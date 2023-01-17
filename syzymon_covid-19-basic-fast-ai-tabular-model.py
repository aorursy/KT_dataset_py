from fastai.tabular import * 

from fastai.callbacks import *



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
PATH = '/kaggle/input/covid19-global-forecasting-week-3/'

train_df = pd.read_csv(PATH + 'train.csv', parse_dates=['Date'])

test_df = pd.read_csv(PATH + 'test.csv', parse_dates=['Date'])



add_datepart(train_df, 'Date', drop=False)

add_datepart(test_df, 'Date', drop=False)
PATH1 = '/kaggle/input/covid19-country-data-wk3-release/'

meta_df = pd.read_csv(PATH1 + 'Data Join - RELEASE.csv', thousands=",")



PATH2 = '/kaggle/input/countryinfo/'

countryinfo = pd.read_csv(PATH2 + 'covid19countryinfo.csv', thousands=",", parse_dates=['quarantine', 'schools', 'publicplace', 'gathering', 'nonessential'])

testinfo = pd.read_csv(PATH2 + 'covid19tests.csv', thousands=",")



countryinfo.rename(columns={'region': 'Province_State', 'country': 'Country_Region'}, inplace=True)

testinfo.rename(columns={'region': 'Province_State', 'country': 'Country_Region'}, inplace=True)

testinfo = testinfo.drop(['alpha3code', 'alpha2code', 'date'], axis=1)



PATH3 = '/kaggle/input/covid19-forecasting-metadata/'

continent_meta = pd.read_csv(PATH3 + 'region_metadata.csv')

continent_meta = continent_meta[['Country_Region' ,'Province_State', 'continent']]



def fill_unknown_state(df):

    df.fillna({'Province_State': 'Unknown'}, inplace=True)

    

for d in [train_df, test_df, meta_df, countryinfo, testinfo, continent_meta]:

    fill_unknown_state(d)
idx_group = ['Country_Region', 'Province_State']



def day_reached_cases(df, name, no_cases=1):

    """For each country/province get first day of year with at least given number of cases."""

    gb = df[df['ConfirmedCases'] >= no_cases].groupby(idx_group)

    return gb.Dayofyear.first().reset_index().rename(columns={'Dayofyear': name})



def area_fatality_rate(df):

    """Get average fatality rate for last known entry, for each country/province."""

    gb = df[df['Fatalities'] >= 22].groupby(idx_group)

    res_df = (gb.Fatalities.last() / gb.ConfirmedCases.last()).reset_index()

    return res_df.rename(columns={0 : 'FatalityRate'})
def joined_data(df):

    res = df.copy()

    

    fatality = area_fatality_rate(train_df)

    first_nonzero = day_reached_cases(train_df, 'FirstCaseDay', 1)

    first_fifty = day_reached_cases(train_df, 'First50CasesDay', 50)

    

    # Add external features

    res = pd.merge(res, continent_meta, how='left')

    res = pd.merge(res, meta_df, how='left')

    res = pd.merge(res, countryinfo, how='left')

    res = pd.merge(res, testinfo, how='left', left_on=idx_group, right_on=idx_group)

    

    # Add calculated features

    res = pd.merge(res, fatality, how='left')

    res = pd.merge(res, first_nonzero, how='left')

    res = pd.merge(res, first_fifty, how='left')

    return res



train_df = joined_data(train_df)

test_df = joined_data(test_df)
# It turns out any country in train dataset has at least one case.

train_df.FirstCaseDay.isna().sum()
def with_new_features(df, train=True):

    res = df.copy()

    add_datepart(res, 'quarantine', prefix='qua')

    add_datepart(res, 'schools', prefix='sql')

    

    res['DaysSinceFirst'] = res['Dayofyear'] - res['FirstCaseDay']

    res['DaysSince50'] = res['Dayofyear'] - res['First50CasesDay']

    res['DaysQua'] = res['Dayofyear'] - res['quaDayofyear']

    res['DaysSql'] = res['Dayofyear'] - res['sqlDayofyear']

    

    # Since we will take log of dependent variable, we won't make it nonzero.

    if train:

        res['ConfirmedCases'] += 1

    return res

    

train_df = with_new_features(train_df)

test_df = with_new_features(test_df, train=False)
# Categorical variables - only basic identifiers, some features like continent will be worth adding.

cat_vars = ['Country_Region', 'Province_State',

            'continent'

#             'publicplace', 'gathering', 'nonessential'

           ]



# Continuous variables - just ones directly connected with time.

cont_vars = ['DaysSinceFirst', 'DaysSince50', 'Dayofyear',

            'DaysQua', 'DaysSql',

            'TRUE POPULATION', 

            'testper1m', 'positiveper1m',

            'casediv1m', 'deathdiv1m', 

            'FatalityRate',

#             'density', 'urbanpop', 'medianage', 'hospibed','healthperpop', 'fertility',

#             'smokers', 'lung', 

#             'continent_gdp_pc', 'continent_happiness', 'continent_Life_expectancy','GDP_region', 

#             'latitude', 'abs_latitude', 'longitude', 'temperature', 'humidity',

            ]



# We will predict only confirmed cases. 

# For fatalities, one could train another model but we won't do it - multiplying by average fatality in each area is enough for a sample submission.

dep_var = 'ConfirmedCases'



df = train_df[cont_vars + cat_vars + [dep_var,'Date']].copy().sort_values('Date')
print(test_df.Date.min())

MAX_TRAIN_IDX = df[df['Date'] < test_df.Date.min()].shape[0]
procs=[FillMissing, Categorify, Normalize]

path = '/kaggle/working/'

data = (TabularList.from_df(df, path=path, cat_names=cat_vars.copy(), cont_names=cont_vars.copy(), procs=procs,)

                # Take only rows before beginning of the test set - as explained above.

                .split_by_idx(list(range(MAX_TRAIN_IDX, len(df))))

                .label_from_df(cols=dep_var, label_cls=FloatList, log=True)

                .add_test(TabularList.from_df(test_df, path=path, cat_names=cat_vars, cont_names=cont_vars))

                .databunch())
data.show_batch(ds_type=DatasetType.Train, rows=5)
learn = tabular_learner(data, layers=[1000, 500], ps=[0.001,0.01], emb_drop=0.04, 

                        metrics=root_mean_squared_error)



calls = [SaveModelCallback(learn, mode='min', every='improvement', monitor='root_mean_squared_error', name='best')]
learn.lr_find(start_lr=1e-8, end_lr=1e-2, num_it=300)

learn.recorder.plot()
learn.fit_one_cycle(25, callbacks=calls)
learn.recorder.plot_losses()
learn.load('best')

learn.fit_one_cycle(10, 5e-5, callbacks=calls)
learn.load('best')

learn.fit_one_cycle(10, 5e-6, callbacks=calls)
learn.load('best')

learn.fit_one_cycle(10, 5e-7, callbacks=calls)
learn.load('best')
preds1, y = learn.get_preds(ds_type=DatasetType.Test)
res1 = np.expm1(preds1)

res2 = list(map(lambda x: x[0], res1.numpy()))

submit = pd.DataFrame({'ConfirmedCases': res2})

submit.index = test_df.ForecastId
fatality_series = test_df.FatalityRate.copy()

fatality_series.index += 1

fatality_series.fillna(0.02137, inplace=True)



submit['Fatalities'] = (submit.ConfirmedCases > 69) * fatality_series * submit.ConfirmedCases
submit.to_csv('submission.csv')
import seaborn as sns



min_date = test_df.Date.min()

max_date = train_df.Date.max()



f, axes = plt.subplots(10, 1, figsize=(16, 60))



def plot_preds(country, ax):

    targets = train_df[(train_df['Country_Region'] == country) & (train_df['Date'] >= min_date)].ConfirmedCases

    subset = test_df[(test_df['Country_Region'] == country) & (test_df['Date'] <= max_date)]

    

    idx = subset.index

    dates = subset.Date

    predicted = submit.iloc[idx].ConfirmedCases

    

    targets.index = dates

    predicted.index = dates

    

    combined = pd.DataFrame({'real' : targets, 'pred': predicted})

    

    sns.lineplot(data=combined, ax=axes[ax]).set_title(country)



plot_preds('Italy', 0)

plot_preds('Spain', 1)

plot_preds('Germany', 2)

plot_preds('Poland', 3)

plot_preds('Czechia', 4)

plot_preds('Russia', 5)

plot_preds('Iran', 6)

plot_preds('Sweden', 7)

plot_preds('Japan', 8)

plot_preds('Belgium', 9)
res = submit.iloc[test_df[(test_df['Country_Region'] == 'Poland')].index]

res.index = test_df[(test_df['Country_Region'] == 'Poland')].Date

res
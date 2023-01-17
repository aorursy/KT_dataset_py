

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
covid_df = pd.read_csv('/kaggle/input/covid-19-data-exploration/covid19_data.csv')

covid_df.head()
covid_df['Date'] = pd.to_datetime(covid_df['Date'])

covid_df = covid_df.rename(columns={'Cumulative tests': 'agg_tests',

                                    'Cumulative tests per million': 'agg_tests_per_mil',

                                    'Total confirmed cases (cases)': 'agg_cases',

                                    'Confirmed cases per million (cases per million)': 'agg_cases_per_mil',

                                    'Total confirmed deaths (deaths)': 'agg_deaths',

                                    'Confirmed deaths per million (deaths per million)': 'agg_deaths_per_mil'})

# covid_df[['agg_tests', 'agg_tests_per_mil', 'agg_cases', 'agg_cases_per_mil', 'agg_deaths', 'agg_deaths_per_mil']] = covid_df[[

#     'agg_tests', 'agg_tests_per_mil', 'agg_cases', 'agg_cases_per_mil', 'agg_deaths', 'agg_deaths_per_mil']].fillna(value=0)

covid_df[['agg_tests', 'agg_cases', 'agg_deaths']] = covid_df[['agg_tests', 'agg_cases', 'agg_deaths']].fillna(value=0)

covid_df.head()

def days_since_first(col):

    first_dates = covid_df[covid_df[col] != 0].groupby('Entity').first()['Date']

    

    def day_diff(row):

        if row['Entity'] not in first_dates:

            return None

        return (row['Date'] - first_dates[row['Entity']]).days

    

    return covid_df.apply(day_diff, axis=1)



covid_df['days_since_1st_case'] = days_since_first('agg_cases')

covid_df['days_since_1st_test'] = days_since_first('agg_tests')

covid_df_first_case = covid_df[covid_df['days_since_1st_case'] >= 0]



def us_vs_sk_plots_since_1st_case(x, y, title):

    covid_df_first_case = covid_df[covid_df['days_since_1st_case'] >= 0]

    plt.figure(figsize=(15, 5))

    sns.relplot(x=x, y=y, hue='Entity', kind='line', data=covid_df_first_case[covid_df_first_case['Entity'].isin(['United States', 'South Korea'])])

    plt.title(title)

    plt.xticks(rotation=90)

    

us_vs_sk_plots_since_1st_case('days_since_1st_case', 'agg_cases', 'Total cases by first day for US and South Korea')
us_vs_sk_plots_since_1st_case('days_since_1st_case', 'agg_cases_per_mil', 'Cases per million by first day for US and SK')
us_vs_sk_plots_since_1st_case('days_since_1st_case', 'agg_tests_per_mil', 'Tests per million by first day for US and SK')
def generate_rolling_mean(days, mean_col):

    return covid_df.reset_index().set_index('days_since_1st_case').groupby('Entity').rolling(days, min_periods=1)[mean_col].mean().values



covid_df['agg_cases_weekly_mean'] = generate_rolling_mean(7, 'agg_cases')

covid_df['agg_tests_weekly_mean'] = generate_rolling_mean(7, 'agg_tests')



covid_df['agg_cases_weekly_roc'] = covid_df['agg_cases_weekly_mean'].pct_change()

covid_df['agg_tests_weekly_roc'] = covid_df['agg_tests_weekly_mean'].pct_change()
def plot_against(df, entity, col_list, x, title):

    melted_df = df[df['Entity'] == entity][col_list].melt(x, var_name='cols',  value_name='vals')



    fig = plt.figure(figsize=(10,5))

    g = sns.catplot(x=x, y='vals', hue='cols', data=melted_df, kind='point', aspect=3)

    plt.title(title)

    plt.xticks(rotation=90)

    return g



covid_df_first_test = covid_df[covid_df['days_since_1st_test'] >= 0]

plot_against(df=covid_df_first_test,

             entity='United States',

             col_list=['days_since_1st_test', 'agg_cases_weekly_roc', 'agg_tests_weekly_roc'],

             x='days_since_1st_test', 

             title='US Confirmed Case ROC versus Test ROC')
plot_against(df=covid_df_first_test,

             entity='South Korea',

             col_list=['days_since_1st_test', 'agg_cases_weekly_roc', 'agg_tests_weekly_roc'],

             x='days_since_1st_test', 

             title='SK Confirmed Case ROC versus Test ROC')
plot_against(df=covid_df_first_test,

             entity='Italy',

             col_list=['days_since_1st_test', 'agg_cases_weekly_roc', 'agg_tests_weekly_roc'],

             x='days_since_1st_test', 

             title='Italy Confirmed Case ROC versus Test ROC')
covid_df['agg_deaths_weekly_mean'] = generate_rolling_mean(7, 'agg_deaths')



covid_df['agg_deaths_weekly_roc'] = covid_df['agg_deaths_weekly_mean'].pct_change()
covid_df_first_test = covid_df[covid_df['days_since_1st_test'] >= 0]

plot_against(df=covid_df_first_test,

             entity='Italy',

             col_list=['days_since_1st_test', 'agg_cases_weekly_roc', 'agg_tests_weekly_roc', 'agg_deaths_weekly_roc'],

             x='days_since_1st_test', 

             title='Italy Confirmed Case ROC vs Test ROC vs Death ROC')
plot_against(df=covid_df_first_test,

             entity='South Korea',

             col_list=['days_since_1st_test', 'agg_cases_weekly_roc', 'agg_tests_weekly_roc', 'agg_deaths_weekly_roc'],

             x='days_since_1st_test', 

             title='South Korea Confirmed Case ROC vs Test ROC vs Death ROC')
plot_against(df=covid_df_first_test,

             entity='United States',

             col_list=['days_since_1st_test', 'agg_cases_weekly_roc', 'agg_tests_weekly_roc', 'agg_deaths_weekly_roc'],

             x='days_since_1st_test', 

             title='US Confirmed Case ROC vs Test ROC vs Death ROC')
covid_df['case_roc_to_test_roc_ratio'] = covid_df['agg_cases_weekly_roc'] / covid_df['agg_tests_weekly_roc']

covid_df['death_roc_to_test_roc_ratio'] = covid_df['agg_deaths_weekly_roc'] / covid_df['agg_tests_weekly_roc']



covid_df_first_test = covid_df[covid_df['days_since_1st_test'] >= 0]

plot_against(df=covid_df_first_test,

             entity='South Korea',

             col_list=['days_since_1st_test', 'case_roc_to_test_roc_ratio', 'death_roc_to_test_roc_ratio'],

             x='days_since_1st_test', 

             title='SK Case ROC and Death ROC to Test ROC Ratio')

plot_against(df=covid_df_first_test,

             entity='United States',

             col_list=['days_since_1st_test', 'case_roc_to_test_roc_ratio', 'death_roc_to_test_roc_ratio'],

             x='days_since_1st_test', 

             title='US Case ROC and Death ROC to Test ROC Ratio')

plot_against(df=covid_df_first_test,

             entity='Italy',

             col_list=['days_since_1st_test', 'case_roc_to_test_roc_ratio', 'death_roc_to_test_roc_ratio'],

             x='days_since_1st_test', 

             title='Italy Case ROC and Death ROC to Test ROC Ratio')
covid_df['test_positive_rate'] = covid_df['agg_cases_per_mil'].astype(float) / covid_df['agg_tests_per_mil'].astype(float)



covid_df_first_test = covid_df[covid_df['days_since_1st_test'] >= 0]

# plot_against(df=covid_df_first_test,

#              entity='South Korea',

#              col_list=['days_since_1st_test', 'test_positive_rate'],

#              x='days_since_1st_test', 

#              title='South Korea Test Positive Ratio Over Time')

# plot_against(df=covid_df_first_test[covid_df_first_test['test_positive_rate'] <= 1],

#              entity='United States',

#              col_list=['days_since_1st_test', 'test_positive_rate'],

#              x='days_since_1st_test',

#              title='US Test Positive Ratio Over Time')

# plot_against(df=covid_df_first_test,

#              entity='Italy',

#              col_list=['days_since_1st_test', 'test_positive_rate'],

#              x='days_since_1st_test', 

#              title='Italy Test Positive Ratio Over Time')

# plot_against(df=covid_df_first_test,

#              entity='New Zealand',

#              col_list=['days_since_1st_test', 'test_positive_rate'],

#              x='days_since_1st_test', 

#              title='New Zealand Test Positive Ratio Over Time')



def plot_against_entity(df, entity_list, y, x, title):

    # melted_df = df[df['Entity'].isin(entity_list)][col_list].melt(x, var_name='cols',  value_name='vals')

    entity_df = df[df['Entity'].isin(entity_list)][['Entity', x, y]]



    fig = plt.figure(figsize=(10,5))

    g = sns.relplot(x=x, y=y, hue='Entity', data=entity_df, kind='line', aspect=3)

    plt.title(title)

    plt.xticks(rotation=90)

    return g



covid_df_first_test = covid_df[covid_df['days_since_1st_test'] >= 5]

plot_against_entity(df=covid_df_first_test,

                     entity_list=['Italy', 'New Zealand', 'South Korea', 'United States'],

                     y='test_positive_rate',

                     x='days_since_1st_test',

                     title='Test Positive Ratio Over Time')
plot_against(df=covid_df_first_test,

             entity='Sweden',

             col_list=['days_since_1st_test', 'agg_cases_weekly_roc', 'agg_tests_weekly_roc', 'agg_deaths_weekly_roc'],

             x='days_since_1st_test', 

             title='Sweden Confirmed Case ROC vs Test ROC vs Death ROC')

plot_against(df=covid_df_first_test,

             entity='Sweden',

             col_list=['days_since_1st_test', 'case_roc_to_test_roc_ratio', 'death_roc_to_test_roc_ratio'],

             x='days_since_1st_test', 

             title='Sweden Case ROC and Death ROC to Test ROC Ratio')
covid_df['test_positive_rate_ROC'] = covid_df[covid_df['test_positive_rate'] <= 1]['test_positive_rate'].pct_change()



# Replace infinite values with NaN

covid_df = covid_df.replace([np.inf, -np.inf], np.nan)



covid_df['test_positive_rate_ROC_weekly_mean'] = generate_rolling_mean(7, 'test_positive_rate_ROC')

max_tpr_roc_weekly_mean_per_entity = covid_df.groupby('Entity')['test_positive_rate_ROC_weekly_mean'].max()
fig = plt.figure(figsize=(32,5))

plt.scatter(x=max_tpr_roc_weekly_mean_per_entity.index, y=max_tpr_roc_weekly_mean_per_entity.values)

plt.title('Maximum Test Positive Rate ROC Weekly Mean by Entity')

plt.xticks(rotation=90)

plt.show()
covid_df_first_test = covid_df[covid_df['days_since_1st_test'] >= 7]

# g = plot_against(df=covid_df_first_test,

#                  entity='South Korea',

#                  col_list=['days_since_1st_test', 'test_positive_rate_ROC_weekly_mean'],

#                  x='days_since_1st_test', 

#                  title='SK TPR ROC Weekly Mean Over Time')

# g.ax.axhline(0, ls='--')

# g = plot_against(df=covid_df_first_test,

#                  entity='United States',

#                  col_list=['days_since_1st_test', 'test_positive_rate_ROC_weekly_mean'],

#                  x='days_since_1st_test', 

#                  title='US TPR ROC Weekly Mean Over Time')

# g.ax.axhline(0, ls='--')

# g = plot_against(df=covid_df_first_test,

#                  entity='Italy',

#                  col_list=['days_since_1st_test', 'test_positive_rate_ROC_weekly_mean'],

#                  x='days_since_1st_test', 

#                  title='Italy TPR ROC Weekly Mean Over Time')

# g.ax.axhline(0, ls='--')

# g = plot_against(df=covid_df_first_test,

#                  entity='New Zealand',

#                  col_list=['days_since_1st_test', 'test_positive_rate_ROC_weekly_mean'],

#                  x='days_since_1st_test', 

#                  title='New Zealand TPR ROC Weekly Mean Over Time')

# g.ax.axhline(0, ls='--')



covid_df_first_test = covid_df[covid_df['days_since_1st_test'] >= 7]

g = plot_against_entity(df=covid_df_first_test,

                        entity_list=['Italy', 'New Zealand', 'South Korea', 'United States'],

                        y='test_positive_rate_ROC_weekly_mean',

                        x='days_since_1st_test',

                        title='Test Positive Ratio ROC Weekly Mean Over Time')

g.ax.axhline(0, ls='--')
FILTER_DAYS = 7



covid_df_first_test = covid_df[covid_df['days_since_1st_test'] >= FILTER_DAYS]

maxidx_tpr_roc_weekly_mean_per_entity = covid_df_first_test.replace([np.inf, -np.inf], np.nan).groupby('Entity')['test_positive_rate_ROC_weekly_mean'].idxmax().dropna()
# Let's find the first negative TPR ROC after the maximum TPR ROC.

def find_neg_tpr_roc_after_max(row):

    if row['Entity'] not in maxidx_tpr_roc_weekly_mean_per_entity:

        return

    if row.name < maxidx_tpr_roc_weekly_mean_per_entity[row['Entity']]:

        return

    if row['test_positive_rate_ROC_weekly_mean'] < 0:

        return row['Date']

    

covid_df_first_test = covid_df[covid_df['days_since_1st_test'] >= FILTER_DAYS]

entity_with_completed_peak_list = []

first_neg_tpr_roc_weekly_mean_after_max = []

for entity in covid_df_first_test['Entity'].unique():

    tmp = covid_df_first_test[covid_df_first_test['Entity'] == entity].apply(find_neg_tpr_roc_after_max, axis=1)

    if len(tmp.dropna().index) > 0:

        entity_with_completed_peak_list.append(entity)

        first_neg_tpr_roc_weekly_mean_after_max.append(tmp.dropna().iloc[0])



entity_first_neg_dict = dict(zip(entity_with_completed_peak_list, first_neg_tpr_roc_weekly_mean_after_max))



# Now using this dictionary we can find the length of the peaks for those countries that had a negative TPR ROC after the max.

entities_completed_peak_length = {}

for entity, first_neg_date in entity_first_neg_dict.items():

    tpr_roc_max_idx = maxidx_tpr_roc_weekly_mean_per_entity[entity]

    tpr_roc_max_date = covid_df.loc[tpr_roc_max_idx, 'Date']

    length = (first_neg_date - tpr_roc_max_date).days

    entities_completed_peak_length.update({entity: length})



print(entities_completed_peak_length)
fig = plt.figure(figsize=(25,5))

sns.barplot(x=list(entities_completed_peak_length.keys()), y=list(entities_completed_peak_length.values()))

plt.title('Length in Days from Beginning of Outbreak to Peak')

plt.xticks(rotation=90)
average_outbreak_length = np.array([entities_completed_peak_length[entity] for entity in entities_completed_peak_length]).mean()

print('Average Outbreak Length: {} days'.format(average_outbreak_length))



entities_not_yet_through_peak = set(maxidx_tpr_roc_weekly_mean_per_entity.index).difference(set(entity_with_completed_peak_list))

print('Entities not yet through peak: {}'.format(entities_not_yet_through_peak))
# Let's find the average peak date based off this average outbreak length.

predicted_peak_based_off_avg = {}

for entity in entities_not_yet_through_peak:

    maxdate = covid_df.loc[maxidx_tpr_roc_weekly_mean_per_entity[entity], 'Date']

    predicted_peakdate = maxdate + datetime.timedelta(days=average_outbreak_length)

    predicted_peak_based_off_avg.update({entity: predicted_peakdate})

    

print('Simple Prediction of Peak Date based off Average Outbreak Length')

print(predicted_peak_based_off_avg)
def calc_features(covid_entity):

    # Let's build a DataFrame of maximum aggregate X per million with the outbreak length in days.

    covid_entity['max_agg_deaths_per_mil'] = covid_entity['agg_deaths_per_mil'].max()

    covid_entity['max_agg_cases_per_mil'] = covid_entity['agg_cases_per_mil'].max()

    covid_entity['max_agg_tests_per_mil'] = covid_entity['agg_tests_per_mil'].max()



    # Calculate Case Fatality Rate (CFR) and find the maximum and mean.

    covid_entity['mean_cfr'] = covid_entity['cfr'].mean()

    covid_entity['max_cfr'] = covid_entity['cfr'].max()



    # Calculate mean/max Test Positive Rate

    covid_entity['mean_tpr'] = covid_entity['test_positive_rate'].mean()

    covid_entity['max_tpr'] = covid_entity['test_positive_rate'].max()



    # Calculate mean Test Positive Rate Rate of Change

    covid_entity['mean_tpr_roc'] = covid_entity['test_positive_rate_ROC'].mean()

    covid_entity['max_tpr_roc_weekly_mean'] = covid_entity['test_positive_rate_ROC_weekly_mean'].max()



    # Calculate Area Under the Curve of Test Positive Rate over Time

    covid_entity_prev_days = covid_entity['days_since_1st_test'].shift(periods=1)

    covid_entity_prev_tpr = covid_entity['test_positive_rate'].shift(periods=1)

    day_diff = covid_entity['days_since_1st_test'] - covid_entity_prev_days

    tpr_sum = covid_entity['test_positive_rate'] + covid_entity_prev_tpr

    tpr_auc = 0.5 * day_diff * tpr_sum

    covid_entity['tpr_iauc'] = tpr_auc.sum()



    return covid_entity

    

covid_df['cfr'] = covid_df['agg_deaths'] / covid_df['agg_cases']



calculated_features = covid_df.groupby('Entity').apply(calc_features)

print(calculated_features[calculated_features['Entity'] == 'South Korea'])
feature_list = ['max_agg_deaths_per_mil', 'max_agg_cases_per_mil', 'max_agg_tests_per_mil',

                'mean_cfr', 'max_cfr',

                'mean_tpr', 'max_tpr', 'max_tpr_roc_weekly_mean', 'mean_tpr_roc', 'tpr_iauc']

# feature_list = ['max_agg_deaths_per_mil', 'max_agg_cases_per_mil', 'max_agg_tests_per_mil',

#                 'mean_cfr', 'mean_tpr_roc', 'tpr_iauc']



outbreak_length_df = pd.DataFrame(entities_completed_peak_length.values(), index=entities_completed_peak_length.keys(), columns=['outbreak_length_in_days'])

single_calc_features = calculated_features.groupby('Entity')[feature_list].first()

for feature in feature_list:

    outbreak_length_df[feature] = single_calc_features[feature]



print(outbreak_length_df)
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures



X = outbreak_length_df[feature_list].values

y = outbreak_length_df['outbreak_length_in_days'].values



poly_reg = PolynomialFeatures(degree=2)

X_poly = poly_reg.fit_transform(X)

        

poly_model = LinearRegression(normalize=True)

poly_model.fit(X_poly, y)



print('Poly Model Coefficients:\n{}\n'.format(poly_model.coef_))

print('Poly Model Score: {}'.format(poly_model.score(X_poly, y)))



model = LinearRegression()

model.fit(X, y)



print('Linear Model Coefficients:\n{}\n'.format(model.coef_))

print('Linear Model Score: {}'.format(model.score(X, y)))



def predict_peak(entity):

    values = [single_calc_features.loc[entity, feature_list].values]

    return model.predict(values)[0]



def predict_peak_poly(entity):

    values = poly_reg.fit_transform([single_calc_features.loc[entity, feature_list].values])

    return poly_model.predict(values)[0]



print('Predicted Italy Outbreak Length: {} days'.format(predict_peak('Italy')))



print('Predicted South Korea Outbreak Length: {} days'.format(predict_peak('South Korea')))
# Let's find the average peak date based off this average outbreak length.

predicted_peak = {}

for entity in entities_not_yet_through_peak:

    maxdate = covid_df.loc[maxidx_tpr_roc_weekly_mean_per_entity[entity], 'Date']

    predicted_peaklength = predict_peak_poly(entity)

    predicted_peakdate = maxdate + datetime.timedelta(days=predicted_peaklength)

    print('Entity: {}'.format(entity))

    print('Peak Length: {}'.format(predicted_peaklength))

    print('Max Date: {}'.format(maxdate))

    print('Peak Date: {}\n'.format(predicted_peakdate))

    predicted_peak.update({entity: {'start': maxdate, 'peak': predicted_peakdate, 'length': predicted_peaklength}})



fig = plt.figure(figsize=(15,5))

sns.barplot(x=list(predicted_peak.keys()), y=[val['length'] for val in predicted_peak.values()])

plt.title('Length in Days from Beginning of Outbreak to Peak')

plt.xticks(rotation=90)
predicted_peak = {}

for entity in entities_not_yet_through_peak:

    maxdate = covid_df.loc[maxidx_tpr_roc_weekly_mean_per_entity[entity], 'Date']

    predicted_peaklength = predict_peak(entity)

    predicted_peakdate = maxdate + datetime.timedelta(days=predicted_peaklength)

    print('Entity: {}'.format(entity))

    print('Peak Length: {}'.format(predicted_peaklength))

    print('Max Date: {}'.format(maxdate))

    print('Peak Date: {}\n'.format(predicted_peakdate))

    predicted_peak.update({entity: {'start': maxdate, 'peak': predicted_peakdate, 'length': predicted_peaklength}})



fig = plt.figure(figsize=(15,5))

sns.barplot(x=list(predicted_peak.keys()), y=[val['length'] for val in predicted_peak.values()])

plt.title('Length in Days from Beginning of Outbreak to Peak')

plt.xticks(rotation=90)
covid_df_og = pd.read_csv('/kaggle/input/covid-19-data-exploration/covid19_data.csv')

covid_df_og['Date'] = pd.to_datetime(covid_df_og['Date'])

covid_df_og = covid_df_og.rename(columns={'Cumulative tests': 'agg_tests',

                                          'Cumulative tests per million': 'agg_tests_per_mil',

                                          'Total confirmed cases (cases)': 'agg_cases',

                                          'Confirmed cases per million (cases per million)': 'agg_cases_per_mil',

                                          'Total confirmed deaths (deaths)': 'agg_deaths',

                                          'Confirmed deaths per million (deaths per million)': 'agg_deaths_per_mil'})
entity = covid_df[covid_df['days_since_1st_case'] > 0].groupby('Entity')

entity_trr = entity['agg_tests_per_mil'].count() / entity['Code'].count()



# Let's drop entities with zero tests.

entity_trr = entity_trr[entity_trr.ne(0)]



fig = plt.figure(figsize=(15,5))

g = sns.barplot(x=entity_trr.index, y=entity_trr.values)

g.axes.axhline(1, ls='--')

plt.title('Test Reporting Rate by Entity')

plt.xticks(rotation=90)
entity = covid_df[covid_df['days_since_1st_case'] > 0].groupby('Entity')

entity_crr = entity['agg_cases_per_mil'].count() / entity['Code'].count()



# Let's drop entities with zero cases, and entities that fully reported all cases.

entity_crr = entity_crr[entity_crr.ne(0) & entity_crr.ne(1)]



fig = plt.figure(figsize=(25,5))

g = sns.barplot(x=entity_crr.index, y=entity_crr.values)

g.axes.axhline(1, ls='--')

plt.title('Case Reporting Rate by Entity (filtered by non-perfect examples)')

plt.xticks(rotation=90)
entity = covid_df[covid_df['days_since_1st_case'] > 0].groupby('Entity')

entity_drr = entity['agg_deaths_per_mil'].count() / entity['Code'].count()



# Let's drop entities with zero deaths, and entities that fully reported all cases.

entity_drr = entity_drr[entity_drr.ne(0) & entity_drr.ne(1)]



fig = plt.figure(figsize=(25,5))

g = sns.barplot(x=entity_drr.index, y=entity_drr.values)

g.axes.axhline(1, ls='--')

plt.title('Death Reporting Rate by Entity (filtered by non-perfect examples)')

plt.xticks(rotation=90)
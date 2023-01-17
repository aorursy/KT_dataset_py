import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from matplotlib.gridspec import GridSpec

import seaborn as sns

from scipy import stats

from scipy.special import boxcox1p



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor



import warnings

warnings.filterwarnings("ignore") # ignoring annoying warnings



from pandasql import sqldf

pysqldf = lambda q: sqldf(q, globals())
features = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/features.csv.zip')

train = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/train.csv.zip')

stores = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/stores.csv')

test = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/test.csv.zip')

sample_submission = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip')
feat_sto = features.merge(stores, how='inner', on='Store')
feat_sto.head(5)
pd.DataFrame(feat_sto.dtypes, columns=['Type'])
train.head(5)
pd.DataFrame({'Type_Train': train.dtypes, 'Type_Test': test.dtypes})
feat_sto.Date = pd.to_datetime(feat_sto.Date)

train.Date = pd.to_datetime(train.Date)

test.Date = pd.to_datetime(test.Date)
feat_sto['Week'] = feat_sto.Date.dt.week 

feat_sto['Year'] = feat_sto.Date.dt.year
train_detail = train.merge(feat_sto, 

                           how='inner',

                           on=['Store','Date','IsHoliday']).sort_values(by=['Store',

                                                                            'Dept',

                                                                            'Date']).reset_index(drop=True)
test_detail = test.merge(feat_sto, 

                           how='inner',

                           on=['Store','Date','IsHoliday']).sort_values(by=['Store',

                                                                            'Dept',

                                                                            'Date']).reset_index(drop=True)
del features, train, stores, test
null_columns = (train_detail.isnull().sum(axis = 0)/len(train_detail)).sort_values(ascending=False).index

null_data = pd.concat([

    train_detail.isnull().sum(axis = 0),

    (train_detail.isnull().sum(axis = 0)/len(train_detail)).sort_values(ascending=False),

    train_detail.loc[:, train_detail.columns.isin(list(null_columns))].dtypes], axis=1)

null_data = null_data.rename(columns={0: '# null', 

                                      1: '% null', 

                                      2: 'type'}).sort_values(ascending=False, by = '% null')

null_data = null_data[null_data["# null"]!=0]

null_data
pysqldf("""

SELECT

    T.*,

    case

        when ROW_NUMBER() OVER(partition by Year order by week) = 1 then 'Super Bowl'

        when ROW_NUMBER() OVER(partition by Year order by week) = 2 then 'Labor Day'

        when ROW_NUMBER() OVER(partition by Year order by week) = 3 then 'Thanksgiving'

        when ROW_NUMBER() OVER(partition by Year order by week) = 4 then 'Christmas'

    end as Holyday,

    case

        when ROW_NUMBER() OVER(partition by Year order by week) = 1 then 'Sunday'

        when ROW_NUMBER() OVER(partition by Year order by week) = 2 then 'Monday'

        when ROW_NUMBER() OVER(partition by Year order by week) = 3 then 'Thursday'

        when ROW_NUMBER() OVER(partition by Year order by week) = 4 and Year = 2010 then 'Saturday'

        when ROW_NUMBER() OVER(partition by Year order by week) = 4 and Year = 2011 then 'Sunday'

        when ROW_NUMBER() OVER(partition by Year order by week) = 4 and Year = 2012 then 'Tuesday'

    end as Day

    from(

        SELECT DISTINCT

            Year,

            Week,

            case 

                when Date <= '2012-11-01' then 'Train Data' else 'Test Data' 

            end as Data_type

        FROM feat_sto

        WHERE IsHoliday = True) as T""")
weekly_sales_2010 = train_detail[train_detail.Year==2010]['Weekly_Sales'].groupby(train_detail['Week']).mean()

weekly_sales_2011 = train_detail[train_detail.Year==2011]['Weekly_Sales'].groupby(train_detail['Week']).mean()

weekly_sales_2012 = train_detail[train_detail.Year==2012]['Weekly_Sales'].groupby(train_detail['Week']).mean()

plt.figure(figsize=(20,8))

sns.lineplot(weekly_sales_2010.index, weekly_sales_2010.values)

sns.lineplot(weekly_sales_2011.index, weekly_sales_2011.values)

sns.lineplot(weekly_sales_2012.index, weekly_sales_2012.values)

plt.grid()

plt.xticks(np.arange(1, 53, step=1))

plt.legend(['2010', '2011', '2012'], loc='best', fontsize=16)

plt.title('Average Weekly Sales - Per Year', fontsize=18)

plt.ylabel('Sales', fontsize=16)

plt.xlabel('Week', fontsize=16)

plt.show()
train_detail.loc[(train_detail.Year==2010) & (train_detail.Week==13), 'IsHoliday'] = True

train_detail.loc[(train_detail.Year==2011) & (train_detail.Week==16), 'IsHoliday'] = True

train_detail.loc[(train_detail.Year==2012) & (train_detail.Week==14), 'IsHoliday'] = True

test_detail.loc[(test_detail.Year==2013) & (test_detail.Week==13), 'IsHoliday'] = True
weekly_sales_mean = train_detail['Weekly_Sales'].groupby(train_detail['Date']).mean()

weekly_sales_median = train_detail['Weekly_Sales'].groupby(train_detail['Date']).median()

plt.figure(figsize=(20,8))

sns.lineplot(weekly_sales_mean.index, weekly_sales_mean.values)

sns.lineplot(weekly_sales_median.index, weekly_sales_median.values)

plt.grid()

plt.legend(['Mean', 'Median'], loc='best', fontsize=16)

plt.title('Weekly Sales - Mean and Median', fontsize=18)

plt.ylabel('Sales', fontsize=16)

plt.xlabel('Date', fontsize=16)

plt.show()
weekly_sales = train_detail['Weekly_Sales'].groupby(train_detail['Store']).mean()

plt.figure(figsize=(20,8))

sns.barplot(weekly_sales.index, weekly_sales.values, palette='dark')

plt.grid()

plt.title('Average Sales - per Store', fontsize=18)

plt.ylabel('Sales', fontsize=16)

plt.xlabel('Store', fontsize=16)

plt.show()
weekly_sales = train_detail['Weekly_Sales'].groupby(train_detail['Dept']).mean()

plt.figure(figsize=(25,8))

sns.barplot(weekly_sales.index, weekly_sales.values, palette='dark')

plt.grid()

plt.title('Average Sales - per Dept', fontsize=18)

plt.ylabel('Sales', fontsize=16)

plt.xlabel('Dept', fontsize=16)

plt.show()
sns.set(style="white")



corr = train_detail.corr()



mask = np.triu(np.ones_like(corr, dtype=np.bool))



f, ax = plt.subplots(figsize=(20, 15))



cmap = sns.diverging_palette(220, 10, as_cmap=True)



plt.title('Correlation Matrix', fontsize=18)



sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)



plt.show()
train_detail = train_detail.drop(columns=['Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])

test_detail = test_detail.drop(columns=['Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])
def make_discrete_plot(feature):

    fig = plt.figure(figsize=(20,8))

    gs = GridSpec(1,2)

    sns.boxplot(y=train_detail.Weekly_Sales, x=train_detail[feature], ax=fig.add_subplot(gs[0,0]))

    plt.ylabel('Sales', fontsize=16)

    plt.xlabel(feature, fontsize=16)

    sns.stripplot(y=train_detail.Weekly_Sales, x=train_detail[feature], ax=fig.add_subplot(gs[0,1]))

    plt.ylabel('Sales', fontsize=16)

    plt.xlabel(feature, fontsize=16)

    fig.show()
def make_continuous_plot(feature):

    

    fig = plt.figure(figsize=(18,15))

    gs = GridSpec(2,2)

    

    j = sns.scatterplot(y=train_detail['Weekly_Sales'], 

                        x=boxcox1p(train_detail[feature], 0.15), ax=fig.add_subplot(gs[0,1]), palette = 'blue')



    plt.title('BoxCox 0.15\n' + 'Corr: ' + str(np.round(train_detail['Weekly_Sales'].corr(boxcox1p(train_detail[feature], 0.15)),2)) +

              ', Skew: ' + str(np.round(stats.skew(boxcox1p(train_detail[feature], 0.15), nan_policy='omit'),2)))

    

    j = sns.scatterplot(y=train_detail['Weekly_Sales'], 

                        x=boxcox1p(train_detail[feature], 0.25), ax=fig.add_subplot(gs[1,0]), palette = 'blue')



    plt.title('BoxCox 0.25\n' + 'Corr: ' + str(np.round(train_detail['Weekly_Sales'].corr(boxcox1p(train_detail[feature], 0.25)),2)) +

              ', Skew: ' + str(np.round(stats.skew(boxcox1p(train_detail[feature], 0.25), nan_policy='omit'),2)))

    

    j = sns.distplot(train_detail[feature], ax=fig.add_subplot(gs[1,1]), color = 'green')



    plt.title('Distribution\n')

    

    j = sns.scatterplot(y=train_detail['Weekly_Sales'], 

                        x=train_detail[feature], ax=fig.add_subplot(gs[0,0]), color = 'red')



    plt.title('Linear\n' + 'Corr: ' + str(np.round(train_detail['Weekly_Sales'].corr(train_detail[feature]),2)) + ', Skew: ' + 

               str(np.round(stats.skew(train_detail[feature], nan_policy='omit'),2)))

    

    fig.show()
make_discrete_plot('IsHoliday')
make_discrete_plot('Type')
train_detail.Type = train_detail.Type.apply(lambda x: 3 if x == 'A' else(2 if x == 'B' else 1))

test_detail.Type = test_detail.Type.apply(lambda x: 3 if x == 'A' else(2 if x == 'B' else 1))
make_continuous_plot('Temperature')
train_detail = train_detail.drop(columns=['Temperature'])

test_detail = test_detail.drop(columns=['Temperature'])
make_continuous_plot('CPI')
train_detail = train_detail.drop(columns=['CPI'])

test_detail = test_detail.drop(columns=['CPI'])
make_continuous_plot('Unemployment')
train_detail = train_detail.drop(columns=['Unemployment'])

test_detail = test_detail.drop(columns=['Unemployment'])
make_continuous_plot('Size')
def WMAE(dataset, real, predicted):

    weights = dataset.IsHoliday.apply(lambda x: 5 if x else 1)

    return np.round(np.sum(weights*abs(real-predicted))/(np.sum(weights)), 2)
def random_forest(n_estimators, max_depth):

    result = []

    for estimator in n_estimators:

        for depth in max_depth:

            wmaes_cv = []

            for i in range(1,5):

                print('k:', i, ', n_estimators:', estimator, ', max_depth:', depth)

                x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)

                RF = RandomForestRegressor(n_estimators=estimator, max_depth=depth)

                RF.fit(x_train, y_train)

                predicted = RF.predict(x_test)

                wmaes_cv.append(WMAE(x_test, y_test, predicted))

            print('WMAE:', np.mean(wmaes_cv))

            result.append({'Max_Depth': depth, 'Estimators': estimator, 'WMAE': np.mean(wmaes_cv)})

    return pd.DataFrame(result)
def random_forest_II(n_estimators, max_depth, max_features):

    result = []

    for feature in max_features:

        wmaes_cv = []

        for i in range(1,5):

            print('k:', i, ', max_features:', feature)

            x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)

            RF = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=feature)

            RF.fit(x_train, y_train)

            predicted = RF.predict(x_test)

            wmaes_cv.append(WMAE(x_test, y_test, predicted))

        print('WMAE:', np.mean(wmaes_cv))

        result.append({'Max_Feature': feature, 'WMAE': np.mean(wmaes_cv)})

    return pd.DataFrame(result)
def random_forest_III(n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf):

    result = []

    for split in min_samples_split:

        for leaf in min_samples_leaf:

            wmaes_cv = []

            for i in range(1,5):

                print('k:', i, ', min_samples_split:', split, ', min_samples_leaf:', leaf)

                x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)

                RF = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, 

                                           min_samples_leaf=leaf, min_samples_split=split)

                RF.fit(x_train, y_train)

                predicted = RF.predict(x_test)

                wmaes_cv.append(WMAE(x_test, y_test, predicted))

            print('WMAE:', np.mean(wmaes_cv))

            result.append({'Min_Samples_Leaf': leaf, 'Min_Samples_Split': split, 'WMAE': np.mean(wmaes_cv)})

    return pd.DataFrame(result)
X_train = train_detail[['Store','Dept','IsHoliday','Size','Week','Type','Year']]

Y_train = train_detail['Weekly_Sales']
n_estimators = [56, 58, 60]

max_depth = [25, 27, 30]



random_forest(n_estimators, max_depth)
max_features = [2, 3, 4, 5, 6, 7]



random_forest_II(n_estimators=58, max_depth=27, max_features=max_features)
min_samples_split = [2, 3, 4]

min_samples_leaf = [1, 2, 3]



random_forest_III(n_estimators=58, max_depth=27, max_features=6, 

                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
RF = RandomForestRegressor(n_estimators=58, max_depth=27, max_features=6, min_samples_split=3, min_samples_leaf=1)

RF.fit(X_train, Y_train)
X_test = test_detail[['Store', 'Dept', 'IsHoliday', 'Size', 'Week', 'Type', 'Year']]

predict = RF.predict(X_test)
Final = X_test[['Store', 'Dept', 'Week']]

Final['Weekly_Sales'] = predict
Final_adj = pysqldf("""

    SELECT

        Store,

        Dept,

        Week,

        Weekly_Sales,

        case 

            when Week = 52 and last_sales > 2*Weekly_Sales then Weekly_Sales+(2.5/7)*last_sales

            else Weekly_Sales 

        end as Weekly_Sales_Adjusted

    from(

        SELECT

            Store, 

            Dept, 

            Week, 

            Weekly_Sales,

            case 

                when Week = 52 then lag(Weekly_Sales) over(partition by Store, Dept) 

            end as last_sales

        from Final)""")
sample_submission['Weekly_Sales'] = Final_adj['Weekly_Sales_Adjusted']

sample_submission.to_csv('submission.csv',index=False)
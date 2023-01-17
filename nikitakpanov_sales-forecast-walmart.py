import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.special import boxcox1p

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

# tslearn - time series clusters
!pip install tslearn
from tslearn.clustering import TimeSeriesKMeans

from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())


from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
features = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/features.csv.zip')
train = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
stores = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/stores.csv')
test = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/test.csv.zip')
sample_submission = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip')

print("import done")
sep = '--------------------------------------------------------'

d = globals()
df_list = [v for k, v in d.items() if isinstance(v, pd.DataFrame)]
df_dict = {k: v for k, v in d.items() if isinstance(v, pd.DataFrame)}

def quick_look(df):
    print(df.head())
    print("shape")
    print(df.shape)
    print("info")
    print(df.info())
    return

def WMAE(dataset, real, predicted):
    weights = dataset.IsHoliday.apply(lambda x: 5 if x else 1)
    return np.round(np.sum(weights*abs(real-predicted))/(np.sum(weights)), 2)

def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.
for i in df_dict:
    print(i+sep)
    quick_look(eval(i))
# stores2
# pysqldf("""

# SELECT DISTINCT DATE 
#     ,SUBSTR(date,0,11) as date2 ,CAST(SUBSTR(date,6,2) as int ) as month ,WEEK
# FROM store_fact str
# WHERE CAST(SUBSTR(date,6,2) as int )  = 7
# """)

# type dummy
# stores = pd.get_dummies(stores, columns=['Type'])

stores.Type = stores.Type.apply(lambda x: 3 if x == 'A' else(2 if x == 'B' else 1))

# store over time
store_fact = features.merge(stores, how='inner', on='Store')

# time variables
store_fact.Date = pd.to_datetime(store_fact.Date)
train.Date = pd.to_datetime(train.Date)
test.Date = pd.to_datetime(test.Date)


store_fact['Week'] = store_fact.Date.dt.week 
store_fact['Year'] = store_fact.Date.dt.year
store_fact['WeekYear'] = store_fact.Year*100+store_fact.Week

# ADD pre-superbowl
store_fact.loc[(store_fact.Week==5), 'IsHoliday'] = True

# ADD EASTER
store_fact.loc[(store_fact.Year==2010) & (store_fact.Week==13), 'IsHoliday'] = True
store_fact.loc[(store_fact.Year==2011) & (store_fact.Week==16), 'IsHoliday'] = True
store_fact.loc[(store_fact.Year==2012) & (store_fact.Week==14), 'IsHoliday'] = True
store_fact.loc[(store_fact.Year==2013) & (store_fact.Week==13), 'IsHoliday'] = True

# ADD INDEPENDENCE DAY
store_fact.loc[(store_fact.Week==26), 'IsHoliday'] = True


# ADD WEEK OF MONTH
store_fact = pysqldf(
"""
with cal1 as 
    (SELECT distinct date ,SUBSTR(date,0,11) as date2 ,CAST(SUBSTR(date,6,2) as int ) as month ,CAST(SUBSTR(date,9,2) as int ) as days ,year
        ,case when CAST(SUBSTR(date,6,2) as int) = 12 then 0 else IsHoliday end as IsHoliday 
        ,case when CAST(SUBSTR(date,6,2) as int) = 12 then IsHoliday
           else 0 end as IsHoliday_Dec
    FROM store_fact
    )
,cal_dim as 
    (select *
        ,rank() OVER ( PARTITION BY month2 ORDER BY date) AS week_of_month
    from
        (select * 
            ,CASE WHEN month = 2 and days >= 26 THEN year*100 + month+1
                when days >= 28 THEN year*100 + month+1 
                ELSE year*100 + month end AS month2
            ,lag(IsHoliday, -1, 0) over (order by date) AS IsHoliday_lag
            ,lag(IsHoliday_Dec, -1, 0) over (order by date) AS IsHoliday_Dec_lag
        from cal1
        )
    ORDER BY date
    )
select store ,str.Date ,Temperature ,Fuel_Price
    ,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5
    ,CPI,Unemployment
    ,Size
    ,Type --Type_A,Type_B,Type_C
    ,str.Week,str.Year,str.WeekYear
    ,cal.IsHoliday 
    ,cal.IsHoliday_Dec
    ,cal.IsHoliday_lag ,cal.IsHoliday_Dec_lag
    ,cal.week_of_month
from store_fact str
left join cal_dim cal
on str.date = cal.date
""")

# pd.set_option('display.max_rows', test.shape[0]+1)
# print(test)

pysqldf(
""" 
select
distinct date ,SUBSTR(date,0,11) as date2 ,CAST(SUBSTR(date,6,2) as int ) as month ,CAST(SUBSTR(date,9,2) as int ) as days ,year
        ,case when CAST(SUBSTR(date,6,2) as int) = 12 then 0 else IsHoliday end as IsHoliday 
        ,case when CAST(SUBSTR(date,6,2) as int) = 12 and year = 2010 and CAST(SUBSTR(date,9,2) as int ) in (24,31) then 1
           when CAST(SUBSTR(date,6,2) as int) = 12 and year = 2011 and CAST(SUBSTR(date,9,2) as int ) in (23,30) then 1
           when CAST(SUBSTR(date,6,2) as int) = 12 and year = 2012 and CAST(SUBSTR(date,9,2) as int ) in (21,28) then 1
           else 0 end as IsHoliday_Dec ,week
    FROM store_fact
   where CAST(SUBSTR(date,6,2) as int) = 12
""")

# pysqldf(
# """ 

#         SELECT
#             Store, 
#             Dept, 
#             Week, 
#             Weekly_Sales,
#             case 
#                 when Week = 52 then lag(Weekly_Sales) over(partition by Store, Dept) 
#             end as last_sales
#         from Final where Week>48
#         """)
train=train.drop(columns=['IsHoliday'])
test=test.drop(columns=['IsHoliday'])

train['Store_Dept'] = train['Store'].apply(str) + '_' + train['Dept'].apply(str)
test['Store_Dept'] = test['Store'].apply(str) + '_' + test['Dept'].apply(str)

store_fact.Date = pd.to_datetime(store_fact.Date)

# train2 - full data
train2 = train.merge(store_fact 
                           ,how='inner'
                           ,on=['Store','Date']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
test2 = test.merge(store_fact 
                           ,how='inner'
                           ,on=['Store','Date']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
train2.head()

# STORE CLUSTER
weekly_summary = train2.groupby(['Store','WeekYear']).sum()['Weekly_Sales'].reset_index()
df = weekly_summary.pivot(index='WeekYear', columns='Store', values='Weekly_Sales').reset_index().rename_axis(None, axis=1).drop(columns=['WeekYear'])
X = df.transpose().values

distortions = []
K = range(1,10)
for k in K:
    model = TimeSeriesKMeans(n_clusters=k, metric="euclidean", max_iter=10, n_init=2)
    model.fit(X)
    distortions.append(model.inertia_)
model.inertia_

plt.figure(figsize=(5,5))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
# model = TimeSeriesKMeans(n_clusters=5, metric="euclidean", max_iter=10, n_init=2)
model = TimeSeriesKMeans(n_clusters=6, metric="euclidean", max_iter=10, n_init=2)
model.fit(X)

# build helper df to map metrics to their cluster labels
df_scluster = pd.DataFrame(list(zip(df.columns, model.labels_)), columns=['Store', 'cluster'])
df_scluster.head()

stores_cluster = stores.merge(df_scluster, how='inner', on='Store')
stores_cluster = pd.get_dummies(stores_cluster, columns=['cluster'])

filter_col = [col for col in stores_cluster if (col.startswith('Type_') or col.startswith('cluster_'))]

# Plot Heatmap from variable's correlation
plt.figure(figsize=(5,5))
m = stores_cluster[filter_col].corr()
np.fill_diagonal(m.values, np.nan)
sns.heatmap(m, cmap='seismic', annot=True, fmt='.2f', annot_kws={"size": 9})
# DEPARTMENT CLUSTER
weekly_summary = train2.groupby(['Dept','WeekYear']).sum()['Weekly_Sales'].reset_index()
df = weekly_summary.pivot(index='WeekYear', columns='Dept', values='Weekly_Sales').reset_index().rename_axis(None, axis=1).drop(columns=['WeekYear'])

X = df.fillna(0).transpose().values

distortions = []
K = range(1,10)
for k in K:
    model = TimeSeriesKMeans(n_clusters=k, metric="euclidean", max_iter=10, n_init=2)
    model.fit(X)
    distortions.append(model.inertia_)
model.inertia_

plt.figure(figsize=(5,5))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
# model = TimeSeriesKMeans(n_clusters=4, metric="euclidean", max_iter=10, n_init=2)
model = TimeSeriesKMeans(n_clusters=6, metric="euclidean", max_iter=10, n_init=2)
model.fit(X)

# # build helper df to map metrics to their cluster labels
df_dcluster = pd.DataFrame(list(zip(df.columns, model.labels_)), columns=['Dept', 'dcluster'])
# STORE / DEPARTMENT CLUSTER
weekly_summary = train2.groupby(['Store_Dept','WeekYear']).sum()['Weekly_Sales'].reset_index()
df = weekly_summary.pivot(index='WeekYear', columns='Store_Dept', values='Weekly_Sales').reset_index().rename_axis(None, axis=1).drop(columns=['WeekYear'])

X = df.fillna(0).transpose().values

distortions = []
K = range(1,10)
for k in K:
    model = TimeSeriesKMeans(n_clusters=k, metric="euclidean", max_iter=10, n_init=2)
    model.fit(X)
    distortions.append(model.inertia_)
model.inertia_

plt.figure(figsize=(5,5))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
# model = TimeSeriesKMeans(n_clusters=5, metric="euclidean", max_iter=10, n_init=2)
model = TimeSeriesKMeans(n_clusters=7, metric="euclidean", max_iter=10, n_init=2)
model.fit(X)

# # build helper df to map metrics to their cluster labels
df_sdcluster = pd.DataFrame(list(zip(df.columns, model.labels_)), columns=['Store_Dept', 'sdcluster'])
# visualize clusters 
train3 = train2.merge(df_scluster, how='inner', on='Store')
train3 = train3.merge(df_dcluster, how='inner', on='Dept')
train3 = train3.merge(df_sdcluster, how='inner', on='Store_Dept')
train3.to_csv('cluster_test.csv',index=False)
#viz made in powerbi (adjusted store clusters)
def random_forest(n_estimators, max_depth):
    result = []
    for estimator in n_estimators:
        for depth in max_depth:
            wmaes_cv = []
            for i in range(1,3):
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
        for i in range(1,3):
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
            for i in range(1,3):
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

def XGBRegressor_I(n_estimators, early_stopping_rounds ,learning_rate):
    result = []
    for estimator in n_estimators:
        for rounds in early_stopping_rounds:
            for rates in learning_rate:
                wmaes_cv = []
                for i in range(1,3):
                    print('k:', i, ', n_estimators:', estimator, ', early_stopping_rounds:', rounds, ', learning_rate:', rates)
                    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)

                    XGBR = XGBRegressor(n_estimators=estimator, early_stopping_rounds=rounds ,learning_rate=rates)
                    XGBR.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)

                    predicted = XGBR.predict(x_test)
                    wmaes_cv.append(WMAE(x_test, y_test, predicted))
                print('WMAE:', np.mean(wmaes_cv))
                result.append({'Early Stopping Rounds': rounds, 'Estimators': estimator, 'WMAE': np.mean(wmaes_cv)})
    return pd.DataFrame(result)
var_list = ['Dept','IsHoliday' ,'IsHoliday_Dec','Week','Year','Type','Size','cluster','dcluster']

X_train = train3[var_list]
Y_train = train3['Weekly_Sales']

n_estimators = [25,50,75,125]
max_depth = [20,25]
random_forest(n_estimators, max_depth)
var_list = ['Dept','IsHoliday' ,'IsHoliday_Dec','Week','Year','Type','Size','cluster','dcluster']

X_train = train3[var_list]
Y_train = train3['Weekly_Sales']

n_estimators = [125]
max_depth = [24,26]
random_forest(n_estimators, max_depth)
var_list = ['Dept','IsHoliday' ,'IsHoliday_Dec','Week','Year','Type','Size','cluster','dcluster']

X_train = train3[var_list]
Y_train = train3['Weekly_Sales']

n_estimators = [125]
max_depth = [27]
random_forest(n_estimators, max_depth)
var_list =  ['Store','Dept','IsHoliday' ,'IsHoliday_Dec','Week','Year','Type','Size']

X_train = train3[var_list]
Y_train = train3['Weekly_Sales']

n_estimators = [25,50,75,125]
max_depth = [20,25]
random_forest(n_estimators, max_depth)
var_list =  ['Store','Dept','IsHoliday' ,'IsHoliday_Dec','Week','Year','Type','Size']

X_train = train3[var_list]
Y_train = train3['Weekly_Sales']

n_estimators = [75]
max_depth = [24,26]
random_forest(n_estimators, max_depth)
var_list =  ['Store','Dept','IsHoliday' ,'IsHoliday_Dec','Week','Year','Type','Size']

X_train = train3[var_list]
Y_train = train3['Weekly_Sales']

n_estimators = [75]
max_depth = [22,23]
random_forest(n_estimators, max_depth)
var_list = ['Dept','IsHoliday' ,'IsHoliday_Dec','Week','Year','Type','Size','cluster','dcluster']
X_train = train3[var_list]
Y_train = train3['Weekly_Sales']
max_features = ['auto' ,'sqrt', 'log2', 5, 7]
random_forest_II(n_estimators=125, max_depth=26, max_features=max_features)
var_list =  ['Store','Dept','IsHoliday' ,'IsHoliday_Dec','Week','Year','Type','Size']
X_train = train3[var_list]
Y_train = train3['Weekly_Sales']
max_features = ['auto' ,'sqrt', 'log2', 5, 7]
random_forest_II(n_estimators=75, max_depth=23, max_features=max_features)
var_list = ['Dept','IsHoliday' ,'IsHoliday_Dec','Week','Year','Type','Size','cluster','dcluster']
X_train = train3[var_list]
Y_train = train3['Weekly_Sales']

min_samples_split = [2, 3, 4]
min_samples_leaf = [1, 2, 3]
random_forest_III(n_estimators=125, max_depth=26, max_features=7, 
                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
var_list =  ['Store','Dept','IsHoliday' ,'IsHoliday_Dec','Week','Year','Type','Size']
X_train = train3[var_list]
Y_train = train3['Weekly_Sales']

min_samples_split = [2, 3, 4]
min_samples_leaf = [1, 2, 3]
random_forest_III(n_estimators=75, max_depth=23, max_features='auto', 
                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
var_list =  ['Store','Dept','IsHoliday' ,'IsHoliday_Dec','Week','Year','Type','Size','cluster','dcluster']
X_train = train3[var_list]
Y_train = train3['Weekly_Sales']


n_est = [3000]
esr = [5]
lrs = [0.1,0.5,0.8]
XGBRegressor_I(n_est,esr,lrs)
var_list =  ['Store','Dept','IsHoliday' ,'IsHoliday_Dec','Week','Year','Type','Size','cluster','dcluster']
X_train = train3[var_list]
Y_train = train3['Weekly_Sales']


n_est = [4000]
esr = [5]
lrs = [0.1,0.5]
XGBRegressor_I(n_est,esr,lrs)
var_list = ['Dept','IsHoliday' ,'IsHoliday_Dec','Week','Year','Type','Size','cluster','dcluster']

X_train = train3[var_list]
Y_train = train3['Weekly_Sales']

RF = RandomForestRegressor(n_estimators=125, max_depth=26, max_features=7, min_samples_split=4, min_samples_leaf=1)
RF.fit(X_train, Y_train)

test3 = test2.merge(df_scluster, how='left', on='Store')
test3 = test3.merge(df_dcluster, how='left', on='Dept')
# test3 = test3.merge(df_sdcluster, how='left', on='Store_Dept')

X_test = test3[var_list]

# X_test.fillna(7, inplace=True)
predict = RF.predict(X_test)
Final = test3[['Store', 'Dept', 'Week']]
Final['Weekly_Sales'] = predict
var_list = ['Store','Dept','IsHoliday' ,'IsHoliday_Dec','Week','Year','Type','Size']

X_train = train3[var_list]
Y_train = train3['Weekly_Sales']

RF = RandomForestRegressor(n_estimators=75, max_depth=23, max_features='auto', min_samples_split=3, min_samples_leaf=1)
RF.fit(X_train, Y_train)

test3 = test2.merge(df_scluster, how='left', on='Store')
test3 = test3.merge(df_dcluster, how='left', on='Dept')
# test3 = test3.merge(df_sdcluster, how='left', on='Store_Dept')

X_test = test3[var_list]

# X_test.fillna(7, inplace=True)
predict = RF.predict(X_test)
Final_rf2 = test3[['Store', 'Dept', 'Week']]
Final_rf2['Weekly_Sales'] = predict
estimator = 3000
rounds = 5
rates = 0.5

var_list = ['Store','Dept','IsHoliday' ,'IsHoliday_Dec','Week','Year','Type','Size','cluster','dcluster']

X_train = train3[var_list]
Y_train = train3['Weekly_Sales']

XGBR = XGBRegressor(n_estimators=estimator, early_stopping_rounds=rounds ,learning_rate=rates)
XGBR.fit(X_train, Y_train, verbose=False)
      
test3 = test2.merge(df_scluster, how='left', on='Store')
test3 = test3.merge(df_dcluster, how='left', on='Dept')

X_test = test3[var_list]
predict = XGBR.predict(X_test)

Final_XGBR = test3[['Store', 'Dept', 'Week']]
Final_XGBR['Weekly_Sales'] = predict
estimator = 4000
rounds = 5
rates = 0.5

var_list = ['Store','Dept','IsHoliday' ,'IsHoliday_Dec','Week','Year','Type','Size','cluster','dcluster']

X_train = train3[var_list]
Y_train = train3['Weekly_Sales']

XGBR = XGBRegressor(n_estimators=estimator, early_stopping_rounds=rounds ,learning_rate=rates)
XGBR.fit(X_train, Y_train, verbose=False)
      
test3 = test2.merge(df_scluster, how='left', on='Store')
test3 = test3.merge(df_dcluster, how='left', on='Dept')

X_test = test3[var_list]
predict = XGBR.predict(X_test)

Final_XGBR2 = test3[['Store', 'Dept', 'Week']]
Final_XGBR2['Weekly_Sales'] = predict
# Final_adj = pysqldf("""
#     with trans as 
#         (select cluster ,dcluster ,year ,sum(Weekly_Sales) as Weekly_Sales
#         from train3 where week <= 43 and week >=32
#         group by cluster ,dcluster ,year)

#     ,lkp as 
#         (select distinct cluster,store ,dept ,dcluster from train3)

#     ,final_pred as
#         (SELECT a.Store ,a.Dept ,a.Week ,(a.Weekly_Sales*0.25 + b.Weekly_Sales*0.25 + c.Weekly_Sales*0.25 + d.Weekly_Sales*0.25) as Weekly_Sales
#          from Final a
#          left join  Final_XGBR b
#          on a.Store = b.Store
#          and a.Dept = b.Dept
#          and a.Week = b.Week
#          left join  Final_rf2 c
#          on a.Store = c.Store
#          and a.Dept = c.Dept
#          and a.Week = c.Week
#          left join  Final_XGBR2 d
#          on a.Store = d.Store
#          and a.Dept = d.Dept
#          and a.Week = d.Week
#         )
#     SELECT fin.Store ,fin.Dept ,Week ,Weekly_Sales
#          ,case when Week = 52 and last_sales > 2*Weekly_Sales then Weekly_Sales+(2.5/7)*last_sales 
#             when week > 43 and week <=51 and growth > 1.01 then 1.01*Weekly_Sales 
#             when week > 43 and week <=51 and growth < 0.99 then 0.99*Weekly_Sales 
#             when week > 43 and week <=51 and growth is null then 1*Weekly_Sales 
#             when week > 43 and week <=51 then growth*Weekly_Sales 
#              else Weekly_Sales 
#         end as Weekly_Sales_Adjusted
#     from
#         (SELECT Store ,Dept ,Week ,Weekly_Sales
#             ,case when Week = 52 then lag(Weekly_Sales) over(partition by Store, Dept)  end as last_sales
#         from final_pred
#         ) fin
#     left join lkp on fin.store = lkp.store and fin.dept = lkp.dept
#     --
#     left join 
#         (select a.cluster ,a.dcluster ,a.Weekly_Sales/b.Weekly_Sales as growth
#         from trans a
#         inner join trans b
#         on a.cluster= b.cluster 
#         and a.dcluster = b.dcluster
#         where a.year = 2012 and b.year = 2011
#         ) b
#     on lkp.dcluster = b.dcluster
#     and lkp.cluster = b.cluster
#     left join (select store ,dept ,min(Weekly_Sales) as Min_Weekly_Sales from train3 group by store ,dept) c
#     on fin.Dept = c.Dept
#     and fin.store = c.store 
# """)

# sample_submission['Weekly_Sales'] = Final_adj['Weekly_Sales_Adjusted']
# sample_submission.to_csv('submission.csv',index=False)

# from xgboost import XGBRegressor

# x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)

# my_model = XGBRegressor()
# # Add silent=True to avoid printing out updates with each cycle
# my_model.fit(x_train, y_train, verbose=False)

# # make predictions
# # predicted = my_model.predict(x_test)

# from sklearn.metrics import mean_absolute_error
# print("WMAE: " + str(WMAE(x_test, y_test, predicted)))

# XGBR = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# XGBR.fit(x_train, y_train, early_stopping_rounds=1, eval_set=[(x_test, y_test)], verbose=False)
    
# # my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# # my_model.fit(x_train, y_train, early_stopping_rounds=5, 
# #              eval_set=[(x_test, y_test)], verbose=False)






# #Retrodiction of Markdown data
# store_fact = store_fact.fillna(0)

# store_fact_temp  = pysqldf("""
#     select a.Store 
#         ,case when a.MarkDown1 =  0 then b.MarkDown1 else a.MarkDown1 end as MarkDown1
#         ,case when a.MarkDown2 =  0 then b.MarkDown2 else a.MarkDown2 end as MarkDown2
#         ,case when a.MarkDown3 =  0 then b.MarkDown3 else a.MarkDown3 end as MarkDown3
#         ,case when a.MarkDown4 =  0 then b.MarkDown4 else a.MarkDown4 end as MarkDown4
#         ,case when a.MarkDown5 =  0 then b.MarkDown5 else a.MarkDown5 end as MarkDown5
#         ,WeekYear
#     from store_fact a
#     left join 
#         (select Week,Year-1 as Year,Store ,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5
#         from store_fact a
#         where year = 2012
#         union 
#         select Week,Year-2 as Year,Store ,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5
#         from store_fact a
#         where year = 2012
#         ) b
#     on a.Store =  b.Store
#     and a.Week =  b.Week
#     and a.Year =  b.Year
#     """)
# store_fact['MarkDown1'] = store_fact_temp['MarkDown1']
# store_fact['MarkDown2'] = store_fact_temp['MarkDown2']
# store_fact['MarkDown3'] = store_fact_temp['MarkDown3']
# store_fact['MarkDown4'] = store_fact_temp['MarkDown4']
# store_fact['MarkDown5'] = store_fact_temp['MarkDown5']
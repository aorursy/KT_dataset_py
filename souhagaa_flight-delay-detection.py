import datetime, warnings, scipy 

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.patches as patches

from matplotlib.patches import ConnectionPatch

from collections import OrderedDict

from matplotlib.gridspec import GridSpec

from sklearn import metrics, linear_model

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from scipy.optimize import curve_fit

from catboost import CatBoostRegressor

from sklearn.model_selection import GridSearchCV

from datetime import datetime





plt.rcParams["patch.force_edgecolor"] = True

plt.style.use('fivethirtyeight')

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "last_expr"

pd.options.display.max_columns = 50

%matplotlib inline

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/Train.csv')

df_test = pd.read_csv('../input/Test.csv')
df.head()
df['DATOP'].unique()
df.info()
df.shape
df['target'].mean()
# # df[['DATOP', 'STD', 'STA']]

df['STD'] =  pd.to_datetime(df['STD'], format='%Y-%m-%d %H:%M:%S')

df['STA'] =  pd.to_datetime(df['STA'], format='%Y-%m-%d %H.%M.%S')

df['DATOP'] =  pd.to_datetime(df['DATOP'], format='%Y-%m-%d')



df_test['STD'] =  pd.to_datetime(df_test['STD'], format='%Y-%m-%d %H:%M:%S')

df_test['STA'] =  pd.to_datetime(df_test['STA'], format='%Y-%m-%d %H.%M.%S')

df_test['DATOP'] =  pd.to_datetime(df_test['DATOP'], format='%Y-%m-%d')
# there is a space after the flight id

df['FLTID'] = df['FLTID'].astype(str).str[:-1]

df_test['FLTID'] = df_test['FLTID'].astype(str).str[:-1]
df['FLTID'][0]
df['STATUS'].unique()
df.isnull().sum(axis=0).reset_index()
df['AC'].unique()
df['target'].mean()
# function that extract statistical parameters from a grouby objet:

def get_stats(group):

    return {'min': group.min(), 'max': group.max(),

            'count': group.count(), 'mean': group.mean()}

#_______________________________________________________________

# Creation of a dataframe with statitical infos on each airline:

global_stats = df['target'].groupby(df['AC']).apply(get_stats).unstack()

global_stats = global_stats.sort_values('count')

global_stats
global_stats.sort_values('max')
font = {'family' : 'normal', 'weight' : 'bold', 'size'   : 18}

mpl.rc('font', **font)

import matplotlib.patches as mpatches

#__________________________________________________________________

df2 = df.loc[:, ['AC', 'target']]

#________________________________________________________________________

colors = ['royalblue', 'grey', 'wheat', 'c', 'firebrick', 'seagreen', 'lightskyblue',

          'lightcoral', 'yellowgreen', 'gold', 'tomato', 'violet', 'aquamarine', 'chartreuse']



fig = plt.figure(1, figsize=(70,70))

gs=GridSpec(2,2)             

ax1=fig.add_subplot(gs[0,0]) 

# Pie chart nº1: nb of flights

labels = [s for s in  global_stats.index]

sizes  = global_stats['count'].values

explode = [0.3 if sizes[i] < 1.0 else 0.0 for i in range(len(sizes))]

patches, texts, autotexts = ax1.pie(sizes, explode=explode,

                                labels=labels, colors = colors,  autopct='%1.0f%%',

                                shadow=False, startangle=0)

ax1.axis('equal')

# ax1.set_title('% of flights per company', bbox={'facecolor':'midnightblue', 'pad':10},

#               color = 'w',fontsize=18)
# ax2=fig.add_subplot(gs[0,1]) 

# Pie chart nº2: mean delay at departure

fig = plt.figure(1, figsize=(50,50))

ax2=fig.add_subplot(gs[0,0])

labels = [s for s in  global_stats.index]

sizes  = global_stats['mean'].values

sizes  = [max(s,0) for s in sizes]

patches, texts, autotexts = ax2.pie(sizes, labels = labels,

                                colors = colors, shadow=False, startangle=0,

                                autopct = lambda p :  '{:.0f}'.format(p * sum(sizes) / 100))

ax2.axis('equal')

# ax2.set_title('Mean delay', bbox={'facecolor':'midnightblue', 'pad':5},

#               color='w', fontsize=24)

plt.show()
fig = plt.figure(1, figsize=(25,25))

df2 = df.loc[:, ['AC', 'target']]

df2 = df2.drop(df2[df2.target == 0].index)

ax = sns.stripplot(x="target", y="AC", data=df2, size = 12, linewidth = 2,  jitter=True)

plt.xlabel('Departure delay', fontsize=18, bbox={'facecolor':'midnightblue', 'pad':5},

           color='w', labelpad=30)

ax3.yaxis.label.set_visible(False)

plt.tight_layout(w_pad=3) 
# Function that define how delays are grouped

delay_type = lambda x:((0,1)[x > 5],2)[x > 45]

df['DELAY_LEVEL'] = df['target'].apply(delay_type)

#____________________________________________________

fig = plt.figure(1, figsize=(70,70))

ax = sns.countplot(y="AC", hue='DELAY_LEVEL', data=df)

#____________________________________________________________________________________

# We replace the abbreviations by the full names of the companies and set the labels

labels = df['AC'].unique().tolist()

ax.set_yticklabels(labels)

plt.setp(ax.get_xticklabels(), fontsize=24, weight = 'normal', rotation = 0);

plt.setp(ax.get_yticklabels(), fontsize=24, weight = 'bold', rotation = 0);

ax.yaxis.label.set_visible(False)

plt.xlabel('Flight count', fontsize=24, weight = 'bold', labelpad=10)

#________________

# Set the legend

L = plt.legend()

L.get_texts()[0].set_text('on time (t < 5 min)')

L.get_texts()[1].set_text('small delay (5 < t < 45 min)')

L.get_texts()[2].set_text('large delay (t > 45 min)')

plt.show()
print("Nb of airports: {}".format(len(df['DEPSTN'].unique())))
list_of_airpots = df['DEPSTN'].unique().tolist()

list_of_ac = df['AC'].unique().tolist()

origin_nb = dict()

for carrier in list_of_ac:

    liste_origin_airport = df[df['AC'] == carrier]['DEPSTN'].unique().tolist()

    origin_nb[carrier] = len(liste_origin_airport)
test_df = pd.DataFrame.from_dict(origin_nb, orient='index')

test_df.rename(columns = {0:'count'}, inplace = True)

ax = test_df.plot(kind='bar', figsize = (25,25))

labels = [x for x in list_of_ac]

ax.set_xticklabels(labels)

plt.ylabel('Number of airports visited', fontsize=14, weight = 'bold', labelpad=12)

plt.setp(ax.get_xticklabels(), fontsize=11, ha = 'right', rotation = 80)

ax.legend().set_visible(False)

plt.show()
# df['month'] = pd.DatetimeIndex(df['DATOP']).month

df['year'] = pd.DatetimeIndex(df['DATOP']).year

df.head()
flights_dict = dict()

ac_group = df.groupby('AC')

for x in list_of_ac:

    df_ac = ac_group.get_group(x)

    flights_dict[x] = df[df['AC'] == x]['year'].value_counts().to_dict()
flights_dict
# flights_dict

nbr_flights = pd.DataFrame.from_dict(flights_dict, orient='index')

nbr_flights.fillna(0, inplace=True)

# nbr_flights
nbr_flights.columns = ['flights_in_2018', 'flights_in_2017', 'flights_in_2016']
nbr_flights['flights_in_2016'] = nbr_flights['flights_in_2016'].astype(np.int64)

nbr_flights['flights_in_2017'] = nbr_flights['flights_in_2017'].astype(np.int64)

nbr_flights['flights_in_2018'] = nbr_flights['flights_in_2018'].astype(np.int64)

nbr_flights['total_flights'] = nbr_flights['flights_in_2016'] + nbr_flights['flights_in_2017'] + nbr_flights['flights_in_2018']

# nbr_flights
nbr_flights['total_flights'].unique().min()
nbr_flights['total_flights'].unique().max()
nbr_flights['flights_in_2018'].unique()
# add a variable describing the frequency of that AC's flights 3 levels (100 <, 100< < 200, >200) but per year

df['ac_frequency'] = 0

def define_frequency(nbr):

        if nbr<100:

            return 0

        

        else:

            if (nbr>100 and nbr<200): 

                return 1

            else:

                return 2
for i in range(df.shape[0]):

    ac = df.loc[i, 'AC']

    year = df.loc[i, 'year']

    if year == 2016:

        freq_2016 = define_frequency(nbr_flights.loc[ac, 'flights_in_2016'])

        if freq_2016 !=0:

            df.at[i, 'ac_frequency'] = freq_2016

    else:

        if year == 2017:

            freq_2017 = define_frequency(nbr_flights.loc[ac, 'flights_in_2017'])

            if freq_2017 !=0:

                df.at[i, 'ac_frequency'] = freq_2017

        else:

            freq_2018 = define_frequency(nbr_flights.loc[ac, 'flights_in_2018'])

            if freq_2018 !=0:

                df.at[i, 'ac_frequency'] = freq_2018
df.to_csv('Train_interm.csv', index=False)
airport_mean_delays = pd.DataFrame(pd.Series(df['DEPSTN'].unique()))

airport_mean_delays.set_index(0, drop = True, inplace = True)



for carrier in list_of_ac:

    df1 = df[df['AC'] == carrier]

    test = df1['target'].groupby(df['DEPSTN']).apply(get_stats).unstack()

    airport_mean_delays[carrier] = test.loc[:, 'mean'] 
airport_mean_delays
sns.set(context="paper")

fig = plt.figure(1, figsize=(15,15))



ax = fig.add_subplot(1,2,1)

subset = airport_mean_delays.iloc[:34,:]

mask = subset.isnull()

sns.heatmap(subset, linewidths=0.01, cmap="Accent", mask=mask, vmin = 0, vmax = 35)

plt.setp(ax.get_xticklabels(), fontsize=10, rotation = 85) ;

ax.yaxis.label.set_visible(False)



ax = fig.add_subplot(1,2,2)

subset = airport_mean_delays.iloc[34:,:]

fig.text(0.5, 1.02, "Delays: impact of the origin airport", ha='center', fontsize = 18)

mask = subset.isnull()

sns.heatmap(subset, linewidths=0.01, cmap="Accent", mask=mask, vmin = 0, vmax = 35)

plt.setp(ax.get_xticklabels(), fontsize=10, rotation = 85) ;

ax.yaxis.label.set_visible(False)



plt.tight_layout()
df.head()
df = pd.read_csv('/home/souhagaa/Bureau/hackathon/tunisair/Train_interm.csv')
df.head()
df['trajectory'] = df['DEPSTN'] + '-' + df['ARRSTN']

df_test['trajectory'] = df_test['DEPSTN'] + '-' + df_test['ARRSTN']
df['trajectory'].nunique()
df['month'] = pd.DatetimeIndex(df['DATOP']).month

df['day'] = pd.DatetimeIndex(df['DATOP']).day

df['day_of_week'] = pd.DatetimeIndex(df['DATOP']).dayofweek

df['year'] = pd.DatetimeIndex(df['DATOP']).year

df['week_of_year'] = pd.DatetimeIndex(df['DATOP']).week
df_test['month'] = pd.DatetimeIndex(df_test['DATOP']).month

df_test['day'] = pd.DatetimeIndex(df_test['DATOP']).day

df_test['day_of_week'] = pd.DatetimeIndex(df_test['DATOP']).dayofweek

df_test['year'] = pd.DatetimeIndex(df_test['DATOP']).year

df_test['week_of_year'] = pd.DatetimeIndex(df_test['DATOP']).week
# Adding a week of month variable

data = [df, df_test]

for dataset in data:

    dataset.loc[ dataset['day'] <= 7, 'week_of_month'] = 0

    dataset.loc[(dataset['day'] > 7) & (dataset['day'] <= 14), 'week_of_month'] = 1

    dataset.loc[(dataset['day'] > 14) & (dataset['day'] <= 21), 'week_of_month'] = 2

    dataset.loc[(dataset['day'] > 21) & (dataset['day'] <= 28), 'week_of_month'] = 3

    dataset.loc[(dataset['day'] > 28) & (dataset['day'] <= 31), 'week_of_month'] = 4

    dataset['week_of_month'] = dataset['week_of_month'].astype(int)
# Adding a season column depicting the season the flight has taken place in

data = [df, df_test]

for dataset in data:

    dataset.loc[ (dataset['month'] < 3) | (dataset['month'] == 12), 'season'] = 0

    dataset.loc[(dataset['month'] >= 3) & (dataset['month'] < 6), 'season'] = 1

    dataset.loc[(dataset['month'] >= 6) & (dataset['month'] < 9), 'season'] = 2

    dataset.loc[(dataset['month'] >= 9) & (dataset['month'] < 12), 'season'] = 3

    dataset['season'] = dataset['season'].astype(int)
df['dep_hour'] = pd.DatetimeIndex(df['STD']).hour 

df_test['dep_hour'] = pd.DatetimeIndex(df_test['STD']).hour





df['arr_hour'] = pd.DatetimeIndex(df['STA']).hour 

df_test['arr_hour'] = pd.DatetimeIndex(df_test['STA']).hour
df['dep_minute'] = pd.DatetimeIndex(df['STD']).minute

df_test['dep_minute'] = pd.DatetimeIndex(df_test['STD']).minute



df['arr_minute'] = pd.DatetimeIndex(df['STA']).minute 

df_test['arr_minute'] = pd.DatetimeIndex(df_test['STA']).minute
df['flight_duration_sec'] = (df['STA'] - df['STD']).values.astype(np.int64) // 10 ** 9

df_test['flight_duration_sec'] = (df_test['STA'] - df_test['STD']).values.astype(np.int64) // 10 ** 9
df['flight_duration_hours'] = df['arr_hour'] - df['dep_hour'] 

df_test['flight_duration_hours'] = df_test['arr_hour'] - df_test['dep_hour']



df['flight_duration_minutes'] = (df['flight_duration_sec'] / 60).astype(np.int64)

df_test['flight_duration_minutes'] = (df_test['flight_duration_sec'] / 60).astype(np.int64)
data = [df, df_test]

for dataset in data:

    dataset.loc[ (dataset['dep_hour'] < 12) , 'dep_hour_AM_PM'] = 0

    dataset.loc[(dataset['dep_hour'] >= 12) , 'dep_hour_AM_PM'] = 1

    dataset['dep_hour_AM_PM'] = dataset['dep_hour_AM_PM'].astype(int)



    dataset.loc[ (dataset['arr_hour'] < 12) , 'arr_hour_AM_PM'] = 0

    dataset.loc[(dataset['arr_hour'] >= 12) , 'arr_hour_AM_PM'] = 1

    dataset['arr_hour_AM_PM'] = dataset['arr_hour_AM_PM'].astype(int)
df['S_dep_hour'] = np.sin(2*np.pi*df['dep_hour']/24)

df['C_dep_hour'] = np.cos(2*np.pi*df['dep_hour']/24)

df_test['S_dep_hour'] = np.sin(2*np.pi*df_test['dep_hour']/24)

df_test['C_dep_hour'] = np.cos(2*np.pi*df_test['dep_hour']/24)





df['S_arr_hour'] = np.sin(2*np.pi*df['arr_hour']/24)

df['C_arr_hour'] = np.cos(2*np.pi*df['arr_hour']/24)

df_test['S_arr_hour'] = np.sin(2*np.pi*df_test['arr_hour']/24)

df_test['C_arr_hour'] = np.cos(2*np.pi*df_test['arr_hour']/24)
# '2016-01-03'.strftime('%j')

data = [df, df_test]

for dataset in data:

    for i in range(dataset.shape[0]):

        dataset.loc[i, 'day_of_year'] = int(dataset.loc[i, 'DATOP'].strftime('%j'))

# int(df['DATOP'].datetime.strftime('%j'))
corrmat = df.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
df.shape
# df = pd.read_csv('/home/souhagaa/Bureau/hackathon/tunisair/Train_clean.csv')
# we shuffle data

# df = df.sample(frac=1).reset_index(drop=True)
df.head()
# !!!! only in case df is read from the file we need to convert the types back to datetime objects

# df['STD'] =  pd.to_datetime(df['STD'], format='%Y-%m-%d %H:%M:%S')

# df['STA'] =  pd.to_datetime(df['STA'], format='%Y-%m-%d %H:%M:%S')

# df['DATOP'] =  pd.to_datetime(df['DATOP'], format='%Y-%m-%d')
df.to_csv('/home/souhagaa/Bureau/hackathon/tunisair/Train_clean.csv', index=False)
df.columns
train_cols = ['DATOP', 'FLTID', 'DEPSTN', 'ARRSTN', 'STD', 'STA', 'STATUS',

       'AC', 'trajectory', 'month', 'day', 'day_of_week', 'year',

       'week_of_year', 'week_of_month', 'season', 'dep_hour', 'arr_hour',

       'dep_minute', 'arr_minute', 'flight_duration_sec',

       'flight_duration_hours', 'flight_duration_minutes', 'dep_hour_AM_PM',

       'arr_hour_AM_PM', 'S_dep_hour', 'C_dep_hour', 'S_arr_hour',

       'C_arr_hour', 'day_of_year']

X = df[train_cols]

test = df_test[train_cols]

y = df['target']
# y = df['target']

# del df['target']

# X = df
X.head()
# del X['year']

# del X['ac_frequency']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=17)
categ_feat_idx = np.where(X_train.dtypes == 'object')[0]

categ_feat_idx
# from catboost import CatBoostRegressor

# model = CatBoostRegressor(iterations=700, depth= 10, l2_leaf_reg= 7, learning_rate= 0.1) 

# #depth= 10, l2_leaf_reg= 7, learning_rate= 0.1

# model.fit(X_train,y_train,verbose=False,cat_features=categ_feat_idx)



# from catboost import CatBoostRegressor

model = CatBoostRegressor(iterations=699, depth=10, learning_rate=0.1,l2_leaf_reg= 7, loss_function='RMSE')

model.fit(X_train, y_train,cat_features=categ_feat_idx,eval_set=(X_test, y_test),plot=True)
predictions = model.predict(X_test)

from math import sqrt

from sklearn.metrics import mean_squared_error



rmse = sqrt(mean_squared_error(y_test,predictions))

print(rmse)

#106.61722528055746

model = CatBoostRegressor()

parameters = {'depth'         : [6,8,10],

              'learning_rate' : [0.01, 0.05, 0.1],

              'iterations'    : [650, 700, 750]

             }

grid = GridSearchCV(estimator=model, param_grid = parameters, cv = 2, n_jobs=-1)

grid.fit(X_train, y_train,cat_features=categ_feat_idx,eval_set=(X_test, y_test))    



# Results from Grid Search

print("\n========================================================")

print(" Results from Grid Search " )

print("========================================================")    



print("\n The best estimator across ALL searched params:\n",

      grid.best_estimator_)



print("\n The best score across ALL searched params:\n",

      grid.best_score_)



print("\n The best parameters across ALL searched params:\n",

      grid.best_params_)
importances = pd.DataFrame({'feature':X.columns,'importance':np.round(model.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances
importances.plot.bar()

# I again fit the model but with the whole dataset to get more data into training

# model.fit(X, y,cat_features=categ_feat_idx,eval_set=(X_test, y_test))
y_pred = model.predict(test)
any(y_pred<0)
y_pred[y_pred < 0] =0
submission= pd.DataFrame({'ID':df_test["ID"],'target':y_pred})

submission.to_csv("submission.csv",index=False)
one_hot = pd.get_dummies(X['STATUS'])

X = X.drop('STATUS',axis = 1)

X = X.join(one_hot)
one_hot = pd.get_dummies(test['STATUS'])

test = test.drop('STATUS',axis = 1)

test = test.join(one_hot)
# one_hot = pd.get_dummies(df['DEPSTN'],prefix='depart_')

# df = df.drop('DEPSTN',axis = 1)

# df = df.join(one_hot)
# one_hot = pd.get_dummies(df['ARRSTN'], prefix='arrival_')

# df = df.drop('ARRSTN',axis = 1)

# df = df.join(one_hot)
# one_hot = pd.get_dummies(df['AC'], prefix='ac_')

# df = df.drop('AC',axis = 1)

# df = df.join(one_hot)
le = LabelEncoder()

# enc = OneHotEncoder(sparse=False)

X['DEPSTN'] = le.fit_transform(X['DEPSTN'])

X['ARRSTN'] = le.fit_transform(X['ARRSTN'])

X['AC'] = le.fit_transform(X['AC'])

X['FLTID'] = le.fit_transform(X['FLTID'])

X['trajectory'] = le.fit_transform(X['trajectory'])
test['DEPSTN'] = le.fit_transform(test['DEPSTN'])

test['ARRSTN'] = le.fit_transform(test['ARRSTN'])

test['AC'] = le.fit_transform(test['AC'])

test['FLTID'] = le.fit_transform(test['FLTID'])

test['trajectory'] = le.fit_transform(test['trajectory'])
X.head()
X['DATOP_ts'] = X.DATOP.values.astype(np.int64) // 10 ** 9

X['STD_ts'] = X.STD.values.astype(np.int64) // 10 ** 9

X['STA_ts'] = X.STA.values.astype(np.int64) // 10 ** 9

del X['DATOP']

del X['STA']

del X['STD']
test['DATOP_ts'] = test.DATOP.values.astype(np.int64) // 10 ** 9

test['STD_ts'] = test.STD.values.astype(np.int64) // 10 ** 9

test['STA_ts'] = test.STA.values.astype(np.int64) // 10 ** 9

del test['DATOP']

del test['STA']

del test['STD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=17)
model = linear_model.LinearRegression()

model.fit(X_train, y_train)
results = model.predict(X_test)

score1 = np.sqrt(metrics.mean_squared_error(y_test,results))

score2 = metrics.r2_score(y_test,results )

print('RMSE: ',score1, '  R2 Score: ', score2)

# MSE:  5954526.582315134   R2 Score:  -461.04557347688393

# when deleting year and ac_frequency MSE:  13643.94079805509   R2 Score:  0.058992522154372895

# WITH label encoding MSE:  12795.027061727462   R2 Score:  0.036762754570380474

# MSE:  12059.485969460224   R2 Score:  0.0421820337368739 labelenc with delet of year and ac_freq
plt.scatter(X_train['AC'], y_train)

#memory error

for i in range(1,3):

    poly = PolynomialFeatures(degree = i)

    regr = linear_model.LinearRegression()

    X_ = poly.fit_transform(X_train)

    regr.fit(X_, y_train)

    X_ = poly.fit_transform(X_test)

    results = regr.predict(X_)

    score1 = np.sqrt(metrics.mean_squared_error(y_test,results))

    score2 = metrics.r2_score(y_test,results )

    print('Model with Polynominal Degree', i, 'MSE: ',score1, '  R2 Score: ', score2)

# 5 : Model with Polynominal Degree 5 MSE:  11915.27537981318   R2 Score:  0.053635879617132454

clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)

clf.fit(X_train, y_train)
results = clf.predict(X_test)

score1 = np.sqrt(metrics.mean_squared_error(y_test,results))

score2 = metrics.r2_score(y_test,results )

print('RMSE: ',score1, '  R2 Score: ', score2)
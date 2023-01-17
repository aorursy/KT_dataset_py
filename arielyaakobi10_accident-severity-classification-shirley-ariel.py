# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime as dt

import time

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as split

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.linear_model import LogisticRegression

from pandas.tools.plotting import scatter_matrix

import warnings

from sklearn.metrics import roc_auc_score

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, OneHotEncoder, KBinsDiscretizer, MaxAbsScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer

import seaborn as sns

sns.set()

import math



warnings.filterwarnings('ignore')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Load Data and encode to latin

acc = pd.read_csv('../input/Accident_Information.csv', encoding = 'latin')

veh = pd.read_csv('../input/Vehicle_Information.csv', encoding = 'latin')



# Merging two data sets into one with inner join by index

df = pd.merge(veh, acc, how = 'inner', on = 'Accident_Index')



#Check data sample

print(df.shape)

df.head()
#Distribution of original data by targets



ax = sns.countplot(x = df.Accident_Severity ,palette="Set2")

sns.set(font_scale=1)

ax.set_xlabel(' ')

ax.set_ylabel(' ')

fig = plt.gcf()

fig.set_size_inches(8,4)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(df.Accident_Severity)), (p.get_x()+ 0.3, p.get_height()+10000))



plt.title('Distribution of 2 Million Targets',)

plt.xlabel('Accident Severity')

plt.ylabel('Frequency [%]')

plt.show()

# Creating weights that are opposite to the weights of target

weights = np.where(df['Accident_Severity'] == 'Slight', .2, .8)



#Sampling only 30% of the data with new weights  

df = df.sample(frac=0.3, replace=True, weights=weights)

print(df.shape)

#df.Accident_Severity.value_counts(normalize=True)
#Distribution of sample data by targets



ax = sns.countplot(x = df.Accident_Severity ,palette="Set2")

sns.set(font_scale=1.5)

ax.set_xlabel(' ')

ax.set_ylabel(' ')

fig = plt.gcf()

fig.set_size_inches(8,4)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(df.Accident_Severity)), (p.get_x()+ 0.3, p.get_height()+10000))



plt.title('Distribution of 600K Targets',)

plt.xlabel('Accident Severity')

plt.ylabel('Frequency [%]')

plt.show()

#Missing values for each column

null_count = df.isnull().sum()

null_count[null_count>0]#.plot('bar', figsize=(30,10))
(df.Age_of_Vehicle

 .value_counts()

 .plot(title = "Age of Vehicle", 

       logx = True, 

       figsize=(14,5)))



print('Min:',    df.Age_of_Vehicle.min(), '\n'

      'Max:',    df.Age_of_Vehicle.max(), '\n'

      'Median:', df.Age_of_Vehicle.median())
(df['Engine_Capacity_.CC.']

 .plot('hist',

       bins = 1000,

       title = "Engine Capacity", 

       figsize=(14,5),

       logx = True

      ))



print('Min:',    df['Engine_Capacity_.CC.'].min(), '\n'

      'Max:',    df['Engine_Capacity_.CC.'].max(), '\n'

      'Median:', df['Engine_Capacity_.CC.'].median())
df2 = df[['Accident_Index', '1st_Road_Class','Day_of_Week', 'Junction_Detail','Light_Conditions', 'Number_of_Casualties',

          'Number_of_Vehicles', 'Road_Surface_Conditions', 'Road_Type', 'Special_Conditions_at_Site', 'Speed_limit',

          'Time', 'Urban_or_Rural_Area', 'Weather_Conditions', 'Age_Band_of_Driver', 'Age_of_Vehicle',

          'Hit_Object_in_Carriageway', 'Hit_Object_off_Carriageway', 'make', 'Engine_Capacity_.CC.', 'Sex_of_Driver',

          'Skidding_and_Overturning', 'Vehicle_Manoeuvre', 'Vehicle_Type', 'Accident_Severity'

         ]]
plt.figure(figsize=(9,5))

sns.heatmap(df2.corr(),linewidths=.5,cmap="YlGnBu")

plt.show()
plt.figure(figsize=(14,5))

sns.distplot(df2.Number_of_Vehicles).set_xlim(0,20)

print('Min:',    df2.Number_of_Vehicles.min(), '\n'

      'Max:',    df2.Number_of_Vehicles.max(), '\n'

      'Median:', df2.Number_of_Vehicles.median())
plt.figure(figsize=(14,5))

sns.distplot(df2.Number_of_Casualties).set_xlim(0,20)

print('Min:',    df2.Number_of_Casualties.min(), '\n'

      'Max:',    df2.Number_of_Casualties.max(), '\n'

      'Median:', df2.Number_of_Casualties.median())
time_x = pd.to_datetime(df2['Time'], format='%H:%M').dt.hour

plt.figure(figsize=(14,5))

time_x.value_counts().sort_index().plot('area')
df2['Accident_Severity'] = df2['Accident_Severity'].replace(['Serious', 'Fatal'], 'Serious or Fatal')

df2 = pd.get_dummies(df2, columns=['Accident_Severity'])

df2 = df2.drop('Accident_Severity_Serious or Fatal', axis=1)

df2.Accident_Severity_Slight.value_counts(normalize=True)
plt.figure(figsize=(14,5))

acc_slight = df2.Accident_Severity_Slight == 1

acc_severe = df2.Accident_Severity_Slight == 0



sns.kdeplot(df2.Number_of_Casualties[acc_slight],shade=True,color='Blue', label='Slight').set_xlim(0,20)

sns.kdeplot(df2.Number_of_Casualties[acc_severe],shade=True,color='Red', label='Severe').set_xlim(0,20)



plt.title('Number of Casualties dist by accident severity')

plt.show()



#print("we can see distribution between failed (under 2000), and successful (bigger the 2000)")

plt.figure(figsize=(14,5))



sns.kdeplot(df2.Number_of_Vehicles[acc_slight],shade=True,color='Blue', label='Slight').set_xlim(0,20)

sns.kdeplot(df2.Number_of_Vehicles[acc_severe],shade=True,color='Red', label='Severe').set_xlim(0,20)



plt.title('Number of vehicles dist by accident severity')

plt.show()



#print("we can see distribution between failed (under 2000), and successful (bigger the 2000)")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,10))

plt.subplots_adjust(hspace=1.4)



(df2.groupby(['Age_Band_of_Driver'])

 .mean()

 ['Accident_Severity_Slight']

 .sort_index()

 .plot

 .bar(title = "Mean Age Band of Driver vs. Accident Severity",

      ax = axes[0,0]))



(df2.groupby(['Speed_limit'])

 .mean()

 ['Accident_Severity_Slight']

 .sort_index()

 .plot

 .bar(title = "Mean Speed limit vs. Accident Severity",

      ax = axes[0,1]))



(df2.groupby(['Urban_or_Rural_Area'])

 .mean()

 ['Accident_Severity_Slight']

 .sort_index()

 .plot

 .bar(title = "Mean Urban or Rural Area vs. Accident Severity",

      ax = axes[1,0]))



(df2.groupby(['Vehicle_Manoeuvre'])

 .mean()

 ['Accident_Severity_Slight']

 .sort_values()

 .plot

 .bar(title = "Mean Vehicle Manoeuvre vs. Accident Severity",

      ax = axes[1,1]))



plt.show()
X = df2.drop(['Accident_Index','Accident_Severity_Slight'], axis=1)

y = df2.Accident_Severity_Slight

print(X.shape,

      y.shape)
def get_Speed_limit(df):

    return df[['Speed_limit']]



FullTransformerOnSpeedLimit = Pipeline([("Select_Speed_Limit", FunctionTransformer(func=get_Speed_limit, validate=False)),

                                        ("Fill_Null",          SimpleImputer(missing_values=np.nan, strategy='most_frequent')),

                                        ("One_Hot_Encoder",    OneHotEncoder(sparse = False, handle_unknown='ignore'))

                                       ])



#FullTransformerOnSpeedLimit.fit_transform(X[:5000], y[:5000])
def get_Time(df):

    return pd.to_datetime(df['Time'], format='%H:%M').dt.time



def find_time_group(time_object):

    if time_object<pd.datetime.time(pd.datetime(2000,1,1,5,0)):

        return 'Night'

    elif time_object<pd.datetime.time(pd.datetime(2000,1,1,7,0)):

        return 'Early Morning'

    elif time_object<pd.datetime.time(pd.datetime(2000,1,1,10,0)):

        return 'Morning'

    elif time_object<pd.datetime.time(pd.datetime(2000,1,1,15,0)):

        return 'Midday'

    elif time_object<pd.datetime.time(pd.datetime(2000,1,1,18,0)):

        return 'Afternoon'

    elif time_object<pd.datetime.time(pd.datetime(2000,1,1,20,0)):

        return 'Evening'

    elif time_object<=pd.datetime.time(pd.datetime(2000,1,1,23,59)):

        return 'Late Evening'

    return np.nan



FullTransformerOnTime = Pipeline([("Select_Time",     FunctionTransformer(func=get_Time, validate=False)),

                                  ("Group_Time",      FunctionTransformer(func=lambda x: x.apply(find_time_group).to_frame(), validate=False)),

                                  ("Fill_Null",       SimpleImputer(missing_values=np.nan, strategy='most_frequent')),

                                  ("One_Hot_Encoder", OneHotEncoder(sparse = False, handle_unknown='ignore'))

                                 ])



#FullTransformerOnTime.fit_transform(X[:5000], y[:5000])
def get_Age_of_Vehicle(df):

    return df[['Age_of_Vehicle']]



FullTransformerOnAgeofVehicle = Pipeline([("Select_Age_of_Vehicle", FunctionTransformer(func=get_Age_of_Vehicle, validate=False)),

                                          ("Fill_Null",             SimpleImputer(missing_values=np.nan, strategy='median'))

                                         ])



#FullTransformerOnAgeofVehicle.fit_transform(X[:5000], y[:5000])
def get_make(df):

    list_of_small_makers = list(df['make'].value_counts()[df['make'].value_counts() < 2000].index)

    return df['make'].replace(list_of_small_makers, 'Other').to_frame()



FullTransformerOnMake = Pipeline([("Select_Make",      FunctionTransformer(func=get_make, validate=False)),

                                   ("Fill_Null",       SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='Other')),

                                   ("One_Hot_Encoder", OneHotEncoder(sparse = False, handle_unknown='ignore'))])



#FullTransformerOnMake.fit_transform(X[:5000], y[:5000])
def get_Engine_Capacity(df):

    return df[['Engine_Capacity_.CC.']]



FullTransformerOnEngineCapacity = Pipeline([("Select_Engine_Capacity",       FunctionTransformer(func=get_Engine_Capacity, validate=False)),

                                            ("Fill_Null",                    SimpleImputer(missing_values=np.nan, strategy='most_frequent')),

                                            ("Car_Types_by_Engine_Capacity", KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='quantile')),

                                            ("One_Hot_Encoder",              OneHotEncoder(sparse = False, handle_unknown='ignore'))

                                           ])



#FullTransformerOnEngineCapacity.fit_transform(X[:5000], y[:5000])

#FullTransformerOnEngineCapacity.named_steps["Car_Types_by_Engine_Capacity"].bin_edges_[0]
def get_columns_to_one_hot(df):

    return df[['1st_Road_Class', 'Day_of_Week', 'Junction_Detail', 'Light_Conditions', 'Number_of_Casualties', 

               'Number_of_Vehicles', 'Road_Surface_Conditions', 'Road_Type', 'Special_Conditions_at_Site', 

               'Urban_or_Rural_Area', 'Weather_Conditions', 'Age_Band_of_Driver', 'Hit_Object_in_Carriageway',

               'Hit_Object_off_Carriageway', 'Sex_of_Driver', 'Skidding_and_Overturning',

               'Vehicle_Manoeuvre', 'Vehicle_Type'

              ]]



DataToOneHotTransformerOnColumns = Pipeline([("Select_Columns",  FunctionTransformer(func=get_columns_to_one_hot, validate=False)),

                                             ("One_Hot_Encoder", OneHotEncoder(sparse = False, handle_unknown='ignore'))])



#DataToOneHotTransformerOnColumns.fit_transform(X[:5000], y[:5000])
FeatureUnionTransformer = FeatureUnion([

                                        ("FTAgeofVehicle",   FullTransformerOnAgeofVehicle),

                                        ("FTEngineCapacity", FullTransformerOnEngineCapacity),

                                        ("FTMake",           FullTransformerOnMake),

                                        ("FTSpeedLimit",     FullTransformerOnSpeedLimit),

                                        ("FTTime",           FullTransformerOnTime),

                                        ("OHEColumns",       DataToOneHotTransformerOnColumns)])



#FeatureUnionTransformer.fit_transform(X[:5000], y[:5000])
Full_Transformer = Pipeline([

                           ("Feature_Engineering", FeatureUnionTransformer),

                           ("Min_Max_Transformer", MaxAbsScaler())

                           ])



#Full_Transformer.fit(X[:5000], y[:5000])
X_train, X_test, y_train, y_test = split(X, y)
%%time



clf = LogisticRegression(class_weight = "balanced")



Full_Transformer.fit(X_train)

X_train_transformed = Full_Transformer.transform(X_train)

clf.fit(X_train_transformed, y_train)



X_test_transformed = Full_Transformer.transform(X_test)



y_pred = clf.predict(X_test_transformed)



print('Classification Report:',classification_report(y_test, y_pred))



print('Score:',roc_auc_score(y_test.values, clf.predict_proba(X_test_transformed)[:, 1]))
%%time



clf = RandomForestClassifier(n_estimators=100, n_jobs=3)



Full_Transformer.fit(X_train)

X_train_transformed = Full_Transformer.transform(X_train)

clf.fit(X_train_transformed, y_train)



X_test_transformed = Full_Transformer.transform(X_test)



y_pred = clf.predict(X_test_transformed)



print('Classification Report:',classification_report(y_test, y_pred))



print('Score:',roc_auc_score(y_test.values, clf.predict_proba(X_test_transformed)[:, 1]))
LogisticRegression_Full_Estimator = Pipeline([

                                              ("Feature_Engineering", FeatureUnionTransformer),

                                              ("Min_Max_Transformer", MaxAbsScaler()),

                                              ("Clf",                 LogisticRegression(class_weight = "balanced"))

                                             ])



#LogisticRegression_Full_Estimator.fit(X[:5000], y[:5000])
%%time



LogisticRegression_Full_Estimator.fit(X_train, y_train)

LogisticRegression_Full_Estimator.predict(X_train)

LogisticRegression_Full_Estimator.predict(X_test)



print('Classification Report:' '\n',

      classification_report(y_test, LogisticRegression_Full_Estimator.predict(X_test)))

print('Score:',roc_auc_score(y_test.values, LogisticRegression_Full_Estimator.predict_proba(X_test)[:, 1]))
RandomForest_Full_Estimator = Pipeline([

                                        ("Feature_Engineering", FeatureUnionTransformer),

                                        ("Min_Max_Transformer", MaxAbsScaler()),

                                        ("Clf",                 RandomForestClassifier(n_estimators=100, n_jobs=3))

                                       ])



#RandomForest_Full_Estimator.fit(X[:5000], y[:5000])
%%time



RandomForest_Full_Estimator.fit(X_train, y_train)

RandomForest_Full_Estimator.predict(X_train)

RandomForest_Full_Estimator.predict(X_test)



print('Classification Report:' '\n',

      classification_report(y_test, RandomForest_Full_Estimator.predict(X_test)))

print('Score:',roc_auc_score(y_test.values, RandomForest_Full_Estimator.predict_proba(X_test)[:, 1]))
#%%time

#scoreTest_RF = []

#scoreTrain_RF = []

#for number in range(1,10):

#    clf = RandomForestClassifier(max_depth = number,n_estimators = 100, n_jobs=3, class_weight = "balanced")

#    Full_Transformer.fit(X_train)

#    X_train_transformed = Full_Transformer.transform(X_train)

#    clf.fit(X_train_transformed, y_train)

#    X_test_transformed = Full_Transformer.transform(X_test)

#    y_score_train = clf.predict_proba(X_train_transformed)[:,1]

#    y_score_test = clf.predict_proba(X_test_transformed)[:,1]

#

#    scoreTrain_RF.append(round(roc_auc_score(y_train, y_score_train) , 3))

#    scoreTest_RF.append(round(roc_auc_score(y_test, y_score_test) , 3))

    
#pd.DataFrame({'test roc score':scoreTest_RF,'train roc score':scoreTrain_RF}).plot(grid = True)

#plt.xlabel('Max depth')

#plt.ylabel('Score')

#plt.title("RandomForestClassifier")

#plt.show()
#%%time

#cls_RF = RandomForestClassifier(max_depth = np.array(scoreTest_RF).argmax(),n_estimators = 100, n_jobs=3, class_weight = "balanced")

#Full_Transformer.fit(X_train)

#X_train_transformed = Full_Transformer.transform(X_train)

#clf.fit(X_train_transformed, y_train)

#X_test_transformed = Full_Transformer.transform(X_test)

#

#print("RF roc_train:",round(roc_auc_score(y_train, cls_RF.predict_proba(X_train_transformed)[:,1]) , 3))

#print("RF roc_test:",round(roc_auc_score(y_test, cls_RF.predict_proba(X_test_transformed)[:,1]) , 3))

#

#print('RF_cross_score:', cross_val_score(cls_RF, X_train_transformed, y_train, cv=5, scoring='roc_auc').mean())
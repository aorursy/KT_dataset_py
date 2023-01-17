import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import random
import math
import time
# from sklearn.linear_model import LinearRegression, BayesianRidge
# from sklearn.model_selection import RandomizedSearchCV, train_test_split
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator 
plt.style.use('fivethirtyeight')
%matplotlib inline 
PatientInfo = pd.read_csv(r'../input/PatientInfo.csv')
PatientInfo_Copy = PatientInfo
PatientInfo.isna().sum()
PatientInfo.columns
PatientInfo["disease"] = PatientInfo["disease"].fillna("False")
PatientInfo["city"] = PatientInfo["city"].fillna("Unknown")
PatientInfo["country"] = PatientInfo["country"].fillna("Unknown")
PatientInfo["contact_number"] = PatientInfo["contact_number"].fillna("0")
PatientInfo["infected_by"] = PatientInfo["infected_by"].fillna("0")
PatientInfo["deceased_date"] = PatientInfo["deceased_date"].fillna("None")
PatientInfo["released_date"] = PatientInfo["released_date"].fillna("None")
PatientInfo["symptom_onset_date"] = PatientInfo["symptom_onset_date"].fillna("None")
PatientInfo["sex"] = PatientInfo["sex"].fillna("Unknown")
PatientInfo["infection_case"] = PatientInfo["infection_case"].fillna("Others")
PatientInfo.isna().sum()
PatientInfo = PatientInfo.drop(['global_num','birth_year','infection_order',
                                'symptom_onset_date','released_date','deceased_date'], axis=1)
PatientInfo["age"] = PatientInfo["age"].str.replace("s","").astype(float)
## Updating null values in age column by using mean number
index_NaN_age = list(PatientInfo["age"][PatientInfo["age"].isnull()].index)
index_NaN_age

for i in index_NaN_age :
    age_med = PatientInfo["age"].mean()
    age_pred = PatientInfo["age"].mean()
    if not np.isnan(age_pred) :
        PatientInfo["age"].iloc[i] = age_pred
    else :
        PatientInfo["age"].iloc[i] = age_med
# #Converting certain features to categorical form
categorical_features = ['sex', 'country','province','city', 'disease',
                        'infection_case', 'state']
PatientInfo[categorical_features] = PatientInfo[categorical_features].astype('category')
PatientInfo.info()
# #Converting certain features to Numerical form
Numerical_features = ['contact_number','infected_by']
PatientInfo[Numerical_features] = PatientInfo[Numerical_features].astype('int64')
PatientInfo.info()
## Converting Date column to Date form
from datetime import datetime, timedelta
from pandas import DataFrame
PatientInfo['confirmed_date'] = PatientInfo['confirmed_date'].astype('datetime64[ns]') 
PatientInfo.info()
import seaborn as sns

# def categorical_eda(df):
#     """Given dataframe, generate EDA of categorical data"""
#     print("To check: Unique count of non-numeric data")
#     print(df.select_dtypes(include=['category']).nunique())
#     # Plot count distribution of categorical data
    
#     for col in df.select_dtypes(include='category').columns:
#         if df[col].nunique() < 20:
#             fig = sns.catplot(x=col,hue='state', kind="count", data=df)
#             fig.set_xticklabels(rotation=90)
#             plt.show()
        
        
# categorical_eda(PatientInfo)
############# Current Status ################
g  = sns.factorplot(y="age",x="state",data=PatientInfo,kind="bar", size = 6 , palette = "muted")
#############  Patients State by age group and province  ###########
g  = sns.factorplot(y="age",x="country",data=PatientInfo, hue = "state", kind="bar", size = 6 ,
                    palette = "muted",height=8.27, aspect=11.7/5)
#############  Patients State by age group and province  ###########
fig,axes = plt.subplots(1,1,figsize=(20,5))
g = sns.barplot(y="age",x="province",data=PatientInfo,hue = "state")
########## Top 20 infected province #############
fig,axes = plt.subplots(1,1,figsize=(20,5))
sns.countplot(PatientInfo.province, order = PatientInfo.province.value_counts()[:20].index)
plt.xticks(rotation = 50)
plt.show()
########## Top 20 infected cities #############
fig,axes = plt.subplots(1,1,figsize=(20,5))
sns.countplot(PatientInfo.city, order = PatientInfo.city.value_counts()[:20].index)
plt.xticks(rotation = 50)
plt.show()
#### Most infection casued by Patients ############
Infected_by = PatientInfo[PatientInfo["infected_by"] != 0 ]

fig,axes = plt.subplots(1,1,figsize=(20,5))
sns.countplot(Infected_by.infected_by, order = Infected_by.infected_by.value_counts()[:30].index)
plt.xticks(rotation = 50)
plt.show()
######## infection_case Analysis #####
fig,axes = plt.subplots(1,1,figsize=(20,5))
sns.countplot(PatientInfo.infection_case, order = PatientInfo.infection_case.value_counts()[:30].index)
plt.xticks(rotation = 90)
plt.show()
Confirmed_Cases = PatientInfo.groupby('confirmed_date').count()
plt.figure(figsize=(16, 9))
Confirmed_Case_History = Confirmed_Cases['infection_case'].plot(title = "Confirmed Case History")
Death_History = PatientInfo_Copy[PatientInfo_Copy['state']=='deceased'].groupby('deceased_date').count()
plt.figure(figsize=(16, 9))
Death_Case_History = Death_History['infection_case'].plot(title = "Death Case History")
Released_History = PatientInfo_Copy[PatientInfo_Copy['state']=='released'].groupby('released_date').count()
plt.figure(figsize=(16, 9))
Released_Case_History = Released_History['infection_case'].plot(title = "Released Case History")
PatientRoute = pd.read_csv('../input/PatientRoute.csv')
PatientRoute_Copy = PatientRoute
PatientRoute
PatientInfo.isna().sum()
#### Most infection casued by Patients ############
# Infected_by = PatientInfo[PatientInfo["infected_by"] != 0 ]

fig,axes = plt.subplots(1,1,figsize=(20,5))
sns.countplot(PatientRoute.patient_id, order = PatientRoute.patient_id.value_counts()[:70].index)
plt.xticks(rotation = 50)
plt.show()
#### Most infection casued by Patients ############
# Infected_by = PatientInfo[PatientInfo["infected_by"] != 0 ]

fig,axes = plt.subplots(1,1,figsize=(20,5))
sns.countplot(PatientRoute.province, order = PatientRoute.province.value_counts()[:70].index)
plt.xticks(rotation = 50)
plt.show()
#### Most infection_case ############
# Infected_by = PatientInfo[PatientInfo["infected_by"] != 0 ]

fig,axes = plt.subplots(1,1,figsize=(20,5))
sns.countplot(PatientRoute.type, order = PatientRoute.type.value_counts()[:70].index)
plt.xticks(rotation = 50)
plt.show()
#### Most infection casued in city ############
# Infected_by = PatientInfo[PatientInfo["infected_by"] != 0 ]

fig,axes = plt.subplots(1,1,figsize=(20,5))
sns.countplot(PatientRoute.city, order = PatientRoute.city.value_counts()[:70].index)
plt.xticks(rotation = 50)
plt.show()
PatientRoute_History = PatientRoute.groupby('date').count()
plt.figure(figsize=(16, 9))
PatientRoute_History = PatientRoute_History['patient_id'].plot(title = "Patient Gathering History")
PatientRoute_details = PatientRoute.groupby('patient_id').count()
PatientRoute_details.reset_index(inplace = True) 
PatientRoute_details

PatientRoute_details = PatientRoute_details.drop([ 'global_num', 'date', 'city', 'type',
       'latitude', 'longitude'], axis=1)
PatientRoute_details.rename(columns = {'province':'Places_Visited'}, inplace = True)
result = pd.merge(PatientInfo,
                 PatientRoute_details[['patient_id', 'Places_Visited']],
                 on='patient_id')
result
TimeAge = pd.read_csv('../input/TimeAge.csv')
TimeAge_Copy = TimeAge
TimeAge['date'] = TimeAge['date'].astype('datetime64[ns]') 
TimeAge.set_index('date')
TimeAge_0 = TimeAge[TimeAge['age'] == '0s'].set_index('date')
TimeAge_10 = TimeAge[TimeAge['age'] == '10s'].set_index('date')
TimeAge_20 = TimeAge[TimeAge['age'] == '20s'].set_index('date')
TimeAge_30 = TimeAge[TimeAge['age'] == '30s'].set_index('date')
TimeAge_40 = TimeAge[TimeAge['age'] == '40s'].set_index('date')
TimeAge_50 = TimeAge[TimeAge['age'] == '50s'].set_index('date')
TimeAge_60 = TimeAge[TimeAge['age'] == '60s'].set_index('date')
TimeAge_70 = TimeAge[TimeAge['age'] == '70s'].set_index('date')
TimeAge_80 = TimeAge[TimeAge['age'] == '80s'].set_index('date')
TimeAge_90 = TimeAge[TimeAge['age'] == '90s'].set_index('date')
plt.figure(figsize=(16, 9))
plt.plot(TimeAge_0['confirmed'])
plt.plot(TimeAge_10['confirmed'])
plt.plot(TimeAge_20['confirmed'])
plt.plot(TimeAge_30['confirmed'])
plt.plot(TimeAge_40['confirmed'])
plt.plot(TimeAge_50['confirmed'])
plt.plot(TimeAge_60['confirmed'])
plt.plot(TimeAge_70['confirmed'])
plt.plot(TimeAge_80['confirmed'])
plt.title('Confirmed Coronavirus Cases', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('Confirmed Cases', size=30)
plt.legend(['0s', '10s', '20s', '30s', '40s','50s','60s','70s','80s'], prop={'size': 20})
plt.figure(figsize=(16, 9))
plt.plot(TimeAge_0['deceased'])
plt.plot(TimeAge_10['deceased'])
plt.plot(TimeAge_20['deceased'])
plt.plot(TimeAge_30['deceased'])
plt.plot(TimeAge_40['deceased'])
plt.plot(TimeAge_50['deceased'])
plt.plot(TimeAge_60['deceased'])
plt.plot(TimeAge_70['deceased'])
plt.plot(TimeAge_80['deceased'])
plt.title('Deaths', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('Confirmed Deaths', size=30)
plt.legend(['0s', '10s', '20s', '30s', '40s','50s','60s','70s','80s'], prop={'size': 20})
TimeAge_Copy
TimeAge_Stat = TimeAge_Copy.groupby('age')['confirmed','deceased'].sum()
TimeAge_Stat.reset_index(inplace = True) 
TimeAge_Stat["age"] = TimeAge_Stat["age"].str.replace("s","").astype(float)
TimeAge_Stat
 ####### Confirmed cases w.r.t. Age ######
g  = sns.factorplot(y="confirmed",x="age",data=TimeAge_Stat, kind="bar", size = 6 ,
                    palette = "muted",height=8.27, aspect=11.7/5)
g  = sns.factorplot(y="deceased",x="age",data=TimeAge_Stat, kind="bar", size = 6 ,
                    palette = "muted",height=8.27, aspect=11.7/5)
 ####### Deceased cases w.r.t. Age ######
g  = sns.factorplot(y="deceased",x="age",data=TimeAge_Stat, kind="bar", size = 6 ,
                    palette = "muted",height=8.27, aspect=11.7/5)
### Confirmed cases and Death Comparision ##

indices = TimeAge_Stat.index
width = np.min(np.diff(indices))/3

fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(111)
ax.bar(indices-width,TimeAge_Stat['confirmed'],width,color='b',label='-Ymin')
ax.bar(indices,TimeAge_Stat['deceased'],width,color='r',label='Ymax')
ax.set_xlabel('Confimed and deaths')
plt.show()
Time = pd.read_csv('../input/Time.csv')
Time_Copy = Time
Time['date'] = Time['date'].astype('datetime64[ns]') 
Time.set_index('date')
plt.figure(figsize=(16, 9))
plt.plot(Time['test'])
plt.plot(Time['negative'])
plt.plot(Time['confirmed'])
plt.plot(Time['released'])
plt.plot(Time['deceased'])
plt.title('Test Analysis', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('Tests', size=30)
plt.legend(['test', 'negative', 'confirmed', 'released', 'deceased'], prop={'size': 20})
Time_Stat = Time.sum()
Time_Stat = Time_Stat.to_frame()
Time_Stat.reset_index(inplace = True) 
Time_Stat.rename(columns = {'index':'Analysis',0: 'Count'}, inplace = True)
g  = sns.barplot(x='Analysis', y='Count',data=Time_Stat)

#Time_Stat = Time_Copy['negative', 'confirmed'].sum()
TimeGender = pd.read_csv('../input/TimeGender.csv')
TimeGender

TimeGender_Copy = TimeGender
TimeGender['date'] = TimeGender['date'].astype('datetime64[ns]') 
TimeGender.set_index('date')
plt.figure(figsize=(16, 9))
MaleCases = TimeGender[TimeGender['sex'] == 'male']
FemaleCases = TimeGender[TimeGender['sex'] == 'female']
plt.plot(MaleCases['confirmed'])
plt.plot(FemaleCases['confirmed'])
plt.title('Test Analysis', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('Tests', size=30)
plt.legend(['Male', 'Female'], prop={'size': 20})
plt.figure(figsize=(16, 9))
MaleCases = TimeGender[TimeGender['sex'] == 'male']
FemaleCases = TimeGender[TimeGender['sex'] == 'female']
plt.plot(MaleCases['deceased'])
plt.plot(FemaleCases['deceased'])
plt.title('deceased Analysis', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('Tests', size=30)
plt.legend(['Male', 'Female'], prop={'size': 20})
TimeProvince = pd.read_csv('../input/TimeProvince.csv')
TimeProvince

TimeProvince_Copy = TimeProvince
TimeProvince['date'] = TimeProvince['date'].astype('datetime64[ns]') 
TimeProvince.set_index('date')
TimeProvince_Stat = TimeProvince_Copy.groupby('date')['confirmed','deceased','released'].sum()
TimeProvince_Stat
plt.figure(figsize=(16, 9))
plt.plot(TimeProvince_Stat['confirmed'])
plt.plot(TimeProvince_Stat['released'])
plt.plot(TimeProvince_Stat['deceased'])
plt.title('Test Analysis', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('Tests', size=30)
plt.legend(['Confirmed', 'released','deceased'], prop={'size': 20})
Province_Stat = TimeProvince_Copy.groupby('province')['confirmed','deceased','released'].max()
Province_Stat.reset_index(inplace = True) 
Province_Stat
g = sns.factorplot(y="confirmed",x="province",data=Province_Stat, kind="bar", size = 6 ,
                    palette = "muted",height=8.27, aspect=11.7/3.5)
plt.xticks(rotation = 90)
# TimeProvince_Copy
g = sns.factorplot(y="released",x="province",data=Province_Stat, kind="bar", size = 6 ,
                    palette = "muted",height=8.27, aspect=11.7/3.5)
plt.xticks(rotation = 90)
# TimeProvince_Copy
g = sns.factorplot(y="deceased",x="province",data=Province_Stat, kind="bar", size = 6 ,
                    palette = "muted",height=8.27, aspect=11.7/3.5)
plt.xticks(rotation = 90)
Weather = pd.read_csv('../input/Weather.csv')
Weather['date'] = Weather['date'].astype('datetime64[ns]') 
Weather_2020 = Weather[(Weather['date'] > '12/31/2019')]
Weather_2020.rename(columns = {'date':'confirmed_date'}, inplace = True)
Weather_2020
# apply "Vlookup" in padas
Covid_2020_Details = PatientInfo.merge(Weather_2020, how='left', on=['province', 'confirmed_date'])
Covid_2020_Details
## Droping Unwanted columns 
Covid_2020_Details = Covid_2020_Details.drop(['code','infected_by','confirmed_date','patient_id'], axis=1)
# Covid_2020_Details.fillna(Covid_2020_Details.mean())
Covid_2020_Details['avg_temp'].fillna((Covid_2020_Details['avg_temp'].mean()), inplace=True)
Covid_2020_Details['min_temp'].fillna((Covid_2020_Details['min_temp'].mean()), inplace=True)
Covid_2020_Details['max_temp'].fillna((Covid_2020_Details['max_temp'].mean()), inplace=True)
Covid_2020_Details['precipitation'].fillna((Covid_2020_Details['precipitation'].mean()), inplace=True)
Covid_2020_Details['max_wind_speed'].fillna((Covid_2020_Details['max_wind_speed'].mean()), inplace=True)
Covid_2020_Details['most_wind_direction'].fillna((Covid_2020_Details['most_wind_direction'].mean()), inplace=True)
Covid_2020_Details['avg_relative_humidity'].fillna((Covid_2020_Details['avg_relative_humidity'].mean()), inplace=True)
# Covid_2020_Details.to_csv("Covid_2020_Details.csv")
Covid_2020_Details = Covid_2020_Details[(Covid_2020_Details['sex'] != 'Unknown')]
Covid_2020_Details
# #Converting certain features to categorical form
categorical_features_Covid_2020 = ['province']
Covid_2020_Details[categorical_features_Covid_2020] = Covid_2020_Details[categorical_features_Covid_2020].astype('category')
Covid_2020_Details.info()
plt.figure(figsize=(20,20))
g = sns.heatmap(Covid_2020_Details.corr(),cmap="BrBG",annot=True)
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
# Here applying label encoder to categorical attribute by using column key name.
Covid_2020_Details['sex']=labelencoder.fit_transform(Covid_2020_Details['sex'])
Covid_2020_Details['country']=labelencoder.fit_transform(Covid_2020_Details['country'])
Covid_2020_Details['province']=labelencoder.fit_transform(Covid_2020_Details['province'])
Covid_2020_Details['city']=labelencoder.fit_transform(Covid_2020_Details['city'])
Covid_2020_Details['infection_case']=labelencoder.fit_transform(Covid_2020_Details['infection_case'])
Covid_2020_Details['state']=labelencoder.fit_transform(Covid_2020_Details['state'])
Covid_2020_Details['disease'] = Covid_2020_Details['disease'].map({'TRUE':1, 'FALSE':0})
Covid_2020_Details['disease']=labelencoder.fit_transform(Covid_2020_Details['disease'])
Covid_2020_Details.info()
y=Covid_2020_Details['state'].values
x=Covid_2020_Details.drop(['state'],axis=1).values
# dataset split.
train_size=0.80
test_size=0.20
seed=5

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,train_size=train_size,test_size=test_size,random_state=seed)
print(X_train.shape)
print(y_train.shape)
n_neighbors=5
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# keeping all models in one list
models=[]
models.append(('LogisticRegression',LogisticRegression()))
models.append(('knn',KNeighborsClassifier(n_neighbors=n_neighbors)))
models.append(('SVC',SVC()))
models.append(("decision_tree",DecisionTreeClassifier()))
models.append(('Naive Bayes',GaussianNB()))

# Evaluating Each model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

names=[]
predictions=[]
error='accuracy'
for name,model in models:
    fold=KFold(n_splits=10,random_state=0)
    result=cross_val_score(model,X_train,y_train,cv=fold,scoring=error)
    predictions.append(result)
    names.append(name)
    msg="%s : %f (%f)"%(name,result.mean(),result.std())
    print(msg)\
    
# # Visualizing the Model accuracy
fig=plt.figure()
fig.suptitle("Comparing Algorithms")
plt.boxplot(predictions)
plt.show()
# Spot Checking and Comparing Algorithms With StandardScaler Scaler
from sklearn.pipeline import Pipeline
from sklearn. preprocessing import StandardScaler
pipelines=[]
pipelines.append(('scaled Logisitic Regression',Pipeline([('scaler',StandardScaler()),('LogisticRegression',LogisticRegression())])))
pipelines.append(('scaled KNN',Pipeline([('scaler',StandardScaler()),('KNN',KNeighborsClassifier(n_neighbors=n_neighbors))])))
pipelines.append(('scaled SVC',Pipeline([('scaler',StandardScaler()),('SVC',SVC())])))
pipelines.append(('scaled DecisionTree',Pipeline([('scaler',StandardScaler()),('decision',DecisionTreeClassifier())])))
pipelines.append(('scaled naive bayes',Pipeline([('scaler',StandardScaler()),('scaled Naive Bayes',GaussianNB())])))

# # Evaluating Each model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
names=[]
predictions=[]
for name,model in models:
    fold=KFold(n_splits=10,random_state=0)
    result=cross_val_score(model,X_train,y_train,cv=fold,scoring=error)
    predictions.append(result)
    names.append(name)
    msg="%s : %f (%f)"%(name,result.mean(),result.std())
    print(msg)
    

# # Visualizing the Model accuracy
fig=plt.figure()
fig.suptitle("Comparing Algorithms")
plt.boxplot(predictions)
plt.show()
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                          random_state=0, shuffle=False)
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
AdaBoostClassifier(n_estimators=100, random_state=0)
clf.feature_importances_
clf.score(X_test,y_test)
import xgboost
classifier=xgboost.XGBClassifier()

classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.4, learning_rate=0.1,
       max_delta_step=0, max_depth=6, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)

from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X_train, y_train,cv=10)

score
score.mean()
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout
# Initialising the ANN
classifier = Sequential()
X_train.shape
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 12, init = 'he_uniform',activation='relu',input_dim = 15))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 12, init = 'he_uniform',activation='relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'glorot_uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, nb_epoch = 100)
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)
y_pred

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test, y_pred))
# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

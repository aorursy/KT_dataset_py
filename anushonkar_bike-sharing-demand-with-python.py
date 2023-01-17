#Data visualisation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns



#warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



#For classifiaction.

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier





#For Model selection

from sklearn.model_selection import train_test_split,cross_validate



#regression

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor



#forevaluation metrics

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression
# Print multiple statements in same line

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
#read test and train data

train=pd.read_csv(r'../input/train.csv',parse_dates = [0])

test=pd.read_csv(r'../input/test.csv',parse_dates = [0])

df=train.copy()

test_df=test.copy()
#to view data

df.head()
#to view unique columns and check null values

df.columns.unique()

df.isnull().sum()
#visualization of data



#holiday

df.holiday.value_counts()#total holiday count

sns.factorplot(x='holiday',data=df,kind='count',size=4,aspect=1)

plt.show();


#for Season

df.season.value_counts()

sns.factorplot(x='season',data=df,kind='count',size=4,aspect=1)

plt.show();
#for weather

df.weather.value_counts()

sns.factorplot(x='weather',data=df,kind='count',size=4,aspect=1)

plt.xlabel("Weather details")

plt.show();
#boxplot

sns.boxplot(data=df[['temp','atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']], width=0.8,linewidth=1.5)

fig=plt.gcf()

fig.set_size_inches(10,10)
#Visulaization using histograms for all the continuous variables.



fig,axes=plt.subplots(2,2)

axes[0,0].hist(x="temp",data=df,edgecolor="black",linewidth=1,color='blue')

axes[0,0].set_title("Variation of temp")

axes[0,1].hist(x="atemp",data=df,edgecolor="black",linewidth=1,color='blue')

axes[0,1].set_title("Variation of atemp")

axes[1,0].hist(x="windspeed",data=df,edgecolor="black",linewidth=1,color='blue')

axes[1,0].set_title("Variation of Windspeed")

axes[1,1].hist(x="humidity",data=df,edgecolor="black",linewidth=1,color='blue')

axes[1,1].set_title("Variation of Humidity")

fig.set_size_inches(10,10)

plt.show();
#Dummy coding Season

season=pd.get_dummies(df['season'],prefix='season')

df=pd.concat([df,season],axis=1)

df.head()

#for test data

season=pd.get_dummies(test_df['season'],prefix='season')

test_df=pd.concat([test_df,season],axis=1)

test_df.head()
#Dummy coding weather

weather=pd.get_dummies(df['weather'],prefix='weather')

df=pd.concat([df,weather],axis=1)

df.head()

#for test data

weather=pd.get_dummies(test_df['weather'],prefix='weather')

test_df=pd.concat([test_df,weather],axis=1)

test_df.head()
#Drop Weather and season

df.drop(['season','weather'],inplace=True,axis=1)



#test data

test_df.drop(['season','weather'],inplace=True,axis=1)

#feature extraction of date



def feature_extraction(df):

    df['year'] = df.datetime.dt.year

    df['year'] = df['year'].map({2011:0, 2012:1})

    df['month'] = df.datetime.dt.month

    df['dayofweek'] = df.datetime.dt.dayofweek

    df['hour'] = df.datetime.dt.hour

    df['day'] = df.datetime.dt.day

    

feature_extraction(df)

feature_extraction(test_df)
df.head()
#Drop datetime and Casual registered columns

df.drop('datetime',axis=1,inplace=True)

df.drop(['casual','registered'],axis=1,inplace=True)
#Variation of count 

# with hour.

sns.factorplot(x="hour",y="count",data=df,kind='point',size=4,aspect=1.5)

plt.xlabel("Varation of count with hour")

plt.show();
#with month

sns.factorplot(x="month",y="count",data=df,kind='bar',size=4,aspect=1)

plt.xlabel("Varation of count with month")

plt.show();
#with year

sns.factorplot(x="year",y="count",data=df,kind='bar',size=4,aspect=1)

plt.xlabel("Varation of count with year")

plt.show();
#with day

sns.factorplot(x="day",y='count',kind='bar',data=df,size=5,aspect=1)

plt.xlabel("Varation of count with day")

plt.show();
#MODELLING 

x_train,x_test,y_train,y_test=train_test_split(df.drop('count',axis=1),df['count'],test_size=0.25,random_state=42)



models=[RandomForestRegressor(),AdaBoostRegressor(),BaggingRegressor(),SVR(),KNeighborsRegressor()]

model_names=['RandomForestRegressor','AdaBoostRegressor','BaggingRegressor','SVR','KNeighborsRegressor']

rmsle=[]

d={}

for model in range (len(models)):

    clf=models[model]

    clf.fit(x_train,y_train)

    test_pred=clf.predict(x_test)

    rmsle.append(np.sqrt(mean_squared_log_error(test_pred,y_test)))

d={'Modelling Algo':model_names,'RMSLE':rmsle}   

rmsle_frame=pd.DataFrame(d)

rmsle_frame
sns.factorplot(x='Modelling Algo',y='RMSLE',data=rmsle_frame,kind='bar',size=5,aspect=2)
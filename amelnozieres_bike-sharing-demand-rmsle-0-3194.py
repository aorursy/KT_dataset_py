# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



# data visualisation and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

import missingno as msno

#configure

# sets matplotlib to inline and displays graphs below the corressponding cell.

% matplotlib inline  

style.use('fivethirtyeight')

sns.set(style='whitegrid',color_codes=True)



#import the necessary modelling algos.



#classifiaction.

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC,SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB



#regression

from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor



#model selection

from sklearn.model_selection import train_test_split,cross_validate

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



#evaluation metrics

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  # for classification

 
train=pd.read_csv(r'../input/train.csv')

test=pd.read_csv(r'../input/test.csv')

df=train.copy()

test_df=test.copy()

df.head()
df.columns.unique()
df.info()
df.isnull().sum()  # implies no null values and hence no imputation needed ::).
msno.matrix(df)  # just to visualize. no missing value.
# let us consider season.

df.season.value_counts()
#sns.factorplot(x='season',data=df,kind='count',size=5,aspect=1)

sns.factorplot(x='season',data=df,kind='count',size=5,aspect=1.5)
#holiday

df.holiday.value_counts()

sns.factorplot(x='holiday',data=df,kind='count',size=5,aspect=1) # majority of data is for non holiday days.
#holiday

df.workingday.value_counts()

sns.factorplot(x='workingday',data=df,kind='count',size=5,aspect=1) # majority of data is for working days.
#weather

df.weather.value_counts()
sns.factorplot(x='weather',data=df,kind='count',size=5,aspect=1)  

# 1-> spring

# 2-> summer

# 3-> fall

# 4-> winter
df.describe()
# just to visualize.

sns.boxplot(data=df[['temp',

       'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']])

fig=plt.gcf()

fig.set_size_inches(10,10)
# can also be visulaized using histograms for all the continuous variables.

df.temp.unique()

fig,axes=plt.subplots(2,2)

axes[0,0].hist(x="temp",data=df,edgecolor="black",linewidth=2,color='#ff4125')

axes[0,0].set_title("Variation of temp")

axes[0,1].hist(x="atemp",data=df,edgecolor="black",linewidth=2,color='#ff4125')

axes[0,1].set_title("Variation of atemp")

axes[1,0].hist(x="windspeed",data=df,edgecolor="black",linewidth=2,color='#ff4125')

axes[1,0].set_title("Variation of windspeed")

axes[1,1].hist(x="humidity",data=df,edgecolor="black",linewidth=2,color='#ff4125')

axes[1,1].set_title("Variation of humidity")

fig.set_size_inches(10,10)
#corelation matrix.

cor_mat= df[:].corr()

mask = np.array(cor_mat)

mask[np.tril_indices_from(mask)] = False

fig=plt.gcf()

fig.set_size_inches(30,12)

sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)
# # seperating season as per values. this is bcoz this will enhance features.

season=pd.get_dummies(df['season'],prefix='season')

df=pd.concat([df,season],axis=1)

df.head()

season=pd.get_dummies(test_df['season'],prefix='season')

test_df=pd.concat([test_df,season],axis=1)

test_df.head()
# # # same for weather. this is bcoz this will enhance features.

weather=pd.get_dummies(df['weather'],prefix='weather')

df=pd.concat([df,weather],axis=1)

df.head()

weather=pd.get_dummies(test_df['weather'],prefix='weather')

test_df=pd.concat([test_df,weather],axis=1)

test_df.head()
# # # now can drop weather and season.

df.drop(['season','weather'],inplace=True,axis=1)

df.head()

test_df.drop(['season','weather'],inplace=True,axis=1)

test_df.head()





# # # also I dont prefer both registered and casual but for ow just let them both.
df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]

df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]

df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]

df['year'] = [t.year for t in pd.DatetimeIndex(df.datetime)]

df['year'] = df['year'].map({2011:0, 2012:1})

df.head()
test_df["hour"] = [t.hour for t in pd.DatetimeIndex(test_df.datetime)]

test_df["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_df.datetime)]

test_df["month"] = [t.month for t in pd.DatetimeIndex(test_df.datetime)]

test_df['year'] = [t.year for t in pd.DatetimeIndex(test_df.datetime)]

test_df['year'] = test_df['year'].map({2011:0, 2012:1})

test_df.head()
# now can drop datetime column.

df.drop('datetime',axis=1,inplace=True)

df.head()
cor_mat= df[:].corr()

mask = np.array(cor_mat)

mask[np.tril_indices_from(mask)] = False

fig=plt.gcf()

fig.set_size_inches(30,12)

sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)
df.drop(['casual','registered'],axis=1,inplace=True)
df.head()
# with hour.

sns.factorplot(x="hour",y="count",data=df,kind='bar',size=5,aspect=1.5)
sns.factorplot(x="month",y="count",data=df,kind='bar',size=5,aspect=1.5)

# note that month affects season and that effects wheteher people take bike or not. like climate conditions rainy,hazy etc... .
sns.factorplot(x="year",y="count",data=df,kind='bar',size=5,aspect=1.5)

# 0 for 2011 and 1 for 2012. Hence demand has increased over the years.
sns.factorplot(x="day",y='count',kind='bar',data=df,size=5,aspect=1)
# for temp

plt.scatter(x="temp",y="count",data=df,color='#ff4125')
new_df=df.copy()

new_df.temp.describe()

new_df['temp_bin']=np.floor(new_df['temp'])//5

new_df['temp_bin'].unique()

# now we can visualize as follows

sns.factorplot(x="temp_bin",y="count",data=new_df,kind='bar')
# and similarly we can do for other continous variables and see how it effect the target variable.
df.head()
df.columns.to_series().groupby(df.dtypes).groups
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

d

    
rmsle_frame=pd.DataFrame(d)

rmsle_frame
sns.factorplot(y='Modelling Algo',x='RMSLE',data=rmsle_frame,kind='bar',size=5,aspect=2)
sns.factorplot(x='Modelling Algo',y='RMSLE',data=rmsle_frame,kind='point',size=5,aspect=2)
#for random forest regresion.

no_of_test=[500]

params_dict={'n_estimators':no_of_test,'n_jobs':[-1],'max_features':["auto",'sqrt','log2']}

clf_rf=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='neg_mean_squared_log_error')

clf_rf.fit(x_train,y_train)

pred=clf_rf.predict(x_test)

print((np.sqrt(mean_squared_log_error(pred,y_test))))
clf_rf.best_params_
# for KNN

n_neighbors=[]

for i in range (0,50,5):

    if(i!=0):

        n_neighbors.append(i)

params_dict={'n_neighbors':n_neighbors,'n_jobs':[-1]}

clf_knn=GridSearchCV(estimator=KNeighborsRegressor(),param_grid=params_dict,scoring='neg_mean_squared_log_error')

clf_knn.fit(x_train,y_train)

pred=clf_knn.predict(x_test)

print((np.sqrt(mean_squared_log_error(pred,y_test))))

clf_knn.best_params_
pred=clf_rf.predict(test_df.drop('datetime',axis=1))

d={'datetime':test['datetime'],'count':pred}

ans=pd.DataFrame(d)

ans.to_csv('submission_19.csv',index=False) # saving to a csv file for predictions on kaggle.

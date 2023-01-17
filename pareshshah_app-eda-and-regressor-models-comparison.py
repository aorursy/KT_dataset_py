# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt   

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression





from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb



%matplotlib inline 
df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
df.shape
df.dtypes
df.sample(10)
# Dropping Current Ver and Android Ver as they are not useful for EDA, Analysis or Model

df.drop(['Android Ver','Current Ver'],axis=1,inplace=True)
# While performing Data Analysis - Finding top 10 Reviewed Apps, it was observed that there are duplicate entries in the dataset

# There are 483 rows that are exactly duplicate for all columns hence these will be removed.

print("Total Rows before:",df.shape)

df.drop_duplicates(inplace=True)

print("Total Rows after:",df.shape)

# There are still more duplicates except Reviews Columns that has different numbers see below example

df.sort_values(['Reviews'],ascending=False)

# In this case we will sort dataset using App+Reviews in descending order and then delete duplicate rows keeping first row

df=df.sort_values(['App','Reviews'],ascending=False)



# Check if data is sorted

df.loc[(df.App=='Facebook')]

cols=list(df.columns.values)

cols.remove('Reviews')

cols

print("Total Rows before:",df.shape)

df.drop_duplicates(subset=cols,inplace=True)

print("Total Rows after:",df.shape)

# Let's check if there are more duplicate apps

dup_df=df[df.duplicated(subset=['App'],keep=False)]

dup_df.sort_values('App',ascending=False)

# There are still 121 duplicate apps however it is difficult to decide which one to keep as Duplicate

# Apps are either have different categories or reviews or last updates or combination of fields.

# Hence Category will be updated with Genres first and then 

# dataset will be sorted by Last Updated + Reviews and First row will be retained.

dup_index = list(df[df.duplicated(subset=['App'],keep=False)].index)

df.loc[dup_index]['Category']=df['Genres'][dup_index].str.upper()



# Sort on Last Updated and Reviews

df=df.sort_values(['Last Updated','Reviews'],ascending=False)



print("Total Rows before:",df.shape)

df.drop_duplicates(subset=['App'],inplace=True)

print("Total Rows after:",df.shape)

# Let's check if there are more duplicate apps

dup_df=df[df.duplicated(subset=['App'],keep=False)]

dup_df.sort_values('App',ascending=False)

df.isnull().sum()
# Delete all rows that have NA Rating



df.dropna(subset=['Rating'],inplace=True)

df.loc[(df.Type.isna()==True)]
# Since above row doesn't provide any useful/meaningful info hence will remove this row

df.dropna(subset=['Type'],inplace=True)
df.loc[(df['Content Rating'].isna()==True)]
# Something's wrong with above record as values in columns looks displaced. Hence will delete this row from the dataset

df.dropna(subset=['Content Rating'],inplace=True)
df.shape
df.isnull().sum()
df.dtypes

# Reviews contains numeric values and will convert into int data type

# Price contains float values and will convert into float data type

# Last Updated contains date and will convert into datatime data type

df['Reviews']=df['Reviews'].astype('int64')

df['Price'] = df['Price'].str.replace('$','')

df['Price']=df['Price'].astype('float64')

df['Installs']=df['Installs'].str.replace(',','').replace('+','')

df['Installs']=df['Installs'].str.replace('+','')

df['Installs']=df['Installs'].astype('float64')

df['Last Updated']=df['Last Updated'].astype('datetime64[ns]')

df.dtypes
# Let's find out number of apps in each category in graphical view

df[['Category','App']].groupby('Category').count().sort_values('App',ascending=False).plot(kind='bar',figsize=(20,10),title='Number of Apps per Category')

family_df=df.loc[df.Category=='FAMILY']
# graphical view of Free vs Paid Apps

df[['Type','App']].groupby('Type').count().sort_values('App',ascending=False).plot(kind='pie',subplots=True,figsize=(20,10),table=True,title='Free vs Paid Apps Distribution',autopct='%.2f')

family_df[['Type','App']].groupby('Type').count().sort_values('App',ascending=False).plot(kind='pie',subplots=True,figsize=(20,10),table=True,title='FAMILY Apps - Free vs Paid  Distribution',autopct='%.2f')

# graphical view of Apps count across Content Ratings 

df[['Content Rating','App']].groupby('Content Rating').count().sort_values('App',ascending=False).plot(kind='pie',subplots=True,figsize=(20,10),table=True,title='Apps Distribution based on Content Ratings',autopct='%.2f')

family_df[['Content Rating','App']].groupby('Content Rating').count().sort_values('App',ascending=False).plot(kind='pie',subplots=True,figsize=(20,10),table=True,title='FAMILY Apps Distribution based on Content Ratings',autopct='%.2f')

# Graphical view of number of apps for each ratings

df[['Rating','App']].groupby('Rating').count().sort_values('Rating',ascending=False).plot(kind='bar',figsize=(20,10),title='App Ratings')

family_df[['Rating','App']].groupby('Rating').count().sort_values('Rating',ascending=False).plot(kind='bar',figsize=(20,10),title='FAMILY App Ratings')

# Graphical view of Top 10 Apps based on number of Reviews

df[['App','Reviews']].sort_values('Reviews',ascending=False).head(10).plot(kind='bar',figsize=(20,10),x='App',y='Reviews',title='Top 10 Reviewed Apps ')

family_df[['App','Reviews']].sort_values('Reviews',ascending=False).head(10).plot(kind='bar',figsize=(20,10),x='App',y='Reviews',title='Top 10 Reviewed FAMILY Apps ')
# Graphical view of Number of Installations



df[['Installs','App']].groupby('Installs').count().sort_values('Installs',ascending=False).plot(kind='bar',figsize=(20,10),title='Number of Apps for Installations Size')

family_df[['Installs','App']].groupby('Installs').count().sort_values('Installs',ascending=False).plot(kind='bar',figsize=(20,10),title='Number of FAMILY Apps for Installations Size')
# Top Apps with 1+ billion Installs and Highest review

df.loc[(df.Installs==1000000000)].sort_values('Reviews',ascending=False)
# Top FAMILY Apps with 1+ billion Installs and Highest review

family_df.loc[(family_df.Installs==1000000000)].sort_values('Reviews',ascending=False)

# Top FAMILY Apps with 100 million Installs and Highest review

family_df.loc[(family_df.Installs==100000000)].sort_values('Reviews',ascending=False)
# Graphical view of how many apps are kept updated

year_df = df[['App']]

year_df['Last Updated']=df['Last Updated'].dt.year

year_df.groupby('Last Updated').count().sort_values('Last Updated',ascending=False).plot(kind='bar',figsize=(20,10),title='Number of Apps Last Updated Year')

# Removing following columns as not relevant for Model

df.drop(['Last Updated'],axis=1,inplace=True)
df.dtypes
# Convert String values to Numerical - Category, Type, Content Rating, Genres



lblencoder=LabelEncoder()

df['App']=lblencoder.fit_transform(df['App'])

df['Category']=lblencoder.fit_transform(df['Category'])

df['Genres']=lblencoder.fit_transform(df['Genres'])

df['Content Rating']=lblencoder.fit_transform(df['Content Rating'])

df['Type']=lblencoder.fit_transform(df['Type'])



df.dtypes
# Convert Size

# Size is numeric value 

# Multiple values in Kb with 1024 and Mb with 1048576 (1024*1024)





dup_index = list(df[df['Size'].str.endswith('k')==True].index)

df['Size'][dup_index]=df['Size'][dup_index].str.replace('k','').astype('float64') * 1024



dup_index = list(df[df['Size'].str.endswith('M')==True].index)

df['Size'][dup_index]=df['Size'][dup_index].str.replace('M','').astype('float64') * 1024 * 1024



df[df['Size'] == 'Varies with device'] = 0

df['Size']=df['Size'].astype('float64')

model_results=[]





def runModel(model, x_tr, x_te, y_tr, y_te,model_name):

    model.fit(x_tr,y_tr)

    r2_tr=model.score(x_tr, y_tr)

    r2_te=model.score(x_te, y_te)

    print(model_name,'- Train R2',r2_tr)

    print(model_name,'- Test R2',r2_te)



    y_predict = model.predict(x_te)

    mse=metrics.mean_squared_error(y_predict, y_te)



    rmse=np.sqrt(mse)

    print('RMSE',rmse)

    model_results.append({'Model':model_name,'Train R2':r2_tr,'Test R2':r2_te,'RMSE':rmse})

    

X=df.drop(['Rating'],axis=1)

Y=df['Rating']

Y=Y.astype('int')



x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=1)

std_scale=StandardScaler()

x_train=std_scale.fit_transform(x_train)

x_test=std_scale.fit_transform(x_test)


lgr=LogisticRegression()

y_train = lblencoder.fit_transform(y_train)

y_test = lblencoder.fit_transform(y_test)



runModel(lgr,x_train,x_test,y_train,y_test,'LogisticRegression')
X=df.drop(['Rating'],axis=1)

Y=df['Rating']



x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=1)



linreg=LinearRegression()



runModel(linreg,x_train,x_test,y_train,y_test,'LinearRegression')


dtr=DecisionTreeRegressor(random_state=1)

runModel(dtr,x_train,x_test,y_train,y_test,'DecisionTreeRegressor')



dtr1=DecisionTreeRegressor(random_state=7)

criterion=('mse', 'friedman_mse', 'mae')

max_depth = [int(x) for x in np.linspace(1, 30, num = 10)]

max_depth.append(None)

max_leaf_nodes = [int(x) for x in np.linspace(2, 30, num = 10)]



min_samples_split = [2,3,4,5,6,7,8,9,10]

min_samples_leaf = [1,2,3,4]



random_grid1 = {'criterion': criterion,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'max_leaf_nodes': max_leaf_nodes}



rf_random1 = RandomizedSearchCV(estimator=dtr1,param_distributions=random_grid1,n_iter=100,cv=3,verbose=2,random_state=7,n_jobs=-1)

rf_random1.fit(x_train,y_train)

rf_random1.best_params_


dtr2=DecisionTreeRegressor(min_samples_split=8,min_samples_leaf=4,max_leaf_nodes=14,max_depth=4,criterion='mse',random_state=1)

runModel(dtr2,x_train,x_test,y_train,y_test,'DecisionTreeRegressor with Hyperparamters')





rfr1 = RandomForestRegressor(n_estimators=100,random_state=1)

runModel(rfr1,x_train,x_test,y_train,y_test,'RandomForestRegressor')



rfr2=RandomForestRegressor(random_state=7)



n_estimators = [int(x) for x in np.linspace(start=50, stop=110, num=10)]

criterion=('mse', 'friedman_mse', 'mae')

max_depth = [int(x) for x in np.linspace(1, 30, num = 10)]

max_depth.append(None)

max_leaf_nodes = [int(x) for x in np.linspace(2, 30, num = 10)]



min_samples_split = [2,3,4,5,6,7,8,9,10]

min_samples_leaf = [1,2,3,4]



random_grid1 = {'criterion': criterion,

               'n_estimators': n_estimators,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'max_leaf_nodes': max_leaf_nodes}



rf_random1 = RandomizedSearchCV(estimator=rfr2,param_distributions=random_grid1,n_iter=100,cv=3,verbose=2,random_state=7,n_jobs=-1)

rf_random1.fit(x_train,y_train)

rf_random1.best_params_


rfr3 = RandomForestRegressor(n_estimators=63,max_depth=7,criterion='mse',min_samples_split=5,min_samples_leaf=4,max_leaf_nodes=30,random_state=1)

runModel(rfr3,x_train,x_test,y_train,y_test,'RandomForestRegressor with Hyperparameters')





br = BaggingRegressor(n_estimators=300,random_state=1)

runModel(br,x_train,x_test,y_train,y_test,'BaggingRegressor')


br1=BaggingRegressor(random_state=7)

n_estimators = [int(x) for x in np.linspace(start = 1, stop = 300, num = 50)]

max_features = [1,2,3,4,5,6,7,8,9]

max_samples = [1,2,3,4,5,6,7,8,9]

bootstrap = [True]

bootstrap_features = [True, False]

oob_score = [True, False]

warm_start = [False]

random_grid1 = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_samples': max_samples,

               'bootstrap': bootstrap,

               'bootstrap_features': bootstrap_features,

               'oob_score': oob_score,

               'warm_start': warm_start}



rf_random1 = RandomizedSearchCV(estimator=br1,param_distributions=random_grid1,n_iter=100,cv=3,verbose=2,random_state=7,n_jobs=-1)

rf_random1.fit(x_train,y_train)

rf_random1.best_params_


br2 = BaggingRegressor(n_estimators=500,max_samples=1000,random_state=1)

runModel(br2,x_train,x_test,y_train,y_test,'BaggingRegressor with Hyperparameters')


abr = AdaBoostRegressor(n_estimators=300,random_state=1)

runModel(abr,x_train,x_test,y_train,y_test,'AdaBoostRegressor')
abr1=AdaBoostRegressor(random_state=7)



n_estimators = [int(x) for x in np.linspace(start = 1, stop = 300, num = 50)]

learning_rate = [1,2,3,4,5,6,7,8,9]

random_grid1 = {'n_estimators': n_estimators,

               'learning_rate': learning_rate}



rf_random1 = RandomizedSearchCV(estimator=abr1,param_distributions=random_grid1,n_iter=100,cv=3,verbose=2,random_state=7,n_jobs=-1)

rf_random1.fit(x_train,y_train)

rf_random1.best_params_


abr2 = AdaBoostRegressor(n_estimators=1,learning_rate=5,random_state=1)

runModel(abr2,x_train,x_test,y_train,y_test,'AdaBoostRegressor with Hyperparameters')


gbr = GradientBoostingRegressor(n_estimators=300,random_state=1)

runModel(gbr,x_train,x_test,y_train,y_test,'GradientBoostingRegressor')
gbr1=GradientBoostingRegressor(random_state=7)



n_estimators = [int(x) for x in np.linspace(start = 1, stop = 300, num = 10)]



max_depth = [int(x) for x in np.linspace(1, 40, num = 10)]

max_depth.append(None)

min_samples_split = [2,3,4,5,6,7,8,9,10]

min_samples_leaf = [1,2,3,4,5,6,7,8,9,10]

criterion = ['friedman_mse','mse','mae']

random_grid1 = {'n_estimators': n_estimators,

                'criterion': criterion,

                'max_depth': max_depth,

                'min_samples_split': min_samples_split,

                'min_samples_leaf': min_samples_leaf}



rf_random1 = RandomizedSearchCV(estimator=gbr1,param_distributions=random_grid1,n_iter=100,cv=3,verbose=2,random_state=7,n_jobs=-1)

rf_random1.fit(x_train,y_train)

rf_random1.best_params_
gbr2 = GradientBoostingRegressor(n_estimators=300,random_state=1)

runModel(gbr2,x_train,x_test,y_train,y_test,'GradientBoostingRegressor with Hyperparameters')


xgbr1=xgb.XGBRegressor(n_estimators=100,random_state=5)



runModel(xgbr1,x_train,x_test,y_train,y_test,'XGBRegressor')
xgbr2=xgb.XGBRegressor(random_state=7)



n_estimators = [int(x) for x in np.linspace(start = 100, stop = 600, num = 50)]



max_depth = [int(x) for x in np.linspace(5, 50, num = 20)]

max_depth.append(None)

max_delta_step = [0,1,2,3,4]

random_grid1 = {'n_estimators': n_estimators,

                'max_depth': max_depth,

                'max_delta_step': max_delta_step}



rf_random1 = RandomizedSearchCV(estimator=xgbr2,param_distributions=random_grid1,n_iter=100,cv=3,verbose=2,random_state=7,n_jobs=-1)

rf_random1.fit(x_train,y_train)

rf_random1.best_params_


xgbr3=xgb.XGBRegressor(n_estimators=191,max_depth=5,max_delta_step=3,random_state=1)



runModel(xgbr3,x_train,x_test,y_train,y_test,'XGBRegressor with Hyperparameters')
dtr_df=pd.DataFrame(model_results,columns=['Model','Train R2','Test R2','RMSE'])

dtr_df


dtr_df.plot(kind='bar',figsize=(20,10),x='Model',title='Comparison of Regressor Models Results')

import numpy as np 

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

data.head()

# Read to data set
data.isnull().any()
CopyData = data.copy() # I copy the data set

CopyData.fillna(CopyData['salary'].mean(),inplace=True) # I fill in the missing values with the average.

#I will continue with all transactions with the copy data set.
CopyData.isnull().any()

# Its Done
CopyData.describe().T

# Students in the classroom are on the same average.
import plotly.express as px

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Box(y=CopyData['ssc_p'],name='Secondary Education percentage',boxpoints='all',jitter=0.3,marker_color='rgb(0,255,0)',line_color='rgb(50,205,50)'))

fig.add_trace(go.Box(y=CopyData['hsc_p'],name='Higher Secondary Education percentage',boxpoints='suspectedoutliers',jitter=0.3,marker_color='rgb(32,178,170)',line_color='rgb(102,205,170)'))

fig.add_trace(go.Box(y=CopyData['degree_p'],name='Degree Percentage',boxpoints='suspectedoutliers',jitter=0.3,marker_color='rgb(47,79,79)',line_color='rgb(0,128,128)'))

fig.add_trace(go.Box(y=CopyData['etest_p'],name='Employability test percentage',boxpoints='all',jitter=0.3,marker_color='rgb(127,255,212)',line_color='rgb(64,224,208)'))

fig.add_trace(go.Box(y=CopyData['mba_p'],name='MBA percentage',boxpoints='all',jitter=0.3,marker_color='rgb(0,191,255)',line_color='rgb(25,25,112)'))
import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(),ax=ax,annot=True,linewidths=.5,cmap="YlGnBu");
sns.pairplot(CopyData, kind  ="reg");
import plotly.express as px

colors = ['Gold','GoldenRod','Gray']

fig = go.Figure()

fig.add_trace(go.Bar(x=['Commerce','Science','Arts'],y=[113,91,11],marker_color=colors))

fig.show()
colors = ['Crimson','LightPink']

fig = go.Figure()

fig.add_trace(go.Bar(x=['Others','Central'],y=[131,84],marker_color=colors))

fig.show()
f, ax = plt.subplots(figsize=(8, 5))

sns.barplot(x='salary', y='gender',hue='hsc_s', data=CopyData,ax=ax,color='Crimson');
f, ax = plt.subplots(figsize=(10, 8))

sns.barplot(x='salary', y='gender',hue=CopyData['degree_t'], data=CopyData,ax=ax,color='BlueViolet');
CopyDataX = CopyData.copy() # Let's copy whatever happens ;)

dms = pd.get_dummies(CopyData[['ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation','status','gender']])

dms.head() # Let's clean this one too
y = CopyData['status']

X_ = CopyData.drop(['ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation','status','gender'],axis=1).astype('float64')

X_.head()
X = pd.concat([X_,dms[['ssc_b_Central','hsc_b_Central','hsc_s_Arts','hsc_s_Commerce','hsc_s_Science','degree_t_Comm&Mgmt','degree_t_Others','degree_t_Sci&Tech','workex_Yes','specialisation_Mkt&Fin','status_Placed','gender_M']]],axis=1)

X.head()
CopyDataT = X
from sklearn.model_selection import train_test_split

y = CopyDataT['status_Placed']

x = CopyDataT.drop(['status_Placed'],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)

# We made our train and test distinctions.
from sklearn.ensemble import RandomForestClassifier

RandomFC = RandomForestClassifier()

RandomFCmodel = RandomFC.fit(x_train,y_train)

print(RandomFCmodel)
from sklearn.metrics import accuracy_score

y_prediction = RandomFCmodel.predict(x_test)

accuracy_score(y_test,y_prediction)

# Yes, we could have built a beautiful model.
Y_p = pd.DataFrame({

    'y_pred':y_prediction,

    'y_real':y_test

})

Y_p.head(10)

# Quite successful.
from sklearn.model_selection import GridSearchCV

RF_params = {

    'max_depth':[2,3,5,8,10],

    'max_features':[2,5,8],

    'n_estimators':[10,500,1000,2000],

    'min_samples_split':[2,5,10]

}

RF = RandomForestClassifier()

RF_model_cv = GridSearchCV(RF,RF_params,cv=10,n_jobs=-1,verbose=2)

RF_model_cv.fit(x_train,y_train)

# We are searching for hyperparameters.
print('Best Parameter:' + str(RF_model_cv.best_params_))
RF_tuned = RandomForestClassifier(max_depth=RF_model_cv.best_params_['max_depth'],

                                  max_features=RF_model_cv.best_params_['max_features'],

                                 min_samples_split=RF_model_cv.best_params_['min_samples_split'],

                                 n_estimators=RF_model_cv.best_params_['n_estimators']).fit(x_train,y_train)
y_predic_new = RF_tuned.predict(x_test)

accuracy_score(y_test,y_predic_new)

# Not much has changed, but we used the optimum parameters.
Importance = pd.DataFrame({'Importance':RF_tuned.feature_importances_*100},

                         index=x_train.columns)

Importance_sorted = Importance.sort_values(by='Importance',

                                          axis=0,

                                          ascending=True).plot(kind='barh',color='Coral');
MPData = pd.DataFrame({

    'ssc_p':CopyDataT['ssc_p'],

    'hsc_p':CopyDataT['hsc_p'],

    'degree_p':CopyDataT['degree_p'],

    'etest_p':CopyDataT['etest_p'],

    'mba_p':CopyDataT['mba_p'],

    'salary':CopyDataT['salary']

})

MPData.head()

# I don't think the rest will affect the rest except mba but I want to include them too.
# We do train and test separation.

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict

x_x = data.drop(['salary'],axis=1)

y_y = data['salary']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)
from sklearn.linear_model import LinearRegression

Linear = LinearRegression()

LinearModel = Linear.fit(x_train,y_train)

print('Linear Model İntercept' + str(LinearModel.intercept_))

print('Linear Model Coef' + str(LinearModel.coef_))
rmse = np.sqrt(mean_squared_error(y_train,LinearModel.predict(x_train)))

print('Train Error:',rmse)

t_rmse = np.sqrt(mean_squared_error(y_test,LinearModel.predict(x_test)))

print('Test Error:',t_rmse)

print('Verified Error:',cross_val_score(LinearModel,x_train,y_train,cv=10,scoring='r2').mean())
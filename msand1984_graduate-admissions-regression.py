# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
df.info()
df.columns
# Check any number of columns with NaN

print(df.isnull().any().sum(), ' / ', len(df.columns))

# Check any number of data points with NaN

print(df.isnull().any(axis=1).sum(), ' / ', len(df))
df.rename(columns= {'Chance of Admit ':'Chance of Admit', 'LOR ' :'LOR'},inplace=True)
df.drop(df.columns[0] , axis = 1 , inplace= True)
df.head()
f, ax = plt.subplots(2,2,figsize=(8,4))

vis1 = sns.distplot(df['GRE Score'],bins=10, ax= ax[0][0]) ;

vis2 = sns.distplot(df['TOEFL Score'],bins=10, ax=ax[0][1]) ;

vis3 = sns.distplot(df['SOP'],bins=10, ax=ax[1][0]) ;

vis4 = sns.distplot(df["University Rating"],bins=10, ax=ax[1][1]) ;
fig = plt.figure(figsize=(50,50))

for i in range(1, 7):

    ax = fig.add_subplot(3, 3, i)

    sns.scatterplot(x=df['Chance of Admit'], y= df.iloc[:,i], hue=df.Research) ;    

    plt.xlabel('Chance of Admit')

    plt.ylabel(df.columns[i])
fig,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(df.corr() , annot= True ) ;
df.Research.value_counts()
plt.scatter(data = df , x= 'GRE Score' , y = 'CGPA') ;
sns.lmplot(data=df , hue='Research' , x= 'GRE Score' , y = 'CGPA') ;
plt.scatter(data = df , x= 'GRE Score' , y = 'University Rating') ;
df[df['CGPA'] > 0.80].plot(kind = 'scatter' , x = 'GRE Score' , y ='Chance of Admit') ;
df[df['Chance of Admit'] > 0.80].plot(kind = 'scatter' , x = 'GRE Score' , y ='University Rating') ;
from numpy import percentile



for col in df.columns :

    q25 = percentile(df[col] ,25)

    q75 = percentile(df[col] , 75)

    IQR = q75-q25

    cutoff = IQR*1.5

    upper_cutoff = q75 + cutoff

    lowe_cutoff = q25 - cutoff

    outliers = [ x  for x in df[col] if x < lowe_cutoff or x > upper_cutoff]

    print('Identified outliers: %d' % len(outliers))

    outliers_removed = [ x  for x in df[col] if x >= lowe_cutoff and x <= upper_cutoff]

    print('Non-outlier observations: %d' % len(outliers_removed))

    df_out = df.loc[(df[col] > lowe_cutoff) & (df[col] < upper_cutoff)]
from sklearn.model_selection import train_test_split



y = df['Chance of Admit']

x = df.drop('Chance of Admit',axis =1)

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error , r2_score

lr = LinearRegression()

lr.fit(x_train,y_train)

y_predict = lr.predict(x_test)

R2score = r2_score( y_test , y_predict )*100

print('The R2score for Linear Regression is ' , R2score)
# Import the model we are using

from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data

rf.fit(x_train, y_train);

y_predict = rf.predict(x_test)

print('The R2score for Random Forest Regressor is ' , r2_score( y_test , y_predict ))



#from sklearn.model_selection import train_test_split , cross_val_score



#scores =cross_val_score(rf,x_data,y_data,cv=2,scoring='r2')
featureImportance = pd.Series(rf.feature_importances_ , index = x_train.columns).sort_values(ascending = True)

sns.barplot(featureImportance,y=featureImportance.index);

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Visualizing Important Features")
df_x_train = x_train[ ['CGPA', 'GRE Score' ,'TOEFL Score']]

df_x_test = x_test[ ['CGPA', 'GRE Score', 'TOEFL Score' ]]

rf_improved = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data

rf_improved.fit(df_x_train, y_train);

y_predict = rf_improved.predict(df_x_test)



R2score = r2_score( y_test , y_predict )

print('The R2score for Random Forest Regressor is ' , (R2score*100))

# Import the model we are using

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state = 42)

# Train the model on training data

dt.fit(x_train, y_train);

y_predict = dt.predict(x_test)



R2score = r2_score( y_test , y_predict )

print('The R2score for Decision tree Regressor is ' , (R2score *100))
from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)



y_rbf = svr_rbf.fit(x_train, y_train).predict(x_test)



R2score_rbf = r2_score( y_test , y_rbf )

print('The R2score for RBF is ' , R2score_rbf )
import xgboost

xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,

                           colsample_bytree=1, max_depth=7)

xgb_pred = xgb.fit(x_train,y_train).predict(x_test)

print('The R2score for xgboost is ' , (r2_score( y_test , xgb_pred)*100))
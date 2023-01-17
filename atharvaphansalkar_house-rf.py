# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import sklearn 	

	

	

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score 	

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

 	

from sklearn.metrics import mean_squared_error, r2_score,accuracy_score











sets= pd.read_csv('/kaggle/input/housepricesalespredication/Data.csv')





sets.info()

newset=sets.loc[(sets['Market_dist']>=10000) & (sets['Parking_type']=='Covered') & (sets['City_type']=="CAT B")]



newset=newset.reset_index()



newset.head(7)
newset.loc[newset['Parking_type']=='Open' ,'Parking_type']=1



newset.loc[newset['City_type']=='CAT B' ,'City_type']=2



sets.loc[sets['Parking_type']=="Open",'Parking_type']=2



sets.loc[sets['Parking_type']=="No Parking",'Parking_type']=0



sets.loc[sets['Parking_type']=="Covered",'Parking_type']=4



sets.loc[sets['Parking_type']=="Not Provided",'Parking_type']=1





sets.loc[sets['City_type']=="CAT A",'City_type']=200





sets.loc[sets['City_type']=="CAT B",'City_type']=100





sets.loc[sets['City_type']=="CAT C",'City_type']=40



sets.head(5)
sets["Market_dist"].fillna(sets['Market_dist'].mean(),inplace=True)

                           

sets["Taxi_dist"].fillna(sets['Taxi_dist'].mean(),inplace=True)



sets["Hospital_dist"].fillna(sets['Hospital_dist'].mean(),inplace=True)



sets["Carpet_area"].fillna(sets['Carpet_area'].mean(),inplace=True)



sets["Builtup_area"].fillna(sets['Builtup_area'].mean(),inplace=True)



sets["Rainfall"].fillna(sets['Rainfall'].mean(),inplace=True)







sets["newpark"]=sets['Parking_type'].factorize()[0]

sets["newcity"]=sets['City_type'].factorize()[0]



sets.head(5)
y=sets.Price_house

x=sets.drop(['Price_house','newpark','newcity'],axis=1)



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=4)
pipe=make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(n_estimators=73))



hyperparameters={'randomforestregressor__max_features' :['auto'],

                 'randomforestregressor__max_depth':[5] }



clf=GridSearchCV(pipe, hyperparameters,cv=10)



clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)



sets.info()
print(mean_squared_error(y_test, y_pred))



print(r2_score(y_test,y_pred))

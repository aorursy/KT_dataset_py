import pandas as pd

import numpy as np                            #import libariy

from matplotlib import pyplot as plt 

import urllib



df=pd.read_csv("../input/heaptitis.csv")

#reading csv file from uci 

df.head()       

df.tail()# to view are datasets
# we can see in our dataset missing values are present(?)





df.shape                                                       #it return data shape 155 rows and 20 columns

print("No of featurs:",df.shape[1])
''' analysis'''

df.dtypes



# some column are object type we need to convert it into numeric because machine learning model work with numeric values

col_mask=df.isnull().any(axis=0)   #to check how may columns that have missing values



print(col_mask)

                         # those column have missing values they are True and those not have False




df.head()
#df=df.astype(float)

print("Object are changed into numeric")

print(df.dtypes)





                                            # print(df.median()) it will print median value for each comlumn





df.fillna(df.median(), inplace=True)           # filing missing values  with median

df.head()           
y=df['CLASS']   #target variable 

#x=df.drop('CLASS',axis=1)  #input varuable or training data

x=df.iloc[0:154,2:].values
# spliting training and testing data with help of scikit learn

y.shape
#now we handle missing values and use fillna method for this and we replace missing value with mean of that columns

              #input data without target variable thats why i drop first columns of dataset and assign to x



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

print("Original data x:",x.shape)

print("After spliting x_traing shape:",np.shape(x_train))

print("After spliting x_test size:",np.shape(x_test))





from sklearn.ensemble import RandomForestClassifier

tree=RandomForestClassifier(criterion='gini',max_depth=2,n_estimators=3,random_state=0)



                                                     

tree.fit(x_train,y_train)                                                                                 # put data orignal scale

y_pred=tree.predict(x_test)



from sklearn.metrics import accuracy_score

print('misclassified samples: %d'%(y_test!=y_pred).sum())

print("accuracy:%f"%accuracy_score(y_test,y_pred))











# to find best parameter we need to do hyperparameter tunning 
from sklearn.model_selection import GridSearchCV



#setup the parametrs and distribution to sample 

param_dist={'n_estimators':[5,10,15,20,],'max_features':['auto','sqrt'],'max_depth':[3,None],'min_samples_leaf':np.arange(1,10),"criterion": ["gini", "entropy"],'bootstrap':[True,False]}

rfc=RandomForestClassifier()

cv=GridSearchCV(rfc,param_dist,cv=5)

cv.fit(x_train,y_train)

y_p=cv.predict(x_test)



print("tuned parametrs:{}".format(cv.best_params_))

print("best Score is: {}".format(cv.best_score_))
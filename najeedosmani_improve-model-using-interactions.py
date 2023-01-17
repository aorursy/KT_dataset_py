# This is a simple demonstration of how interactions are made.
# EX:

feat1=[1,2,3,4]
feat2=[2,3,4,5]
inter_feat=[]
for i  in range(0,4):
    inter_feat.append(feat1[i]*feat2[i])
inter_feat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston

# This line of code would import our dataset.
l=load_boston()

# This is a bunch format datatype which looks almost similar to a dictionary.
l.keys()

# This keys would help us to get the info of all that we want to know about the dataset.
l.data
# This loads the data.
l.feature_names

# This is a list of all the feature's names that we gonna use.
boston_data1=pd.DataFrame(data=l.data,columns=l.feature_names)
boston_data2=pd.DataFrame(data=l.data,columns=l.feature_names)

# Here we have created two copies of the datasets.
# Lets just check the head of our dataset.
# We would load first five rows.

boston_data1.head(n=5)
boston_data1['Target']=l.target
boston_data2['Target']=l.target

# We add target variable in both the datasets. 
# Lets now again check the head of the datset.

boston_data1.head(n=5)
# Lets just quickly plot a pairplot to have an overview of whats happening.

sns.pairplot(boston_data1)
corr_dataset=boston_data1.corr()
corr_dataset
sns.heatmap(corr_dataset)
X=boston_data1.drop('Target',axis=1)
y=boston_data1['Target']
from sklearn.model_selection import train_test_split
# Here a test size of 20% is used with a random state of 40.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)
from sklearn.linear_model import LinearRegression
lr_bef_inter=LinearRegression()

# Linear model before using interactions.
lr_bef_inter.fit(X_train,y_train)
lr_bef_inter.score(X_test,y_test)
# Lets again load the correlation dataset.

corr_dataset
sns.heatmap(corr_dataset)
# This is the function that we made to get interacting features.

def int_feat(cols):
    col1=cols[0]
    col2=cols[1]
    return col1*col2
boston_data2['int_CRIM_RAD']=boston_data2[['CRIM','RAD']].apply(int_feat,axis=1)
boston_data2['int_DIS_ZN']=boston_data2[['DIS','ZN']].apply(int_feat,axis=1)
boston_data2['int_NOX_INDUS']=boston_data2[['NOX','INDUS']].apply(int_feat,axis=1)
boston_data2['int_AGE_INDUS']=boston_data2[['AGE','INDUS']].apply(int_feat,axis=1)
boston_data2['int_DIS_INDUS']=boston_data2[['DIS','INDUS']].apply(int_feat,axis=1)
boston_data2['int_RAD_INDUS']=boston_data2[['RAD','INDUS']].apply(int_feat,axis=1)
boston_data2['int_TAX_INDUS']=boston_data2[['TAX','INDUS']].apply(int_feat,axis=1)
boston_data2['int_LSTAT_INDUS']=boston_data2[['LSTAT','INDUS']].apply(int_feat,axis=1)
boston_data2['int_NOX_AGE']=boston_data2[['NOX','AGE']].apply(int_feat,axis=1)
boston_data2['int_NOX_DIS']=boston_data2[['NOX','DIS']].apply(int_feat,axis=1)
boston_data2['int_NOX_RAD']=boston_data2[['NOX','RAD']].apply(int_feat,axis=1)
boston_data2['int_NOX_TAX']=boston_data2[['NOX','TAX']].apply(int_feat,axis=1)
boston_data2['int_LSTAT_RM']=boston_data2[['LSTAT','RM']].apply(int_feat,axis=1)
boston_data2['int_DIS_AGE']=boston_data2[['DIS','AGE']].apply(int_feat,axis=1)
boston_data2['int_LSTAT_AGE']=boston_data2[['LSTAT','AGE']].apply(int_feat,axis=1)


# This lines of code will add new interacted feature of our selected features to our dataset.
# We can see some new features been added to our dataset.

boston_data2.head()
X_new=boston_data2.drop('Target',axis=1)
y_new=boston_data2['Target']
X_New_train, X_New_test, y_New_train, y_New_test = train_test_split(X_new, y_new, test_size=0.20, random_state=40)
# This is the new model that we created.

lr_aft_int=LinearRegression()
lr_aft_int.fit(X_New_train,y_New_train)
lr_aft_int.score(X_New_test,y_New_test)
print('Accuracy of the model before adding interacting features     : {} %'.format(lr_bef_inter.score(X_test,y_test)*100))
print('Accuracy of the model after adding interacting features      : {} %'.format(lr_aft_int.score(X_New_test,y_New_test)*100))
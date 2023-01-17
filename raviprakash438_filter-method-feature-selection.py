import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import collections
import os
print(os.listdir("../input"))

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
#Load the train dataset. It contain more then 76000 records. Lets load 10000 records only to make things fast.
df=pd.read_csv('../input/santander-customer-satisfaction/train.csv',nrows=10000)
df.shape
df.info()
#Find the missing data in the columns
[col for col in df.columns if df[col].isnull().sum()>0]
#After executing the above code we found that there is no missing data.
# separate dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(df.drop(labels=['TARGET'], axis=1),df['TARGET'],test_size=0.3,random_state=0)
#Shape of training set and test set.
X_train.shape, X_test.shape

# I keep a copy of the dataset with all the variables
# to measure the performance of machine learning models
# at the end of the notebook
X_train_org=X_train.copy()
X_test_org=X_test.copy()
varModel=VarianceThreshold(threshold=0) #Setting variance threshold to 0 which means features that have same value in all samples.
varModel.fit(X_train)
constArr=varModel.get_support()
constArr
#get_support() return True and False value for each feature.
#True: Not a constant feature
#False: Constant feature(It contains same value in all samples.)
#To find total number of constant and non constant features we will be using collections.Counter function.
collections.Counter(constArr)
#Non Constant feature:284
#Constant feature: 86
#Print out constant feature name
constCol=[col for col in X_train.columns if col not in X_train.columns[constArr]]
constCol
print(X_train['ind_var2_0'].value_counts())
print(X_train['ind_var13_medio'].value_counts())
print(X_train['ind_var27'].value_counts())
print('Shape before drop-->',X_train.shape, X_test.shape)
#X_train=varModel.transform(X_train)
#X_test=varModel.transform(X_test)
X_train.drop(columns=constCol,axis=1,inplace=True)
X_test.drop(columns=constCol,axis=1,inplace=True)
print('Shape after drop-->',X_train.shape, X_test.shape)
#Create variance threshold model
quasiModel=VarianceThreshold(threshold=0.01) #It will search for the features having 99% of same value in all samples.
quasiModel.fit(X_train)
quasiArr=quasiModel.get_support()
quasiArr
#get_support() return True and False value for each feature.
#True: Not a quasi constant feature
#False: Quasi constant feature(It contains 99% same value in all samples.)
#To find total number of quasi constant and non quasi constant features we will be using collections.Counter function.
collections.Counter(quasiArr)
#Non quasi Constant feature:241
#Quasi constant feature: 43
#Print out quasi constant feature name
quasiCols=[col for col in X_train.columns if col not in X_train.columns[quasiArr]]
quasiCols
totalSampleCount=len(X_train)
print(X_train['num_aport_var33_ult1'].value_counts()/totalSampleCount)
print(X_train['num_var29'].value_counts()/totalSampleCount)
print(X_train['num_venta_var44_ult1'].value_counts()/totalSampleCount)
print('Shape before drop-->',X_train.shape, X_test.shape)
X_train.drop(columns=quasiCols,axis=1,inplace=True)
X_test.drop(columns=quasiCols,axis=1,inplace=True)
print('Shape after drop-->',X_train.shape, X_test.shape)
#The method will find the duplicate columns and return name of duplicated columns in an array
def duplicateColumns(data):
    dupliCols=[]
    for i in range(0,len(data.columns)):
        col1=data.columns[i]
        for col2 in data.columns[i+1:]:
            if data[col1].equals(data[col2]):
                dupliCols.append(col1+','+col2)
    return dupliCols
duplCols=duplicateColumns(X_train)
duplCols
print('Total Duplicated columns',len(duplCols))
#Lets verify the columns are Identical or not.
X_train[['ind_var1_0','ind_var40_0']]
#Get the duplicate column names
dCols=[col.split(',')[1] for col in duplCols]
dCols
#Find the count of unique columns
len(set(dCols))
print('Shape of our data before applying filter technique-->',df.shape)
print('Shape before droping duplicate columns-->',X_train.shape, X_test.shape)
X_train=X_train.drop(columns=dCols,axis=1)
X_test=X_test.drop(columns=dCols,axis=1)
print('Shape after droping duplicate columns-->',X_train.shape, X_test.shape)

# I keep a copy of the dataset except constant and duplicated variables
# to measure the performance of machine learning models
# at the end of the notebook
X_train_fil=X_train.copy()
X_test_fil=X_test.copy()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
houseDf=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
houseDf.head()
houseDf.info()
#Currently I will be dealling with numerical columns only.
colType = ['int64','float64']
#Select the columns which are either int64 or float64.
numCols=list(houseDf.select_dtypes(include=colType).columns)
#Assigning numerical columns from df to data variable. We can use the same variable as well.
data=houseDf[numCols]
data.shape
#Check if there is any missing data.
data.isnull().sum().max()
#Filling missing data
data.fillna(0,axis=1,inplace=True)
#Re-check if there is any missing data.
data.isnull().sum().max()
#Split our data in training and test set.
x_train,x_test,y_train,y_test=train_test_split(data.drop('SalePrice',axis=1),data['SalePrice'],test_size=.2,random_state=1)
# visualise correlated features
# I will build the correlation matrix, which examines the 
# correlation of all features (for all possible feature combinations)
# and then visualise the correlation matrix using seaborn
plt.figure(figsize=(20,15))
sns.heatmap(x_train.corr())
plt.show()
def correlation(dataset,threshold):
    col_corr=set() # set will contains unique values.
    corr_matrix=dataset.corr() #finding the correlation between columns.
    for i in range(len(corr_matrix.columns)): #number of columns
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold: #checking the correlation between columns.
                colName=corr_matrix.columns[i] #getting the column name
                col_corr.add(colName) #adding the correlated column name heigher than threshold value.
    return col_corr #returning set of column names
col=correlation(x_train,0.75)
print('Correlated columns:',col)
#X_train is train dataset for Santander database.
scol=correlation(X_train,0.8)
print('Correlated columns:',scol)
print(len(scol))
print('Shape of our data before applying filter technique-->',df.shape)
print('Shape before droping duplicate columns-->',X_train.shape, X_test.shape)
X_train=X_train.drop(columns=scol,axis=1)
X_test=X_test.drop(columns=scol,axis=1)
print('Shape after droping duplicate columns-->',X_train.shape, X_test.shape)
# create a function to build random forests and compare performance in train and test set
def RandomForest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=200, random_state=1, max_depth=4)
    rf.fit(X_train, y_train)
    print('Train set')
    pred = rf.predict_proba(X_train)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
    print('Test set')
    pred = rf.predict_proba(X_test)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
# original dataset result
RandomForest(X_train_org.drop(labels=['ID'], axis=1),
                  X_test_org.drop(labels=['ID'], axis=1),
                  Y_train, Y_test)
#Result after applying basic filter method on dataset.
RandomForest(X_train_fil.drop(labels=['ID'], axis=1),
                  X_test_fil.drop(labels=['ID'], axis=1),
                  Y_train, Y_test)
#Result after removing correlated features from filtered dataset.
RandomForest(X_train.drop(labels=['ID'], axis=1),
                  X_test.drop(labels=['ID'], axis=1),
                  Y_train, Y_test)

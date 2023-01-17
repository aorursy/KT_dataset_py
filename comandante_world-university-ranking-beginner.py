# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/cwurData.csv")
data.info() # Memory and data types used
data.head(10) #head function calls default  first 5 samples.
data.tail() #Last 5 samples.
data.shape  # Row-Column
data.describe() # Only Numeric Statistics
data.columns # Column names
data.isnull().sum() # Missing values control 
data.corr() # Indicates the presence, direction and intensity of the relationship between the two variables
#Correlation heatmap
f,ax= plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot=True,linewidths=.6,fmt='.2f',ax=ax) #annot=True :Visible numbers 
plt.show() 
data.cov() #Covaryans 
#Line plot
plt.plot(data.score,data.world_rank,color='blue',label='world_rank',alpha=0.7)
plt.plot(data.score,data.patents,color='red',label='patents',alpha=0.7)
plt.plot(data.score,data.quality_of_faculty,color='green',label='quality_of_faculty',alpha=0.7)
plt.legend()
plt.xlabel('Score')
plt.ylabel('y axis')
plt.title('Line plot')
plt.show()
#Scatter plot
data.plot(kind='scatter',x='world_rank',y='patents',alpha=0.5,color='blue')
plt.xlabel('world_rank')
plt.ylabel('patents')
plt.title('world_rank - patents Scatter Plot')
plt.show()
#Histogram 
data.score.plot(kind='hist',bins=40,figsize=(15,15),color='green')
plt.show()
#Score frequency 
#Boxplot
plt.figure()
plt.boxplot(data.score, 0, 'gD')
plt.show()
#Score frequency and the change of outliers according to years
df = data[data.year.isin(data.year.value_counts().head(10).index)]

sns.boxplot(
    x='year',
    y='score',
    data=df
)
plt.show()
P = np.percentile(data.score, [10, 100])
P
data[(data['year']==2012)& (data['score']>=85) & (data['country']=='USA')].sort_values('score',axis=0,ascending=False) #Filtering and ordering by score

data['country'].unique() #Unique method doesn't show the same country again.
filter_1=data['year']==2012
filter_2=data['score']<45
filter_3=data['country']=='United Kingdom'
data[filter_1 & filter_2 & filter_3]
#List of universities in the United Kingdom that were below 45 scores in 2012
#Let's add a new feature
score_average=sum(data.score) /len(data.score)
median=data.score.median()
data['score_level']=['high' if i>score_average else 'Average'if median<i<=score_average else 'Low' for i in data.score]
data1=data.head()
data2=data.tail()
conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True) # axis=0 adds dataframes in row
#Let's see the data again.
conc_data_row
#For example lets look frequency of country types
print(data['country'].value_counts(dropna=False))
data.describe() #ignore null entries and not number features

data_new=data.head(10)
data_new
#melt function
#frame : DataFrame
#id_vars : tuple, list, or ndarray, optional
#Column(s) to use as identifier variables.
#value_vars : tuple, list, or ndarray, optional
#Column(s) to unpivot. If not specified, uses all columns that are not set as id_vars.
melted=pd.melt(frame=data_new,id_vars='institution',value_vars=['score','patents'])
melted
#Index is name
#Column to use to make new frame’s columns.
#values:Column(s) to use for populating new frame’s values. 
#If not specified, all remaining columns will be used and the result will have hierarchically indexed columns.
#Return reshaped DataFrame organized by given index / column values.
melted.pivot(index='institution',columns='variable',values='value')
#Lets make a sample
data1=data.head()
data2=data.tail()
conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True) # axis=0 adds dataframes in row
conc_data_row
data1=data['institution'].head(7)
data2=data['world_rank'].head(7)
conc_data_col=pd.concat([data1,data2],axis=1) # axis=1 : adds dataframes in column
conc_data_col
#Let's see the distribution of countries by groupby method.
data.groupby("country").size()
data.dtypes
#Convert object(str) to categorical and int to float
data['country']=data['country'].astype('category')
data['quality_of_faculty']=data['quality_of_faculty'].astype('float')

data.dtypes
#country converted from object to categorical
#quality_of_faculty converted from int to float
#Let's look at  does World University Ranking data have non value
data.isnull().sum() # Missing values control
#Copy the data, opening a new memory
data1=data.copy()
#Convert categorical(str) to object again.
data1['country']=data1['country'].astype('object')
#Let's drop the broad_impact feature
data1_target = data1.broad_impact
data1_predictors = data1.drop(['broad_impact'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
data1_numeric_predictors = data1_predictors.select_dtypes(exclude=['object'])
data1_numeric_predictors #Only numeric features and non null
from sklearn.model_selection import KFold
#Sample dataset
ds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

#K-Fold Cross Validation
kfold = KFold(3, True, 1)

#Return on the folds
for egitim, test in kfold.split(ds):
    print('egitim: %s, test: %s' % (ds[egitim], ds[test]))
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection

#We select attribute values for training
X = data1_numeric_predictors.iloc[:, :-1].values

#We select attribute values for classification
#I selected score_level feature
Y = data.iloc[: ,-1].values
Y
#Separation of training and test data sets
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
#Calculation of ACC with K-fold cross validation of NB model
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
cv_results
msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
msg
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(Y_pred,Y_test))

from sklearn.linear_model import LinearRegression 
x=data1_numeric_predictors.iloc[:,[0,1,2,3,4,5,6,7,8,10]].values #all features except score
y=data1_numeric_predictors.score.values.reshape(-1,1) #Score
#fitting data
multiple_linear_regression=LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0 :",multiple_linear_regression.intercept_) #Intercept
print("b1,b2,b3,b4,b5,b6,b7,b8,b9,b10 :",multiple_linear_regression.coef_) #coefficients
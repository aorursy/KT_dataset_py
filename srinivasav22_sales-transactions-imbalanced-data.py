import pandas as pd

import numpy as np

from pandas import ExcelWriter

from pandas import ExcelFile

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn import ensemble

from scipy.stats import spearmanr

import warnings

warnings.filterwarnings("ignore")

import copy

from sklearn.preprocessing import StandardScaler, LabelEncoder

import scipy.stats as stats
df_train = pd.read_excel('/kaggle/input/sales-transactions-dataset/Train.xlsx')

df_test = pd.read_excel('/kaggle/input/sales-transactions-dataset/Train.xlsx')

print("=="*45)

print("The Training dataset has {0} rows and {1} columns".format(df_train.shape[0], df_train.shape[1]))

print(" The Testing dataset has {0} rows and {1} columns".format(df_test.shape[0], df_test.shape[1]))

print("=="*45)
# Check the Head of the Data Set

df_train.head()
# Check the Head of the Data Set

df_test.head()
# Check the Tail of the Data Set

df_train.tail()
df_train.dtypes,df_test.dtypes
# Check the Unique Values in the Data Set

print('Unique Records in Train Data: {}'.format(df_train.nunique()))

print('Unique Records in Test Data : {}'.format(df_test.nunique()))
df_test.dtypes
# Check the Data Types in the Data Set

print('Stas:{}'.format(df_train.describe()))



print('Stas:{}'.format(df_test.describe()))

# Check the Null Values in the Data sets

print('Null Values in Train Data: {}'.format(df_train.isnull().sum()))

print('Null Values in Test Data : {}'.format(df_test.isnull().sum()))

# Sum of Quantity and Total Sales Values by Grouping Suspicious



df_train.groupby('Suspicious').agg({'Quantity':'sum','TotalSalesValue':'sum'})

# 'Quantity' seems left skewed hence imputing it with median

df_train['Quantity'].fillna(df_train.TotalSalesValue.median(),inplace=True)
# 'Total  Sales Value' seems left skewed hence imputing it with median

df_train['TotalSalesValue'].fillna(df_train.TotalSalesValue.median(),inplace=True)
# Making NAN Values Where 



#df_train['TotalSalesValue']= df_train['TotalSalesValue'].where(df_train['TotalSalesValue']>250000)
#df_train['Quantity']= df_train['Quantity'].where(df_train['Quantity']>50000)
# Check the Null Values in the Data sets

print('Null Values in Train Data: {}'.format(df_train.isnull().sum()))

print('Null Values in Test Data : {}'.format(df_test.isnull().sum()))

# Since data in 'LoanAmount' column seems right skewed hence imputing it with median seems good option.

df_train['Quantity'].fillna(df_train.Quantity.median(),inplace=True)

df_train['TotalSalesValue'].fillna(df_train.TotalSalesValue.median(),inplace=True)
# Check the Null Values in the Data sets

print('Null Values in Train Data: {}'.format(df_train.isnull().sum()))

print('Null Values in Test Data : {}'.format(df_test.isnull().sum()))

df_train.insert(4,"Price",df_train['TotalSalesValue']/df_train['Quantity'])

df_train.head()
df_test.insert(4,"Price",df_test['TotalSalesValue']/df_test['Quantity'])

df_test.head()
# Converted Target Variable "Suspicious" to Numeric



df_train = df_train.replace(to_replace ="Yes",value =1) 

df_train = df_train.replace(to_replace ="No",value =2) 

df_train = df_train.replace(to_replace ="indeterminate",value =3)



df_train.head()
# To Check the Suspicious in Interger Type

df_train.dtypes
# in the Above we can see that the Suspicious Class in Object Type so we need convert that in to integer

#df_train['Suspicious'] = df_train.Suspicious.astype(int)
# '''To Remove white Space Using strip leading and trailing space'''

df_train['ProductID'] = df_train['ProductID'].str.strip()

print (df_train.head(10))
## label_encoder object knows how to understand word labels.



from sklearn.preprocessing import LabelEncoder



label_encoder = preprocessing.LabelEncoder() 

  

df_train['ProductID']= label_encoder.fit_transform(df_train['ProductID'])

df_train['SalesPersonID']= label_encoder.fit_transform(df_train['SalesPersonID'])

df_test['ProductID']= label_encoder.fit_transform(df_test['ProductID'])

df_test['SalesPersonID']= label_encoder.fit_transform(df_test['SalesPersonID'])

#df_train['ProductID'].unique()

 
suspicious = df_train['Suspicious']



mean=suspicious.mean()

median=suspicious.median()

mode=suspicious.mode()



print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])

plt.figure(figsize=(10,5))

plt.hist(suspicious,bins=100,color='grey')

plt.axvline(mean,color='red',label='Mean')

plt.axvline(median,color='yellow',label='Median')

plt.axvline(mode[0],color='green',label='Mode')

plt.xlabel('suspicious')

plt.ylabel('Frequency')

plt.legend()

plt.show()
quantity = df_train['Quantity']



mean=quantity.mean()

median=quantity.median()

mode=quantity.mode()



print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])

plt.figure(figsize=(10,5))

plt.hist(quantity,bins=100,color='grey')

plt.axvline(mean,color='red',label='Mean')

plt.axvline(median,color='yellow',label='Median')

plt.axvline(mode[0],color='green',label='Mode')

plt.xlabel('quantity')

plt.ylabel('Frequency')

plt.legend()

plt.show()
sales = df_train['TotalSalesValue']



mean=sales.mean()

median=sales.median()

mode=sales.mode()



print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])

plt.figure(figsize=(10,5))

plt.hist(sales,bins=100,color='grey')

plt.axvline(mean,color='red',label='Mean')

plt.axvline(median,color='yellow',label='Median')

plt.axvline(mode[0],color='green',label='Mode')

plt.xlabel('sales')

plt.ylabel('Frequency')

plt.legend()

plt.show()
df_train['Quantity'].hist(bins=1000)
df_train['TotalSalesValue'].hist(bins=1000)
df_train.head(100)
# Log Transformation of Skewed Data



df_train['Quantity']= np.log10(df_train['Quantity'])

df_train['TotalSalesValue']= np.log10(df_train['TotalSalesValue'])



#print(df_train(10))

sales = df_train['TotalSalesValue']



mean=sales.mean()

median=sales.median()

mode=sales.mode()



print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])

plt.figure(figsize=(10,5))

plt.hist(sales,bins=100,color='grey')

plt.axvline(mean,color='red',label='Mean')

plt.axvline(median,color='yellow',label='Median')

plt.axvline(mode[0],color='green',label='Mode')

plt.xlabel('sales')

plt.ylabel('Frequency')

plt.legend()

plt.show()
quantity = df_train['Quantity']



mean=quantity.mean()

median=quantity.median()

mode=quantity.mode()



print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])

plt.figure(figsize=(10,5))

plt.hist(quantity,bins=100,color='grey')

plt.axvline(mean,color='red',label='Mean')

plt.axvline(median,color='yellow',label='Median')

plt.axvline(mode[0],color='green',label='Mode')

plt.xlabel('quantity')

plt.ylabel('Frequency')

plt.legend()

plt.show()
# Delete the Unique Records which matches to rows 



df_train = df_train.drop(['ReportID'], axis=1)



df_train.dtypes
# skewness along the index axis 

df_train.skew(axis = 0, skipna = True) 
# skewness along the index axis 

df_train.skew(axis = 0, skipna = True) 
# Separate Target "Suspicious" as y and others as x to split the data set



X = df_train.loc[:, df_train.columns != 'Suspicious']

y = df_train.loc[:, df_train.columns == 'Suspicious']
seed = 2

test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report



dtc = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=100,

            max_features=None, max_leaf_nodes=50, min_samples_leaf=10,

            min_samples_split=4, min_weight_fraction_leaf=0.0,

            presort=False, random_state=42, splitter='random')
dtc.fit(X_train,y_train)



y_dtc = dtc.predict(X_test)
print(classification_report(y_test,y_dtc))
#Create a Random Forest Classifier

RFC=RandomForestClassifier(n_estimators=1000)



#Train the model using the training sets y_pred=clf.predict(X_test)

RFC=RFC.fit(X_train, y_train)



RFC_Pred=RFC.predict(X_test)
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score



Accuracy_Score = accuracy_score(y_test, RFC_Pred)

Precision_Score = precision_score(y_test, RFC_Pred,  average="macro")

Recall_Score = recall_score(y_test, RFC_Pred,  average="macro")

F1_Score = f1_score(y_test, RFC_Pred,  average="macro")



print('Average Accuracy: %0.2f +/- (%0.1f) %%' % (Accuracy_Score.mean()*100, Accuracy_Score.std()*100))

print('Average Precision: %0.2f +/- (%0.1f) %%' % (Precision_Score.mean()*100, Precision_Score.std()*100))

print('Average Recall: %0.2f +/- (%0.1f) %%' % (Recall_Score.mean()*100, Recall_Score.std()*100))

print('Average F1-Score: %0.2f +/- (%0.1f) %%' % (F1_Score.mean()*100, F1_Score.std()*100))



CM = confusion_matrix(y_test, RFC_Pred)
print(classification_report(y_test, RFC_Pred))
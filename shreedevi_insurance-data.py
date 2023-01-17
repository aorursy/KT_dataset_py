# IMPORT 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
#! head -n 3 ../input/exercise_02_train.csv
insu = pd.read_csv('../input/exercise_02_train.csv')
insu.head(2).T
insu.y.value_counts().plot(kind='bar')
insu.info(verbose=True, max_cols=101)
insu.head(1).T
# CATEGORICAL COLUMNS
insu.loc[:,insu.dtypes==object].head()
# Remove the $ symbol
insu['x41'] = insu['x41'].str.replace('$','').astype(float)
#Remove the % symbol
insu['x45'] = insu['x45'].str.replace('%','').astype(float)
insu.loc[:,insu.dtypes==object].head()
insu['x34'].value_counts().plot(kind='barh')
# Make all brand names lowercase
insu['x34'] = insu['x34'].str.lower()
insu['x35'].value_counts().plot(kind='barh')
s1 = insu['x35']
# Standardize the day names
insu['x35'] = s1.replace({'monday':'mon', 'tuesday':'tue', 'wednesday':'wed',
        'thurday':'thu', 'thur':'thu','friday':'fri'})
insu['x35'].value_counts().plot(kind='barh')
insu.loc[:,insu.dtypes==object].head()
insu['x68'].value_counts().sort_values().plot(kind='barh')
insu['x68'] = insu['x68'].str.lower()
#Standardize the month names
insu['x68'] = insu['x68'].replace({'january':'jan', 'dev':'dec', 'sept.':'sep',
        'july':'jul'})
insu['x93'].value_counts().sort_values().plot(kind='barh')
insu.loc[:,insu.dtypes==object].head()
# Look at missing rows
insu[insu.isnull().any(axis=1)].shape
#Drop rows with missing data
insu.dropna(how='any', inplace=True)
# Look at missing rows AGAIN
insu[insu.isnull().any(axis=1)].shape
insu.x0.plot(kind='hist')

cols = insu.columns
insu.boxplot(column=['x0', 'x1', 'x2'])
target = insu.y
insu.drop('y', axis=1, inplace=True)
insu2 = pd.get_dummies(insu, columns=['x34', 'x35', 'x68', 'x93'])
insu2.head()
X = insu2.values
y = target.values
corr = np.corrcoef(X.T,y)
sns.heatmap(data = corr,vmin=0, vmax=1)
# Split the data for train and dev/test purpose 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=0.20, 
                                                    random_state=10)
#Normalizer or Standardized
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
Xn_train = scaler.transform(X_train)
Xn_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegressionCV
lrm = LogisticRegressionCV(tol=0.0001)
lrm.fit(Xn_train, Y_train)
print("Test Accuracy: ", 100*lrm.score(Xn_test, Y_test))
from sklearn.metrics import classification_report
print ("CLASSIFICATION REPORT:\n")
print (classification_report(Y_test, lrm.predict(Xn_test)))

from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(base_estimator=LogisticRegressionCV())
abc.fit(Xn_train, Y_train)
print("Test Accuracy: \n", 100*abc.score(Xn_test, Y_test))
print( "CLASSIFICATION REPORT:\n")
print (classification_report(Y_test,abc.predict(Xn_test)))


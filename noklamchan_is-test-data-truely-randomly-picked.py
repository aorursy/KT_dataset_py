import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
%matplotlib inline
import matplotlib.pyplot as plt
data = Path('../input')
application_train = pd.read_csv(data/'application_train.csv')
application_test = pd.read_csv(data/'application_test.csv')

tables = [application_train, application_test]
def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns",1000):       
            display(df)

for i in tables:
    display_all(i.head())
cols = [i for i in application_train.columns.values if 'AMT' in i]
cols
train = application_train[cols]
test = application_test[cols]
from sklearn.ensemble import  RandomForestClassifier
train['is_test'] = 0
test['is_test'] = 1
n = len(test)
train = train.iloc[np.random.permutation(len(train))[:n]]
train.shape,test.shape
df = pd.concat([train,test])
df = df.iloc[np.random.permutation(len(df))]
df.fillna(0,inplace=True) # Fill na with 0 as RandomForest do not accept NaN
df.head()
df.is_test.mean() # Now the dataset is balanced
y = df['is_test'].copy()
df.drop('is_test',1,inplace=True)
def split_vals(a,n): return a[:n].copy(), a[n:].copy()
n_trn = int(len(df)*0.75) # reserve 25% of data as validation
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape
m = RandomForestClassifier(n_estimators=100, max_depth=8)
m.fit(X_train,y_train)
print('Accuracy: ',m.score(X_train,y_train))
print('Accuracy: ',m.score(X_valid,y_valid))
for i in cols:
    plt.figure()
    plt.title(i)
    plt.hist([application_train[i],application_test[i]],alpha=0.3,log=True,density=True)
    plt.xticks( rotation='vertical')
    plt.legend(['train','test'])
    plt.show()
   

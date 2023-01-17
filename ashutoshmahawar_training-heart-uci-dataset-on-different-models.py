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
import matplotlib.pyplot as plt
import seaborn as sns

dataframe = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv', dtype={'sex':'object', 'cp':'object','fbs':'object', 'restecg':'object',
                                                                            'exang':'object','slope':'object', 'thal':'object'})
dataframe.head()
print('No. of rows: {0} and columns: {1}'.format(dataframe.shape[0], dataframe.shape[1]))
# Gives information about the data type of each feature and Non-Null count.
dataframe.info()
# Gives statistical description of each numerical feature.
dataframe.describe().T
dataframe.target.value_counts()
type(dataframe.loc[0, 'sex'])
column = dataframe.select_dtypes('object').columns

for col in column:
    print(col,":\n", dataframe[col].value_counts())
    print('-----------------------------------')
dataframe.isnull().sum(axis=0)
# to find out rows which are having thal==0
(dataframe.thal== str(0)).sum()
dataframe = dataframe.loc[dataframe.thal != str(0)]
dataframe.shape
dataframe.thal.value_counts()
df_num = dataframe.select_dtypes('int64', 'float64')
df_num.columns
df_num.hist(figsize=(10,10))
df_cat = dataframe.select_dtypes('object')
df_cat.columns
dataframe.columns
from sklearn.model_selection import train_test_split
y = dataframe['target']
dataframe.drop(['target'], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(dataframe, y, test_size=0.2, stratify=y, random_state=0)

# importing libraries required for scaling and encoding features. 
# we will fit on train data first and then transform test data.
from sklearn.preprocessing import StandardScaler, OneHotEncoder
scaler = StandardScaler()
encoder = OneHotEncoder(drop='first', sparse=False)
df_num = X_train.select_dtypes(['int64','float64'])
df_cat = X_train.select_dtypes('object')

#standardizing numerical train data
df_num = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num.columns)
df_num.columns
df_cat.columns
#encoding categorical data
data = encoder.fit_transform(df_cat)
col = encoder.get_feature_names(df_cat.columns)
df_cat = pd.DataFrame(data, columns=col)
df_cat.columns
X_train = pd.concat([df_num, df_cat], axis=1)
X_train.isna().sum(axis=0)
# transforming the features on test data
df_num = X_test.select_dtypes(['int64', 'float64'])
df_cat = X_test.select_dtypes('object')
df_num = pd.DataFrame(scaler.transform(df_num), columns=df_num.columns)
df_num.columns

data = encoder.transform(df_cat)
col = encoder.get_feature_names(df_cat.columns)
df_cat = pd.DataFrame(data, columns=col)
df_cat.columns
X_test = pd.concat([df_num, df_cat], axis=1)
X_test.isna().sum(axis=0)
from sklearn.linear_model import LogisticRegression
regress = LogisticRegression()
regress.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
pred_train = regress.predict(X_train)
pred_test = regress.predict(X_test)
print('Accuracy on Training data using Logistic Regression: ', accuracy_score(y_train, pred_train))
print('Accuracy on Test data using Logistic Regression', accuracy_score(y_test, pred_test))
plt.plot(X_train.T, '*')
plt.xticks(rotation='vertical')
plt.show()
X_binarised_train = X_train.apply(pd.cut, bins=2, labels=[0,1])
plt.plot(X_binarised_train.T, '*')
plt.xticks(rotation='vertical')
plt.show()
plt.plot(X_test.T, '*')
plt.xticks(rotation='vertical')
plt.show()
X_binarised_test = X_test.apply(pd.cut, bins=2, labels=[0,1])
plt.plot(X_binarised_test.T, '*')
plt.xticks(rotation='vertical')
plt.show()
X_binarised_train = X_binarised_train.values
X_binarised_test = X_binarised_test.values
class mp_neuron:
    
    def __init__(self):
        self.b = None
        
    def model(self, x):
        return (sum(x) >= self.b)
    
    def predict(self, X):
        Y =[]
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)
        
    def fit(self, X, Y):
        accuracy = {}
        
        for b in range(X.shape[1] + 1):
            self.b = b
            Y_pred = self.predict(X)
            accuracy[b] = accuracy_score(Y_pred, Y)
            print(b,":",accuracy[b])
        best_b = max(accuracy, key = accuracy.get)
        self.b = best_b
        
        print('Optimal value of b is', best_b)
        print('Highest accuracy is', accuracy[best_b])
mp_neuron = mp_neuron()
mp_neuron.fit(X_binarised_train, y_train)
class Perceptron:
    
    def __init__(self):
        self.w = None
        self.b = None
    
    def model(self, x):
        return 1 if (np.dot(self.w, x) >= self.b) else 0
    
    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)
    
    def fit(self, X, Y, epochs = 1, lr = 1):
        self.w = np.random.rand(X.shape[1])
        self.b = 0
        
        accuracy = {}
        max_accuracy = 0
        
        for i in range(epochs):
            for x, y in zip(X, Y):
                y_pred = self.model(x)
                if y == 1 and y_pred == 0:
                    self.w = self.w + lr * x
                    self.b = self.b + lr * 1
                elif y == 0  and y_pred == 1:
                    self.w = self.w - lr * x
                    self.b = self.b - lr * 1
            accuracy[i] = accuracy_score(self.predict(X), Y)
            if accuracy[i] > max_accuracy:
                max_accuracy = accuracy[i]
                chkptw = self.w
                chkptb = self.b
        self.w = chkptw
        self.b = chkptb
        print(max_accuracy)
X_train_perceptron = X_train.values
X_test_perceptron = X_test.values
perceptron = Perceptron()
perceptron.fit(X_train_perceptron, y_train, epochs=100000, lr=0.45)
y_pred = perceptron.predict(X_test_perceptron)
print("Accuracy on test data:", accuracy_score(y_test, y_pred))




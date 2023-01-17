# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
len(train)
len(test)
df = pd.concat([train,test], axis=0)
df.describe()
df.isnull().sum()
del df["MarkDown1"]
del df["MarkDown2"]
del df["MarkDown3"]
del df["MarkDown4"]
del df["MarkDown5"]
df.fillna(0, inplace=True)
df.dtypes
# Make sure we can later recognize what a dummy once belonged to
df['Type'] = 'Type_' + df['Type'].map(str)
df['Store'] = 'Store_' + df['Store'].map(str)
df['Dept'] = 'Dept_' + df['Dept'].map(str)
plt.hist(df['Fuel_Price'])
plt.hist(df['Size'])
plt.hist(df['Temperature'])
plt.hist(df['Unemployment'])
# CPI, Fuel_Price, Size, Temperature, Unemployment
df['CPI'] = (df['CPI'] - df['CPI'].mean())/(df['CPI'].std())
df['Fuel_Price'] = (df['Fuel_Price'] - df['Fuel_Price'].mean())/(df['Fuel_Price'].std())
df['Size'] = (df['Size'] - df['Size'].mean())/(df['Size'].std())
df['Unemployment'] = (df['Unemployment'] - df['Unemployment'].mean())/(df['Unemployment'].std())
plt.hist(df["Size"])
plt.hist(df["Unemployment"])
plt.hist(df["CPI"])
plt.hist(df["Fuel_Price"])
# Create dummies
type_dummies = pd.get_dummies(df['Type'])
store_dummies = pd.get_dummies(df['Store'])
dept_dummies = pd.get_dummies(df['Dept'])
# Add dummies
df = pd.concat([df,type_dummies,store_dummies,dept_dummies],axis=1)
# Remove originals
del df['Type']
del df['Store']
del df['Dept']
del df['Date']
df.dtypes
train = df.iloc[:282451]
test = df.iloc[282451:]
test = test.drop('Weekly_Sales',axis=1) # We should remove the nonsense values from test
y = train['Weekly_Sales'].values
X = train.drop('Weekly_Sales',axis=1).values
X.shape
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import optimizers
model = Sequential()
model.add(Dense(65, activation='relu', input_dim=135))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
adam=optimizers.Adam(lr=0.15, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss='mae')
Sequential_feature_scaled = model.fit(X,y,batch_size=2048,epochs=10)
import seaborn as sns
five_thirty_eight = [
    "#30a2da",
    "#fc4f30",
    "#e5ae38",
    "#6d904f",
    "#8b8b8b"
]
sns.set(palette=five_thirty_eight)
plt.plot(Sequential_feature_scaled.history['loss'], label="model")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
X_test = test.values
y_pred = model.predict(X_test, batch_size=2048)
testfile = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({'id':testfile['Store'].map(str) + '_' + testfile['Dept'].map(str) + '_' + testfile['Date'].map(str),
                          'Weekly_Sales':y_pred.flatten()})
submission.head()
submission.to_csv('submission_2.csv',index=False)

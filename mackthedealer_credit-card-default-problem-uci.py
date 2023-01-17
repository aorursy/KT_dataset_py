# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
dirname = 'C:/Users/mayan/Documents/Credit Card Dataset/'
filename = 'UCI_Credit_Card.csv'

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../input/UCI_Credit_Card.csv')
df.head()
# Let us first check for null values
df.isnull().sum()
# Now let us study the dataset a little bit
df.info()
# Let us now describe the dataset
df.describe()
# Limit Balance histogram
df['LIMIT_BAL'].hist()
# Boxplot of Income 
sns.boxplot(x='LIMIT_BAL', data=df)
# Countplot of gender
sns.countplot(x='SEX', data=df)
# Marital Status-wise countplot
sns.countplot(x='MARRIAGE', data=df)
df['MARRIAGE'].value_counts()
# Since our dataset legend does not have any values in the legend for marriage in 0 category. So let us club values with 0 into others i.e 3.
df.loc[df['MARRIAGE']==0,'MARRIAGE']=3
# lET US CHECK IF OUR TRANSFORMATION WORKED
df['MARRIAGE'].value_counts()
# Let us see the distribution of Age
df['AGE'].hist()
# Boxplot of Age column
sns.boxplot(y='AGE', data=df)
# Let us see the age distribution with respect to sex
sns.boxplot(x='SEX',y='AGE', data=df)
# Let us now check the payment statuses of each month
sns.countplot(x='PAY_0', data=df)
# Since our legend has no description for -2 and 0 in this. Let us now remove them and replace by -1 assuming that those customer paid their dues duly.
df.loc[df['PAY_0']==-2,"PAY_0"]=-1
df.loc[df['PAY_0']==0,"PAY_0"]=-1
# Let us now check the dataset for the values
sns.countplot(x='PAY_0', data=df)
# Let us move to the next month
sns.countplot(x='PAY_2', data=df)
# Since our legend has no description for -2 and 0 in this. Let us now remove them and replace by -1 assuming that those customer paid their dues duly.
df.loc[df['PAY_2']==-2,"PAY_2"]=-1
df.loc[df['PAY_2']==0,"PAY_2"]=-1
sns.countplot(x='PAY_2', data=df)
# Now let us do the same for the other such PAY_ columns
sns.countplot(x='PAY_3', data=df)
# Since our legend has no description for -2 and 0 in this. Let us now remove them and replace by -1 assuming that those customer paid their dues duly.
df.loc[df['PAY_3']==-2,"PAY_3"]=-1
df.loc[df['PAY_3']==0,"PAY_3"]=-1
sns.countplot(x='PAY_3', data=df)
sns.countplot(x='PAY_4', data=df)
# Since our legend has no description for -2 and 0 in this. Let us now remove them and replace by -1 assuming that those customer paid their dues duly.
df.loc[df['PAY_4']==-2,"PAY_4"]=-1
df.loc[df['PAY_4']==0,"PAY_4"]=-1
sns.countplot(x='PAY_4', data=df)
sns.countplot(x='PAY_5', data=df)
# Since our legend has no description for -2 and 0 in this. Let us now remove them and replace by -1 assuming that those customer paid their dues duly.
df.loc[df['PAY_5']==-2,"PAY_5"]=-1
df.loc[df['PAY_5']==0,"PAY_5"]=-1
sns.countplot(x='PAY_5', data=df)
sns.countplot(x='PAY_6', data=df)
# Since our legend has no description for -2 and 0 in this. Let us now remove them and replace by -1 assuming that those customer paid their dues duly.
df.loc[df['PAY_6']==-2,"PAY_6"]=-1
df.loc[df['PAY_6']==0,"PAY_6"]=-1
sns.countplot(x='PAY_6', data=df)
# Let us do a bit of EDA With deeper insights to be derived
default = df[df['default.payment.next.month']==1]
# histogram of age of defaulters
default['AGE'].hist()
# Countplot of Age by defaulters
sns.countplot(x='SEX', data=default)
# Let us now see the defaulters as per education qualifications
sns.countplot(x='EDUCATION', data=default)
# Boxplot of different categories in terms of education
sns.boxplot(x='EDUCATION', y='LIMIT_BAL', data=default)
# Let us see the number of defaulters with respect to education and their respective sex.
sns.countplot(x='EDUCATION', hue='SEX', data=default)
# Let us see the whether more defaulters are married or single
sns.countplot(x='MARRIAGE', data=default)
from scipy import stats
z = np.abs(stats.zscore(df))
print(z)
threshold = 3
print(np.where(z>3))
dfout = df[(z<3).all(axis=1)]
dfout
dfout.to_pickle('dfpreprocessed.pkl')
# Now let us check if the dataset is balanced or not
sns.countplot(x='default.payment.next.month', data=df)
# Since the dataset is heavily imbalanced we will have to balance it first in order to train a model on the dataset
from imblearn.combine import SMOTETomek
X = dfout.drop('default.payment.next.month', axis=1, inplace=False)
y = dfout['default.payment.next.month']
sampler = SMOTETomek(1)
Xout, yout = sampler.fit_resample(X,y)
yout.value_counts()
# Now we have a balanced dataset with 19736 points each side. Let us now shuffle the dataset and create training, validation and testing sets.
Xout
# Now let us create dummies on the relevant columns i.e. SEX, MARRIAGE, EDUCATION.
sex = pd.get_dummies(Xout['SEX'], prefix='SEX', drop_first=True)
edu = pd.get_dummies(Xout['EDUCATION'], prefix='EDUCATION', drop_first=True)
marriage = pd.get_dummies(Xout['MARRIAGE'], prefix='MARRIAGE', drop_first=True)
Xout = pd.concat([Xout, sex, edu, marriage], axis=1)
Xout.drop(['SEX', 'EDUCATION', 'MARRIAGE'], axis=1, inplace=True)

Xout.columns
ordercol = ['ID', 'LIMIT_BAL','SEX_2','EDUCATION_1',
       'EDUCATION_2', 'EDUCATION_3', 'EDUCATION_4','MARRIAGE_2',
       'MARRIAGE_3', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5',
       'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
       'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
       'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'   ]
Xout = Xout[ordercol]
# Now let us scale the given data before proceeding to ML Training
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xscaled = scaler.fit_transform(Xout)
Xscaled= pd.DataFrame(Xscaled, columns=Xout.columns)
Xscaled
# Importing train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xscaled, yout, train_size=0.9, random_state=42)
X_train2, X_valid, y_train2, y_valid = train_test_split(X_train, y_train, train_size=(8/9), random_state=42)
# Since we don't need ID Column in our classifier. Let us drop it from all our X columns.
X_train2.drop('ID', axis=1, inplace=True)
X_valid.drop('ID', axis=1, inplace=True)
X_test.drop('ID', axis=1, inplace=True)
# Logistic Regression
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(X_train2, y_train2)
model1 = reg.fit(X_train2, y_train2)
# Now let us evaluate the model
reg.score(X_train2, y_train2)
# It has a training accuracy of 76.14 %. Let us check the validation and testing accuracy
reg.score(X_valid, y_valid)
reg.score(X_test, y_test)
# Now let us try K-Nearest Neighbours Classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(5)
model2 = knn.fit(X_train2, y_train2)
model2.score(X_train2, y_train2)
model2.score(X_valid, y_valid)
model2.score(X_test, y_test)
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(100)
rfc.fit(X_train2, y_train2)
rfc.score(X_train2, y_train2)
rfc.score(X_valid, y_valid)
rfc.score(X_test, y_test)
# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train2, y_train2)
nb.score(X_train2, y_train2)
from xgboost import XGBClassifier, XGBRFClassifier
xgb = XGBClassifier()
xgb.fit(X_train2, y_train2)
xgb.score(X_train2, y_train2)
xgb.score(X_valid, y_valid)
xgb.score(X_test, y_test)
xgbrfc = XGBRFClassifier()
xgbrfc.fit(X_train2, y_train2)
xgbrfc.score(X_train2, y_train2)
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier()
gbm.fit(X_train2, y_train2)
gbm.score(X_train2, y_train2)
# Now let us try out neural networks
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
model = Sequential()
input_size = 27

model.add(Dense(input_size, activation='relu'))
model.add(Dense(27, activation='relu'))

model.add(Dense(14, activation='relu'))
model.add(Dense(7, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
batch_size = 10
max_epochs = 1000
early =EarlyStopping(patience=50)
model.fit(X_train2, y_train2, batch_size=batch_size, epochs=max_epochs, callbacks=[early], validation_data=(X_valid, y_valid), verbose=2)
predmodel = model.predict_classes(X_train2)
from sklearn.metrics import classification_report
print(classification_report(y_train2, predmodel))
predvalid = model.predict_classes(X_valid)
print(classification_report(y_valid, predvalid))
predtest = model.predict_classes(X_test)
print(classification_report(y_test, predtest))
predtrainxgb = xgb.predict(X_train2)
predvalxgb = xgb.predict(X_valid)
predtestxgb = xgb.predict(X_test)

print(classification_report(y_test, predtestxgb))
print(classification_report(y_valid, predvalxgb))
print(classification_report(y_train2, predtrainxgb))
# checking rfc
predtrainrfc = rfc.predict(X_train2)
predvalrfc = rfc.predict(X_valid)
predtestrfc = rfc.predict(X_test)
print(classification_report(y_train2, predtrainrfc))
# 100 % train accuracy seems rather wrong. but let's check validation and test
print(classification_report(y_valid, predvalrfc))
print(classification_report(y_test, predtestrfc))
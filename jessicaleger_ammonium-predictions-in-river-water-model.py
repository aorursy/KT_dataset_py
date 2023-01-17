import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
ammonium_train=pd.read_csv('/kaggle/input/ammonium-prediction-in-river-water/train.csv')

ammonium_train.head()
ammonium_train.info()
ammonium_train.describe()
ammonium_train.drop(ammonium_train[['3','4','5','6','7']], axis=1, inplace=True)

ammonium_train.head()
ammonium_train.count()
ammonium_train.dropna(inplace=True)

ammonium_train.count()
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(10,5))

plt.plot(ammonium_train.index, ammonium_train['2'], color='blue', label='station 2')

plt.plot(ammonium_train.index, ammonium_train['1'], color='magenta', label='station 1')

plt.plot(ammonium_train.index, ammonium_train['target'], color='orange', label='target station')

plt.legend()



plt.ylabel('Ammonium \n concentration (mg/cub. dm)')

plt.title('Ammonium Concentrations Over Time')
plt.figure(figsize=(13,5))

ax1=plt.scatter(ammonium_train['1'], ammonium_train['target'], label='station 1', color='magenta')

plt.scatter(ammonium_train['2'], ammonium_train['target'], label='station 2', color='blue')

plt.legend()

plt.xlabel('Downstream ammonium \n concentration (mg/cub. dm)')

plt.ylabel('Target ammonium \n concentration (mg/cub. dm)')

plt.title('Relationship Between Downstream and \n Target Station Ammonium Concentrations')
from scipy import stats
fig = plt.figure(figsize = (10, 7))

sns.residplot(ammonium_train['1'], ammonium_train['target'], color='magenta', label='station 1')

sns.residplot(ammonium_train['2'], ammonium_train['target'], color='blue', label='station 2')



plt.title('Residual plot', size=24)

plt.xlabel('upstream stations ammonium', size=18)

plt.ylabel('target station ammonium', size=18)

plt.legend()
stats.shapiro(ammonium_train['target'])
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson') 
values=ammonium_train[['target','1','2']]

pt.fit(values)

yeoj_transform=pt.transform(values)

ammonium_train_transformed=pd.DataFrame(data=yeoj_transform, columns=['target','1','2'])

ammonium_train_transformed.head()
ax1=plt.scatter(ammonium_train_transformed['1'], ammonium_train_transformed['target'], label='station 1')

plt.scatter(ammonium_train_transformed['2'], ammonium_train_transformed['target'], label='station 2')

plt.legend()

plt.xlabel('Downstream ammonium \n concentration (mg/cub. dm)')

plt.ylabel('Target ammonium \n concentration (mg/cub. dm)')

plt.title('Relationship Between Downstream and \n Target Station Ammonium Concentrations')
fig = plt.figure(figsize = (10, 7))

sns.residplot(ammonium_train_transformed['1'], ammonium_train_transformed['target'], color='magenta', label='station 1')

sns.residplot(ammonium_train_transformed['2'], ammonium_train_transformed['target'], color='blue', label='station 2')



# title and labels

plt.title('Residual plot', size=24)

plt.xlabel('upstream stations ammonium', size=18)

plt.ylabel('target station ammonium', size=18)

plt.legend()
stats.shapiro(ammonium_train_transformed['target'])
ammonium_train_transformed[['target','1','2']].corr()
X=ammonium_train_transformed[['1']].values

y=ammonium_train_transformed['target'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn import linear_model
linreg = linear_model.LinearRegression()
linreg.fit(X_train,y_train)
from sklearn.metrics import r2_score, mean_squared_error as mse
y_pred_train=linreg.predict(X_train)
print("The Mean Squared Error on the Train set is:\t{:0.5f}".format(mse(y_train, y_pred_train)))

print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_pred_train)))
plt.plot(X_train,y_pred_train)

plt.scatter(ammonium_train['1'], ammonium_train['target'], label='station 1')

plt.scatter(ammonium_train['2'], ammonium_train['target'], label='station 2')

plt.legend()
y_pred_test=linreg.predict(X_test)
print("The Mean Squared Error on the Test set is:\t{:0.1f}".format(mse(y_test, y_pred_test)))

print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test)))
ammonium_test=pd.read_csv('/kaggle/input/ammonium-prediction-in-river-water/test.csv')

ammonium_test.head()
newest_pred=linreg.predict(ammonium_test[['1']])

ammonium_test['target station guesses']=newest_pred

ammonium_test.head()
print("The Mean Squared Error on the Test set is:\t{:0.1f}".format(mse(y[0:63],newest_pred)))

print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y[0:63], newest_pred)))
ammonium_test['1'].corr(ammonium_test['2'])
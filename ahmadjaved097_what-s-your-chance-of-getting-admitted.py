import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt     #data visualization

import seaborn as sns               #data visualization



%matplotlib inline
df = pd.read_csv('../input/Admission_Predict.csv')

df.head()
df.drop('Serial No.', axis = 1, inplace = True)
print(df.info())
df.describe()
plt.figure(figsize = (10,6))

sns.heatmap(df.corr(), annot = True, linewidths=0.5, cmap = 'coolwarm')

plt.show()
sns.set_style('whitegrid')

sns.pairplot(df)

plt.show()
sns.countplot(df['University Rating'])

plt.show()
sns.lmplot('CGPA','GRE Score', data = df,palette='hls',hue = 'University Rating',fit_reg=False)

plt.show()
sns.lmplot('CGPA','TOEFL Score', data = df,palette='hls',hue = 'University Rating',fit_reg=False)

plt.show()
sns.lmplot('CGPA','Chance of Admit ', data = df,palette='hls',hue = 'University Rating',fit_reg=False)

plt.show()
plt.figure(figsize=(10,6))

sns.distplot(df['TOEFL Score'], kde = False, bins = 30, color = 'blue')

plt.title('TOEFL Score Distribution')

plt.show()
plt.figure(figsize=(10,6))

sns.distplot(df['GRE Score'], kde = False, bins = 30, color = 'red')

plt.title('GRE Score Distribution')

plt.show()
plt.figure(figsize=(10,6))

sns.distplot(df['CGPA'], kde = False, bins = 30, color = 'purple')

plt.title('CGPA Distribution')

plt.show()
X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',

       'Research']]

y = df['Chance of Admit ']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 101)
from sklearn.linear_model import LinearRegression

linearmodel = LinearRegression()
linearmodel.fit(X_train,y_train)
cdf = pd.DataFrame(linearmodel.coef_, X.columns, columns=['Coefficient'])

cdf
prediction = linearmodel.predict(X_test)
plt.figure(figsize=(9,6))

plt.scatter(y_test,prediction)

plt.xlabel('Test Values for y')

plt.ylabel('Predicted Values for y')

plt.title('Scatter Plot of Real Test Values vs. Predicted Values ')

plt.show()
from sklearn import metrics

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test,prediction))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, prediction))

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test,prediction)))
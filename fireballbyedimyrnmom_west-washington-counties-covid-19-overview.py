import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pandas as pd

import numpy as np

df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')



data=df.drop(['fips'], axis = 1) 

data
plt.figure(figsize=(12,8)) # Figure size

data.groupby("state")['deaths'].max().plot(kind='bar', color='darkred')
WA=data.loc[data['state']== 'Washington']

WA
WA.sort_values(by=['cases'], ascending=False)
WA.sort_values(by=['deaths'], ascending=False)
plt.figure(figsize=(12,8)) # Figure size

WA.groupby("county")['cases'].max().plot(kind='bar', color='olive')
plt.figure(figsize=(12,8)) # Figure size

WA.groupby("county")['deaths'].max().plot(kind='bar', color='darkgray')
plt.figure(figsize=(16,11))

sns.lineplot(x="date", y="deaths", hue="county",data=WA)

plt.xticks(WA.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees

plt.show()
plt.figure(figsize=(16,11))

sns.lineplot(x="date", y="cases", hue="county",data=WA)

plt.xticks(WA.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees

plt.show()
King=WA.loc[WA['county']== 'King']

King
plt.figure(figsize=(16,9)) # Figure size

sns.lineplot(x='date', y='cases', data=King, marker='o', color='peru') 

plt.title('Cases in King county') # Title

plt.xticks(King.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees

plt.show()
plt.figure(figsize=(16,9)) # Figure size

sns.lineplot(x='date', y='deaths', data=King, marker='o', color='steelblue') 

plt.title('Reported deaths in King county') # Title

plt.xticks(King.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees

plt.show()
Prc=WA.loc[WA['county']== 'Pierce']

Prc
plt.figure(figsize=(16,9)) # Figure size

sns.lineplot(x='date', y='cases', data=Prc, marker='o', color='tomato') 

plt.title('Cases in Pierce county') # Title

plt.xticks(Prc.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees

plt.show()
plt.figure(figsize=(16,9)) # Figure size

sns.lineplot(x='date', y='deaths', data=Prc, marker='o', color='darkkhaki') 

plt.title('Deaths in Pierce county') # Title

plt.xticks(Prc.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees

plt.show()
Sno=WA.loc[WA['county']== 'Snohomish']

Sno
plt.figure(figsize=(16,9)) # Figure size

sns.lineplot(x='date', y='cases', data=Sno, marker='o', color='indigo') 

plt.title('Cases in Snohomish county') # Title

plt.xticks(Sno.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees

plt.show()
plt.figure(figsize=(16,9)) # Figure size

sns.lineplot(x='date', y='deaths', data=Sno, marker='o', color='dimgrey') 

plt.title('Deaths in Snohomish county') # Title

plt.xticks(Sno.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees

plt.show()
#concat dfs



DF1 = pd.concat([King,Prc, Sno])

DF1
plt.figure(figsize=(16,11))

sns.lineplot(x="cases", y="deaths", hue="county",data=DF1)

plt.title('Death across counties') # Title
plt.figure(figsize=(16,11))

sns.lineplot(x="date", y="cases", hue="county",data=DF1)

plt.title('Cases across counties') # Title

plt.xticks(DF1.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees

plt.show()
DF1.corr().style.background_gradient(cmap='magma')
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split 
X = King['cases'].values.reshape(-1,1)

y = King['deaths'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm
print(regressor.intercept_)

print(regressor.coef_)
y_pred = regressor.predict(X_test)
preds1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

preds1
plt.scatter(X_test, y_test,  color='palevioletred')

plt.plot(X_test, y_pred, color='black', linewidth=1)

plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
X1 = Sno['cases'].values.reshape(-1,1)

y1 = Sno['deaths'].values.reshape(-1,1)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=1)
regressor1 = LinearRegression()  

regressor1.fit(X_train1, y_train1) #training the algorithm
y_pred2 = regressor1.predict(X_test1)
preds2 = pd.DataFrame({'Actual': y_test1.flatten(), 'Predicted': y_pred2.flatten()})

preds2
plt.scatter(X_test1, y_test1,  color='rosybrown')

plt.plot(X_test1, y_pred2, color='black', linewidth=1)

plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test1, y_pred2))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test1, y_pred2))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test1, y_pred2)))
X2 = Prc['cases'].values.reshape(-1,1)

y2 = Prc['deaths'].values.reshape(-1,1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.1, random_state=0)
regressor2 = LinearRegression()  

regressor2.fit(X_train2, y_train2) #training the algorithm
y_pred3 = regressor2.predict(X_test2)
preds3 = pd.DataFrame({'Actual': y_test2.flatten(), 'Predicted': y_pred3.flatten()})

preds3
plt.scatter(X_test2, y_test2,  color='goldenrod')

plt.plot(X_test2, y_pred3, color='black', linewidth=1)

plt.show()
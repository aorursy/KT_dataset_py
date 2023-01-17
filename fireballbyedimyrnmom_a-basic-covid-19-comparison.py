import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pandas as pd
import numpy as np
df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')

data=df.drop(['fips'], axis = 1) 
data
data.sort_values(by=['cases'], ascending=False)
data.sort_values(by=['deaths'], ascending=False)
plt.figure(figsize=(12,8)) # Figure size
data.groupby("state")['cases'].max().plot(kind='bar', color='olivedrab')
data.plot.line()
plt.figure() # for defining figure sizes
data.plot(x='state', y='deaths', figsize=(12,8), color='goldenrod')
WA=data.loc[data['state']== 'Washington']
WA
WA.groupby('county').plot(x='date', y='deaths')
NY=data.loc[data['state']== 'New York']
NY
DF1 = pd.concat([WA,NY])
DF1
DFGroup = DF1.groupby(['cases'])

DFGPlot = DFGroup.sum().unstack().plot(kind='bar', figsize=(15,10))
plt.style.use('ggplot')
WA.plot(kind='bar', figsize=(16,9))
plt.ylabel('total')

DF1.plot('state',['deaths','cases'],kind = 'line', figsize=(16,9))
plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=WA, marker='o', color='lightseagreen') 
plt.title('Cases in Washington') # Title
plt.xticks(DF1.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()
plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=NY, marker='o', color='darkmagenta') 
plt.title('Cases in New York state') # Title
plt.xticks(DF1.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()
plt.figure(figsize=(16,9)) # Figure size
sns.lineplot(x='date', y='cases', data=DF1, marker='o', color='royalblue') 
plt.title('Cases in the states of WA and NY') # Title
plt.xticks(DF1.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees
plt.show()
DF1.corr().style.background_gradient(cmap='magma')
ax = WA.plot()
NY.plot(ax=ax)
plt.figure(figsize=(16,11))
sns.lineplot(x="cases", y="deaths", hue="county",data=WA)
plt.figure(figsize=(16,11))
sns.lineplot(x="cases", y="deaths", hue="state",data=DF1)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split 
X = WA['cases'].values.reshape(-1,1)
y = WA['deaths'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(X_test)
preds1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
preds1
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
LogReg = LogisticRegression()  
LogReg.fit(X_train, y_train) #training the algorithm
y_pred = LogReg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(LogReg.score(X_test, y_test)))
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_percentage = 100 * accuracy
accuracy_percentage
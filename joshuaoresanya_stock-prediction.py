import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib



plt.style.use('fivethirtyeight')

matplotlib.rcParams['axes.labelsize'] = 14

matplotlib.rcParams['xtick.labelsize'] = 12

matplotlib.rcParams['ytick.labelsize'] = 12

matplotlib.rcParams['text.color'] = 'k'
df = pd.read_csv('../input/stock-dataset/Stock.csv', parse_dates = ['Date'])
df.head(10)
df.dtypes
df.columns
df.set_index('Date', inplace=True)

df.head(10)
df = df[['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume']]
df.head(10)
df.describe()
fig = plt.figure()

axes = fig.add_axes([1, 1, 1, 1])

df['Adj_Volume'].plot(kind='hist')
df.plot.scatter(x='Adj_Close', y='Adj_Volume')
fig = plt.figure()

axes = fig.add_axes([1, 1, 1, 1])

sns.distplot(df['Adj_Volume'])
df.corr()
fig = plt.figure()

axes = fig.add_axes([1, 1, 2, 1.0])

axes.plot(df.index, df['Adj_Close'])

plt.title('XYZ Company')

plt.ylabel('Adj_Close($)')

plt.show()
#plot heatmap to find correlation among features

#We can see that Adj_Volume is negatively corellated with all the remaining features observed. 

corrmat = df.corr()

f, ax = plt.subplots(figsize = (15,8))

sns.heatmap(corrmat, square = True, annot = True, linewidth = 0.8, cmap = 'RdBu')
df.head()
df['Adj_Close'] = df['Adj_Close'].rolling(20).mean()
df['Label'] = df['Adj_Close'].shift(-1)
df1 = df.copy()
df
df.dropna(inplace=True)
df.head(10) #from the new dataframe, every row have been mapped against its future value. 
X = df[['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume']]

y = df['Label']
#Let's do some preprocessing but before then, let's convert both the labels and features to np.arrays for easy mathematical calculation

X = np.array(df[['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume']])

y = np.array(df['Label'])
#import preprocessing

from sklearn import preprocessing, model_selection
X = preprocessing.scale(X) #this is to scale down the features between 0 and -1 for fast computation
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=1)
from sklearn.linear_model import LinearRegression
reg = LinearRegression() #creating an instance of the linear regression model

reg.fit(X_train, y_train) #this will map the X_train and y_train and create an hypotesis. after the hypotesis is done, then, 

#it will plot a best fit line between the X and the y to give us the right prediction for the unseen data. 
accuracy = reg.score(X_test, y_test)

print('The accuracy value is:', accuracy)
y_pred=reg.predict(X_test) # now predict
df2 = pd.DataFrame(y_pred, y_test)

df2.head()
from sklearn.metrics import classification_report

import statsmodels.api as sm
x = sm.add_constant(X)

results = sm.OLS(y, x).fit()

results.summary()
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
#Rounding the R Squared score 

print("R2 score : %.2f" % r2_score(y_test,y_pred))
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, y_pred)
from sklearn.metrics import explained_variance_score

explained_variance_score(y_test, y_pred)
from sklearn.linear_model import LinearRegression

from sklearn.datasets import make_regression
# generate regression dataset

X, y = make_regression(n_samples=100, n_features=2, noise=0.1)

# fit final model

model = LinearRegression()

model.fit(X, y)
# new instances where we do not know the answer

y_test, _ = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=1)

# make a prediction

y_pred = model.predict(y_test)

# show the inputs and predicted outputs

for i in range(len(y_test)):

	print("y=%s, Predicted=%s" % (y_test[i], y_pred[i]))
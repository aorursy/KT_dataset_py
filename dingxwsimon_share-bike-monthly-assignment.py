import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
data = pd.read_csv("../input/train.csv")
data.dtypes
testdata = pd.read_csv("../input/test.csv")
testdata.dtypes
null_columns=testdata.columns[testdata.isnull().any()]
testdata[null_columns].isnull().sum()
print(data['count'].max())
print(data['count'].min())
data[data['weather'] == 0].count()
null_columns=data.columns[data.isnull().any()]
data[null_columns].isnull().sum()
X = data[['datetime','season','holiday','workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']]
Y = data['count']
X['date'] = pd.to_datetime(X['datetime'])
X['weekday'] = X.apply (lambda row: row['date'].weekday(),axis=1)
X['hour'] = X.apply (lambda row: row['date'].hour,axis=1)
print(X.isnull().values.any())
plt.scatter(X['atemp'], Y)
plt.show()
X['atemp'].hist()
plt.show()
plt.scatter(X['humidity'], Y)
plt.show()
plt.scatter(X['windspeed'], Y)
plt.show()
#box plot overallqual/saleprice

data = pd.concat([X['weather'],Y], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='weather', y="count", data=data)
fig.axis(ymin=0, ymax=800);
corrMatrix=pd.concat([X,Y], axis=1).corr()

sns.set(font_scale=1.10)
plt.figure(figsize=(10, 10))

sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between features');
X = X[['hour','season','holiday','workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']]
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=0)
xtrain = np.asmatrix(xtrain)
xtest = np.asmatrix(xtest)
ytrain = np.ravel(ytrain)
ytest = np.ravel(ytest)
model = LinearRegression()
model.fit(xtrain, ytrain)
model.intercept_
model.coef_
pred = model.predict(xtrain)
((pred-ytrain)*(pred-ytrain)).sum() / len(ytrain)
(abs(pred-ytrain)/ytrain).sum() / len(ytrain)
testdata['date'] = pd.to_datetime(testdata['datetime'])
testdata['weekday'] = testdata.apply (lambda row: row['date'].weekday(),axis=1)
testdata['hour'] = testdata.apply (lambda row: row['date'].hour,axis=1)
print(testdata.isnull().values.any())
test_X = testdata[['hour','season','holiday','workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']]
pred = model.predict(test_X)
df = pd.DataFrame({"datetime": testdata['datetime'],"count": pred})
df['count'] = df.apply (lambda row: int(row['count']) if int(row['count']) > 0 else 1 ,axis=1)

df.to_csv('submission.csv',index = False, header=True)

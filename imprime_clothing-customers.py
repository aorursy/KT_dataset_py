import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/Ecommerce Customers.csv")

df.columns
df.info()
df.head()
df.describe()
sns.jointplot("Time on Website", "Yearly Amount Spent", data=df)
sns.jointplot("Time on App", "Yearly Amount Spent", data=df)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

# For Notebooks

init_notebook_mode(connected=True)

# For offline use

cf.go_offline()
df3 = df[df.columns[3:6]]

df3 =df3.loc[:20]

df3.iplot(kind='scatter')
sns.jointplot("Time on App", "Length of Membership", data=df, kind="hex")
sns.pairplot(df)
sns.lmplot('Length of Membership','Yearly Amount Spent', df, scatter_kws={"s": 10})
X = df[['Avg. Session Length', 'Time on App',

       'Time on Website', 'Length of Membership']]

X.shape
y =df[['Yearly Amount Spent']]

y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
line = LinearRegression()


line.fit(X_train, y_train)
line.score(X_test,y_test)
line.coef_
pridict_y = line.predict(X_test)
plt.scatter(y_test, pridict_y, s=10)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score
print(mean_absolute_error(y_test, pridict_y))

print(mean_squared_error(y_test, pridict_y))

print(np.sqrt(mean_squared_error(y_test, pridict_y)))

print(explained_variance_score(y_test, pridict_y))

print(line.score(X_test, y_test))

print(r2_score(y_test, pridict_y))
coeffecients = pd.DataFrame(np.array(line.coef_).reshape(-1,1),X.columns)

coeffecients.columns = ['Coeffecient']

coeffecients
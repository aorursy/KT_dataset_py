# Import Libreries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('../input/StudentsPerformance.csv')
data.head()
# Some statistics about the data
data.describe()
data.info()
data['gender'].value_counts()
sns.catplot(x='gender',kind='count',data=data,height=4.5,palette='viridis')

plt.title('Gender')
data['gender'].replace({'male':'0','female':'1'},inplace=True)
data['race/ethnicity'].value_counts()
data["race/ethnicity"].sort_values()

sns.catplot(x='race/ethnicity',kind='count',data=data,height=4.5,palette='viridis',

            order=['group A','group B','group C','group D','group E'])
data['race/ethnicity'].replace({'group A':'1','group B':'2', 'group C':'3',

                               'group D':'4','group E':'5'},inplace=True)
data['lunch'].value_counts()
sns.catplot(x='lunch',kind='count',data=data,height=4.5,palette='viridis')
data['lunch'].replace({'free/reduced':'0','standard':'1'},inplace=True)
data['test preparation course'].value_counts()
sns.catplot(x='test preparation course',kind='count',data=data,height=4.5,palette='viridis')
data['test preparation course'].replace({'none':'0','completed':'1'},inplace=True)
data['parental level of education'].value_counts()
data["race/ethnicity"].sort_values()

sns.catplot(x='parental level of education',kind='count',data=data,height=4.5,aspect=2,palette='viridis',

            order=["some high school","high school","associate's degree","some college",

                   "bachelor's degree","master's degree"],)
data['parental level of education'].replace({'some high school':'1','high school':'1',"associate's degree":'2',

                                        'some college':'3',"bachelor's degree":'4',"master's degree":'5'},inplace=True)
data.head()
# Now we will look a little deeper on some interesting plots of our data, in order to get some insights
sns.set(rc={'figure.figsize':(20,6)})

sns.countplot(x='writing score', hue='test preparation course',data=data, palette='viridis')

plt.title('Writing Score by Test Preparation Course')
plt.figure(figsize=(10, 6))

sns.scatterplot(x='math score',y='reading score', hue='gender',data=data, palette='viridis')

plt.title('Math score VS Readind Score by Gender')
# Take a look at our objectives distribution

plt.figure(figsize=(8, 4))

plt.hist(x='writing score',bins=10,data=data)
sns.set(rc={'figure.figsize':(20,6)})

sns.countplot(x='writing score', hue='lunch',data=data, palette='viridis')
X = data[['gender','race/ethnicity','parental level of education','lunch','test preparation course','math score','reading score']]
y = data['writing score']
# Split the data to train and test

from sklearn import model_selection

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# implementation of Linear Regression model using scikit-learn and K-fold for stable model

from sklearn.linear_model import LinearRegression

kfold = model_selection.KFold(n_splits=10)

lr = LinearRegression()

scoring = 'r2'

results = model_selection.cross_val_score(lr, X, y, cv=kfold, scoring=scoring)

lr.fit(X_train,y_train)

lr_predictions = lr.predict(X_test)

print('Coefficients: \n', lr.coef_)
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, lr_predictions))

print('MSE:', metrics.mean_squared_error(y_test, lr_predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lr_predictions)))
from sklearn.metrics import r2_score

print("R_square score: ", r2_score(y_test,lr_predictions))
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state = 42)

dtr.fit(X_train,y_train)

dtr_predictions = dtr.predict(X_test) 



# R^2 Score

print("R_square score: ", r2_score(y_test,dtr_predictions))
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators = 100)

rfr.fit(X_train,y_train)

rfr_predicitions = rfr.predict(X_test) 



# R^2 Score

print("R_square score: ", r2_score(y_test,rfr_predicitions))
from sklearn import ensemble

clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,

          learning_rate = 0.1, loss = 'ls')

clf.fit(X_train, y_train)

clf_predicitions = clf.predict(X_test) 

print("R_square score: ", r2_score(y_test,clf_predicitions))
y = np.array([r2_score(y_test,lr_predictions),r2_score(y_test,dtr_predictions),r2_score(y_test,rfr_predicitions),

           r2_score(y_test,clf_predicitions)])

x = ["LinearRegression","RandomForest","DecisionTree","Grdient Boost"]

plt.bar(x,y)

plt.title("Comparison of Regression Algorithms")

plt.ylabel("r2_score")

plt.show()
import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



sns.set(style="whitegrid")

sns.set_color_codes("pastel")

%matplotlib inline
# I'm use only student-por.csv

data = pd.read_csv('../input/student-por.csv')
data.head(5)
data.info()
f, ax = plt.subplots(figsize=(4, 4))

plt.pie(data['sex'].value_counts().tolist(), 

        labels=['Female', 'Male'], colors=['#ffd1df', '#a2cffe'], 

        autopct='%1.1f%%', startangle=90)

axis = plt.axis('equal')
fig, ax = plt.subplots(figsize=(5, 4))

sns.distplot(data['age'],  

             hist_kws={"alpha": 1, "color": "#a2cffe"}, 

             kde=False, bins=8)

ax = ax.set(ylabel="Count", xlabel="Age")
f, ax = plt.subplots(figsize=(4, 4))

plt.pie(data['studytime'].value_counts().tolist(), 

        labels=['2 to 5 hours', '<2 hours', '5 to 10 hours', '>10 hours'], 

        autopct='%1.1f%%', startangle=0)

axis = plt.axis('equal')
f, ax = plt.subplots(figsize=(4, 4))

plt.pie(data['romantic'].value_counts().tolist(), 

        labels=['No', 'Yes'], autopct='%1.1f%%', startangle=90)

axis = plt.axis('equal')
fig, ax = plt.subplots(figsize=(5, 4))

sns.distplot(data['Walc'],  

             hist_kws={"alpha": 1, "color": "#a2cffe"}, 

             kde=False, bins=4)

ax = ax.set(ylabel="Students", xlabel="Weekend Alcohol Consumption")
plot1 = sns.factorplot(x="Walc", y="health", hue="sex", data=data)

plot1.set(ylabel="Health", xlabel="Weekend Alcohol Consumption")



plot2 = sns.factorplot(x="Dalc", y="health", hue="sex", data=data)

plot2.set(ylabel="Health", xlabel="Workday Alcohol Consumption")
plot1 = sns.factorplot(x="G3", y="Walc", data=data)

plot1.set(ylabel="Final Grade", xlabel="Weekend Alcohol Consumption")



plot2 = sns.factorplot(x="G3", y="Dalc", data=data)

plot2.set(ylabel="Final Grade", xlabel="Workday Alcohol Consumption")
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, Lasso

from sklearn.linear_model import Ridge



from sklearn.model_selection import cross_val_score
y = data['G3']

X = data.drop(['G3'], axis=1)
X = pd.get_dummies(X)
names = ['DecisionTreeRegressor', 'LinearRegression', 'Ridge', 'Lasso']



clf_list = [DecisionTreeRegressor(),

            LinearRegression(),

            Ridge(),

            Lasso()]
for name, clf in zip(names, clf_list):

    print(name, end=': ')

    print(cross_val_score(clf, X, y, cv=5).mean())
tree = DecisionTreeRegressor()

tree.fit(X, y)
importances = tree.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X.shape[1]):

    print("%d. Feature %s (%f)" % (f + 1, X.columns.values[indices[f]], importances[indices[f]]))
X = data.drop(['G3', 'G2', 'G1'], axis=1)
X = pd.get_dummies(X)
for name, clf in zip(names, clf_list):

    print(name, end=': ')

    print(cross_val_score(clf, X, y, cv=5).mean())
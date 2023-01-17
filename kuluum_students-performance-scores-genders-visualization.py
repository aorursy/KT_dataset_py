import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
data = pd.read_csv('../input/StudentsPerformance.csv')
data.head()
#just shortner parental degree to fit it on plots
data['parental level of education'].replace('bachelor\'s degree', 'bachelor', inplace=True)
data['parental level of education'].replace('some college', 'college', inplace=True)
data['parental level of education'].replace('master\'s degree', 'master', inplace=True)
data['parental level of education'].replace('associate\'s degree', 'associate', inplace=True)

sns.kdeplot(data['math score'])
sns.kdeplot(data['reading score'])
sns.kdeplot(data['writing score'])
plt.legend();
sns.catplot(x="parental level of education", y="math score", kind="violin", hue="gender", split=True, data=data, height=8)
sns.catplot(x="parental level of education", y="reading score", kind="violin", hue="gender", split=True, data=data, size=8)
sns.catplot(x="race/ethnicity", y="writing score", kind="violin", hue="gender", split=True, data=data, size=8);
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,8), sharex=True)
g1 = sns.catplot(x="test preparation course", y="math score", kind="violin", hue="gender", split=True, data=data, ax=ax1)
g2 = sns.catplot(x="test preparation course", y="reading score", kind="violin", hue="gender", split=True, data=data, ax=ax2)
g3 = sns.catplot(x="test preparation course", y="writing score", kind="violin", hue="gender", split=True, data=data, ax=ax3)
plt.close(g1.fig)
plt.close(g2.fig)
plt.close(g3.fig)
sns.jointplot(x=data['math score'], y=data['reading score'], kind="hex");
sns.jointplot(x=data['math score'], y=data['writing score'], kind="hex");
sns.jointplot(x=data['writing score'], y=data['reading score'], kind="hex");
f, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
sns.scatterplot(x="math score", y="reading score", data=data, ax=axes[0], hue="gender")
sns.scatterplot(x="reading score", y="writing score", data=data, ax=axes[1], hue="gender")
sns.scatterplot(x="math score", y="writing score", data=data, ax=axes[2], hue="gender")
X = data[['gender','race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']]
y = data[['math score', 'math score', 'reading score']]
overall_score = y.values.mean(axis=1)
Xa2 = pd.get_dummies(X, drop_first = True)
#Model
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
models = []
models.append(['Ridge', Ridge()])
models.append(['GBM', GradientBoostingRegressor()])
models.append(['ada', AdaBoostRegressor()])
models.append(['SVR', SVR(gamma='auto')])
models.append(['NuSVR', NuSVR(gamma='auto')])
models.append(['LinearSVR', LinearSVR()])
results = []
names =[]

for name, model in models:    
    kfold = KFold(n_splits = 15, random_state = 11)
    cv_result = cross_val_score(model, Xa2, overall_score, cv =kfold, scoring = 'r2')
    results.append(cv_result)
    names.append(name)
    msg = '%s: %f (%f)' % (name, cv_result.mean(), cv_result.std())
    print(msg)
plt.figure(figsize = (10,5))
sns.boxplot(x = names, y = results)
plt.show()

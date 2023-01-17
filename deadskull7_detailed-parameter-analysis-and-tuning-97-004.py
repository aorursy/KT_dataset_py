# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd
import math 
# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV

import keras
from keras import backend as K

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
%matplotlib inline
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6
train = pd.read_csv('../input/voice.csv')
df = train.copy()
df.head()
df.shape
df.columns
df.describe()
df.isnull().sum()
temp = []
for i in df.label:
    if i == 'male':
        temp.append(1)
    else:
        temp.append(0)
df['label'] = temp
df.label.value_counts()
correlation_map = df.corr()
obj = np.array(correlation_map)
obj[np.tril_indices_from(obj)] = False
fig,ax= plt.subplots()
fig.set_size_inches(15,15)
sns.heatmap(correlation_map, mask=obj,vmax=.7, square=True,annot=True)
kws = dict(s=50, linewidth=.5, edgecolor="w")
g = sns.FacetGrid(df, col="label", palette="Set1")
g = (g.map(plt.scatter, "meanfun", "IQR", **kws).add_legend())
kws = dict(s=50, linewidth=.5, edgecolor="g")
g = sns.FacetGrid(df, col="label", palette="Pal")
g = (g.map(plt.scatter, "sp.ent", "Q25", **kws).add_legend())
kws = dict(s=50, linewidth=.5, edgecolor="y")
g = sns.FacetGrid(df, col="label")
g = (g.map(plt.scatter, "Q25", "meanfun", **kws).add_legend())
kws = dict(s=50, linewidth=.5, edgecolor="m")
g = sns.FacetGrid(df, col="label", palette="Set1")
g = (g.map(plt.scatter, "IQR", "sd", **kws).add_legend())
sns.set(style="white", color_codes=True)
sns.jointplot("meanfun", "IQR", data=df, kind="reg" , color='k')
sns.set(style="darkgrid", color_codes=True)
sns.jointplot("sp.ent", "Q25", data=df, kind="hex" , color='g')
sns.set(style="whitegrid", color_codes=True)
g = (sns.jointplot(df.Q25 , df.meanfun , kind="hex", stat_func=None).set_axis_labels("Q25", "meanfun"))
g = sns.PairGrid(df[["meanfun" , "Q25" , "sd" , "meanfreq" , "IQR" , "sp.ent" , "centroid", "label"]] , palette="Set2" , hue = "label")
g = g.map(plt.scatter, linewidths=1, edgecolor="r", s=60)
g = g.add_legend()
from sklearn import svm
X = df[df.columns[0:20]]
y = df.label
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2)
clf = svm.SVC()
clf.fit(X_train , y_train)
pred = clf.predict(X_test)
print(accuracy_score(y_test, pred))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
df = scaler.transform(df)
X = df[: , 0:20]
y = df[: , 20]
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2)
clf = svm.SVC()
clf.fit(X_train , y_train)
pred = clf.predict(X_test)
print(accuracy_score(y_test, pred))
from sklearn.model_selection import cross_val_score
clf = svm.SVC()
scores = cross_val_score(clf , X , y , cv=10)
scores
scores.mean()
temp = np.arange(11)[1 : 11]
plt.plot(temp , scores)
plt.show()
from sklearn.model_selection import GridSearchCV
model = svm.SVC()
param_grid = {
"C" : np.linspace(0.01,1,10)
}
grid = GridSearchCV( model , param_grid , cv = 10 , scoring = "accuracy")
grid.fit(X,y)
grid.best_score_
grid.best_params_
grid.grid_scores_
for i in grid.grid_scores_:
    print(i[2])
gridscores = []
for i in grid.grid_scores_:
    gridscores.append(i[1])
    
plt.xlabel('C')
plt.ylabel('Mean cross-validation Accuracy')
plt.plot(np.linspace(0.01,1,10) , gridscores , 'r')
plt.show()
from sklearn.model_selection import GridSearchCV
model = svm.SVC()
param_grid = {
"gamma" : [0.0001,0.001,0.01,0.1,1,10,100,300,600]
}
grid = GridSearchCV( model , param_grid , cv = 10 , scoring = "accuracy")
grid.fit(X,y)
grid.best_score_
grid.best_params_
grid.grid_scores_
gridscores = []
for i in grid.grid_scores_:
    gridscores.append(i[1])
    
plt.xlabel('Gamma')
plt.ylabel('Mean cross-validation Accuracy')
plt.plot([0.0001,0.001,0.01,0.1,1,10,100,300,600] , gridscores , 'k')
plt.show()
model = svm.SVC()
param_grid = {
"C" : [0.50 , 0.55 , 0.59 , 0.63] ,
"gamma" : [0.005, 0.008, 0.010, 0.012 , 0.015]
}
grid = GridSearchCV( model , param_grid , cv = 10 , scoring = "accuracy")
grid.fit(X,y)
grid.best_score_
grid.best_params_
model = svm.SVC(kernel='linear')
param_grid = {
"C" : [0.50 , 0.55 , 0.59 , 0.63] ,
"gamma" : [0.005, 0.008, 0.010, 0.012 , 0.015]
}
grid = GridSearchCV( model , param_grid , cv = 10 , scoring = "accuracy")
grid.fit(X,y)
grid.best_score_
grid.best_params_
model = svm.SVC(kernel='linear',C=0.1 , gamma=0.447)
scores = cross_val_score(model , X, y, cv=10, scoring='accuracy')
print(scores.mean())
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 61 , max_depth = 37)
scores = cross_val_score(model , X , y , cv=10)
scores
scores.mean()
model.fit(X , y)
print(model.feature_importances_)
ranks = np.argsort(-model.feature_importances_)
f, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x=model.feature_importances_[ranks], y=train.columns.values[ranks], orient='h')
ax.set_xlabel("Importance Of Features in RandomForestClassifier")
plt.tight_layout()
plt.show()
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
scores = cross_val_score(model, X , y , cv=5)
scores
scores.mean()
from xgboost import XGBClassifier
xgb = XGBClassifier()
scores = cross_val_score(model, X , y , cv=10)
scores
scores.mean()
xgb = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30)
xgb.fit(X_train , y_train)
pred = xgb.predict(X_test)
print(accuracy_score(y_test , pred))
ranks = np.argsort(-xgb.feature_importances_)
f, ax = plt.subplots(figsize=(12, 7))

sns.barplot(x=xgb.feature_importances_[ranks], y=train.columns.values[ranks], orient='h')
ax.set_xlabel("Importance Of Features in XGBClassifier")
plt.tight_layout()
plt.show()





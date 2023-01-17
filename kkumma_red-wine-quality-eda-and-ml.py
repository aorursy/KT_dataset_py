import numpy as np 
import pandas as pd

!pip install plotnine   
from plotnine import *
%matplotlib inline
#Loading dataset
wine = pd.read_csv('../input/winequality-red.csv')
wine.head(5)
import seaborn as sns
color = sns.color_palette()

import matplotlib.pyplot as plt
sns.set(style="white")
plt.subplots(figsize=(10,8))
cmap = sns.diverging_palette(200, 10, as_cmap=True)
ax = plt.axes()
ax.set_title("Red Wine Quality Correlation Heatmap")
corr = wine.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, cmap=cmap)
wine = wine.drop(['fixed acidity','residual sugar','free sulfur dioxide','pH'],axis=1)
bins = [0, 4, 6, 10]
labels = ["poor","normal","excellent"]
wine['binned_quality'] = pd.cut(wine['quality'], bins=bins, labels=labels)
sns.boxplot(y='alcohol', x='binned_quality', 
                 data=wine, 
                 width=0.5,
                 palette="rocket")
# Bin "alcohol" variable into three levels: low, median and high
bins = [0, 10, 12, 15]
labels = ["low alcohol","median alcohol","high alcohol"]
wine['binned_alcohol'] = pd.cut(wine['alcohol'], bins=bins, labels=labels)
wine.drop('alcohol',axis =1, inplace = True)
(ggplot(wine, aes('citric acid', 'volatile acidity', color = 'binned_alcohol'))
 + geom_point(alpha=0.3)
 + facet_wrap("binned_quality",ncol =1))
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
wine = pd.read_csv('../input/winequality-red.csv')
bins = [0, 4, 6, 10]
labels = ["poor","normal","excellent"]
wine['binned_quality'] = pd.cut(wine['quality'], bins=bins, labels=labels)
wine = wine.drop('quality', axis = 1)
X = wine.drop(['binned_quality'], axis = 1)
y = wine['binned_quality']
#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
from sklearn.metrics import confusion_matrix, classification_report
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))
from sklearn.linear_model import SGDClassifier

svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
print(classification_report(y_test, pred_svc))

#Finding best parameters for our SVC model
param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(X_train, y_train)
grid_svc.best_params_
svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print(classification_report(y_test, pred_svc2))
rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)
rfc_eval.mean()
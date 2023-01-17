# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
### Import Lib

%matplotlib inline
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge,BayesianRidge
from sklearn.metrics import mean_squared_error
from math import radians, cos, sin, asin, sqrt
import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
df_wine=pd.read_csv("../input/winequality-red.csv")

df_wine.describe()
df_wine.info()
np.round(df_wine.describe())

m=np.mean(df_wine['fixed acidity'])
st=np.std(df_wine['fixed acidity'])
df_wine=df_wine[df_wine['fixed acidity']<=m+2*st]
df_wine=df_wine[df_wine['fixed acidity']>=m-2*st]
m=np.mean(df_wine['volatile acidity'])
st=np.std(df_wine['volatile acidity'])
df_wine=df_wine[df_wine['volatile acidity']<=m+2*st]
df_wine=df_wine[df_wine['volatile acidity']>=m-2*st]
m=np.mean(df_wine['citric acid'])
st=np.std(df_wine['citric acid'])
df_wine=df_wine[df_wine['citric acid']<=m+2*st]
df_wine=df_wine[df_wine['citric acid']>=m-2*st]
m=np.mean(df_wine['residual sugar'])
st=np.std(df_wine['residual sugar'])
df_wine=df_wine[df_wine['residual sugar']<=m+2*st]
df_wine=df_wine[df_wine['residual sugar']>=m-2*st] 
#chlorides free sulfur dioxide total sulfur dioxide density pH sulphates alcohol
m=np.mean(df_wine['chlorides'])
st=np.std(df_wine['chlorides'])
df_wine=df_wine[df_wine['chlorides']<=m+2*st]
df_wine=df_wine[df_wine['chlorides']>=m-2*st] 

m=np.mean(df_wine['free sulfur dioxide'])
st=np.std(df_wine['free sulfur dioxide'])
df_wine=df_wine[df_wine['free sulfur dioxide']<=m+2*st]
df_wine=df_wine[df_wine['free sulfur dioxide']>=m-2*st]


m=np.mean(df_wine['total sulfur dioxide'])
st=np.std(df_wine['total sulfur dioxide'])
df_wine=df_wine[df_wine['total sulfur dioxide']<=m+2*st]
df_wine=df_wine[df_wine['total sulfur dioxide']>=m-2*st]


m=np.mean(df_wine['density'])
st=np.std(df_wine['density'])
df_wine=df_wine[df_wine['density']<=m+2*st]
df_wine=df_wine[df_wine['density']>=m-2*st]

m=np.mean(df_wine['pH'])
st=np.std(df_wine['pH'])
df_wine=df_wine[df_wine['pH']<=m+2*st]
df_wine=df_wine[df_wine['pH']>=m-2*st]

m=np.mean(df_wine['sulphates'])
st=np.std(df_wine['sulphates'])
df_wine=df_wine[df_wine['sulphates']<=m+2*st]
df_wine=df_wine[df_wine['sulphates']>=m-2*st]


m=np.mean(df_wine['alcohol'])
st=np.std(df_wine['alcohol'])
df_wine=df_wine[df_wine['alcohol']<=m+2*st]
df_wine=df_wine[df_wine['alcohol']>=m-2*st]
plt.hist(df_wine['quality'].values, bins=5)
plt.xlabel('quality')
plt.ylabel('number of  records')
plt.show()
y=df_wine.quality
#feature1=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
feature=['fixed acidity','residual sugar','sulphates','alcohol']
x=df_wine[feature]
df_wine.drop(['quality'],axis=1,inplace=True)
x.head()
y.head()

#sns.pairplot(df_wine, x_vars=['fixed acidity','volatile acidity','citric acid'],y_vars='y', palette="husl")
sns.pairplot(df_wine, x_vars=['fixed acidity','volatile acidity','citric acid'],y_vars=['fixed acidity','volatile acidity','citric acid'], palette="husl",kind='reg')
sns.pairplot(df_wine,x_vars=['residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide'],y_vars=['residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide'],palette="husl",kind='reg')
sns.pairplot(df_wine,x_vars=['density','pH','sulphates','alcohol'],y_vars=['density','pH','sulphates','alcohol'],palette="husl",kind='reg')
X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)

clf1 = LogisticRegression(random_state=5)
clf1.fit(X_train, y_train)
pred = clf1.predict(X_test)
clf1.score(X_test, y_test)
#metrics.accuracy_score(y_test, pred)
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=200, 
                                 learning_rate=0.2, 
                                 max_depth=5, 
                                 random_state=5)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
clf.score(X_test, y_test)
parameter_gridsearch = {
                 'max_depth' : [3, 4],  #depth of each decision tree
                 'n_estimators': [50, 20],  #count of decision tree
                 'max_features': ['sqrt', 'auto', 'log2'],      
                 'min_samples_split': [2],      
                 'min_samples_leaf': [1, 3, 4],
                 'bootstrap': [True, False],
                 }
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


x, y = make_classification(n_samples=1000, n_features=2,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x, y)

print(clf.score(x, y))
print(list(zip(feature,clf.feature_importances_)))
pred = clf.predict(x)
from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(y, pred)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# ROC curve on Predicted probabilities
pred_proba = clf.predict_proba(x)
fpr, tpr, _ = metrics.roc_curve(y, pred_proba[:,1])
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

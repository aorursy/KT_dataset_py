# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
df.head()
df.info()
df.describe()
df['class'].value_counts()
#change the class(e,p) to (0,1). That means 0 is eatable and 1 is poisonous
df['class'].replace(['e','p'],[0,1],inplace=True)
#get dummy variables
df = pd.get_dummies(df,drop_first=True)
from sklearn.model_selection import train_test_split
X = df.drop(['class'],axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
# import different algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Run cross validation to get accuracy score and confusion matrix
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# prepare models
models=[]
models.append(('LR',LogisticRegression()))
models.append(('RFC',RandomForestClassifier()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVC',svm.SVC()))
models.append(('GBC',GradientBoostingClassifier()))
models.append(('XGC',XGBClassifier()))
# evaluate each model in turn
for name, model in models:
    model.fit(X_train,y_train)
    score = model.score(X_test,y_test)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    ax = sns.heatmap(cm,fmt='d',annot=True,xticklabels=['Predicted_0','Predicted_1'],yticklabels=['True_0','True_1'])
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()
    accuracy = "%s: %f" % (name, score)
    print(accuracy)
#hyper-parameters of logisticRegression
from sklearn.model_selection import cross_val_score

c_n=np.logspace(-3,3,7)
cross_val_scores=[]
for i in c_n:
    logmodel = LogisticRegression(C=i,solver='liblinear')
    scores=cross_val_score(logmodel, X_train, y_train, cv=10,scoring='accuracy')
    cross_val_scores.append(np.mean(scores))
print("best cross-validation score: {:.3f}".format(np.max(cross_val_scores)))
best_c_n=c_n[np.argmax(cross_val_scores)]
print("best c_n: {}".format(best_c_n))

logmodel=LogisticRegression(C=best_c_n,solver='liblinear')
logmodel.fit(X_train, y_train)
print("test score: {:.3f}".format(logmodel.score(X_test, y_test)))
from sklearn.model_selection import GridSearchCV
params1 = {'n_estimators':range(30, 200,5)}
grid = GridSearchCV(RandomForestClassifier(random_state=0), param_grid=params1, cv=10, scoring='accuracy', return_train_score=True)
grid.fit(X_train, y_train)

print(grid.best_estimator_)
print(grid.best_params_)
print(grid.best_score_)
#tuning hyper-parameter of XGBClassifier
import time
params2={'max_depth':np.arange(3,7,1)}
grid = GridSearchCV(XGBClassifier(), param_grid=params2, scoring='accuracy', cv=10 )

start=time.time()
grid.fit(X_train, y_train)
end=time.time()
print(end-start)

print(grid.best_estimator_)
print(grid.best_score_)
print(grid.best_params_)
# Find top 10 important features for RFC
model = RandomForestClassifier()
model.fit(X_train,y_train)
importance = model.feature_importances_
indices = np.argsort(importance)[::-1][0:10]
labels = X_train.columns
# Visualize top 10 important features
plt.title('Feature Importance')
plt.bar(range(10),importance[indices],color='lightblue',align='center')
plt.xticks(range(10),labels[indices],rotation=90)
plt.show()

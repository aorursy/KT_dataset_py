import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head()
df.describe()
df['qual']= df['quality'].map(lambda x: 1 if x>6 else 0)
df = df.drop('quality',1)
plt.figure(figsize= (18,10))
sns.heatmap(df.corr(), annot =True)
plt.show()
df.qual.value_counts()
df.info()
df = df.drop(['volatile acidity','free sulfur dioxide'], 1)
plt.figure(figsize = (18,10))
sns.heatmap(df.corr(), annot= True)
plt.show()
df =df.drop('citric acid',1)
plt.figure(figsize = (18,10))
sns.heatmap(df.corr(), annot= True)
plt.show()
from sklearn.model_selection import train_test_split
x= df.drop('qual',1)
y= df['qual']
x_train,x_test, y_train, y_test = train_test_split(x,y, train_size= 0.7, test_size=0.3, random_state=100)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
pred = rfc.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print(classification_report(y_test, pred))
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test, pred))
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
n_folds = 5
parameters = {'max_depth' : range(2,10)}

rf = RandomForestClassifier()
rf = GridSearchCV(rf,parameters, cv = n_folds, scoring = 'accuracy', return_train_score = True )
rf.fit(x_train,y_train)
scores = rf.cv_results_
scores = pd.DataFrame(scores)
scores
plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


n_folds = 5

parameters = {'n_estimators': range(100, 1500, 400)}

rf = RandomForestClassifier(max_depth=4)


rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",return_train_score = True)
rf.fit(x_train, y_train)
scores1 = rf.cv_results_
scores1 = pd.DataFrame(scores1)
scores1
plt.figure()
plt.plot(scores1["param_n_estimators"], 
         scores1["mean_train_score"], 
         label="training accuracy")
plt.plot(scores1["param_n_estimators"], 
         scores1["mean_test_score"], 
         label="test accuracy")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
n_folds = 5
parameters = {'max_features' : [2,3,4,5,6,7,8,9]}
rf = RandomForestClassifier(max_depth = 4)
rf = GridSearchCV(rf,parameters, cv= n_folds,  scoring = 'accuracy', return_train_score=True)
rf.fit(x_train,y_train)
scores2 = rf.cv_results_
scores2 = pd.DataFrame(scores2)
scores2
plt.figure()
plt.plot(scores2['param_max_features'],
        scores2['mean_train_score'],
        label = 'train accuracy')
plt.plot(scores2['param_max_features'],
        scores2['mean_test_score'],
        label = 'test accuracy')
plt.xlabel('max_features')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


n_folds = 5

parameters = {'min_samples_leaf': range(100, 900, 50)}

rf = RandomForestClassifier()


rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy")
rf.fit(x_train, y_train)
scores3 = rf.cv_results_
scores3 = pd.DataFrame(scores3)
scores
plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


n_folds = 5

parameters = {'min_samples_split': range(200, 400, 50)}

rf = RandomForestClassifier()


rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy", return_train_score=True)
rf.fit(x_train, y_train)
scores4 = rf.cv_results_
scores4 = pd.DataFrame(scores4)
scores4
param_grid = {
    'max_depth': [2,3,4,5,6],
    'min_samples_leaf': range(100, 400, 50),
    'min_samples_split': range(200, 400, 100),
    'n_estimators': [100,200, 300], 
    'max_features': [3, 6]
}
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1,verbose = 1)
grid_search.fit(x_train, y_train)
print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=6,
                             min_samples_leaf=100, 
                             min_samples_split=200,
                             max_features=6,
                             n_estimators=100)
rfc.fit(x_train,y_train)
predictions = rfc.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test,predictions))

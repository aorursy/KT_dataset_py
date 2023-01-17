import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
matplotlib.style.use('fivethirtyeight')
df=pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
#checking the dataframe head
df.head()
df.info()
#Here we don't have any null values. Lets check the outlier values if any
sns.pairplot(df)
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)
df.corr()
plt.figure(figsize=(10,10))
for feature in (df.columns):
        plt.hist(df[feature])
        plt.title(feature)
        plt.show()
bins=(2,6.5,8)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)
from sklearn.preprocessing import  LabelEncoder
#for changing into 
label_quality = LabelEncoder()
df['quality'] = label_quality.fit_transform(df['quality'])
df.head()
plt.figure(figsize=(10,10))
for feature in (df.columns):
        sns.barplot(y=df[feature],x=df['quality'])
        plt.title(feature)
        plt.show()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
X=df.drop(['quality'],axis=1)
y=df['quality']
ms=StandardScaler()
X= ms.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=101)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
prediction=rf.predict(X_test)
print("Score on Train Set",rf.score(X_train,y_train))
print("Score on Test Set",rf.score(X_test,y_test))
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print("Classification Report on Random Forest\n",classification_report(y_test,prediction))
print("Classification Report on Random Forest\n",confusion_matrix(y_test,prediction))
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Criterian to select
criterion=['gini','entropy']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'criterion':criterion,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
randomforest=RandomForestClassifier()
from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator = randomforest, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=101, n_jobs = 1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
prediction=rf_random.predict(X_test)
print("Classification Report on Random Forest\n",confusion_matrix(y_test,prediction))
print("Classification Report on Random Forest\n",classification_report(y_test,prediction))
from xgboost import XGBClassifier
xg=XGBClassifier()
xg.fit(X_train,y_train)
prediction=xg.predict(X_test)
print("Classification Report on XGBoost\n",confusion_matrix(y_test,prediction))
print("Classification Report on XGBoost\n",classification_report(y_test,prediction))
from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)
prediction=svc.predict(X_test)
print("Classification Report on SVC\n",confusion_matrix(y_test,prediction))
print("Classification Report on SVC\n",classification_report(y_test,prediction))
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear','rbf','sigmoid']}  
sv=SVC()
rf_random = RandomizedSearchCV(estimator = sv, param_distributions = param_grid, n_iter = 100, cv = 5, verbose=2, random_state=101, n_jobs = 1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
prediction=rf_random.predict(X_test)
print("Classification Report on SVC\n",confusion_matrix(y_test,prediction))
print("Classification Report on SVC\n",classification_report(y_test,prediction))

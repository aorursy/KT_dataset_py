%%time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
%%time
df = pd.read_csv('../input/disease/liver.csv')
df.head()
%%time
df['Outcome']=df['Dataset']
df.drop('Dataset',axis=1,inplace=True)
df.head()
#check null values
%%time
df.isna().sum()
%%time
#fill null values with mean
df['Albumin_and_Globulin_Ratio']=df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean())
df.isna().sum()
# Perform one-hot encoding to categorical variables 
df=pd.concat([df,pd.get_dummies(df['Gender'], prefix = 'Gender')],axis=1)
X =df.drop(['Gender','Outcome'], axis=1)
X=StandardScaler().fit_transform(X)
y=df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
model = SVC()

parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5)
grid.fit(X_train, y_train)
roc_auc = np.around(np.mean(cross_val_score(grid, X_test, y_test, cv=5, scoring='roc_auc')), decimals=4)
print('Score: {}'.format(roc_auc))
model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)
predictions = cross_val_predict(model, X_test, y_test, cv=5)
print(classification_report(y_test, predictions))
score = np.around(np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc')),decimals=4)
print('Score: {}'.format(score))
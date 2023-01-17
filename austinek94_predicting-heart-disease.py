#Import the usual packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/heart-disease-uci/heart.csv')
df.head()
df.describe()
df.info()
df.shape
df['goal'] = df['target'] + 1
df['goal'] = df['goal'].replace(2,0)
df.head()
sns.boxplot(x=df['goal'],y=df['age'])
sns.swarmplot(x=df['goal'],y=df['age'],color='black',alpha=0.25)
sns.barplot(x=df['goal'],y=df['sex'])
sns.swarmplot(x=df['goal'],y=df['sex'],color='black',alpha=0.25)
sns.boxplot(x=df['goal'],y=df['trestbps'])
sns.swarmplot(x=df['goal'],y=df['trestbps'],color='black',alpha=0.25)
sns.heatmap(df.corr(),cmap='Blues')
corr_columns = ['exang','oldpeak','ca','thal']
for i in range(len(corr_columns)):
    fig, axs = plt.subplots(1,1)
    sns.barplot(x=df['goal'],y=df[corr_columns[i]])
    sns.swarmplot(x=df['goal'],y=df[corr_columns[i]],color='black',alpha=0.25)
    
#Import the usual data modeling packages
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from xgboost import XGBClassifier
X = df.drop(['target','goal'],axis=1)
y = df.goal.values

X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)
score=[]
lr = LogisticRegression()
lr.fit(X_train,y_train)

s1=np.mean(cross_val_score(lr,X_train,y_train,scoring='roc_auc',cv=5))
score.append(s1*100)
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

s2 = np.mean(cross_val_score(knn,X_train,y_train,scoring='roc_auc',cv=5))
score.append(s2*100)
xgb = XGBClassifier()
xgb.fit(X_train,y_train)

s3 = np.mean(cross_val_score(xgb,X_train,y_train,scoring='roc_auc',cv=5))
score.append(s3*100)
rf = RandomForestClassifier()
rf.fit(X_train,y_train)

s4 = np.mean(cross_val_score(rf,X_train,y_train,scoring='roc_auc',cv=5))
score.append(s4*100)
svc = svm.SVC()
svc.fit(X_train,y_train)

s5 =np.mean(cross_val_score(svc,X_train,y_train,scoring='roc_auc',cv=5))
score.append(s5*100)
labels = ['LogisticRegression', 'KNN', 'XGB', 'RandomForest', 'SVM']
for i in range(len(labels)):
    print('The score for',label[i],'is:',score[i])

parameters = {'n_estimators':range(10,300,10),'criterion':('gini', 'entropy'),'max_features':('auto','sqrt','log2')}
gs = GridSearchCV(rf,parameters,scoring='roc_auc',cv=5)
gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_estimator_)
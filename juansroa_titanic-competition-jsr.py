#Importings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Loading data
training=pd.read_csv('../input/titanic/train.csv')
#Exploring data shape and info
print(training.shape)
training.info()
sns.heatmap(training.isna())

#How the data looks
training.head(10)
#Exploring numerical features
print(training.columns)
numerical=['Age','SibSp','Parch','Fare']
training_numerical=training[numerical]
#Histograms
for i in training_numerical.columns:
  plt.figure(figsize=(5,5))
  plt.hist(training_numerical[i],color='green')
  plt.title=i
  plt.show()
# Mean, median, etc
print(training_numerical.describe())
# Correlation between variables
training_numerical.corr()
sns.heatmap(training_numerical.corr(),cmap="Greens")
#Pivot tables
pd.pivot_table(training,training_numerical,index='Survived')
#Exploring categorical variables
categorical=['Survived','Sex', 'Cabin', 'Embarked','Pclass']
training_categorical=training[categorical]
#Barplots
for i in training_categorical.columns:
  sns.barplot(training_categorical[i].value_counts().index,training_categorical[i].value_counts(),data=training_categorical)
  plt.show()
#Exploring survival rate acros the other categorical values
forpivot=['Sex','Embarked','Pclass']
for i in forpivot:
  print('\n\n',pd.pivot_table(training,values='PassengerId',index='Survived',columns=i,aggfunc='count'))

#Create a new feature relatives: sum of the SibSp+Parch
training['Relatives']=training.SibSp+training.Parch
#Create a variable that indicates the Cabin Classification
training['Cabin_type']=training['Cabin'].str[0]
training.Cabin_type.fillna('No Cabin',inplace=True)
print(training.Cabin_type.value_counts(sort=True))

#Fixing problem with Cabin T
training[training['Cabin_type']=='T']
training.drop(339,inplace=True)
#Create a feature that counts the number of cabins per passenger
training['CabinPerPassenger']=training.Cabin.apply(lambda x:0 if pd.isna(x) else len(x.split()))
training.CabinPerPassenger.value_counts()
training['CabinPerPassenger']=training['CabinPerPassenger'].astype('str')
training.info()
#Impute missing values with median for Age
training.Age.fillna(training.Age.median(),inplace=True)

#Delete missing values from Embarke (2 obs)
training.dropna(subset=['Embarked'],inplace=True)

#Normalize Fare
training['Fare_norm']=np.log(training.Fare+1)
plt.hist(training['Fare_norm'])
plt.show()
#Convert PClass to categorical
training['Pclass']=training['Pclass'].astype('str')

#Get dummy variables for categorical and create me X dataframe
X=pd.get_dummies(training[['Pclass','Fare_norm','SibSp','Parch','Age','Sex','Embarked','Relatives','Cabin_type','CabinPerPassenger']],drop_first=True)
X.dtypes

#Create y dataframe
y=training['Survived']
#Standard Scaler
scaler=StandardScaler()

X_scaled=X.copy()
X_scaled[['Fare_norm','SibSp','Parch','Age','Relatives']]=scaler.fit_transform(X_scaled[['Fare_norm','SibSp','Parch','Age','Relatives']])
X_scaled.shape
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state=42)
#Logistic Regression
logreg=LogisticRegression(max_iter=2500)
cv_logreg=cross_val_score(logreg,X_train,y_train,cv=5)
print(cv_logreg)
print(cv_logreg.mean())
#Support Vector Machine (SVM)
svm=SVC()
cv_svm=cross_val_score(svm,X_train,y_train,cv=5)
print(cv_svm)
print(cv_svm.mean())
#KNN
knn=KNeighborsClassifier()
cv_knn=cross_val_score(knn,X_train,y_train,cv=5)
print(cv_knn)
print(cv_knn.mean())
#Random Forests
rf=RandomForestClassifier(random_state=1)
cv_rf=cross_val_score(rf,X_train,y_train,cv=5)
print(cv_rf)
print(cv_rf.mean())
#Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=200, 
            max_depth=2,
            random_state=2)
cv_gb=cross_val_score(gb,X_train,y_train,cv=5)
print(cv_gb)
print(cv_gb.mean())
#Function to extract best score and best parameters for each model
def perfomance(model,model_name):
  print(model_name)
  print('Best score: '+str(model.best_score_))
  print('Best parameters: '+str(model.best_params_))

#Tuning KNN
neighbors=[x for x in range(5,31)]
weights=['uniform','distance']
p=[1,2]

knn=KNeighborsClassifier()
params:knn={'n_neighbors':neighbors,
            'weights':weights,
            'p':p}

cv_knn=GridSearchCV(estimator=knn,param_grid=params,cv=5,n_jobs=-1)
cv_knn.fit(X_train,y_train)
perfomance(cv_knn,'KNN')
best_knn=cv_knn.best_estimator_

best_knn.fit(X_train,y_train)
y_pred=best_knn.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print('KNN accuracy in test set: {:.3f}'.format(accuracy))

#Tuning SVM
svm=SVC(probability=True)
params_svm={'C':[0.02,0.5,1,10,100,1000],
            'gamma':[0.1,0.01,0.001]}

cv_svm=GridSearchCV(svm,params_svm,cv=5,n_jobs=-1)
cv_svm.fit(X_train,y_train)
perfomance(cv_svm,'SVM')
best_svm=cv_svm.best_estimator_

best_svm.fit(X_train,y_train)
y_pred=best_svm.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print('SVM accuracy in test set: {:.3f}'.format(accuracy))
#Tuning Logistic Regression
logreg=LogisticRegression()
params_logreg={'penalty':['l1','l2'],
               'C':[x for x in np.linspace(0,1)],
               'fit_intercept':[True,False]}
cv_logreg=GridSearchCV(logreg,params_logreg,cv=5,n_jobs=-1)
cv_logreg.fit(X_train,y_train)
perfomance(cv_logreg,'Logistic Regression')
best_logreg=cv_logreg.best_estimator_

best_logreg.fit(X_train,y_train)
y_pred=best_logreg.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print('Logistic Regression accuracy in test set: {:.3f}'.format(accuracy))
#Gradient Boosting Classifier
gb=GradientBoostingClassifier()
params_gb={'learning_rate':[x for x in np.linspace(0,0.2,8)],
           'n_estimators':[100,300],
           'max_depth':[x for x in range(1,6)]}
cv_gb=GridSearchCV(gb,params_gb,cv=5,n_jobs=-1)
cv_gb.fit(X_train,y_train)
perfomance(cv_gb,'Gradient Boosting')
best_gb=cv_gb.best_estimator_

best_gb.fit(X_train,y_train)
y_pred=best_gb.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print('Gradient Boosting accuracy in test set: {:.3f}'.format(accuracy))

#Tuning Random Forest
rf=RandomForestClassifier()
params_rf={'n_estimators':[100,200,300,400],
           'criterion':['gini','entropy'],
           'random_state':[1]}
cv_rf=GridSearchCV(rf,params_rf,cv=5,n_jobs=-1)
cv_rf.fit(X_train,y_train)
perfomance(cv_rf,'Random Forest')
best_rf=cv_rf.best_estimator_

best_rf.fit(X_train,y_train)
y_pred=best_rf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print('Random Forest accuracy in test set: {:.3f}'.format(accuracy))
#Ensemble learning using voting classifier
from sklearn.ensemble import VotingClassifier

classifiers=[('Logistic Regression',best_logreg),('Gradient Boosting',best_gb),('SVM',best_svm)]
vc_hard=VotingClassifier(estimators=classifiers)

vc_hard.fit(X_train,y_train)
y_pred=vc_hard.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print('Voting Classifier accuracy in test set: {:.3f}'.format(accuracy))

#Load test data
test=pd.read_csv('../input/titanic/test.csv')

#Evaluating Missing Values
sns.heatmap(test.isna())
#Impute NaN values in Age
test.Age.fillna(test.Age.median(),inplace=True)

#Drop NaN in fare
test.Fare.fillna(test.Fare.mean(),inplace=True)
#Check again Missing Values
sns.heatmap(test.isna())
#Normalize Fare
test['Fare_norm']=np.log(test.Fare+1)

#Convert PClass to categorical
test['Pclass']=test['Pclass'].astype('str')

#Create a new feature relatives: sum of the SibSp+Parch
test['Relatives']=test.SibSp+test.Parch


#Create a variable that indicates the Cabin Classification
test['Cabin_type']=test['Cabin'].str[0]
test.Cabin_type.fillna('No Cabin',inplace=True)

#Create a feature that counts the number of cabins per passenger
test['CabinPerPassenger']=test.Cabin.apply(lambda x:0 if pd.isna(x) else len(x.split()))
test['CabinPerPassenger']=test['CabinPerPassenger'].astype('str')

#Get dummy variables for categorical and create me X dataframe
test_final=pd.get_dummies(test[['Pclass','Fare_norm','SibSp','Parch','Age','Sex',
                                'Embarked','Relatives','Cabin_type','CabinPerPassenger']],drop_first=True)

test_final_scaled=test_final.copy()
test_final_scaled[['Fare_norm','SibSp','Parch','Age','Relatives']]=scaler.fit_transform(test_final_scaled[['Fare_norm','SibSp','Parch','Age','Relatives']])
#Fixing problem with cabin T
print(test_final_scaled.columns)
print(X_scaled.columns)
print(test.Cabin.unique())
print(training.Cabin.value_counts())

print(X_scaled.shape)
print(test_final_scaled.shape)
#Predictions
y_pred=vc_hard.predict(test_final_scaled)

#Add predictions to test
test['Survived']=y_pred
test.head()

#Extract Passenger ID and predictions
results=test[['PassengerId','Survived']]
results.shape

#Create Submision file
results.to_csv('submision.csv',index=False)
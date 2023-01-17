import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Plotting function

%matplotlib inline

import seaborn as sns #Plotting Aesthetics

sns.set_style('ticks')

import os #For importing the data



#For Accuracy and Prediction

from sklearn.metrics import confusion_matrix # For Machine Learning Accuracy Representation

import itertools

from sklearn import metrics 



#Cleaning Data

from sklearn.preprocessing import LabelEncoder



#Scaling Data

from sklearn.preprocessing import StandardScaler



#Feature Selection

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



#Model Selection and Data Splitting

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV





#Classifiers

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import RidgeClassifier





# For Importing data

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.info()

print('----------------------------------------')

test.info()
train.columns
train_new=train.drop(['PassengerId','Cabin','Ticket','Name'],axis=1)

test_new=test.drop(['PassengerId','Cabin','Ticket','Name'],axis=1)
train_new['Age'].fillna((train['Age'].mean()), inplace=True)

train_new['Embarked'].fillna('', inplace=True)

test_new['Age'].fillna((test['Age'].mean()), inplace=True)

test_new['Embarked'].fillna('', inplace=True)

test_new['Fare'].fillna(test['Fare'].mean(),inplace=True)
train_new.info()

print('----------------------------------------')

test_new.info()
#plot multiple histograms with Seaborn

survived = train_new[train.Survived == 0]

sns.distplot(train_new['Age'],  kde=False,color='blue', label='Survived')



not_survived = train_new[train.Survived == 1]

sns.distplot(not_survived['Age'], kde=False,color='red', label='Not Survived')



plt.legend(prop={'size': 12})

plt.title('Age of Titanic Passengers', fontsize=18)

plt.xlabel('Age', fontsize=16)

plt.ylabel('Count', fontsize=16)
sns.countplot(train_new['Survived'],palette="GnBu_d")

plt.title('Titanic Passenger Survival')
sns.countplot(train_new['Sex'],palette="GnBu_d")

plt.title('Sex of Titanic Passengers')
sns.countplot(train_new['Pclass'],hue=train_new['Survived'],palette="GnBu_d")

plt.title('Survival by Class of Titanic Passengers')
sns.countplot(train_new['Embarked'],hue=train_new['Pclass'],palette="GnBu_d")

plt.title('Embarked by Class of Titanic Passengers')
sns.countplot(train_new['Sex'],hue=train_new['Pclass'],palette="GnBu_d")

plt.title('Sex by Class of Titanic Passengers')
sns.scatterplot(train_new['Age'],train_new['Fare'],train_new['Sex'],palette=['red','blue'])

plt.title('Fare by Age and Sex')

plt.xlabel('Age')

plt.ylabel('Fare')

plt.show()
le=LabelEncoder()

train_new['Embarked_Code']=le.fit_transform(train_new['Embarked'])

test_new['Embarked_Code']=le.transform(test_new['Embarked'])

train_new['Sex_Code']=le.fit_transform(train_new['Sex'])

test_new['Sex_Code']=le.transform(test_new['Sex'])

train_new
train_new.info()
sns.heatmap(train_new.corr(),annot=True,cmap='Blues')
train_new.columns
train_accuracy= []

accuracy_list = []

algorithm = []



X_train, X_test, y_train, y_test = train_test_split(train_new[['Pclass','Age','SibSp','Parch', 'Fare','Embarked_Code', 'Sex_Code']],

                                                    train_new['Survived'],test_size=0.3, random_state=0)
scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)

X_test_scaled=scaler.transform(X_test)
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion Matrix',

                          cmap=plt.cm.BuGn):



    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()
Log_param={'C':[0.005,0.01,0.1,0.5,1],'solver':['lbfgs', 'liblinear', 'sag'],'max_iter':[100,500,1000],'multi_class':['ovr']}

Log_Reg=LogisticRegression()

Log_parm=GridSearchCV(Log_Reg, Log_param, cv=5)

Log_parm.fit(X_train_scaled, y_train)

y_reg=Log_parm.predict(X_test_scaled)

print("The best parameters are ",Log_parm.best_params_)

print("Train Accuracy {0:.3f}".format(Log_parm.score(X_train_scaled, y_train)))

print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_reg)))

cm = metrics.confusion_matrix(y_test, y_reg)

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm, classes=['Dead', 'Survived'],

                          title='Logistic Regression')

accuracy_list.append(metrics.accuracy_score(y_test, y_reg)*100)

train_accuracy.append(Log_parm.score(X_train_scaled, y_train))

algorithm.append('Logistic Regression')
SVC_param={'kernel':['sigmoid','rbf','poly'],'C':[0.005,0.01,0.1,0.5,1,1.25],'decision_function_shape':['ovr'],'random_state':[0]}

SVC_pol=SVC()

SVC_parm=GridSearchCV(SVC_pol, SVC_param, cv=5)

SVC_parm.fit(X_train_scaled, y_train)

y_pol=SVC_parm.predict(X_test_scaled)

print("The best parameters are ",SVC_parm.best_params_)

print("Train Accuracy {0:.3f}".format(SVC_parm.score(X_train_scaled, y_train)))

print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_pol)))

cm = metrics.confusion_matrix(y_test, y_pol)

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm, classes=['Dead', 'Survived'],

                          title='SVM')

train_accuracy.append(SVC_parm.score(X_train_scaled, y_train))

accuracy_list.append(metrics.accuracy_score(y_test, y_pol)*100)

algorithm.append('SVM')
error = []

# Calculating error for K values between 1 and 40

for i in range(1, 40):

    K_NN =KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

                     metric_params=None, n_jobs=None, n_neighbors=i, p=2,

                     weights='uniform')

    K_NN.fit(X_train_scaled, y_train)

    pred_i = K_NN.predict(X_test_scaled)

    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))

plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',

         markerfacecolor='blue', markersize=10)

plt.title('Error Rate K Value')

plt.xlabel('K Value')

plt.ylabel('Mean Error')
K_NN =KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

                     metric_params=None, n_jobs=None, n_neighbors=14, p=2,

                     weights='uniform')

K_NN.fit(X_train_scaled, y_train)

y_KNN=K_NN.predict(X_test_scaled)

print("Train Accuracy {0:.3f}".format(K_NN.score(X_train_scaled, y_train)))

print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_KNN)))

cm = metrics.confusion_matrix(y_test, y_KNN)

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm, classes=['Dead', 'Survived'],

                          title='KNN')

train_accuracy.append(K_NN.score(X_train_scaled, y_train))

accuracy_list.append(metrics.accuracy_score(y_test, y_KNN)*100)

algorithm.append('KNN')
RFC_param={'max_depth':[1,2,3,4,5],'n_estimators':[10,25,50,100,150],'criterion':['entropy','gini'],

           'ccp_alpha':[0,0.01,0.1],'max_features':[0.5,'auto']}

RFC=RandomForestClassifier()

RFC_parm=GridSearchCV(RFC, RFC_param, cv=5)

RFC_parm.fit(X_train_scaled, y_train)

y_RFC=RFC_parm.predict(X_test_scaled)

print("The best parameters are ",RFC_parm.best_params_)

print("Train Accuracy {0:.3f}".format(RFC_parm.score(X_train_scaled, y_train)))

print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_RFC)))

cm = metrics.confusion_matrix(y_test, y_RFC)

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm, classes=['Dead', 'Survived'],

                          title='RFC')

train_accuracy.append(RFC_parm.score(X_train_scaled, y_train))

accuracy_list.append(metrics.accuracy_score(y_test, y_RFC)*100)

algorithm.append('Random Forest')
GBC_parma={'loss':['deviance','exponential'],'n_estimators':[10,25,50,100,150],'learning_rate':[0.1,0.25, 0.5, 0.75],

          'criterion':['friedman_mse'], 'max_features':[None],'max_depth':[1,2,3,4,5,10],'ccp_alpha':[0,0.01,0.1]}

GBC = GradientBoostingClassifier()

GBC_parm=GridSearchCV(GBC, GBC_parma, cv=5)

GBC_parm.fit(X_train_scaled, y_train)

y_GBC=GBC_parm.predict(X_test_scaled)

print("The best parameters are ",GBC_parm.best_params_)

print("Train Accuracy {0:.3f}".format(GBC_parm.score(X_train_scaled, y_train)))

print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_GBC)))

cm = metrics.confusion_matrix(y_test, y_GBC)

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm, classes=['Dead', 'Survived'],

                          title='GBC')

train_accuracy.append(GBC_parm.score(X_train_scaled, y_train))

accuracy_list.append(metrics.accuracy_score(y_test, y_GBC)*100)

algorithm.append('GBC')
RC_parma={'solver':['svd','lsqr','cholesky'],'alpha':[0,0.5,0.75,1,1.5,2],'normalize':[True,False]}

RC=RidgeClassifier()

RC_parm=GridSearchCV(RC, RC_parma, cv=5)

RC_parm.fit(X_train_scaled, y_train)

y_RC=RC_parm.predict(X_test_scaled)

print("The best parameters are ",RC_parm.best_params_)

print("Train Accuracy {0:.3f}".format(RC_parm.score(X_train_scaled, y_train)))

print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_RC)))

cm = metrics.confusion_matrix(y_test, y_RC)

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm, classes=['Dead', 'Survived'],

                          title='Ridge Classifier')

train_accuracy.append(RC_parm.score(X_train_scaled, y_train))

accuracy_list.append(metrics.accuracy_score(y_test, y_RC)*100)

algorithm.append('Ridge Classifier')
X_feat=train_new[['Pclass','Age', 'SibSp', 'Parch', 'Fare','Embarked_Code', 'Sex_Code']]

y_feat=train_new['Survived']
#Feature Selection

bestfeatures = SelectKBest(score_func=chi2, k=5)

fit = bestfeatures.fit(X_feat,y_feat)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X_feat.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Feature','Score']  #naming the dataframe columns

print(featureScores.nlargest(3,'Score'))  #print 10 best features
train_accuracy_fet= []

accuracy_list_fet = []

algorithm_fet = []

X_train_fet, X_test_fet, y_train_fet, y_test_Fet = train_test_split(train_new[['Pclass', 'Fare', 'Sex_Code']],

                                                    train_new['Survived'],test_size=0.3, random_state=0)
X_train_fet_scaled=scaler.fit_transform(X_train_fet)

X_test_fet_scaled=scaler.transform(X_test_fet)
Log_param={'C':[0.005,0.01,0.1,0.5,1],'solver':['lbfgs', 'liblinear', 'sag'],'max_iter':[100,500,1000],'multi_class':['ovr']}

Log_Reg=LogisticRegression()

Log_parm=GridSearchCV(Log_Reg, Log_param, cv=5)

Log_parm.fit(X_train_fet_scaled, y_train)

y_reg=Log_parm.predict(X_test_fet_scaled)

print("The best parameters are ",Log_parm.best_params_)

print("Train Accuracy {0:.3f}".format(Log_parm.score(X_train_fet_scaled, y_train)))

print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_reg)))

cm = metrics.confusion_matrix(y_test, y_reg)

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm, classes=['Dead', 'Survived'],

                          title='Logistic Regression')

accuracy_list_fet.append(metrics.accuracy_score(y_test, y_reg)*100)

train_accuracy_fet.append(Log_parm.score(X_train_fet_scaled, y_train))

algorithm_fet.append('Logistic Regression')
SVC_param={'kernel':['sigmoid','rbf','poly'],'C':[0.005,0.01,0.1,0.5,1,1.25],'decision_function_shape':['ovr'],'random_state':[0]}

SVC_pol=SVC()

SVC_parm=GridSearchCV(SVC_pol, SVC_param, cv=5)

SVC_parm.fit(X_train_fet_scaled, y_train)

y_pol=SVC_parm.predict(X_test_fet_scaled)

print("The best parameters are ",SVC_parm.best_params_)

print("Train Accuracy {0:.3f}".format(SVC_parm.score(X_train_fet_scaled, y_train)))

print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_pol)))

cm = metrics.confusion_matrix(y_test, y_pol)

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm, classes=['Dead', 'Survived'],

                          title='SVM')

train_accuracy_fet.append(SVC_parm.score(X_train_fet_scaled, y_train))

accuracy_list_fet.append(metrics.accuracy_score(y_test, y_pol)*100)

algorithm_fet.append('SVM')
error = []

# Calculating error for K values between 1 and 40

for i in range(1, 40):

    K_NN =KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

                     metric_params=None, n_jobs=None, n_neighbors=i, p=2,

                     weights='distance')

    K_NN.fit(X_train_fet_scaled, y_train)

    pred_i = K_NN.predict(X_test_fet_scaled)

    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))

plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',

         markerfacecolor='blue', markersize=10)

plt.title('Error Rate K Value')

plt.xlabel('K Value')

plt.ylabel('Mean Error')
K_NN =KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

                     metric_params=None, n_jobs=None, n_neighbors=12, p=2,

                     weights='distance')

K_NN.fit(X_train_fet_scaled, y_train)

y_KNN=K_NN.predict(X_test_fet_scaled)

print("Train Accuracy {0:.3f}".format(K_NN.score(X_train_fet_scaled, y_train)))

print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_KNN)))

cm = metrics.confusion_matrix(y_test, y_KNN)

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm, classes=['Dead', 'Survived'],

                          title='KNN')

train_accuracy_fet.append(K_NN.score(X_train_fet_scaled, y_train))

accuracy_list_fet.append(metrics.accuracy_score(y_test, y_KNN)*100)

algorithm_fet.append('KNN')
RFC_param={'max_depth':[1,2,3,4,5],'n_estimators':[10,25,50,100,150,200],'criterion':['entropy','gini'],

           'ccp_alpha':[0,0.01,0.1],'max_features':[0.5,'auto']}

RFC=RandomForestClassifier()

RFC_parm=GridSearchCV(RFC, RFC_param, cv=5)

RFC_parm.fit(X_train_fet_scaled, y_train)

y_RFC=RFC_parm.predict(X_test_fet_scaled)

print("The best parameters are ",RFC_parm.best_params_)

print("Train Accuracy {0:.3f}".format(RFC_parm.score(X_train_fet_scaled, y_train)))

print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_RFC)))

cm = metrics.confusion_matrix(y_test, y_RFC)

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm, classes=['Dead', 'Survived'],

                          title='RFC')

train_accuracy_fet.append(RFC_parm.score(X_train_fet_scaled, y_train))

accuracy_list_fet.append(metrics.accuracy_score(y_test, y_RFC)*100)

algorithm_fet.append('Random Forest')
GBC_parma={'loss':['deviance','exponential'],'n_estimators':[10,25,50,100,150],'learning_rate':[0.1,0.25, 0.5, 0.75],

          'criterion':['friedman_mse'], 'max_features':[None],'max_depth':[1,2,3,4,5,10],'ccp_alpha':[0,0.01,0.1]}

GBC = GradientBoostingClassifier()

GBC_parm=GridSearchCV(GBC, GBC_parma, cv=5)

GBC_parm.fit(X_train_fet_scaled, y_train)

y_GBC=GBC_parm.predict(X_test_fet_scaled)

print("The best parameters are ",GBC_parm.best_params_)

print("Train Accuracy {0:.3f}".format(GBC_parm.score(X_train_fet_scaled, y_train)))

print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_GBC)))

cm = metrics.confusion_matrix(y_test, y_GBC)

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm, classes=['Dead', 'Survived'],

                          title='GBC')

train_accuracy_fet.append(GBC_parm.score(X_train_fet_scaled, y_train))

accuracy_list_fet.append(metrics.accuracy_score(y_test, y_GBC)*100)

algorithm_fet.append('GBC')
RC_parma={'solver':['svd','lsqr','cholesky'],'alpha':[0,0.5,0.75,1,1.5,2],'normalize':[True,False]}

RC=RidgeClassifier()

RC_parm=GridSearchCV(RC, RC_parma, cv=5)

RC_parm.fit(X_train_fet_scaled, y_train)

y_RC=RC_parm.predict(X_test_fet_scaled)

print("The best parameters are ",RC_parm.best_params_)

print("Train Accuracy {0:.3f}".format(RC_parm.score(X_train_fet_scaled, y_train)))

print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_RC)))

cm = metrics.confusion_matrix(y_test, y_RC)

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm, classes=['Dead', 'Survived'],

                          title='Ridge Classifier')

train_accuracy_fet.append(RC_parm.score(X_train_fet_scaled, y_train))

accuracy_list_fet.append(metrics.accuracy_score(y_test, y_RC)*100)

algorithm_fet.append('Ridge Classifier')
f,ax = plt.subplots(figsize = (10,5))

sns.barplot(x=accuracy_list,y=algorithm,palette = sns.dark_palette("blue",len(accuracy_list)))

plt.xlabel("Accuracy")

plt.ylabel("Algorithm")

plt.title('Algorithm Test Accuracy No Feature Selection')

plt.show()
f,ax = plt.subplots(figsize = (10,5))

sns.barplot(x=accuracy_list_fet,y=algorithm_fet,palette = sns.dark_palette("blue",len(accuracy_list)))

plt.xlabel("Accuracy")

plt.ylabel("Algorithm")

plt.title('Algorithm Test Accuracy Feature Selection')

plt.show()
#Setting test data up

Testing_set=test_new[['Pclass', 'Fare', 'Sex_Code']]

Testing_set_scaled=scaler.transform(Testing_set)
#Predicting and Submission

ID = test['PassengerId']

test_prediction = K_NN.predict(Testing_set_scaled)



Submission = pd.DataFrame({ 'PassengerId' : ID, 'Survived': test_prediction })

Submission.to_csv('submission.csv', index=False)
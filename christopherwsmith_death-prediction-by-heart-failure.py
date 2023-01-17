#Simple Data processing

import numpy as np #linear algebra

import pandas as pd # data processing, .csv load



#Feature Selection

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



#Data Visualization

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from matplotlib.ticker import NullFormatter

import matplotlib.ticker as ticker

import itertools #For Confusion Matrix

%matplotlib inline

import seaborn as sns



# Scaling

from sklearn import preprocessing #For data normalization



# Model Selection

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV # For parameterization and splitting data

from sklearn.metrics import confusion_matrix

from sklearn import metrics # For Accuracy



#Classification Algorithms

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import RidgeClassifier
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
heart=pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

heart
heart.describe()
heart.info()
print(heart.columns.unique)
#Separating the data to asses with feature selection 

X_feat=heart[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',

       'ejection_fraction', 'high_blood_pressure', 'platelets',

       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']]

y_feat=heart['DEATH_EVENT']
#Feature Selection

bestfeatures = SelectKBest(score_func=chi2, k=5)

fit = bestfeatures.fit(X_feat,y_feat)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X_feat.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Factors','Score']  #naming the dataframe columns

print(featureScores.nlargest(5,'Score'))  #print 5 best features
train_accuracy= []

accuracy_list = []

algorithm = []



X_train,X_test,y_train,y_test = train_test_split(heart[['platelets','time','creatinine_phosphokinase','ejection_fraction','age']]

                                                 ,heart['DEATH_EVENT'],test_size=0.2, random_state=0)

print("X_train shape :",X_train.shape)

print("Y_train shape :",y_train.shape)

print("X_test shape :",X_test.shape)

print("Y_test shape :",y_test.shape)
scaler_ss=preprocessing.StandardScaler()
X_train_scaled=scaler_ss.fit_transform(X_train)

X_test_scaled=scaler_ss.transform(X_test)
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
Log_Reg=LogisticRegression(C=1, class_weight='balanced', dual=False,

                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,

                   max_iter=1000, multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=0, solver='lbfgs', tol=0.0001, verbose=0,

                   warm_start=False)

Log_Reg.fit(X_train_scaled, y_train)

y_reg=Log_Reg.predict(X_test_scaled)

print("Train Accuracy {0:.3f}".format(Log_Reg.score(X_train_scaled, y_train)))

print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_reg)))

cm = metrics.confusion_matrix(y_test, y_reg)

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm, classes=['Alive', 'Death'],

                          title='Logistic Regression')

accuracy_list.append(metrics.accuracy_score(y_test, y_reg)*100)

train_accuracy.append(Log_Reg.score(X_train_scaled, y_train))

algorithm.append('Logistic Regression')
SVC_param={'kernel':['sigmoid','rbf','poly'],'C':[1],'decision_function_shape':['ovr'],'random_state':[0]}

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

plot_confusion_matrix(cm, classes=['Alive', 'Death'],

                          title='SVM')

train_accuracy.append(SVC_parm.score(X_train_scaled, y_train))

accuracy_list.append(metrics.accuracy_score(y_test, y_pol)*100)

algorithm.append('SVM')
error = []

# Calculating error for K values between 1 and 40

for i in range(1, 40):

    K_NN =KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

                     metric_params=None, n_jobs=None, n_neighbors=i, p=2,

                     weights='distance')

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

                     metric_params=None, n_jobs=None, n_neighbors=2, p=2,

                     weights='distance')

K_NN.fit(X_train_scaled, y_train)

y_KNN=K_NN.predict(X_test_scaled)

print("Train Accuracy {0:.3f}".format(K_NN.score(X_train_scaled, y_train)))

print('Test Accuracy' "{0:.3f}".format(metrics.accuracy_score(y_test, y_KNN)))

cm = metrics.confusion_matrix(y_test, y_KNN)

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm, classes=['Alive', 'Death'],

                          title='KNN')

train_accuracy.append(K_NN.score(X_train_scaled, y_train))

accuracy_list.append(metrics.accuracy_score(y_test, y_KNN)*100)

algorithm.append('KNN')
RFC_param={'max_depth':[1,2,3,4,5],'n_estimators':[10,25,50,100,150],'random_state':[None],

           'criterion':['entropy','gini'],'max_features':[0.5]}

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

plot_confusion_matrix(cm, classes=['Alive', 'Death'],

                          title='RFC')

train_accuracy.append(RFC_parm.score(X_train_scaled, y_train))

accuracy_list.append(metrics.accuracy_score(y_test, y_RFC)*100)

algorithm.append('Random Forest')
GBC_parma={'loss':['deviance','exponential'],'n_estimators':[10,25,50,100,150],'learning_rate':[0.1,0.25, 0.5, 0.75],

          'criterion':['friedman_mse'], 'max_features':[None],'max_depth':[1,2,3,4,5,10],'random_state':[None]}

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

plot_confusion_matrix(cm, classes=['Alive', 'Death'],

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

plot_confusion_matrix(cm, classes=['Alive', 'Death'],

                          title='Ridge Classifier')

train_accuracy.append(RC_parm.score(X_train_scaled, y_train))

accuracy_list.append(metrics.accuracy_score(y_test, y_RC)*100)

algorithm.append('Ridge Classifier')
#Train Accuracy

f,ax = plt.subplots(figsize = (10,5))

sns.barplot(x=train_accuracy,y=algorithm,palette = sns.dark_palette("blue",len(accuracy_list)))

plt.xlabel("Accuracy")

plt.ylabel("Algorithm")

plt.title('Algorithm Train Accuracy')

plt.show()
#Testing Accuracy

f,ax = plt.subplots(figsize = (10,5))

sns.barplot(x=accuracy_list,y=algorithm,palette = sns.dark_palette("blue",len(accuracy_list)))

plt.xlabel("Accuracy")

plt.ylabel("Algorithm")

plt.title('Algorithm Test Accuracy')

plt.show()
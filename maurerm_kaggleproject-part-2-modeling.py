#Data Analysis Libraries

import numpy as np

import pandas as pd

from pandas import Series,DataFrame

from scipy import stats

import statistics as st

import math

import os

from datetime import datetime



#Visualization Libraries

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)

from IPython.display import HTML

from IPython.display import display

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected = True)

%matplotlib inline

from IPython.core.display import HTML

def multi_table(table_list):

    ''' Acceps a list of IpyTable objects and returns a table which contains each IpyTable in a cell

    '''

    return HTML(

        '<table><tr style="background-color:white;">' + 

        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +

        '</tr></table>'

    )



#sklearn

from sklearn import metrics

from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score,confusion_matrix, classification_report, confusion_matrix, jaccard_similarity_score, f1_score, fbeta_score



from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler, Imputer,MinMaxScaler



from sklearn import model_selection

from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score, validation_curve, RandomizedSearchCV, cross_val_predict



from sklearn import linear_model

from sklearn.linear_model import LogisticRegression, LinearRegression



from sklearn import naive_bayes

from sklearn.naive_bayes import GaussianNB



from sklearn import neighbors

from sklearn.neighbors import KNeighborsClassifier



from sklearn import tree

from sklearn.tree import DecisionTreeClassifier



from sklearn import ensemble

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostRegressor



from sklearn.svm import SVC



from sklearn.ensemble import AdaBoostClassifier



from sklearn import datasets



#misc

from functools import singledispatch

import eli5

from eli5.sklearn import PermutationImportance

import shap

from mpl_toolkits.mplot3d import Axes3D

import os

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input"))

import itertools

from xgboost import XGBClassifier



heart = pd.read_csv("../input/heart.csv")

heart2= heart.drop(heart.index[164])



heart2.columns=['age', 'sex', 'cpain','resting_BP', 'chol', 'fasting_BS', 'resting_EKG', 

                'max_HR', 'exercise_ANG', 'ST_depression', 'm_exercise_ST', 'no_maj_vessels', 'thal', 'target']



heart2['chol']=heart2['chol'].replace([417, 564], 240)

heart2['chol']=heart2['chol'].replace([407, 409], 249)



heart2['ST_depressionAB']=heart2['ST_depression'].apply(lambda row: 1 if row > 0 else 0)

heart2A=heart2.iloc[:,0:11]

heart2B=heart2.iloc[:,11:14]

heart2C=heart2.loc[:,'ST_depressionAB']

heart2C=pd.DataFrame(heart2C)

heart2C.head()

heart2 = pd.concat([heart2A, heart2C, heart2B], axis=1, join_axes=[heart2A.index])



heart2.loc[48, 'thal']=2.0

heart2.loc[281, 'thal']=3.0



#seperate independent (feature) and dependent (target) variables

#KNN cannot process text/ categorical data unless they are be converted to numbers

#For this reason I did not input the heart3 DataFrame created above

X=heart2.drop('target',1)

y=heart2.loc[:,'target']



#Scale the data

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



#Split the data into training and testing sets

X_train,X_test,y_train,y_test = train_test_split(X_scaled, y,test_size=.2,random_state=40)



#Call classifier and, using GridSearchCV, find the best parameters

knn = KNeighborsClassifier()

params = {'n_neighbors':[i for i in range(1,33,2)]}

modelKNN = GridSearchCV(knn,params,cv=10)

modelKNN.fit(X_train,y_train)

modelKNN.best_params_   



#Use the above model (modelKNN) to predict the y values corresponding to the X testing set

predictKNN = modelKNN.predict(X_test)



#Compare the results of the model's predictions (predictKNN) to the actual y values

accscoreKNN=accuracy_score(y_test,predictKNN)



import warnings



def fxn():

    warnings.warn("deprecated", DeprecationWarning)



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    fxn()
conf_matrixKNN = confusion_matrix(y_test,predictKNN)



print('Confusion Matrix:\n{}\n'.format(conf_matrixKNN))

print('True Positive:\t{}'.format(conf_matrixKNN[1,1]))

print('True Negative:\t{}'.format(conf_matrixKNN[0,0]))

print('False Positive:\t{}'.format(conf_matrixKNN[0,1]))

print('False Negative:\t{}'.format(conf_matrixKNN[1,0]))



class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(conf_matrixKNN), annot = True, cmap = 'YlGnBu',

           fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for K-Nearest Neighbors Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
total=sum(sum(conf_matrixKNN))



sensitivityKNN = conf_matrixKNN[1,1]/(conf_matrixKNN[1,1]+conf_matrixKNN[1,0])

print('Sensitivity : ', sensitivityKNN)



specificityKNN = conf_matrixKNN[0,0]/(conf_matrixKNN[0,0]+conf_matrixKNN[0,1])

print('Specificity : ', specificityKNN)
KNNF1=metrics.f1_score(y_test,predictKNN)

print(classification_report(y_test,predictKNN))

print("F1 Score:" ,round((100*KNNF1),2))
predictKNN_quant = modelKNN.predict_proba(X_test)[:, 1]



fprKNN, tprKNN, thresholds = roc_curve(y_test, predictKNN_quant)



fig, ax = plt.subplots()

ax.plot(fprKNN, tprKNN)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC Curve for Heart Disease Classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)



aucKNN=auc(fprKNN, tprKNN)

print("Area under the curve:", aucKNN)
KNNscores = cross_val_score(modelKNN, X_scaled, y, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (KNNscores.mean(), KNNscores.std() * 2))

print ('Cross_validated scores:', KNNscores)
KNNR2 = metrics.r2_score(y_test,predictKNN)

print ('R2:', KNNR2)
#Data Analysis Libraries

import numpy as np

import pandas as pd

from pandas import Series,DataFrame

from scipy import stats

import statistics as st

import math

import os

from datetime import datetime



#Visualization Libraries

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)

from IPython.display import HTML

from IPython.display import display

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected = True)

%matplotlib inline

from IPython.core.display import HTML

def multi_table(table_list):

    ''' Acceps a list of IpyTable objects and returns a table which contains each IpyTable in a cell

    '''

    return HTML(

        '<table><tr style="background-color:white;">' + 

        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +

        '</tr></table>'

    )



#sklearn

from sklearn import metrics

from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score,confusion_matrix, classification_report, confusion_matrix, jaccard_similarity_score, f1_score, fbeta_score



from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler, Imputer,MinMaxScaler



from sklearn import model_selection

from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score, validation_curve, RandomizedSearchCV, cross_val_predict



from sklearn import linear_model

from sklearn.linear_model import LogisticRegression, LinearRegression



from sklearn import naive_bayes

from sklearn.naive_bayes import GaussianNB



from sklearn import neighbors

from sklearn.neighbors import KNeighborsClassifier



from sklearn import tree

from sklearn.tree import DecisionTreeClassifier



from sklearn import ensemble

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostRegressor



from sklearn.svm import SVC



from sklearn.ensemble import AdaBoostClassifier



from sklearn import datasets



#misc

from functools import singledispatch

import eli5

from eli5.sklearn import PermutationImportance

import shap

from mpl_toolkits.mplot3d import Axes3D

import os

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input"))

import itertools

from xgboost import XGBClassifier
heart = pd.read_csv("../input/heart.csv")
heart2= heart.drop(heart.index[164])



heart2.columns=['age', 'sex', 'cpain','resting_BP', 'chol', 'fasting_BS', 'resting_EKG', 

                'max_HR', 'exercise_ANG', 'ST_depression', 'm_exercise_ST', 'no_maj_vessels', 'thal', 'target']



heart2['chol']=heart2['chol'].replace([417, 564], 240)

heart2['chol']=heart2['chol'].replace([407, 409], 249)



heart2['ST_depressionAB']=heart2['ST_depression'].apply(lambda row: 1 if row > 0 else 0)

heart2A=heart2.iloc[:,0:11]

heart2B=heart2.iloc[:,11:14]

heart2C=heart2.loc[:,'ST_depressionAB']

heart2C=pd.DataFrame(heart2C)

heart2C.head()

heart2 = pd.concat([heart2A, heart2C, heart2B], axis=1, join_axes=[heart2A.index])



heart2.loc[48, 'thal']=2.0

heart2.loc[281, 'thal']=3.0
PHD=heart2.loc[heart2.loc[:,"target"]==1]

AHD=heart2.loc[heart2.loc[:,"target"]==0]



from scipy.stats import ttest_ind

def rowz(ttest): 

    name=ttest_ind(PHD[ttest], AHD[ttest])

    name=list(name)

    name = pd.DataFrame(np.array(name))

    name=name.T

    col=["t-statistic", "p_value"]

    name.columns=col

    return name



AGE=rowz('age')

AGE.loc[:,"Names"]="Age"

RESTING_BP=rowz('resting_BP')

RESTING_BP.loc[:,"Names"]="Resting_BP"

CHOLESTEROL=rowz('chol')

CHOLESTEROL.loc[:,"Names"]="Cholesterol"

MAX_HR=rowz('max_HR')

MAX_HR.loc[:,"Names"]="Max_HR"

ST_DEP=rowz('ST_depression')

ST_DEP.loc[:,"Names"]="ST_Depression"



PVALS = pd.concat([AGE, RESTING_BP,CHOLESTEROL,MAX_HR, ST_DEP], axis=0)

PVALS=PVALS.set_index(PVALS["Names"])

P_VALS= PVALS.drop('Names',axis=1)



P_VALS
X=heart2.drop('target',1)

y=heart2.loc[:,'target']
X.head()
y.head()
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=.3,random_state=40)
print ('Train set:  ', X_train.shape,  y_train.shape)

print ('Test set:   ', X_test.shape,  y_test.shape)
#seperate independent (feature) and dependent (target) variables

#KNN cannot process text/ categorical data unless they are be converted to numbers

#For this reason I did not input the heart3 DataFrame created above

X=heart2.drop('target',1)

y=heart2.loc[:,'target']



#Scale the data

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



#Split the data into training and testing sets

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=.2,random_state=40)



#Use the above model (modelKNN) to predict the y values corresponding to the X testing set

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

y_predictLR = LR.predict(X_test)



#Compare the results of the model's predictions (predictKNN) to the actual y values

accscoreLR=accuracy_score(y_test,y_predictLR)

print('Using Logistic Regression we get an accuracy score of: ',

      round(accuracy_score(y_test,y_predictLR),5)*100,'%')
conf_matrixLR = confusion_matrix(y_test, y_predictLR)



print('Confusion Matrix:\n{}\n'.format(conf_matrixLR))

print('True Positive:\t{}'.format(conf_matrixLR[1,1]))

print('True Negative:\t{}'.format(conf_matrixLR[0,0]))

print('False Positive:\t{}'.format(conf_matrixLR[0,1]))

print('False Negative:\t{}'.format(conf_matrixLR[1,0]))



class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(conf_matrixLR), annot = True, cmap = 'YlGnBu',

           fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for Logistic Regression Model Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
total=sum(sum(conf_matrixLR))



specificityLR = conf_matrixLR[0,0]/(conf_matrixLR[0,0]+conf_matrixLR[1,0])

print('Specificity : ', specificityLR)



sensitivityLR = conf_matrixLR[1,1]/(conf_matrixLR[1,1]+conf_matrixLR[0,1])

print('Sensitivity : ', sensitivityLR )
y_pred_quantLR = LR.predict_proba(X_test)[:, 1]



fprLR, tprLR, thresholdsLR = roc_curve(y_test, y_pred_quantLR)



fig, ax = plt.subplots()

ax.plot(fprLR, tprLR)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC Curve for Heart Disease Classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
aucLR=auc(fprLR, tprLR)

print("Area under the curve:", aucLR)
LRF1=metrics.f1_score(y_test,y_predictLR)

print(classification_report(y_test,y_predictLR))
LRscores = cross_val_score(LR, X_scaled, y, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (LRscores.mean(), LRscores.std() * 2))



LRR2 = metrics.r2_score(y_test,y_predictLR)

print ('R2:', LRR2)
#seperate independent (feature) and dependent (target) variables

#KNN cannot process text/ categorical data unless they are be converted to numbers

#For this reason I did not input the heart3 DataFrame created above

X=heart2.drop('target',1)

y=heart2.loc[:,'target']



#Scale the data

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



#Split the data into training and testing sets

X_train,X_test,y_train,y_test = train_test_split(X_scaled, y,test_size=.2,random_state=40)



#Call classifier and, using GridSearchCV, find the best parameters

knn = KNeighborsClassifier()

params = {'n_neighbors':[i for i in range(1,33,2)]}

modelKNN = GridSearchCV(knn,params,cv=10)

modelKNN.fit(X_train,y_train)

modelKNN.best_params_   



#Use the above model (modelKNN) to predict the y values corresponding to the X testing set

predictKNN = modelKNN.predict(X_test)



#Compare the results of the model's predictions (predictKNN) to the actual y values

accscoreKNN=accuracy_score(y_test,predictKNN)

print('Accuracy Score: ',accuracy_score(y_test,predictKNN))

print('Using k-NN we get an accuracy score of: ',

      round(accuracy_score(y_test,predictKNN),5)*100,'%')

conf_matrixKNN = confusion_matrix(y_test,predictKNN)



print('Confusion Matrix:\n{}\n'.format(conf_matrixKNN))

print('True Positive:\t{}'.format(conf_matrixKNN[1,1]))

print('True Negative:\t{}'.format(conf_matrixKNN[0,0]))

print('False Positive:\t{}'.format(conf_matrixKNN[0,1]))

print('False Negative:\t{}'.format(conf_matrixKNN[1,0]))



class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(conf_matrixKNN), annot = True, cmap = 'YlGnBu',

           fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for K-Nearest Neighbors Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
total=sum(sum(conf_matrixKNN))



specificityKNN = conf_matrixKNN[0,0]/(conf_matrixKNN[0,0]+conf_matrixKNN[1,0])

print('Specificity : ', specificityKNN)



sensitivityKNN = conf_matrixKNN[1,1]/(conf_matrixKNN[1,1]+conf_matrixKNN[0,1])

print('Sensitivity : ', sensitivityKNN )
predictKNN_quant = modelKNN.predict_proba(X_test)[:, 1]



fprKNN, tprKNN, thresholds = roc_curve(y_test, predictKNN_quant)



fig, ax = plt.subplots()

ax.plot(fprKNN, tprKNN)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC Curve for Heart Disease Classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
aucKNN=auc(fprKNN, tprKNN)

print("Area under the curve:", aucKNN)
KNNF1=metrics.f1_score(y_test,predictKNN)

print(classification_report(y_test,predictKNN))
KNNscores = cross_val_score(modelKNN, X_scaled, y, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (KNNscores.mean(), KNNscores.std() * 2))



KNNR2 = metrics.r2_score(y_test,predictKNN)

print ('R2:', KNNR2)
X=heart2.drop('target',1)

y=heart2.loc[:,'target']



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled, y,test_size=.3,random_state=40)



dtree= DecisionTreeClassifier()

params = {'max_features': ['auto', 'sqrt', 'log2'],

          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 

          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11]}#



tree_model = GridSearchCV(dtree, param_grid=params, n_jobs=-1)

tree_model.fit(X_train,y_train)



#Printing best parameters selected through GridSearchCV



tree_model.best_params_

predictTREE = tree_model.predict(X_test)



accsocreTREE=accuracy_score(y_test,predictTREE)

print('Accuracy Score: ',accuracy_score(y_test,predictTREE))

print('Using Decision Tree we get an accuracy score of: ',

      round(accuracy_score(y_test,predictTREE),5)*100,'%')
conf_matrixTREE = confusion_matrix(y_test, predictTREE)



print('Confusion Matrix:\n{}\n'.format(conf_matrixTREE))

print('True Positive:\t{}'.format(conf_matrixTREE[1,1]))

print('True Negative:\t{}'.format(conf_matrixTREE[0,0]))

print('False Positive:\t{}'.format(conf_matrixTREE[0,1]))

print('False Negative:\t{}'.format(conf_matrixTREE[1,0]))

class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(conf_matrixTREE), annot = True, cmap = 'YlGnBu',

           fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for Decision Tree Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
total=sum(sum(conf_matrixTREE))



specificityTREE = conf_matrixTREE[0,0]/(conf_matrixTREE[0,0]+conf_matrixTREE[1,0])

print('Specificity : ', specificityTREE)



sensitivityTREE = conf_matrixTREE[1,1]/(conf_matrixTREE[1,1]+conf_matrixTREE[0,1])

print('Sensitivity : ', sensitivityTREE )
predictTREE_quant = tree_model.predict_proba(X_test)[:, 1]



fprTREE, tprTREE, thresholds = roc_curve(y_test, predictTREE_quant)



fig, ax = plt.subplots()

ax.plot(fprTREE, tprTREE)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC Curve for Heart Disease Classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
aucTREE=auc(fprTREE, tprTREE)

print("Area under the curve:", aucTREE)
TREEF1=metrics.f1_score(y_test,predictTREE)

print(classification_report(y_test,predictTREE))
TREEscores = cross_val_score(tree_model, X_scaled, y, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (TREEscores.mean(), TREEscores.std() * 2))



TREER2 = metrics.r2_score(y_test,predictTREE)

print ('R2:', TREER2)
X= heart2.drop('target',1)

y= heart2['target']



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y, test_size=.3,random_state=40)



modelABC = AdaBoostClassifier(base_estimator=dtree)

param_dist = {

 'n_estimators': [i for i in range(1,100)],

 'learning_rate' : [0.01,0.05,0.1,0.3,1]

 }



ABC = RandomizedSearchCV(AdaBoostClassifier(),

 param_distributions = param_dist,

 cv=3,

 n_iter = 10,

 n_jobs=-1)



ABC.fit(X_train, y_train)

ABC.best_params_ 

y_predictABC = ABC.predict(X_test)





accscoreAB=accuracy_score(y_test,y_predictABC)

print('Using AdaBoost we get an accuracy score of: ',

      round(accuracy_score(y_test,y_predictABC),5)*100,'%')
conf_matrixAB = confusion_matrix(y_test, y_predictABC)



print('Confusion Matrix:\n{}\n'.format(conf_matrixAB))

print('True Positive:\t{}'.format(conf_matrixAB[1,1]))

print('True Negative:\t{}'.format(conf_matrixAB[0,0]))

print('False Positive:\t{}'.format(conf_matrixAB[0,1]))

print('False Negative:\t{}'.format(conf_matrixAB[1,0]))



class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(conf_matrixAB), annot = True, cmap = 'YlGnBu',

           fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for AdaBoost Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
total=sum(sum(conf_matrixAB))



specificityAB = conf_matrixAB[0,0]/(conf_matrixAB[0,0]+conf_matrixAB[1,0])

print('Specificity : ', specificityAB)



sensitivityAB = conf_matrixAB[1,1]/(conf_matrixAB[1,1]+conf_matrixAB[0,1])

print('Sensitivity : ', sensitivityAB )
y_predABC_quant = ABC.predict_proba(X_test)[:, 1]



fprABC, tprABC, thresholds = roc_curve(y_test, y_predABC_quant)



fig, ax = plt.subplots()

ax.plot(fprABC, tprABC)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC Curve for Heart Disease Classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
aucAB=auc(fprABC, tprABC)

print("Area under the curve:", aucAB)
ABF1=metrics.f1_score(y_test,y_predictABC)

print(classification_report(y_test,y_predictABC))
ABscores = cross_val_score(ABC, X_scaled, y, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (ABscores.mean(), ABscores.std() * 2))



ABR2 = metrics.r2_score(y_test,y_predictABC)

print ('R2:', ABR2)
X= heart2.drop('target',1)

y= heart2['target']



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y, test_size=.3,random_state=40)



clf=RandomForestClassifier()

params = {'n_estimators':[i for i in range(1,100)]}

modelRF = GridSearchCV(clf,params,cv=10)

modelRF.fit(X_train,y_train)

modelRF.best_params_   



y_predCLF=modelRF.predict(X_test)



accscoreRF=accuracy_score(y_test,y_predCLF)

print('Using Random Forest we get an accuracy score of: ',

      round(accuracy_score(y_test,y_predCLF),5)*100,'%')
#y_predCLFH=modelRFH.predict(H_test)

conf_matrixRF = confusion_matrix(y_test, y_predCLF)



print('Confusion Matrix:\n{}\n'.format(conf_matrixRF))

print('True Positive:\t{}'.format(conf_matrixRF[1,1]))

print('True Negative:\t{}'.format(conf_matrixRF[0,0]))

print('False Positive:\t{}'.format(conf_matrixRF[0,1]))

print('False Negative:\t{}'.format(conf_matrixRF[1,0]))



class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(conf_matrixRF), annot = True, cmap = 'YlGnBu',

           fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for Random Forest Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
total=sum(sum(conf_matrixRF))



specificityRF = conf_matrixRF[0,0]/(conf_matrixRF[0,0]+conf_matrixRF[1,0])

print('Specificity : ', specificityRF)



sensitivityRF = conf_matrixRF[1,1]/(conf_matrixRF[1,1]+conf_matrixRF[0,1])

print('Sensitivity : ', sensitivityRF)
y_predCLF_quant = modelRF.predict_proba(X_test)[:, 1]



fprRF, tprRF, thresholds = roc_curve(y_test, y_predCLF_quant)



fig, ax = plt.subplots()

ax.plot(fprRF, tprRF)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC Curve for Heart Disease Classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)

aucRF=auc(fprRF, tprRF)

print("Area under the curve:", aucRF)
RFF1=metrics.f1_score(y_test,y_predCLF)

print(classification_report(y_test,y_predCLF))
RFscores = cross_val_score(modelRF, X_scaled, y, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (RFscores.mean(), RFscores.std() * 2))



RFR2 = metrics.r2_score(y_test,y_predCLF)

print ('Cross-Predicted Accuracy:', RFR2)
X= heart2.drop('target',1)

y= heart2['target']



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y, test_size=.3,random_state=40)



param_dist = {

 'n_estimators': [i for i in range(1,100)],

 'learning_rate' : [0.01,0.05,0.1,0.3,1]

 }



XGB = RandomizedSearchCV(XGBClassifier(),

 param_distributions = param_dist,

 cv=3,

 n_iter = 10,

 n_jobs=-1)



XGB.fit(X_train, y_train)

XGB.best_params_ 

y_predictXGB = XGB.predict(X_test)





accscoreXGB=accuracy_score(y_test,y_predictXGB)

print('Using XGBoost we get an accuracy score of: ',

      round(accuracy_score(y_test,y_predictXGB),5)*100,'%')
conf_matrixXGB = confusion_matrix(y_test, y_predictXGB)



print('Confusion Matrix:\n{}\n'.format(conf_matrixXGB))

print('True Positive:\t{}'.format(conf_matrixXGB[1,1]))

print('True Negative:\t{}'.format(conf_matrixXGB[0,0]))

print('False Positive:\t{}'.format(conf_matrixXGB[0,1]))

print('False Negative:\t{}'.format(conf_matrixXGB[1,0]))



class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(conf_matrixXGB), annot = True, cmap = 'YlGnBu',

           fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for XGBoost Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
total=sum(sum(conf_matrixXGB))



specificityXGB = conf_matrixXGB[0,0]/(conf_matrixXGB[0,0]+conf_matrixXGB[1,0])

print('Specificity : ', specificityXGB)



sensitivityXGB = conf_matrixXGB[1,1]/(conf_matrixXGB[1,1]+conf_matrixXGB[0,1])

print('Sensitivity : ', sensitivityXGB)
y_predXGB_quant = XGB.predict_proba(X_test)[:, 1]



fprXGB, tprXGB, thresholds = roc_curve(y_test, y_predXGB_quant)



fig, ax = plt.subplots()

ax.plot(fprXGB, tprXGB)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC Curve for Heart Disease Classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
aucXGB=auc(fprXGB, tprXGB)

print("Area under the curve:", aucXGB)
XGF1=metrics.f1_score(y_test,y_predictXGB)

print(classification_report(y_test,y_predictXGB))
XGscores = cross_val_score(XGB, X_scaled, y, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (XGscores.mean(), XGscores.std() * 2))



XGR2 = metrics.r2_score(y_test,y_predictXGB)

print ('Cross-Predicted Accuracy:', XGR2)
plt.subplot(3,2,1)

sns.heatmap(pd.DataFrame(conf_matrixLR), annot = True, cmap = 'YlGnBu',

           fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for Logisitc Regression', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()



plt.subplot(3,2,2)

sns.heatmap(pd.DataFrame(conf_matrixKNN), annot = True, cmap = 'YlGnBu',

           fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for KNN', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()



plt.subplot(3,2,3)

sns.heatmap(pd.DataFrame(conf_matrixTREE), annot = True, cmap = 'YlGnBu',

           fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for Decision Tree', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()



plt.subplot(3,2,4)

sns.heatmap(pd.DataFrame(conf_matrixRF), annot = True, cmap = 'YlGnBu',

           fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for Random Forest', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()



plt.subplot(3,2,5)

sns.heatmap(pd.DataFrame(conf_matrixAB), annot = True, cmap = 'YlGnBu',

           fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for AdaBoost', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()



plt.subplot(3,2,6)

sns.heatmap(pd.DataFrame(conf_matrixXGB), annot = True, cmap = 'YlGnBu',

           fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for XGBoost', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()



comparisons=pd.DataFrame()

comparisons.loc["Logistic Regression","Accuracy"]=accscoreLR

comparisons.loc["Logistic Regression","Sensitivity"]=sensitivityLR

comparisons.loc["Logistic Regression","Specificity"]=specificityLR

comparisons.loc["Logistic Regression","F1"]=LRF1

comparisons.loc["Logistic Regression","Area Under the Curve"]=aucLR

comparisons.loc["Logistic Regression","Cross Validation Accuracy"]=LRscores.mean()

comparisons.loc["Logistic Regression","R2"]=LRR2



comparisons.loc["KNN","Accuracy"]=accscoreKNN

comparisons.loc["KNN","Sensitivity"]=sensitivityKNN

comparisons.loc["KNN","Specificity"]=specificityKNN

comparisons.loc["KNN","F1"]=KNNF1

comparisons.loc["KNN","Area Under the Curve"]=aucKNN

comparisons.loc["KNN","Cross Validation Accuracy"]=KNNscores.mean()

comparisons.loc["KNN","R2"]=KNNR2



comparisons.loc["Decision Tree","Accuracy"]=accsocreTREE

comparisons.loc["Decision Tree","Sensitivity"]=sensitivityTREE

comparisons.loc["Decision Tree","Specificity"]=specificityTREE

comparisons.loc["Decision Tree","F1"]=TREEF1

comparisons.loc["Decision Tree","Area Under the Curve"]=aucTREE

comparisons.loc["Decision Tree","Cross Validation Accuracy"]=TREEscores.mean()

comparisons.loc["Decision Tree","R2"]=TREER2



comparisons.loc["AdaBoost","Accuracy"]=accscoreAB

comparisons.loc["AdaBoost","Sensitivity"]=sensitivityAB

comparisons.loc["AdaBoost","Specificity"]=specificityAB

comparisons.loc["AdaBoost","F1"]=ABF1

comparisons.loc["AdaBoost","Area Under the Curve"]=aucAB

comparisons.loc["AdaBoost","Cross Validation Accuracy"]=ABscores.mean()

comparisons.loc["AdaBoost","R2"]=ABR2



comparisons.loc["Random Forest","Accuracy"]=accscoreRF

comparisons.loc["Random Forest","Sensitivity"]=sensitivityRF

comparisons.loc["Random Forest","Specificity"]=specificityRF

comparisons.loc["Random Forest","F1"]=RFF1

comparisons.loc["Random Forest","Area Under the Curve"]=aucRF

comparisons.loc["Random Forest","Cross Validation Accuracy"]=RFscores.mean()

comparisons.loc["Random Forest","R2"]=RFR2



comparisons.loc["XGBoost","Accuracy"]=accscoreXGB

comparisons.loc["XGBoost","Sensitivity"]=sensitivityXGB

comparisons.loc["XGBoost","Specificity"]=specificityXGB

comparisons.loc["XGBoost","F1"]=XGF1

comparisons.loc["XGBoost","Area Under the Curve"]=aucXGB

comparisons.loc["XGBoost","Cross Validation Accuracy"]=XGscores.mean()

comparisons.loc["XGBoost","R2"]=XGR2





def highlight_max(s):

    '''

    highlight the maximum in a Series yellow.

    '''

    is_max = s == s.max()

    return ['background-color: yellow' if v else '' for v in is_max]



compare=comparisons.style.apply(highlight_max)

compare
error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

    

plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,marker='o', linestyle='dashed',color='b',markerfacecolor='r', markersize=10)

plt.title('K-Value vs Error Mean')

plt.show()
modelKNN.best_params_   
Default=pd.DataFrame()

Default.loc["1","Type"]=1

Default.loc["2","Type"]=2

Default.loc["3","Type"]=3



After_OneHot=pd.DataFrame()

After_OneHot.loc[:,"Type1"]=Default.loc[:,"Type"].apply(lambda x: 1 if x==1 else 0)

After_OneHot.loc[:,"Type2"]=Default.loc[:,"Type"].apply(lambda x: 1 if x==2 else 0)

After_OneHot.loc[:,"Type3"]=Default.loc[:,"Type"].apply(lambda x: 1 if x==3 else 0)



print("Before --> After One Hot Encoding")

multi_table([Default, After_OneHot])
heart4=pd.DataFrame.copy(heart2)

heart4.loc[:,"sex_female"]=heart4.loc[:,"sex"].apply(lambda x: 0 if x==1 else 1)

heart4.loc[:,"CP_Asymptomatic"]=heart4.loc[:,"cpain"].apply(lambda x: 1 if x==0 else 0)

heart4.loc[:,"CP_TypicalAng"]=heart4.loc[:,"cpain"].apply(lambda x: 1 if x==1 else 0)

heart4.loc[:,"CP_AtypicalAng"]=heart4.loc[:,"cpain"].apply(lambda x: 1 if x==2 else 0)

heart4.loc[:,"CP_NonAnginal"]=heart4.loc[:,"cpain"].apply(lambda x: 1 if x==3 else 0)

heart4.loc[:,"BSLT120"]=heart4.loc[:,"fasting_BS"].apply(lambda x: 1 if x==0 else 0)

heart4.loc[:,"EKG_Normal"]=heart4.loc[:,"resting_EKG"].apply(lambda x: 1 if x==0 else 0)

heart4.loc[:,"EKG_LVH"]=heart4.loc[:,"resting_EKG"].apply(lambda x: 1 if x==1 else 0)

heart4.loc[:,"EKG_STT"]=heart4.loc[:,"resting_EKG"].apply(lambda x: 1 if x==2 else 0)

heart4.loc[:,"ExerciseANG_No"]=heart4.loc[:,"exercise_ANG"].apply(lambda x: 1 if x==0 else 0)

heart4.loc[:,"STUpsloping"]=heart4.loc[:,"m_exercise_ST"].apply(lambda x: 1 if x==0 else 0)

heart4.loc[:,"STFlat"]=heart4.loc[:,"m_exercise_ST"].apply(lambda x: 1 if x==1 else 0)

heart4.loc[:,"STDownsloping"]=heart4.loc[:,"m_exercise_ST"].apply(lambda x: 1 if x==2 else 0)

heart4.loc[:,"STABNNormal"]=heart4.loc[:,"ST_depressionAB"].apply(lambda x: 1 if x==0 else 0)



heart4= heart4.drop('cpain',axis=1)

heart4= heart4.drop('m_exercise_ST',axis=1)

heart4= heart4.drop('resting_EKG',axis=1)

heart4=heart4[['age', 'sex','sex_female', 'CP_Asymptomatic', 'CP_TypicalAng', 'CP_AtypicalAng',

              'CP_NonAnginal', 'resting_BP', 'chol', 'fasting_BS', 'BSLT120', 'EKG_Normal',

              'EKG_LVH', 'EKG_STT', 'max_HR', 'ExerciseANG_No','exercise_ANG', 'ST_depression',

              'STUpsloping', 'STFlat', 'STDownsloping','ST_depressionAB', 'STABNNormal', 

               'no_maj_vessels', 'thal']]

heart4.columns=['Age', 'Sex_Male','Sex_Female', 'CP_Asymptomatic', 'CP_TypicalAng', 'CP_AtypicalAng',

              'CP_NonAnginal', 'Resting_BP', 'Chol', 'BSMT120', 'BSLT120', 'EKG_Normal',

              'EKG_LVH', 'EKG_STT', 'max_HR', 'ExerciseANG_No','ExerciseANG_Yes', 'ST_Depression',

              'STUpsloping', 'STFlat', 'STDownsloping','ST_depressionAB', 'STABNNormal',

             '#Major_Vessels', 'Thalium_ST']



H=heart4

X=heart2.drop('target',1)

y=heart2.loc[:,'target']
H.head()
H=heart4

y=heart2.loc[:,'target']

scaler = StandardScaler()

H_scaled = scaler.fit_transform(H)

H_train,H_test,y_train,y_test = train_test_split(H_scaled,y,test_size=.2,random_state=40)



#Logisitc Regression

LRH = LogisticRegression(C=0.01, solver='liblinear').fit(H_train,y_train)

y_predictLRH = LRH.predict(H_test)

accscoreLRH=accuracy_score(y_test,y_predictLRH)

LRHscores = cross_val_score(LRH, H_scaled, y, cv=5)

LRHR2 = metrics.r2_score(y_test,y_predictLRH)





#KNN

knn =KNeighborsClassifier()

params = {'n_neighbors':[i for i in range(1,33,2)]}

modelKNNH = GridSearchCV(knn,params,cv=10)

modelKNNH.fit(H_train,y_train)

modelKNNH.best_params_   

predictKNNH = modelKNNH.predict(H_test)

accscoreKNNH=accuracy_score(y_test,predictKNNH)

KNNHscores = cross_val_score(modelKNNH, H_scaled, y, cv=5)

KNNHR2 = metrics.r2_score(y_test,predictKNNH)



#Decision Tree

dtreeH= DecisionTreeClassifier()

paramsH = {'max_features': ['auto', 'sqrt', 'log2'],

          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 

          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11]}#

tree_modelH = GridSearchCV(dtreeH, param_grid=paramsH, n_jobs=-1)

tree_modelH.fit(H_train,y_train)

tree_modelH.best_params_

predictTREEH = tree_modelH.predict(H_test)

accscoreTREEH=accuracy_score(y_test,predictTREEH)

TREEHscores = cross_val_score(tree_modelH, H_scaled, y, cv=5)

TREEHR2 = metrics.r2_score(y_test, predictTREEH)



#AdaBoost with Decision Tree Base

modelABCH = AdaBoostClassifier(base_estimator=dtree)

param_distH = {

 'n_estimators': [i for i in range(1,100)],

 'learning_rate' : [0.01,0.05,0.1,0.3,1]

 }

ABCH = RandomizedSearchCV(AdaBoostClassifier(), param_distributions = param_distH, cv=3, n_iter = 10,n_jobs=-1)

ABCH.fit(H_train, y_train)

ABCH.best_params_ 

y_predictABCH = ABCH.predict(H_test)

accscoreABH=accuracy_score(y_test,y_predictABCH)

ABHscores = cross_val_score(ABCH, H_scaled, y, cv=5)

ABHR2 = metrics.r2_score(y_test, y_predictABCH)



#Random Forest

clfH=RandomForestClassifier()

paramsH = {'n_estimators':[i for i in range(1,100)]}

modelRFH = GridSearchCV(clfH,paramsH,cv=10)

modelRFH.fit(H_train,y_train)

modelRFH.best_params_   

y_predCLFH=modelRFH.predict(H_test)

accscoreRFH=accuracy_score(y_test,y_predCLFH)

RFHscores = cross_val_score(modelRFH, H_scaled, y, cv=5)

RFHR2 = metrics.r2_score(y_test, y_predCLFH)



#XGBoost

param_distH = {

 'n_estimators': [i for i in range(1,100)],

 'learning_rate' : [0.01,0.05,0.1,0.3,1]

 }

XGBH = RandomizedSearchCV(XGBClassifier(),param_distributions = param_distH, cv=3, n_iter = 10,n_jobs=-1)

XGBH.fit(H_train, y_train)

XGBH.best_params_ 

y_predictXGBH = XGBH.predict(H_test)

accscoreXGBH=accuracy_score(y_test,y_predictXGBH)

XGHscores = cross_val_score(XGBH, H_scaled, y, cv=5)

XGHR2 = metrics.r2_score(y_test, y_predictXGBH)
Accuracy=pd.DataFrame()

Accuracy.loc["Logistic Regression Accuracy","One Hot Encoded"]=accscoreLRH

Accuracy.loc["Logistic Regression Accuracy","Unaltered"]=accscoreLR

Accuracy.loc["KNN Accuracy","One Hot Encoded"]=accscoreKNNH

Accuracy.loc["KNN Accuracy","Unaltered"]=accscoreKNN

Accuracy.loc["Decision Tree Accuracy","One Hot Encoded"]=accscoreTREEH

Accuracy.loc["Decision Tree Accuracy","Unaltered"]=accsocreTREE

Accuracy.loc["AdaBoost Accuracy","One Hot Encoded"]=accscoreABH

Accuracy.loc["AdaBoost Accuracy","Unaltered"]=accscoreAB

Accuracy.loc["Random Forest Accuracy","One Hot Encoded"]=accscoreRFH

Accuracy.loc["Random Forest Accuracy","Unaltered"]=accscoreRF

Accuracy.loc["XGBoost Accuracy","One Hot Encoded"]=accscoreXGBH

Accuracy.loc["XGBoost Accuracy","Unaltered"]=accscoreXGB



Accuracy=Accuracy.T

Accuracy.style.apply(highlight_max)
CVA=pd.DataFrame()

CVA.loc["Logistic Regression CVA","One Hot Encoded"]=LRHscores.mean()

CVA.loc["Logistic Regression CVA","Unaltered"]=LRscores.mean()

CVA.loc["KNN CVA","One Hot Encoded"]=KNNHscores.mean()

CVA.loc["KNN CVA","Unaltered"]=KNNscores.mean()

CVA.loc["Decision Tree CVA","One Hot Encoded"]=TREEHscores.mean()

CVA.loc["Decision Tree CVA","Unaltered"]=TREEscores.mean()

CVA.loc["AdaBoost CVA","One Hot Encoded"]=ABHscores.mean()

CVA.loc["AdaBoost CVA","Unaltered"]=ABscores.mean()

CVA.loc["Random Forest CVA","One Hot Encoded"]=RFHscores.mean()

CVA.loc["Random Forest CVA","Unaltered"]=RFscores.mean()

CVA.loc["XGBoost Accuracy CVA","One Hot Encoded"]=XGHscores.mean()

CVA.loc["XGBoost Accuracy CVA","Unaltered"]=XGscores.mean()



CVA=CVA.T

CVA.style.apply(highlight_max)
R2=pd.DataFrame()

R2.loc["Logistic Regression R2","One Hot Encoded"]=LRHR2

R2.loc["Logistic Regression R2","Unaltered"]=LRR2

R2.loc["KNN R2","One Hot Encoded"]=KNNHR2

R2.loc["KNN R2","Unaltered"]=KNNR2

R2.loc["Decision Tree R2","One Hot Encoded"]=TREEHR2

R2.loc["Decision Tree R2","Unaltered"]=TREER2

R2.loc["AdaBoost R2","One Hot Encoded"]=ABHR2

R2.loc["AdaBoost R2","Unaltered"]=ABR2

R2.loc["Random Forest R2","One Hot Encoded"]=RFHR2

R2.loc["Random Forest R2","Unaltered"]=RFR2

R2.loc["XGBoost Accuracy R2","One Hot Encoded"]=XGHR2

R2.loc["XGBoost Accuracy R2","Unaltered"]=XGR2



R2=R2.T

R2.style.apply(highlight_max)
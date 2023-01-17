import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



import matplotlib as mpl

mpl.rc('font',family='Malgun Gothic')



pd.options.display.max_columns=100
heart=pd.read_csv('../input/heart-disease-prediction-using-logistic-regression/framingham.csv')
heart.head()
heart.shape
heart.describe()
heart.isna().sum()
heart['education_clean']=heart['education'].fillna(0)
heart['cigsPerDay_clean']=heart['cigsPerDay'].fillna(0)
heart['BPMeds_clean']=heart['BPMeds'].fillna(2)
heart['totChol_clean']=heart['totChol'].fillna(heart['totChol'].mean())
heart['BMI_clean']=heart['BMI'].fillna(heart['BMI'].mean())
heart['glucose_clean']=heart['glucose'].fillna(heart['glucose'].mean())
heart.columns
data=heart[['male', 'age','currentSmoker','prevalentStroke', 'prevalentHyp', 'diabetes', 'sysBP',

       'diaBP', 'heartRate','TenYearCHD', 'education_clean','cigsPerDay_clean', 'BPMeds_clean', 'totChol_clean', 'BMI_clean',

       'glucose_clean']]
data.dropna(inplace=True)
print(data.isnull().sum())

print(data.shape)
sns.countplot(data['TenYearCHD'])
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score, precision_score
heart.columns
X=data[['male', 'age', 'currentSmoker', 'prevalentStroke', 'prevalentHyp',

       'diabetes', 'sysBP', 'diaBP', 'heartRate',

       'education_clean', 'cigsPerDay_clean', 'BPMeds_clean', 'totChol_clean',

       'BMI_clean', 'glucose_clean']]

Y=data['TenYearCHD']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=142)



print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
dt=DecisionTreeClassifier()

rf=RandomForestClassifier()

xgb=XGBClassifier()

lgb=LGBMClassifier()

logit=LogisticRegression(class_weight='balanced')
dt.fit(X_train,Y_train)

rf.fit(X_train,Y_train)

xgb.fit(X_train,Y_train)

lgb.fit(X_train,Y_train)

logit.fit(X_train,Y_train)
#DT

Y_pred=dt.predict(X_test)

cmdf=pd.DataFrame(confusion_matrix(Y_test,Y_pred),columns=[0,1],index=[0,1])

sns.heatmap(cmdf,annot=True)

plt.xlabel('예측값')

plt.ylabel('실제값')



print('F1: ',f1_score(Y_test,Y_pred))
#rf

Y_pred=rf.predict(X_test)

cmdf=pd.DataFrame(confusion_matrix(Y_test,Y_pred),columns=[0,1],index=[0,1])

sns.heatmap(cmdf,annot=True)

plt.xlabel('예측값')

plt.ylabel('실제값')



print('F1: ',f1_score(Y_test,Y_pred))
#xgb

Y_pred=xgb.predict(X_test)

cmdf=pd.DataFrame(confusion_matrix(Y_test,Y_pred),columns=[0,1],index=[0,1])

sns.heatmap(cmdf,annot=True)

plt.xlabel('예측값')

plt.ylabel('실제값')



print('F1: ',f1_score(Y_test,Y_pred))
#lgbm

Y_pred=lgb.predict(X_test)

cmdf=pd.DataFrame(confusion_matrix(Y_test,Y_pred),columns=[0,1],index=[0,1])

sns.heatmap(cmdf,annot=True)

plt.xlabel('예측값')

plt.ylabel('실제값')



print('F1: ',f1_score(Y_test,Y_pred))
#logit

Y_pred=logit.predict(X_test)

cmdf=pd.DataFrame(confusion_matrix(Y_test,Y_pred),columns=[0,1],index=[0,1])

sns.heatmap(cmdf,annot=True)

plt.xlabel('예측값')

plt.ylabel('실제값')



print('F1: ',f1_score(Y_test,Y_pred))
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=142)



print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
dt=DecisionTreeClassifier()

dt.fit(X_train,Y_train)
parameters={'max_depth':[2,3,5,10],'criterion':['gini','entropy'],

            'min_samples_split':[2,3,5],'class_weight':['balanced']}



grid_model=GridSearchCV(dt,param_grid=parameters,scoring='f1',refit=True)

grid_model.fit(X_train,Y_train)



estimator=grid_model.best_estimator_

pred=estimator.predict(X_test)



print("최적 하이퍼파라미터: ", grid_model.best_params_)

print('f1_score: ',f1_score(Y_test,pred))
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=142)



print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
#logit

parameters={'C':[2,3,5,10,20,50],'penalty':['l1','l2']}



grid_model=GridSearchCV(logit,param_grid=parameters,cv=5,scoring='f1',refit=True)

grid_model.fit(X_train,Y_train)



lg=grid_model.best_estimator_

pred=lg.predict(X_test)



print("최적 하이퍼파라미터: ", grid_model.best_params_)

print('f1_score: ',f1_score(Y_test,pred))
#Precision-recall curve

from sklearn.metrics import precision_recall_curve



target_proba=lg.predict_proba(X_test)[:,1]
def precision_recall_curve_plot(Y_test, prdict_proba_class1):

    precisions, recalls, thresholds = precision_recall_curve(Y_test, prdict_proba_class1)

    

    plt.figure(figsize=[7,7])

    threshold_boundary = thresholds.shape[0]

    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--',label='precision')

    plt.plot(thresholds, recalls[0:threshold_boundary], label='recall')

    

    # 그래프 X축 Scaling (Scale 단위를 0.1로 지정)

    start ,end = plt.xlim()

    plt.xticks(np.round(np.arange(start,end,0.1),2))

    

    # 그래프 설정 

    plt.xlabel('Threshold value')

    plt.title('Precision & Recall value')

    plt.legend()

    plt.show()
precision_recall_curve_plot(Y_test,target_proba)
#바뀐 threshold값에 따라 다시 실행을 시켜줌

from sklearn.preprocessing import Binarizer
# 지정해준 threshold 각각 마다 모두 confusion matrix, precision, recall, accuracy, f1_score 산출

def classification_evaluation(Y_test,Y_pred_test):

    confusion = confusion_matrix(Y_test,Y_pred_test)

    accuracy = accuracy_score(Y_test,Y_pred_test)

    precision = precision_score(Y_test,Y_pred_test)

    recall = recall_score(Y_test,Y_pred_test)

    f1 = f1_score(Y_test,Y_pred_test)

    print("Confusion Matrix")

    print(confusion)

    print(" ")

    print("정확도 : ",accuracy.round(3))

    print("정밀도 : ",precision.round(3))

    print("재현률 : ",recall.round(3))

    print("F1 Score : ",f1.round(3))

    print(" ")



def classification_evaluation_Threshold(Y_test,Y_pred_test,thresholds):

    for customer_threshold in thresholds:

        customer_Binarizer = Binarizer(threshold = customer_threshold).fit(Y_pred_test)

        customer_Predict = customer_Binarizer.transform(Y_pred_test)

        print("======= Threshold : ", customer_threshold,"=======")

        classification_evaluation(Y_test,customer_Predict)
thresholds= [0.55,0.56,0.57,0.58,0.59,0.6,0.61]



target_proba=lg.predict_proba(X_test)[:,1].reshape(-1,1)



classification_evaluation_Threshold(Y_test,target_proba,thresholds)
lgb=LGBMClassifier(n_estimators=200,min_child_samples=50,max_depth=10,num_leaves=20,learning_rate=0.9)

lgb.fit(X_train,Y_train)



parameters={'learning_rate':[0.7,0.8,0.9]}



grid_model=GridSearchCV(lgb,param_grid=parameters,cv=5,scoring='f1',refit=True)

grid_model.fit(X_train,Y_train)



lgbm=grid_model.best_estimator_

pred=lgbm.predict(X_test)



print("최적 하이퍼파라미터: ", grid_model.best_params_)

print('f1_score: ',f1_score(Y_test,pred))
target_proba=lgbm.predict_proba(X_test)[:,1]

precision_recall_curve_plot(Y_test,target_proba)
thresholds= [0.003,0.008,0.01,0.02,0.03,0.04]



target_proba=lgbm.predict_proba(X_test)[:,1].reshape(-1,1)



classification_evaluation_Threshold(Y_test,target_proba,thresholds)
vot=VotingClassifier(estimators=[('LR',lg),('LGB',lgbm)],voting='soft')
vot.fit(X_train,Y_train)

pred=vot.predict(X_test)

f1_score(pred,Y_test)
target_proba=vot.predict_proba(X_test)[:,1]

precision_recall_curve_plot(Y_test,target_proba)
thresholds= [0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.4]



target_proba=vot.predict_proba(X_test)[:,1].reshape(-1,1)



classification_evaluation_Threshold(Y_test,target_proba,thresholds)
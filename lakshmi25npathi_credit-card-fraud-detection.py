import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score

from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,auc,classification_report,roc_auc_score

from scikitplot.metrics import plot_confusion_matrix,plot_precision_recall_curve



import lightgbm as lgb

from scipy.stats import randint as sp_randint

from sklearn.model_selection import StratifiedKFold



import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')



sns.set_style('whitegrid')

#import the training dataset

cc_df=pd.read_csv('../input/creditcard.csv')

cc_df.head()
#Shape of the dataset

cc_df.shape
#count of target classes

print(cc_df['Class'].value_counts())

#count for target classes

fig,ax=plt.subplots(figsize=(20,5))

sns.countplot(cc_df.Class.values,palette='husl')
#Percentage of target classes count

cc_df['Class'].value_counts(normalize=True)
%%time

#Distribution of attributes

attributes=cc_df.columns.values[1:30]

def plot_attribute_distribution(attributes):

    i=0

    sns.set_style('whitegrid')

    

    fig=plt.figure()

    ax=plt.subplots(5,6,figsize=(22,18))

    

    for var in attributes:

        i+=1

        plt.subplot(5,6,i)

        sns.distplot(cc_df[var],hist=False)

        plt.xlabel('var',)

        sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

    plt.show()



plot_attribute_distribution(attributes)
#Correlations in training attributes

attributes=cc_df.columns.values[1:30]

correlations=cc_df[attributes].corr().abs().unstack().sort_values(kind='quicksort').reset_index()

correlations=correlations[correlations['level_0']!=correlations['level_1']]

print(correlations)
#normalized the amount variable by using standard scaler

ss=StandardScaler()

#convert to numpy array

amount=np.array(cc_df['Amount']).reshape(-1,1)

#fit transform the data

amount_ss=ss.fit_transform(amount)

#Create a dataframe

amount_df=pd.DataFrame(amount_ss,columns=['Amount'])

amount_df.head()
#Drop the amount variable

cc_df=cc_df.drop(['Amount'],axis=1)

cc_df.head()
#Creating the amount variable

cc_df['Amount']=amount_df

cc_df.head()
%%time

#Training data

X=cc_df.drop(['Time','Class'],axis=1)

Y=cc_df['Class']



X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25)

print('Shape of X_train :',X_train.shape)

print('Shape of X_test:',X_test.shape)

print('Shape of y_train :',y_test.shape)

print('Shape of y_test :',y_test.shape)
%%time

x=X_train

y=y_train

#StratifiedKFold cross validator

cv=StratifiedKFold(n_splits=5,random_state=42)

for train_index,valid_index in cv.split(x,y):

    X_t, X_v=x.iloc[train_index], x.iloc[valid_index]

    y_t, y_v=y.iloc[train_index], y.iloc[valid_index]



print('Shape of X_train :',X_t.shape)

print('Shape of X_test:',X_v.shape)

print('Shape of y_train :',y_t.shape)

print('Shape of y_test :',y_v.shape)
%%time

#Synthetic Minority Oversampling Technique

sm = SMOTE(random_state=42, ratio=1.0)

#Generating synthetic data points

X_smote,y_smote=sm.fit_sample(X_t,y_t)

X_smote_v,y_smote_v=sm.fit_sample(X_v,y_v)

print('shape of X_smote :',X_smote.shape)

print('Shape of y_smote :',y_smote.shape)

print('shape of X_smote_v :',X_smote_v.shape)

print('Shape of y_smote_v :',y_smote_v.shape)
%%time

#Logistic regression model for SMOTE

smote=LogisticRegression(random_state=42)

#fitting the smote model

smote.fit(X_smote,y_smote)
#Accuracy of the model

smote_score=smote.score(X_smote,y_smote)

print('Accuracy of the smote_model :',smote_score)
%%time

#Cross validation prediction

cv_pred=cross_val_predict(smote,X_smote_v,y_smote_v,cv=5)

#Cross validation score

cv_score=cross_val_score(smote,X_smote_v,y_smote_v,cv=5)

print('cross_val_score :',np.average(cv_score))
%%time

#Predicting the model

smote_pred=smote.predict(X_test)

print(smote_pred)
#Confusion matrix

cm=confusion_matrix(y_test,smote_pred)

#Plot the confusion matrix

plot_confusion_matrix(y_test,smote_pred,normalize=False,figsize=(15,8))
#ROC_AUC score

roc_score=roc_auc_score(y_test,smote_pred)

print('ROC score :',roc_score)



#ROC_AUC curve

plt.figure()

false_positive_rate,recall,thresholds=roc_curve(y_test,smote_pred)

roc_auc=auc(false_positive_rate,recall)

plt.title('Reciver Operating Characteristics(ROC)')

plt.plot(false_positive_rate,recall,'b',label='ROC(area=%0.3f)' %roc_auc)

plt.legend()

plt.plot([0,1],[0,1],'r--')

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.0])

plt.ylabel('Recall(True Positive Rate)')

plt.xlabel('False Positive Rate')

plt.show()
#Classification report

scores=classification_report(y_test,smote_pred)

print(scores)
%%time

#train data

lgb_train=lgb.Dataset(X_t,y_t)

#validation data

lgb_valid=lgb.Dataset(X_v,y_v)
#choosing the hyperparameters

params={'boosting_type': 'gbdt', 

          'max_depth' : 25,

          'objective': 'binary',

          'boost_from_average':False, 

          'nthread': 12,

          'num_leaves': 120,

          'learning_rate': 0.07,

          'max_bin': 1000,  

          'subsample_for_bin': 200,

          'is_unbalance':True,

          'metric' : 'auc',

          }
%%time

#training the model

num_round=5000

lgbm= lgb.train(params,lgb_train,num_round,valid_sets=[lgb_train,lgb_valid],verbose_eval=500,early_stopping_rounds = 4000)

lgbm
#predict the model

lgbm_predict_prob=lgbm.predict(X_test,random_state=42,num_iteration=lgbm.best_iteration)

print(lgbm_predict_prob)

#Convert to binary output

lgbm_predict=np.where(lgbm_predict_prob>=0.5,1,0)

print(lgbm_predict)
lgb.plot_importance(lgbm,max_num_features=29,importance_type="split",figsize=(15,8))
plt.figure()

#confusion matrix

cm=confusion_matrix(y_test,lgbm_predict)

print(cm)

labels=['True','False']

plt.figure(figsize=(10,5))

sns.heatmap(cm,xticklabels=labels,yticklabels=labels,cmap='Blues',vmin=0.2,annot=True,fmt='d')

plt.title('Confusion_matrix')

plt.xlabel('Predicted Class')

plt.ylabel('True Class')

plt.show()
#printing the classification report

print(classification_report(y_test,lgbm_predict))
#submitting the prediction results

sub_df=pd.DataFrame(data=lgbm_predict_prob,columns=['lgbm_pred'])

sub_df['smote_pred']=smote_pred

#Save to csv file

sub_df.to_csv('submission.csv',index=False)
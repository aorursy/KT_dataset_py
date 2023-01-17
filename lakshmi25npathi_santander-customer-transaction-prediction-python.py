import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score

from sklearn.metrics import roc_auc_score,confusion_matrix,make_scorer,classification_report,roc_curve,auc

from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import SMOTE, RandomOverSampler

from imblearn.under_sampling import ClusterCentroids,NearMiss, RandomUnderSampler

import lightgbm as lgb

import eli5

from eli5.sklearn import PermutationImportance

from sklearn import tree

import graphviz

from pdpbox import pdp, get_dataset, info_plots

import scikitplot as skplt

from scikitplot.metrics import plot_confusion_matrix,plot_precision_recall_curve





from scipy.stats import randint as sp_randint

import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))



random_state=42

np.random.seed(random_state)
#importing the train dataset

train_df=pd.read_csv('../input/train.csv')

train_df.head()
#Shape of the train dataset

train_df.shape
#Summary of the dataset

train_df.describe()
%%time

#target classes count

target_class=train_df['target'].value_counts()

print('Count of target classes :\n',target_class)

#Percentage of target classes count

per_target_class=train_df['target'].value_counts()/len(train_df)*100

print('percentage of count of target classes :\n',per_target_class)



#Countplot and violin plot for target classes

fig,ax=plt.subplots(1,2,figsize=(20,5))

sns.countplot(train_df.target.values,ax=ax[0],palette='husl')

sns.violinplot(x=train_df.target.values,y=train_df.index.values,ax=ax[1],palette='husl')

sns.stripplot(x=train_df.target.values,y=train_df.index.values,jitter=True,color='black',linewidth=0.5,size=0.5,alpha=0.5,ax=ax[1],palette='husl')

ax[0].set_xlabel('Target')

ax[1].set_xlabel('Target')

ax[1].set_ylabel('Index')
%%time

#Distribution of train attributes

def plot_train_attribute_distribution(t0,t1,label1,label2,train_attributes):

    i=0

    sns.set_style('whitegrid')

    

    fig=plt.figure()

    ax=plt.subplots(10,10,figsize=(22,18))

    

    for attribute in train_attributes:

        i+=1

        plt.subplot(10,10,i)

        sns.distplot(t0[attribute],hist=False,label=label1)

        sns.distplot(t1[attribute],hist=False,label=label2)

        plt.legend()

        plt.xlabel('Attribute',)

        sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

    plt.show()
%%time

#corresponding to negative class

t0=train_df[train_df.target.values==0]

#corresponding to positive class

t1=train_df[train_df.target.values==1]

#train attributes from 2 to 102

train_attributes=train_df.columns.values[2:102]

#plot distribution of train attributes

plot_train_attribute_distribution(t0,t1,'0','1',train_attributes)
%%time

#train attributes from 102 to 203

train_attributes=train_df.columns.values[102:203]

#plot distribution of train attributes

plot_train_attribute_distribution(t0,t1,'0','1',train_attributes)
#importing the test dataset

test_df=pd.read_csv('../input/test.csv')

test_df.head()
#Shape of the test dataset

test_df.shape
%%time

#Distribution of test attributes

def plot_test_attribute_distribution(test_attributes):

    i=0

    sns.set_style('whitegrid')

    

    fig=plt.figure()

    ax=plt.subplots(10,10,figsize=(22,18))

    

    for attribute in test_attributes:

        i+=1

        plt.subplot(10,10,i)

        sns.distplot(test_df[attribute],hist=False)

        plt.xlabel('Attribute',)

        sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

    plt.show()
%%time

#test attribiutes from 1 to 101

test_attributes=test_df.columns.values[1:101]

#plot distribution of test attributes

plot_test_attribute_distribution(test_attributes)
%%time

#test attributes from 101 to 202

test_attributes=test_df.columns.values[101:202]

#plot the distribution of test attributes

plot_test_attribute_distribution(test_attributes)
%%time

#Distribution of mean values per column in train and test dataset

plt.figure(figsize=(16,8))

#train attributes

train_attributes=train_df.columns.values[2:202]

#test attributes

test_attributes=test_df.columns.values[1:201]

#Distribution plot for mean values per column in train attributes

sns.distplot(train_df[train_attributes].mean(axis=0),color='blue',kde=True,bins=150,label='train')

#Distribution plot for mean values per column in test attributes

sns.distplot(test_df[test_attributes].mean(axis=0),color='green',kde=True,bins=150,label='test')

plt.title('Distribution of mean values per column in train and test dataset')

plt.legend()

plt.show()



#Distribution of mean values per row in train and test dataset

plt.figure(figsize=(16,8))

#Distribution plot for mean values per row in train attributes

sns.distplot(train_df[train_attributes].mean(axis=1),color='blue',kde=True,bins=150,label='train')

#Distribution plot for mean values per row in test attributes

sns.distplot(test_df[test_attributes].mean(axis=1),color='green',kde=True, bins=150, label='test')

plt.title('Distribution of mean values per row in train and test dataset')

plt.legend()

plt.show()
%%time

#Distribution of std values per column in train and test dataset

plt.figure(figsize=(16,8))

#train attributes

train_attributes=train_df.columns.values[2:202]

#test attributes

test_attributes=test_df.columns.values[1:201]

#Distribution plot for std values per column in train attributes

sns.distplot(train_df[train_attributes].std(axis=0),color='red',kde=True,bins=150,label='train')

#Distribution plot for std values per column in test attributes

sns.distplot(test_df[test_attributes].std(axis=0),color='blue',kde=True,bins=150,label='test')

plt.title('Distribution of std values per column in train and test dataset')

plt.legend()

plt.show()



#Distribution of std values per row in train and test dataset

plt.figure(figsize=(16,8))

#Distribution plot for std values per row in train attributes

sns.distplot(train_df[train_attributes].std(axis=1),color='red',kde=True,bins=150,label='train')

#Distribution plot for std values per row in test attributes

sns.distplot(test_df[test_attributes].std(axis=1),color='blue',kde=True, bins=150, label='test')

plt.title('Distribution of std values per row in train and test dataset')

plt.legend()

plt.show()
%%time

#Distribution of skew values per column in train and test dataset

plt.figure(figsize=(16,8))

#train attributes

train_attributes=train_df.columns.values[2:202]

#test attributes

test_attributes=test_df.columns.values[1:201]

#Distribution plot for skew values per column in train attributes

sns.distplot(train_df[train_attributes].skew(axis=0),color='green',kde=True,bins=150,label='train')

#Distribution plot for skew values per column in test attributes

sns.distplot(test_df[test_attributes].skew(axis=0),color='blue',kde=True,bins=150,label='test')

plt.title('Distribution of skewness values per column in train and test dataset')

plt.legend()

plt.show()



#Distribution of skew values per row in train and test dataset

plt.figure(figsize=(16,8))

#Distribution plot for skew values per row in train attributes

sns.distplot(train_df[train_attributes].skew(axis=1),color='green',kde=True,bins=150,label='train')

#Distribution plot for skew values per row in test attributes

sns.distplot(test_df[test_attributes].skew(axis=1),color='blue',kde=True, bins=150, label='test')

plt.title('Distribution of skewness values per row in train and test dataset')

plt.legend()

plt.show()
%%time

#Distribution of kurtosis values per column in train and test dataset

plt.figure(figsize=(16,8))

#train attributes

train_attributes=train_df.columns.values[2:202]

#test attributes

test_attributes=test_df.columns.values[1:201]

#Distribution plot for kurtosis values per column in train attributes

sns.distplot(train_df[train_attributes].kurtosis(axis=0),color='blue',kde=True,bins=150,label='train')

#Distribution plot for kurtosis values per column in test attributes

sns.distplot(test_df[test_attributes].kurtosis(axis=0),color='red',kde=True,bins=150,label='test')

plt.title('Distribution of kurtosis values per column in train and test dataset')

plt.legend()

plt.show()



#Distribution of kutosis values per row in train and test dataset

plt.figure(figsize=(16,8))

#Distribution plot for kurtosis values per row in train attributes

sns.distplot(train_df[train_attributes].kurtosis(axis=1),color='blue',kde=True,bins=150,label='train')

#Distribution plot for kurtosis values per row in test attributes

sns.distplot(test_df[test_attributes].kurtosis(axis=1),color='red',kde=True, bins=150, label='test')

plt.title('Distribution of kurtosis values per row in train and test dataset')

plt.legend()

plt.show()
%%time

#Finding the missing values in train and test data

train_missing=train_df.isnull().sum().sum()

test_missing=test_df.isnull().sum().sum()

print('Missing values in train data :',train_missing)

print('Missing values in test data :',test_missing)
%%time

#Correlations in train attributes

train_attributes=train_df.columns.values[2:202]

train_correlations=train_df[train_attributes].corr().abs().unstack().sort_values(kind='quicksort').reset_index()

train_correlations=train_correlations[train_correlations['level_0']!=train_correlations['level_1']]

print(train_correlations.head(10))

print(train_correlations.tail(10))
%%time

#Correlations in test attributes

test_attributes=test_df.columns.values[1:201]

test_correlations=test_df[test_attributes].corr().abs().unstack().sort_values(kind='quicksort').reset_index()

test_correlations=test_correlations[test_correlations['level_0']!=test_correlations['level_1']]

print(test_correlations.head(10))

print(test_correlations.tail(10))
%%time

#Correlations in train data

train_correlations=train_df[train_attributes].corr()

train_correlations=train_correlations.values.flatten()

train_correlations=train_correlations[train_correlations!=1]

#Correlations in test data

test_correlations=test_df[test_attributes].corr()

test_correlations=test_correlations.values.flatten()

test_correlations=test_correlations[test_correlations!=1]



plt.figure(figsize=(20,5))

#Distribution plot for correlations in train data

sns.distplot(train_correlations, color="Red", label="train")

#Distribution plot for correlations in test data

sns.distplot(test_correlations, color="Blue", label="test")

plt.xlabel("Correlation values found in train and test")

plt.ylabel("Density")

plt.title("Correlation distribution plot for train and test attributes")

plt.legend()
#training and testing data

X=train_df.drop(columns=['ID_code','target'],axis=1)

test=test_df.drop(columns=['ID_code'],axis=1)

y=train_df['target']
#Split the training data

X_train,X_valid,y_train,y_valid=train_test_split(X,y,random_state=42)



print('Shape of X_train :',X_train.shape)

print('Shape of X_valid :',X_valid.shape)

print('Shape of y_train :',y_train.shape)

print('Shape of y_valid :',y_valid.shape)
%%time

#Random forest classifier

rf_model=RandomForestClassifier(n_estimators=10,random_state=42)

#fitting the model

rf_model.fit(X_train,y_train)
%%time

#Permutation importance

from eli5.sklearn import PermutationImportance

perm_imp=PermutationImportance(rf_model,random_state=42)

#fitting the model

perm_imp.fit(X_valid,y_valid)
%%time

#Important features

eli5.show_weights(perm_imp,feature_names=X_valid.columns.tolist(),top=200)
%%time

#Create the data we will plot 'var_53'

features=[v for v in X_valid.columns if v not in ['ID_code','target']]

pdp_data=pdp.pdp_isolate(rf_model,dataset=X_valid,model_features=features,feature='var_53')

#plot feature "var_53"

pdp.pdp_plot(pdp_data,'var_53')

plt.show()
%%time

#Create the data we will plot 

pdp_data=pdp.pdp_isolate(rf_model,dataset=X_valid,model_features=features,feature='var_6')

#plot feature "var_6"

pdp.pdp_plot(pdp_data,'var_6')

plt.show()
%%time

#Create the data we will plot 

pdp_data=pdp.pdp_isolate(rf_model,dataset=X_valid,model_features=features,feature='var_146')

#plot feature "var_146"

pdp.pdp_plot(pdp_data,'var_146')

plt.show()
#Training data

X=train_df.drop(['ID_code','target'],axis=1)

Y=train_df['target']

#StratifiedKFold cross validator

cv=StratifiedKFold(n_splits=5,random_state=42,shuffle=True)

for train_index,valid_index in cv.split(X,Y):

    X_train, X_valid=X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid=Y.iloc[train_index], Y.iloc[valid_index]



print('Shape of X_train :',X_train.shape)

print('Shape of X_valid :',X_valid.shape)

print('Shape of y_train :',y_train.shape)

print('Shape of y_valid :',y_valid.shape)
%%time

#Logistic regression model

lr_model=LogisticRegression(random_state=42)

#fitting the lr model

lr_model.fit(X_train,y_train)
#Accuracy of the model

lr_score=lr_model.score(X_train,y_train)

print('Accuracy of the lr_model :',lr_score)
%%time

#Cross validation prediction

cv_predict=cross_val_predict(lr_model,X_valid,y_valid,cv=5)

#Cross validation score

cv_score=cross_val_score(lr_model,X_valid,y_valid,cv=5)

print('cross_val_score :',np.average(cv_score))
#Confusion matrix

cm=confusion_matrix(y_valid,cv_predict)

#Plot the confusion matrix

plot_confusion_matrix(y_valid,cv_predict,normalize=False,figsize=(15,8))
#ROC_AUC score

roc_score=roc_auc_score(y_valid,cv_predict)

print('ROC score :',roc_score)



#ROC_AUC curve

plt.figure()

false_positive_rate,recall,thresholds=roc_curve(y_valid,cv_predict)

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

print('AUC:',roc_auc)
#Classification report

scores=classification_report(y_valid,cv_predict)

print(scores)
%%time

#Predicting the model

X_test=test_df.drop(['ID_code'],axis=1)

lr_pred=lr_model.predict(X_test)

print(lr_pred)
%%time

from imblearn.over_sampling import SMOTE

#Synthetic Minority Oversampling Technique

sm = SMOTE(random_state=42, ratio=1.0)

#Generating synthetic data points

X_smote,y_smote=sm.fit_sample(X_train,y_train)

X_smote_v,y_smote_v=sm.fit_sample(X_valid,y_valid)
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
#Confusion matrix

cm=confusion_matrix(y_smote_v,cv_pred)

#Plot the confusion matrix

plot_confusion_matrix(y_smote_v,cv_pred,normalize=False,figsize=(15,8))
#ROC_AUC score

roc_score=roc_auc_score(y_smote_v,cv_pred)

print('ROC score :',roc_score)



#ROC_AUC curve

plt.figure()

false_positive_rate,recall,thresholds=roc_curve(y_smote_v,cv_pred)

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

print('AUC:',roc_auc)
#Classification report

scores=classification_report(y_smote_v,cv_pred)

print(scores)
%%time

#Predicting the model

X_test=test_df.drop(['ID_code'],axis=1)

smote_pred=smote.predict(X_test)

print(smote_pred)
#Training the model

#training data

lgb_train=lgb.Dataset(X_train,label=y_train)

#validation data

lgb_valid=lgb.Dataset(X_valid,label=y_valid)
#Selecting best hyperparameters by tuning of different parameters

params={'boosting_type': 'gbdt', 

          'max_depth' : -1, #no limit for max_depth if <0

          'objective': 'binary',

          'boost_from_average':False, 

          'nthread': 20,

          'metric':'auc',

          'num_leaves': 50,

          'learning_rate': 0.01,

          'max_bin': 100,      #default 255

          'subsample_for_bin': 100,

          'subsample': 1,

          'subsample_freq': 1,

          'colsample_bytree': 0.8,

          'bagging_fraction':0.5,

          'bagging_freq':5,

          'feature_fraction':0.08,

          'min_split_gain': 0.45, #>0

          'min_child_weight': 1,

          'min_child_samples': 5,

          'is_unbalance':True,

          }
num_rounds=10000

lgbm= lgb.train(params,lgb_train,num_rounds,valid_sets=[lgb_train,lgb_valid],verbose_eval=1000,early_stopping_rounds = 5000)

lgbm
X_test=test_df.drop(['ID_code'],axis=1)

#predict the model

#probability predictions

lgbm_predict_prob=lgbm.predict(X_test,random_state=42,num_iteration=lgbm.best_iteration)

#Convert to binary output 1 or 0

lgbm_predict=np.where(lgbm_predict_prob>=0.5,1,0)

print(lgbm_predict_prob)

print(lgbm_predict)
#plot the important features

lgb.plot_importance(lgbm,max_num_features=50,importance_type="split",figsize=(20,50))
#final submission

sub_df=pd.DataFrame({'ID_code':test_df['ID_code'].values})

sub_df['lgbm_predict_prob']=lgbm_predict_prob

sub_df['lgbm_predict']=lgbm_predict

sub_df.to_csv('submission.csv',index=False)

sub_df.head()
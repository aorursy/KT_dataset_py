import warnings

warnings.filterwarnings('ignore') #to ignore unnecessary warnings



import numpy as np

import pandas as pd



#plotting

import matplotlib.pyplot as plt

import seaborn as sns







#data preprocessing

from sklearn.model_selection import train_test_split



from sklearn.preprocessing import StandardScaler



# ML algorithms

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from xgboost import XGBClassifier

from xgboost import plot_importance



#cross validation

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import StratifiedKFold



# evaluation metrics

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score, classification_report











import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#raw train file



train_raw = pd.read_csv('/kaggle/input/system-hack-highly-imbalanced-data/Train.csv')

train_raw.head(10)
#checking distributuion of class 0



train_raw.loc[train_raw['MULTIPLE_OFFENSE'] == 0]
# raw test file



test_raw = pd.read_csv('/kaggle/input/system-hack-highly-imbalanced-data/Test.csv')

test_raw.head(10)


#Shape of the dataframe



print('No. of rows in train_raw = ', train_raw.shape[0],'\n','No. of columns in train_raw = ', train_raw.shape[1] )
print('No. of rows in test_raw = ', test_raw.shape[0],'\n','No. of columns in test_raw = ', test_raw.shape[1] )
# General information - train_raw



train_raw.info()
# Unique value count for train_raw 



for i in train_raw.columns:

    print(train_raw[i].value_counts(dropna=False))
train_raw['X_1'].unique()
# Converting Date from object type to pd.datetime



train_raw['DATE'] = pd.to_datetime(train_raw['DATE'])

test_raw['DATE'] = pd.to_datetime(test_raw['DATE'])



print('Lower Date: ' ,train_raw['DATE'].min(), '\n', 'Higher date : ', train_raw['DATE'].max())
#General information: test_raw



test_raw.info()
# Unique value count for test_raw 



for i in test_raw.columns:

    print(test_raw[i].value_counts(dropna=False))
#Checking Data Imbalance // Uniques value counts in 'MULTIPLE_OFFENSE' // 



uv = pd.DataFrame(train_raw['MULTIPLE_OFFENSE'].value_counts(dropna =False))

uv  # 1 - Hack

    # 0 - Not a Hack
# Function to print height of barcharts on the bars



def barh(ax):

    

    for p in ax.patches:

        val = p.get_height() #height of the bar

        x = p.get_x()+ p.get_width()/2 # x- position 

        y = p.get_y() + p.get_height() + 500 #y-position

        ax.annotate(round(val,2),(x,y))

# Ploting imbalance in dataframe 



plt.figure(figsize = (15,7))



cols = ['r','b']



plt.subplot(1,2,1)

ax0 = sns.countplot(x = 'MULTIPLE_OFFENSE',data = train_raw)

barh(ax0)

plt.title('Data Imbalance in counts')



plt.subplot(1,2,2)

labels = ['Hack','Not a Hack' ]



plt.pie(uv['MULTIPLE_OFFENSE'], labels=labels, autopct='%1.1f%%',explode = (0,1),colors = cols)

plt.title('Data Imbalance in Percentages (%)')





plt.show()



# 1 - Hack

    # 0 - Not a Hack
# All columns of train_raw dataframe



col = train_raw.columns

col 
# Numerical columns



num_col = [i for i in col if i not in ['INCIDENT_ID', 'DATE','MULTIPLE_OFFENSE']]

# Plotting boxplots to see the distribution of each variable with 'MULTIPLE_OFFENSE'



for i in num_col:

    plt.figure(figsize=(15,5))

    sns.boxplot(x='MULTIPLE_OFFENSE',y=i,data=train_raw)

    plt.show()
train_raw.describe(percentiles = [0.05,0.25,0.5,0.75,0.8,0.85,0.95,0.99,0.995])
#Removing outliers



train_df = train_raw.loc[(train_raw['X_8'] <=train_raw['X_8'].quantile(0.995)) & (train_raw['X_10'] <=train_raw['X_10'].quantile(0.995))

                          & (train_raw['X_12'] <=train_raw['X_12'].quantile(0.995))]

train_df.describe(percentiles = [0.05,0.25,0.5,0.75,0.8,0.85,0.95,0.99,0.995])
# Checking outliers for test_raw dataframe



for i in num_col:

    plt.figure(figsize=(15,5))

    sns.boxplot(x=i,data=test_raw)

    plt.show()
test_raw.describe(percentiles = [0.05,0.25,0.5,0.75,0.8,0.85,0.95,0.99,0.995])
# Checking for null values



train_df.isnull().sum()
#Searching null values in test_df



pd.DataFrame({"Null value count": test_raw.isnull().sum(), "Null value in %": round(100*(test_raw.isnull().sum()/test_raw.shape[0]),2)})
# Treating null values



test_raw['X_12']= test_raw['X_12'].fillna(test_raw['X_12'].median())

test_raw.isnull().sum()
test_df = test_raw #NEW NAME 
#Checking skewness 



for i in num_col:

    plt.figure(figsize = (20,7))

    sns.distplot(train_df[i],kde_kws={'bw':0.05})

    plt.show()
#Creating function to extract features from date column



def date_feature(df):

    df['Day'] = df['DATE'].dt.day

    df['Month'] = df['DATE'].dt.month

    df['Year'] = df['DATE'].dt.year

    

    return df
# Extracting features



train_df = date_feature(train_df)

test_df = date_feature(test_df)

train_df
# Dropping columns 'INCIDENT_ID', 'DATE'



train_df.drop(['INCIDENT_ID', 'DATE'],axis=1, inplace=True)



test_df_id = test_df.pop('INCIDENT_ID') # we need id column for submission 

test_df.drop(['DATE'],axis=1,inplace=True)



train_df.head()
test_df.head()
# Spliting into X and y



y = train_df.pop('MULTIPLE_OFFENSE')

X = train_df

X.head()
#Spiliting data set into train and test using strtify= y as the data set is highly imbalanced



X_train,X_test,y_train,y_test = train_test_split(train_df,y, train_size = 0.8, random_state = 100,stratify=y) #stratitify y as the data set is highly imbalanced



print('Train data : \n',y_train.value_counts())

print('Test data : \n',y_test.value_counts())
#Variations of different classes in train & test data set



print('Train data : \n',(y_train.value_counts()/len(y_train))*100,'\n \n')

print('Test data : \n',(y_test.value_counts()/len(y_test))*100)
# Checking dimension after spliting



print('len of X = ', len(X),', len of X_train + X_test = ', len(X_train)+len(X_test),', X_train = ',len(X_train),', X_test = ',len(X_test))
#Standardizing



scaler = StandardScaler() #initialization of StandardScaler



colx = X_train.columns



X_train[colx] = scaler.fit_transform( X_train[colx])

X_train.head()
# Standardizing X_test



X_test[colx] = scaler.transform(X_test[colx])

X_test.head()
#Standardizing TRUE test data



test_df[colx] = scaler.transform(test_df[colx])

test_df.head()
# Using RandomUnderSampler

from imblearn.under_sampling import RandomUnderSampler #undersampling



usm = RandomUnderSampler(random_state =25)



X_train_us,y_train_us = usm.fit_resample(X_train,y_train)



X_train_us =pd.DataFrame(X_train_us,columns = colx)



print('Shape of X_train after random undersampling : {}'.format(X_train_us.shape))

print('Shape of original X_train : {}'.format(X_train.shape))

X_train_us.head()

X_train_us
np.bincount(y_train_us)
np.bincount(y_train) # original
# HYPERPARAMETER TUNNING 



# Tuning logistic regression



param_log = {'penalty':['l1','l2'],'C':[0.1,.2,.3,.4,.5]}



log = LogisticRegression(class_weight ='balanced',random_state=5)



folds= StratifiedKFold(n_splits = 3, shuffle = True, random_state = 90)

grid_log = GridSearchCV(estimator = log, param_grid = param_log, 

                          cv = folds, n_jobs = -1,verbose = 1, scoring = 'roc_auc')



grid_log.fit(X_train_us,y_train_us)



grid_log.best_params_
#building model on best params



logistic = LogisticRegression(class_weight ='balanced',random_state=5,C = 0.5, penalty='l2')



logistic.fit(X_train_us,y_train_us)



#prediction

pred_log_sm_train = logistic.predict_proba(X_train_us)[:,1]

pred_log_sm_test = logistic.predict_proba(X_test)[:,1]



y_pred = logistic.predict(X_test)

# Score



print ( 'Train auc score : ', roc_auc_score(y_train_us,pred_log_sm_train))

print ( 'Test auc score : ', roc_auc_score(y_test,pred_log_sm_test))



print(classification_report(y_test,y_pred))



log_us_auc_test_cv = roc_auc_score(y_test, pred_log_sm_test)

precision_log_us_cv = precision_score(y_test,y_pred)

recall_log_us_cv = recall_score(y_test,y_pred)

f1_log_us_cv = f1_score(y_test,y_pred)
# plotting ROC curve on test data

fpr, tpr, thresholds = roc_curve(y_test, pred_log_sm_test)

plt.figure(figsize =(7,5))

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.5f)' % log_us_auc_test_cv)

plt.plot([0, 1], [0, 1])

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
# Random forest



# Hyperparameter tuning for random forest



param_rf = {

    'max_depth': [8,10],

    'min_samples_leaf': range(50, 200, 50),

    'min_samples_split':range(50, 200, 50),

    'n_estimators': [100,150,200,300], 

    'max_features': [5, 10,15,20]

    

}



rf = RandomForestClassifier(n_jobs=-1,class_weight ='balanced',random_state=105)



folds= StratifiedKFold(n_splits = 3, shuffle = True, random_state = 90)

# Instantiate 

grid_rf = RandomizedSearchCV(estimator = rf, param_distributions = param_rf, 

                          cv = folds, n_jobs = -1,verbose = 1,scoring = 'roc_auc',random_state =100)





#fitting

grid_rf.fit(X_train_us,y_train_us)



#best params



grid_rf.best_params_
#using best params



forest_cv = RandomForestClassifier(n_estimators=300,

                                   min_samples_split=100, min_samples_leaf=50,

                                   max_features=15,max_depth=10,

                                   n_jobs=-1,class_weight ='balanced',random_state=1055)



forest_cv.fit(X_train_us,y_train_us)





#prediction



pred_rf_train_cv= forest_cv.predict_proba(X_train_us)[:,1]

pred_rf_test_cv = forest_cv.predict_proba(X_test)[:,1]



y_pred = forest_cv.predict(X_test)

#score

print ( 'Train auc score : ', roc_auc_score(y_train_us,pred_rf_train_cv))

print ( 'Test auc score : ', roc_auc_score(y_test,pred_rf_test_cv))



print(classification_report(y_test,y_pred))



forest_us_auc_test_cv = roc_auc_score(y_test,pred_rf_test_cv)

precision_rf_us_cv = precision_score(y_test,y_pred)

recall_rf_us_cv = recall_score(y_test,y_pred)

f1_rf_us_cv = f1_score(y_test,y_pred)
# plotting ROC curve on test data 

fpr, tpr, thresholds = roc_curve(y_test, pred_rf_test_cv)

plt.figure(figsize =(7,5))

plt.plot(fpr, tpr, label='Random Forest (area = %0.5f)' % forest_us_auc_test_cv )

plt.plot([0, 1], [0, 1])

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
# Hypertunning the xgb



param_xgb = {

    'max_depth': [4,6,8],

    'learning_rate': [0.1,0.3,0.5,0.75],

    'n_estimators': [100,150,200],

    'subsample':[0.3,0.50,.75]

    

}



xgb= XGBClassifier(booster='gbtree',

       n_jobs=-1, objective='binary:logistic', random_state=20,

       reg_alpha=1, reg_lambda=0)



folds= StratifiedKFold(n_splits = 3, shuffle = True, random_state = 90)

# Instantiate

grid_xgb = RandomizedSearchCV(estimator = xgb, param_distributions = param_xgb, 

                          cv = folds, n_jobs = -1,verbose = 1,scoring = 'roc_auc',random_state =100)



grid_xgb.fit(X_train_us,y_train_us)
#best parameters



grid_xgb.best_params_
xgb= XGBClassifier(booster='gbtree',subsample=0.75,

                   n_estimators = 100,

                   max_depth = 8,

                   learning_rate=0.3,

                   n_jobs=-1,objective='binary:logistic', random_state=20,

                   reg_alpha=0, reg_lambda=1)   



xgb.fit(X_train_us,y_train_us)

#prediction 



pred_xgb_train_cv= xgb.predict_proba(X_train_us)[:,1]

pred_xgb_test_cv = xgb.predict_proba(X_test)[:,1]



y_pred = xgb.predict(X_test)

#score

print ( 'Train auc score : ', roc_auc_score(y_train_us,pred_xgb_train_cv))

print ( 'Test auc score : ', roc_auc_score(y_test,pred_xgb_test_cv))

print(classification_report(y_test,y_pred))





xgb_us_auc_test_cv = roc_auc_score(y_test,pred_xgb_test_cv)

precision_xgb_us_cv = precision_score(y_test,y_pred)

recall_xgb_us_cv = recall_score(y_test,y_pred)

f1_xgb_us_cv = f1_score(y_test,y_pred)
# plotting ROC curve on test data 

fpr, tpr, thresholds = roc_curve(y_test, pred_xgb_test_cv )

plt.figure(figsize =(7,5))

plt.plot(fpr, tpr, label='XGBClassifier (area = %0.5f)' % xgb_us_auc_test_cv )

plt.plot([0, 1], [0, 1])

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
#SCORE after RandomUnderSampling



auc_score_all =[log_us_auc_test_cv,forest_us_auc_test_cv,xgb_us_auc_test_cv]

recall_all = [recall_log_us_cv,recall_rf_us_cv,recall_xgb_us_cv]

precision_all = [precision_log_us_cv,precision_rf_us_cv,precision_xgb_us_cv]

f1_all =[f1_log_us_cv,f1_rf_us_cv,f1_xgb_us_cv]



us_cv = pd.DataFrame({'auc_score':auc_score_all,'recall':recall_all,'precision':precision_all,'f1_score':f1_all},index =['Logistic Regression','Random Forest','XGBoost'])

us_cv
# USING ADASYN - oversampling tech



from imblearn.over_sampling import ADASYN



ada = ADASYN (random_state=50)

X_train_adasyn, y_train_ada = ada.fit_resample(X_train, y_train)



X_train_ada =pd.DataFrame(X_train_adasyn,columns = colx) #we are creating a dataframe after the resampling

X_train_ada.head() 



#vizualisation

X_train_adasyn_0 = X_train_adasyn[X_train.shape[0]:]  #Synthetic data points are appended after the original datapoints in the dataframe.

                                                # Hence X_train.shape[0] - original data points and ater this length all are synthetic





# Creating different dataframe for class 0 and 1 separately



X_train_1 = X_train.to_numpy()[np.where(y_train==1.0)]

X_train_0 = X_train.to_numpy()[np.where(y_train==0.0)]









plt.rcParams['figure.figsize'] = [20, 20]

fig = plt.figure()



#Scatter plot to show orignal class-0 data points (two columns of the same dataframe are taken for scatter plot)

plt.subplot(3, 1, 1)

plt.scatter(X_train_0[:, 0], X_train_0[:, 1], label='Actual Class-0 Examples') 

plt.legend()



#Scatter plot for original data vs synthetic data 



plt.subplot(3, 1, 2)

plt.scatter(X_train_0[:, 0], X_train_0[:, 1], label='Actual Class-0 Examples')

plt.scatter(X_train_adasyn_0.iloc[:X_train_0.shape[0], 0], X_train_adasyn_0.iloc[:X_train_0.shape[0], 1],

            label='Artificial ADASYN Class-0 Examples')  # X_train_0.shape[0] = 804 data points 

                                                        # X_train_adasyn_0.shape[0] = 17148 data points

                                                        # X_train_adasyn_0[:X_train_0.shape[0], 0] - so that only 804 data points will be considered for the scatterplot

        

plt.legend()



# Scatter plot to show distribution of original class-0 and class-1 data points

plt.subplot(3, 1, 3)

plt.scatter(X_train_1[:, 0], X_train_1[:, 1], label='Actual Class-1 Examples')

plt.scatter(X_train_0[:, 0], X_train_0[:, 1], label='Actual Class-0 Examples')

plt.legend()
X_train_ada.shape
X_train.shape
np.bincount(y_train_ada)
np.bincount(y_train)


# Tuning logistic regression



param_log = {'penalty':['l1','l2'],'C':[0.1,.2,.3,.4,.5]}



log = LogisticRegression(class_weight ='balanced',random_state=5)



folds= StratifiedKFold(n_splits = 3, shuffle = True, random_state = 90)

grid_log = GridSearchCV(estimator = log, param_grid = param_log, 

                          cv = folds, n_jobs = -1,verbose = 1, scoring = 'roc_auc')



grid_log.fit(X_train_ada,y_train_ada)



grid_log.best_params_
#building model on best params



logistic = LogisticRegression(class_weight ='balanced',random_state=5,C = 0.5, penalty='l2')



logistic.fit(X_train_ada,y_train_ada)



#prediction

pred_log_ada_train = logistic.predict_proba(X_train_ada)[:,1]

pred_log_ada_test = logistic.predict_proba(X_test)[:,1]



y_pred = logistic.predict(X_test)

# Score



print ( 'Train auc score : ', roc_auc_score(y_train_ada,pred_log_ada_train))

print ( 'Test auc score : ', roc_auc_score(y_test,pred_log_ada_test))



print(classification_report(y_test,y_pred))



log_ada_auc_test_cv = roc_auc_score(y_test, pred_log_ada_test)

precision_log_ada_cv = precision_score(y_test,y_pred)

recall_log_ada_cv = recall_score(y_test,y_pred)

f1_log_ada_cv = f1_score(y_test,y_pred)
# plotting ROC curve on test data

fpr, tpr, thresholds = roc_curve(y_test, pred_log_ada_test)

plt.figure(figsize =(7,5))

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.5f)' % log_ada_auc_test_cv)

plt.plot([0, 1], [0, 1])

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
# Random forest



# Hyperparameter tuning for random forest



param_rf = {

    'max_depth': [8,10],

    'min_samples_leaf': range(50, 150, 50),

    'min_samples_split':range(50, 200, 50),

    'n_estimators': [100,150,200], 

    'max_features': [5, 10]

    

}



rf = RandomForestClassifier(n_jobs=-1,class_weight ='balanced',random_state=105)



folds= StratifiedKFold(n_splits = 3, shuffle = True, random_state = 90)

# Instantiate

grid_rf = RandomizedSearchCV(estimator = rf, param_distributions = param_rf, 

                          cv = folds, n_jobs = -1,verbose = 1,scoring = 'roc_auc',random_state =100)





# Fitting

grid_rf.fit(X_train_ada,y_train_ada)



#best params



grid_rf.best_params_
#using best params



forest_cv = RandomForestClassifier(n_estimators=150,

                                   min_samples_split=50, min_samples_leaf=50,

                                   max_features=10,max_depth=10,

                                   n_jobs=-1,class_weight ='balanced',random_state=1055)



forest_cv.fit(X_train_ada,y_train_ada)





#prediction



pred_rf_train_cv= forest_cv.predict_proba(X_train_ada)[:,1]

pred_rf_test_cv = forest_cv.predict_proba(X_test)[:,1]



y_pred = forest_cv.predict(X_test)

#score

print ( 'Train auc score : ', roc_auc_score(y_train_ada,pred_rf_train_cv))

print ( 'Test auc score : ', roc_auc_score(y_test,pred_rf_test_cv))



print(classification_report(y_test,y_pred))



forest_ada_auc_test_cv = roc_auc_score(y_test,pred_rf_test_cv)

precision_rf_ada_cv = precision_score(y_test,y_pred)

recall_rf_ada_cv = recall_score(y_test,y_pred)

f1_rf_ada_cv = f1_score(y_test,y_pred)
# plotting ROC curve on test data 

fpr, tpr, thresholds = roc_curve(y_test, pred_rf_test_cv)

plt.figure(figsize =(7,5))

plt.plot(fpr, tpr, label='Random Forest (area = %0.5f)' % forest_ada_auc_test_cv )

plt.plot([0, 1], [0, 1])

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
# Hypertunning the xgb

param_xgb = {

    'max_depth': [4,6,8],

    'learning_rate': [0.1,0.3,0.5,0.75],

    'n_estimators': [100,150,200],

    'subsample':[0.3,0.50,.75]

    

}



xgb= XGBClassifier(booster='gbtree',

       n_jobs=-1, objective='binary:logistic', random_state=20,

       reg_alpha=1, reg_lambda=0)



folds= StratifiedKFold(n_splits = 3, shuffle = True, random_state = 90)

# Instantiate

grid_xgb = RandomizedSearchCV(estimator = xgb, param_distributions = param_xgb, 

                          cv = folds, n_jobs = -1,verbose = 1,scoring = 'roc_auc',random_state =100)



grid_xgb.fit(X_train_ada,y_train_ada)
grid_xgb.best_params_
xgb_ada= XGBClassifier(booster='gbtree',subsample=0.75,

                   n_estimators = 100,

                   max_depth = 4,

                   learning_rate=0.75,

                   n_jobs=-1,objective='binary:logistic', random_state=20,

                   reg_alpha=0, reg_lambda=1)   



xgb_ada.fit(X_train_ada,y_train_ada)

#prediction 



pred_xgb_train_cv= xgb_ada.predict_proba(X_train_ada)[:,1]

pred_xgb_test_cv = xgb_ada.predict_proba(X_test)[:,1]



y_pred = xgb_ada.predict(X_test)

#score

print ( 'Train auc score : ', roc_auc_score(y_train_ada,pred_xgb_train_cv))

print ( 'Test auc score : ', roc_auc_score(y_test,pred_xgb_test_cv))

print(classification_report(y_test,y_pred))





xgb_ada_auc_test_cv = roc_auc_score(y_test,pred_xgb_test_cv)

precision_xgb_ada_cv = precision_score(y_test,y_pred)

recall_xgb_ada_cv = recall_score(y_test,y_pred)

f1_xgb_ada_cv = f1_score(y_test,y_pred)
# plotting ROC curve on test data 

fpr, tpr, thresholds = roc_curve(y_test, pred_xgb_test_cv )

plt.figure(figsize =(7,5))

plt.plot(fpr, tpr, label='XGBClassifier (area = %0.5f)' % xgb_ada_auc_test_cv )

plt.plot([0, 1], [0, 1])

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
#SCORE after ADASYN



auc_score_all =[log_ada_auc_test_cv,forest_ada_auc_test_cv,xgb_ada_auc_test_cv]

recall_all = [recall_log_ada_cv,recall_rf_ada_cv,recall_xgb_ada_cv]

precision_all = [precision_log_ada_cv,precision_rf_ada_cv,precision_xgb_ada_cv]

f1_all =[f1_log_ada_cv,f1_rf_ada_cv,f1_xgb_ada_cv]



ada_cv = pd.DataFrame({'auc_score':auc_score_all,'recall':recall_all,'precision':precision_all,'f1_score':f1_all},index =['Logistic Regression','Random Forest','XGBoost'])

ada_cv
#Importing SMOTE - oversampling tech



from imblearn.over_sampling import SMOTE



smt = SMOTE(random_state=42)

X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
X_train_sm
# Visualising the distribution of actual and synthetic datapoints



X_train_smote_0 = X_train_sm[X_train.shape[0]:] # Synthetic data points are appended after the original datapoints in the dataframe.

                                                # Hence X_train.shape[0] - original data points and ater this length all are synthetic



    

# Creating different dataframe for class 0 and 1 separately



X_train_1 = X_train.to_numpy()[np.where(y_train==1.0)] 

X_train_0 = X_train.to_numpy()[np.where(y_train==0.0)]





plt.rcParams['figure.figsize'] = [20, 20]

fig = plt.figure()





#Scatter plot to show orignal class-0 data points (two columns of the same dataframe are taken for scatter plot)

plt.subplot(3, 1, 1)

plt.scatter(X_train_0[:, 0], X_train_0[:, 1], label='Actual Class-0 Examples') 

plt.legend()



#Scatter plot for original data vs synthetic data 

plt.subplot(3, 1, 2)

plt.scatter(X_train_0[:, 0], X_train_0[:, 1], label='Actual Class-0 Examples')

plt.scatter(X_train_smote_0.iloc[:X_train_0.shape[0], 0], X_train_smote_0.iloc[:X_train_0.shape[0], 1],

            label='Artificial SMOTE Class-0 Examples')  # X_train_0.shape[0] = 804 data points 

                                                        # X_train_smote_0.shape[0] = 17148 data points

                                                        # X_train_smote_0[:X_train_0.shape[0], 0] - so that only 804 data points will be considered for the scatterplot

        

plt.legend()





# Scatter plot to show distribution of original class-0 and class-1 data points

plt.subplot(3, 1, 3)

plt.scatter(X_train_1[:, 0], X_train_1[:, 1], label='Actual Class-1 Examples')

plt.scatter(X_train_0[:, 0], X_train_0[:, 1], label='Actual Class-0 Examples')

plt.legend()
X_train_sm = pd.DataFrame(X_train_sm,columns = colx) #colx = columns of X_train

X_train_sm.shape #after SMOTE
X_train.shape #original
np.bincount(y_train_sm) #after SMOTE
np.bincount(y_train) #original
# Hyperparameters Tunning

#Logistic regression



param_log = {'penalty':['l1','l2'],'C':[0.1,.2,.3,.4,.5]}



log = LogisticRegression(class_weight ='balanced',random_state=5)



folds= StratifiedKFold(n_splits = 3, shuffle = True, random_state = 90)

grid_log = GridSearchCV(estimator = log, param_grid = param_log, 

                          cv = folds, n_jobs = -1,verbose = 1, scoring = 'roc_auc')



grid_log.fit(X_train_sm,y_train_sm)



grid_log.best_params_
#building model on best params



logistic = LogisticRegression(class_weight ='balanced',random_state=5,C = 0.5, penalty='l2')



logistic.fit(X_train_sm,y_train_sm)



#prediction

pred_log_sm_train = logistic.predict_proba(X_train_sm)[:,1]

pred_log_sm_test = logistic.predict_proba(X_test)[:,1]



y_pred = logistic.predict(X_test)



# Score



print ( 'Train auc score : ', roc_auc_score(y_train_sm,pred_log_sm_train))

print ( 'Test auc score : ', roc_auc_score(y_test,pred_log_sm_test))

print(classification_report(y_test,y_pred))



log_sm_auc_test_cv = roc_auc_score(y_test, pred_log_sm_test)

precision_log_sm_cv = precision_score(y_test,y_pred)

recall_log_sm_cv = recall_score(y_test,y_pred)

f1_log_sm_cv = f1_score(y_test,y_pred)
# plotting ROC curve on test data

fpr, tpr, thresholds = roc_curve(y_test, pred_log_sm_test)

plt.figure(figsize =(7,5))

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.5f)' % log_sm_auc_test_cv)

plt.plot([0, 1], [0, 1])

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
# RandomForestClassifier

# Hyperparameter tuning for random forest



param_rf = {

    'max_depth': [8,10],

    'min_samples_leaf': range(50, 150, 50),

    'min_samples_split':range(50, 200, 50),

    'n_estimators': [100,150,200], 

    'max_features': [5, 10]

    

}



rf = RandomForestClassifier(n_jobs=-1,class_weight ='balanced',random_state=105)



folds= StratifiedKFold(n_splits = 3, shuffle = True, random_state = 90)



grid_rf = RandomizedSearchCV(estimator = rf, param_distributions = param_rf, 

                          cv = folds, n_jobs = -1,verbose = 1,scoring = 'roc_auc',random_state =100)





# Fitting

grid_rf.fit(X_train_sm, y_train_sm)



#best params



grid_rf.best_params_
#using best params



forest_cv = RandomForestClassifier(n_estimators=150,

                                   min_samples_split=50, min_samples_leaf=50,

                                   max_features=10,max_depth=10,

                                   n_jobs=-1,class_weight ='balanced',random_state=105)



forest_cv.fit(X_train_sm, y_train_sm)





#prediction



pred_rf_train_cv= forest_cv.predict_proba(X_train_sm)[:,1]

pred_rf_test_cv = forest_cv.predict_proba(X_test)[:,1]



y_pred = forest_cv.predict(X_test)



#score

print ( 'Train auc score : ', roc_auc_score(y_train_sm,pred_rf_train_cv))

print ( 'Test auc score : ', roc_auc_score(y_test,pred_rf_test_cv))

print(classification_report(y_test,y_pred))



forest_sm_auc_test_cv = roc_auc_score(y_test,pred_rf_test_cv)

precision_rf_sm_cv = precision_score(y_test,y_pred)

recall_rf_sm_cv = recall_score(y_test,y_pred)

f1_rf_sm_cv = f1_score(y_test,y_pred)

# plotting ROC curve on test data 

fpr, tpr, thresholds = roc_curve(y_test, pred_rf_test_cv)

plt.figure(figsize =(7,5))

plt.plot(fpr, tpr, label='Random Forest (area = %0.5f)' % forest_sm_auc_test_cv )

plt.plot([0, 1], [0, 1])

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
#XGBClassifier

# Hyperparameter tuning 



param_xgb = {

    'max_depth': [4,6,8],

    'learning_rate': [0.1,0.3,0.5,0.75],

    'n_estimators': [100,150,200],

    'subsample':[0.3,0.50,.75]

    

}



xgb= XGBClassifier(booster='gbtree',

       n_jobs=-1, objective='binary:logistic', random_state=20,

       reg_alpha=1, reg_lambda=0)



folds= StratifiedKFold(n_splits = 3, shuffle = True, random_state = 90)

# Instantiate 

grid_xgb = RandomizedSearchCV(estimator = xgb, param_distributions = param_xgb, 

                          cv = folds, n_jobs = -1,verbose = 1,scoring = 'roc_auc',random_state =100)



grid_xgb.fit(X_train_sm, y_train_sm)
#best parameters



grid_xgb.best_params_
xgb= XGBClassifier(booster='gbtree',subsample=0.75,

                   n_estimators = 150,

                   max_depth = 4,

                   learning_rate=0.3,

                   n_jobs=-1,objective='binary:logistic', random_state=20,

                   reg_alpha=1, reg_lambda=0)   



xgb.fit(X_train_sm, y_train_sm)

#prediction 



pred_xgb_train_cv= xgb.predict_proba(X_train_sm)[:,1]

pred_xgb_test_cv = xgb.predict_proba(X_test)[:,1]



y_pred = xgb.predict(X_test)



#score

print ( 'Train auc score : ', roc_auc_score(y_train_sm,pred_xgb_train_cv))

print ( 'Test auc score : ', roc_auc_score(y_test,pred_xgb_test_cv))

print(classification_report(y_test,y_pred))



xgb_sm_auc_test_cv = roc_auc_score(y_test,pred_xgb_test_cv)

precision_xgb_sm_cv = precision_score(y_test,y_pred)

recall_xgb_sm_cv = recall_score(y_test,y_pred)

f1_xgb_sm_cv = f1_score(y_test,y_pred)

# plotting ROC curve on test data 

fpr, tpr, thresholds = roc_curve(y_test, pred_xgb_test_cv )

plt.figure(figsize =(7,5))

plt.plot(fpr, tpr, label='XGBClassifier (area = %0.5f)' % xgb_sm_auc_test_cv)

plt.plot([0, 1], [0, 1])

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
# scores 



auc_score_all =[log_sm_auc_test_cv,forest_sm_auc_test_cv,xgb_sm_auc_test_cv]

recall_all = [recall_log_sm_cv,recall_rf_sm_cv,recall_xgb_sm_cv]

precision_all = [precision_log_sm_cv,precision_rf_sm_cv,precision_xgb_sm_cv]

f1_all =[f1_log_sm_cv,f1_rf_sm_cv,f1_xgb_sm_cv]



sm_cv = pd.DataFrame({'auc_score':auc_score_all,'recall':recall_all,'precision':precision_all,'f1_score':f1_all},index =['Logistic Regression','Random Forest','XGBoost'])

sm_cv
us_cv
ada_cv


hack= xgb_ada.predict(test_df)



result = pd.DataFrame({'INCIDENT_ID':test_df_id,'MULTIPLE_OFFENSE':hack})

result.head()
#Important features

xgb_ada.feature_importances_

# Important features



plot_importance(xgb,importance_type='gain')

plt.show()
result.to_csv ('subx.csv', index = None, header=True)

pd.read_csv('subx.csv').head()
#saving the final model as pickle file



import pickle



#open a file where you want to store the model

file = open('xgb.pkl','wb')



pickle.dump(xgb,file) #dumping / saving the model inside file xgb.pkl
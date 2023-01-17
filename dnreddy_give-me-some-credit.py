import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

import graphviz



from sklearn import preprocessing,model_selection

import itertools



import xgboost as xgb



import warnings

warnings.filterwarnings("ignore")
import os

print(os.listdir("/kaggle/input/GiveMeSomeCredit"))
train_df = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-training.csv')

test_df = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-test.csv')

print ("training dataset shape is {}".format(train_df.shape))

print ("testing dataset shape is {}".format(test_df.shape))
train_df.head()
col_names = train_df.columns.values

col_names[0] = 'ID' ## rename first column to ID

train_df.columns = col_names ## assign new column name to training dataset

test_df.columns = col_names ## assign new column name to testing dataset
train_df.head()
test_df.head()
print(train_df.dtypes)
print(test_df.dtypes)
train_df.isnull().sum()
test_df.isnull().sum()
# remove ID, target variable Dlqin2yrs and variables with missing values

feature_list=list(train_df.columns.values)

remove_list = ['ID','SeriousDlqin2yrs','MonthlyIncome','NumberOfDependents']

for each in remove_list:

    feature_list.remove(each)



for each in feature_list:

    sns.distplot(train_df[each])

    plt.show()
print (train_df.columns.values)
log_trans_list = train_df.columns.values[[2,4,5,8,9,10]]

log_trans_list

for each in log_trans_list:

    train_df[each] = np.log(1+train_df[each].values)
for each in feature_list:

    sns.distplot(train_df[each])

    plt.show()
partial_train_df = train_df[['MonthlyIncome','NumberOfDependents']]

#partial_train_df.dropna(how='any')

partial_train_df = partial_train_df.dropna(how='any')



sns.distplot(partial_train_df['MonthlyIncome'])

plt.show()

sns.distplot(partial_train_df['NumberOfDependents'])

plt.show()
partial_train_df['MonthlyIncome'] = np.log(1+partial_train_df['MonthlyIncome'].values)

partial_train_df['NumberOfDependents'] = np.log(1+partial_train_df['NumberOfDependents'].values)





sns.distplot(partial_train_df['MonthlyIncome'])

plt.show()

sns.distplot(partial_train_df['NumberOfDependents'])

plt.show()
train_df['MonthlyIncome'] = np.log(1+train_df['MonthlyIncome'].values)

train_df['NumberOfDependents'] = np.log(1+train_df['NumberOfDependents'].values)
grouped_df = train_df.groupby('age')

dlinq_age = grouped_df['SeriousDlqin2yrs'].aggregate([np.mean,'count']).reset_index()

print(dlinq_age)

dlinq_age.columns =['age','DlqinFreq','count']

sns.regplot(x='age',y='DlqinFreq',data=dlinq_age)

plt.show()
## remove outlier

train_df = train_df[train_df['age'] != 0]

train_df = train_df[train_df['age'] !=99]

train_df = train_df[train_df['age'] !=101]

grouped_df = train_df.groupby('age')

dlinq_age = grouped_df['SeriousDlqin2yrs'].aggregate([np.mean,'count']).reset_index()

dlinq_age.columns =['age','DlqinFreq','count']

sns.regplot(x='age',y='DlqinFreq',data=dlinq_age)

plt.show()



## create new features

train_df['age_sqr'] = train_df['age'].values^2 

## apply the same operation on testing set

test_df['age_sqr'] = test_df['age'].values^2
train_y = train_df['SeriousDlqin2yrs']

#'RevolvingUtilizationOfUnsecuredLines'

train_X = train_df.drop(['SeriousDlqin2yrs','ID'],axis=1,inplace=False)

test_X = test_df.drop(['SeriousDlqin2yrs','ID'],axis=1,inplace=False)

print(type(train_y))



skf = model_selection.StratifiedKFold(n_splits=5,random_state=100)

xgb_params = {

'eta':0.03,

'max_depth':4,

'sub_sample':0.9,

'colsample_bytree':0.5,

'objective':'binary:logistic',

'eval_metric':'auc',

'silent':0

}



print(train_X.shape)

print(train_X.columns)

print(test_X.shape)
best_iteration =[]

best_score= []

training_score = []

for train_ind,val_ind in skf.split(train_X,train_y):

    #print (set(train_y))

    #print (type(train_y))

    X_train,X_val = train_X.iloc[train_ind,],train_X.iloc[val_ind,]

    y_train,y_val = train_y.iloc[train_ind],train_y.iloc[val_ind]

    #print (set(train_y))

    #print (max(train_ind),min(train_ind),max(val_ind),min(val_ind))

    #print (train_ind,val_ind)

    #print(set(y_train))

    dtrain = xgb.DMatrix(X_train,y_train,feature_names = X_train.columns)

    dval = xgb.DMatrix(X_val,y_val,feature_names = X_val.columns)

    model = xgb.train(xgb_params,dtrain,num_boost_round=1000,

                      evals=[(dtrain,'train'),(dval,'val')],verbose_eval=True,early_stopping_rounds=30)

    best_iteration.append(model.attributes()['best_iteration'])

    best_score.append(model.attributes()['best_score'])

    # training_score.append(model.attributes()['best_msg'].split()[1][-8:])

    xgb.plot_importance(model)

    plt.show()
def xgbCV(eta=[0.05],max_depth=[6],sub_sample=[0.9],colsample_bytree=[0.9]):

    train_y = train_df['SeriousDlqin2yrs'] # label for training data

    train_X = train_df.drop(['SeriousDlqin2yrs','ID'],axis=1,inplace=False) # feature for training data

    test_X = test_df.drop(['SeriousDlqin2yrs','ID'],axis=1,inplace=False) # feature for testing data

    skf = model_selection.StratifiedKFold(n_splits=5,random_state=100) # stratified sampling

    train_performance ={} 

    val_performance={}

    for each_param in itertools.product(eta,max_depth,sub_sample,colsample_bytree): # iterative over each combination in parameter space

        xgb_params = {

                    'eta':each_param[0],

                    'max_depth':each_param[1],

                    'sub_sample':each_param[2],

                    'colsample_bytree':each_param[3],

                    'objective':'binary:logistic',

                    'eval_metric':'auc',

                    'silent':0

                    }

        best_iteration =[]

        best_score=[]

        training_score=[]

        for train_ind,val_ind in skf.split(train_X,train_y): # five fold stratified cross validation

            X_train,X_val = train_X.iloc[train_ind,],train_X.iloc[val_ind,] # train X and train y

            y_train,y_val = train_y.iloc[train_ind],train_y.iloc[val_ind] # validation X and validation y

            dtrain = xgb.DMatrix(X_train,y_train,feature_names = X_train.columns) # convert into DMatrix (xgb library data structure)

            dval = xgb.DMatrix(X_val,y_val,feature_names = X_val.columns) # convert into DMatrix (xgb library data structure)

            model = xgb.train(xgb_params,dtrain,num_boost_round=1000, 

                              evals=[(dtrain,'train'),(dval,'val')],verbose_eval=False,early_stopping_rounds=30) # train the model

            best_iteration.append(model.attributes()['best_iteration']) # best iteration regarding AUC in valid set

            best_score.append(model.attributes()['best_score']) # best score regarding AUC in valid set

            training_score.append(model.attributes()['best_msg'].split()[1][10:]) # best score regarding AUC in training set

        valid_mean = (np.asarray(best_score).astype(np.float).mean()) # mean AUC in valid set

        train_mean = (np.asarray(training_score).astype(np.float).mean()) # mean AUC in training set

        val_performance[each_param] =  train_mean

        train_performance[each_param] =  valid_mean

        print ("Parameters are {}. Training performance is {:.4f}. Validation performance is {:.4f}".format(each_param,train_mean,valid_mean))

    return (train_performance,val_performance)

#xgbCV(eta=[0.01,0.02,0.03,0.04,0.05],max_depth=[4,6,8,10],colsample_bytree=[0.3,0.5,0.7,0.9]) 

xgbCV(eta=[0.04],max_depth=[4],colsample_bytree=[0.5])
print(train_X.columns)

any(train_X.columns == test_X.columns)
train = xgb.DMatrix(train_X,train_y,feature_names=train_X.columns)

test = xgb.DMatrix(test_X,feature_names=test_X.columns)

xgb_params = {

                    'eta':0.03,

                    'max_depth':4,

                    'sub_sample':0.9,

                    'colsample_bytree':0.5,

                    'objective':'binary:logistic',

                    'eval_metric':'auc',

                    'silent':0

                    }



final_model = xgb.train(xgb_params,train,num_boost_round=500)

ypred = final_model.predict(test)
xgb.plot_importance(final_model)

plt.show()
SUB_1 = pd.DataFrame({'Id':test_df.ID.values,'Probability':ypred})

SUB_1.to_csv('Submission.csv',index=False)

SUB_1.head()
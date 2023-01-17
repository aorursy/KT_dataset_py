# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





#Catagorical conversion libraries

from sklearn.preprocessing import LabelEncoder

from category_encoders import HashingEncoder,TargetEncoder



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold,train_test_split

from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier,plot_importance

from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE



import xgboost as xgb



from sklearn.metrics import roc_curve,roc_auc_score



from sklearn.metrics import confusion_matrix,classification_report



import seaborn as sns

sns.set(style='whitegrid')

import matplotlib.pyplot as plt

%matplotlib inline



import gc

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

train_df.head()
test_df = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')

test_df.head()
print("Train DF dimension:",train_df.shape)

print("Test DF dimension:",test_df.shape)
# Target variable value counts .There is a imbalance in target .Have to apply oversampling techniques such as SMOTe,Near Miss

train_df.target.value_counts()
# Unique Values for each feature

[(c,train_df[c].unique()) for c in train_df.columns[1:]]
%%time

print("Feature Engineering ....")



train_encoded_df = train_df[['id']].copy()

test_encoded_df = test_df[['id']].copy()





# Target Encoder technique

target_encoding_feat = ['bin_0','bin_1','bin_2','bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7','nom_8',

                        'nom_9','ord_0','ord_1','ord_2','ord_3','ord_4','ord_5']



print("Starting Target Encoding .....")

te = TargetEncoder(cols=target_encoding_feat,smoothing=1.0)

te_encoded_df = te.fit_transform(train_df[target_encoding_feat],train_df['target'])

te_test_encoded_df = te.transform(test_df[target_encoding_feat])



te_encoded_df.columns = 'te_' + te_encoded_df.columns

te_test_encoded_df.columns = 'te_' + te_test_encoded_df.columns



train_encoded_df = pd.concat([train_encoded_df,te_encoded_df],axis=1)

test_encoded_df = pd.concat([test_encoded_df,te_test_encoded_df],axis=1)



#print("Target Encoding Done!..")





# Features day and month are in cyclic in nature . So using cyclic catagorical encoding technique 

print("Cyclic Encoding begin!..")

def cyclic_feat_encoding(df,col):

    

    df['sine_'+col] = np.sin(2 * np.pi * (df[col])/max(df[col]))

    df['cos_'+col] = np.cos(2 * np.pi * (df[col])/max(df[col]))

    return df



train_df = cyclic_feat_encoding(train_df,'day')

train_df = cyclic_feat_encoding(train_df,'month')

test_df = cyclic_feat_encoding(test_df,'day')

test_df = cyclic_feat_encoding(test_df,'month')



train_encoded_df = pd.concat([train_encoded_df,train_df[['sine_day','cos_day','sine_month','cos_month']]],axis=1)

test_encoded_df = pd.concat([test_encoded_df,test_df[['sine_day','cos_day','sine_month','cos_month']]],axis=1)



print("Dimension of Train Encoded DF :",train_encoded_df.shape)

print("Dimension of Test Encoded DF :",test_encoded_df.shape)



print("Feature Engineering Done!..")
y= train_df['target']

y.head()
train_encoded_df.head()
test_encoded_df.head()
del train_df

del test_df

gc.collect()
# Definition to plot roc curve

def roc_curve_plot(fpr,tpr,auc):

    fig,ax = plt.subplots()

    ax.plot(fpr,tpr,'b-',linewidth=2)

    ax.plot([0,1],[0,1],color='navy',linestyle='--')

    ax.set_title(f'AUC:{auc}')

    ax.set(xlabel="False Positive Rate",ylabel="True Positive Rate")

    plt.show()
with_target_encoded_cols = train_encoded_df.columns.values.tolist()[1:]

with_target_encoded_cols
train_encoded_df[with_target_encoded_cols].head()
# Function to fit and predict the classifier 

def fit_clf(clf,X_train,y_train,X_valid,y_valid,t_df):

        

    clf.fit(X_train,y_train)

    preds = clf.predict(X_valid)

    auc = roc_auc_score(y_valid,preds)

    print("roc_auc_score :",auc)

    prep_proba = clf.predict_proba(t_df)[:,1]

    #fpr,tpr,threshold = roc_curve(y_valid,preds,pos_label=1)

    #roc_curve_plot(fpr,tpr,auc)

    return clf,auc,prep_proba
# Function to split df using startifiedkfold and apply SMOTE Upsampling technique



def train_model(clf,X,y,test_df,upsample=False,before_split=False):

    kfold = 10

    skf = StratifiedKFold(n_splits=kfold)

    test_pred = 0.0

    auc_score = 0.0

    if upsample == False:

        print("Only kfold split ...")                    

        for k,(train_idx,valid_idx) in enumerate(skf.split(X,y)):

            print("Split =",k+1)

            X_train,X_valid = X.iloc[train_idx],X.iloc[valid_idx]

            y_train,y_valid = y[train_idx],y[valid_idx]



            ## call classifier 

            clfs,auc,test_pred_proba = fit_clf(clf,X_train,y_train,X_valid,y_valid,test_df)

            test_pred += test_pred_proba

            auc_score += auc

        print("Average AUC Score :",auc_score/kfold)

        return test_pred/kfold

        



    elif upsample == True:

        

        smote = SMOTE(random_state=42)

        if before_split == True:

            print("Upsampling before kfold split.....")

            x_train_sm,y_train_sm =smote.fit_sample(X,y)

        

            for k,(train_idx,valid_idx) in enumerate(skf.split(x_train_sm,y_train_sm)):

                print("Split =",k+1)



                X_train,X_valid = x_train_sm[train_idx],x_train_sm[valid_idx]

                y_train,y_valid = y_train_sm[train_idx],y_train_sm[valid_idx]



                ## call classifier 

                clfs,auc,test_pred_proba = fit_clf(clf,X_train,y_train,X_valid,y_valid,test_df)

                test_pred += test_pred_proba

                auc_score +=auc

            print("Average AUC Score:",auc_score/kfold)

            return test_pred/kfold

        

        else:

            print("Upsampling during kfold split.....")

            for k,(train_idx,valid_idx) in enumerate(skf.split(X,y)):

                

                print("Split =",k+1)



                X_train,X_valid = X.iloc[train_idx],X.iloc[valid_idx]

                y_train,y_valid = y[train_idx],y[valid_idx]



                x_train_sm,y_train_sm =smote.fit_sample(X_train,y_train)

                x_valid_sm,y_valid_sm =smote.fit_sample(X_valid,y_valid)



                ## call classifier 

                clfs,auc,test_pred_proba = fit_clf(clf,x_train_sm,y_train_sm,x_valid_sm,y_valid_sm,test_df)

                test_pred += test_pred_proba

                auc_score +=auc

            print("Average AUC Score:",auc_score/kfold)

            return test_pred/kfold

    else:

        print("None options")

                 

        

    print("Training done!..")
# %%time

# sm = SMOTE(random_state=42)

# x_train_gc,y_train_gc =sm.fit_sample(train_encoded_df[with_target_encoded_cols],y)



# print("x_train_gc dim :",x_train_gc.shape)

# print("y_train_gc dim :",y_train_gc.shape)
#np.logspace(0,4,10)
# Gridsearch taking too much time so commented this code and noted the C value.

# C = np.logspace(0,4,5)

# param_grid = {"penalty":['l2'],"C":C}



# glr = LogisticRegression(solver='lbfgs',max_iter=10000)

# grid_search = GridSearchCV(glr,param_grid=param_grid,cv=10)

# grid_search.fit(train_encoded_df[with_target_encoded_cols],y)



# print("Best Parameters :",grid_search.best_params_)

# print("Best Score :",grid_search.best_score_)
# lr_clf = LogisticRegression(solver='lbfgs',C=166.81,penalty='l2',max_iter=4000) 

# predict_prob = train_model(lr_clf,train_encoded_df[with_target_encoded_cols],y,test_encoded_df[with_target_encoded_cols],upsample=True)
#predict_prob
# predict_prob1 = train_model(LogisticRegression(solver='lbfgs',C=1.0,penalty='l2',max_iter=5000),

#                            train_encoded_df[with_target_encoded_cols],y,test_encoded_df[with_target_encoded_cols],upsample=True)
#predict_prob1
proba_ = train_model(LogisticRegression(solver='lbfgs',C=166.81,penalty='l2',max_iter=5000),

                           train_encoded_df[with_target_encoded_cols],y,test_encoded_df[with_target_encoded_cols],upsample=True)
proba_
# AUC 0.76094191

# lg_reg = LogisticRegression(solver='lbfgs',C=166.81,penalty='l2',max_iter=4000) 

# lg_reg_model = train_model(lg_reg,train_encoded_df[with_target_encoded_cols],y,upsample=True)
# lg_reg_pred_prob = lg_reg_model.predict_proba(test_encoded_df[with_target_encoded_cols])[:,1]

# lg_reg_pred_prob
# AUC 0.7598

# lg_reg_model2 = train_model(lg_reg,train_encoded_df[with_target_encoded_cols],y,upsample=True,before_split=True)

# lg_reg_pred_prob2 = lg_reg_model2.predict_proba(test_encoded_df[with_target_encoded_cols])[:,1]

# lg_reg_pred_prob2
# Increasing the Max Iteration to 10000  AUC 0.7612564509292037



# lg_reg_10000 = LogisticRegression(solver='lbfgs',C=1.0,penalty='l2',max_iter=10000)

# lg_reg_model_1000 = train_model(lg_reg_10000,train_encoded_df[with_target_encoded_cols],y,upsample=True)
# lg_reg_pred_prob_10000 = lg_reg_model_1000.predict_proba(test_encoded_df[with_target_encoded_cols])[:,1]

# lg_reg_pred_prob_10000
submission = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')

submission.head()
# Logistic Regression submission



# submission['target'] = lg_reg_pred_prob

# submission.head()



submission['target'] = proba_

submission.head()

submission.to_csv('submission.csv',index=False)
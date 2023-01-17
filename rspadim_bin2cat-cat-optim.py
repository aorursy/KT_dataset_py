import pandas as pd

print("reading files...")

train  =pd.read_csv("../input/train.csv")

predict=pd.read_csv("../input/test.csv")

bin_cols = [col for col in train.columns if '_bin' in col]

print("done :)")
# from https://www.kaggle.com/rspadim/convert-binary-to-categorical/notebook



import warnings

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin



class BinToCat(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    def fit(self, X, y=None, **kwargs):

        cols=X.columns

        if(len(cols)>64):

            warnings.warn("Caution, more than 64 bin columns, 2**64 can overflow int64")

        for i in cols:

            unique_vals=X[i].unique()

            if(len(unique_vals)>2):

                raise Exception("Column "+i+" have more than 2 values, is it binary? values: "+str(unique_vals))

            if not (0 in unique_vals and 1 in unique_vals):

                raise Exception("Column "+i+" have values different from 0/1, is it binary? values: "+str(unique_vals))

        self.scale=np.array([1<<i for i in range(np.shape(X)[1])])

        

    def transform(self, X):

        return np.sum(self.scale*X,axis=1)

        
a=BinToCat()

a.fit(train[bin_cols])

train['bins']  =a.transform(train[bin_cols])

predict['bins']=a.transform(predict[bin_cols])

# from https://www.kaggle.com/rspadim/categorical-optimization-tree-and-logistic

import time

import numpy as np

from math import factorial

from itertools import permutations

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

from sklearn.metrics import roc_auc_score,log_loss,mean_absolute_error,mean_squared_error,r2_score

from xgboost import XGBClassifier





def getModel(classifier=True,tree_seed=19870425):

    ## tests with xgb

    #return XGBClassifier(max_depth=10000,

    #                     learning_rate=0.1,

    #                     n_estimators=10000, 

    #                     silent=True, 

    #                     objective='binary:logistic', 

    #                     booster='gbtree', 

    #                     n_jobs=1, 

    #                     nthread=None, 

    #                     gamma=0, 

    #                     min_child_weight=1, 

    #                     max_delta_step=0, 

    #                     subsample=1, 

    #                     colsample_bytree=1, 

    #                     colsample_bylevel=1, 

    #                     reg_alpha=0, 

    #                     reg_lambda=1, 

    #                     scale_pos_weight=1, 

    #                     base_score=0.5, 

    #                     random_state=tree_seed, 

    #                     seed=tree_seed, 

    #                     missing=None)

    

    

    if(classifier):

        return DecisionTreeClassifier(max_depth=None,presort=True,criterion='entropy',class_weight='balanced',random_state=tree_seed)

    return DecisionTreeRegressor(max_depth=None,presort=True,random_state=tree_seed)



def getMetric(model):

    ## tests with xgb

    #print(type(model))

    #print(vars(model))

    #print(model._Booster.get_dump())

    #__die

    #if(type(model)==DecisionTreeClassifier):

    #    return model.tree_.max_depth

    #return 0

    return model.tree_.max_depth



# small black magic

def reorderCategorical(df,feature_col,target_col,classifier=True,

                                 max_iterations=721,verbose=False,random_permutation=None,

                                 tree_seed=19870425,random_seed=19870425):

    #time it

    start     = time.time()

    values    =df[feature_col].sort_values().unique() #nd array, since df[col] is a series

    len_values=len(values)



    #min dictionary (l<=>l)

    optimized=False

    default_dict={l:l for l in values}

    min_dict    ={l:l for l in values}

    if(len_values<3):

        if(verbose):

            print(feature_col,': uniques=',len_values,', values=',values)

            print('\t\tLESS THAN 3 UNIQUE VALUES, Time spent (seconds):',time.time() - start)

        return df[feature_col],min_dict

    

    #Current Values

    model=getModel(classifier,tree_seed)

    model.fit(df[feature_col].values.reshape(-1,1),df[target_col])

    min_depth_count=getMetric(model)

    if(verbose):

        print(feature_col,': uniques=',len_values,', depth=',min_depth_count,', values=',values)

        if(classifier):

            print('\t\tROC_AUC/LogLoss: ',

                      roc_auc_score(df[target_col],model.predict_proba(df[feature_col].values.reshape(-1,1))[:,1] ),'/',

                      log_loss(     df[target_col],model.predict_proba(df[feature_col].values.reshape(-1,1))[:,1]))

        else:

            print('\t\tMAE/MSE/R²: ',

                      mean_absolute_error(df[target_col],model.predict(df[feature_col].values.reshape(-1,1))[:,1] ),'/',

                      mean_squared_error( df[target_col],model.predict(df[feature_col].values.reshape(-1,1))[:,1]),'/',

                      r2_score(           df[target_col],model.predict(df[feature_col].values.reshape(-1,1))[:,1]))

    if(min_depth_count==1):

        if(verbose):

            print('\t\tDEPTH=1, Time spent (seconds):',time.time() - start)

        return df[feature_col],min_dict

    

    #Naive order by count

    if(classifier):

        first_try=df[df[target_col]==0].groupby(feature_col)[feature_col].count().sort_values(ascending=True)

    else:

        #maybe a median/mean order? for example, target_col>mean(target) ?

        first_try=df.groupby(feature_col)[feature_col].count().sort_values(ascending=True)

    l,values_dict=0,{}

    for i in first_try.index:

        values_dict[values[l]]=i

        l+=1

    

    model=getModel(classifier,tree_seed)

    model.fit(df[feature_col].replace(values_dict).values.reshape(-1,1),df[target_col])

    # better than l<=>l ?

    if(min_depth_count>getMetric(model)):

        optimized=True

        if(verbose):

            print('\tNaive order by count: from ',min_depth_count,' to ',getMetric(model),', dict:',min_dict)

            if(classifier):

                print('\t\tROC_AUC/LogLoss: ',

                          roc_auc_score(df[target_col],model.predict_proba(df[feature_col].replace(values_dict).values.reshape(-1,1))[:,1] ),'/',

                          log_loss(     df[target_col],model.predict_proba(df[feature_col].replace(values_dict).values.reshape(-1,1))[:,1]))

            else:

                print('\t\tMAE/MSE/R²: ',

                          mean_absolute_error(df[target_col],model.predict(df[feature_col].replace(values_dict).values.reshape(-1,1))[:,1] ),'/',

                          mean_squared_error( df[target_col],model.predict(df[feature_col].replace(values_dict).values.reshape(-1,1))[:,1]),'/',

                          r2_score(           df[target_col],model.predict(df[feature_col].replace(values_dict).values.reshape(-1,1))[:,1]))

        min_depth_count,min_dict=getMetric(model),values_dict

        if(min_depth_count==1):

            if(verbose):

                print('\t\tDEPTH=1, Time spent (seconds):',time.time() - start)

            return df[feature_col].replace(values_dict),values_dict

    elif(verbose):

        print('\t\t=[ No optimization using naive order by Count')

    

    # Search Space:

    # maybe random_permutatition isn't the best method... 

    #     if len(permutations)~=factorial(len_values) < max_iterations, we can use permutatition (real brute force)

    if(random_permutation==None):

        random_permutation=False

        if(factorial(len_values)>max_iterations):

            random_permutation=True

            if(verbose):

                print('\t\tToo big search space, using RANDOM SAMPLING')

        elif(verbose):

            print('\t\tmax_iterations (',max_iterations,') >Factorial(length) (',factorial(len_values),'), USING PERMUTATION')

    

    # TODO: maybe we can do better with GA ?!

    if(random_permutation):

        # random permutation ( good lucky =] )

        np.random.seed(random_seed)

        space=range(max_iterations)

    else:

        # default itertools permutation

        space=permutations(values)



    count=0

    for perm in space:

        if(count>max_iterations):

            break

        # random permutation

        if(random_permutation):

            perm=np.random.permutation(values)

        

        values_dict={values[i]:perm[i] for i in range(0,len_values)}

        model=getModel(classifier,tree_seed)

        model.fit(df[feature_col].replace(values_dict).values.reshape(-1,1),df[target_col])

        if(min_depth_count>getMetric(model)):

            optimized=True

            if(verbose):

                print('\t',count,'/',max_iterations,'NEW!!! from',min_depth_count,' to ',getMetric(model),' dict:',values_dict)

                if(classifier):

                    print('\t\tROC_AUC/LogLoss: ',

                              roc_auc_score(df[target_col],model.predict_proba(df[feature_col].replace(values_dict).values.reshape(-1,1))[:,1] ),'/',

                              log_loss(     df[target_col],model.predict_proba(df[feature_col].replace(values_dict).values.reshape(-1,1))[:,1]))

                else:

                    print('\t\tMAE/MSE/R²: ',

                              mean_absolute_error(df[target_col],model.predict(df[feature_col].replace(values_dict).values.reshape(-1,1))[:,1] ),'/',

                              mean_squared_error( df[target_col],model.predict(df[feature_col].replace(values_dict).values.reshape(-1,1))[:,1]),'/',

                              r2_score(           df[target_col],model.predict(df[feature_col].replace(values_dict).values.reshape(-1,1))[:,1]))

            min_depth_count,min_dict=getMetric(model),values_dict

            if(min_depth_count==1):

                print('\t\tDEPTH=1')

                break

        count+=1

    if(verbose):

        print('\t\tTime spent (seconds):',time.time() - start)

    if(not optimized):

        return df[feature_col],default_dict

    return df[feature_col].replace(values_dict),values_dict

# SINGLE THREAD

_,values_dict=reorderCategorical(train,'bins','target',verbose=True,max_iterations=100)

train['bins_reordered']  =train['bins'].replace(values_dict)

predict['bins_reordered']=predict['bins'].replace(values_dict)

print('Nice job! =]')
train.to_csv(  'train.bin2cat-reordered.csv',index=False)

predict.to_csv('test.bin2cat-reordered.csv',index=False)
from sklearn import tree

from graphviz import Source

import matplotlib.pyplot as plt
#output tree using binaries:)

model=DecisionTreeClassifier(criterion='gini',class_weight='balanced',max_depth=None)

# dataset 1

model.fit(train[bin_cols],train['target'])

y_hat1=model.predict_proba(train[bin_cols])[:,1]

loss1 =log_loss(train['target'],y_hat1)

print('    depth:  ',model.tree_.max_depth)

print('    logloss:',loss1)
#plot tree :)

Source( tree.export_graphviz(model, out_file=None))
model.fit(train['bins_reordered'].reshape(-1,1),train['target'])

y_hat1=model.predict_proba(train['bins_reordered'].reshape(-1,1))[:,1]

loss1 =log_loss(train['target'],y_hat1)

print('    depth:  ',model.tree_.max_depth)

print('    logloss:',loss1)
#plot tree :)

Source( tree.export_graphviz(model, out_file=None))
import pandas as pd

print("reading files...")

train  =pd.read_csv("../input/train.csv")

predict=pd.read_csv("../input/test.csv")

cat_cols = [col for col in train.columns if '_cat' in col]

print("done :)")
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



#MULTI THREAD VERSION:

import psutil 

import threading



lock = threading.Lock()

def threaded_function(args):

    global train,predict,lock

    #print('cat_cols:',len(args))

    for i in args:

        reordered,values_dict=reorderCategorical(train,i,'target',verbose=True)

        with lock:

            train[  i+'_reordered']=reordered

            predict[i+'_reordered']=predict[i].replace(values_dict)

        print(i)



if __name__ == "__main__":

    # 4 threads

    print("Dream machine: :P, 128GB, 16cores")

    print('cores: ',psutil.cpu_count(),' threads:',psutil.cpu_count(logical=False),

         'freq: ',psutil.cpu_freq())

    print('memory: ',psutil.virtual_memory())

    print('swap: ',psutil.swap_memory())



    cores=16

    lencat =len(cat_cols)

    lencatdiv=lencat//cores

    start_end=[]

    for i in range(0,cores):

        if(i==cores-1):

            start_end.append([i*lencatdiv+1,lencat]) # last one

        else:

            start_end.append([i*lencatdiv+1,lencatdiv*(i+1)])

    print('start/end: ',len(start_end))

    threads,l=[],0

    

    for i in start_end:

        #print("cols: ",i[0],i[1],' - ',cat_cols[i[0] : i[1]])

        #print("thread: ",l)

        threads.append( threading.Thread(target = threaded_function, args = (cat_cols[i[0] : i[1]],) ) )

        threads[l].start()

        l+=1

    l=0

    for i in start_end:

        threads[l].join()

        l+=1

    print("thread finished...exiting")

## SINGLE THREAD

#for i in cat_cols:

#    train[i+'_reordered'],values_dict=reorderCategorical(train,i,'target',verbose=True,max_iterations=5)

#    predict[i+'_reordered']=predict[i].replace(values_dict)

#print('Nice job! =]')
train.to_csv(  'Reordered-train.csv',index=False)

predict.to_csv('Reordered-test.csv',index=False)
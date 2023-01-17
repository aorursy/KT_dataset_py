import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



dat = pd.read_csv('../input/creditcard.csv')



print(dat.head())

print('\nThe distribution of the target variable\n')

dat['Class'].value_counts()
print(dat.describe())



columns = dat.columns[1:30] 

fig, axes = plt.subplots(nrows=6, ncols=5,figsize=(10,10))

axes = axes.flatten()



for i in range(29):

  axes[i].hist(dat[columns[i]], normed=1,facecolor='b',alpha=0.75)

  axes[i].set_title(columns[i])

  plt.setp(axes[i].get_xticklabels(), visible=False) 

  plt.setp(axes[i].get_yticklabels(), visible=False) 



plt.setp(axes[29].get_xticklabels(), visible=False) 

plt.setp(axes[29].get_yticklabels(), visible=False) 
import random



random.seed(1234)



Class = dat['Class'].values

dat2 = dat.drop(['Class'], axis=1)



allIndices = np.arange(len(Class))



numTrain = int(round(0.40*len(Class)))

numValid = int(round(0.30*len(Class)))

numTest = len(Class)-numTrain-numValid



inTrain = sorted(np.random.choice(allIndices, size=numTrain, replace=False))

inValidTest = list(set(allIndices)-set(inTrain))

inValid= sorted(np.random.choice(inValidTest, size=numValid, replace=False))

inTest = list(set(inValidTest)-set(inValid))



train = dat2.iloc[inTrain,:]

valid= dat2.iloc[inValid,:]

test =  dat2.iloc[inTest,:]



trainY = Class[inTrain]

validY = Class[inValid]

testY = Class[inTest]
import xgboost as xgb



dtrain = xgb.DMatrix(train, label=trainY)

dvalid = xgb.DMatrix(valid, label=validY)

dtest = xgb.DMatrix(test, label=testY)



## fixed parameters

scale_pos_weight = sum(trainY==0)/sum(trainY==1)  

num_rounds=10 # number of boosting iterations



param = {'silent':1,

         'min_child_weight':1, ## unbalanced dataset

         'objective':'binary:logistic',

         'eval_metric':'auc', 

         'scale_pos_weight':scale_pos_weight}



def do_train(param, train,train_s,trainY,valid,valid_s,validY):

    ## train with given fixed and variable parameters

    ## and report performance on validation dataset

    evallist  = [(train,train_s), (valid,valid_s)]

    model = xgb.train( param, train, num_boost_round=num_rounds, 

                      evals=evallist )    

    preds = model.predict(valid)

    labels = valid.get_label()

      

    act_pos=sum(validY==1)

    act_neg=valid.num_row()-act_pos

    true_pos=sum(1 for i in range(len(preds)) if (preds[i]>=0.5) & (labels[i]==1))

    false_pos=sum(1 for i in range(len(preds)) if (preds[i]>=0.5) & (labels[i]==0))

    false_neg=act_pos-true_pos

    true_neg=act_neg-false_pos

    

    ## precision: tp/(tp+fp) percentage of correctly classified predicted positives

    ## recall: tp/(tp+fn) percentage of positives correctly classified

    ## F-score with beta=1

    ## see Sokolova et al., 2006 "Beyond Accuracy, F-score and ROC:

    ## a Family of Discriminant Measures for Performance Evaluation"

    ## fscore <- 2*precision.neg*recall.neg/(precision.neg+recall.neg)

    

    precision = true_pos/(true_pos+false_pos)

    recall = true_pos/(true_pos+false_neg)

    f_score = 2*precision*recall/(precision+recall)  

    

    print('\nconfusion matrix')

    print('----------------')

    print( 'tn:{:6d} fp:{:6d}'.format(true_neg,false_pos))

    print( 'fn:{:6d} tp:{:6d}'.format(false_neg,true_pos))

    return(f_score)    

from collections import OrderedDict



## parameters to be tuned

tune_dic = OrderedDict()



tune_dic['max_depth']= np.array([20,25,30]) ## maximum tree depth

tune_dic['colsample_bytree']= np.linspace(0.5,1.0,6) ## subsample ratio of columns

tune_dic['eta']= np.linspace(0.3,0.6,4) ## learning rate



best_params = dict()

best_f_score = -1



import itertools

var_params = [ i for i in itertools.product(*tune_dic.values())]

search=np.random.choice(np.arange(len(var_params)),10,replace=False)



columns=[*tune_dic.keys()]+['F Score']



results = pd.DataFrame(index=range(len(search)), columns=columns) ## to check results



for i in range(len(search)): ## len(search)

    

    for (key,val) in zip(tune_dic.keys(),var_params[search[i]]):

        param[key]=val



    print()    

    f_score = do_train(param, dtrain,'train',trainY,dvalid,'valid',validY)

    

    results.loc[i,[*tune_dic.keys()]]=var_params[search[i]]

    results.loc[i,'F Score']=f_score

    

    if f_score > best_f_score:

        best_f_score = f_score

        print('\n*** better f-score',f_score)

        for (key,val) in zip(tune_dic.keys(),var_params[search[i]]):

            best_params[key]=val        

            print(key,': ',val,' ',end='')

        print()    

       
print('\nevaluation on the test dataset\n')  



for (key,val) in best_params.items():

    print(key,': ',val,' ',end='')

    param[key]=val

print('\n\n')

    

best_f_score = do_train(param, dtrain,'train',trainY,dtest,'test',testY)

print('\nf-score on the test dataset: {:6.2f}'.format(best_f_score))
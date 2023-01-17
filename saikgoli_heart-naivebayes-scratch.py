import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import scipy



df=pd.read_csv("../input/heart.csv")

df.head()

cols_cat=['age','cp','restecg','ca','slope','sex','fbs','thal']

cols_num=['trestbps', 'chol','thalach','exang', 'oldpeak']

df['age']=(df['age']/10).astype(int)
#shuffel input data

df=df.sample(frac=1,random_state=42)

#take training data 70percent

df_train=df.sample(frac=0.7,random_state=42)



#make test sample from df

df_test=df.drop(index=df_train.index)



#drop indices

df_train.reset_index(drop=True,inplace=True)

df_test.reset_index(drop=True,inplace=True)



#make test input data set and output vector 

X_test=df_test[df_test.columns[:-1]]

y_test=df_test['target']
df_test.head()
#g_mean,g_std for mean & s.d of continious variables g:gaussian 

g_mean=df_train.groupby('target')[cols_num].mean()

g_std =df_train.groupby('target')[cols_num].mean()



#find log probabilities of the target variables

ytrue_plog=np.log(len(df_train['target']==1)/len(df_train))

yfalse_plog=np.log(len(df_train['target']==0)/len(df_train))

#dictionary for naive bayes categorical variables to fill with log probabilities

dict_cat={0:{},1:{}}



for cc in cols_cat:

    x=df_train.groupby(['target',cc])['target'].size().unstack(fill_value=0).stack()

    x=x.astype(float)

    x+=1

    dict_cat[0][cc]=np.log(x[0]/x[0].sum())

    dict_cat[1][cc]=np.log(x[1]/x[1].sum())

    
#function to calculate the log probabilities of the test data instance

def findprob(x):

    test_index=x.index

    prob_1=0

    for cc in test_index:

        if cc in cols_cat:

            prob_1+=dict_cat[1][cc][x[cc]]

        else :

            prob_1+=np.log(scipy.stats.norm(g_mean[cc][1],g_std[cc][1]).pdf(x[cc]))

        prob_1+=ytrue_plog



    prob_0=0

    for cc in test_index:

        if cc in cols_cat:

            prob_0+=dict_cat[0][cc][x[cc]]

        else :

            prob_0+=np.log(scipy.stats.norm(g_mean[cc][0],g_std[cc][0]).pdf(x[cc]))

        prob_0+=yfalse_plog

    

    return int(prob_1>prob_0)

#prediction vector

pred=np.zeros(shape=len(X_test))



#ignore exceptions, twoexceptions may occured with random state 

for i in range(0,len(X_test)):

    try:

        pred[i]=findprob(X_test.iloc[i])

    except:

        pred[i]=np.nan

        

#print the accuracy of model

print(sum(y_test==pred)/len(y_test))
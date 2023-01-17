import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

train = pd.read_csv('/kaggle/input/cs-challenge/training_set.csv', index_col="ID")
train = train.dropna(axis=1)
column_list = column_list = train.columns.to_list()

non_redundant_cols = [x for x in column_list if x.find('_max') == -1 and x.find('_min') == -1 and x.find('_c') == -1]
non_redundant_cols
#train_non_re = train[non_redundant_cols]

train_non_re=train
def ind_max(l):

    M=l[0]

    ind=0

    for i in range(1,len(l)):

        if l[i]>M:

            M=l[i]

            ind=i

    return ind
#squared_cols = []

pows=[-5+i/2 for i in range(21)]

pows.remove(0.0)

res=[]

#for col in non_redundant_cols:

for col in column_list:

    if col != 'MAC_CODE':

        corr=[]

        for p in pows:

            if (p%1==0 or not any(train[col]<0)) and (p>0 or not any(train[col]==0)):                

                    corr.append(abs(train_non_re['TARGET'].corr(train_non_re[col]**p)))

            else:

                corr.append(0)

        p=pows[ind_max(corr)]

        res.append(p)

        train_non_re[col] = np.power(train_non_re[col], p)

res

            

            

            #cor1 = abs(train_non_re['TARGET'].corr(train_non_re[col]))

            #cor2 = abs(train_non_re['TARGET'].corr(train_non_re[col]**2))

            #if(cor2 > cor1):

                #train_non_re[col] = np.power(train_non_re[col], 2)

                #squared_cols.append(col)

        
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import RidgeCV

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer





col_transformer = ColumnTransformer([

    ('MAC_CODE', OneHotEncoder(dtype='int'),['MAC_CODE'])],

    remainder = StandardScaler())



reg = RidgeCV(cv=5)



pipe = Pipeline([('col_transformer', col_transformer),('reg', reg)], verbose=True)

train_non_re
from sklearn.model_selection import train_test_split



Xtr, Xte, ytr,  yte = train_test_split(train_non_re.drop('TARGET', axis=1), train_non_re['TARGET'], test_size=0.2)



pipe.fit(Xtr,ytr)
from sklearn.metrics import mean_absolute_error



mean_absolute_error(yte, pipe.predict(Xte))
pipe.fit(train_non_re.drop('TARGET', axis=1),train_non_re['TARGET'])
test = pd.read_csv('/kaggle/input/cs-challenge/test_set.csv', index_col="ID")

#test = test[[x for x in non_redundant_cols if x != 'TARGET']]

test = test[[x for x in column_list if x != 'TARGET']]



for col in squared_cols:

    test[col] = np.power(test[col],2)



predict = pipe.predict(test)
test['TARGET'] = predict

test['TARGET'].to_csv("squared_ridge.csv")
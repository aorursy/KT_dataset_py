import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

df_train = pd.read_csv('../input/train.csv')

df_train.head()
line_cnt = len(df_train)-1

train_new = df_train[2:line_cnt].sort_values(by='question1')

print('start')

i=2

df_obj = pd.DataFrame()

while i< line_cnt:



    if i==0:

        break

    else:

        i=i-1

        print(i)

        for line_out in train_new[i:line_cnt].values:

            q1=line_out[3]

            q2=line_out[4]

            flg_out=line_out[5]

            if flg_out==0:

                #print(line_out[0])

                for line_in in train_new[i+1:line_cnt].values:

                    q3=line_in[3]

                    q4=line_in[4]                

                    flg_in=line_in[5]

                    if(q1==q3 and flg_in==0):

                        print('out: '+'%d' %line_out[0])

                        print('in: '+'%d' %line_in[0])

                        newline.add(q2,q4)



for newline in df_obj.values:

    print(newline[0]+' '+newline[1])

        

        
line1=df_train[1:2] 

print(len(df_train))
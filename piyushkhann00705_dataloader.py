# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd

import pickle



traindf= pd.read_csv("../input/audiolstm/train_split2.csv")

testdf=pd.read_csv("../input/audiolstm/test_split2.csv")

valdf=pd.read_csv("../input/audiolstm/val_split2.csv")



with open('../input/audiolstm/text_train2.pkl', 'rb') as f:

    text_train=pickle.load(f)

    

with open('../input/audiolstm/text_test2.pkl', 'rb') as f:

    text_test=pickle.load(f)

    

with open('../input/audiolstm/text_val2.pkl', 'rb') as f:

    text_val=pickle.load(f)
error=[]



def ModTextData(df,text_dict):

    X=[]

    for index,row in df.iterrows():

        try:

            X.append(text_dict[row['text_file_name'][:-9]]['out'])

        except:

            error.append(row['text_file_name'][:-9])

    return X



X_train=ModTextData(traindf,text_train)   

X_test=ModTextData(testdf,text_test)   

X_val=ModTextData(valdf,text_val)   
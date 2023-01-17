# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LinearRegression







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

file= pd.read_csv("../input/train_V2.csv")

sample=file



sample.drop('Id',axis=1,inplace=True)

sample.drop('groupId',axis=1,inplace=True)

sample.drop('matchId',axis=1,inplace=True)

sample.drop('matchType',axis=1,inplace=True)



test_sample=pd.read_csv("../input/test_V2.csv")



test_sample.drop('Id',axis=1,inplace=True)

test_sample.drop('groupId',axis=1,inplace=True)

test_sample.drop('matchId',axis=1,inplace=True)

test_sample.drop('matchType',axis=1,inplace=True)



for i, row in sample.iterrows():

    if sample.at[i,'winPlacePerc']<0.5:

        sample.at[i,'winPlacePerc'] = 0

    else :

        sample.at[i,'winPlacePerc'] = 1



x=np.array(sample.drop(['winPlacePerc'],1))

y=np.array(sample['winPlacePerc'])



test_x=np.array(test_sample)



regressor = LinearRegression() 

regressor.fit(x,y)



pred=regressor.predict(test_x)



pr=[]

for y in np.nditer(pred, op_flags=['readwrite']):

    if y>0.5:

        pr.append(1)    

    else:

        pr.append(0)



file2= pd.read_csv("../input/test_V2.csv")



submission=pd.DataFrame({

    "Id" :file2["Id"],

    "winPlacePerc" :pr

    

})



submission.to_csv("sub2.csv",index=False)

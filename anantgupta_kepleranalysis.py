# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

data=pd.read_csv('../input/exoTest.csv')
import matplotlib.pyplot as plt

colors = {'1.0':'red', '2.0':'blue'}

plt.figure(figsize=(20,10))

for x in range(data.shape[0]):

    if(data.values[x,0]==1):

        plt.plot(data.values[x,1:],color=colors[str(data.values[x,0])],alpha=0.4)  

plt.show()



plt.figure(figsize=(20,10))

for x in range(data.shape[0]):

    if(data.values[x,0]==2):

        plt.plot(data.values[x,1:],color=colors[str(data.values[x,0])],alpha=0.4)  

plt.show()



# As we can see that we need to have different models for different ranges of data points




# FIRST CUT : SIMPLE LOGISTIC

columnNames=[x for x in data.columns.values if x != 'LABEL']

from sklearn.model_selection import train_test_split



datax,datatest,y,ytest=train_test_split(data[columnNames],data['LABEL'],test_size=0.33,random_state=42)

from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

lr=LogisticRegression(C=0.1,class_weight={1:1,2:300})

lr.fit(datax,y)



from sklearn.metrics import confusion_matrix

confusion_matrix(ytest,lr.predict(datatest[columnNames]))

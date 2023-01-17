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
df=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

df
import seaborn as sns

sns.pairplot(df.iloc[:,1:6])
df.info()
df.describe()
#Output Values

y=df.Outcome.values

y
#Other Values

x_data=df.drop(['Outcome'],axis=1) #Dropped Outputs. thats column name is 'Outcome'

x_data
#Normalization 

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

x
#Train-Test Split



from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=30) # test_size=0.2 so %20 test %80 train



print("x_train: ",x_train)

print("x_test: ",x_test)

print("y_train: ",y_train)

print("y_test: ",y_test)



#print how many variables

print("x_train : ",x_train.shape)

print("x_test : ",x_test.shape)

print("y_train : ",y_train.shape)

print("y_test : ",y_test.shape)

#Using sklearn library for LogisticRegression

from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(x_train,y_train)

print("-test accuracy{}".format(lr.score(x_test,y_test)))
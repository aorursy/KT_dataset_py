# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
training_data = pd.read_csv("../input/minor-project-2020/train.csv");

testing_data = pd.read_csv("../input/minor-project-2020/test.csv");
data1 = training_data.to_numpy()

X_train = data1[:,1:-1]

train_length = X_train.shape[0]

Y_train = data1[:,-1:].flatten()
testing_data.drop('id',axis=1,inplace=True)

data2 = testing_data.to_numpy()

X_combined = np.append(X_train,data2,axis=0)
from sklearn.preprocessing import StandardScaler



re = StandardScaler().fit(X_combined)

X_combined = re.transform(X_combined)
X_train = X_combined[:train_length,:]

x_test = X_combined[train_length:,:]
from sklearn.linear_model import LogisticRegression

litreg = LogisticRegression(solver='liblinear',penalty='l1')

# we searched in gridsearch using parameters soolver and penalty and after running on local machine these 2 were best. 

# parameter = {

#penalty':['l1','l2','none'], solver:['liblinear']

#}
litreg.fit(X_train,Y_train)
y_probability = litreg.predict_proba(x_test)

finalans = y_probability[:,-1:]
test = pd.read_csv("../input/minor-project-2020/test.csv");
test_ids = test[{'id'}]
probabilities_dataframe = pd.DataFrame(finalans)
finale = pd.concat([test_ids,probabilities_dataframe],axis=1)
finale.columns = ['id','target']
finale.to_csv('prob.csv',index=False)
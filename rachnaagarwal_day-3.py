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
import pandas as pd

from sklearn.model_selection import train_test_split

apndcts = pd.read_csv("../input/apndcts/apndcts.csv")

print(apndcts.shape)

data_train,data_test=train_test_split(apndcts,test_size=0.3)

print(data_train.shape)

print(data_test.shape)
#Kfold

import pandas as pd

from sklearn.model_selection import KFold

apndcts = pd.read_csv("../input/apndcts/apndcts.csv")

kf=KFold(n_splits=9)

for train_index,test_index in kf.split(apndcts):

    data_train=apndcts.iloc[train_index]

    data_test=apndcts.iloc[test_index]

print(data_train.shape)

print(data_test.shape)

    

from sklearn.utils import resample

X= apndcts.iloc[:,0:9]

resample(X,n_samples=200,random_state=1)
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix



data = pd.read_csv("../input/apndcts/apndcts.csv")



predictors = data.iloc[:,0:7] # Segregating the predictors

target = data.iloc[:,7] # Segregating the target/class

predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size = 0.3, random_state = 123) # Holdout of data

dtree_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5) #Model is initialized

# Finally the model is trained

model = dtree_entropy.fit(predictors_train, target_train)

prediction = model.predict(predictors_test)



acc_score = 0



acc_score = accuracy_score(target_test, prediction, normalize = True)

print(acc_score)

conf_mat = confusion_matrix(target_test, prediction)

print(conf_mat)

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
#importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#importing dataset

train = pd.read_csv('/kaggle/input/datamaestro2020/astro_train.csv')

test = pd.read_csv('/kaggle/input/datamaestro2020/astro_test.csv')
y_train= train['class']

train = train.drop('class', axis= 1)



dataset = pd.concat([train, test])

dataset = dataset.drop(['id','skyVersion', 'camCol', 'run', 'rerun'], axis= 1)

dataset["err_g_log"] = np.log(dataset["err_g"])

dataset["err_i_log"] = np.log(dataset["err_i"])

dataset["err_r_cbrt"] = np.cbrt(dataset["err_r"])

dataset["err_u_log"] = np.log(dataset["err_u"])

dataset["err_z_log"] = np.log(dataset["err_z"])

dataset["obj_log"] = np.log(dataset["obj"])

dataset["photoz_log"] = np.log(dataset["photoz"])

dataset["dec_cbrt"] = np.cbrt(dataset["dec"])







dataset = dataset.drop(['#ra',"err_g","err_i","err_r","err_u","err_z", 'field', 'obj',"photoz", 'dec'], axis= 1)



ss = dataset.iloc[:, :].values



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

transformed_dataset = sc.fit_transform(ss)



train_prepared = transformed_dataset[:45000, :]

test_prepared = transformed_dataset[45000:, :]
#over-sampling data

from imblearn.over_sampling import SMOTE

smote = SMOTE()

x_res, y_res = smote.fit_sample(train_prepared, y_train)
#creating model using random forest classifier

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 250,criterion = 'entropy', random_state = 0)

classifier.fit(x_res, y_res)
#predicting output

y_pre = classifier.predict(test_prepared)
#checking accuracy

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_prepared, y_train, test_size=0.2)



#from sklearn.model_selection import cross_val_score

#accuracies = cross_val_score(estimator = classifier, X= X_train, y = y_train, cv = 3)

#accuracies

#saving predictionsn to csv file

y = pd.DataFrame(y_pre)

#y.to_csv('/content/drive/My Drive/filename.csv')
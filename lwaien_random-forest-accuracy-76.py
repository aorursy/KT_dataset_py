# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

train.head()
plt.subplots(figsize=(20,10))

sns.heatmap(train.corr(),annot=True, vmax=2, square=True)
cor = train.corr().abs()

deathcor = cor['DEATH_EVENT'].sort_values(ascending=False)

deathcor
death = train['DEATH_EVENT']

creatinine = train['serum_creatinine']

ejection = train['ejection_fraction']

age = train['age']

sodium = train['serum_sodium']



sns.boxplot(x=death,y=creatinine)
sns.boxplot(x=death,y=ejection)
sns.boxplot(x=death,y=age)
sns.boxplot(x=death,y=sodium)

sns.swarmplot(x=death,y=sodium, color='0.25')




y = train['DEATH_EVENT']

features = ['age','serum_creatinine','ejection_fraction','serum_sodium']

x = train[features]

x
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .33, random_state = 40)



classifier = RandomForestClassifier(n_estimators=100, random_state=1)



classifier.fit(x_train,y_train)



x_train.shape,x_test.shape,y_train.shape,y_test.shape



accuracy = classifier.predict(x_test)



print(accuracy_score(y_test,accuracy))

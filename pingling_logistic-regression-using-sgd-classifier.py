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

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

classify = pd.read_csv('/kaggle/input/cte-ml-hack-2019/train_real.csv')

classify.head()
classify.describe()
classify.info()
class1 = pd.get_dummies(classify.Soil, prefix ='soil')

class1.head()
classify = classify.join(class1)

classify.drop('Soil', axis =1 , inplace = True)

classify.corr()
y = classify.label.copy()

X = classify.drop('label', axis = 1)

X.drop('Altitude',axis=1,inplace = True)

X.drop('Hillshade_3pm',axis = 1, inplace = True)

X.drop('Id', axis =1 , inplace = True)

X.drop('soil_Soil_Type_7',axis =1 , inplace = True)

X.drop('soil_Soil_Type_8', axis =1 , inplace = True)

X.drop('soil_Soil_Type_40', axis =1 , inplace = True)

X.drop('soil_Soil_Type_36', axis =1 , inplace = True)

X.drop('soil_Soil_Type_37', axis = 1 , inplace = True)

X.drop('soil_Soil_Type_25', axis =1 , inplace = True)

X.info()
from sklearn.model_selection import train_test_split

  # 80 % go into the training test, 20% in the validation test

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=15)
from sklearn.linear_model import LogisticRegression

f,ax=plt.subplots(1,2,figsize=(18,8))

classify['label'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('label')

ax[0].set_ylabel('')

sns.countplot('label',data=classify,ax=ax[1])

ax[1].set_title('label')

plt.show()
from sklearn.linear_model import SGDClassifier
model = LogisticRegression(class_weight = 'balanced')


model.fit(X_train,y_train)


model.score(X_train,y_train)

model.score(X_valid,y_valid)
Test = pd.read_csv('/kaggle/input/cte-ml-hack-2019/test_real.csv')
class2 = pd.get_dummies(Test.Soil, prefix = 'Soil')


Test = Test.join(class2)

Test.drop('Soil', axis =1 , inplace = True)



Test.drop('Altitude', axis =1 , inplace = True)

Test.drop('Hillshade_3pm', axis =1 , inplace = True)

Test.drop('Id', axis = 1 ,inplace = True)





Test.head()
predictions = model.predict(Test)
model.predict(Test)
submission= pd.DataFrame()
Test_df = pd.read_csv('/kaggle/input/cte-ml-hack-2019/test_real.csv')
submission['Id'] = Test_df['Id']
submission['Predicted'] = predictions.tolist()
submission.tail()
submission.to_csv('Submissionx.csv',index=False)
!ls
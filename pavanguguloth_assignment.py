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
data=pd.read_csv('/kaggle/input/math-students/student-mat.csv')

data
school=data.groupby('school')

school.first()
data.isna().sum()
data.describe()
data['Overall'] = np.where(data['G3'] >=10.415, 'Pass', 'Fail')

data.head()
data.describe()
fail_by_gender=data[(data['Overall']=='Fail')]['sex'].value_counts()

fail_by_gender.plot.bar()
pass_by_gender=data[(data['Overall']=='Pass')]['sex'].value_counts()

pass_by_gender.plot.bar()
columns_list=data.columns

data.romantic.head()
import matplotlib.pyplot as plt

fail=data[(data['Overall']=='Fail') & (data['romantic']=='yes')]['sex'].value_counts()

passed=data[(data['Overall']=='Fail') & (data['romantic']=='no')]['sex'].value_counts()



fail,passed
fail.plot.bar()
passed.plot.bar()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

cols=data.columns[data.dtypes==object].tolist()

for col in cols:

    n = len(data[col].unique())

    if (n == 2):

        data[col] = pd.get_dummies(data[col], drop_first=True)
data=pd.get_dummies(data)

data=pd.get_dummies(data)

data
check_for_school=data[data['Overall']=='Fail'].groupby('school')

check_for_school
cols=data.columns.tolist()

cols
new_corr=data.corr()

new_corr['G3'].sort_values(ascending=False)
data=data.drop(['G1','G2','G3','activities','guardian_mother','guardian_father','guardian_other','famsup','reason_home','Fjob_services','Fjob_at_home'],axis=1)

data.head()
y=data.Overall

x=data.drop('Overall',axis=1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

print("train:",len(x_train),"test:",len(x_test))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

lr = LogisticRegression()

lr.fit(x_train, y_train)

predictions = lr.predict(x_test)

cm = confusion_matrix(y_test, predictions)

print(accuracy_score(y_test,predictions)*100)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 50)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

print(accuracy_score(y_test,y_pred)*100)
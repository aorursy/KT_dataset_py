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
data = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
data.head()
data.isnull().sum()
data1 = data.fillna(0)

data1
from sklearn.preprocessing import LabelEncoder



my_label_Encoder = LabelEncoder()



data1['status'] = my_label_Encoder.fit_transform(data1['status'])



data1.dtypes
# "1" represent Male

# "0" represent Female

data1['status'].value_counts()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



values = sns.countplot(data['status'],hue=data['gender'])







for total_count in values.patches:

    height = total_count.get_height()

    width=total_count.get_x()+total_count.get_width()/2.

    values.text(total_count.get_x()+total_count.get_width()/2.,height + 3,'{:1.2f}'.format(height),ha="center") 





    

total_students = data1['gender'].value_counts()

print("Number of male and female\n",total_students)



values = sns.countplot(data['status'],hue=data['degree_t'])



# The below code is to see the total values of each bar on the top

for total_count in values.patches:

    height = total_count.get_height()

    width=total_count.get_x()+total_count.get_width()/2.

    values.text(total_count.get_x()+total_count.get_width()/2.,height + 3,'{:1.2f}'.format(height),ha="center")

    

degrees = data1['degree_t'].value_counts()

print("Students with different degrees: \n",degrees)

values = sns.countplot(data['status'],hue=data['specialisation'])



# The below code is to see the total values of each bar on the top

for total_count in values.patches:

    height = total_count.get_height()

    width=total_count.get_x()+total_count.get_width()/2.

    values.text(total_count.get_x()+total_count.get_width()/2.,height + 3,'{:1.2f}'.format(height),ha="center") 

    

total_students_specialisation = data1['specialisation'].value_counts()

print("Number of students in each specialisation\n",total_students_specialisation)
values = sns.countplot(data['status'],hue=data['hsc_b'])



# The below code is to see the total values of each bar on the top

for total_count in values.patches:

    height = total_count.get_height()

    width=total_count.get_x()+total_count.get_width()/2.

    values.text(total_count.get_x()+total_count.get_width()/2.,height + 3,'{:1.2f}'.format(height),ha="center") 



hsc_board = data1['hsc_b'].value_counts()

print("Number of students in different hsc boards\n",hsc_board)

sns.regplot(x='ssc_p',y='hsc_p',data=data1)



print("We can see the students perfomed well in ssc also performed well in hsc as well and students who did not perform well in ssc did not perform well in hsc but some performed well in hsc")
sns.regplot(x='hsc_p',y='degree_p',data=data1)
sns.regplot(x='degree_p',y='mba_p',data=data1)

sns.catplot(x="status", y="ssc_p", data=data,kind="swarm",hue='gender')

sns.catplot(x="status", y="hsc_p", data=data,kind="swarm",hue='gender')

sns.catplot(x="status", y="degree_p", data=data,kind="swarm",hue='gender')

sns.catplot(x="status", y="mba_p", data=data,kind="swarm",hue='gender')

sns.catplot(x="status", y="etest_p", data=data,kind="swarm",hue='gender')
data1.head()
y = data1['status']

df = data1.drop(['salary','gender','degree_t','hsc_s','hsc_b','ssc_b','specialisation','workex','status'],axis=1)

x = df

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import accuracy_score as acc_score

from sklearn.ensemble import RandomForestClassifier as rfc



X_train,X_test,y_train,y_test = tts(x,y,test_size=0.33)

model = rfc()

model.fit(X_train,y_train)

Z = model.predict(X_test)

print (acc_score(y_test,Z))
import eli5

from eli5.sklearn import PermutationImportance

feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]

perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
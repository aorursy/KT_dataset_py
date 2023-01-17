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
df2 = pd.read_csv('../input/autism-screening-for-toddlers/Toddler Autism dataset July 2018.csv',na_values='?')
df2.columns
df2=df2.rename(columns={'Class/ASD Traits ': "label"})

df2.drop(["Case_No","Ethnicity",'Who completed the test'],axis=1,inplace=True)
df2.head()
list_of_sex = list(set(df2['Sex'].unique()))

sex_map = dict(zip(list_of_sex, np.arange(len(list_of_sex))))

list_of_Jaundice = list(set(df2['Jaundice'].unique()))

Jaundice_map = dict(zip(list_of_Jaundice, np.arange(len(list_of_Jaundice))))

list_of_fam = list(set(df2['Family_mem_with_ASD'].unique()))

fam_map = dict(zip(list_of_fam, np.arange(len(list_of_fam))))

list_of_class = list(set(df2['label'].unique()))

lab_map = dict(zip(list_of_class, np.arange(len(list_of_class))))
df2['Sex'] = df2['Sex'].map(sex_map)

df2['Jaundice'] = df2['Jaundice'].map(Jaundice_map)

df2['Family_mem_with_ASD'] = df2['Family_mem_with_ASD'].map(fam_map)

df2['label'] = df2['label'].map(lab_map)
df2.head()
data2= df2[df2['label']==1]

print("Toddlers:",len(data2)/len(df2) * 100)
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
df2.columns!="label"
Y=df2["label"]

X=df2.drop("label",axis=1)

seed = 7

test_size = 0.33

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,classification_report

rfc= RandomForestClassifier(n_estimators=500)

rfc.fit(X,Y)

pred_rfc= rfc.predict(X)

print(confusion_matrix(Y,pred_rfc))

print(classification_report(Y,pred_rfc))
import pickle

pickle.dump(rfc, open("mod.pkl", 'wb'))

X.ix[0].values
# model = pickle.load(open('model.pkl','rb'))

# print(model.predict([[1.8]]))
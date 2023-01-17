import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir('../input'))
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import Imputer
import datetime
train_df = pd.read_csv('../input/train.csv')
train_target = np.array(train_df['Survived'])
test_df  = pd.read_csv('../input/test.csv')
test_df['Survived'] = 3
train_df_concat = pd.concat([train_df,test_df])
print(train_df_concat['Embarked'].value_counts())
print(train_df_concat['Sex'].value_counts())
train_df_concat['Embarked_map'] = train_df_concat['Embarked'].map({'S':0,'C':1,'Q':2})
train_df_concat['Sex_map'] = train_df_concat['Sex'].map({'male':0,'female':1})
name_first_place = []
name_second_place = []
name_third_place = []
for i in np.array(train_df_concat['Name'].str.split(',')):
    name_first_place.append(i[0])
    second_place = i[1].split('. ')[0]
    third_place = i[1].split('. ')[1]
    name_second_place.append(second_place)
    name_third_place.append(third_place)

train_df_concat['name_first_place'] = np.array(name_first_place)
train_df_concat['name_second_place'] = np.array(name_second_place)
train_df_concat['name_third_place'] = np.array(name_third_place)
train_df_concat.drop(columns=['Name','Embarked','Sex'])
cabin_types = []
for i in train_df_concat['Cabin']:
    if i is not None:
        cabin_type = str(i)[0]
        cabin_types.append(cabin_type)
    else:
        cabin_types.append(None)
Ticket_Prefix = []
for i in train_df_concat['Ticket']:
    if i.isdecimal():
        Ticket_Prefix.append('')
    else:
        Ticket_Prefix.append(i.split(' ')[0])
train_df_concat['Ticket_Prefix'] = np.array(Ticket_Prefix)
train_df_concat = pd.get_dummies(train_df_concat)
train_df = train_df_concat.loc[(train_df_concat['Survived']<3)]
test_df = train_df_concat.loc[(train_df_concat['Survived']==3)]
train_df = train_df.drop(columns=['Survived'])
test_df = test_df.drop(columns=['Survived'])
train_df_array = np.array(train_df)
miss_input = Imputer(missing_values='NaN',strategy='mean',axis=0)
train_df_array = miss_input.fit_transform(train_df_array)

test_df_array = np.array(test_df)
test_df_array = miss_input.fit_transform(test_df_array)
columns = list(range(2,len(train_df_array)))
X_train,X_test,Y_train,Y_test = train_test_split(train_df_array,train_target,test_size=0.2,random_state=5)
clf = svm.SVC(kernel = 'linear', C=1).fit(X_train,Y_train)
test_target = clf.predict(test_df_array)
subm_df = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':test_target})
now=datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
subm_df.to_csv("subm"+ now +".csv",index=False)
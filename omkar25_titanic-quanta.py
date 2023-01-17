# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/train.csv')
import matplotlib.pyplot as plt

male_survived=data['Survived'][data['Sex']=='male'].sum()

female_survived=data['Survived'][data['Sex']=='female'].sum()

total_survived=data['Survived'].sum()

survived_stats=[male_survived, female_survived]

print("Total male survived:",male_survived,"\nTotal Female survived:",female_survived,"\nTotal survived:",total_survived)

#counts = data.groupby(['Survived','Sex']).count()

sex=[1,0]

sexlabel=['male','female']

plt.bar(sex,survived_stats,)

plt.title('Genderwise Survival Stats')

plt.xlabel('Gender')

plt.ylabel('Survived')

plt.xticks(sex,sexlabel)

plt.ylim((0,350))
#Now we will check the age wise survival stats and their gender distribution

data=data.fillna(0)

male_survived=[]

female_survived=[]

agedistr=[10,50,70,100]

for i in range(len(agedistr)):

    male_survived.append(data['Survived'][data['Age']<=agedistr[i]][data['Age']>[0 if i==0 else agedistr[i-1]]][data['Sex']=='male'].sum())

    female_survived.append(data['Survived'][data['Age']<=agedistr[i]][data['Age']>[0 if i==0 else agedistr[i-1]]][data['Sex']=='female'].sum())

for i in range(1,len(male_survived)):

    plt.bar(sex[0],male_survived[i],bottom=male_survived[i-1])

plt.hold(True)

for i in range(1,len(female_survived)):

    plt.bar(sex[1],female_survived[i],bottom=female_survived[i-1])

plt.xticks(sex,sexlabel)

plt.xlabel('Gender')

plt.title('Age wise Gender Survival Distribution')

plt.ylabel('Survived')
class_survival=[]

Passenger_class=['First_Class','Second_Class','General_Class']

a=[1,2,3]

for i in a:

    class_survival.append(data['Survived'][data['Pclass']==i].sum())

    plt.bar(i,class_survival[i-1])

plt.xlabel('Class')

plt.title('Class wise Survival')

plt.ylabel('Survived') 

plt.xticks(a,Passenger_class)
data=data.drop(['Name','PassengerId','Fare','Ticket','Cabin','Embarked'],axis=1)
data=data.replace(['female','male'],[0,1])
survived=data['Survived']

data=data.drop('Survived',axis=1)
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

numerical=['Age']

features_raw= pd.DataFrame(data = data)

features_raw[numerical] = scaler.fit_transform(features_raw[numerical])
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.svm import SVC

#RandomForestClassifier(max_depth=3,n_estimators=50)

clf=AdaBoostClassifier(RandomForestClassifier(max_depth=3,n_estimators=50), n_estimators=10, learning_rate=0.1,random_state=42)

best_clf=clf.fit(features_raw, survived)
test_data=pd.read_csv('../input/test.csv')

test_data=test_data.drop(['Name','PassengerId','Fare','Ticket','Cabin','Embarked'],axis=1)

test_data=test_data.replace(['female','male'],[0,1])

test_data=test_data.fillna(0)
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

numerical=['Age']

features_test_raw= pd.DataFrame(data = test_data)

features_test_raw[numerical] = scaler.fit_transform(features_test_raw[numerical])

#features_test_raw
survived_test=best_clf.predict(features_test_raw)
prediction_data=pd.read_csv('../input/gendermodel.csv')

passengerid=np.array(prediction_data['PassengerId'])

correct_value=prediction_data['Survived']

predictions_csv={'PassengerId':passengerid,'Survived':survived_test}

predictions_csv=pd.DataFrame(predictions_csv,

                      columns=['PassengerId','Survived'])

from sklearn.metrics import accuracy_score

accurate=accuracy_score(correct_value,survived_test)

print("Accuracy of the model is:",accurate)
predictions_csv.to_csv('gender_submission_quanta_1.csv',header=True,mode='a', index=False)
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

train_path = "../input/train.csv"

data_set = pd.read_csv(train_path)

data_set.head()



data_set.info()
data_set['Cabin'].value_counts()
data_set.describe()
import matplotlib.pyplot as plt

data_set.hist(bins=50,figsize=(20,10))
#corr_matrix["Survived"].sort_values(ascending=False)
data_set.head()



child_passengers = data_set[data_set['Age']<18]
child_passengers.count()

child_survied_percent  = child_passengers['PassengerId'].count() 



data_set['Age'] = data_set[['Age']].fillna(value=data_set['Age'].mean())



data_set.SibSp.unique()

sorted_values = data_set.sort_values('Ticket', ascending=True)

print(sorted_values[sorted_values['SibSp']>0].head(n=3))

                    

                    #or  (sorted_values['gender_class'] == 'AF') ].head(n=3))

def assign_gender_class(row):

    if row['Age'] > 60:

        return "O"

    elif (row['Age'] >= 18 and row['Age'] <= 60 and row['Sex'] == 'male'):

        return "AM"

    elif (row['Age'] >= 18 and row['Age'] <= 60 and row['Sex'] == 'female'):

        return "AF"

    else:

        return "C"

    

gender_class = data_set.apply(assign_gender_class,axis = 1)



data_set['gender_class'] = gender_class 


def assign_married_status(df):

    unique_ticket = df.Ticket.unique()

    for i in unique_ticket:

        same_ticket_user = df[df['Ticket']==i]

        print (same_ticket_user.head())

        #same_ticket_user.query("same_ticket_user['SibSp'] > 0 and  same_ticket_user.Cabin.nunique()==1")

        data_query = same_ticket_user[(same_ticket_user['SibSp'] > 0 )]

                                       #&  same_ticket_user.Cabin.nunique()==1)]



print (assign_married_status(data_set))
data_set['gender_class'].value_counts()



data_set.head()
features = ['gender_class','Pclass','SibSp','Parch']



from sklearn.model_selection import train_test_split



train_set , test_set = train_test_split(data_set, test_size = 0.2,random_state =42)





corr_matrix = train_set.corr()



corr_matrix["Survived"].sort_values(ascending=False)
from sklearn.ensemble import RandomForestClassifier



y = train_set["Survived"]



X = pd.get_dummies(train_set[features])

X_test = pd.get_dummies(test_set[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_set.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
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
csv_train = pd.read_csv('/kaggle/input/train.csv', na_values= '?')

csv_test = pd.read_csv('/kaggle/input/test.csv', na_values= '?')
def process_data(csv, train_flag):

    csv_droped_ID = csv.drop("PassengerId", 1)

    #sv_droped_ID.head()

    

    csv_sex = csv_droped_ID["Sex"]

    csv_sex = (csv_sex == "female")

    csv_sex = csv_sex.replace(True, 1)

    csv_droped_ID.Sex = csv_sex

    #csv_droped_ID.head()



    csv_droped_ID_Cabin = csv_droped_ID.drop("Cabin", 1)

    csv_droped_ID_Cabin_Ticket = csv_droped_ID_Cabin.drop("Ticket", 1)

    csv_droped_ID_Cabin_Ticket_Name = csv_droped_ID_Cabin_Ticket.drop("Name", 1)

    #csv_droped_ID_Cabin_Ticket_Name.head()

    csv_embarked = csv_droped_ID_Cabin_Ticket_Name['Embarked']



    csv_embarked = csv_embarked.replace('S', 0)

    csv_embarked = csv_embarked.replace('Q', 1)

    csv_embarked = csv_embarked.replace('C', 2)

    csv_droped_ID_Cabin_Ticket_Name.Embarked = csv_embarked

    for i in csv_droped_ID_Cabin_Ticket_Name.columns:

        temp = csv_droped_ID_Cabin_Ticket_Name[i]

        csv_droped_ID_Cabin_Ticket_Name[i] = temp.fillna(int(temp.mean()))

    #csv_embarked.head()

    if train_flag == 1:

        return csv_droped_ID_Cabin_Ticket_Name.loc[:, "Pclass" : "Embarked"],csv_droped_ID_Cabin_Ticket_Name["Survived"]

    else:

        return csv_droped_ID_Cabin_Ticket_Name
x_train_all, y_train_all = process_data(csv_train, train_flag= 1)

x_for_output = process_data(csv_test, train_flag= 0)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier



rnd_clf = RandomForestClassifier(n_estimators= 1000, max_leaf_nodes= 24)

rnd_clf.fit(x_train, y_train)
rnd_clf.feature_importances_
rnd_clf_pred = rnd_clf.predict(x_test)

from sklearn.metrics import accuracy_score



accuracy_score(rnd_clf_pred, y_test)
from sklearn.ensemble import AdaBoostClassifier



ada_clf = AdaBoostClassifier(

    RandomForestClassifier(), algorithm="SAMME.R", learning_rate= 0.01

    )
ada_clf.fit(x_train, y_train)
ada_clf_pred = ada_clf.predict(x_test)

accuracy_score(ada_clf_pred, y_test)
from sklearn.tree import DecisionTreeClassifier



ada_clf_dT = AdaBoostClassifier(

    DecisionTreeClassifier(max_depth = 1), n_estimators= 200, algorithm="SAMME.R", learning_rate= 0.5

    )

ada_clf_dT.fit(x_train, y_train)
ada_clf_dT_pred = ada_clf_dT.predict(x_test)

accuracy_score(ada_clf_dT_pred, y_test)
from sklearn.ensemble import GradientBoostingClassifier



gbrt = GradientBoostingClassifier(max_depth=2, n_estimators= 3, learning_rate=1)

gbrt.fit(x_train, y_train)
gbrt_pred = gbrt.predict(x_test)



accuracy_score(gbrt_pred, y_test)
rnd_clf_all = RandomForestClassifier(n_estimators= 1000, max_leaf_nodes= 24)

rnd_clf_all.fit(x_train_all, y_train_all)
output = rnd_clf_all.predict(x_for_output)
output = rnd_clf.predict(x_for_output)

out = pd.DataFrame()

out["Survived"] = output

out.to_csv('output.csv', sep= '\t', index= False)
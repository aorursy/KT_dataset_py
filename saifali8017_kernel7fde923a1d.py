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

import numpy as np

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

#importing the datasets



df = pd.read_csv("/kaggle/input/titanic/train.csv")



x = df.iloc[:,[2,4,5]]

y = df.iloc[:,1].values



labelencoder = LabelEncoder()

x["Sex"] = labelencoder.fit_transform(x["Sex"])

impute = Imputer(missing_values=np.nan,strategy = "mean")

x=impute.fit_transform(x)

onehotencoder = OneHotEncoder(categorical_features=[1])

x  = onehotencoder.fit_transform(x).toarray()

x = x[:,1:]



from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)

#using SVM Classifier

from sklearn.svm import SVC

clf = SVC(kernel="rbf",gamma=5,C=5)

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

a = classification_report(y_test,y_pred)

confus = confusion_matrix(y_test,y_pred)
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

passengerId = test_df['PassengerId']


test_df = test_df.iloc[:,[1,3,4]]

test_df["Sex"] = labelencoder.transform(test_df["Sex"])

test_df = impute.fit_transform(test_df)

test_df = onehotencoder.transform(test_df).toarray()

test_df = test_df[:,1:]

test_df = sc.transform(test_df)

prediction = clf.predict(test_df)

output = pd.DataFrame({"PassengerId": passengerId,"Survived" : prediction})
print(output.to_string())
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
data=pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
data.head()
data.describe().T
data.dtypes.sample(10)
y=data["gender"]
y.head()
y.tail()
features=["hsc_p","ssc_p","salary"]

x=data[features]
x.head()
from sklearn.impute import SimpleImputer



# Imputation

my_imputer = SimpleImputer()

X = pd.DataFrame(my_imputer.fit_transform(x))

X.columns= x.columns
X.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski',weights='distance')

knn.fit(X_train,y_train)



y_pred = knn.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted':y_pred})

df
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

print(cm)
from sklearn.model_selection import cross_val_score

success = cross_val_score(estimator = knn, X=X_train, y=y_train , cv = 4)

print(success.mean())

print(success.std())
output = pd.DataFrame({'sscp':X_test.ssc_p,'hscp':X_test.hsc_p,'Salary': X_test.salary,'Gender':y_test,'PredictGender':y_pred})

output.to_csv('KnnSubmission.csv', index=False)

print("Your submission was successfully saved!")
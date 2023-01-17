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
import pandas as pd

pd.set_option('display.max_columns', None)



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report
train= pd.read_csv('../input/forest-cover-type-prediction/train.csv')
train.head()
train.info()
train['Cover_Type'].value_counts(normalize=True)
X= train.drop('Cover_Type', axis=1)

y= train['Cover_Type']



X_train, X_test, y_train, y_test= train_test_split(X,y, test_size= 0.25)
tree= DecisionTreeClassifier(random_state=24)

forest= RandomForestClassifier(random_state=24)

knn= KNeighborsClassifier()

svm= SVC(random_state=24)

xboost= XGBClassifier(random_state=24)





models= [tree, forest, knn, svm, xboost]



for model in models:

    model.fit(X_train, y_train)

    y_pred= model.predict(X_test)

    accuracy= accuracy_score(y_test, y_pred)

    clf_report= classification_report(y_test, y_pred) 

    print(f"The accuracy of model {type(model).__name__} is {accuracy:.2f}")

    print(clf_report)

    print("\n")
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
data = pd.read_excel("/kaggle/input/phishing-websites-detection/Detection.xlsx")

data.head(5)
from sklearn.model_selection import train_test_split

training_set, test_set = train_test_split(data, test_size = 0.2, random_state = 1)

X=data.iloc[:,1:11].values

Y=data.iloc[:,11].values
X_train = training_set.iloc[:,1:11].values

Y_train = training_set.iloc[:,11].values

X_test = test_set.iloc[:,1:11].values

Y_test = test_set.iloc[:,11].values
from sklearn.svm import SVC

classifier = SVC(kernel='rbf', random_state = 1,C=10)

classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)

test_set["Predictions"] = Y_pred

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



print(confusion_matrix(Y_test,Y_pred.round()))

print(classification_report(Y_test,Y_pred.round()))

print(accuracy_score(Y_test, Y_pred.round()))
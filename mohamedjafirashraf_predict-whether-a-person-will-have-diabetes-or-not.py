# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/diabetes/diabetes.csv')

data.head()
not_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']



for column in not_zero:

    data[column] = data[column].replace(0,np.NaN)

    mean = int(data[column].mean(skipna=True))

    data[column] = data[column].replace(np.NaN,mean)

    
X = data.iloc[:, 0:8]

y = data['Outcome']



X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2)
import math

math.sqrt(len(y_test))
knn = KNeighborsClassifier(n_neighbors=19, p=2, metric='euclidean')

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

y_pred
accuracy_score(y_pred,y_test)
prediction=knn.predict([[6,148.0,62.0,35.0,455.0,33.6,0.627,30]])

if prediction ==1:

    print("The person have Diabetes")

else:

    print("The person is not have Diabetes")

prediction
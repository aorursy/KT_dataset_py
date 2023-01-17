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
csv_file_name = "/kaggle/input/titanic/train.csv"

data = pd.read_csv(csv_file_name)

data.info()
print(data)
print(data.values[0])
data = data.dropna()



target = data['Survived']

data = data[['Age', 'Fare']]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0)





from sklearn.neighbors import KNeighborsClassifier

score_from_training_data = []

score_from_test_data = []



for i in np.arange(1,20,1):

    clf = KNeighborsClassifier(n_neighbors=i)

    clf.fit(X_train, y_train)

    

    score_from_training_data.append(clf.score(X_train, y_train))

    score_from_test_data.append(clf.score(X_test, y_test))
import matplotlib.pyplot as plt

plt.plot(np.arange(1,20,1), score_from_training_data, label="score from training data")

plt.plot(np.arange(1,20,1), score_from_test_data, label="score from test data")

plt.legend()

plt.show()
plt.scatter(X_train['Age'], X_train['Fare'], c=y_train, s=30)
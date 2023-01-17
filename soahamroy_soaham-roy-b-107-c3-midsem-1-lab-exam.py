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
import numpy as np

u=np.array([8,4,6,7])

v=np.array([2,5,3,9])

id(u),id(v)



d1=u%3==0

d2=v%3==0

print(d1)

print(d2)



print(u[d1])

print(v[d2])



print(np.sort(v))

print(np.sum(u))

import pandas as pd

d= pd.read_csv("../input/titanic/train_and_test2.csv")

missing_values=data.dropna(axis=1,how='all')

print(missing_values)

d1=d.head(50)

print(d.mean())

male=d['Sex']==1

print(male.mean())

print(d['Fare'].max())
import matplotlib.pyplot as plt

slices=[ 86,83,86,90,88 ]

subj=['English','Maths', 'Science','History','Geography']

cols=['r','b','g','m','c']

plt.pie(slices,labels=subj,colors=cols,startangle=90,shadow= True,explode=(0,0.2,0,0,0),autopct='%1.1f%%')

plt.title('MARKS IN SUBJECT')

plt.show()
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB



a = pd.read_csv('../input/iris/Iris.csv', error_bad_lines=False)

a = a.drop(['Id'], axis=1)

a['Species'] = pd.factorize(a["Species"])[0] 

Target = 'Species'

Features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']



model = LogisticRegression(solver='lbfgs', multi_class='auto')

Features = ['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']



b, c = train_test_split(a, 

                        test_size = 0.2, 

                        train_size = 0.8, 

                        random_state= 3)



b1 = b[Features]

b2 = b[Target]

c1 = c[Features]

c2 = c[Target]



nb_model = GaussianNB() 

nb_model.fit(X=b1, y=b2)

result= nb_model.predict(y[Features]) 



f1_sc = f1_score(c2, result, average='micro')

confusion_m = confusion_matrix(c2, result)



print("F1 Score    : ", f1_sc)

print("Confusion Matrix: ")

print(confusion_m)
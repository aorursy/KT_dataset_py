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
#QUESTION 1



import numpy as np

x = np.array([2,5,3,0])

y = np.array([1,9,4,2])

id(x),id(y)





div = x%3==0

divv = y%3 == 0

print(div)

print(divv)



print(x[div])

print(y[divv])



print(np.sort(y))

print(np.sum(x))

#QUESTION 2







# importing pandas module 

import pandas as pd 



# making data frame from csv file 

data = pd.read_csv("../input/titanic/train_and_test2.csv") 



# making new data frame with dropped NA values 

new_dataset = data.dropna(axis = 0, how ='any') 

new_dataset.describe()



x = new_dataset.head(50)

print("mean is :", x.mean())

male=data['Sex']==1

print("mean of male passengers :", male.mean())

print(data['Fare'].max())

#QUESTION 3

import matplotlib.pyplot as plt

slices=[ 86,83,86,90,88 ]

subj=['English','Maths', 'Science','History','Geography']

cols=['b','g','r','m','c']

plt.pie(slices,labels=subj,colors=cols,startangle=90,shadow= True,explode=(0,0.2,0,0,0),autopct='%1.1f%%')

plt.title('Marks in Each subject')

plt.show()
#QUESTION 4





import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB



df = pd.read_csv('../input/iris/Iris.csv', error_bad_lines=False)

df = df.drop(['Id'], axis=1)

df['Species'] = pd.factorize(df["Species"])[0] 

Target = 'Species'

Features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']



model = LogisticRegression(solver='lbfgs', multi_class='auto')

Features = ['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']



x, y = train_test_split(df, 

                        test_size = 0.2, 

                        train_size = 0.8, 

                        random_state= 3)



x1 = x[Features]

x2 = x[Target]

y1 = y[Features]

y2 = y[Target]



nb_model = GaussianNB() 

nb_model.fit(X=x1, y=x2)

result= nb_model.predict(y[Features]) 



f1_sc = f1_score(y2, result, average='micro')

confusion_m = confusion_matrix(y2, result)



print("F1 Score    : ", f1_sc)

print("Confusion Matrix: ")

print(confusion_m)
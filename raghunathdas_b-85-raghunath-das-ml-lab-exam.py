import numpy as np

a=np.array([9,5,4,3,2,6])

b=np.array([5,8,6,9,2,1])

print("CHECK IF B HAS SAME VIEWS TO MEMORY IN A")

print(b.base is a)

print("CHECK IF A HAS SAME VIEWS TO MEMORY IN B")

print(a.base is b)

div_by_3=a%3==0

div1_by_3=b%3==0

print("Divisible By 3")

print(a[div_by_3])

print(b[div1_by_3])

b[::-1].sort()

print("SECOND ARRAY SORTED")

print(b)

print("SUM OF ELEMENTS OF FIRST ARRAY")

print(np.sum(a))
import pandas as pd

data= pd.read_csv("../input/titanic/train_and_test2.csv")

miss=data.dropna(axis=1,how='all')

print(miss)

data1=data.head(50)

print(data.mean())

male=data['Sex']==1

print(male.mean())

print(data['Fare'].max())
import matplotlib.pyplot as plt

slices=[86,83,86,90,88]

subj=['English','Math','Science','History','Geography']

cols=['c','m','r','b','y']

plt.pie(slices,labels=subj,colors=cols,startangle=90,shadow=True,explode=(0,0.3,0,0,0),autopct='%1.1f%%')

plt.title('SUBJECT RESULTS')

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



x, y = train_test_split(a, 

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
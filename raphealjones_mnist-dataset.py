import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np
df=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df.head()
y=df[['label']]

feature=[]

for i in df.columns:

    if i=='label':

        continue

    else:

        feature.append(i)

x=df[feature]
# from sklearn.linear_model import LogisticRegression
# model=LogisticRegression()

# model.fit(x,y)

# model.score(x,y)
# from sklearn.tree import DecisionTreeClassifier
# model=DecisionTreeClassifier()

# model.fit(x,y)

# model.score(x,y)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=32)
# model=DecisionTreeClassifier()

# model.fit(x_train,y_train)

# model.score(x_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

def knn(k):

    model=KNeighborsClassifier(k)

    model.fit(x_train,y_train)

    return model.score(x_test,y_test)
# from sklearn.naive_bayes import GaussianNB

# model=GaussianNB()

# model.fit(x_train,y_train)

# model.score(x_test,y_test)
# from sklearn.ensemble import AdaBoostClassifier

# model=AdaBoostClassifier()

# model.fit(x_train,y_train)

# model.score(x_test,y_test)
# predictions=[]

# for i in range(1,11):

#     predictions.append(knn(i))
# j=0

# for i in predictions:

#     j+=1

#     print(j,' : ',i)



# maxi=0

# for i in predictions:

#     maxi=i if maxi<i else maxi

# predictions.index(maxi)+1
validate=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

validate
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test.head()
model=KNeighborsClassifier(6)

model.fit(x,y)
final=model.predict(test)

final
new_final=pd.DataFrame(final)

new_final
id=[i for i in range(1,test.shape[0]+1)]

Id=pd.DataFrame(id)

    
submission=pd.concat([Id,new_final],axis=1)

submission
submission.columns=['ImageId','Label']
submission
submission.to_csv('submission.csv')
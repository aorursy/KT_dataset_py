import pandas as pd

import  numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error,confusion_matrix,accuracy_score

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv('../input/mushroom-classification/mushrooms.csv')
df.head()
df.columns
df.info()
df.describe()
df.isnull().sum()
df['class'].value_counts()
df[df['class']=='e'].describe().T
df[df['class']=='p'].describe().T
X=df.drop('class',axis=1)

y=df['class']
X.head()
y.head()
Encoder_x=LabelEncoder()

for col in X.columns:

    X[col]=Encoder_x.fit_transform(X[col])

Encoder_y=LabelEncoder()

y=Encoder_y.fit_transform(y)
X.head()
y
train_x,test_x,train_y,test_y=train_test_split(X,y)
acc = []

for neighbors in range(3,10,1):

    classifier = KNeighborsClassifier(n_neighbors=neighbors, metric='minkowski')

    classifier.fit(train_x,train_y)

    y_pred = classifier.predict(test_x)

    acc.append(accuracy_score(test_y,y_pred))

    

plt.figure(figsize=(15,7))

plt.plot(list(range(3,10,1)), acc)

plt.show()
print(f"Best accuracy is {np.max(acc)} and the k value is {1+acc.index(np.max(acc))}")
k=1+acc.index(np.max(acc))

knn=KNeighborsClassifier(n_neighbors=k)

knn.fit(train_x,train_y)

pred=knn.predict(test_x)
print("Mean absolute error:",mean_absolute_error(pred,test_y))
pred2=[]

for i in pred:

    if i==1:

        pred2.append('p')

    else:

        pred2.append('e')

        

# pred=map(lambda x:'p' if x==1 else 'e',pred)
pred2=pd.DataFrame({'index':test_x.index,'class':pred2})

pred2.head()
model=RandomForestClassifier(random_state=1)

model.fit(train_x,train_y)

pred=model.predict(test_x)
print("Model score:",model.score(test_x,test_y))

print("Mean absolute error:",mean_absolute_error(pred,test_y))

print("Accuracy score:",accuracy_score(pred,test_y))
pred=map(lambda x:'p' if x==1 else 'e',pred)
pred2=pd.DataFrame({'index':test_x.index,'class':pred})

pred2.head()
pred2.to_csv('submission.csv',index=False)
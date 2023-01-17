import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
train=pd.read_csv('../input/iris/Iris.csv')
train.head()
train.info()
train.shape
train.isnull().sum()
train.Species.unique()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
train.Species=label_encoder.fit_transform(train.Species)
train.Species.unique()
train.head()
from sklearn.model_selection import train_test_split
y=train['Species']

y.unique()
train=train.drop(['Id'],axis=1)
X=train.drop(['Species'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.linear_model import LogisticRegression

model=LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=1000)
model.fit(X_train,y_train)
model.score(X_test,y_test)
print('as we can see above the score of our model on the test data is ' + str(model.score(X_test,y_test)) + ' ie ' + str(model.score(X_test,y_test)*100) +'%'+ " (which will change if we run the train test split again as the splitting of data is random)")





print("Anyway it's good!!")
category=['iris-setosa','iris-versicolor','iris-virginica']
pred=model.predict([[5.2,3.2,1.4,0.3]])

pred
category[int(pred)]
from sklearn.utils import shuffle
#train=shuffle(train)
import seaborn as sns

import matplotlib.pyplot as plt
sns.countplot(train['Species'])
plt.scatter(train['SepalLengthCm'],train['Species'])
train.loc[(train.SepalLengthCm<5) & (train.Species==2)]
train.drop(106,inplace=True)
plt.scatter(train['SepalWidthCm'],train['Species'])
train.loc[(train.SepalWidthCm<2.3) & (train.Species==2)]
train.loc[(train.SepalWidthCm<2.5) & (train.Species==0)]
train.drop(119,inplace=True)

train.drop(41,inplace=True)
plt.scatter(train['PetalLengthCm'],train['Species'])
train.loc[(train.PetalLengthCm<3.2) & (train.Species==1)]
train.drop(98,inplace=True)
plt.scatter(train['PetalWidthCm'],train['Species'])
plt.subplot(2, 2, 1)

plt.scatter(train['SepalLengthCm'],train['Species'])

plt.subplot(2, 2, 2)

plt.scatter(train['SepalWidthCm'],train['Species'])

plt.subplot(2, 2, 3)

plt.scatter(train['PetalLengthCm'],train['Species'])

plt.subplot(2, 2, 4)

plt.scatter(train['PetalWidthCm'],train['Species'])
train.info()
train.head()
y=train['Species']

y.unique()
X=train.drop(['Species'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
model.fit(X_train,y_train)
model.score(X_test,y_test)
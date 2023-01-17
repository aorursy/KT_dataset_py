#Data Analysis Liabrarise

import numpy as np 

import pandas as pd



#Data visualization liabraries

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import preprocessing 





import pandas as pd

test = pd.read_csv("../input/titanic-machine-learning-from-disaster/test.csv")

train = pd.read_csv("../input/titanic-machine-learning-from-disaster/train.csv")
#Take backup of data set for some other use if needed.

train_data = train.copy()

test_data  = test.copy()
train.head(5)
train.dtypes
train.isna().sum()
train = train.fillna(method='ffill')
train.isna().sum()

#Dropping columns not required in data set

train = train.drop(['Cabin','Ticket','Parch'], axis = 1)
train.isna().sum()
train.describe().T
sns.boxplot(x = train['Fare'])
def outliers_transform(base_dataset):

    for i in base_dataset.var().sort_values(ascending=False).index[0:10]:

        x=np.array(base_dataset[i])

        qr1=np.quantile(x,0.25)

        qr3=np.quantile(x,0.75)

        iqr=qr3-qr1

        utv=qr3+(1.5*(iqr))

        ltv=qr1-(1.5*(iqr))

        y=[]

        for p in x:

            if p <ltv or p>utv:

                y.append(np.median(x))

            else:

                y.append(p)

        base_dataset[i]=y
outliers_transform(train)
sns.boxplot(x = train['Fare'])
train.describe().T
train.drop('PassengerId',axis=1,inplace=True)

train.drop('Name',axis=1,inplace=True)
label_encoder = preprocessing.LabelEncoder() 

train['Embarked']= label_encoder.fit_transform(train['Embarked'])

train['Sex']= label_encoder.fit_transform(train['Sex'])
train
for i in train.var().index:

    sns.distplot(train[i],kde=False)

    plt.show()
plt.figure(figsize=(20,10))

sns.heatmap(train.corr())
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score



target = train['Survived']

predictors = train.drop(['Survived'], axis = 1) 



x_train,x_test,y_train,y_test=train_test_split(predictors, target, test_size=0.20, random_state=43)
x_train.shape
x_test.shape
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier





classifier = DecisionTreeClassifier()

# Train Decision Tree Classifer

classifier = classifier.fit(x_train,y_train)

#Predict the response for test dataset

y_pred = classifier.predict(x_test)

print("Accuracy:",accuracy_score(y_test, y_pred))
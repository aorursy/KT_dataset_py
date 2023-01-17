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
import matplotlib.pyplot as plt
dt = pd.read_csv('/kaggle/input/titanic/train.csv')

dt_test = pd.read_csv('/kaggle/input/titanic/test.csv')
dt.head()
dt.info()
cols_to_drop = ['PassengerId','Name','Ticket','Cabin','Embarked']
data_clean = dt.drop(columns=cols_to_drop,axis=1)

data_clean_test = dt_test.drop(columns=cols_to_drop,axis=1)

data_clean.info()
data_clean.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data_clean['Sex'] = le.fit_transform(data_clean['Sex'])

data_clean_test['Sex'] = le.fit_transform(data_clean_test['Sex'])

data_clean.head()
import seaborn as sns
def bar_chart(feature):

    survived = dt[dt['Survived']==1][feature].value_counts()

    dead = dt[dt['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True)

    

bar_chart('Sex')
bar_chart('SibSp')
data_clean.corr()

plt.figure(figsize=(11,11))

sns.heatmap(data_clean.corr(),cmap='rainbow',annot=True)

plt.show()
data_clean.info()
data_clean['Age'] = data_clean.fillna(data_clean['Age'].mean())['Age']

data_clean_test['Age'] = data_clean_test.fillna(data_clean_test['Age'].mean())['Age']

data_clean.info()
data_clean_test.info()
data_clean_test['Fare'] = data_clean_test.fillna(data_clean_test['Fare'].mean())['Fare']

data_clean_test.info()
input_cols = ['Pclass',"Sex","Age","SibSp","Parch","Fare"]

output_cols = ["Survived"]
x_train = data_clean[input_cols]

y_train = data_clean[output_cols]

x_test = data_clean_test[input_cols]
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



dtc = DecisionTreeClassifier(max_depth=4,criterion='entropy')

rfc = RandomForestClassifier(max_depth=5,criterion='entropy',n_estimators=16)
dtc.fit(x_train,y_train)
dtc.score(x_train,y_train)
rfc.fit(x_train,y_train)
rfc.score(x_train,y_train)
y_pred = dtc.predict(x_test)

y_pred1 = rfc.predict(x_test)
print(y_pred)

print(y_pred1)
y_test = []

y_test1 = []

for i in range(x_test.shape[0]):

    y_test.append([dt_test['PassengerId'][i],y_pred[i]])

    y_test1.append([dt_test['PassengerId'][i],y_pred1[i]])
y_test
dt = pd.DataFrame(y_test,columns=['PassengerId','Survived'])

dt1 = pd.DataFrame(y_test1,columns=['PassengerId','Survived'])
dt.head()
dt1.head()
dt.to_csv("Submit1.csv",index=False)

dt1.to_csv("Submit2.csv",index=False)
def entropy(col):



    counts = np.unique(col, return_counts=True)

    n = float(col.shape[0])



    ent = 0.0

    for ix in counts[1]:

        p = ix/n

        ent += (-1.0*p*np.log2(p))



    return ent
def divide_data(x_data,fkey,fval):

    

    x_right = pd.DataFrame([],columns=x_data.columns)

    x_left = pd.DataFrame([],columns=x_data.columns)

    

    for ix in range(x_data.shape[0]):

        val = x_data[fkey].loc[ix]

        

        if val>fval:

            x_right = x_right.append(x_data.loc[ix])

        else:

            x_left = x_left.append(x_data.loc[ix])

    

    return x_left,x_right
x_left,x_right = divide_data(data_clean[:10],'Sex',0.5)

print(x_left)

print(x_right)
def information_gain(x_data,fkey,fval):

    

    left,right = divide_data(x_data,fkey,fval)

    

    l = float(left.shape[0])/x_data.shape[0]

    r = float(right.shape[0])/x_data.shape[0]    

    

    # If all examples come on one side

    if left.shape[0] == 0 or right.shape[0] == 0:

        return -1000000000

    

    info_gain = entropy(x_data.Survived) - (l*entropy(left.Survived)+r*entropy(right.Survived))

    

    return info_gain
for fx in X.columns:

    print(fx)

    print(information_gain(data_clean,fx,data_clean[fx].mean()))
class DecisionTree:

    

    def __init__(self,depth=0,max_depth=5):

        self.left = None

        self.right = None

        self.fkey = None

        self.fval = None

        self.depth = depth

        self.max_depth = max_depth

        self.target = None

        

    def train(self,X_train):

        

        features = ['Pclass','Sex','Age','SibSp','Parch','Fare']

        info_gain=[]

        

        for ix in features:

            i_gain = information_gain(X_train,ix,X_train[ix].mean())

            info_gain.append(i_gain)

            

        self.fkey = features[np.argmax(info_gain)]

        self.fval = X_train[self.fkey].mean()

        print("Making node, feature is ",self.fkey)

        

        #Split data

        data_left,data_right = divide_data(X_train,self.fkey,self.fval)

        data_left = data_left.reset_index(drop=True)

        data_right = data_right.reset_index(drop=True)

        

        if data_left.shape[0]==0 or data_right.shape[0]==0:

            if X_train.Survived.mean() >= 0.5:

                self.target = 'Survived'

            else:

                self.target = 'Dead'

            return

        

        if (self.depth >= self.max_depth):

            if X_train.Survived.mean() >= 0.5:

                self.target = 'Survived'

            else:

                self.target = 'Dead'

            return

        

        self.left = DecisionTree(depth=self.depth+1,max_depth=self.max_depth)

        self.left.train(data_left)

        

        self.right = DecisionTree(depth=self.depth+1,max_depth=self.max_depth)

        self.right.train(data_right)

        

        if X_train.Survived.mean() >= 0.5:

                self.target = 'Survived'

        else:

                self.target = 'Dead'

        return

    

    def predict(self,test):

        if test[self.fkey] >= str(self.fval):

            if self.right is None:

                return self.target

            return self.right.predict(test)

        else:

            if self.left is None:

                return self.target

            return self.left.predict(test)
d = DecisionTree()
d.train(data_clean)
print(d.fkey,d.fval)

print(d.left.fkey,d.right.fkey)
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

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier,RandomForestClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.model_selection import GridSearchCV
train_data = pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/train.csv.zip')

test_data = pd.read_csv('../input/ghouls-goblins-and-ghosts-boo/test.csv.zip')
train_data.head()
train_data.info()
train_data.describe()
train_data.isnull().sum()
X = train_data.drop(['id','type'],axis=1)

y = train_data['type']
X = pd.get_dummies(X,columns=['color'])

X
# Ghoul  0

# Goblin 1

# Ghost  2

y.unique()


y[y=='Ghoul'] = 0

y[y=='Goblin'] = 1

y[y=='Ghost'] = 2
y.unique()
y = y.astype('int')
cret = pd.concat([X,y],axis=1)

cret
from sklearn import preprocessing

color_ = preprocessing.LabelEncoder()

color_.fit(train_data['color'])

train_data['color_int'] = color_.transform(train_data['color'])



sns.pairplot(train_data.drop('color', axis = 1), hue = 'type')



train_data.drop('color_int', axis = 1, inplace = True)



# Ghoul  0

# Goblin 1

# Ghost  2
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=77,test_size=0.3)
scaler = MinMaxScaler()

X_test = scaler.fit_transform(X_test)

X_train = scaler.fit_transform(X_train)
clflog = LogisticRegression(random_state=1)

clfdt = DecisionTreeClassifier(random_state=1)

clfrf = RandomForestClassifier()

clfgnb = GaussianNB()

eclf_h =VotingClassifier(estimators = [('log',clflog),('dt',clfdt),('rf',clfrf),('gnb',clfgnb)],voting='hard')

eclf_s =VotingClassifier(estimators = [('log',clflog),('dt',clfdt),('rf',clfrf),('gnb',clfgnb)],voting='soft')



models = [clflog,clfdt,clfrf,clfgnb,eclf_h,eclf_s]

for model in models:

  model.fit(X_train,y_train)  

  predictions = model.predict(X_test)  

  print(classification_report(y_test,predictions))

  print("%s"%model)
models_ = [clflog,clfdt,clfrf,clfgnb]

for model in models_:

  pred = model.predict(X_test)

  print("%s"%model)

  print(confusion_matrix(y_test,pred))  
#LogisticRegression GridSearchCV

c_params = [0.001,0.01,0.1, 1.0, 10.0, 100.0]

params = {

  'solver':['newton-cg','lbfgs','liblinear','saga','sag'],

  'penalty':['l1','l2','elasticnet'],

  'C':c_params,

  'max_iter':[1,10,100,1000],

  'multi_class':['auto','ovr','multinomial']

 }

grid = GridSearchCV(LogisticRegression(),param_grid=params,refit=True,verbose=True,cv=5);

grid = grid.fit(X_train,y_train);
grid.best_params_
lr = LogisticRegression(C=1.0,max_iter=10,multi_class='multinomial',penalty='l2',solver='sag')

lr.fit(X_train,y_train)

predictions = lr.predict(X_test)

print(classification_report(y_test,predictions))
test_data
# id는 predict에 사용되지 않으므로 따로 저장해둔다

id_ = test_data['id'] 

test_data = test_data.drop(['id'],axis=1)
test_data = pd.get_dummies(test_data, columns = ['color'])
test_data
result = pd.DataFrame(lr.predict(test_data))

result.columns=['type']
result[result['type']==0]='Ghoul'

result[result['type']==1]='Goblin'

result[result['type']==2]='Ghost'



result
result = pd.concat([id_,result],axis=1)
result.info()
result.to_csv('result.csv',index=False)
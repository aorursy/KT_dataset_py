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
%matplotlib inline

import os

import warnings

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as image

import pandas as pd

plt.style.use("ggplot")

warnings.simplefilter("ignore")
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_train.head()
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_test.head()
df_train.info()
pd.crosstab(df_train.Sex ,df_train.Survived).plot(kind='bar')

plt.title('Survival rate as per Sex')

plt.xlabel('Sex')

plt.ylabel('Survived')

plt.show()
pd.crosstab(df_train.Pclass ,df_train.Survived).plot(kind='bar')

plt.title('Survival rate as per Pclass')

plt.xlabel('Pclass')

plt.ylabel('Survived')

plt.show()
df_train['Age'].isna().sum()
df_train['Age'] = df_train['Age'].fillna(0)

df_train['Age'].isna().sum()
pd.crosstab(df_train.Embarked ,df_train.Survived).plot(kind='bar')

plt.title('Survival rate as per Embarked')

plt.xlabel('Embarked')

plt.ylabel('Survived')

plt.show()
pd.crosstab(df_train.Parch ,df_train.Survived).plot(kind='bar')

plt.title('Survival rate as per Parch')

plt.xlabel('Parch')

plt.ylabel('Survived')

plt.show()
df_train.drop(columns = ['Name','Ticket' ,'Fare' , 'Cabin']  , inplace= True)
df_train.head(5)
df_train.info()
cat_vars=['Sex' , 'Embarked']

for var in cat_vars:

    cat_list='var'+'_'+var

    cat_list = pd.get_dummies(df_train[var], prefix=var)

    df_train1=df_train.join(cat_list)

    df_train=df_train1
df_train.columns
df_train.drop(columns = ['Sex','Embarked']  , inplace= True)

df_train.head(5)
df_train.info()
s=0

d=0

for i in df_train['Survived']:

  if i==0:

    d+=1

  else:

    s+=1



print('Dead - ',d,' Survived -',s)
X = df_train.loc[:, df_train.columns != 'Survived']

Y = df_train.Survived
df_test['Age'] = df_test['Age'].fillna(0)

df_test.info()
df_test.drop(columns = ['Name','Ticket' ,'Fare','Cabin']  , inplace= True)

df_test.info()
cat_vars=['Sex' , 'Embarked']

for var in cat_vars:

    cat_list='var'+'_'+var

    cat_list = pd.get_dummies(df_test[var], prefix=var)

    df_test1=df_test.join(cat_list)

    df_test=df_test1
df_test.head()
df_test.drop(columns=['Sex' , 'Embarked'] , inplace=True)

df_test.info()
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.tree import export_graphviz # display the tree within a Jupyter notebook

from IPython.display import SVG

from graphviz import Source

from IPython.display import display

from ipywidgets import interactive, IntSlider, FloatSlider, interact

import ipywidgets

from IPython.display import Image

from subprocess import call

import matplotlib.image as mpimg
@interact

def plot_tree_rf(crit=["gini", "entropy"],

                 bootstrap=["True", "False"],

                 depth=IntSlider(min=1,max=30,value=3, continuous_update=False),

                 forests=IntSlider(min=1,max=200,value=100,continuous_update=False),

                 min_split=IntSlider(min=2,max=5,value=2, continuous_update=False),

                 min_leaf=IntSlider(min=1,max=5,value=1, continuous_update=False)):

    

    estimator = RandomForestClassifier(random_state=1,

                                       criterion=crit,

                                       bootstrap=bootstrap,

                                       n_estimators=forests,

                                       max_depth=depth,

                                       min_samples_split=min_split,

                                       min_samples_leaf=min_leaf,

                                       n_jobs=-1,

                                      verbose=False).fit(X, Y)



    print('Random Forest Training Accuracy: {:.3f}'.format(accuracy_score(Y, estimator.predict(X))))

    num_tree = estimator.estimators_[0]

    print('\nVisualizing Decision Tree:', 0)

    

    graph = Source(tree.export_graphviz(num_tree,

                                        out_file=None,

                                        feature_names=X.columns,

                                        class_names=['0', '1'],

                                        filled = True))

    

    display(Image(data=graph.pipe(format='png')))

    

    return estimator
estimator = plot_tree_rf(crit='gini', bootstrap='False' , depth =16  , forests=100 , min_split=3 , min_leaf= 3)



y_pred_rf = estimator.predict(df_test)

print('len',len(y_pred_rf))



sub = pd.DataFrame(columns=['PassengerId' , 'Survived'])



sub['PassengerId'] = df_test['PassengerId'].astype(int)

sub['Survived'] = y_pred_rf.astype(int)



sub.to_csv('sub_rf.csv', index=False)
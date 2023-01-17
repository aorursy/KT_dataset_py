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
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.impute import SimpleImputer;
from sklearn.compose import ColumnTransformer;
from sklearn.pipeline import Pipeline;
from sklearn.preprocessing import LabelEncoder;
from sklearn.preprocessing import StandardScaler;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression ;
from sklearn.linear_model import Ridge, Lasso;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import r2_score;
from sklearn.preprocessing import PolynomialFeatures;
from sklearn.svm import SVR;
from sklearn.svm import SVC;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.naive_bayes import GaussianNB;
import pickle;
x = pd.read_csv('../input/titanic/train.csv');
y = pd.read_csv('../input/titanic/test.csv');
pd.set_option('display.max_columns', None);
pd.set_option('display.max_rows', None);
x.shape, y.shape
x.isnull().sum()
y.isnull().sum()
x1 = x.drop(columns = 'Cabin', axis =1);
y1 = y.drop(columns = 'Cabin', axis =1);

x1.isnull().sum()
y1.isnull().sum()
x2 = x1.fillna({'Embarked': 'C'})
x2.isnull().sum()
for var in y1['Embarked'].unique():
    y1.update(y1[y1.loc[:,'Embarked'] == var]['Fare'].replace(np.nan,y1[y1.loc[:,'Embarked'] == var]['Fare'].mean() ))
y1.isnull().sum()
x2['Age'].mean()
for varr in x2['Embarked'].unique():
    x2.update(x2[x2.loc[:,'Embarked'] == varr]['Age'].replace(np.nan,x2[x2.loc[:,'Embarked'] == varr]['Age'].mean() ))
x2.isnull().sum()
x2['Age'].mean()
for varrr in y1['Embarked'].unique():
    y1.update(y1[y1.loc[:,'Embarked'] == varrr]['Age'].replace(np.nan,y1[y1.loc[:,'Embarked'] == varrr]['Age'].mean() ))
y1.isnull().sum()
x3 = x2.drop(columns = ['Name', 'Ticket'], axis = 1);
y2 = y1.drop(columns = ['Name', 'Ticket'], axis = 1)
x3.columns
order = {'male':1, 'female': 0};
x3['Sex']=x3['Sex'].map(order);
y2['Sex']=y2['Sex'].map(order);
orders = {'S':1, 'Q': 3, 'C': 2};
x3['Embarked']=x3['Embarked'].map(orders);
y2['Embarked']=y2['Embarked'].map(orders);
y2
xx = x3.drop(columns = 'Survived', axis =1 );
yy = x3[['Survived']]
x_train, x_test, y_train, y_test = train_test_split(xx, yy, test_size = 0.2, random_state = 51)
#sc = StandardScaler();
#sc.fit_transform(x_train);
#xxc_train = sc.fit_transform(x_train);
#xxc_test = sc.fit_transform(x_test);

#xx_train = pd.DataFrame(xxc_train, columns = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked' ]);
#xx_test = pd.DataFrame(xxc_test, columns = ['PassengerId',  'Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked' ]);


dfc = DecisionTreeClassifier()
dfc.fit(x_train, y_train)
#sv =SVC();
#sv.fit(xx_train, y_train)

dfc.score(x_test, y_test)
#y2

#sv.score(xx_test, y_test)

#model = sv.predict(y2)
model = dfc.predict(y2);
modelData = pd.DataFrame(model, columns = ['Survived']);
modelData.set_index('Survived').to_csv('submission.csv')
#pd.merge(y2, modelData)
#sub = pd.concat(y2, modelData, left_on = 'old', right_on = 'newww')
#y2
#sub = pd.merge(y2, mm, how = 'outer');
#sub
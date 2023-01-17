# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/5dc975b0250f1_Training_Dataset.csv')
train.drop(['Unnamed: 0','no_of_rounds'],1,inplace=True)
na_columns = train.isna().sum()

print(na_columns[na_columns!=0])
submission = pd.read_csv('/kaggle/input/5dc97593aef65_Testing_Dataset_final.csv')
test = pd.read_csv('/kaggle/input/5dc97593aef65_Testing_Dataset_final.csv')

test.drop(['Sex','Unnamed: 0'],1,inplace=True)
na_columns = test.isna().sum()

print(na_columns[na_columns!=0])
test['Winner'] = 'M'
data = pd.concat([train, test], sort=False).reset_index(drop = True)
data
categorical = data.select_dtypes(exclude = np.number)

categoricalCols = list(categorical.columns)

numerical = data.select_dtypes(include = np.number)

numericCols = list(numerical.columns)
import scipy.stats as stats

from scipy.stats import chi2_contingency



class ChiSquare:

    def __init__(self, dataframe):

        self.df = dataframe

        self.p = None #P-Value

        self.chi2 = None #Chi Test Statistic

        self.dof = None

        

        self.dfObserved = None

        self.dfExpected = None

        

    def _print_chisquare_result(self, colX, alpha):

        result = ""

        if self.p<alpha:

            result="{0} is IMPORTANT for Prediction".format(colX)

        else:

            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)



        print(result)

        

    def TestIndependence(self,colX,colY, alpha=0.05):

        X = self.df[colX].astype(str)

        Y = self.df[colY].astype(str)

        

        self.dfObserved = pd.crosstab(Y,X) 

        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)

        self.p = p

        self.chi2 = chi2

        self.dof = dof 

        

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)

        

        self._print_chisquare_result(colX,alpha)
cT = ChiSquare(data)

#Feature Selection

for var in categoricalCols:

    cT.TestIndependence(colX=var,colY="Winner" )
categoricalCols = ['date','location','Winner','title_bout']
correlationMatrix = train.corr().abs()

upper = correlationMatrix.where(np.triu(np.ones(correlationMatrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]

to_drop

for i_var in to_drop:

    numericCols.remove(i_var)

numericCols
to_drop
from sklearn.preprocessing import LabelEncoder

for c in categoricalCols:

    lb = LabelEncoder()

    data[c] = lb.fit_transform(data[c])
data
features = numericCols + categoricalCols

data = data.loc[:,features]

data.head()
train = data[data['Winner']!=2]

test = data[data['Winner']==2]
y = train['Winner']

train.drop(['Winner'],1,inplace=True)
test.drop(['Winner'],1,inplace=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.2, random_state = 42)

y_train = pd.DataFrame(y_train)

y_test = pd.DataFrame(y_test)
from sklearn.tree import DecisionTreeClassifier 

dtree_model = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train) 

dtree_predictions = dtree_model.predict(X_test) 
accuracy = dtree_model.score(X_test, y_test) 

print(accuracy)

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, dtree_predictions))

from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test, dtree_predictions)

cm
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 12,max_depth=3, criterion = 'entropy',random_state = 0)

classifier.fit(X_train, y_train)

dtree_predictions = dtree_model.predict(X_test) 
accuracy = dtree_model.score(X_test, y_test) 

print(accuracy)

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, dtree_predictions))

from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test, dtree_predictions)

cm
from xgboost import XGBClassifier

classifier = XGBClassifier(objective='multi:softmax', random_state = 0)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test) 
accuracy = classifier.score(X_test, y_test) 

print(accuracy)

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, predictions))

from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test, predictions)

cm
from xgboost import XGBClassifier

classifier = XGBClassifier(objective='multi:softmax', random_state = 0)

classifier.fit(train, y)

predictions = classifier.predict(test) 
submission['Winner'] = predictions
submission['Winner'] = np.where(submission['Winner']==0, 'Blue', submission['Winner'])

submission['Winner'] = np.where(submission['Winner']=='3', 'Red', submission['Winner'])
submission['Winner'].unique()
submission.to_csv('submission.csv')
from lightgbm import LGBMClassifier

import lightgbm as lgb

classifier = LGBMClassifier(objective='multi:softmax', random_state = 0)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test) 
accuracy = classifier.score(X_test, y_test) 

print(accuracy)

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, predictions))

from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test, predictions)

cm
feature_importances = pd.DataFrame()

feature_importances['feature'] = X_train.columns

feature_importances['average'] = classifier.feature_importances_
plt.figure(figsize=(15, 8))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(20), x='average', y='feature');

plt.title(' TOP 20 feature importance',fontsize=15)

plt.xticks(fontsize=15)

plt.ylabel('Features',fontsize=20)

plt.yticks(fontsize=15)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(multi_class='multinomial',solver='lbfgs', random_state = 42,class_weight='balanced')
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, predictions)
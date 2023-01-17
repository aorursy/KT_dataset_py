# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report



from scipy import stats

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score





import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LinearRegression



from sklearn.linear_model import Ridge



import matplotlib.pyplot as plt



from sklearn.model_selection import KFold

import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier





from sklearn.ensemble import RandomForestRegressor



from sklearn.preprocessing import OneHotEncoder



from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB



from sklearn.model_selection import cross_validate



heart = pd.read_csv('../input/heart-disease-uci/heart.csv')
heart.head(10)
heart.info()
heart.describe()
heart.hist(figsize=(20,20))
heart.skew()
(np.log1p(heart)).skew()
heart.columns
sns.boxplot(x="cp", y ='target', data=heart )
sns.countplot(x="cp", data=heart )
sns.boxplot(data=heart, x='sex',y= 'target')
sns.distplot(heart['sex'])
sns.distplot(heart['cp'])
heart.corr()['target']
#Correlation Matrix

fig = plt.figure(figsize=[15,15])

plt.title('Pearson Correlation of Features', y=1.05, size=15)



sns.heatmap(heart.corr(), annot = True, square=True,linecolor='white',cmap='coolwarm' )
heart.tail()
y = heart['target']

heart.sample(10)
heart.nunique()
hea = heart.copy()
hea_dummy = pd.get_dummies(data=hea, columns= ['cp','fbs','restecg','exang','slope','ca','thal'], drop_first=True)
hea_dummy.head()
hea_dummy.columns
scaler = StandardScaler()

X_new_dummy = scaler.fit_transform(hea_dummy.drop(columns='target'))
selector = SelectFromModel(estimator=RandomForestRegressor(n_jobs=-1, n_estimators=100)).fit(X_new_dummy, y)

X_new_dummy = selector.transform(X_new_dummy)
X_train, X_test, y_train, y_test = train_test_split(

    X_new_dummy, y, test_size=0.2, random_state=42)
model ={'clf':DecisionTreeClassifier(max_depth=3,min_samples_leaf =10, random_state=0),

       'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0),

       'Logistic Regression': LogisticRegression(random_state=0),

      'Gaussian Naive': GaussianNB() }



for keys, items in model.items():

    cv_results = cross_validate(items, X_new_dummy, y, cv=5, scoring=('r2', 'f1','precision','recall','roc_auc'))

    print("keys" + "  " + str(keys),"\n")

    print("Recall:  ", cv_results['test_recall'])

    print("Precision:  ", cv_results['test_precision'],"\n")

    print("AUC:  ", cv_results['test_roc_auc'])

    print("max AUC:  ", max(cv_results['test_roc_auc']),"\n")

    
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
titanic = pd.read_csv("/kaggle/input/titanic/train.csv")
titanic.head()
titanic.info() 
import matplotlib.pyplot as plt 

titanic.hist(figsize=(40,30))
plt.show()
corr = titanic.corr() 
corr
import matplotlib.pyplot as plt 
import seaborn as sns 

plt.figure(figsize = (14, 7))
sns.heatmap(corr,annot=True,linecolor="black",lw=0.5)
plt.show()
def analyze_data_correlations(df, res_column_name="Survived"):
    columns_list = list(df); columns_list.remove(res_column_name)
    for x in columns_list:
        print(titanic[[x, 'Survived']].groupby([x], as_index=False).mean().sort_values(by='Survived', ascending=False))
        print("*" * 40)
analyze_data_correlations(titanic)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import RobustScaler

useless_attribs = ["Ticket", "Name", "Cabin", "PassengerId"]
cat_attribs = ["Embarked", "Sex"]
num_attribs = ["Age", "Fare", ]

cat_to_ord_dict = {"Embarked": {'S': 0, 'C': 1, 'Q': 2}, "Sex": {"male": 0, "female": 1}}

class FullTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, fillna_values=None): 
        self.fillna_vals = fillna_values.copy() if fillna_values != None else None 
    def fit(self, X, y=None):
        return self 
    @staticmethod
    def get_age_ord(age):
        if   age <= 16: return 0 
        elif age <= 32: return 1
        elif age <= 48: return 2
        elif age <= 64: return 3
        else          : return 4 
    @staticmethod 
    def get_fare_ord(fare):
        if   fare <= 7.91  : return 0
        elif fare <= 14.454: return 1
        elif fare <= 31    : return 2
        else               : return 3
    def transform(self, X, y = None):
        X_transformed = X.copy() 
        fillna_vals = self.fillna_vals  
        
        #delete useless attributes
        X_transformed.drop(useless_attribs, axis=1, inplace=True)
        
        #fill missing numerical values with mean value and categorical+PClass with mode 
        if fillna_vals == None: 
            num_median_values = {x: X_transformed[x].median() for x in num_attribs}
            cat_mode_values = {x: X_transformed[x].mode()[0] for x in cat_attribs+["Pclass"]}
            fillna_vals = {**num_median_values, **cat_mode_values} #merge two dictionaries
        X_transformed.fillna(value=fillna_vals, inplace=True)
        
        #transform age and fare to ordinal
        X_transformed["Age"] = X_transformed["Age"].apply(FullTransformer.get_age_ord) 
        X_transformed["Fare"] = X_transformed["Fare"].apply(FullTransformer.get_fare_ord) 
        #transform categorical to ordinal 
        for attr in cat_attribs: X_transformed[attr] = X_transformed[attr].map(cat_to_ord_dict[attr]).astype(int) 
        
        #replace SibSp and Parch features with more useful IsAlone feature 
        X_transformed["IsAlone"] = ((X_transformed["SibSp"] + X_transformed["Parch"] + 1) == 1).astype(int)
        X_transformed.drop(["SibSp", "Parch"], axis=1, inplace=True)    
        
        if self.fillna_vals == None:
            return X_transformed, fillna_vals
        return X_transformed
y_train = titanic["Survived"]
X_train = titanic.drop("Survived", axis=1)
X_train.info()
X_train_prepared, fillna_vals = FullTransformer().fit_transform(X_train)
fillna_vals
X_train_prepared
pd.DataFrame(X_train_prepared).head(30)
pd.DataFrame(titanic).head(30)
embarked = {0: 0, 1: 0, 2: 0}
sex = {0: 0, 1:0}
for x in range(len(X_train_prepared)):
    if y_train[x] == 1:
        embarked[X_train_prepared[x][5]]+=1
        sex[X_train_prepared[x][6]]+=1
print(embarked)
print(sex)
corr_pred = pd.DataFrame(X_train_prepared).copy(); corr_pred["Survived"] = y_train; corr_pred = corr_pred.corr()  

plt.figure(figsize = (14, 7))
sns.heatmap(corr_pred, annot=True,linecolor="black",lw=0.5)
plt.show()
coeff = pd.DataFrame(X_train_prepared.columns)
coeff.columns = ['Feature']
coeff["Correlation"] = pd.Series(lin_clf.coef_[0])

coeff.sort_values(by='Correlation', ascending=False)
pd.DataFrame(X_train_prepared).hist(figsize=(30,25))
plt.show()
from sklearn.linear_model import SGDClassifier
lin_clf = SGDClassifier()
lin_clf.fit(X_train_prepared, y_train)
some_data = X_train.iloc[:5] 
some_labels = y_train[:5] 
some_data_prepared = FullTransformer(fillna_vals).transform(some_data)
print("Predictions: ", lin_clf.predict(some_data_prepared))
print("Labels: ", list(some_labels))
lin_clf.score(X_train_prepared, y_train)
from sklearn.model_selection import cross_val_score 
cross_val_score(lin_clf, X_train_prepared, y_train, cv=5, scoring="accuracy")
from sklearn.svm import LinearSVC 
svc_clf = LinearSVC(C=1, loss="hinge")
cross_val_score(svc_clf, X_train_prepared, y_train, cv=3, scoring="accuracy")
svc_clf.fit(X_train_prepared, y_train)
svc_clf.score(X_train_prepared, y_train)
from sklearn.tree import DecisionTreeClassifier 

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train_prepared, y_train) 
tree_clf.score(X_train_prepared, y_train)
cross_val_score(tree_clf, X_train_prepared, y_train, cv=3, scoring="accuracy")
X_test = pd.read_csv("/kaggle/input/titanic/test.csv")
X_test_prepared = FullTransformer(fillna_vals).fit_transform(X_test)
y_test_predict = tree_clf.predict(X_test_prepared)
result = pd.DataFrame({"PassengerId": X_test["PassengerId"],
                       "Survived": y_test_predict})
result.to_csv("submission.csv", index=False)
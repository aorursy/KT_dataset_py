#import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%%capture

!pip install trelawney
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()
train = train.drop(["Name", "Ticket", "Fare"],axis=1)

test = test.drop(["Name", "Ticket", "Fare"],axis=1)
train.head()
train.isna().sum()
#fill the missing cabin values with mode

train["Cabin"] = train["Cabin"].fillna(str(train["Cabin"].mode().values[0]))

test["Cabin"] = test["Cabin"].fillna(str(test["Cabin"].mode().values[0]))
train["Cabin"] = train["Cabin"].apply(lambda x:str(x).replace(' ','')if ' ' in str(x) else str(x))

test["Cabin"] = test["Cabin"].apply(lambda x:str(x).replace(' ','')if ' ' in str(x) else str(x))
train["Deck"] = train["Cabin"].str.slice(0,1)

test["Deck"] = test["Cabin"].str.slice(0,1)
train = train.drop(["Cabin"],axis=1)

test = test.drop(["Cabin"],axis=1)
def impute_median(series):

    return series.fillna(series.median())
train.Age = train.Age.transform(impute_median)

test.Age = test.Age.transform(impute_median)
train["Embarked"] = train["Embarked"].fillna("S")

test["Embarked"] = test["Embarked"].fillna("S")
train.isnull().sum()
test.isnull().sum()
train['Is_Married'] = np.where(train['SibSp']==1, 1, 0)

test['Is_Married'] = np.where(test['SibSp']==1, 1, 0)



train.head()
train["Family_Size"] = train.SibSp + train.Parch

test["Family_Size"] = test.SibSp + test.Parch



train.head()
train['Elderly'] = np.where(train['Age']>=50, 1, 0)

train.head()
#Split the data set into independent(x) and dependent (y) data sets



y = train["Survived"].values.reshape(-1, 1)

x = train.iloc[:, 2:12]

x_test  = test.drop("PassengerId",axis=1).copy()
x.dtypes
x_test.dtypes
from collections import Counter
##### encode the categorical data values

from sklearn.preprocessing import LabelEncoder





labelEncoder_Y = LabelEncoder()

x.iloc[:,1] = labelEncoder_Y.fit_transform(x.iloc[:, 1].values)

x_test.iloc[:,1] = labelEncoder_Y.transform(x_test.iloc[:, 1].values)



x.iloc[:,5] = labelEncoder_Y.fit_transform(x.iloc[:, 5].values)

x_test.iloc[:,5] = labelEncoder_Y.transform(x_test.iloc[:, 5].values)



x.iloc[:,6] = labelEncoder_Y.fit_transform(x.iloc[:, 6].values)

x_test.iloc[:,6] = labelEncoder_Y.transform(x_test.iloc[:, 6].values)
x.dtypes
x_test.dtypes
#split the data set

from sklearn.model_selection import train_test_split



x_train, x_val, y_train, y_val = train_test_split(x, y , test_size=0.25, random_state=42)
#scale the data(feature scaling)

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

x_train = pd.DataFrame(sc.fit_transform(x_train), columns=x_train.columns, index=x_train.index)

x_val = pd.DataFrame(sc.fit_transform(x_val), columns=x_val.columns, index=x_val.index)

y_train = pd.DataFrame(y_train, index=x_train.index)

y_val = pd.DataFrame(y_val, index=x_val.index)
x_train.shape, x_val.shape, y_train.shape, y_val.shape
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
model = RandomForestClassifier(min_samples_leaf=3, max_depth=5, n_estimators=100)
model.fit(x_train, y_train)

print(metrics.classification_report(y_val, model.predict(x_val)))
from trelawney.shap_explainer import ShapExplainer



explainer = ShapExplainer()

explainer.fit(model, x_train, y_train)
feature_importance_graph = explainer.graph_feature_importance(x_val)

feature_importance_graph.update_layout(title='Shap Feature Importance')

feature_importance_graph.show()
from trelawney.surrogate_explainer import SurrogateExplainer

from sklearn.tree import DecisionTreeClassifier



explainer = SurrogateExplainer(DecisionTreeClassifier(max_depth=4))

explainer.fit(model, x_train, y_train)
from IPython.display import Image

explainer.plot_tree(out_path='./tree_viz')

Image('./tree_viz.png', width=1000, height=500)
explainer.adequation_score()
from sklearn import metrics
explainer.adequation_score(metric=metrics.roc_auc_score)
y_pred = pd.DataFrame(model.predict_proba(x_val)[:, 1], index=x_val.index)
most_probable = y_pred.idxmax()

biggest_false_positive = (y_pred - y_val).idxmax()

biggest_false_negative = (y_pred - y_val).idxmin()
from trelawney.lime_explainer import LimeExplainer
explainer = LimeExplainer()

explainer.fit(model, x_train, y_train)
x_val.loc[most_probable, :]
lime_explanation_graph = explainer.graph_local_explanation(x_val.loc[most_probable, :])

lime_explanation_graph.update_layout(title='Lime individual prediction interpretation')

lime_explanation_graph.show()
x.loc[biggest_false_positive, :]
lime_explanation_graph = explainer.graph_local_explanation(x_val.loc[biggest_false_positive, :])

lime_explanation_graph.update_layout(title='Lime individual prediction interpretation')

lime_explanation_graph.show()
x.loc[biggest_false_negative, :]
lime_explanation_graph = explainer.graph_local_explanation(x_val.loc[biggest_false_negative, :])

lime_explanation_graph.update_layout(title='Lime individual prediction interpretation')

lime_explanation_graph.show()
from trelawney.shap_explainer import ShapExplainer



explainer = ShapExplainer()

explainer.fit(model, x_train, y_train)
shap_explanation_graph = explainer.graph_local_explanation(x_val.loc[most_probable, :])

shap_explanation_graph.update_layout(title='SHAP individual prediction interpretation')

shap_explanation_graph.show()
shap_explanation_graph = explainer.graph_local_explanation(x_val.loc[biggest_false_positive, :])

shap_explanation_graph.update_layout(title='SHAP individual prediction interpretation')

shap_explanation_graph.show()
shap_explanation_graph = explainer.graph_local_explanation(x_val.loc[biggest_false_negative, :])

shap_explanation_graph.update_layout(title='SHAP individual prediction interpretation')

shap_explanation_graph.show()
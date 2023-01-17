import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
df= pd.read_csv("../input/hr-analytics/HR_comma_sep.csv")
df.head()
df.salary = df.salary.map({"low": 1, "medium":2, "high":3})
X =  df[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company","Work_accident","promotion_last_5years", "salary"]]
y= df.left
df.left.value_counts()
sns.countplot(df.left)
corr = X.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot = True, annot_kws={"size": 12},square = True )
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)
rfc.fit(X_train, y_train)
pd.Series(rfc.feature_importances_).sort_values(ascending =False)
from sklearn.feature_selection import RFECV
rfc = RandomForestClassifier()
rfecv = RFECV(estimator=rfc, step = 1, cv =5, scoring ="accuracy")
rfecv= rfecv.fit(X_train, y_train)
rfecv.n_features_
X_train.columns[rfecv.support_]
X= df[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'salary']]
def Classifier(model, param_grid, X,y):
    clf= GridSearchCV(model, param_grid, cv=10, scoring ="accuracy")
    clf.fit(X,y)
    print("The best parameter found on development set is: ")
    print(clf.best_params_)
    print("The best estimator is:")
    print(clf.best_estimator_)
    print ("The best score is:")
    print(clf.best_score_)
#for DecisionTreeClassifier parameters:
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'min_samples_split': [2,3,4,5,6,7,8,9,10], 
              'min_samples_leaf':[2,3,4,5,6,7,8,9,10] }

model = DecisionTreeClassifier()
Classifier(model,param_grid,X,y)

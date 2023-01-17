import pandas as pd

import numpy as np

from IPython.core.display import display 
df = pd.read_csv("../input/train.csv", header=0)

display(df.head(5)) 
X = df.iloc[:, 2:] 

y = df.iloc[:, 1] 

print(display(X.head(5))) 

print(y.head(5))
X1 = X.iloc[:, [0,2,3,4,5,7,9]]

X1.head(5)
print("--------------------------")

print("X1 shape: (%i,%i)" %X1.shape)

print("--------------------------")

print("Y shape: %i" %y.count()) 

print("--------------------------")

print("Check the null count of target variable: %i" % y.isnull().sum())

print("--------------------------")
ohe_columns = ["Sex", "Embarked"]

X_new = pd.get_dummies(X1, dummy_na=False, columns=ohe_columns)

display(X_new.head())
print(X_new.describe())

print(display(X_new))
from sklearn.preprocessing import Imputer



imp = Imputer(missing_values="NaN",

              strategy="mean",

              axis=0)



imp.fit(X_new)

X_new_columns = X_new.columns.values

X_new = pd.DataFrame(imp.transform(X_new), columns=X_new_columns)

print(X_new.describe())

print(display(X_new))
print("--------------------------")

print("X_new shape: (%i,%i)" %X_new.shape)

print("--------------------------")

print("Y shape: %i" %y.count()) 

print("--------------------------")

print("Check the null count of target variable: %i" % y.isnull().sum())

print("--------------------------")
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn. metrics import f1_score
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=1)
pipelines = {

    "knn": 

          Pipeline([("scl", StandardScaler()),

                    ("est", KNeighborsClassifier())]),

    "logistic": 

          Pipeline([("scl", StandardScaler()),

                    ("est", LogisticRegression(random_state=1))]),

    "rsvc": 

          Pipeline([("scl", StandardScaler()),

                    ("est", SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=1))]),

    "lsvc": 

          Pipeline([("scl", StandardScaler()),

                    ("est", LinearSVC(max_iter=1000,C=1.0, class_weight="balanced", random_state=1))]),

    "tree": 

          Pipeline([("scl", StandardScaler()),

                    ("est", DecisionTreeClassifier(max_depth=3, random_state=1))]),

    "rf": 

          Pipeline([("scl", StandardScaler()),

                    ("est", RandomForestClassifier(max_depth=100,random_state=1))]),

    "gb": 

          Pipeline([("scl", StandardScaler()),

                    ("est", GradientBoostingClassifier(learning_rate=0.2,max_depth=50, random_state=1))]),

    "mlp": 

          Pipeline([("scl", StandardScaler()),

                    ("est", MLPClassifier(hidden_layer_sizes=(8,8),max_iter=1000,random_state=1))])                     

}
scores = {}

for pipe_name, pipeline in pipelines.items():

    pipeline.fit(X_train, y_train)

    scores[(pipe_name, "train")] = f1_score(y_train, pipeline.predict(X_train))

    scores[(pipe_name, "test")] = f1_score(y_test, pipeline.predict(X_test))



pd.Series(scores).unstack()
df_s = pd.read_csv("../input/test.csv", header=0) 

display(df_s.head(5)) 
X_s = df_s.iloc[:, [1,3,4,5,6,8,10]]

print(display(X_s.head(5))) 

print(X_s.dtypes)
X_ohe_s = pd.get_dummies(X_s, dummy_na=False, columns=ohe_columns)

print("X_ohe_s shape: (%i,%i)" % X_ohe_s.shape)

X_ohe_s.head(3)
print(display(X_new.head(3)))

print(display(X_ohe_s.head(3)))
X_ohe_s1 = pd.DataFrame(imp.transform(X_ohe_s), columns=X_new_columns)

print(display(X_ohe_s1.head(5)))

print(X_ohe_s1.describe())
r_logistic = pipelines["logistic"].predict(X_ohe_s1)

r_lsvc = pipelines["lsvc"].predict(X_ohe_s1)

r_tree = pipelines["tree"].predict(X_ohe_s1)
#logistic

df1 = pd.DataFrame(df_s.iloc[:,0])

df2 = pd.DataFrame(r_logistic, columns=["Survived"])

pre_logistic = pd.concat([df1,df2], axis=1)
#lsvc

df1 = pd.DataFrame(df_s.iloc[:,0])

df3 = pd.DataFrame(r_lsvc, columns=["Survived"])

pre_lsvc = pd.concat([df1,df3], axis=1)
#tree

df1 = pd.DataFrame(df_s.iloc[:,0])

df4 = pd.DataFrame(r_tree, columns=["Survived"])

pre_tree = pd.concat([df1,df4], axis=1)
#CSV

pre_logistic.to_csv("titanic_result_log.csv")

pre_lsvc.to_csv("titanic_result_lsvc.csv")

pre_tree.to_csv("titanic_result_tree.csv")
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
#Import the needed libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif


from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score

sns.set_style("dark")
## Import the dataset

data = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
data.head()
data.shape
data.info()
to_replace =["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

def makeNaN(x):
    if x == 0:
        return np.nan
    else:
        return x

for col in to_replace: 
    data[col] = data[col].apply(makeNaN)

data.info()
plt.figure(figsize = (10, 5))

sns.boxplot(x = "Insulin", data = data)
data[["Insulin", "Outcome"]].groupby("Outcome").mean()
data[data["Insulin"].isnull()]["Outcome"].value_counts()
plt.figure(figsize = (15, 8))

sns.boxplot(data = data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "BMI", "DiabetesPedigreeFunction", "Age"]])
plt.figure(figsize = (15, 8))

sns.distplot(a = data["Age"], kde = False)
plt.figure(figsize = (15, 8))
sns.heatmap(data = data.corr(), annot = True, cmap="YlGnBu")

fig, ax = plt.subplots(1,2, figsize = (20, 8))
sns.regplot(x = "Glucose", y = "Insulin", data = data, ax = ax[0])
sns.regplot(x = "BMI", y = "SkinThickness", data = data, ax = ax[1], color = "darkorange")
fig.show()
data[data["SkinThickness"] > 70]
data[(data["SkinThickness"] < 99) & (data["Glucose"] > 150) & (data["Glucose"] < 250)]["SkinThickness"].mean()  
data[(data["SkinThickness"] < 99) & (data["BloodPressure"] > 50) & (data["BloodPressure"] < 100)]["SkinThickness"].mean()  
BMI_cat = pd.cut(data["BMI"], bins=(0, 18.5, 25, 30, 100), labels=["Underweight", "Normal Weight", "Overweight", "Obese"])
plt.figure(figsize = (15, 8))
sns.barplot(x = ["Underweight", "Normal Weight", "Overweight", "Obese"], y = BMI_cat.value_counts().sort_index(), color = "skyblue")
plt.ylabel("Count")
plt.show()
plt.figure(figsize = (15, 8))
sns.swarmplot(y = "BMI", x = "Outcome", data = data)
y = data["Outcome"]
X = data.drop(["Outcome"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 5)
print("The train set contains ", X_train.shape[0]," samples")
print("The test set contains ", X_test.shape[0]," samples")
imputer = SimpleImputer(strategy = "mean")
scaler = StandardScaler()
model = LogisticRegression()

pipeline = Pipeline(steps = [
        ("imputing", imputer),
        ('scaling', scaler),
        ("model", model)
        ])


scores = cross_val_score(pipeline, X_train, y_train, cv = 5, scoring = "accuracy")

print("The average score is: ", scores.mean())
print("The standard deviation is: ", scores.std())
scores = cross_validate(pipeline, X_train, y_train, cv = 5, scoring = "accuracy", return_train_score = True)

train_scores = scores["train_score"]
val_scores = scores["test_score"]

gap = np.abs(np.subtract(train_scores, val_scores))

print("The average gap between training and validation fold is: ", gap.mean())
print("The standard deviation of the gaps is: ", gap.std())

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_train)
report = classification_report(y_train, y_pred)
print("The classification report on the training set is: ")
print()
print(report)
to_impute = list(X.columns)
to_impute.remove("Insulin")
regr_imp = IterativeImputer(random_state=0)

imputing = ColumnTransformer(transformers=
                                 [('imputing', imputer, to_impute ),
                                 ('regression', regr_imp, ["Glucose", "Insulin"]),
                                 ])

pipeline_imputing = Pipeline(steps = [
        ("imputing", imputing),
        ('scaling', scaler),
        ("model", model)
        ])


scores = cross_val_score(pipeline_imputing, X_train, y_train, cv = 5, scoring = "accuracy")
print("The average score is: ", scores.mean())
print("The standard deviation is: ", scores.std())
cols = list(X.columns)

poly = PolynomialFeatures(2, include_bias = False)

model = LogisticRegression()

interaction = ColumnTransformer(transformers=
                                 [('imputing', imputer, to_impute ),
                                 ('regression', regr_imp, ["Glucose", "Insulin"])
                                 
                                 ])

pipeline_int = Pipeline(steps = [
        ("interaction", interaction),
        ("scaling", scaler),
        ('inter', poly),
        ("model", model)
        ])


scores = cross_val_score(pipeline_int, X_train, y_train, cv = 5, scoring = "accuracy")

print("The average score is: ", scores.mean())
print("The standard deviation is: ", scores.std())

cols = list(X.columns)

poly = PolynomialFeatures(2, include_bias = False)

model = LogisticRegression()

interaction = ColumnTransformer(transformers=
                                 [('imputing', imputer, to_impute ),
                                 ('regression', regr_imp, ["Glucose", "Insulin"])
                                 
                                 ])

pipeline_int = Pipeline(steps = [
        ("interaction", interaction),
        ("scaling", scaler),
        ('inter', poly),
        ("select", SelectKBest(f_classif, k=3 )),
        ("model", model)
         ])


scores = cross_val_score(pipeline_int, X_train, y_train, cv = 5, scoring = "accuracy")

print("The average score is: ", scores.mean())
print("The standard deviation is: ", scores.std())

model = LogisticRegression()

pipeline = Pipeline(steps = [
        ("imputing", imputer),
        ('scaling', scaler),
        ("model", model)
        ])


scores = cross_val_score(pipeline, X_train, y_train, cv = 5, scoring = "accuracy")

print("The average score is: ", scores.mean())
print("The standard deviation is: ", scores.std())

inner_cv = StratifiedKFold(n_splits=5, shuffle = True)
outer_cv = StratifiedKFold(n_splits=10, shuffle = True)

model_scores = {"train": [],
                "test": []
               }

models = [
    {
        "name": "Rand Forest Classifier",
        "estimator": RandomForestClassifier(),
        "params": {
                    "n_estimators": range(50, 200, 50),
                    "max_depth": [1, 3, 5], 
                    "random_state": [48]
                      }
    },
    
    {
        "name": "KNN Classifier",
        "estimator": KNeighborsClassifier(),
        "params": {
                    "n_neighbors": [3,5,7,10],
                    }
    },
    
    {
        "name": "Logistic Regression",
        "estimator": LogisticRegression(),
        "params": {
                    "C": [0.01, 0.1, 1, 10, 100, 1000],
                    "random_state": [48]
                }
    },
    
    {
        "name": "Support Vector Machine",
        "estimator": SVC(),
        "params": {
                    "C": [0.01, 0.1, 1, 10, 100, 1000],
                    "gamma": ["auto"],
                    "random_state": [48]
                }
    }
    
    ]

for alg in models:

    model = GridSearchCV(alg["estimator"], alg["params"], scoring ="accuracy", cv = inner_cv)

    pipeline = Pipeline(steps = [
        ("imputing", imputer),
        ('scaling', scaler),
        ("model", model)
        ])




    scores = cross_validate(pipeline, X_train, y_train, cv =  outer_cv, return_train_score = True)

    train_scores = np.array(scores["train_score"])
    test_scores = np.array(scores["test_score"])
    
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print()
    print(alg["name"])
    print("The average  training score is:", train_scores.mean())
    print("The standard deviation of the training score is:", train_scores.std())
    print()
    print("The average  validation score is:", test_scores.mean())
    print("The standard deviation of the validation score is:", test_scores.std())
    print()
    
    model_scores["train"].append(train_scores.mean())
    model_scores["test"].append(test_scores.mean())

model_scores["train"]
scores_data = pd.DataFrame({"Name" : ["RF", "KNN", "LR", "SVM"],
                            "train": model_scores["train"],
                            "test": model_scores["test"]
                           })

plt.figure(figsize = (15, 8))
sns.barplot(x = "Name", y = "test", data = scores_data)


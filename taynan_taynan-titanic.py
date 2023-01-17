import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
# print(os.listdir("../input"))

training = pd.read_csv("../input/train.csv")
training.head()
training.shape
training.describe()
training.info()
categoricas = ["Sex", "Embarked"]
ordinais = ["Pclass"]
numericas = ["Age", "SibSp", "Parch", "Fare"]
label = ["Survived"]
training["classe"] = training["Pclass"] / 3.0
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
label_encoder = LabelEncoder()
label_encoder.fit(training["Sex"])
training["SexLE"] = label_encoder.transform(training["Sex"])
training["EmbarkedNoMissing"] = training["Embarked"].fillna(value='S')
label_encoder = LabelEncoder()
label_encoder.fit(training["EmbarkedNoMissing"])
training["EmbarkedLE"] = label_encoder.transform(training["EmbarkedNoMissing"])
training["Embarked_0"] = np.where(training["EmbarkedLE"] == 0, 1, 0)
training["Embarked_1"] = np.where(training["EmbarkedLE"] == 1, 1, 0)
training["Embarked_2"] = np.where(training["EmbarkedLE"] == 2, 1, 0)
training.head()
categoricas = ["SexLE", "Embarked_0", "Embarked_1", "Embarked_2"]
ordinais = ["classe"]
numericas = ["Age", "SibSp", "Parch", "Fare"]
label = ["Survived"]
X = training[categoricas + ordinais + numericas]
y = training[label]
X["Age"] = X["Age"].fillna(value=26.7)
categoricas = ["SexLE", "Embarked_0", "Embarked_1", "Embarked_2"]
ordinais = ["classe"]
numericas = ["Age", "SibSp", "Parch", "Fare"]
label = ["Survived"]
X.describe()
X = X[categoricas + ordinais + numericas]
y = y[label]
scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), columns=["SexLE", "Embarked_0", "Embarked_1", "Embarked_2", "classe", "Age", "SibSp", "Parch", "Fare"])
X.describe()
from sklearn.model_selection import StratifiedKFold
K = 10
SEED = 1
KFold = StratifiedKFold(n_splits=K)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
y = np.array(y)
parametros = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
tecnica = LogisticRegression()
RegressaoLogistica = GridSearchCV(tecnica, parametros, cv=10, scoring="roc_auc", return_train_score=True)
RegressaoLogistica.fit(X, y.ravel())
resultados = pd.DataFrame(RegressaoLogistica.cv_results_)
resultados
RegressaoLogistica.best_params_
parametros = {'n_estimators': [120, 300, 500], 'max_depth': [5, 8, 15], 'min_samples_split': [2, 10, 15], 'min_samples_leaf': [2, 5, 10]}
tecnica = RandomForestClassifier()
RandomForest = GridSearchCV(tecnica, parametros, cv=5, scoring="roc_auc", verbose=0, return_train_score=True)
RandomForest.fit(X, y.ravel())
resultados = pd.DataFrame(RandomForest.cv_results_)
resultados[resultados["rank_test_score"] <= 5]
RandomForest.best_params_
parametros = {'n_estimators': [200, 250, 300, 350, 500, 600, 750], 'learning_rate': [0.05, 0.1, 0.125, 0.15, 0.175, 0.2], 'max_depth': [2, 3, 4]}
tecnica = GradientBoostingClassifier()
GB = GridSearchCV(tecnica, parametros, cv=5, scoring="roc_auc", verbose=0, return_train_score=True)
GB.fit(X, y.ravel())
resultados = pd.DataFrame(GB.cv_results_)
resultados[resultados["rank_test_score"] <= 5]
GB.best_params_
test = pd.read_csv("../input/test.csv")
test.head()
test.shape
categoricas = ["Sex", "Embarked"]
ordinais = ["Pclass"]
numericas = ["Age", "SibSp", "Parch", "Fare"]
label = ["Survived"]
test["classe"] = test["Pclass"] / 3.0
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
label_encoder = LabelEncoder()
label_encoder.fit(test["Sex"])
test["SexLE"] = label_encoder.transform(test["Sex"])
test["EmbarkedNoMissing"] = test["Embarked"].fillna(value='S')
label_encoder = LabelEncoder()
label_encoder.fit(test["EmbarkedNoMissing"])
test["EmbarkedLE"] = label_encoder.transform(test["EmbarkedNoMissing"])
test["Embarked_0"] = np.where(test["EmbarkedLE"] == 0, 1, 0)
test["Embarked_1"] = np.where(test["EmbarkedLE"] == 1, 1, 0)
test["Embarked_2"] = np.where(test["EmbarkedLE"] == 2, 1, 0)
test.head()
categoricas = ["SexLE", "Embarked_0", "Embarked_1", "Embarked_2"]
ordinais = ["classe"]
numericas = ["Age", "SibSp", "Parch", "Fare"]
label = ["Survived"]
X = test[categoricas + ordinais + numericas]
X["Age"] = X["Age"].fillna(value=26.7)
X["Fare"] = X["Fare"].fillna(value=35.627)
categoricas = ["SexLE", "Embarked_0", "Embarked_1", "Embarked_2"]
ordinais = ["classe"]
numericas = ["Age", "SibSp", "Parch", "Fare"]
label = ["Survived"]
X.describe()
X = X[categoricas + ordinais + numericas]
scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), columns=["SexLE", "Embarked_0", "Embarked_1", "Embarked_2", "classe", "Age", "SibSp", "Parch", "Fare"])
RF_Predictions = RandomForest.predict(X)
LR_Predictions = RegressaoLogistica.predict(X)
GB_Predictions = GB.predict(X)
tabela_resultados = np.stack((RF_Predictions, LR_Predictions, GB_Predictions), axis=1)
tabela_resultados.shape
tabela_resultados = pd.DataFrame(tabela_resultados, columns=["Random Forest", "Regressão Logística", "Gradient Boosting"])
tabela_resultados["Votos"] = tabela_resultados["Random Forest"] + tabela_resultados["Regressão Logística"] + tabela_resultados["Gradient Boosting"]
def votacao(row):
    if row["Votos"] >= 2:
        return 1
    else:
        return 0
tabela_resultados["Majority Vote"] = tabela_resultados.apply(lambda row: votacao(row), axis=1)
tabela_resultados.head()
tabela_resultados["Majority Vote"].to_csv("submission.csv")

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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {"grid.linestyle":"--"})
from pandas_profiling import ProfileReport
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import SMOTE
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
df_train.head()
df_test.head()
profile = ProfileReport(df_train)
profile.to_notebook_iframe()
sns.distplot(df_train["Age"])
plt.xlabel("Idade")
plt.ylabel("Função de densidade de probabilidade")
sns.boxplot(data=df_train["Age"])
sns.distplot(df_train["Fare"])
plt.xlabel("Tarifa do passageiro")
plt.ylabel("Função de densidade de probabilidade")
sns.boxplot(data=df_train["Fare"])
sns.distplot(df_train["SibSp"])
sns.boxplot(data=df_train["SibSp"])
gender = df_train[["Sex", "Survived", "Age"]].groupby(["Sex","Survived"]).count()
gender
plt.figure(figsize=(10,6))
gender.plot.barh()
plt.xlabel("Número de ocorrências (unidade)")
plt.ylabel("Gênero")
sns.catplot("Survived", col="Pclass", kind="count", data=df_train)
sns.catplot("Survived", col="Sex", kind="count", data=df_train)
sns.catplot("Survived", col="Embarked", kind="count", data=df_train)
df_train.duplicated().sum()
df_train.isna().sum()
df_train.isna().sum()/df_train["PassengerId"].count()
df_train["Age"].interpolate(method="pad",inplace=True)
df_train["Cabin"].interpolate(method="pad",inplace=True)
df_train["Embarked"].interpolate(method="pad",inplace=True)
df_train.isna().sum()
df_train.dropna(inplace=True)
df_train.groupby(by="Survived")["Survived"].count()/df_train["Survived"].count()
for col in df_train.columns:
  print("Valores da coluna {0}: {1}.".format(col,df_train[col].unique()))
df_train[df_train["Fare"] == 0]
df_train = df_train.reset_index(drop=True)
ohe = OneHotEncoder(sparse=True)
df_train = pd.concat([df_train,pd.DataFrame(ohe.fit_transform(df_train["Sex"].values.reshape(-1,1)).toarray(), columns=["sex_m", "sex_f"])], axis=1)
df_train["Cabin_first_letter"] = df_train["Cabin"].apply(lambda x: x[0])
columns_names = [x for x in df_train["Cabin_first_letter"].unique()]
columns_cabin = sorted(columns_names)
df_train = pd.concat([df_train,pd.DataFrame(ohe.fit_transform(df_train["Cabin_first_letter"].values.reshape(-1,1)).toarray(), columns=columns_cabin)], axis=1)
columns_embaked = [x for x in df_train["Embarked"].unique()]
columns_embaked = sorted(columns_embaked)
df_train = pd.concat([df_train,pd.DataFrame(ohe.fit_transform(df_train["Embarked"].values.reshape(-1,1)).toarray(), columns=columns_embaked)], axis=1)
df_train.drop(["Cabin","Embarked", "Sex","Cabin_first_letter", "PassengerId","Name", "Ticket"], axis=1, inplace=True)
df_train.drop(df_train[df_train["Fare"] == 0].index, inplace=True)
df_train.head()
le = LabelEncoder()
df_train["Survived"] = le.fit_transform(df_train["Survived"])
scaler = MinMaxScaler(feature_range=[0,1])
X = df_train.drop("Survived", axis=1)
y = df_train["Survived"]
X = scaler.fit_transform(X)
rfc = RandomForestClassifier()
rfc.fit(X, y)
rfc.feature_importances_
df_train.columns
df_feature_importance = pd.DataFrame({"feature_importance":rfc.feature_importances_*100, "Columns":df_train.columns[1:]})
plt.figure(figsize=(10,8))
sns.barplot(data=df_feature_importance.sort_values(by="feature_importance", ascending=False), y="Columns", x="feature_importance")
plt.ylabel("Feature")
plt.xlabel("Importância (%)")
plt.xlim(0, 100)
X = df_train[["Age", "Fare", "sex_m", "sex_f", "Pclass"]]
y = df_train["Survived"]
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
X_resampled.shape
y_resampled.shape
rfc = RandomForestClassifier()
svm = SVC()
knn = KNeighborsClassifier(n_neighbors=5)
classifiers = [rfc, svm, knn]
df_results = pd.DataFrame()
for clf, clf_name in zip(classifiers, ["Random Forest", "SVM", "k-NN"]):
  clf_results = cross_validate(clf, X_resampled, y_resampled, cv=10, scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"])
  df_temp = pd.DataFrame({"metric":list({x:np.mean(clf_results[x]) for x in clf_results}.keys())[2:], "performance":list({x:np.mean(clf_results[x]) for x in clf_results}.values())[2:]})
  df_temp["classifier"] = clf_name
  df_results = pd.concat([df_results,df_temp])
df_results["metric"].replace("test_accuracy", "Accuracy", inplace=True)
df_results["metric"].replace("test_precision_macro", "Precision macro", inplace=True)
df_results["metric"].replace("test_recall_macro", "Recall macro", inplace=True)
df_results["metric"].replace("test_f1_macro", "F1 macro", inplace=True)
df_results["performance"] = df_results["performance"] * 100
df_results
plt.figure(figsize=(10,6))
sns.barplot(data=df_results, x="metric", y="performance", hue="classifier")
plt.ylabel("Desempenho (%)")
plt.xlabel("Métrica")
plt.title("Comparação entre os classificadores")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, fancybox=True, shadow=True)
plt.ylim(0, 100)
def dataCleaning(dataset):
    dataset["Age"].interpolate(method="pad",inplace=True)
    dataset["Cabin"].interpolate(method="pad",inplace=True)
    dataset["Embarked"].interpolate(method="pad",inplace=True)
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)
    return dataset
def dataPreparation(dataset):
    ohe = OneHotEncoder(sparse=True)
    dataset = pd.concat([dataset,pd.DataFrame(ohe.fit_transform(dataset["Sex"].values.reshape(-1,1)).toarray(), columns=["sex_m", "sex_f"])], axis=1)
    dataset["Cabin_first_letter"] = dataset["Cabin"].apply(lambda x: x[0])
    columns_names = [x for x in dataset["Cabin_first_letter"].unique()]
    columns_cabin = sorted(columns_names)
    dataset = pd.concat([dataset,pd.DataFrame(ohe.fit_transform(dataset["Cabin_first_letter"].values.reshape(-1,1)).toarray(), columns=columns_cabin)], axis=1)
    columns_embaked = [x for x in dataset["Embarked"].unique()]
    columns_embaked = sorted(columns_embaked)
    dataset = pd.concat([dataset,pd.DataFrame(ohe.fit_transform(dataset["Embarked"].values.reshape(-1,1)).toarray(), columns=columns_embaked)], axis=1)
    dataset = dataset.drop(["Cabin","Embarked", "Sex","Cabin_first_letter", "PassengerId","Name", "Ticket"], axis=1)
    dataset = dataset.drop(dataset[dataset["Fare"] == 0].index)
    return dataset
df_test = dataCleaning(df_test)
df_test = dataPreparation(df_test)
df_test.head()
scaler = MinMaxScaler(feature_range=[0,1])
df_test_scaled = scaler.fit_transform(df_test[["Age", "Fare", "sex_m", "sex_f", "Pclass"]])
rfc.fit(X_resampled, y_resampled)
predictions = rfc.predict(df_test_scaled)
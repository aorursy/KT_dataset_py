import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
dfTrain = pd.read_csv("../input/train_data.csv",
          names=[
          "Id", "Age", "Workclass", "Fnlwgt", "Education", "Education-Num", "Marital Status",
          "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
          "Hours per week", "Country", "Income"],
          sep=r'\s*,\s*',
          engine='python',
          na_values="?",
          skiprows=1)
dfTest = pd.read_csv("../input/test_data.csv",
         names=[
         "Id", "Age", "Workclass", "Fnlwgt", "Education", "Education-Num", "Marital Status",
         "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
         "Hours per week", "Country"],
         sep=r'\s*,\s*',
         engine='python',
         na_values="?",
         skiprows=1)
dfTrain = dfTrain.apply(lambda x: x.fillna(x.value_counts().index[0])) # Substituicao pelo mais frequente
# Percentual de missing data por coluna
dfTrain.isna().sum() / dfTrain.shape[0] * 100
dfTest = dfTest.apply(lambda x: x.fillna(x.value_counts().index[0])) # Substituicao pelo mais frequente
# Percentual de missing data por coluna
dfTrain.isna().sum() / dfTrain.shape[0] * 100
dfTrain = dfTrain.drop(["Id"], axis='columns')
dfTrain.shape
dfTest.shape
dfTrain.describe()
strList = ["Workclass", "Education", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Country"]
dfAll = pd.concat([dfTrain[strList], dfTest[strList]]).apply(preprocessing.LabelEncoder().fit_transform)
dfTrain[strList] = dfAll.iloc[:dfTrain.shape[0]]
bkpDfTrain = dfTrain.copy()
bkpDfTest = dfTest.copy()
dfTrain[["Income"]] = dfTrain[["Income"]].apply(preprocessing.LabelEncoder().fit_transform)
dfTest[strList] = dfAll.iloc[dfTrain.shape[0]:]
dfTrain.describe()
dfTrain['Income'].hist()
plt.figure(figsize=(10, 10))
plt.title('Matriz de correlação')
sns.heatmap(dfTrain.corr(), annot=True, linewidths=0.1)
labels = ["Age", "Workclass", "Fnlwgt", "Education", "Education-Num", "Marital Status",
          "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
          "Hours per week", "Country"]
dfTrain[labels] = (dfTrain[labels] - dfTrain[labels].mean()) / dfTrain[labels].std()
dfTest[labels] = (dfTest[labels] - dfTrain[labels].mean()) / dfTrain[labels].std()
dfPlot = pd.melt(dfTrain, id_vars="Income",
                                           var_name="features",
                                           value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="Income", data=dfPlot, split=True, inner="quart")
plt.xticks(rotation=90)
features = ["Age", "Education-Num", "Occupation", "Relationship", "Sex", "Hours per week"]
XTrain = dfTrain[features]
YTrain = bkpDfTrain.Income
XTest = dfTest[features]
rfc = RandomForestClassifier()
gs = GridSearchCV(rfc, {
    'n_estimators': [79, 80, 81, 82, 83],
    'criterion': ['gini', 'entropy'],
    'max_depth': [7, 8, 9, 10, 11],
}, cv=3, n_jobs=-1)
gs.fit(XTrain, YTrain)
gs.best_params_
rfc = RandomForestClassifier(n_estimators=81, criterion='gini', max_depth=9)
cross_val_score(rfc, XTrain, YTrain, cv=10).mean()
svc = SVC()
gs = GridSearchCV(svc, {
    'C': [1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}, cv=3, n_jobs=-1)
gs.fit(XTrain, YTrain)
gs.best_params_
svc = SVC(C=2.0, kernel='rbf', gamma='scale')
cross_val_score(svc, XTrain, YTrain, cv=10).mean()
mlp = MLPClassifier()
gs = GridSearchCV(mlp, {
    'hidden_layer_sizes': [(5,5), (10, 5), (5, 10), (10, 10), (5, 5, 5), (5, 5, 5, 5)],
    'activation': ['tanh', 'relu', 'logistic']
}, cv=3, n_jobs=-1)
gs.fit(XTrain, YTrain)
gs.best_params_
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='tanh')
cross_val_score(mlp, XTrain, YTrain, cv=10).mean()
rfc.fit(XTrain, YTrain)
YTest = rfc.predict(XTest)
pd.DataFrame(YTest, columns = ["income"]).to_csv("Output.csv", index_label="Id")
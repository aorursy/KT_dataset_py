import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
dfTrain = pd.read_csv("../input/database/train_data.csv",
          names=[
          "Id", "Age", "Workclass", "Fnlwgt", "Education", "Education-Num", "Marital Status",
          "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
          "Hours per week", "Country", "Income"],
          sep=r'\s*,\s*',
          engine='python',
          na_values="?",
          skiprows=1)
dfTest = pd.read_csv("../input/database/test_data.csv",
         names=[
         "Id", "Age", "Workclass", "Fnlwgt", "Education", "Education-Num", "Marital Status",
         "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
         "Hours per week", "Country"],
         sep=r'\s*,\s*',
         engine='python',
         na_values="?",
         skiprows=1)
# dfTrain = dfTrain.dropna() # Retirada de NA
dfTrain = dfTrain.apply(lambda x: x.fillna(x.value_counts().index[0])) # Substituicao pelo mais frequente
# Percentual de missing data por coluna
dfTrain.isna().sum() / dfTrain.shape[0] * 100
# dfTest = dfTest.dropna() # Retirada de NA
dfTest = dfTest.apply(lambda x: x.fillna(x.value_counts().index[0])) # Substituicao pelo mais frequente
# Percentual de missing data por coluna
dfTrain.isna().sum() / dfTrain.shape[0] * 100
dfTrain = dfTrain.drop(["Id"], axis='columns')
dfTrain.shape
dfTrain.describe()
strList = ["Workclass", "Education", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Country"]
dfAll = pd.concat([dfTrain[strList], dfTest[strList]]).apply(preprocessing.LabelEncoder().fit_transform)
dfTrain[strList] = dfAll.iloc[:dfTrain.shape[0]]
bkpDfTrain = dfTrain.copy()
bkpDfTest = dfTest.copy()
dfTrain[["Income"]] = dfTrain[["Income"]].apply(preprocessing.LabelEncoder().fit_transform)
dfTest[strList] = dfAll.iloc[dfTrain.shape[0]:]
dfTrain.describe()
bkpDfTrain
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
#XTrain = bkpDfTrain[features]
#YTrain = bkpDfTrain.Income
XTrain = dfTrain[features]
YTrain = bkpDfTrain.Income
XTest = dfTest[features]
knn = KNeighborsClassifier(n_neighbors=16)
scores = cross_val_score(knn, XTrain, YTrain, cv=10)
scores
knn.fit(XTrain, YTrain)
YTest = knn.predict(XTest)
YTest
pd.DataFrame(YTest, columns = ["income"]).to_csv("Output.csv", index_label="Id")
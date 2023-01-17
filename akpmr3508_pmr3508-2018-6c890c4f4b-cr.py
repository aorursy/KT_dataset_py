import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
dfTrain = pd.read_csv("../input/train.csv",
          sep=r'\s*,\s*',
          engine='python',
          na_values="")
dfTest = pd.read_csv("../input/test.csv",
         sep=r'\s*,\s*',
         engine='python',
         na_values="")
# dfTrain = dfTrain.dropna() # Retirada de NA
dfTrain = dfTrain.apply(lambda x: x.fillna(x.value_counts().index[0])) # Substituicao pelo mais frequente
# dfTest = dfTest.dropna() # Retirada de NA
dfTest = dfTest.apply(lambda x: x.fillna(x.value_counts().index[0])) # Substituicao pelo mais frequente
dfTrain.shape
dfTrain.describe()
dfTrain.select_dtypes(exclude='number').head()
strList = ["Id", "idhogar", "dependency", "edjefe", "edjefa"]
dfAll = pd.concat([dfTrain[strList], dfTest[strList]]).apply(preprocessing.LabelEncoder().fit_transform)
bkpDfTrain = dfTrain.copy()
bkpDfTest = dfTest.copy()
dfTrain[strList] = dfAll.iloc[:dfTrain.shape[0]]
dfTest[strList] = dfAll.iloc[dfTrain.shape[0]:]
dfTrain.describe()
labels = dfTrain.columns
labels = labels.drop("Target")
dfTrain[labels] = (dfTrain[labels] - dfTrain[labels].mean()) / dfTrain[labels].std()
dfTest[labels] = (dfTest[labels] - dfTrain[labels].mean()) / dfTrain[labels].std()
dfTrain.corr()["Target"].sort_values()
features = ["hogar_nin", "r4t1", "SQBhogar_nin", "meaneduc", "cielorazo", "escolari"]
XTrain = dfTrain[features]
YTrain = bkpDfTrain.Target
XTest = dfTest[features]
scores = []
for i in range(1, 200, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores.append(cross_val_score(knn, XTrain, YTrain, cv=10).mean())
plt.plot(range(1, 200, 10), scores)
knn = KNeighborsClassifier(n_neighbors=81) # Melhor resultado
knn.fit(XTrain, YTrain)
YTest = knn.predict(XTest)
YTest
dfSave = pd.DataFrame(data={"Id" : bkpDfTest["Id"], "Target" : YTest})
pd.DataFrame(dfSave[["Id", "Target"]], columns = ["Id", "Target"]).to_csv("Output.csv", index=False)
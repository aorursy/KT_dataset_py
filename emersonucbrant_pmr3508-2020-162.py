import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import seaborn as sns
adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",
        header=0,
        index_col=["Id"],
        names=[
        "Id", "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.shape
adult.head()
adult.isnull().sum()
nAdult = adult.copy()
#Workclass
moda_treino_Workclass = nAdult["Workclass"].mode()
nAdult["Workclass"] = nAdult["Workclass"].fillna(moda_treino_Workclass).astype(str)
#Occupation
moda_treino_Occupation = nAdult["Occupation"].mode()
nAdult["Occupation"] = nAdult["Occupation"].fillna(moda_treino_Occupation).astype(str)
#Country
moda_treino_Country = nAdult["Country"].mode()
nAdult["Country"] = nAdult["Country"].fillna(moda_treino_Country).astype(str)
nAdult.describe()
q1 = nAdult.quantile(0.25)
q3 = nAdult.quantile(0.75)
iqr = q3-q1
((nAdult < (q1 - 1.5 * iqr)) | (nAdult > (q3 + 1.5 * iqr))).sum()
plt.figure(figsize = (18,6))
nAdultCorr = nAdult.apply(preprocessing.LabelEncoder().fit_transform)
sns.heatmap(nAdultCorr.corr(), annot=True)
testAdult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",
        header=0,
        index_col=["Id"],
        names=[
        "Id", "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
nTestAdult = testAdult.copy()
#Workclass
moda_teste_Workclass = nTestAdult["Workclass"].mode()
nTestAdult["Workclass"] = nTestAdult["Workclass"].fillna(moda_teste_Workclass).astype(str)
#Occupation
moda_teste_Occupation = nTestAdult["Occupation"].mode()
nTestAdult["Occupation"] = nTestAdult["Occupation"].fillna(moda_teste_Occupation).astype(str)
#Country
moda_teste_Country = nTestAdult["Country"].mode()
nTestAdult["Country"] = nTestAdult["Country"].fillna(moda_teste_Country).astype(str)
numAdult = nAdult.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
XAdult = numAdult[["Age", "Education-Num", "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week"]]
YAdult = numAdult.Target
XTestAdult = numTestAdult[["Age", "Education-Num", "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week"]]
melhorK = 1
melhorScore = 0
for k in range (1,30):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(XAdult,YAdult)
    scores = cross_val_score(knn, XAdult, YAdult, cv=10)
    print(scores)
    print(scores.mean())
    if scores.mean() > melhorScore:
        melhorScore = scores.mean()
        melhorK = k
print("melhorK = " + str(melhorK) + ", score = " + str(melhorScore))
knn = KNeighborsClassifier(n_neighbors=melhorK)
knn.fit(XAdult,YAdult)
YTestPred = knn.predict(XTestAdult)
submissao_np = np.where(YTestPred == 0, '<=50K', '>50K')
submissao = pd.DataFrame({'income':submissao_np})
submissao.to_csv('submission.csv', index=True, index_label='Id')
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", na_values="?")
df.head()
# Vamos utilizar o .info do pandas para analisar melhor nossos dados
df.info()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_analysis = df.copy()
# df_analysis['income']
df_analysis['income'] = le.fit_transform(df_analysis['income'])
df_analysis['income']
plt.figure(figsize=(10,8))
sns.heatmap(df_analysis.corr(),annot=True);
# remoção das colunas de variáveis 'Id' e 'fnlwgt'
df_analysis = df_analysis.drop(['Id','fnlwgt'], axis = 1)
df_analysis.head()
sns.catplot(x="income", y="hours.per.week", kind="boxen", data=df_analysis);
sns.catplot(x="income", y="age", kind="boxen", data=df_analysis);
sns.catplot(x="income", y="education.num", kind="boxen", data=df_analysis);
sns.catplot(x="income", y="capital.gain", kind="boxen", data=df_analysis);
sns.catplot(x="income", y="capital.loss", kind="boxen", data=df_analysis);
menorque50 = df_analysis[df_analysis['income'] == 0] 
maiorque50 = df_analysis[df_analysis['income'] == 1]
fig, axes = plt.subplots(nrows=1, ncols=2)
menorque50['workclass'].value_counts().plot(kind = 'bar',ax=axes[0],title='<=50K');
maiorque50['workclass'].value_counts().plot(kind = 'bar',ax=axes[1],title='>50K');
fig, axes = plt.subplots(nrows=1, ncols=2)
menorque50['education'].value_counts().plot(kind = 'bar',ax=axes[0],title='<=50K');
maiorque50['education'].value_counts().plot(kind = 'bar',ax=axes[1],title='>50K');
fig, axes = plt.subplots(nrows=1, ncols=2)
menorque50['marital.status'].value_counts().plot(kind = 'bar',ax=axes[0],title='<=50K');
maiorque50['marital.status'].value_counts().plot(kind = 'bar',ax=axes[1],title='>50K');
fig, axes = plt.subplots(nrows=1, ncols=2)
menorque50['occupation'].value_counts().plot(kind = 'bar',ax=axes[0],title='<=50K');
maiorque50['occupation'].value_counts().plot(kind = 'bar',ax=axes[1],title='>50K');
fig, axes = plt.subplots(nrows=1, ncols=2)
menorque50['relationship'].value_counts().plot(kind = 'bar',ax=axes[0],title='<=50K');
maiorque50['relationship'].value_counts().plot(kind = 'bar',ax=axes[1],title='>50K');
fig, axes = plt.subplots(nrows=1, ncols=2)
menorque50['race'].value_counts().plot(kind = 'bar',ax=axes[0],title='<=50K');
maiorque50['race'].value_counts().plot(kind = 'bar',ax=axes[1],title='>50K');
fig, axes = plt.subplots(nrows=1, ncols=2)
menorque50['sex'].value_counts().plot(kind = 'pie',ax=axes[0],title='<=50K');
maiorque50['sex'].value_counts().plot(kind = 'pie',ax=axes[1],title='>50K');
from sklearn.neighbors import KNeighborsClassifier

#Inicialmente,utilizamos um K arbitrário de valor igual a 5.
knn = KNeighborsClassifier(n_neighbors=5)
X_train = df[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Y_train = df.income
from sklearn.model_selection import cross_val_score

scores = cross_val_score(knn, X_train, Y_train, cv=10)
scores
#Cálculo da acurácia
print('Acurácia:',scores.mean())
best_n, best_score = 0, 0
new_scores = []

for n in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=n)
    n_score = np.mean(cross_val_score(knn, X_train, Y_train, cv=10))
    print('KNN:',n,'\t Score:',n_score)
    new_scores.append(n_score)
    if n_score > best_score:
        best_score = n_score
        best_n = n
print('Melhor k:', best_n)
print('Melhor score:', best_score)
knn = KNeighborsClassifier(n_neighbors = best_n)
knn.fit(X_train, Y_train)
test_data = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values="?")
X_test = test_data[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
predictions = knn.predict(X_test)
predictions
submission = pd.DataFrame()
submission[0] = test_data.index
submission[1] = predictions
submission.columns = ['Id','income']
submission.head()
submission.to_csv('submission.csv',index = False)

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans

from sklearn.metrics import jaccard_similarity_score



# importando dados

dataset = pd.read_csv('../input/agrupamento-de-dados-ufv-crp/dados.csv')

# importando dados p/ validação

y_true = pd.read_csv('../input/marketingdatatrain/train.csv',sep=';')



dataset.head()

#y_true.head()
plt.figure(figsize=(12,6))

plt.xlabel("Estado Civil")

plt.ylabel("Quantidade")

plt.title("Distribuição númerica por estado civil")

plt.bar(dataset['marital'].value_counts().keys(), dataset['marital'].value_counts().values,color="#B0E0E6")

plt.tight_layout()

plt.show()
enc = LabelEncoder()

dataset.poutcome = enc.fit_transform(dataset['poutcome'])

dataset.day_of_week = enc.fit_transform(dataset['day_of_week'])

dataset.month = enc.fit_transform(dataset['month'])

dataset.contact = enc.fit_transform(dataset['contact'])

dataset.housing = enc.fit_transform(dataset['housing'])

dataset.loan = enc.fit_transform(dataset['loan'])

dataset.marital = enc.fit_transform(dataset['marital'])

dataset.default = enc.fit_transform(dataset['default'])

dataset.education = enc.fit_transform(dataset['education'])

dataset.job = enc.fit_transform(dataset['job'])

del dataset['emp.var.rate']

del dataset['cons.price.idx']

del dataset['cons.conf.idx']

del dataset['euribor3m']

del dataset['nr.employed']

del dataset['duration']

#del dataset['age']

#del dataset['job']

#del dataset['education']

#del dataset['month']

#del dataset['day_of_week']

#del dataset['contact']
display(dataset.head())
f, ax = plt.subplots(1,2, figsize=(16,8))



colors = ["#FA5858", "#64FE2E", '#66b3ff']

labels ="Não tem Empréstimo", "Empréstimo", "Não se sabe"



plt.title('Aluguel X Crédito', fontsize=16)



dataset["loan"].value_counts().plot.pie(explode=[0,0.15,0], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors,labels=labels, fontsize=12, startangle=25)



ax[0].set_title('Empréstimo', fontsize=16)

ax[0].set_xlabel('% de empréstimos', fontsize=14)



#palette = ["#64FE2E", "#FA5858"]

sns.barplot(x="housing", y="month", hue='default' ,data=dataset, palette='Set2', estimator=lambda x: len(x) / len(dataset) * 100,dodge=True)

ax[1].set_xlabel('(%) aluguel', fontsize=14)

ax[1].set(ylabel="(%) quantidade")

ax[1].set_xticklabels(dataset["loan"].unique(), rotation=0, rotation_mode="anchor")

plt.tight_layout()

plt.show()
kmeans = KMeans(n_clusters=2, random_state=0)

model = kmeans.fit(dataset)
display(model.labels_)
plt.figure(figsize=(12,6))

plt.title('Distribuição rótulos k means')

plt.hist(model.labels_,color='#66b3ff')

plt.show()
print('Tamanho -> ',len(model.labels_))
display(y_true.head())
y_true.y = enc.fit_transform(y_true['y'])
y_true['y'].value_counts()
print('Score -> ',jaccard_similarity_score(y_true.y,model.labels_))
previsao = pd.DataFrame()

previsao["Predicted"] = model.labels_

previsao.to_csv('prediction.csv',index=True,index_label="Id")
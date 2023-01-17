import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
t=pd.read_excel("../input/cours1/cours.xls",index_col=0)

t['Retour enquete'] = t['Retour enquete'].map({'Oui': 1, 'Non': 0})

t['Nombre concerts par an'] = t['Nombre concerts par an'].map({'Plus de 10': 10, '3 a 5': 4, "Aucun":0,'5 a 10':7,'1 a 2':1})

t["Tranche d'age"] = t["Tranche d'age"].map({'20-24 ans':22,'Moins de 15 ans':15, '70-74 ans':72, '15-19 ans':17,'30-34 ans':32, '75 ans ou plus':75, '50-54 ans':52, '55-59 ans':57,'40-44 ans':42, '45-49 ans':47, '65-69 ans':67, '25-29 ans':27, '35-39 ans':37,'60-64 ans':62})

t["Genre"] = t["Genre"].map({"Femme":0,"Homme":1})

t["Region"] = t["Region"].astype('category').cat.codes

t
t=pd.get_dummies(t,columns=["Style musical 1"],prefix='Style', prefix_sep='.')

# t
t.corr(method ='pearson')["Retour enquete"].sort_values(ascending=False)
corr=t.corr(method ='kendall')#["Retour enquete"].sort_values(ascending=False)
f,ax=plt.subplots(figsize=(15,15))

display((sns.heatmap(corr,vmax=.8,square=True)).figure)
corr["Retour enquete"].sort_values(ascending=False)
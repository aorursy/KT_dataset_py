# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_eu2019a=pd.read_excel( "https://www.data.gouv.fr/fr/datasets/r/f0b85156-91c2-4c8a-9b3f-beda48d15ecd")



df_eu2019a



df_eu2019a
suite=[]

for c in df_eu2019a.columns[:19]:

    suite.append((c,"",""))





for i in range(34):

    for c in df_eu2019a.columns[19:26]:

        suite.append(("votes",i+1,c))

        



df_eu2019a1=df_eu2019a.iloc[:,:256]

df_eu2019a1["pad"]= np.zeros_like(df_eu2019a.iloc[:,0])



df_eu2019a1.columns=pd.MultiIndex.from_tuples(suite,names=["","liste","valeur"])

df_eu2019a1.to_csv("europeenes2019.csv")

df_eu2019a1

df_eu2019a1[("votes",34,"% Voix/Exp")]=df_eu2019a1[("votes",34,"Voix")]/df_eu2019a1["Exprimés"]

df_eu2019a1[("votes",34,"% Voix/Ins")]=df_eu2019a1[("votes",34,"Voix")]/df_eu2019a1["Inscrits"]

df_eu2019a1
listes=df_eu2019a1.loc[0,"votes"].unstack().iloc[:,1:-3]

listes.to_csv("listes.csv")

listes




col_b=[('Code du département', '', ''),

       ('Libellé du département', '', ''), ('Code de la commune', '', ''),

       ('Libellé de la commune', '', ''), ('Code du b.vote', '', ''),

       #('Inscrits', '', ''),

 ('% Abs/Ins', '', ''),

('% Blancs/Ins', '', '')]

df_eu2019n=pd.concat(

[df_eu2019a1 [col_b],

df_eu2019a1.loc[:,("votes",slice(None),"% Voix/Ins")]],

    axis=1)

df_eu2019n.set_index(['Code du département','Libellé du département', 'Code de la commune','Libellé de la commune','Code du b.vote'],inplace=True)

df_eu2019a1.to_csv("df_eu2019a1.csv")

df_eu2019n
res=listes.copy()

res["score"]=df_eu2019n.votes.mean().values

res
res[res.score>2].index.values
df_eu2019n=df_eu2019n["votes"]

df_eu2019n

import sklearn.decomposition

pca=sklearn.decomposition.PCA()

plt.figure(figsize=(16,8))

plt.subplot(1,2,1)



val=df_eu2019n.fillna(0).values

#val[0]=np.log(val[0])

val_pca=pca.fit_transform(val)

tetes=np.argmax( df_eu2019a1.loc[:,("votes",slice(None),"% Voix/Ins")].values , axis=1)

plt.scatter(val_pca[:,0],val_pca[:,1],c=tetes)



plt.axis('equal')

plt.subplot(1,2,2)



candidats=df_eu2019a1.loc[0,"votes"].unstack().iloc[:,1:-3]["Libellé Abrégé Liste"].values

n=candidats.shape[0]

indexes_cabdidats=np.linspace(0,n-1,n)



plt.scatter(np.zeros(n),indexes_cabdidats,c=indexes_cabdidats,s=200)

for i in range(n):

    plt.text(0.005,i,candidats[i]+"  "+str(i))

plt.axis("off")







print(val_pca.shape)

val_pca
df_eu2019a1.loc[0,"votes"].unstack().iloc[:,1:-3]["Libellé Abrégé Liste"].values

#[type(c[0]) for c in df_eu2019n.columns.values]



col_n=[c[0] for c in df_eu2019n.columns.values[:3] if isinstance(c[0],str)]

col_n+=list(candidats)

a=0

b=1

plt.figure(figsize=(25,25))

plt.scatter(pca.components_[0],pca.components_[1])

plt.axis('equal')



plt.axis('tight')

for i,n in enumerate(col_n):

    x=pca.components_[a,i-1]

    y=pca.components_[b,i-1]

    plt.xlabel(f"comp 0" )

    plt.ylabel(f"comp 1")

    plt.text(x,y,n)



import itertools

ncomp=5





comb=[(a,b) for a,b in itertools.product(range(ncomp), repeat=2) if  a!=b and a<b]

cols=int(np.sqrt(len(comb)))

#plt.figure(figsize=(25,25*cols/(cols+1)))

fig=plt.figure(figsize=(25,25*ncomp))

for j,ab in enumerate(comb):

    a,b=ab

    if a!=b and a<b:

        ax =plt.subplot(len(comb)+1,1,j+1)

        

        plt.scatter(pca.components_[a],pca.components_[b])

        plt.axis('equal')



        for i,n in enumerate(col_n):

            x=pca.components_[a,i-1]

            y=pca.components_[b,i-1]

            plt.xlabel(f"comp {a}" )

            plt.ylabel(f"comp {b}")

            plt.text(x,y,n)



pca.n_components_
plt.plot(np.cumsum(pca.explained_variance_ratio_))



for r in pca.explained_variance_ratio_*100:

    print(r,end=", ")




kpca=sklearn.decomposition.KernelPCA(kernel ="poly",degree=8 ,fit_inverse_transform=True,n_jobs=-1)

val_kpca=pca.fit_transform(val)

plt.scatter(val_kpca[:,0],val_kpca[:,1],c=tetes)

plt.axis('equal')

val_kpca







mmf=sklearn.decomposition.NMF()

val_mmf=mmf.fit_transform(val)

plt.scatter(val_mmf[:,0],val_mmf[:,1],c=tetes)

val_mmf



ica =sklearn.decomposition.FastICA()

val_ica=ica.fit_transform(val)

plt.scatter(val_ica[:,0],val_ica[:,1],c=tetes)

val_ica
fact =sklearn.decomposition.FactorAnalysis()

val_fact=fact.fit_transform(val)

plt.scatter(val_fact[:,0],val_fact[:,1],c=tetes)

val_ica
lda = sklearn.decomposition.LatentDirichletAllocation(n_components=2,n_jobs =-1)

val_lda=lda.fit_transform(val)

plt.scatter(val_lda[:,0],val_lda[:,1],c=tetes)

val_lda




import sklearn.decomposition

pca=sklearn.decomposition.PCA()

plt.figure(figsize=(16,8))

plt.subplot(1,2,1)



val=df_eu2019n.iloc[:,res[res.score>2].index.values-1].values

#val[0]=np.log(val[0])

val_pca=pca.fit_transform(val)

tetes=np.argmax( df_eu2019a1.loc[:,("votes",slice(None),"% Voix/Ins")].values , axis=1)

plt.scatter(val_pca[:,0],val_pca[:,1],c=tetes)



plt.axis('equal')

plt.subplot(1,2,2)



candidats=res.iloc[res[res.score>2].index.values-1,0].values

n=candidats.shape[0]

indexes_cabdidats=np.linspace(0,n-1,n)



plt.scatter(np.zeros(n),indexes_cabdidats,c=indexes_cabdidats,s=200)

for i in range(n):

    plt.text(0.005,i,candidats[i]+"  "+str(i))

plt.axis("off")







print(val_pca.shape)

val_pca
import itertools

ncomp=pca.n_components_





comb=[(a,b) for a,b in itertools.product(range(ncomp), repeat=2) if  a!=b and a<b]

cols=int(np.sqrt(len(comb)))

#plt.figure(figsize=(25,25*cols/(cols+1)))

fig=plt.figure(figsize=(25,25*ncomp))

for j,ab in enumerate(comb):

    a,b=ab

    if a!=b and a<b:

        plt.subplot(len(comb)+1,1,j+1)

        

        plt.scatter(pca.components_[a],pca.components_[b])



        for i,n in enumerate(candidats):

            if i<=pca.n_components_:

                x=pca.components_[a,i-1]

                y=pca.components_[b,i-1]

                plt.xlabel(f"comp {a}" )

                plt.ylabel(f"comp {b}")

                plt.text(x,y,n)



pca.n_components_
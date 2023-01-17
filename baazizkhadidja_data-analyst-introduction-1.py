##numpy

#



import numpy as np

l= [1,2,3]

np.array(l)

b=[(1,2,2),(4,5,6)]

np.array(b)

np.random.randint(1,8,43)



liste=np.arange(20)

print(liste)

np.array(liste)
np.arange(1,10)

np.random.randint(1,10,9)
np.zeros((5,3))
np.ones((5,2))
np.eye(3)
np.random.randint(1,8,43)


np.arange(20)



a=[1,2,3,4,5,6,1,4,7,1,2,5,8,6,4]



b=np.array(a)

print(b)

c=b.reshape(3,5)

c

a=np.arange(2,50,3)

a=np.array(a)

a.reshape(2,8)
#pandas

import pandas as pd

import numpy as np

#x={}"positif":[{'id':'12', 'content':'je suis content'},{'id':'13', 'content':'je suis heureuse'},{'id':'14', 'content':'je suis jolie'}],"negatif":[{'id':'15', 'content':'je suis fatiguée'},{'id':'16', 'content':'j'ai peure'},{'id':'17', 'content':'je suis pressée'}],"objectif":[{'id':'18', 'content':'normal'},{'id':'19', 'content':'indifférent'}],"mixed":[{'id':'20', 'content':'oui mais non'},{'id':'21', 'content':'je ne sais pas exactement'},{'id':'22', 'content':'il y a le bien et le mal'}]]

#y=['positive','négative''objective','mixed']



#pd.Series(x, index=y)
x=[[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7]]

y=['a','b','c','d','e','f','g']

pd.Series(x,y)
d={'id':'12', 'content':'je suis content'}

pd.Series(d)
from numpy.random import randn

df=pd.DataFrame(randn(3,4), ["Etudiant1", "Etudiant2", "Etudiant3"], ["x1", "x2","x3","x4"])

print(df)

df["new"]=df["x1"]+df["x2"]

df["new"]
df.drop("Etudiant1")



df
df.loc[["Etudiant2"],["x2","x4"]]
import pandas as pd

from numpy.random import randn

x=['id1', 'id2', 'id3', 'id4', ]

y=["age", "poid", "taille", "note"]

z=["admis", "ajourné", "admis", "ajourné", ]

tableau=pd.DataFrame(randn(4,4),x,y)

tableau
tableau
tableau['age']>0

tableau['note']>0
tableau[(tableau['age']<0)&(tableau['note']>0)]
tableau[(tableau['age']<0)|(tableau['note']>0)]
# pour ajouter une colonne au tableau

state=['fr','alg','usa','en']

tableau['s']=state

tableau

id='id1 id2 id3 id4 id5'.split()

id

crt='nb gaz surf km'.split()

crt

tb=pd.DataFrame(randn(5,4),id,crt)

tb
prix=rand(5)

prix

tb['prix']=prix

tb
tb.reset_index()
tb.set_index('prix')
inp=[1,2,3,4]

outp='a b c d'.split()

t=list(zip(inp,outp))

t
#Missing Data 

id=['id1','id2','id3']

df= pd.DataFrame({'A':[np.nan,2,3],'B':[4,np.nan,6],'C':[7,4,9]}, id)

df
df['B'].fillna('non')
t=pd.read_html

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
#train = train.replace(np.nan, 0)
fra = []
for colonna in train.columns:
    if train[colonna].dtype == np.float64:
        df = pd.DataFrame(data = {'target': train.target, colonna : train[colonna]})
        df = df.sort_values(by=colonna, ascending=False)
        minimo =  np.nan
        massimo =  np.nan
        conteggio = np.nan
        lastTarget = np.nan

        lista = []

        for row in df.itertuples():

            if lastTarget != row.target:
                lista.append([lastTarget, conteggio, minimo, massimo])
                minimo = row[2]
                massimo = row[2]
                conteggio = 1
                lastTarget = row.target
            else:
                conteggio += 1
                minimo = min(minimo, row[2])
                massimo = max(massimo, row[2])

        lista.append([lastTarget, conteggio, minimo, massimo])
        fra.append([colonna, len(lista)])

fra.sort(key=lambda fra: fra[1], reverse=False)
colonna = 'v50'
df = pd.DataFrame(data = {'target': train.target, colonna : train[colonna]})
df = df.sort_values(by=colonna, ascending=False)
minimo =  np.nan
massimo =  np.nan
conteggio = np.nan
lastTarget = np.nan

lista = []

for row in df.itertuples():
    
    if lastTarget != row.target:
        lista.append([lastTarget, conteggio, minimo, massimo])
        minimo = row[2]
        massimo = row[2]
        conteggio = 1
        lastTarget = row.target
    else:
        conteggio += 1
        minimo = min(minimo, row[2])
        massimo = max(massimo, row[2])
    
lista.append([lastTarget, conteggio, minimo, massimo])
len(lista)
lista
v = train.groupby(['v38', 'v62', 'v72', 'v129', 'v3', 'v24', 'v30', 'v31'
                 , 'v47', 'v52', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91'
                 , 'v107', 'v110', 'v112', 'v113', 'v125', 'target']).ID.count()
v.unstack().to_csv('output', sep='\t')

#v56, 

y = x[pd.isnull(x[1])].as_matrix()
y.sum(0)
for colonna in train.columns:
    if train[colonna].dtype == np.object:
        print(colonna)
v.unstack()
train = pd.read_csv("../input/train.csv")
train = train.replace(np.nan, 1)
e = []
for c1 in range(len(c64)):
    for c2 in range(len(c64)):
        c = train[c64[c1]].div(train[c64[c2]])
        e.append([c64[c1], c64[c2], np.abs(train.target.corr(c))])
e.sort(key=lambda e: e[2], reverse=True)     
e
f = [x for x in e if x[2] > 0.1]
f.sort(key=lambda f: f[2], reverse=True)
f
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

import umap
df = pd.read_csv("/kaggle/input/single-cell-rna-seq-nestorova2016-mouse-hspc/nestorowa_corrected_log2_transformed_counts.txt", sep=' ',  )

df
y = pd.read_csv("/kaggle/input/single-cell-rna-seq-nestorova2016-mouse-hspc/nestorowa_corrected_population_annotation.txt", sep=' ')

y
y.iloc[:,0].unique()

# Create cell types markers



# First ones - loaded from anotations

df2 = df.join(y)

df2['celltype'].fillna('no_gate', inplace = True)

vec_cell_types_from_annotations = df2['celltype']

vec_cell_types_from_annotations



# Second ones - l

#  Extract some cell types markers from the cells ids: 

l = []# set()

for i in df.index:

    l.append(i[:4])

l = np.array(l)    

for m in np.unique(l):

    print(m,  (l==m).sum() )

vec_cell_types_from_dataframeindex = l    
import matplotlib.pyplot as plt

import seaborn as sns

import umap



X = df.values



r = umap.UMAP().fit_transform(X.copy())

#plt.scatter(r[:,0],r[:,1],c = adata.obs['cell_types'].values )

plt.figure(figsize = (15,7) )

sns.scatterplot(x=r[:,0], y=r[:,1], hue = vec_cell_types_from_dataframeindex  )

plt.show()



plt.figure(figsize = (15,7) )

sns.scatterplot(x=r[:,0], y=r[:,1], hue = vec_cell_types_from_annotations  )
list_cell_types_and_gens_markers = ["|HSCs | Procr |", 

"|Erythroids | Gata1, Klf1, Epor, Gypa, Hba-a2, Hba-a1, Spi1 |",

"|Neutrophils | Elane, Cebpe, Ctsg, Mpo, Gfi1 |",

    "|Monocytes | Irf8, Csf1r, Ctsg, Mpo |",

"|Basophils | Mcpt8, Prss34 |",

"|Lymphoid| Dntt, Il7r|",

"|Mast cells| Cma1, Gzmb, Kit|",#  CD117/C-Kit |"

"|Megakaryocytes | Itga2b, Pbx1, Sdpr, Vwf |"]



for s0 in list_cell_types_and_gens_markers:

    fig = plt.figure(figsize = (16,5) )

    list_gens = [''] + s0.split('|')[2].split(',')

    i = 0

    for gen in list_gens :

    #for gen in list_gen:

        i += 1

        

        if i == 1:

            fig.add_subplot(1, len(list_gens) , i) 

            sns.scatterplot(x=r[:,0], y=r[:,1], hue = vec_cell_types_from_annotations  )

            continue 

            

        gen = gen.replace(' ','')

        #print(gen)

        #fig = plt.figure(figsize = (7,2) )

        fig.add_subplot(1, len(list_gens) , i) 

        plt.scatter(r[:,0],r[:,1],c = df[gen]) # 'Cd19'] )

        #sns.scatterplot(x=r[:,0], y=r[:,1], hue = adata.obs['cell_types'].values )

        #fig.colorbar()# line, ax=axs[1])

        #plt.title(gen)#  +' ' + s0.split('|')[1])

        plt.title(s0.split('|')[1] + ' ' + gen ) #  +' ' + 

        plt.colorbar()# line, ax=axs[1])

        #plt.show()
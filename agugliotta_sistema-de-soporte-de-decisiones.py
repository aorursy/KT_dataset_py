# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs # load make_blobs to simulate data
from sklearn import decomposition # load decomposition to do PCA analysis with sklearn
chocolate_data = pd.read_csv("../input/flavors_of_cacao.csv")
original_col = chocolate_data.columns
new_col = ['Company', 'Species', 'REF', 'ReviewDate', 'CocoaPercent','CompanyLocation', 'Rating', 'BeanType', 'Country']
chocolate_data =chocolate_data.rename(columns=dict(zip(original_col, 
new_col)))
chocolate_data.head()
#Remove % sign from CocoaPercent column 
chocolate_data['CocoaPercent'] = chocolate_data['CocoaPercent'].str.replace('%','').astype(float)/100
chocolate_data.head()
sb.distplot(chocolate_data['Rating'],kde = False)
plt.show()
sb.distplot(chocolate_data['REF'],kde = False)
plt.show()
sb.distplot(chocolate_data['CocoaPercent'],kde = False)
plt.show()
ax = plt.axes()
ax.scatter(chocolate_data['CocoaPercent'], chocolate_data['Rating'])
ax.set(xlabel='Cocoa Strength',
       ylabel='Rating',
       title='Cocoa Strength vs Rating')
ax = plt.axes()
ax.scatter(chocolate_data['ReviewDate'], chocolate_data['Rating'])
ax.set(xlabel='ReviewDate',
       ylabel='Rating',
       title='Country vs Rating')
bc = plt.axes()

Companyfreq=chocolate_data['Company'].value_counts()
x=[] #init empty lists
y=[]
for i in range (0,5):
    x.append(Companyfreq.axes[0][i])
    y.append(Companyfreq[i])
    
bc.bar(x,y)
mean_by_country = chocolate_data.groupby(["CompanyLocation"])['Rating'].mean()
mean_sorted = mean_by_country.sort_values(ascending=False)
top_bottom_5 = pd.concat([mean_sorted[:5], mean_sorted[-5:]])
top_bottom_5.plot('barh')
unsatisfactory = chocolate_data[chocolate_data['Rating'] < 3.0] 
satisfactory = chocolate_data[(chocolate_data['Rating'] >= 3.0) & (chocolate_data['Rating'] < 4.0)] 
pre_elite = chocolate_data[chocolate_data['Rating'] >= 4.0] 
label_names=['Insatisfactorio','Satisfactorio','Premium']

sizes = [unsatisfactory.shape[0],satisfactory.shape[0],pre_elite.shape[0]]
explode = (0.05,0.05,0.05)
plt.pie(sizes,labels=label_names,explode=explode,autopct='%1.1f%%',pctdistance=0.85
        ,startangle=90,shadow=True)
fig=plt.gcf()
my_circle=plt.Circle((0,0),0.7,color='white') #white center
fig.gca().add_artist(my_circle)
plt.axis('equal')
plt.tight_layout()
plt.show()
X1, Y1 = make_blobs(n_features=10, 
         n_samples=100,
         centers=4, random_state=4,
         cluster_std=2)
print(X1.shape)
pca = decomposition.PCA(n_components=4)
pc = pca.fit_transform(X1)
pc_df = pd.DataFrame(data = pc , 
        columns = ['PC1', 'PC2','PC3','PC4'])
pc_df['Cluster'] = Y1
pc_df.head()
print(pca.explained_variance_ratio_)
print(str(round((pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100,2))+"%")
df = pd.DataFrame({'var':pca.explained_variance_ratio_,
             'PC':['PC1','PC2','PC3','PC4']})
sb.barplot(x='PC',y="var", 
           data=df, color="c");
sb.lmplot( x="PC1", y="PC2",
  data=pc_df, 
  fit_reg=False, 
  hue='Cluster', # color by cluster
  legend=True,
  scatter_kws={"s": 80}) # specify the point size
import datetime
import random

#Definición de constantes
geneSet = "abcdefghijklmnñopqrstuvwxyzABCDEFGHIJKLMNÑOPQRSTUVWXYZ "
target = "Hola Mundo"

#Inicialización de variables
random.seed(2)
startTime = datetime.datetime.now()

def generate_parent(lenght):
    """
        Define muestra aleatoria para que sea nuestro padre
        Ejemplo: random.sample(geneSet, 5) == > ['o', 'S', 'D', 'B', 'L']

    """
    genes = []
    while len(genes) < lenght:
        sampleSize = min(lenght - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sampleSize))
    return ''.join(genes)

def get_fitness(guess):
    """
        Función de aptitud donde sumamos 1 si nuestra muestra aleatoria
        coincide en lugar y caracter de nuestro target
        Ejemplo: 
        zip(target, "Ho iLEPHIZ") sería
        [('H', 'H'),
         ('o', 'o'),
         ('l', ' '),
         ('a', 'i'),
         (' ', 'L'),
         ('M', 'E'),
         ('u', 'P'),
         ('n', 'H'),
         ('d', 'I'),
         ('o', 'Z')]
         
         al recorrerlo  la primera iteración expected = H y actual = H por lo tanto suma 1
         .
         .
         .
         en la ultima iteración expected = o y actual = Z por lo tanto suma 0
         resultado es 2 por las 2 primeras iteraciones
    """
    return sum(1 for expected, actual in zip(target, guess) if expected == actual)

def mutate(parent):
    """
        Creamos una muestra aletoria y la vamos a agregar
    """
    index = random.randrange(0, len(parent))
    childGenes = list(parent)
    newGene, alternate = random.sample(geneSet, 2)
    childGenes[index] = alternate if newGene == childGenes[index] else newGene
    return ''.join(childGenes)

def display(guess):
    timeDiff =  datetime.datetime.now() - startTime
    fitness = get_fitness(guess)
    print('{}\t{}\t{}'.format(guess, fitness, timeDiff))
    
bestParent = generate_parent(len(target)) # Genera String aleatorio de la misma longitud que el target
bestFitness = get_fitness(bestParent)
display(bestParent)

while True:
    child = mutate(bestParent)
    childFitness = get_fitness(child)
    if bestFitness >= childFitness:
        continue
    display(child)
    if childFitness >= len(bestParent):
        break
    bestFitness = childFitness
    bestParent = child
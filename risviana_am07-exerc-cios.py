# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
import math
import statistics
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import scipy.misc
import imageio
from sklearn.neighbors import KNeighborsClassifier
#from sklearn_extra.cluster import KMedoids
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
iris = datasets.load_iris()
x = iris.data[:, :]
y = iris.target
data=pd.DataFrame(iris.data)
df=pd.read_csv("/kaggle/input/iris/Iris.csv")

def kmeans(k, x,data):
    distances = dict()
    group=dict()
    #escolher k centróides aleatórios
    centroids=random.sample(list(x), k)
   
    
    #enquanto algum exemplo troca de grupo
    change=True
    
    
    while(change):
        #calcular distâncias
        for i in range(len(x)):
            for k in range(len(centroids)):
                distances[i,k] = math.sqrt((x[i][0]-centroids[k][0])**2 +(x[i][1]-centroids[k][1])**2 +(x[i][2]- centroids[k][2])**2 + (x[i][3] - centroids[k][3])**2)
        
       
            #para cada exemplo xi, atribuir ao centroide mais próximo
            d_min=min(distances, key=distances.get)[1]
            group[i]=d_min
                
    
        #para cada centroide zi,calcular cada centroide com a média de todos os elementos do grupo
        for j in range(len(centroids)):
            
            #guardar centroides antigos
            old_centroids=centroids
            #calcular novos centroides
            lista=[k for k,v in group.items() if v == j]
            if(len(lista)>0):
                c=data.iloc[lista]
                centroids[j][0]=c[0].mean()
                centroids[j][1]= c[1].mean()
                centroids[j][2]=c[2].mean()
                centroids[j][3]=c[3].mean()
            lista=None
            
        #verificar se o novo grupo é o mesmo que o anterior 
        if(old_centroids==centroids):
            change=False   
        
            
    return group
group=kmeans(3, x,data)
group
def calculate_kmeans(k):
    kmeans = KMeans(n_clusters = k, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    return kmeans.fit_predict(x)
y_kmeans3=calculate_kmeans(3)

def return_species_per_group(y_kmeans3,k):
    lista=pd.DataFrame(y_kmeans3)
    g= lista.loc[lista[0]==k]
    d=df.iloc[g.index]
    g_setosa=lista.iloc[d.loc[d['Species']=='Iris-setosa'].index]
    g_virginica=lista.iloc[d.loc[d['Species']=='Iris-virginica'].index]
    g_versicolor=lista.iloc[d.loc[d['Species']=='Iris-versicolor'].index]
    return g_setosa,g_virginica,g_versicolor
g0_setosa,g0_virginica,g0_versicolor=return_species_per_group(y_kmeans3,0) 
g1_setosa,g1_virginica,g1_versicolor=return_species_per_group(y_kmeans3,1)
g2_setosa,g2_virginica,g2_versicolor=return_species_per_group(y_kmeans3,2)

#Plot Iris_setosa
def plot_setosa(setosa,name,c):
    kwargs = dict(alpha=0.5, bins=10)
    plt.hist(setosa[0], **kwargs, color=c, label=name)
    

plot_setosa(g0_setosa,'g0','g')
plot_setosa(g1_setosa,'g1','r')
plot_setosa(g2_setosa,'g2','b')
plt.gca().set(title='Distribution Histogram of Iris_setosa', ylabel='Distribution')
plt.legend();
#Plot Iris_virginica
def plot_virginica(virginica,name,c):
    kwargs = dict(alpha=0.5, bins=10)
    plt.hist(virginica[0], **kwargs, color=c, label=name)
    

plot_virginica(g0_virginica,'g0','g')
plot_virginica(g1_virginica,'g1','r')
plot_virginica(g2_virginica,'g2','b')
plt.gca().set(title='Distribution Histogram of Iris_virginica', ylabel='Distribution')
plt.legend();




##Plot Iris_versicolor
def plot_versicolor(virginica,name,c):
    kwargs = dict(alpha=0.5, bins=10)
    plt.hist(virginica[0], **kwargs, color=c, label=name)
    

plot_versicolor(g0_versicolor,'g0','g')
plot_versicolor(g1_versicolor,'g1','r')
plot_versicolor(g2_versicolor,'g2','b')
plt.gca().set(title='Distribution Histogram of Iris_versicolor', ylabel='Distribution')
plt.legend();

y_kmeans6=calculate_kmeans(6)
g0_setosa,g0_virginica,g0_versicolor=return_species_per_group(y_kmeans6,0) 
g1_setosa,g1_virginica,g1_versicolor=return_species_per_group(y_kmeans6,1)
g2_setosa,g2_virginica,g2_versicolor=return_species_per_group(y_kmeans6,2)
g3_setosa,g3_virginica,g3_versicolor=return_species_per_group(y_kmeans6,3) 
g4_setosa,g4_virginica,g4_versicolor=return_species_per_group(y_kmeans6,4)
g5_setosa,g5_virginica,g5_versicolor=return_species_per_group(y_kmeans6,5)



plot_setosa(g0_setosa,'g0','g')
plot_setosa(g1_setosa,'g1','r')
plot_setosa(g2_setosa,'g2','b')
plot_setosa(g3_setosa,'g3','purple')
plot_setosa(g4_setosa,'g4','pink')
plot_setosa(g5_setosa,'g5','olive')
plt.gca().set(title='Distribution Histogram of Iris_setosa', ylabel='Distribution')
plt.legend();
plot_virginica(g0_virginica,'g0','g')
plot_virginica(g1_virginica,'g1','r')
plot_virginica(g2_virginica,'g2','b')
plot_virginica(g3_virginica,'g3','purple')
plot_virginica(g4_virginica,'g4','pink')
plot_virginica(g5_virginica,'g5','olive')
plt.gca().set(title='Distribution Histogram of Iris_virginica', ylabel='Distribution')
plt.legend();
plot_versicolor(g0_versicolor,'g0','g')
plot_versicolor(g1_versicolor,'g1','r')
plot_versicolor(g2_versicolor,'g2','b')
plot_versicolor(g3_versicolor,'g3','purple')
plot_versicolor(g4_versicolor,'g4','pink')
plot_versicolor(g5_versicolor,'g5','olive')
plt.gca().set(title='Distribution Histogram of Iris_versicolor', ylabel='Distribution')
plt.legend();
y_kmeans9=calculate_kmeans(9)
g0_setosa,g0_virginica,g0_versicolor=return_species_per_group(y_kmeans9,0) 
g1_setosa,g1_virginica,g1_versicolor=return_species_per_group(y_kmeans9,1)
g2_setosa,g2_virginica,g2_versicolor=return_species_per_group(y_kmeans9,2)
g3_setosa,g3_virginica,g3_versicolor=return_species_per_group(y_kmeans9,3) 
g4_setosa,g4_virginica,g4_versicolor=return_species_per_group(y_kmeans9,4)
g5_setosa,g5_virginica,g5_versicolor=return_species_per_group(y_kmeans9,5)
g6_setosa,g6_virginica,g6_versicolor=return_species_per_group(y_kmeans9,6)
g7_setosa,g7_virginica,g7_versicolor=return_species_per_group(y_kmeans9,7)
g8_setosa,g8_virginica,g8_versicolor=return_species_per_group(y_kmeans9,8)

plot_setosa(g0_setosa,'g0','g')
plot_setosa(g1_setosa,'g1','r')
plot_setosa(g2_setosa,'g2','b')
plot_setosa(g3_setosa,'g3','purple')
plot_setosa(g4_setosa,'g4','pink')
plot_setosa(g5_setosa,'g5','olive')
plot_setosa(g6_setosa,'g6','black')
plot_setosa(g7_setosa,'g7','yellow')
plot_setosa(g8_setosa,'g8','lime')
plt.gca().set(title='Distribution Histogram of Iris_setosa', ylabel='Distribution')
plt.legend();
plot_virginica(g0_virginica,'g0','g')
plot_virginica(g1_virginica,'g1','r')
plot_virginica(g2_virginica,'g2','b')
plot_virginica(g3_virginica,'g3','purple')
plot_virginica(g4_virginica,'g4','pink')
plot_virginica(g5_virginica,'g5','olive')
plot_virginica(g6_virginica,'g6','black')
plot_virginica(g7_virginica,'g7','yellow')
plot_virginica(g8_virginica,'g8','lime')
plt.gca().set(title='Distribution Histogram of Iris_virginica', ylabel='Distribution')
plt.legend();
plot_versicolor(g0_versicolor,'g0','g')
plot_versicolor(g1_versicolor,'g1','r')
plot_versicolor(g2_versicolor,'g2','b')
plot_versicolor(g3_versicolor,'g3','purple')
plot_versicolor(g4_versicolor,'g4','pink')
plot_versicolor(g5_versicolor,'g5','olive')
plot_versicolor(g6_versicolor,'g6','black')
plot_versicolor(g7_versicolor,'g7','yellow')
plot_versicolor(g8_versicolor,'g8','lime')
plt.gca().set(title='Distribution Histogram of Iris_versicolor', ylabel='Distribution')
plt.legend();
def kmeans(k, x,data,df):
    
    group=dict()
    #escolher k centróides aleatórios
    centroids=random.sample(list(x), k)
   
    
    #enquanto algum exemplo troca de grupo
    count=0

    while(count<100):
        
        lista_distance=[]
        #calcular distâncias
        for i in range(len(x)):
            distances = dict()
            for k in range(len(centroids)):
                distances[i,k] = math.sqrt((x[i][0]-centroids[k][0])**2 +(x[i][1]-centroids[k][1])**2 +(x[i][2]- centroids[k][2])**2 + (x[i][3] - centroids[k][3])**2)
                
       
            #para cada exemplo xi, atribuir ao centroide mais próximo
            d_min=min(distances, key=distances.get)[1]
            group[i]=d_min
            lista_distance.append(distances[min(distances, key=distances.get)])
            
        
        df.iloc[count:count+1,:1]=statistics.mean(lista_distance)
        df.iloc[count:count+1,1:]=statistics.stdev(lista_distance)
        
        
        #para cada centroide zi,calcular cada centroide com a média de todos os elementos do grupo
        for j in range(len(centroids)):
            
            #guardar centroides antigos
            old_centroids=centroids
            #calcular novos centroides
            lista=[k for k,v in group.items() if v == j]
            if(len(lista)>0):
                c=data.iloc[lista]
                centroids[j][0]=c[0].mean()
                centroids[j][1]= c[1].mean()
                centroids[j][2]=c[2].mean()
                centroids[j][3]=c[3].mean()
            lista=None
            
        count+=1 
        
            
    return group,df
df=pd.DataFrame(index=range(100))
df['media_3']=0
df['desvio_padrao_3']=0
df['media_6']=0
df['desvio_padrao_6']=0
df['media_9']=0
df['desvio_padrao_9']=0
group,df1=kmeans(3, x,data,df.iloc[:,:2])
group,df2=kmeans(6, x,data,df.iloc[:,2:4])
group,df3=kmeans(9, x,data,df.iloc[:,4:])
df.iloc[:,:2]=df1
df.iloc[:,2:4]=df2
df.iloc[:,4:]=df3
df
def kmedoides(n_clusters, x):
    group=dict()
    #escolher k centróides aleatórios
    sum_distances = dict()
    #enquanto algum exemplo troca de grupo
    cost_decrease=True
    cost=0
    lista_medoides=[]
    count=0
    #enquanto o custo diminui
    while(cost_decrease):
        lista_costs=[]
        medoides=random.sample(list(x), n_clusters)
        lista_medoides.append(medoides)
        #Para cada medóide m, para cada dado o ponto que não é um medóide
        for i in range(len(x)):
            costs=dict()
            for k in range(len(medoides)):
                
                if((medoides[k][0]!=x[i][0]) & (medoides[k][1]!=x[i][1]) & (medoides[k][2]!=x[i][2]) & (medoides[k][3]!=x[i][3])):
                    #calcular custo
                    costs[i,k] =(abs(x[i][0]-medoides[k][0]) + abs(x[i][1]-medoides[k][1]) + abs(x[i][2]- medoides[k][2]) + abs(x[i][3] - medoides[k][3]))
                
            if(len(costs)!=0):
                #para cada exemplo xi, atribuir ao medóide mais próximo
                d_min=min(costs, key=costs.get)[1]
                group[i,d_min]=costs[min(costs, key=costs.get)]
                
        old_cost=cost
        cost=sum(group.values())        
        #recalcular custo
        if(count>0):
            
            if((cost-old_cost)>0):

                cost_decrease=False   
        
        count+=1
    return  group,lista_medoides
group,lista_medoides=kmedoides(3, x)
group
def holdout(x,y,frac):
    X_train, X_test, y_train, y_test = train_test_split(x, y,stratify=y,test_size=frac)
    return X_train, X_test, y_train, y_test
df=pd.read_csv("/kaggle/input/iris/Iris.csv")
x_train, x_test, y_train, y_test=holdout(df.iloc[:,1:5],df['Species'],0.5)
#i.para k = 9, 18, 27, 45 e 72. Execute o k-medóides no conjunto de treino.
x=pd.concat([x_train, pd.DataFrame(y_train)], axis=1).reindex(x_train.index)
group9,lista_medoides9=kmedoides(9, x.values)
group18,lista_medoides18=kmedoides(18, x.values)
group27,lista_medoides27=kmedoides(27, x.values)
group45,lista_medoides45=kmedoides(45, x.values)
group72,lista_medoides72=kmedoides(72, x.values)


def return_data(lista_medoides):
    data=[]
    for l in range(len(lista_medoides)):
            for k in range(len(lista_medoides[l])):
                data.append(lista_medoides[l][k])
    return data
#ii. Remova do conjunto de treino todos os elementos que não são centroide de um grupo
data9=return_data(lista_medoides9)
data9=pd.DataFrame(data9)
data9=data9.drop_duplicates()

data18=return_data(lista_medoides18)
data18=pd.DataFrame(data18)
data18=data18.drop_duplicates()

data27=return_data(lista_medoides27)
data27=pd.DataFrame(data27)
data27=data27.drop_duplicates()

data45=return_data(lista_medoides45)
data45=pd.DataFrame(data45)
data45=data45.drop_duplicates()

data72=return_data(lista_medoides72)
data72=pd.DataFrame(data72)
data72=data72.drop_duplicates()
# iii. Calcule a taxa de acerto POR CLASSE para o conjunto de teste.
neigh9= KNeighborsClassifier(n_neighbors=9)
neigh9.fit(data9.iloc[:,:4], data9.iloc[:,4:5])
y_pred9=neigh9.predict(x_test)

neigh18= KNeighborsClassifier(n_neighbors=18)
neigh18.fit(data18.iloc[:,:4], data18.iloc[:,4:5])
y_pred18=neigh18.predict(x_test)

neigh27= KNeighborsClassifier(n_neighbors=27)
neigh27.fit(data27.iloc[:,:4], data27.iloc[:,4:5])
y_pred27=neigh27.predict(x_test)

neigh45= KNeighborsClassifier(n_neighbors=45)
neigh45.fit(data45.iloc[:,:4], data45.iloc[:,4:5])
y_pred45=neigh45.predict(x_test)

neigh72= KNeighborsClassifier(n_neighbors=72)
neigh72.fit(data72.iloc[:,:4], data72.iloc[:,4:5])
y_pred72=neigh72.predict(x_test)
#k=9
def calculate_score_per_class(y_test,y_pred9):
    score_setosa=0
    score_virginica=0
    score_versicolor=0
    
    for k in range(len(y_test)):
        
        if((y_test.iloc[k:k+1,1:]['Species'][k]== 'Iris-setosa') & (y_pred9.iloc[k:k+1,:][0][k]== 'Iris-setosa')):
                
            score_setosa+=1
            
        elif((y_test.iloc[k:k+1,1:]['Species'][k]=='Iris-virginica') & (y_pred9.iloc[k:k+1,:][0][k]=='Iris-virginica')):
                
            score_virginica+=1
            
        elif((y_test.iloc[k:k+1,1:]['Species'][k]== 'Iris-versicolor') & (y_pred9.iloc[k:k+1,:][0][k]== 'Iris-versicolor')):
                
            score_versicolor+=1
            
    return score_setosa,score_virginica,score_versicolor
               
score_setosa,score_virginica,score_versicolor=calculate_score_per_class(pd.DataFrame(y_test).reset_index(),pd.DataFrame(y_pred9))
score_setosa*100/25,score_virginica*100/25,score_versicolor*100/25
#k=18
score_setosa,score_virginica,score_versicolor=calculate_score_per_class(pd.DataFrame(y_test).reset_index(),pd.DataFrame(y_pred18))
score_setosa*100/25,score_virginica*100/25,score_versicolor*100/25
#27
score_setosa,score_virginica,score_versicolor=calculate_score_per_class(pd.DataFrame(y_test).reset_index(),pd.DataFrame(y_pred27))
score_setosa*100/25,score_virginica*100/25,score_versicolor*100/25
#45
score_setosa,score_virginica,score_versicolor=calculate_score_per_class(pd.DataFrame(y_test).reset_index(),pd.DataFrame(y_pred45))
score_setosa*100/25,score_virginica*100/25,score_versicolor*100/25
#72
score_setosa,score_virginica,score_versicolor=calculate_score_per_class(pd.DataFrame(y_test).reset_index(),pd.DataFrame(y_pred72))
score_setosa*100/25,score_virginica*100/25,score_versicolor*100/25
data_filmes=pd.read_csv("/kaggle/input/maisassistidos/maisAssistidos.csv")
data_filmes.head()
X=data_filmes.iloc[:,1:]
Y=data_filmes['Nome do filme']
# MUDAR ? PARA -1
X=X.replace(to_replace =['?'],value=-1)
X.head()
# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(X)
model.labels_
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)





plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level')
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

image=Image.open('/kaggle/input/dataset-margaridas/Captura de Tela (8).jpg')
image

# Carregando imagem como um array NumPy:
image_vector= imageio.imread("/kaggle/input/dataset-margaridas/Captura de Tela (8).jpg")
data=return_data(image_vector)
image_vector

#k=8
y_kmeans8 = KMeans(n_clusters=8, random_state=0).fit(data)
y_kmeans8.labels_
#k=64
y_kmeans64=KMeans(n_clusters=64, random_state=0).fit(data)
y_kmeans64.labels_
#k=512
y_kmeans512=KMeans(n_clusters=512, random_state=0).fit(data)
y_kmeans512.labels_
#k=8
def converter_centroide_inteiro_prox( y_kmeans):
    lista_cluster=[]
    for k in y_kmeans.cluster_centers_:
        lista_cluster.append([int("%1.f"%k[0]),int("%1.f"%k[1]),int("%1.f"%k[2])])
    return lista_cluster
   
lista_cluster=converter_centroide_inteiro_prox( y_kmeans8)
def reconstruir_vetor_imagem(lista_cluster,y_kmeans):
    new_image=[]
    
    for k in y_kmeans8.labels_:
        new_image.append(lista_cluster[k])
    return new_image
new_image_vector=reconstruir_vetor_imagem(lista_cluster,y_kmeans8)    
def reconstruir_imagem(new_image_vector,image_vector):
    v=[]
    df=pd.DataFrame(new_image_vector)
    i=0
    f=509
    for d in range(len(image_vector)):
        v.append(df.iloc[i:f,:].values)
        i=f
        f=f+509
    
    return plt.imshow(v)
   
reconstruir_imagem(new_image_vector,image_vector)
#k=64
lista_cluster=converter_centroide_inteiro_prox(y_kmeans64)
new_image_vector=reconstruir_vetor_imagem(lista_cluster,y_kmeans64) 
reconstruir_imagem(new_image_vector,image_vector)
#k=512
lista_cluster=converter_centroide_inteiro_prox(y_kmeans512)
new_image_vector=reconstruir_vetor_imagem(lista_cluster,y_kmeans512) 
reconstruir_imagem(new_image_vector,image_vector)

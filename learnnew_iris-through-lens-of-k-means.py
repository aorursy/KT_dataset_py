#Importing the required modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('../input/Iris.csv')
df.tail()       #Will return last 5 records of the dataframe
df.info()
df.describe()
sns.pairplot( df.drop(['Id'],axis=1) ,hue='Species')
plt.legend(prop={'size': 24})
x=np.array(df.drop(['Id','Species'],axis=1))   
y=df['Species']       #No need as K-mean is an unsupervised learning and we'll try to identify the classification through our model 
#Finding the optimum number of clusters for k-means classification using "Elbow method"
from sklearn.cluster import KMeans
wcss = []   #within cluster sum of squares

for i in range(1, 10):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
#Graph to find the suitable k value 
plt.plot(range(1, 10), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')   
plt.show()
#Training our model
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
kmeans.fit(x)
y_kmeans=kmeans.predict(x)
#y_kmeans = kmeans.fit_predict(x)       #perfoms both fit and predict
print(len(y_kmeans))
print(y_kmeans)
#Plot to compare our actual classification with model classification

fig, axs = plt.subplots(ncols=2,figsize=(30,8))
axs[0].scatter(x[y_kmeans == 0, 3], x[y_kmeans == 0, 1], s = 100, c = 'yellow', label = 'Iris-versicolor')
axs[0].scatter(x[y_kmeans == 1, 3], x[y_kmeans == 1, 1], s = 100, c = 'red', label = 'Iris-setosa')
axs[0].scatter(x[y_kmeans == 2, 3], x[y_kmeans == 2, 1], s = 100, c = 'orange', label = 'Iris-virginica')
axs[0].scatter(kmeans.cluster_centers_[:, 3], kmeans.cluster_centers_[:,1], s = 300, c = 'black', label = 'Centroids')
axs[0].set_title("Predictions Classification")
axs[0].legend(prop={'size': 14})

axs[1].scatter(df[df['Species']=='Iris-versicolor']['PetalWidthCm'], df[df['Species']=='Iris-versicolor']['SepalWidthCm'], s = 100, c = 'yellow', label = 'Iris-versicolor')
axs[1].scatter(df[df['Species']=='Iris-setosa']['PetalWidthCm'], df[df['Species']=='Iris-setosa']['SepalWidthCm'], s = 100, c = 'red', label = 'Iris-setosa')
axs[1].scatter(df[df['Species']=='Iris-virginica']['PetalWidthCm'], df[df['Species']=='Iris-virginica']['SepalWidthCm'], s = 100, c = 'orange', label = 'Iris-virginica')
axs[1].set_title("Actual Classification")
axs[1].legend(prop={'size': 14})
df['Actual value']=df['Species']
df['Actual value'].replace(['Iris-versicolor','Iris-setosa','Iris-virginica'],[0,1,2],inplace=True)
df['Model prediction']=list(y_kmeans)
df.head()
uniq=df['Actual value'].nunique()
actual_li=[0]*uniq
model_li=[0]*uniq
true_li=[0]*uniq
true_count= 0
actual_count = 0
for a,p in zip(df['Actual value'],df['Model prediction']):
    actual_li[a]=actual_li[a]+1
    model_li[p]=model_li[p]+1
    true_count=true_count+1
    if a==p:
        true_li[a]=true_li[a]+1
        actual_count=actual_count+1
        
accuracy=(actual_count*100)/true_count
        
print(actual_li)        #Actual classification
print(model_li)       #Model classification
print(true_li)          #Accurate classification
print(true_count)
print(actual_count)
print(accuracy)



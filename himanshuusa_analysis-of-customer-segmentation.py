#import Useful Library
import pandas as pd
import numpy as np

#for making graph
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#for warnings
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
df.columns = ['id','gender','age','income','spending']
df.info()
df.head()
df.describe()
sns.boxplot(x="age", data=df)
sns.boxplot(x="income", data=df)
sns.boxplot(x="spending", data=df)
bins_size = np.arange(18,75,2)
plt.hist(x="age", data=df, bins= bins_size,rwidth=0.9);
plt.title('Distribution of Age');
plt.ylabel('Count');
plt.xlabel('Age');
plt.figure(figsize=(12,3))
bins_size = np.arange(15,134,1)
plt.hist(x="income", data=df, bins= bins_size,rwidth=0.9);
plt.title('Distribution of Annual Income');
plt.ylabel('Count');
plt.xlabel('Annual Income (1000 $)');
plt.figure(figsize=(8,3))
bins_size = np.arange(1,100,3)
plt.hist(x="spending", data=df, bins= bins_size,rwidth=0.9);
plt.title('Distribution of Spending Score');
plt.ylabel('Count');
plt.xlabel('Spending Score');
sns.distplot(df.age);
sns.distplot(df.spending);
sns.distplot(df.income);
df.describe()
df.income.unique()
df.income.nunique()
df.spending.unique()
df.spending.nunique()
df.age.unique()
df.age.nunique()
df.duplicated().sum()
plt.figure(figsize =(8,8))
#ax.set_ylim(4.0, 0)
ax= sns.heatmap(df.corr(),square = True, annot = True,cmap= 'Spectral' )
ax.set_ylim(4.0, 0)
df.info()
plt.figure(1 , figsize = (15 , 7))
n = 0 
for x in ['age' , 'income' , 'spending']:
    for y in ['age' , 'income' , 'spending']:
        n += 1
        plt.subplot(3 , 3 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.regplot(x = x , y = y , data = df)
        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
plt.show()
col = sns.color_palette()[0]
sns.barplot(x="gender", y="spending", data=df, color=col)
sns.barplot(x="gender", y="income", data=df, color=col)
plt.figure(figsize=(15,5))
sns.barplot(x="age", y="spending", data=df, color=col)
plt.figure(figsize=(15,5))
sns.barplot(x="age", y="income", data=df, color=col)
plt.figure(figsize=(20,5))
sns.barplot(x="spending", y="income", data=df, color=col)
plt.figure(figsize=(10,10))
sns.pairplot(df, vars=["age", "income","spending"])
sns.pairplot(df, vars=["age", "income","spending"], hue = "gender")
df.duplicated().sum()
df.drop(columns=['id'], inplace = True)
df.info()
a = df.groupby(['gender', 'age'])
a.first()
plt.figure(figsize = (10 , 5))
sns.scatterplot(x = 'income',y = 'spending', data = df, hue = 'gender', s=200,alpha =0.7)
labels = ['Female', 'Male']
size = df['gender'].value_counts()
colors = ['lightblue', 'pink']
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (5, 5)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Gender', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()
plt.figure(figsize = (5 , 5))
sns.stripplot(df['gender'], df['age'], palette = 'Purples', size = 6)
plt.title('Gender vs Age', fontsize = 10)
plt.show()
x = df.income
y = df.age
z = df.spending
plt.figure(figsize = (10 , 5))
sns.lineplot(x, y, color = 'blue')
sns.lineplot(x, z, color = 'pink')
plt.title('Annual Income vs Age and Spending Score', fontsize = 20)
plt.ylabel('Spending Score');
plt.xlabel('Annual Income (k$)');
plt.show()
x1 = df.iloc[:, [2, 3]].values
x1.shape
from sklearn.cluster import KMeans
wcss_1 = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x1)
    wcss_1.append(km.inertia_)
    
plt.figure(figsize = (9 ,5))
plt.plot(np.arange(1 , 11) , wcss_1 , 'o')
plt.plot(np.arange(1 , 11) , wcss_1 , '-' , alpha = 0.5)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss_1')
plt.show()
kmeans_1=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans_1=kmeans_1.fit_predict(x1)
labels_1 = kmeans_1.labels_
centroids_1 = kmeans_1.cluster_centers_
labels_1
centroids_1
plt.figure(figsize=(15,5))
plt.scatter(x1[y_kmeans_1 == 0, 0], x1[y_kmeans_1 == 0, 1], s = 100, c = 'pink', label = 'miser')
plt.scatter(x1[y_kmeans_1 == 1, 0], x1[y_kmeans_1 == 1, 1], s = 100, c = 'yellow', label = 'general')
plt.scatter(x1[y_kmeans_1 == 2, 0], x1[y_kmeans_1 == 2, 1], s = 100, c = 'cyan', label = 'target')
plt.scatter(x1[y_kmeans_1 == 3, 0], x1[y_kmeans_1 == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')
plt.scatter(x1[y_kmeans_1 == 4, 0], x1[y_kmeans_1 == 4, 1], s = 100, c = 'orange', label = 'careful')
plt.scatter(centroids_1[:,0], centroids_1[:, 1], s = 50, c = 'blue' , label = 'centeroid')

plt.style.use('fivethirtyeight')
plt.title('Clusters: Annual Income vs Spending Score', fontsize = 20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()
x2 = df.iloc[:, [1,3]].values
x2.shape
wcss_2 = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x2)
    wcss_2.append(km.inertia_)
plt.figure(figsize = (9 ,5))
plt.plot(np.arange(1 , 11) , wcss_2 , 'o')
plt.plot(np.arange(1 , 11) , wcss_2 , '-' , alpha = 0.5)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss_2')
plt.show()
kmeans_2=KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans_2=kmeans_2.fit_predict(x2)
y_kmeans_2
labels_2= kmeans_2.labels_
centroids_2 = kmeans_2.cluster_centers_
plt.figure(figsize=(15,5))
plt.title('Clusters: Age vs Spending Score', fontsize = 20)

plt.scatter(x2[y_kmeans_2 == 0, 0], x2[y_kmeans_2 == 0, 1], s = 100, c = 'pink', label = 'Usual Customers' )
plt.scatter(x2[y_kmeans_2 == 1, 0], x2[y_kmeans_2 == 1, 1], s = 100, c = 'orange', label = 'Priority Customers')
plt.scatter(x2[y_kmeans_2 == 2, 0], x2[y_kmeans_2 == 2, 1], s = 100, c = 'lightgreen', label = 'Target Customers(Young)')
plt.scatter(x2[y_kmeans_2 == 3, 0], x2[y_kmeans_2 == 3, 1], s = 100, c = 'red', label = 'Target Customers(Old)')
plt.scatter(centroids_2[:, 0],centroids_2[:, 1], s = 50, c = 'black')

plt.style.use('fivethirtyeight')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid()
plt.show()
x3 = df.iloc[:, [1,2]].values
x3.shape
wcss_3 = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x3)
    wcss_3.append(km.inertia_)
plt.figure(figsize = (9 ,5))
plt.plot(np.arange(1 , 11) , wcss_3 , 'o')
plt.plot(np.arange(1 , 11) , wcss_3 , '-' , alpha = 0.5)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss_3')
plt.show()
kmeans_3=KMeans(n_clusters=6,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans_3=kmeans_3.fit_predict(x3)
labels_3 = kmeans_3.labels_
centroids_3 = kmeans_3.cluster_centers_
plt.figure(figsize=(15,4))
plt.title('Clusters: Age vs Annual Income', fontsize = 20)

plt.scatter(x3[y_kmeans_3 == 0, 0], x3[y_kmeans_3 == 0, 1], s = 100, c = 'pink', label = 'High Earners' )
plt.scatter(x3[y_kmeans_3 == 1, 0], x3[y_kmeans_3 == 1, 1], s = 100, c = 'orange', label = 'Young Low Earners')
plt.scatter(x3[y_kmeans_3 == 2, 0], x3[y_kmeans_3 == 2, 1], s = 100, c = 'lightgreen', label = 'Average Earners')
plt.scatter(x3[y_kmeans_3 == 3, 0], x3[y_kmeans_3 == 3, 1], s = 100, c = 'red', label = 'Old Average Earners')
plt.scatter(x3[y_kmeans_3 == 4, 0], x3[y_kmeans_3 == 4, 1], s = 100, c = 'magenta', label = 'Old Low Earners')
plt.scatter(x3[y_kmeans_3 == 5, 0], x3[y_kmeans_3 == 5, 1], s = 100, c = 'cyan', label = 'Young Average Earners')

plt.scatter(centroids_3[:, 0],centroids_3[:, 1], s = 50, c = 'black')

plt.style.use('fivethirtyeight')
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.legend()
plt.grid()
plt.show()
df_1 = df[['income','spending']]
df_1['Clusters'] = pd.DataFrame(labels_1)
df_1.head()
df_2 = df[['age','spending']]
df_2['Clusters'] = pd.DataFrame(labels_2)
df_2.head()
df_3 = df[['age','income']]
df_3['Clusters'] = pd.DataFrame(labels_3)
df_3.head()
#to csv
df_1.to_csv("Income_and_spendings.csv", index=False)
df_2.to_csv("age_and_spendings.csv", index=False)
df_3.to_csv("age_and_income.csv",index=False)
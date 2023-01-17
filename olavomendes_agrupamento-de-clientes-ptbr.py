import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.cluster import KMeans



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

df.head()
print('Linhas: ', df.shape[0])

print('Colunas: ', df.shape[1])
df.isnull().sum()
df.info()
plt.figure(figsize=(16, 5))



plt.subplot(1, 3, 1)

sns.distplot(df['Annual Income (k$)'], color='g')

plt.title('Renda anual')



plt.subplot(1, 3, 2)

sns.distplot(df['Age'])

plt.title('Idade')



plt.subplot(1, 3, 3)

sns.distplot(df['Spending Score (1-100)'], color='r')

plt.title('Pontuação')



plt.show()
plt.figure(figsize=(5, 5))



plt.pie(df.Gender.value_counts(), colors=['pink', 'lightblue'], explode=[0, 0.1], labels=['Feminino', 'Masculino'], autopct='%.2f%%')

plt.title('Gênero', fontsize=14)

plt.show()
plt.figure(figsize=(10, 6))



sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap='Blues', annot=True, linewidths=2, linecolor='black')



plt.show()
sns.boxplot(df['Gender'], df['Spending Score (1-100)'], palette=['lightblue', 'pink'])

plt.title('Gender vs Spending Score', fontsize = 20)

plt.show()
plt.figure(figsize=(15, 4))



plt.subplot(1, 2, 1)

male_income = df.loc[df.Gender == 'Male', 'Annual Income (k$)']

male_score = df.loc[df.Gender == 'Male', 'Spending Score (1-100)']

sns.kdeplot(male_income, color='b', shade=True)

sns.kdeplot(male_score, color='g', shade=True)

plt.title('Homem')



plt.subplot(1, 2, 2)

female_income = df.loc[df.Gender == 'Female', 'Annual Income (k$)']

female_score = df.loc[df.Gender == 'Female', 'Spending Score (1-100)']

sns.kdeplot(female_income, color='pink', shade=True)

sns.kdeplot(female_score, color='g', shade=True)

plt.title('Mulher')



plt.legend()

plt.tight_layout()

plt.show()
x = df.iloc[:, [2, 4]].values
plt.figure(figsize=(8, 4))

# Within cluster sum o errors

# Inicializa uma lista varia

wcss = []



# Itera entre 1 e 10 clusters

for i in range(1, 11):

    # Cria uma instância do KMeans

    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=500, n_init=10, random_state=42)

    # Treina o modelo

    kmeans.fit(x)

    # Armazena a inertia na lista criada anteriormente

    wcss.append(kmeans.inertia_)

    

plt.plot(range(1, 11), wcss)

plt.xlabel('Quantidade de clusters')

plt.grid(True)

plt.show()
# modelo com 4 clusters

kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=500, n_init=10, random_state=42)

preds = kmeans.fit_predict(x)
plt.figure(figsize=(12, 8))



# Aqui iremos visualizar os resultados dos 4 clusters

plt.scatter(x[preds==0, 0], x[preds==0, 1], s=100, c='red', alpha=0.5, label='Comum')

plt.scatter(x[preds==1, 0], x[preds==1, 1], s=100, c='green', alpha=0.5, label='Ideal')

plt.scatter(x[preds==2, 0], x[preds==2, 1], s=100, c='blue', alpha=0.5, label='Usual')

plt.scatter(x[preds==3, 0], x[preds==3, 1], s=100, c='cyan', alpha=0.5, label='Usual - Mais velho')



# Centróides

plt.title('Clusters da Idade', fontsize=14)

plt.xlabel('Idade')

plt.ylabel('Pontuação')



plt.legend()

plt.show()
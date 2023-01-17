import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import preprocessing

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler  #para normalizar os dados



# Path of the file to read

master = "../input/master.csv"

# Read the file into a variable iris_data

data = pd.read_csv(master,sep=",",decimal=",")

# Print the first 5 rows of the data

data.head()





data.info()
data['country'].unique()
len(data['country'].unique()) #Existem 101 países no dataset
data['age'].unique()
data['generation'].value_counts()   #como tem relação direta com a idade/ano da pesquisa podemos remover
#Taxas médias de suicídio

country_suicide = data.groupby('country').agg('mean')['suicides_no'].sort_values(ascending=False)

x = list(country_suicide.keys())

y = list(country_suicide.values)

plt.figure(figsize=(12,16))

plt.barh(x,y)

plt.show
data.isnull().sum() #Verificando valores nulos
p = sns.countplot(x="sex", data=data)
p = sns.barplot(x='sex', y='suicides_no', hue='age', data=data)
m_pop = data.loc[data.loc[:, 'sex']=='male',:]

f_pop = data.loc[data.loc[:, 'sex']=='female',:]

m_rate = sns.lineplot(x='year', y='suicides_no', data=m_pop)

f_rate = sns.lineplot(x='year', y='suicides_no', data=f_pop)



p = plt.legend(['Masculino', 'Feminino'])
#===== Suicidios por idade



mean_wage_per_age = data.groupby('age')['suicides_no'].mean()

sns.barplot(x = mean_wage_per_age.index, y = mean_wage_per_age.values)

plt.xticks(rotation=90)
df = data.loc[:, 'year':'generation']

df = df.drop(['country-year'], axis=1)



df = df.dropna() 



#===== Transformando os dados para valores numericos



from sklearn import preprocessing



def convertCol(df_value):				#funcao que transforma os sexos de numero para int

	df_value=df_value.replace(",", "")

	df_value=float(df_value)

	return df_value



def alterandoSequencia(df_value):				#funcao que transforma os sexos de numero para int

  if df_value==4:

    df_value=0 #5-14 years

  elif df_value==0:

    df_value=1 #15-24 year

  elif df_value==3:

    df_value=2 #25-34 years

  elif df_value==1:

    df_value=3 #35-54 years

  elif df_value==5:

    df_value=4 #55-74 years

  elif df_value==2:

    df_value=5 #75+ years



  return df_value





df[' gdp_for_year ($) '] = df[' gdp_for_year ($) '].apply(convertCol)	#aplica a transformacao



#===== Mais transformações de valores textuais para numericos



le = preprocessing.LabelEncoder()



le.fit(df['sex'])

df['sex']=le.transform(df['sex'])



le.fit(df['generation'])

df['generation']=le.transform(df['generation'])



le.fit(df['age'])

df['age']=le.transform(df['age'])



df['age']=df['age'].apply(alterandoSequencia)





columns = df.columns
print(columns)
print(df.head())
#======== Distribuicoes



df_new = df[['suicides_no','population',' gdp_for_year ($) ','gdp_per_capita ($)']]



sns.pairplot(df_new, kind="scatter")

plt.show()

sns.jointplot(x=df["suicides_no"], y=df["population"], data=df);


wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i)

    kmeans.fit(df)

    wcss.append(kmeans.inertia_)



plt.rcParams['figure.figsize'] = (15, 5)

plt.plot(range(1, 11), wcss)

plt.title('K-Means Clustering(Método do Cotovelo)', fontsize = 20)

plt.xlabel('Numero de clusters')

plt.ylabel('Soma dos quadrados intra-cluster')

plt.grid()

plt.show()
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=3,

                          random_state=0,

                          batch_size=10)
y_pred = kmeans.fit_predict(df)
plt.scatter(df[" gdp_for_year ($) "],df["suicides_no"], c=y_pred)

plt.title("Clusters")

plt.xlabel("PIB")

plt.ylabel("Número de suicídios")
from sklearn.cluster import AffinityPropagation
clustering = AffinityPropagation().fit(df)
AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,

          damping=0.5, max_iter=200, preference=None, verbose=False)
y_pred = clustering.predict(df)
plt.scatter(df[" gdp_for_year ($) "],df["suicides_no"], c=y_pred)

plt.title("Clusters")

plt.xlabel("PIB")

plt.ylabel("Número de suicídios")
from sklearn.cluster import AgglomerativeClustering

preds = []

for linkage in ('ward', 'average', 'complete', 'single'):

    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=3)

    y_pred = clustering.fit_predict(df)

    preds.append(y_pred)

    


plt.scatter(df[" gdp_for_year ($) "],df["suicides_no"], c=preds[0])

plt.title("Clusters - Ward")

plt.xlabel("PIB")

plt.ylabel("Número de suicídios")



plt.scatter(df[" gdp_for_year ($) "],df["suicides_no"], c=preds[1])

plt.title("Clusters - Average")

plt.xlabel("PIB")

plt.ylabel("Número de suicídios")



plt.scatter(df[" gdp_for_year ($) "],df["suicides_no"], c=preds[2])

plt.title("Clusters - Complete")

plt.xlabel("PIB")

plt.ylabel("Número de suicídios")

plt.scatter(df[" gdp_for_year ($) "],df["suicides_no"], c=preds[3])

plt.title("Clusters - Single")
import scipy.cluster.hierarchy as sch



dendrogram = sch.dendrogram(sch.linkage(df, method = 'ward'))

plt.title('Dendrogam', fontsize = 20)

plt.ylabel('Distância Euclidiana')

plt.show()
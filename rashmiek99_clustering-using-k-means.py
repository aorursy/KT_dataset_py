# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



sns.set(rc={"font.style":"normal",

            "axes.grid":False,

            'figure.figsize':(10.0,10.0)}) 



import nltk

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from wordcloud import WordCloud





from sklearn.cluster import KMeans

# Any results you write to the current directory are saved as output.
retail_df = pd.read_csv('/kaggle/input/online-retail-customer-clustering/OnlineRetail.csv',encoding='ISO_8859-1')

retail_df.head()
retail_df.CustomerID = retail_df.CustomerID.astype(object)

retail_df.InvoiceDate = pd.to_datetime(retail_df.InvoiceDate,format='%d-%m-%Y %H:%M')
retail_df.shape
retail_df.info()
# Droping rows that have missing values



retail_df = retail_df.dropna()



#Dropping negative values

idx = retail_df[retail_df.Quantity < 0].index

retail_df.drop(idx,axis=0,inplace=True)



idx = retail_df[retail_df.UnitPrice < 0].index

retail_df.drop(idx,axis=0,inplace=True)



#Very few outliers, will just drop it for now!

idx = retail_df[retail_df['Quantity'] > 5000].index

retail_df.drop(idx,axis=0,inplace=True)
retail_df.describe()
plt.figure(figsize=(12,5))

sns.countplot(retail_df['Country'],palette= 'Set3')

plt.xticks(rotation=40,ha='right')

plt.title("Country Distribution")

plt.xlabel('Country')

plt.ylabel('Count');
plt.figure(figsize=(8,5))

sns.countplot(retail_df['InvoiceDate'].dt.year,palette= 'Set1')

plt.xticks(rotation=40,ha='right')

plt.title("Year Distribution")

plt.xlabel('Year')

plt.ylabel('Count');
plt.figure(figsize=(8,5))

sns.countplot(retail_df['InvoiceDate'].dt.month_name(),palette= 'Spectral')

plt.xticks(rotation=40,ha='right')

plt.title("Month Distribution")

plt.xlabel('Month')

plt.ylabel('Count');
plt.figure(figsize=(8,5))

sns.countplot(retail_df['InvoiceDate'].dt.day_name(),palette= 'Set1')

plt.xticks(rotation=40,ha='right')

plt.title("Week Distribution")

plt.xlabel('Week')

plt.ylabel('Count');


 #seaborn.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)

    

def wordcloud(text,my_mask=None):

    wordcloud = WordCloud(width=1000,height=1000,max_words=1000,collocations=False,

    min_font_size=10,contour_width=2, mask=my_mask,background_color='white').generate(text)

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()



def tokenize(text):    

    stop_words = set(stopwords.words('english'))

    token =word_tokenize(text)

    

    word_token = []

    for w in token:

        if w not in stop_words and w.isalpha() == True :

            word_token.append(str(w))

    return(str(word_token))
text = tokenize(str(retail_df['Description']).lower())



wordcloud(text)
df1 = pd.DataFrame()

df2 = pd.DataFrame()

df3 = pd.DataFrame()
retail_df['Transaction_Amount'] = retail_df['Quantity'] * retail_df['UnitPrice']

df1['Transaction_Amount'] = retail_df.groupby('CustomerID')['Transaction_Amount'].sum()
df2= retail_df.groupby('CustomerID')['InvoiceNo'].count()

df2 = df2.reset_index()

df2.columns = ['CustomerID','Total_Transaction']

df2.head()
df = pd.merge(df1,df2,on='CustomerID',how='inner')
latest_transaction = retail_df['InvoiceDate'].max()



retail_df['Latest_Transaction'] = latest_transaction - retail_df['InvoiceDate']



df3 = retail_df.groupby('CustomerID')['Latest_Transaction'].min()

df3 = df3.reset_index()



df3['Latest_Transaction'] = df3['Latest_Transaction'].dt.days
df = pd.merge(df,df3,on='CustomerID',how='inner')

df.columns = ['CustomerID','Total_Amount','Total_Transaction','Latest_Transaction']

df.head()
plt.figure(figsize=(10,5))

sns.boxplot(data = df[['Total_Amount','Total_Transaction','Latest_Transaction']],orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
df.describe()
Q1 = df.Total_Amount.quantile(0.05)

Q3 = df.Total_Amount.quantile(0.95)



IQR = Q3 - Q1

df = df[ (df['Total_Amount']  >= Q1 - 1.5 * IQR) & (df['Total_Amount'] <= Q3 + 1.5 * IQR)]



Q1 = df.Total_Transaction.quantile(0.05)

Q3 = df.Total_Transaction.quantile(0.95)



IQR = Q3 - Q1

df = df[ (df['Total_Transaction']  >= Q1 - 1.5 * IQR) & (df['Total_Transaction'] <= Q3 + 1.5 * IQR)]



Q1 = df.Latest_Transaction.quantile(0.05)

Q3 = df.Latest_Transaction.quantile(0.95)



IQR = Q3 - Q1

df = df[ (df['Latest_Transaction']  >= Q1 - 1.5 * IQR) & (df['Latest_Transaction'] <= Q3 + 1.5 * IQR)]



df
plt.figure(figsize=(10,5))

sns.boxplot(data = df[['Total_Amount','Total_Transaction','Latest_Transaction']],orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



scaled = scaler.fit_transform(df[['Total_Amount','Total_Transaction','Latest_Transaction']])



df_scaled = pd.DataFrame(scaled,columns=['Total_Amount','Total_Transaction','Latest_Transaction'])



df_scaled.head()
#Lets fit multiple kmeans

SSE = []



for cluster in range(2,8):

    kmeans = KMeans(n_clusters=cluster,random_state=42)



    kmeans.fit(df_scaled)

    

    centroids = kmeans.cluster_centers_

    pred_clusters = kmeans.predict(df_scaled)

    

    SSE.append(kmeans.inertia_)

    

frame = pd.DataFrame({'Cluster':range(2,8) , 'SSE':SSE})



frame
plt.figure(figsize=(5,5))

plt.plot(frame['Cluster'],frame['SSE'],marker='o')

plt.title('Custers Vs SSE')

plt.xlabel('No of Clusters')

plt.ylabel('Intertia')

plt.show()
kmeans = KMeans(n_clusters=3,random_state=42)

kmeans.fit(df_scaled)

pred = kmeans.predict(df_scaled)
#frame = pd.DataFrame(df_scaled)

df['Cluster'] = kmeans.labels_

df['Cluster'].value_counts()
df.head()
plt.figure(figsize=(10,5))

sns.boxplot(x = df['Cluster'] ,y = df['Total_Amount'],orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)

plt.title("Clusters Vs Total_Amount")

plt.xlabel("Clusters")

plt.ylabel("Total_Amount")

plt.legend();
plt.figure(figsize=(10,5))

sns.boxplot(x = df['Cluster'] ,y = df['Total_Transaction'],orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)

plt.title("Clusters Vs Total_Transaction")

plt.xlabel("Clusters")

plt.ylabel("Total_Transaction")

plt.legend();
plt.figure(figsize=(10,5))

sns.boxplot(x = df['Cluster'] ,y = df['Latest_Transaction'],orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)

plt.title("Clusters Vs Latest_Transaction")

plt.xlabel("Clusters")

plt.ylabel("Latest_Transaction")

plt.legend();
plt.scatter(df['Total_Amount'],df['Total_Transaction'],df['Latest_Transaction'],

                     c=kmeans.labels_, cmap='rainbow');
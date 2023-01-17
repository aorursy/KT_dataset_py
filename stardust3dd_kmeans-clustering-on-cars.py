import pandas as pd

df= pd.read_csv('../input/carsdata/cars.csv')

df.shape # 261 rows, 8 columns

df.isna().sum() # no NaN values in df. However some spaces left blank
# remove all spaces from column names

sn= list(df.columns)

usn= []

for i in sn:

    usn.append(i.strip(' '))

csn= dict(zip(sn, usn))

df.rename(columns= csn, inplace= True)

df.columns # all spaces from column names removed now
df['cubicinches']= pd.to_numeric(df['cubicinches'], errors= 'coerce')

df['weightlbs']= pd.to_numeric(df['weightlbs'], errors= 'coerce')

df.isna().sum() # 5 missing values present in 2 columns. 
from sklearn.preprocessing import LabelEncoder

le_brand= LabelEncoder()

df['encoded_brand']= le_brand.fit_transform(df['brand'])
# there are some empty spaces in df columns that we also need to replace

m_cubi= df['cubicinches'].mean()

m_wt= df['weightlbs'].mean()

df.fillna({'cubicinches': m_cubi, 'weightlbs': m_wt}, inplace= True)

df.isna().sum()
from sklearn.cluster import KMeans

wcss= {} # creating dict to keep track of within cluster sum of squares

for i in range(2, 12):

    km= KMeans(n_clusters= i, init= 'k-means++', max_iter= 500)

    km.fit(df.drop(['brand'], axis= 1))

    wcss[i]= km.inertia_ 
from operator import itemgetter

sorted(wcss.items(), key= itemgetter(1), reverse= 1)
%matplotlib inline

import matplotlib.pyplot as plt



plt.figure(figsize=(10, 6))

plt.plot(list(wcss.keys()), list(wcss.values()))

plt.xlabel('Number of clusters')

plt.ylabel('Within cluster sum of squares')

plt.title('WCSS Chart')

plt.show() # shows the elbow point at clusters= 3
# optimal number of clusters seen to be 3

km3= KMeans(n_clusters= 3, max_iter= 500)

df['pred']= km3.fit_predict(df.drop(['brand'], axis= 1))



df.head()
# let us examine the confusion matrix

from sklearn.metrics import confusion_matrix, classification_report

cm= confusion_matrix(df['pred'], df['encoded_brand'])

cr= classification_report(df['pred'], df['encoded_brand'])

print(cm)

print(cr)
%matplotlib inline

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.title('Original clusters')

plt.scatter(df['mpg'], df['hp'], c= df['encoded_brand'])

plt.show()
plt.figure(figsize=(10,6))

plt.title('Predicted clusters')

plt.scatter(df['mpg'], df['hp'], c= df['pred'])

plt.show()
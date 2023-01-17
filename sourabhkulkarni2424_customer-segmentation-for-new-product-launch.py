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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_excel('../input/customer-survey-dataset/Boaster_Responses_German_Version.xlsx')

df.head()
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()
df['Smartphone_control'] = encoder.fit_transform(df['Smartphone_control'])

df['Ausbildung'] = encoder.fit_transform(df['Ausbildung'])

df['Beschäftigungsstatus'] = encoder.fit_transform(df['Beschäftigungsstatus'])

df['Bevorzugte_Art_des_Frühstücks'] = encoder.fit_transform(df['Bevorzugte_Art_des_Frühstücks'])

df.head()
plt.figure(figsize=(15,10))

sns.heatmap(df.corr(),annot= True,cmap='coolwarm')

plt.show()
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
df['Segment']= model.fit_predict(df.drop('Umfrage_ID',axis=1))

df.head()
sse = []

k_rng = range(1,10)

for k in k_rng:

    model= KMeans(n_clusters=k)

    model.fit(df.drop('Umfrage_ID',axis=1))

    sse.append(model.inertia_)
plt.xlabel('K')

plt.ylabel('Sum of squared error')

plt.plot(k_rng,sse)

plt.show()
model = KMeans(n_clusters=5)



df['Final_Segment']= model.fit_predict(df.drop('Umfrage_ID',axis=1))



df.head()
df.Final_Segment.value_counts()
seg_0 = df[df.Final_Segment == 0]

seg_0
plt.figure(figsize=(15,8))

plt.subplot(2,2,1)

sns.barplot(x= seg_0.Umfrage_ID,y=seg_0.Bewertung )

plt.title('Rating given by customers')

plt.show



plt.subplot(2,2,2)

sns.barplot(x= seg_0.Umfrage_ID,y=seg_0.Akzeptierter_Preis )

plt.title('Accepted Price by customers')

plt.show



plt.subplot(2,2,3)

sns.barplot(x= seg_0.Umfrage_ID,y=seg_0.Alter )

plt.title('Age of customers')

plt.show



plt.subplot(2,2,4)

sns.barplot(x= seg_0.Umfrage_ID,y=seg_0.Menschen_in_der_Familie )

plt.title('Number of People in family')

plt.show





plt.tight_layout(pad=3.0)
seg_1 = df[df.Final_Segment == 1]

seg_1
plt.figure(figsize=(11,8))

plt.subplot(2,2,1)

sns.barplot(x= seg_1.Umfrage_ID,y=seg_1.Bewertung )

plt.title('Rating given by customers')

plt.show



plt.subplot(2,2,2)

sns.barplot(x= seg_1.Umfrage_ID,y=seg_1.Akzeptierter_Preis )

plt.title('Accepted Price by customers')

plt.show



plt.subplot(2,2,3)

sns.barplot(x= seg_1.Umfrage_ID,y=seg_1.Alter )

plt.title('Age of customers')

plt.show



plt.subplot(2,2,4)

sns.barplot(x= seg_1.Umfrage_ID,y=seg_1.Menschen_in_der_Familie )

plt.title('Number of People in family')

plt.show



plt.tight_layout(pad=3.0)
seg_2 = df[df.Final_Segment == 2]

seg_2
plt.figure(figsize=(11,8))

plt.subplot(2,2,1)

sns.barplot(x= seg_2.Umfrage_ID,y=seg_2.Bewertung )

plt.title('Rating given by customers')

plt.show



plt.subplot(2,2,2)

sns.barplot(x= seg_2.Umfrage_ID,y=seg_2.Akzeptierter_Preis )

plt.title('Accepted Price by customers')

plt.show



plt.subplot(2,2,3)

sns.barplot(x= seg_2.Umfrage_ID,y=seg_2.Alter )

plt.title('Age of customers')

plt.show



plt.subplot(2,2,4)

sns.barplot(x= seg_2.Umfrage_ID,y=seg_2.Menschen_in_der_Familie )

plt.title('Number of People in family')

plt.show





plt.tight_layout(pad=3.0)
seg_3 = df[df.Final_Segment == 3]

seg_3
plt.figure(figsize=(13,8))

plt.subplot(2,2,1)

sns.barplot(x= seg_3.Umfrage_ID,y=seg_3.Bewertung )

plt.title('Rating given by customers')

plt.show



plt.subplot(2,2,2)

sns.barplot(x= seg_3.Umfrage_ID,y=seg_3.Akzeptierter_Preis )

plt.title('Accepted Price by customers')

plt.show



plt.subplot(2,2,3)

sns.barplot(x= seg_3.Umfrage_ID,y=seg_3.Alter )

plt.title('Age of customers')

plt.show



plt.subplot(2,2,4)

sns.barplot(x= seg_3.Umfrage_ID,y=seg_3.Menschen_in_der_Familie )

plt.title('Number of People in family')

plt.show





plt.tight_layout(pad=3.0)
seg_4 = df[df.Final_Segment == 4]

seg_4
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)

sns.barplot(x= seg_4.Umfrage_ID,y=seg_4.Bewertung )

plt.title('Rating given by customers')

plt.show



plt.subplot(2,2,2)

sns.barplot(x= seg_4.Umfrage_ID,y=seg_4.Akzeptierter_Preis )

plt.title('Accepted Price by customers')

plt.show



plt.subplot(2,2,3)

sns.barplot(x= seg_4.Umfrage_ID,y=seg_4.Alter )

plt.title('Age of customers')

plt.show



plt.subplot(2,2,4)

sns.barplot(x= seg_4.Umfrage_ID,y=seg_4.Menschen_in_der_Familie )

plt.title('Number of People in family')

plt.show





plt.tight_layout(pad=3.0)
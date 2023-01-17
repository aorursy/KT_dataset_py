import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/COVID19_line_list_data.csv')
df.head(10)
set_01 = df.summary[df.location == 'Shanghai'].iloc[0]

print(set_01)
set_02 = df.summary[df.location == 'Beijing'].iloc[0]

print(set_02)
import nltk

import re
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
# nltk.download("stopwords")
def cleaner(string):

    clean_01 = re.sub(r"[^a-zA-Z]"," ",string)

    clean_02 = clean_01.lower()

    clean_03 = clean_02.split()

    clean_04 = [word for word in clean_03 if not word in stopwords.words("english")]

    

    # Apply Stemming to get the root of the word

    ps = PorterStemmer()

    clean_05 = [ps.stem(word) for word in clean_04]

    return clean_05

    
cleaned_set_01 = cleaner(set_01)

cleaned_set_02 = cleaner(set_02)
!pip install datasketch
from datasketch import MinHash



data1 = cleaned_set_01

data2 = cleaned_set_02



m1, m2 = MinHash(), MinHash()

for d in data1:

    m1.update(d.encode('utf8'))

for d in data2:

    m2.update(d.encode('utf8'))

print("Estimated Jaccard for data1 and data2 is", m1.jaccard(m2))



s1 = set(data1)

s2 = set(data2)

actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))

print("Actual Jaccard for data1 and data2 is", actual_jaccard)
df = pd.read_csv('/kaggle/input/covid_19_data.csv')
df
df.isnull().sum()
df.fillna(method='ffill',inplace=True)
import seaborn as sns
sns.heatmap(df.corr())
df.drop(['SNo','ObservationDate','Last Update'],axis=1, inplace=True)
df
df.dtypes
df['Province/State'] = df['Province/State'].astype('category')

df['Country/Region'] = df['Country/Region'].astype('category')
cat_Data = pd.get_dummies(df.select_dtypes(include='category'))
cat_Data.head(10)
y = df.pop('Deaths')
from sklearn.preprocessing import MinMaxScaler

import numpy as np
sd = MinMaxScaler(feature_range=(0,1))
num_data = pd.DataFrame(sd.fit_transform(df.select_dtypes(exclude='category')), columns=['Confirmed','Deaths'])
num_data
full_data = pd.concat([num_data,cat_Data], axis=1)
full_data.head(10)
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.metrics import mean_squared_error, mean_absolute_error
x_train,x_test,y_train,y_test = train_test_split(full_data,y,test_size = 0.2, random_state = 42)
from sklearn.tree import DecisionTreeRegressor
# Decison Tree is best

DT = DecisionTreeRegressor()
DT.fit(x_train,y_train)
DT.score(x_train,y_train) # Accuracy
folds = KFold(n_splits=10, random_state=42)

score = cross_val_score(DT, x_train, y_train,cv=folds, scoring="neg_mean_squared_error")

accuracy = np.sqrt(-score)

print(accuracy.mean()) # will be on test data 
y_pred = DT.predict(x_test)

error = np.sqrt(mean_squared_error(y_pred=y_pred,y_true=y_test))

print(error)
temp = x_test.iloc[5]

temp2 = pd.DataFrame(temp).T
pred = DT.predict(temp2)
temp2 # input
print(pred[0]) # predicted answer
y_test.iloc[5] # real answer
df = pd.read_csv('/kaggle/input/COVID19_line_list_data.csv')
df.head(5)
male_summary = df.summary[df.gender == 'male']

female_summary = df.summary[df.gender == 'female']
male_summary
female_summary=female_summary.astype('category')
def cleaner_2(df):

    corpus = []

    for i in range(df.shape[0]):

        clean_01 = re.sub(r"[^a-zA-Z]"," ",str(df.iloc[i]))

        clean_02 = clean_01.lower()

        clean_03 = clean_02.split()

        clean_04 = [word for word in clean_03 if not word in stopwords.words("english")]



        # Apply Stemming to get the root of the word

        ps = PorterStemmer()

        clean_05 = [ps.stem(word) for word in clean_04]

        corpus.append(set(clean_05))

    return corpus
set1 = cleaner_2(male_summary)
set2 = cleaner_2(female_summary)
# set1
# set2
from datasketch import MinHash, MinHashLSH



m1 = MinHash(num_perm=128)

m2 = MinHash(num_perm=128)



for se in set1:

    for d in se:

        m1.update(d.encode('utf8'))



for se in set2:

    for d in se:

        m2.update(d.encode('utf8'))        

# Create LSH index

lsh = MinHashLSH(threshold=0.5, num_perm=128)

lsh.insert("m2", m2)

result = lsh.query(m1)
if result:

    print('yes, there are 50% chances that Similarity Exits in Summary of Male and Females using Locality Sensitive Hashing ')

else:

    print('NO, there is not 50% Similarity Exits in Summary of Male and Females using Locality Sensitive Hashing')
# We have already cleaned the data so we will apply Min has Matrix and Jeccard Similarity on It
# set1
# set2
from datasketch import MinHash



data1 = set1

data2 = set2



m1, m2 = MinHash(), MinHash()

for data in data1:

    for d in data:

        m1.update(d.encode('utf8'))

for data in data2:

    for d in data:

        m2.update(d.encode('utf8'))

print("Estimated Jaccard Similarity between data1 and data2 is", m1.jaccard(m2))
df = pd.read_csv('/kaggle/input/time_series_covid_19_deaths.csv')
df.head(5)
countries = set(df['Country/Region'][df['3/23/20'] > 10])
countries
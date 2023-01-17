# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import glob

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        all_files= glob.glob(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import gc

gc.collect()
%matplotlib inline



import matplotlib

import numpy as np

import matplotlib.pyplot as plt
from __future__ import absolute_import, division, print_function, unicode_literals



import tensorflow as tf

from tensorflow import keras



dataset_path= keras.utils.get_file("UserIdToGender_Train.csv", 'https://www.techgig.com/files/DataScienceData/254606/UserIdToGender_Train.csv')
dataset_path2= keras.utils.get_file("UserId_Test.csv", "https://www.techgig.com/files/DataScienceData/254606/UserId_Test.csv")
dataset_2= keras.utils.get_file('Urls_Json_Data.zip', "https://www.techgig.com/files/DataScienceData/254606/Urls_Json_Data.zip", extract= True)
df = pd.read_csv(dataset_2, engine= 'python', error_bad_lines= False, sep= 'delimiter', skipinitialspace= True, compression= 'infer', header= None)

                     

                     
df1= df[0].str.split('":"').apply(pd.Series)

df1.columns= ['id', 'ti', 'title', 'weblink', 'description', 'long_description', 'entities']

df1['weblink']= df1['weblink'].str.split(",", n=1).map(lambda x: x[0]).str.replace('\"', '')



columns_name= ['id', 'weblink', 'others']

columns_name2= ['id', 'gender']

train= pd.read_csv(dataset_path, names= columns_name2, sep= ",", skipinitialspace= True, skiprows= 1)

columns_name2= ['id', 'gender']

test= pd.read_csv(dataset_path2, names= columns_name2, sep= ",", skipinitialspace= True, skiprows= 1)
train.shape
test.shape


df_from_each_file = (pd.read_csv(f, names= columns_name, skipinitialspace= True, index_col=None, skiprows= 1, engine= 'python', error_bad_lines= False, sep= 'delimiter') for f in all_files)

concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
cd= concatenated_df.copy()
cd.shape
df1['number']= df1.ti.map(lambda x: x.split(',')[0]).str.split('[a-z]+').map(lambda x: ''.join(x)).str.replace('\W+', '')
cd['weblink']= cd.id.str.split(',', n= 1).map(lambda x: x[-1])

cd['id']= cd.id.str.split(',', n= 1).map(lambda x: x[0])
cd['id']= cd.id.str.replace('\W+', '')
train.shape

train.gender.value_counts()
train[train.id.isin(cd.id)].index.size
test.shape
test[test.id.isin(cd.id)].index.size
def search(sub):

    seq= df1.number.tolist()

    try:

        for text in seq:

            if sub in text:

                return text

    except:

        pass

    

        
import warnings

warnings.filterwarnings('ignore')
train['id']= train['id'].map(lambda x: str(x)).str.replace('\W+', '')

first= train.astype(str).merge(cd.astype(str), how= 'inner', on= 'id')

second= first.astype(str).merge(df1.astype(str), how= 'outer', on= 'weblink')

second= second[['id_x', 'gender', 'weblink']]

second= second.rename({'id_x': 'id'}, axis= 1)

tt1t= train[~train.id.isin(cd.id)]



tt1t['id_new']= tt1t.id.astype(str).str.replace('\W+', '').map(lambda x: search(x))

dftrain= tt1t.copy()

dftrain['number']= dftrain['id_new'].values

dftrain= df1.merge(dftrain, how= 'inner', on= 'number')

dftrain= dftrain.rename({'id_y': 'id'}, axis= 1)

dftrain= dftrain[['id', 'gender', 'weblink']]

dftrain= pd.concat([dftrain, second])
tt1t.head()
dftrain= dftrain.drop_duplicates(subset=['id', 'weblink'])
dftrain.index.size
test['id']= test['id'].map(lambda x: str(x)).str.replace('\W+', '')
test[test.id.isin(cd.id)].index.size


test['id']= test['id'].map(lambda x: str(x)).str.replace('\W+', '')

first= test.astype(str).merge(cd.astype(str), how= 'inner', on= 'id')

second= first.astype(str).merge(df1.astype(str), how= 'outer', on= 'weblink')

second= second[['id_x', 'title', 'weblink']]

second= second.rename({'id_x': 'id'}, axis= 1)

tt1t= test[~test.id.isin(cd.id)]

tt1t['id_new']= tt1t.id.astype(str).str.replace('\W+', '').map(lambda x: search(x))

dftest= tt1t.copy()

dftest['number']= dftest['id_new'].values

dftest= df1.merge(dftest, how= 'inner', on= 'number')

dftest= dftest.rename({'id_y': 'id'}, axis= 1)

tt1t= tt1t[~tt1t.id.isin(dftest.id)]

dftest= dftest[['id', 'title', 'weblink']]

dftest= pd.concat([dftest, second])

dftest= dftest.drop_duplicates(subset=['id', 'weblink'])
total= pd.concat([dftrain, dftest])
del cd, concatenated_df
labels, uniques = pd.factorize(total.weblink.astype(str))

total['new_label']= labels



total_label = total.groupby('id')['new_label'].apply(np.copy)

df= pd.DataFrame(total_label, columns= ['new_label'])

df= df.reset_index()



df['new_label']= df['new_label'].astype(str).str.replace('[', '').str.replace(']', '').str.replace(',', ' ')

total= total[['id', 'gender']].drop_duplicates(subset= ['id'])

df= df.merge(total, how= 'inner', on= 'id')

df= df.set_index('id')



labels
tt1t['gender'] = (','.join(['M'] * tt1t.index.size)).split(',')
df1= df[(df['gender'].astype(str)== 'M')|(df['gender'].astype(str)== 'F')]

df2= df[~df.index.isin(df1.index)]
df1.shape
df2.shape
del total, total_label

gc.collect()
gl= {'F': 0, 'M': 1}

y = np.array(df1['gender'].astype(str).map(gl))



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import linear_model

import numpy as np

from sklearn.model_selection import train_test_split

import scipy



from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score
count_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')

count_vect.fit((df1['new_label'].astype(str)).unique())

X = count_vect.transform(df1['new_label'].astype(str).values)





Xt = count_vect.transform(df2['new_label'].astype(str).values)



from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score
ml= RandomForestClassifier(n_estimators= 40, n_jobs= -1, random_state= 0)

ml.fit(X, y)

predicted = ml.predict(Xt)
print(list(predicted).count(0))

df2['gender']= predicted

df2= df2.reset_index()

tt1t= tt1t[['id', 'gender']]

df2= df2[['id', 'gender']]



df2= pd.concat([df2, tt1t])

df2= df2.rename({'id': 'userid'}, axis= 1)

df2.to_csv('sk1.csv')

df2= df2.set_index('userid')

df2['gender']= df2.gender.astype(str).str.replace('0', 'F').str.replace('1', 'M')
test.shape
df2.shape
df2.to_csv('sk_test.csv')
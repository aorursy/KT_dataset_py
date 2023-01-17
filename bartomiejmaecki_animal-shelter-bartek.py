import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.display import clear_output



from tensorflow import keras

from tensorflow.keras import layers



tf.keras.backend.set_floatx('float64')

data_df = pd.read_csv("/kaggle/input/animal-shelter-fate/train.csv")

data_df
out_columns = data_df['Outcome Type'].unique()

out_columns.sort()

out_columns
import itertools

from multiprocessing import Pool

import multiprocessing



def parallelize_dataframe(df, func, n_cores=multiprocessing.cpu_count()):

    df_split = np.array_split(df, n_cores)

    pool = Pool(n_cores)

    df = pd.concat(pool.map(func, df_split))

    pool.close()

    pool.join()

    return df

import itertools



def mapAge(row):

  s = str(row['Age upon Outcome']).split()

  if pd.isna(row['Age upon Outcome']):

    s = '8 year'



  days = 2920; # 8 year



  if s[1].startswith("year"):

    days = int(s[0]) * 365

  if s[1].startswith("month"):

    days = int(s[0]) * 30

  if s[1].startswith("week"):

    days = int(s[0]) * 7

  if s[1].startswith("day"):

    days = int(s[0])

  

  if days < 0:

    days *= -1



  return days / (30*365) # normalize



def prepare_data(df):

  df['DateTime']= pd.to_datetime(df['DateTime']) 



  df.sort_values(by='DateTime', inplace=True)

  df['returnShelter'] = df.duplicated(subset=['Animal ID']).astype(int)



  df['hasName'] = df.apply(lambda row : int(pd.notnull(row['Name'])), axis = 1)

  df['ageInDays'] = df.apply(mapAge, axis = 1)

  df['month'] = (pd.DatetimeIndex(df['DateTime']).month).astype(int)



  df['Male'] = df.apply(lambda row : int('Male' in str(row['Sex upon Outcome'])), axis = 1)

  df['Female'] = df.apply(lambda row : int('Female' in str(row['Sex upon Outcome'])), axis = 1)

  df['Spayed'] = df.apply(lambda row : int('Spayed' in str(row['Sex upon Outcome'])), axis = 1)

  df['Neutered'] = df.apply(lambda row : int('Neutered' in str(row['Sex upon Outcome'])), axis = 1)

  df['Intact'] = df.apply(lambda row : int('Intact' in str(row['Sex upon Outcome'])), axis = 1)





  df['Tricolor'] = df.apply(lambda row : int('Tricolor' in str(row['Color'])), axis = 1)

  df['DarkFur'] = df.apply(lambda row : int(('Gray' or 'Black' or 'Brown' or 'Blue') in str(row['Color'])), axis = 1)

  df['MixBreed'] = df.apply(lambda row : int('Mix' in str(row['Breed']) or '/' in str(row['Breed'])), axis = 1)



  breedWords = data_df['Breed'].str.replace('/', ' ').str.split(' ').tolist()

  breedWordsSet = set(itertools.chain.from_iterable(breedWords))



  for val in breedWordsSet: 

        df[val] = df.apply(lambda row : int(val in str(row['Breed'])), axis = 1)



  breedWords = data_df['Breed'].str.replace('/', ' ').str.split(' ').tolist()

  set(itertools.chain.from_iterable(breedWords))





  df = pd.get_dummies(df, prefix='AT', prefix_sep='_', columns=['Animal Type'])

  df = pd.get_dummies(df, prefix='quarter', columns=['month'])



  df = df.drop(['Animal ID', 'Name', 'Age upon Outcome', 'DateTime',

                'Date of Birth', 'Breed', 'Sex upon Outcome', 'Color', 'AT_Livestock'],

               axis=1, errors='ignore')



  return df



# df = prepare_data(data_df.copy().sample(n=1000))

df = parallelize_dataframe(data_df.copy(), prepare_data)

df = df.sample(frac=1).reset_index(drop=True)

pd.set_option('display.max_columns', None)



print(df)
from sklearn.model_selection import train_test_split



shelter_features = df.columns.to_list()

shelter_features.remove('Outcome Type')



# Separate features and labels

shelter_X, shelter_y = df[shelter_features], df['Outcome Type']





# Split data 70%-30% into training set and test set

x_shelter_train, x_shelter_test, y_shelter_train, y_shelter_test = train_test_split(shelter_X, shelter_y,

                                                                                    test_size=0.30,

                                                                                    random_state=1,

                                                                                    stratify=shelter_y)



train_ids = x_shelter_train.pop('ID')

test_ids = x_shelter_test.pop('ID').values



print ('Training Set: %d, Test Set: %d \n' % (x_shelter_train.size, x_shelter_test.size))

from sklearn.linear_model import LogisticRegression





# train a logistic regression model on the training set

model = LogisticRegression(C=0.01, solver='lbfgs', multi_class='auto', max_iter=1000).fit(x_shelter_train, y_shelter_train)

print (model)
import numpy as np

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from sklearn. metrics import classification_report



%matplotlib inline



shelter_prediction = model.predict(x_shelter_test)

print(classification_report(y_shelter_test, shelter_prediction))
from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt



fig, ax = plt.subplots(figsize=(5, 5))



# normalize https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

plot_confusion_matrix(model, x_shelter_test, y_shelter_test,

                                 display_labels=out_columns,

                                 cmap=plt.cm.Blues,

                                 normalize='true',

                                 xticks_rotation=75,

                                 ax=ax)
%%script false --no-raise-error

from datetime import datetime



timestamp = datetime.timestamp(datetime.now())

isodate = datetime.fromtimestamp(timestamp).isoformat()



dftest_orginal = pd.read_csv("drive/My Drive/animal-shelter/test.csv")

dftest = parallelize_dataframe(dftest_orginal.copy(), prepare_data)





dftest_ids = dftest.pop("ID")

shelter_prediction_td = model.predict_proba(dftest).round(2)



out_df = pd.DataFrame(shelter_prediction_td, columns=out_columns)

out_df.insert(0, 'ID', dftest_ids)



out_df.to_csv(f'/kaggle/working/LogisticRegression-{isodate}_full-result.csv', index=False)



out_df

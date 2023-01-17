# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter



import numpy

import pandas

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from itertools import combinations

from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from keras.layers import Flatten, LSTM, Conv1D, Dense, Reshape, Embedding, GlobalAveragePooling1D, Concatenate, Activation, Dropout, Add, Input, BatchNormalization, SpatialDropout1D

from keras.layers.merge import concatenate, add

from keras.models import Model

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from keras.callbacks import TensorBoard

import keras.backend as K

from keras.utils import plot_model

from sklearn import preprocessing

from keras.optimizers import SGD

from keras.callbacks import ReduceLROnPlateau

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD, FastICA

from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection



import matplotlib.pylab as plt

%matplotlib inline  



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv', index_col='PassengerId')

df_test = pd.read_csv('../input/test.csv', index_col='PassengerId')

df_gender_sub = pd.read_csv("../input/gender_submission.csv", index_col='PassengerId')



es = EarlyStopping(monitor='val_acc', mode='auto', verbose=1, patience=4)

# seed = 5

# numpy.random.seed(seed)
train_len = len(df_train)



dataset =  pd.concat(objs=[df_train, df_test], axis=0).reset_index(drop=True)

dataset.head(10)
dataset = dataset.fillna(np.nan)

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]

dataset["Title"] = pd.Series(dataset_title)

dataset["Title"].head()
Ticket = []

for i in list(dataset.Ticket):

    if not i.isdigit() :

        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix

    else:

        Ticket.append("X")

        

dataset["Ticket"] = Ticket

dataset["Ticket"].head()
# # procedure to remove all special characters and change to upper case

# def remove_special(initial):

#     return (''.join(e for e in initial if e.isalnum())).upper()



# dataset['ticket_prefix_v2'] = dataset['ticket_prefix'].apply(remove_special)

#dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



def replace_titles(x):

    title = x['Title']

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady', 'Dona']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title



dataset['Title'] = dataset.apply(replace_titles,axis = 1)

names = 'Mr', 'Mrs', 'Miss', 'Master'

size = list(dict(Counter(dataset['Title'])).values())

fig = plt.figure(figsize=(18,8))

fig.patch.set_facecolor('black')

plt.rcParams['text.color'] = 'white'

my_circle=plt.Circle( (3,3), 0.9, color='black')

plt.pie(size, labels=names)

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.legend(loc='lower left')

plt.show()
dataset.drop(labels = ["Name"], axis = 1, inplace = True)

dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)

dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)

dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)
dataset['Cabin'] = dataset['Cabin'].fillna(value='U')

dataset["Fare"].fillna(dataset.groupby("Pclass")["Fare"].transform("median"), inplace=True)

dataset['Parch'] = dataset['Parch'].fillna(value=0)

dataset['Pclass'] = dataset['Pclass'].fillna(value=0)

dataset['SibSp'] = dataset['SibSp'].fillna(value=0)

dataset['Single'] = dataset['Single'].fillna(value=0)

dataset['SmallF'] = dataset['SmallF'].fillna(value=0)

dataset['MedF'] = dataset['MedF'].fillna(value=0)

dataset['LargeF'] = dataset['LargeF'].fillna(value=0)

#dataset["Fare"] = dataset["Fare"].astype(float)

# # Filling missing value of Age 



# ## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp

# # Index of NaN age rows

# index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)



# for i in index_NaN_age :

#     age_med = dataset["Age"].median()

#     age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()

#     if not np.isnan(age_pred) :

#         dataset['Age'].iloc[i] = age_pred

#     else :

#         dataset['Age'].iloc[i] = age_med

dataset["Age"].fillna(dataset.groupby("Title")["Age"].transform("median"), inplace=True)
def age_bucket(data) :

    if data["Age"] <= 12 :

        return "Age_0-12"

    elif (data["Age"] > 12) & (data["Age"] <= 24 ):

        return "Age_12-24"

    elif (data["Age"] > 24) & (data["Age"] <= 48) :

        return "Age_24-48"

    elif (data["Age"] > 48) & (data["Age"] <= 60) :

        return "Age_48-60"

    elif data["Age"] > 60 :

        return "Age_gt_60"

    

dataset["age_bucket"] = dataset.apply(lambda dataset:age_bucket(dataset), axis=1)
# def age_bucket(data) :

#     if data["Age"] <= 16 :

#         return "Age_0-16"

#     elif (data["Age"] > 16) & (data["Age"] <= 26 ):

#         return "Age_16-26"

#     elif (data["Age"] > 26) & (data["Age"] <= 36) :

#         return "Age_26-36"

#     elif (data["Age"] > 36) & (data["Age"] <= 62) :

#         return "Age_36-62"

#     elif data["Age"] > 60 :

#         return "Age_gt_62"

    

# dataset["age_bucket"] = dataset.apply(lambda dataset:age_bucket(dataset), axis=1)
#embarked_mapping = {"S": 0, "C": 1, "Q": 2}

#dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# # import pandas as pd

testcols = list(dataset.select_dtypes(include=['int64','int']))



pca = PCA(n_components=3)

_X = pca.fit_transform(dataset[testcols])

pca_data = pd.DataFrame(_X, columns=["PCA1", "PCA2", "PCA3"])

dataset[["PCA1", "PCA2", "PCA3"]] = pca_data



fica = FastICA(n_components=3)

_X = fica.fit_transform(dataset[testcols])

fica_data = pd.DataFrame(_X, columns=["FICA1", "FICA2", "FICA3"])

dataset[["FICA1", "FICA2", "FICA3"]] = fica_data



tsvd = TruncatedSVD(n_components=3)

_X = tsvd.fit_transform(dataset[testcols])

tsvd_data = pd.DataFrame(_X, columns=["TSVD1", "TSVD2", "TSVD3"])

dataset[["TSVD1", "TSVD2", "TSVD3"]] = tsvd_data



grp = GaussianRandomProjection(n_components=3)

_X = grp.fit_transform(dataset[testcols])

grp_data = pd.DataFrame(_X, columns=["GRP1", "GRP2", "GRP3"])

dataset[["GRP1", "GRP2", "GRP3"]] = grp_data



srp = SparseRandomProjection(n_components=3)

_X = srp.fit_transform(dataset[testcols])

srp_data = pd.DataFrame(_X, columns=["SRP1", "SRP2", "SRP3"])

dataset[["SRP1", "SRP2", "SRP3"]] = srp_data



dataset["Survived"] = dataset["Survived"].fillna(value=0)

dataset["Survived"] = dataset["Survived"].astype(int)

dataset.drop('Age', axis=1, inplace=True)

#dataset.drop('Ticket', axis=1, inplace=True)

dataset.drop('SibSp', axis=1, inplace=True)

dataset.drop('Parch', axis=1, inplace=True)

#dataset.drop('ticket_prefix', axis=1, inplace=True)

dataset['Cabin'] = dataset['Cabin'].str.replace('\d+', '')

dataset['Cabin'] = dataset['Cabin'].astype(str).str[0]
pandas.set_option('display.max_columns', None)

dataset.head(10)
def add_interactions(dataset):

    # Get feature names

    comb = list(combinations(list(dataset.columns), 2))

    col_names = list(dataset.columns) + ['_'.join(x) for x in comb]

    

    # Find interactions

    poly = PolynomialFeatures(interaction_only=True, include_bias=False)

    dataset = poly.fit_transform(dataset)

    dataset = pd.DataFrame(dataset)

    dataset.columns = col_names

    

    # Remove interactions with 0 values

    no_inter_indexes = [i for i, x in enumerate(list((dataset ==0).all())) if x]

    dataset = dataset.drop(dataset.columns[no_inter_indexes], axis=1)

    

    return dataset
#Answers

y = dataset['Survived']



data = dataset





columnsToEncode = list(data.select_dtypes(include=['category','object']))

#columnsToEncode.append('Parch')

columnsToEncode.append('Pclass')

#columnsToEncode.append('SibSp')

columnsToEncode.append('Fsize')

columnsToEncode.append('Single')

columnsToEncode.append('SmallF')

columnsToEncode.append('MedF')

columnsToEncode.append('LargeF')



data[columnsToEncode] = data[columnsToEncode].fillna(value='none')



quant = (-1, 0, 17, 30, 100, 1000)

label_quants = ['NoInf', 'quart_1', 'quart_2', 'quart_3', 'quart_4']

dataset["Fare_cat"] = pd.cut(dataset.Fare, quant, labels=label_quants)

dataset["Fare_cat"] = dataset["Fare_cat"].fillna('NoInf')

dataset.drop('Fare', axis=1, inplace=True)

columnsToEncode.append('Fare_cat')



y = y.fillna(value=0)



other_cols = [i for i in data.columns if i not in columnsToEncode ]





X = data

X = X.loc[:,~X.columns.duplicated()]

columnsToEncode
#Make all categoricals strings

X[columnsToEncode] = X[columnsToEncode].astype(str)

X[columnsToEncode].head()
other_cols = list(set(other_cols))

other_cols
X[other_cols].head()
scaler = preprocessing.StandardScaler()

#scaler = preprocessing.MinMaxScaler()



###!!!! TODO add scaler if we get more numeric features



#X[other_cols] = scaler.fit_transform(data[other_cols])

from sklearn.model_selection import train_test_split





train = X[:train_len]



y_train = train["Survived"]

X_train = train.drop(labels = ["Survived"],axis = 1)



X_test = X[train_len:]



other_cols = [i for i in X_train.columns if i not in columnsToEncode ]



#Y_train = train["Survived"]



#X_train = train.drop(labels = ["Survived"],axis = 1)

#
X_train.head(10)
X_train[other_cols].head()




#converting data to list format to match the network structure

def preproc2(X_train, X_test):



    input_list_train = []

    input_list_test = []

    

    #the cols to be embedded: rescaling to range [0, # values)

    for c in columnsToEncode:

        #print(c)

        raw_vals = numpy.unique(X_train[c])

        val_map = {}

        for i in range(len(raw_vals)):

            val_map[raw_vals[i]] = i       

        input_list_train.append(X_train[c].map(val_map).values)

        input_list_test.append(X_test[c].map(val_map).fillna(0).values)

     

    other_cols = [c for c in X_train.columns if (not c in columnsToEncode)]

    

    #fransform the individual sets so that we do not have data leakage

    #by doign this before the split!!! v important

    X_train[other_cols] = scaler.fit_transform(X_train[other_cols])

    X_test[other_cols] = scaler.fit_transform(X_test[other_cols])

    

    #Append the rest of the columns

    input_list_train.append(X_train[other_cols].values)

    input_list_test.append(X_test[other_cols].values)

    

    return input_list_train, input_list_test



X_train_list,X_test_list = preproc2( X_train, X_test )



X_train.head()
len(other_cols)
K.clear_session() 



models, inputs = [], []



for categoical_var in columnsToEncode :

    model = Input((1,))

    no_of_unique_cat  = X[categoical_var].nunique()

    embedding_size = min(numpy.ceil((no_of_unique_cat)/2), 128 )

    embedding_size = int(embedding_size)

    output_model = Embedding( no_of_unique_cat+1, embedding_size, name=categoical_var )(model)

    output_model = Reshape(target_shape=(embedding_size,))(output_model)

    inputs.append(model)

    models.append(output_model)



input_numeric = Input((len(other_cols),))

x = Dense(20)(input_numeric)



models.append(x)

inputs.append(input_numeric)



cat = Concatenate(axis=1)(models)

output_layer = Dense(100, activation='relu')(cat)



output_layer = Dense(50, activation='relu')(output_layer)



output_layer = Dense(25, activation='relu')(output_layer)



output_layer = Dense(15, activation='relu')(output_layer)



output_layer = Dense(1, activation='sigmoid')(output_layer)





model = Model(inputs=inputs, outputs=[output_layer])

model.summary()



sgd = SGD(lr = 0.01, momentum = 0.9)

# Define a learning rate decay method:

lr_decay = ReduceLROnPlateau(monitor='loss',

                             patience=1,

                             verbose=0,

                             factor=0.6,

                             min_lr=1e-7)

# lr_decay = ReduceLROnPlateau(monitor='val_loss',

#                              patience=1,

#                              verbose=0,

#                              factor=0.6,

#                              min_lr=0.0001)



#model.compile(loss='binary_crossentropy', optimizer='adam' ,metrics=['mse','acc']) 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras import callbacks

import time

import datetime

log_folder = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')





# Best model callback

bm_callback = callbacks.ModelCheckpoint(

    filepath='bm.h5',

    save_best_only=True,

    save_weights_only=False,

    verbose=1

)





history  =  model.fit(X_train_list, y_train, validation_split=0.1,

                                 epochs = 300, batch_size = 100, verbose=2, callbacks=[bm_callback, lr_decay]) 
from keras.models import load_model

model = load_model('bm.h5', compile=False)
preds = model.predict(X_train_list, batch_size=10000, verbose=1)

from sklearn.metrics import confusion_matrix

preds2 = (preds > 0.5)

confusion_matrix(y_train, preds2)
#Test results

preds = model.predict(X_test_list, batch_size=10000, verbose=1)

from sklearn.metrics import confusion_matrix

preds2 = (preds > 0.5)

confusion_matrix(df_gender_sub.Survived, preds2)
from sklearn.metrics import mean_squared_error

mean_squared_error(df_gender_sub.Survived, preds2)  


# summarize history for accuracy

plt.hist(preds);
submission = pd.DataFrame({

    'PassengerId': dataset[train_len:].index+1,

    'Survived': np.rint(preds[:,0]),

})

submission = submission.astype(int)

submission
for idx, itm in submission.T.iteritems():

    print('{}\t{}\t{}'.format(int(itm[0]), int(itm[1]), df_gender_sub.Survived.iloc[idx]))
submission.sort_values('PassengerId', inplace=True)    

submission.to_csv('submission.csv', index=False)
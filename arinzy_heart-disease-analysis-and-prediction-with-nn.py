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
#Setting the seed value to get consistent results 



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = ""



seed_value= 0



os.environ['PYTHONHASHSEED']=str(seed_value)



import random

random.seed(seed_value)



np.random.seed(seed_value)



import tensorflow as tf



tf.random.set_seed(seed_value)



from keras import backend as K

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,

                                        allow_soft_placement=True) 

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

tf.compat.v1.keras.backend.set_session(sess)
#imports



from sklearn.pipeline import FeatureUnion

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import seaborn as sns

import eli5

from eli5.sklearn import PermutationImportance

from sklearn.compose import ColumnTransformer

from keras import layers

from keras.layers import Input, Dense, Activation

from keras.models import Model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils import plot_model

from keras.layers import Dropout

from keras.regularizers import l2

from keras.regularizers import l1

#np.random.seed(5)



import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
#Lets take a look at the data

df.head()
df.info()
#Good news! we have no NAN values in our dataset

df.isnull().sum()
#Exploring age distribution

plt.figure(figsize=(15,8))

sns.distplot(df.age,color='#86bf91')
labels=['Male','Female']

plt.figure(figsize=(6,6))

plt.pie(df.sex.value_counts(), labels=labels, autopct='%1.1f%%', shadow=True);

#Binning the age data to see how aging affects heart disease

bins = np.linspace(df.age.min(), df.age.max(),8)

bins = bins.astype(int)

df_age = df.copy()

df_age['binned'] = pd.cut(df_age['age'],bins=bins)

df_age.head()
#Lets take a look at the target distribution with respect to ages

import seaborn as sns

plt.figure(figsize=(15,8))

sns.countplot(x='binned',hue = 'target',data = df_age,edgecolor=None)
df_age.binned.value_counts()
bins = [df.age.min(), 35, 55, np.inf]

labels = ['young','middle','older']

df_cat = df.copy()

df_cat['binned'] = pd.cut(df_cat['age'],bins=bins,labels=labels)

df_cat.head()


ax = df_cat.groupby('binned')['cp'].value_counts(normalize=True).unstack('cp').plot(kind='bar',figsize=(15,9),rot=0)

ax.set_xlabel("Binned Age Groups");

ax.set_ylabel("CP Percentages");
ax = df_cat.groupby('cp')['target'].value_counts(normalize=True).unstack('target').plot(kind='bar',figsize=(15,9),rot=0)

ax.legend(['Disease Free','Has Disease'])

ax.set_xlabel('Chest Pain Type');

ax.set_ylabel('Percentages');
df_cat.groupby('target')['cp'].value_counts().unstack('cp').plot(kind='bar',figsize=(15,9),rot=0)
bins = np.linspace(df_cat.chol.min(),df_cat.chol.max(),30)

plt.figure(figsize=(15,8))

sns.distplot(df_cat.chol,bins=bins)


    

bins = [df_cat.chol.min(), 200, 239,np.inf]

labels = ['Normal','Borderline','High']

df_chol = df_cat.copy(deep=True)

df_chol['chol_bin'] = pd.cut(df_cat['chol'],bins = bins,labels = labels)

plt.figure(figsize=(15,8))

sns.countplot(x = 'chol_bin',data=df_chol,hue='target')
ax = df.groupby('ca')['target'].value_counts(normalize=True).unstack('target').plot(kind='bar',figsize=(15,9),rot=0)

ax.legend(['Disease Free','Has Disease'])

ax.set_xlabel('# of major vessels (0-4) colored by flourosopy',fontdict={'fontsize':14});

ax.set_ylabel('Percentages',fontdict={'fontsize':14});
df.ca.value_counts()
ax = df.groupby('thal')['target'].value_counts(normalize=True).unstack('target').plot(kind='bar',figsize=(15,9),rot=0)

ax.legend(['Disease Free','Has Disease'])

ax.set_xlabel('Thalemesia')

ax.set_ylabel = ('Percentages')
df.thal.value_counts()
#First define the limit for normal heart rate limit for each patient according to their age category



df_t = df_cat.copy(deep=True)

df_t.loc[df_t.binned=='young','hr_bin'] = 200

df_t.loc[df_t.binned=='middle','hr_bin'] = 185

df_t.loc[df_t.binned=='older','hr_bin'] = 160

df_t.head()
#Then categorizing the heart rate category as Normal or High in thalach_bin



df_t['thalach_bin'] = np.where(df_t.eval("thalach <= hr_bin "), "Normal", "High")

df_t
#grouping df_t to get the counts of the patients in each group



df_thalach = df_t.groupby(['thalach_bin','target','binned']).count()

df_thalach




#Dividing inital values for each age_binned group by summed up  entries per each age_binned group to get percentages



df_thalach.iloc[[0,3,6,9]]/= df_thalach.iloc[[0,3,6,9]].age.sum()

df_thalach.iloc[[1,4,7,10]]/= df_thalach.iloc[[1,4,7,10]].age.sum()

df_thalach.iloc[[2,5,8,11]]/= df_thalach.iloc[[2,5,8,11]].age.sum()



df_thalach = df_thalach.reset_index(level='target')

df_thalach = df_thalach.reset_index(level='binned')

df_thalach = df_thalach[['age','target','binned']]

df_thalach.rename(columns={'age':'density'},inplace=True)

df_thalach

#df_thalach.reset_index(level='thalach_bin',inplace=True)



df_thalach = df_thalach.reset_index()

plt.figure(figsize=(15,8));

g = sns.FacetGrid(df_thalach, col = 'binned', height=5, aspect=1.2,hue='target');

g.map(sns.barplot,  "thalach_bin", "density",alpha=0.6);

plt.legend();
ax = df.groupby('restecg')['target'].value_counts(normalize=True).unstack('target').plot(kind='bar',figsize=(15,9),rot=0)

ax.legend(['Disease Free','Has Disease']);

ax.set_xlabel('Resting Electrocardiographic Results');

ax.set_ylabel('Percentages');
df.restecg.value_counts()
def trestbps_bin(row):

    

    if row['trestbps'] <= 80:

        value = 'Low'

    

    elif row['trestbps'] > 120:

        value = 'High'

    else:

        value = 'Normal'

        

    return value
df_trestbps = df.copy()

df_trestbps['trestbps_bin'] = df.apply(trestbps_bin,axis=1)





df_trestbps
ax = df_trestbps.groupby('trestbps_bin')['target'].value_counts(normalize=True).unstack('target').plot(kind='bar',figsize=(15,9),rot=0)

ax.legend(['Disease Free','Has Disease'])

ax.set_xlabel('Blood Pressure Bin');

ax.set_ylabel('Percentages');

df_trestbps.trestbps_bin.value_counts()
ax = df.groupby('exang')['target'].value_counts(normalize=True).unstack('target').plot(kind='bar',figsize=(15,9),rot=0)

ax.legend(['Disease Free','Has Disease'])

ax.set_xlabel('Exercise Induced Angina');

ax.set_ylabel('Percentages');

#first seperate categorical and numerical data to apply different transformations



y = df.target

df_transformed = df.copy(deep=True)

numeric_features = ['age','trestbps','chol','thalach','ca','oldpeak']

categorical_features = ['sex','cp','fbs','restecg','exang','slope','thal']



enc = OneHotEncoder(sparse=False,drop='first')

enc.fit(df_transformed[categorical_features])



col_names = enc.get_feature_names(categorical_features)

df_transformed = pd.concat([df_transformed.drop(categorical_features, 1),

          pd.DataFrame(enc.transform(df_transformed[categorical_features]),columns = col_names)], axis=1).reindex()

df_transformed.head()

scaler = StandardScaler()





df_transformed[numeric_features]  = scaler.fit_transform(df_transformed[numeric_features])

df_transformed.head()
#Split the data 

X_train, X_test, y_train, y_test = train_test_split(df_transformed.drop('target',axis=1), df_transformed['target'], test_size = .2, random_state=10)

rf_model = RandomForestClassifier(max_depth=5, random_state=137)

rf_model.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix

def conf_matrix(X_test,y_test,model):

    

    y_pred = model.predict(X_test)

    return confusion_matrix(y_test, y_pred)

    

    
conf_matrix(X_test,y_test,rf_model)
from sklearn.metrics import accuracy_score

y_test_pred = rf_model.predict(X_test)

accuracy_score(y_test, y_test_pred)
from sklearn.model_selection import GridSearchCV

n_estimators = [10, 30, 50, 100]

max_depth = [5, 8, 15]

min_samples_split = [2, 5, 10, 15, 40]

min_samples_leaf = [1, 2, 5, 10] 



hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  

              min_samples_split = min_samples_split, 

             min_samples_leaf = min_samples_leaf)



gridF = GridSearchCV(rf_model, hyperF, cv = 3, verbose = 1, 

                      n_jobs = -1)

bestF = gridF.fit(X_train, y_train)
gridF.best_params_
hp_list = gridF.best_params_



improved_model = RandomForestClassifier(max_depth=hp_list['max_depth'], min_samples_leaf = hp_list['min_samples_leaf'],

                                        min_samples_split = hp_list['min_samples_split'], n_estimators = hp_list['n_estimators'],random_state = 137)
improved_model.fit(X_train, y_train)
conf_matrix(X_test,y_test,improved_model)
y_test_pred = improved_model.predict(X_test)

accuracy_score(y_test, y_test_pred)


def model_nn(input_shape):

    



    # Define the input placeholder as a tensor with shape input_shape

    X_input = Input(input_shape)



  

    X = Dense(512,kernel_regularizer=l2(0.01),kernel_initializer = 'random_uniform')(X_input)

    X = Activation('relu')(X)

    X = Dropout(0.5,seed=10)(X)

    X = Dense(256,kernel_regularizer=l2(0.01),kernel_initializer = 'random_uniform')(X)

    X = Activation('relu')(X)

    X = Dropout(0.25,seed=10)(X)

    X = Dense(16,kernel_regularizer=l2(0.01),kernel_initializer = 'random_uniform')(X)

    X = Activation('relu')(X)

    X = Dropout(0.25,seed=10)(X)

    X = Dense(1, activation='sigmoid')(X)



    # Create model. 

    model = Model(inputs = X_input, outputs = X, name='nnModel')



    return model
nnModel = model_nn(X_train.shape)

nnModel.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

nnModel.fit(x = X_train, y = y_train, epochs = 100, batch_size = 16)
preds = nnModel.evaluate(x= X_test, y=y_test)

print()

print ("Loss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))
df_transformed.head()
import eli5

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rf_model, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm,feature_names = X_test.columns.tolist())
import eli5

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(improved_model, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm,feature_names = X_test.columns.tolist())
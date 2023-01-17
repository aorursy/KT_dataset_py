# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.layers import Dense,Lambda,Flatten,Input

from keras.models import Model

from keras.losses import binary_crossentropy

from keras import backend as K



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/train.csv')

test_set = pd.read_csv('/kaggle/input/test.csv')

df1 = df.copy()

y = df1['Attrition']

df1 = df1.drop(['Attrition'],axis=1)

ID = np.array(test_set['ID'])

df1 = df1.drop(['ID'],axis=1)

test_set = test_set.drop(['ID'],axis=1)


df2 = df1.select_dtypes(include=['object'])

print(df2.head())

test_set1 = test_set.select_dtypes(include=['object'])

print(test_set1.head())

for col in df2.columns:

    df[col] = pd.Categorical(df[col])

    

df2.info()



    

df2.info()
df3 = pd.get_dummies(df2, prefix = 'category')

test_set2 = pd.get_dummies(test_set1, prefix = 'category')


#df3.head()

cat_col = []

for col in df2.columns:

    cat_col.append(col)

df_m = df1.drop(cat_col,axis=1)

df_m.info()

test_setm = test_set.drop(cat_col,axis=1)
for col in df_m.columns:

    if df_m[col].mean() > 1:

        df_m[col] = (df_m[col] - df_m[col].mean()) / df_m[col].var()

        

for col in test_setm.columns:

    if test_setm[col].mean() > 1:

        test_setm[col] = (test_setm[col] - test_setm[col].mean()) / test_setm[col].var()

df_main = pd.concat([df_m,df3],axis=1)

df_main = df_main.drop(['EmployeeNumber'],axis=1)

print(df_main.head())

test_set_main = pd.concat([test_setm,test_set2],axis=1)

print(test_set_main.head())


from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

#print(np.array(df_main)[0].shape[0])



df_main['Attrition'] = y

label_1 = df_main[df_main['Attrition'] == 1]

label_0 = df_main[df_main['Attrition'] == 0]

print(label_1.shape)

label_11 = label_1.drop(['Attrition'],axis=1)

label_00 = label_0.drop(['Attrition'],axis=1)

label_1 = label_1['Attrition']

label_0 = label_0['Attrition']
print(label_11.shape)


#Reparametrization

def sample_z(args):

  mu, sigma = args

  batch     = K.shape(mu)[0]

  dim       = K.int_shape(mu)[1]

  eps       = K.random_normal(shape=(batch, dim))

  return mu + K.exp(sigma / 2) * eps

#KL Divergence Loss

def kl_reconstruction_loss(true, pred):

  # Reconstruction loss

  reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred))

  # KL divergence loss

  kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)

  kl_loss = K.sum(kl_loss, axis=-1)

  kl_loss *= -0.5

  # Total loss = 50% rec + 50% KL divergence loss

  return K.mean(reconstruction_loss + kl_loss)

input_shape = np.array(label_11)[0].shape

lat_dim = 2



#X as Inputs 53 Features

X = Input(shape=input_shape)

Xi = Dense(20,activation='relu')(X)

mu = Dense(lat_dim)(Xi)

sigma = Dense(lat_dim)(Xi)

z = Lambda(sample_z,output_shape=(lat_dim,))([mu,sigma])

enc = Model(X,[mu,sigma,z])

Input_de = Input(shape=(lat_dim,))

#Yii = Dense(20,activation='relu')(Input_de)

Yi = Dense(input_shape[0],activation='relu')(Input_de)

dec = Model(Input_de,Yi)

out = dec(enc(X)[2])

vae1 = Model(X,out)

vae2 = Model(X,out)



vae1.compile(optimizer='adam', loss=kl_reconstruction_loss)

vae2.compile(optimizer='adam', loss=kl_reconstruction_loss)



# Train autoencoder

vae1.fit(label_11, label_11, epochs = 20, batch_size = 1, validation_split = .2)

vae2.fit(label_00, label_00, epochs = 20, batch_size = 1, validation_split = .2)

aug_dat1 = vae1.predict(label_11)

aug_dat2 = vae2.predict(label_00)

print(label_11.shape)

label_1 = np.array(label_1).reshape(-1,1)

aug_dat1 = np.concatenate((aug_dat1,label_1),axis=1)



#print(df_main.head())

#df_main = df_main.concatenate(df)

#df_main.head()
label_0 = np.array(label_0).reshape(-1,1)

aug_dat2 = np.concatenate((aug_dat2,label_0),axis=1)
aug_dat = np.concatenate((aug_dat1,aug_dat2),axis=0)

print(aug_dat.shape)
data = np.concatenate((np.array(df_main),aug_dat),axis=0)

y= data[:,-1]

data = data[:,:-1]
print(data.shape)
X_train, X_test, Y_train, Y_test = train_test_split(data,y,test_size=0.3)

clf = SVC(kernel='poly',C=.1)

clf.fit(X_train,Y_train)

print(clf.score(X_test,Y_test))

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

clf2 = RandomForestClassifier(max_depth=5, random_state=0)

clf2.fit(X_train,Y_train)

print(clf2.score(X_test,Y_test))

clf3 = AdaBoostClassifier(n_estimators=100, random_state=0)

clf3.fit(X_train,Y_train)

print(clf3.score(X_test,Y_test))

#df_m.mean()



pred = clf3.predict(np.array(test_set_main.drop(['EmployeeNumber'],axis=1)))
pred = pred.reshape(441,1)

pred.shape



ID = ID.reshape(441,1)

ID.shape

final = np.concatenate((ID,pred),axis=1)

#final.shape

final = pd.DataFrame(data=final,columns=['ID','Attrition'])
final.to_csv('submission2.csv',index=False)
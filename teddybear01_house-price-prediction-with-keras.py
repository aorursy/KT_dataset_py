# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 

import os

import keras

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error



#dosyaları çek

df_train  =  pd.read_csv('../input/train.csv')

df_test =  pd.read_csv('../input/test.csv')

submission = pd.read_csv('../input/sample_submission.csv')
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice']);

plt.show()
corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);

plt.show()
#kayıp veri



total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max() #just checking that there's no missing data missing...





k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
df_train_x= df_train[['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']]

df_train_y= df_train['SalePrice']



df_test_x= df_test[['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']]

df_test_y= submission['SalePrice']



print(str(df_train_x.shape))

print(str(df_train_y.shape))

print(str(df_test_x.shape))

print(str(df_test_y.shape))

df_train_x.dtypes
# define base model



# create model

model = Sequential()

model.add(Dense(6, input_dim=6, kernel_initializer='normal', activation='relu'))

model.add(Dense(1, kernel_initializer='normal'))

# Compile model

model.compile(loss='mean_squared_error', optimizer='adam')







seed = 7

np.random.seed(seed)

# evaluate model with standardized dataset

#estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)





history = model.fit(np.array(df_train_x), np.array(df_train_y), epochs=800, batch_size=100)
print(history.history.keys())

plt.plot(history.history['loss'])

#plt.plot(history.history['val_acc'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y = model.predict(np.array(df_test_x))

y.shape
predictions = y.transpose()

reality = df_test_y.transpose()







plt.figure(figsize=(20,10))

plt.scatter(predictions, reality)

plt.title('Tahmin x Gerçek',fontsize = 30)

plt.xlabel('Tahmin',fontsize = 30)

plt.ylabel('Gerçek',fontsize = 30)

plt.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k', lw=4)

plt.show()
error=[]



for i in range (1458):

    error.append(abs(reality[i]-predictions[0][i])/reality[i]*100)

    

mean = np.nansum(error)

print ('Hata orani = %',mean/len(error))
sonuc= pd.DataFrame(y, columns=['SalesPrice'])



test =  pd.read_csv('../input/test.csv')



df = pd.concat([test['Id'], sonuc['SalesPrice']], axis=1, keys=['Id', 'SalePrice'])



df['SalePrice'].fillna(df['SalePrice'].mean(), inplace=True)
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







## load data

train_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/train_set.csv',index_col=0) 

test_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/test_set.csv',index_col=0) 



features = ['Rooms','Distance','Bedroom2','Bathroom','Type_t','Type_u','Regionname_Eastern Victoria',  'Regionname_Northern Metropolitan','Regionname_Northern Victoria','Regionname_South-Eastern Metropolitan',

           'Regionname_Southern Metropolitan', 'Regionname_Western Metropolitan', 'Regionname_Western Victoria']





#Creem els dummies per a les variables categòriques Type i Region



type_train = pd.get_dummies(train_set.Type, drop_first=True, prefix='Type')

train_set = pd.concat([train_set, type_train], axis=1)



type_test = pd.get_dummies(test_set.Type, drop_first=True, prefix='Type')

test_set = pd.concat([test_set, type_test], axis=1)







regionname_train = pd.get_dummies(train_set.Regionname, drop_first=True, prefix='Regionname')

train_set = pd.concat([train_set, regionname_train], axis=1)



regionname_test = pd.get_dummies(test_set.Regionname, drop_first=True, prefix='Regionname')

test_set = pd.concat([test_set, regionname_test], axis=1)





train_set[features] = train_set[features].fillna(train_set.mean())







#Elimino outliers que poden modificar negativament la predicció 



train_set = train_set.drop(train_set[train_set.Landsize > 30000].index)



train_set = train_set.drop(train_set[train_set.Bathroom > 6].index)



train_set = train_set.drop(train_set[train_set.BuildingArea > 10000].index)















#Normalitzem les dades de totes les columnes.



X=(train_set[features]-train_set[features].min())/(train_set[features].max()-train_set[features].min())



y = train_set[['Price']]



## the model

# KNeighborsRegressor

from sklearn import neighbors

n_neighbors = 14 # you can modify this paramenter (ONLY THIS ONE!!!)

model = neighbors.KNeighborsRegressor(n_neighbors)



## fit the model

model.fit(X, y)



## predict training set

y_pred = model.predict(X)









from sklearn.metrics import mean_squared_error, r2_score

print("----- EVALUATION ON TRAIN SET ------")

print("RMSE",np.sqrt(mean_squared_error(y, y_pred)))

print("R^2: ",r2_score(y, y_pred))











## predict the test set and generate the submission file



#Omplim els nans del test amb la mitjana de les columnes.

X_test = test_set[features].fillna(test_set.mean())

#Normalitzem les columnes del test

X_test=(X_test-X_test.min())/(X_test.max()-X_test.min())



y_pred = model.predict(X_test)







df_output = pd.DataFrame(y_pred)

df_output = df_output.reset_index()

df_output.columns = ['index','Price']



df_output.to_csv('baseline.csv',index=False)





'''

He utilitzat el model knn degut a que he optat per l'ús de normalitzar les columnes i utilitzar poques característiques per a donar una bona predicció.



Primer he creat el dummies per a dues de les caractarístiques categòriques que m'han millorat més la predicció: Type i Regionname.



Després he aplicat la mitjana de la columna a tots els NaNs d'aquelles columnes i posteriorment les he normalitzades.



He aplicat el model de knn amb una n = 14 ja que es la que m'ha donat millors resultats.





'''





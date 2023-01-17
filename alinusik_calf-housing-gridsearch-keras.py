# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
housing=pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
housing.head()
housing.tail()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()
plt.figure(figsize=(10,7))
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
        s=housing["population"]/100, label="population", figsize=(15,8),
        c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True,
    )
plt.legend

corr_matrix=housing.corr()
print(corr_matrix)
corr_matrix["median_house_value"].sort_values(ascending=False)
housing.hist(bins=50,figsize=(20,15))
housing.hist(bins=500,figsize=(20,15))
#combining 2 attributes
housing["rooms_per_household"]=housing["total_rooms"]/housing["households"]
corr_matrix=housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


bedroom_median=housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(bedroom_median,inplace=True)
#splitting data
from sklearn.model_selection import train_test_split

housing_test=housing.drop("median_house_value",axis=1)
housing_labels_test= housing["median_house_value"].copy()

#housing= strat_train_set.drop("median_house_value",axis=1)
#housing_labels= strat_train_set["median_house_value"].copy()


X_train, X_test, Y_train, Y_test = train_test_split(housing_test, housing_labels_test, 
                                    test_size=0.2, random_state=3)
#SimpleImputer
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")
housing_num=housing_test.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)

print(housing_num.mean().values)
print()

x=imputer.transform(housing_num)
print(x)
#OrdinalEncoder
housing_cat=housing_test[["ocean_proximity"]]
print(housing_cat[900:1000])

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder=OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[900:1000])

print(ordinal_encoder.categories_)
#OneHotEncoder
housing_cat=housing_test[["ocean_proximity"]]

from sklearn.preprocessing import OneHotEncoder

cat_encoder=OneHotEncoder()
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot[900:1000])

housing_cat_1hot.toarray()
#Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

print(housing_num[990:1000])


num_pipeline = Pipeline([('imputer',SimpleImputer(strategy="median")),
                        ('std_scalar',StandardScaler())
                        ])


#housing_num_tr =num_pipeline.fit_transform(housing_num)
#print(housing_num_tr[990:1000])

#Full Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#print(housing[990:1000])

num_attribs=list(housing_num)
cat_attribs=["ocean_proximity"]

full_pipeline=ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat",OneHotEncoder(),cat_attribs)
])
housing_prepared=full_pipeline.fit_transform(housing_test)
#print()
#print()

print(housing_prepared[990:1000])
#keras model definition

from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential

def create_model(lyrs=[8,8,16,32] , act='relu' , opt='Adam' , dr=0.0):
    model=Sequential()
    model.add(Dense(lyrs[0], input_dim=housing_prepared.shape[1],activation=act))
    
    for i in range(1,len(lyrs)):
        model.add(Dense(lyrs[i] , activation=act))
        
    model.add(Dropout(dr))
    
    model.add(Dense(1, activation=act))
    
    model.compile(loss='mean_absolute_error' , optimizer=opt , metrics=['accuracy'])
    return model


#default model to check structure 
#model 1

model1 = create_model()
print(model1.summary())
#to check accuracy of default training

training1 = model1.fit(X_train, Y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0, validation_data=(X_test, Y_test))
acc = np.mean(training1.history['val_accuracy'])
print('Error: '(val_acc*100))
plt.plot(training1.history['accuracy'])
plt.plot(training1.history['val_accuracy'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
print(housing_prepared.shape)
#model 2

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

model2 = KerasClassifier(build_fn=create_model, verbose=3)

#activations = ['relu', 'tanh', 'sigmoid']
#param_grid = dict(act=activations)

batch_size = [16,32]
epochs = [50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)

# search the grid
grid = GridSearchCV(estimator=model2, 
                    param_grid=param_grid,
                    cv=10,
                    verbose=3
                    )  

gridfit = grid.fit(X_train, Y_train)
gridfit.best_score_ 
gridfit.best_params_

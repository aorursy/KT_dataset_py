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
data = pd.read_csv("../input/Pokemon.csv")
len(data)
data.head(9)
data.columns
data['Type 2'].fillna(value='None',inplace=True)
data.head(9)
data['Type 1'].value_counts().plot.bar()
data['Type 2'].value_counts().plot.bar()
data['Legendary'].value_counts().plot.bar()
from sklearn.model_selection import train_test_split

legendaryPokemon = data.loc[data['Legendary']==True]

normalPokemon = data.loc[data['Legendary']==False]

# we will only use the pokemon battle stats + types to determine whether it is legendary or not 

legendaryPokemon = legendaryPokemon[['Type 1','Type 2','Total','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Legendary']]

normalPokemon = normalPokemon[['Type 1','Type 2','Total','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Legendary']]



# now we will randomly sample random non-legendary pokemon from the data set to balance our dataset



sampledNormalPokemon = normalPokemon.sample(100)





x = pd.concat([legendaryPokemon, sampledNormalPokemon])

x = pd.get_dummies(x)

# take last column as training labels and drop it from the training data

y = x['Legendary']

x = x.drop('Legendary', 1)
testNormalPokemon = pd.get_dummies(normalPokemon)

testNormalPokemon.head()
#Using the train_test_split to create train and test sets.

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 47, test_size = 0.30)
# now that we have split our train, test data. Let's increase the amount of Legendary pokemon in our training data, 

# by creating synthetic examples using the SMOTE algorithm

from imblearn.over_sampling import SMOTE



# sampling ration of 1.0 will equally balance the binary classes

sm = SMOTE(random_state=15,sampling_strategy= 1.0)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train)



X_train_res.shape
(y_train_res == True).sum()
from sklearn.ensemble import RandomForestClassifier # for random forest classifier

model = RandomForestClassifier(n_estimators=100,max_depth=7)
#Training the random forest classifier. 

model.fit(X_train_res, y_train_res)

#Predicting labels on the test set.

y_pred =  model.predict(X_test)
#Importing the accuracy metric from sklearn.metrics library



from sklearn.metrics import accuracy_score

print('Accuracy Score on train data: ', accuracy_score(y_true=y_train_res, y_pred=model.predict(X_train_res)))

print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))
# feature importance

importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(model.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.head(10)
plot = importances.plot.pie(y='importance', figsize=(10, 10))
import sklearn.tree 

import graphviz 



# Extract single tree

estimator = model.estimators_[4]



dot_data = dot_data = sklearn.tree.export_graphviz(estimator, out_file=None, 

               feature_names=x.columns,  

                class_names=['normal','legendary'] , filled=True, rounded=True,  special_characters=True)  

graph = graphviz.Source(dot_data) 



graph
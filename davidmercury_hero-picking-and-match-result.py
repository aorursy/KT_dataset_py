# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
test_players = pd.read_csv('../input/test_player.csv')

hero_names = pd.read_csv('../input/hero_names.csv')
hero_df = pd.merge(test_players,hero_names, on = 'hero_id',how = 'left')

hero_df.set_index(['match_id','player_slot'],inplace = True)

hero_df.head(10)
id_df = hero_df['hero_id'].unstack()

id_df.head()
name_df = hero_df['localized_name'].unstack()

name_df.head()
test_labels = pd.read_csv('../input/test_labels.csv', index_col=0)


test_labels.head()
new_test = pd.merge(test_labels,id_df,left_index = True,right_index = True)

new_test = pd.merge(new_test,name_df,left_index = True,right_index = True)

new_test.head()



       

    
new_test.dropna(inplace = True)

new_test.info()


small_test = new_test.iloc[:2000].copy()



small_test['radiant_id'] = [np.zeros(112) for _ in range(len(small_test))]

small_test['dire_id'] = [np.zeros(112) for _ in range(len(small_test))]

small_test['radiant_name'] = [[] for _ in range(len(small_test))]

small_test['dire_name'] = [[] for _ in range(len(small_test))]





for ind in small_test.index:

    for num in range(0,5):

        d = small_test.loc[ind,str(num)+'_x']

        name = small_test.loc[ind,str(num)+'_y']

        small_test.loc[ind,'radiant_id'][d-1] = 1

        small_test.loc[ind,'radiant_name'].append(name)

    for num in range(128,133):

        d = small_test.loc[ind,str(num)+'_x']

        name = small_test.loc[ind,str(num)+'_y']

        small_test.loc[ind,'dire_id'][d-1] = 1

        small_test.loc[ind,'dire_name'].append(name)



    
small_test.info()
frame = small_test[['radiant_id','dire_id']]

def f(x):

    return list(x.radiant_id)+list(x.dire_id)

small_test['combine_id'] = frame.apply(f,axis = 1)
small_test.info()
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier



x = small_test.combine_id.values.tolist()

y = small_test.radiant_win.values.tolist()



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=4)
clf = svm.SVC()

clf.fit(x_train,y_train)

y_predict = clf.predict(x_test)

accuracy_score(y_test,y_predict)
scores = cross_val_score(clf, x, y, cv=5)

scores
clf = RandomForestClassifier(n_estimators=10)

clf.fit(x_train,y_train)

y_predict = clf.predict(x_test)

accuracy_score(y_test,y_predict)
scores = cross_val_score(clf, x, y, cv=5)

scores
clf = KNeighborsClassifier(n_neighbors=100)

clf.fit(x_train,y_train)

y_predict = clf.predict(x_test)

accuracy_score(y_test,y_predict)
scores = cross_val_score(clf, x, y, cv=5)

scores
from keras.models import Sequential, Model

from keras.optimizers import Adam

from keras.regularizers import l2

from keras.layers import Dense, BatchNormalization, Dropout

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau



model = Sequential()

model.add(Dense(100, activation='relu',input_dim=224))

model.add(Dense(20, activation='relu'))

model.add(Dense(5, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 10,batch_size=20,validation_data=(x_test, y_test))
z = small_test.combine_id
a = np.array(z)

np.shape(a)
a = list(z)

np.shape(a)
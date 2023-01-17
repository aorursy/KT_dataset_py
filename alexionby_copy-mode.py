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
data = pd.read_csv('../input/train.csv')

or_data = data.copy()

ID = data.pop('id')

data.head()
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import train_test_split

from sklearn.model_selection import GridSearchCV
from keras.models import Sequential

from keras.layers import Dense,Dropout,Activation

from keras.utils.np_utils import to_categorical

from keras.wrappers.scikit_learn import KerasClassifier
y = data.pop('species')

y = LabelEncoder().fit(y).transform(y)

y_cat = to_categorical(y)

y_cat.shape, y.shape
X = StandardScaler().fit(data).transform(data)

X.shape
def create_model(dropout_rate_l1=0.1 , dropout_rate_l2=0.1):

    

    model = Sequential()

    model.add(Dense(2048,input_dim=192,  init='uniform', activation='relu'))

    model.add(Dropout(dropout_rate_l1))

    model.add(Dense(1536, activation='relu'))

    model.add(Dropout(dropout_rate_l2))

    model.add(Dense(99, activation='softmax'))

    

    model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])

    

    return model
dropout_rate_l1 = [0.3]

dropout_rate_l2 = [0.3]



models = []



for i in dropout_rate_l1:

    for j in dropout_rate_l2:

        models.append([i,j])

        

histories = []



for i in models:

    print(i)

    model = create_model(i[0],i[1])

    

    history = model.fit(X,y_cat,batch_size=192,

                   nb_epoch=10,verbose=0, validation_split=0.1)

    

    histories.append(history)
min_val_loss, param = 1,[]



for history,model in zip(histories,models):

    if min(history.history['val_loss']) < min_val_loss :

        min_val_loss = min(history.history['val_loss'])

        param = model

    print(model)

    print(history.history.keys())

    #print('val_acc: ',max(history.history['val_acc']))

    print('val_loss: ',min(history.history['val_loss']))

    #print('acc: ',max(history.history['acc']))

    print('loss: ',min(history.history['loss']))

    print('\n')



print(min_val_loss, param)
%pylab inline

pylab.rcParams['figure.figsize'] = (9, 6)



import seaborn as sns

import matplotlib.pyplot as plt



for history in histories:

    plt.plot(history.history['val_acc'],'o-')

plt.xlabel('Number of Iterations')

plt.ylabel('Categorical Crossentropy')

plt.title('Train Error vs Number of Iterations')
model = create_model(param[0],param[1])

model.fit(X,y_cat,batch_size=192,nb_epoch=10,verbose=0, validation_split=0.1)
test = pd.read_csv('../input/test.csv')

index = test.pop('id')

test = StandardScaler().fit(test).transform(test)
yPred = model.predict_proba(test)
yPred = pd.DataFrame(yPred, index=index, columns=sort(or_data.species.unique()))

yPred

#pd.concat((index,yPred), axis=1)
yPred.to_csv('result.csv', index=True)
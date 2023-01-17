# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

test_data  = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")



test_id = test_data['id'].values.reshape(-1,1)

train_data = train_data.drop(['nom_9','nom_8','nom_7','nom_6','nom_5','ord_5','id'],axis=1)

test_data = test_data.drop(['nom_9','nom_8','nom_7','nom_6','nom_5','ord_5','id'],axis=1)
columns = [i for i in train_data.columns]



for each in columns:

    

    print(train_data['{}'.format(each)].value_counts())
data_label_columns = ["nom_0","nom_1","nom_2","ord_1","ord_2","nom_3","nom_2","nom_4","ord_3","ord_4","bin_3","bin_4"]



from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()



for i in data_label_columns:

    

    train_data["{}".format(i)] = label_encoder.fit_transform(train_data["{}".format(i)])



#%% im doing same thing with test data

    

for i in data_label_columns:

    

   test_data["{}".format(i)] = label_encoder.fit_transform(test_data["{}".format(i)])
y = train_data['target'].values.reshape(-1,1)



x_data = train_data.drop(['target'],axis=1)



#%% normalizatiton of x_data



x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data))


test_data = (test_data - np.min(test_data)) / (np.max(test_data)-np.min(test_data))
from sklearn.model_selection import train_test_split



x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2 ,random_state = 42)

# Evaluating the ANN

from keras.wrappers.scikit_learn import KerasClassifier

from keras.layers import Dense # build our layers library

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

def build_classifier():

    classifier = Sequential() # initialize neural network

    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))

    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier



#%%



classifier = KerasClassifier(build_fn = build_classifier, epochs = 6)



classifier.fit(x,y)





predicted = classifier.predict(test_data).reshape(-1,1)









submission = np.hstack((test_id,predicted))







test_id.shape
submission.shape
submission1 = pd.DataFrame(submission,columns = ["id","target"])

submission1.to_csv("submission.csv",index=False)
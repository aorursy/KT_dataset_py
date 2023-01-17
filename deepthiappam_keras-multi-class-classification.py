import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import np_utils

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline
seed = 7

np.random.seed(seed)
df = pd.read_csv("../input/Iris.csv")

dataset = df.values

X = dataset[:,1:5].astype(float)

Y = dataset[:,5]
#encode class values as integers

encode = LabelEncoder()

encode.fit(Y)

encoded_Y = encode.transform(Y)

encoded_Y

#convert integers to dummy variables

dummy_y = np_utils.to_categorical(encoded_Y)
def my_model():

    model = Sequential()

    model.add(Dense(4,input_dim=4,init='normal',activation='relu'))

    model.add(Dense(3,init='normal',activation='sigmoid'))

    

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model



from keras.wrappers.scikit_learn import KerasClassifier



estimator = KerasClassifier(build_fn=my_model,nb_epoch=200,batch_size=5,verbose=0)







kfold = KFold(n_splits=10,shuffle=True,random_state=seed)



results = cross_val_score(estimator,X,dummy_y,cv=kfold)



print("Accuracy: %.2f%% (%.2f%%)"%(results.mean()*100,

                            results.std()*100))
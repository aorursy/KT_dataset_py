import numpy as np

import pandas as pd



from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import adam

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils.np_utils import to_categorical





from sklearn.cross_validation import StratifiedKFold, train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")
X_train = train.loc[:,"pixel0":"pixel783"]

y_train = train.label

X_test = test.loc[:,"pixel0":"pixel783"]
y_dummy_train = to_categorical(y_train)
y_dummy_train.shape
#pca = PCA(n_components = 100)

#X_pca = pca.fit_transform(X_train)

X_pca = StandardScaler().fit_transform(X_train)
#X_test_pca = pca.transform(X_test)

X_test_pca = StandardScaler().fit_transform(X_test)
#building the keras model:

model = Sequential()

model.add(Dense(800, input_dim=784, init='normal', activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(10, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ["accuracy"])
hist = model.fit(X_pca, y_dummy_train, nb_epoch=20, batch_size=64, verbose = 0) #validation_split=0.2)
#scores = pd.DataFrame(hist.history)

#scores.loc[:5,["acc", "val_acc"]].plot()
#scores.val_acc.max()
preds = model.predict_classes(X_test_pca)
solution = pd.DataFrame({"ImageId": np.arange(1,(len(preds)+1)),  "Label":preds})
solution.to_csv("solution.csv", index=False)
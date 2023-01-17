import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.callbacks import EarlyStopping

from sklearn.feature_extraction import DictVectorizer
train = pd.read_csv("../input/train.csv").set_index("id")

test = pd.read_csv("../input/test.csv").set_index("id")
test.head(10)
def prepare_data(data, train=True, dv=None):

    cat_keys = [k for k in data.keys() if k.startswith("cat")]

    cat_x = data[cat_keys]

    cont_keys = [k for k in data.keys() if k.startswith("cont")]

    cont_x = data[cont_keys]

    if train:

        y = data["loss"]

    else:

        y = None

    cat_x_dict = [r[1].to_dict() for r in cat_x.iterrows()]

    del cat_x

    if dv is None:

        dv = DictVectorizer().fit(cat_x_dict)

    cat_cont_x = dv.transform(cat_x_dict).toarray()

    del cat_x_dict

    return np.column_stack([cat_cont_x, cont_x]), y, dv
train_x, train_y, dv = prepare_data(train)

test_x, _ , _ = prepare_data(test, False, dv)
print(train_x.shape)

print(test_x.shape)
model = Sequential()

model.add(Dense(output_dim=256, input_dim=train_x.shape[1]))

model.add(Activation("relu"))

model.add(Dropout(0.20))

model.add(Dense(output_dim=128))

model.add(Activation("relu"))

model.add(Dropout(0.30))

model.add(Dense(output_dim=64))

model.add(Activation("relu"))

model.add(Dropout(0.40))

model.add(Dense(output_dim=1))

model.compile("nadam","mae")
es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

model.fit(train_x, train_y.values, nb_epoch=100, batch_size=256, validation_split=0.1, verbose=2, callbacks=[es_cb])

del train_x

del train_y

del train
pred_y = model.predict(test_x)

result = pd.DataFrame(pred_y, index=test.index, columns=["loss"])

result.to_csv("submission.csv")
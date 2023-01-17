%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



plt.style.use("ggplot")
from sklearn.feature_extraction import DictVectorizer



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
train_data = pd.read_csv("../input/train.csv").set_index("id")

train_data.head(5)
train_x, train_y, dv = prepare_data(train_data) 
test_data = pd.read_csv("../input/test.csv").set_index("id")

test_x, _, _ = prepare_data(test_data, False, dv)
from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout

from keras.objectives import MSE, MAE

from keras.callbacks import EarlyStopping
model = Sequential()

model.add(Dense(input_dim=train_x.shape[1], output_dim=256))

model.add(Activation("tanh"))

model.add(Dropout(0.50))

model.add(Dense(output_dim=128))

model.add(Activation("relu"))

model.add(Dropout(0.50))

model.add(Dense(output_dim=64))

model.add(Activation("relu"))

model.add(Dropout(0.50))

model.add(Dense(output_dim=1))

model.compile("nadam", "mae")
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

train_log = model.fit(train_x, train_y.values, batch_size=256, nb_epoch=200, validation_split=0.1, verbose=2, callbacks=[early_stopping])
plt.plot(train_log.history["loss"], label="loss")

plt.plot(train_log.history["val_loss"], label="val_loss")

plt.legend()
pred_y = model.predict(test_x)

result = pd.DataFrame(pred_y, index=test_data.index, columns=["loss"])

result.to_csv("submission.csv")
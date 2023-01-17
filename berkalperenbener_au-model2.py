from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Input

from tensorflow.keras.models import Model

from keras.models import model_from_json

import pickle

import json

import numpy as np



arrays = []



trainImagesX = np.load("../input/inputs/trainImagesX.pickle",allow_pickle=True)

validationImagesX = np.load("../input/inputs/validationImagesX.pickle",allow_pickle=True)

trainAttrX = np.load("../input/inputs/trainAttrX.pickle",allow_pickle=True)

validationAttrX = np.load("../input/inputs/validationAttrX.pickle",allow_pickle=True)

testImagesX = np.load("../input/inputs/images.pickle",allow_pickle=True)

testAttrX = np.load("../input/inputs/list_altitudes.pickle",allow_pickle=True)



trainImagesX = np.asarray(trainImagesX)

validationImagesX = np.asarray(validationImagesX)

trainAttrX = np.asarray(trainAttrX)

validationAttrX = np.asarray(validationAttrX)

testImagesX = np.asarray(testImagesX)

testAttrX = np.asarray(testAttrX)



trainy = trainAttrX

validationy = validationAttrX

testy = testAttrX



print("lşjk")





angle_theta = []

angle_psi = []

angle_phi = []

linear_x = []

linear_y = []

linear_z = []

altitude = []



print("lşjk")

with open("../input/inputs/trainImagesXname.pickle", 'rb') as trainImagesXname:

    lines1 = pickle.load(trainImagesXname)



with open("../input/inputs/validationImagesXname.pickle", 'rb') as validationImagesXname:

    lines2 = pickle.load(validationImagesXname)



with open("../input/inputs/list_image_names.pickle", 'rb') as list_image_names:

    lines3 = pickle.load(list_image_names)





print("lşjk")





with open('../input/auair2019annotations1/annotations.json') as json_file:

    data = json.load(json_file)

    for p in lines1:

        k = data['annotations']['image_name'== p]

        angle_theta.append(k['angle_theta'])

        angle_psi.append(k['angle_psi'])

        angle_phi.append(k['angle_phi'])

        linear_x.append(k['linear_x'])

        linear_y.append(k['linear_y'])

        linear_z.append(k['linear_z'])

        altitude.append(k['altitude'])



train = np.hstack([angle_theta, angle_psi,linear_x,linear_y,linear_z])



with open("train.pickle", "wb") as fp:

    pickle.dump(train, fp)





with open('../input/auair2019annotations1/annotations.json') as json_file:

    data = json.load(json_file)

    for p in lines2:

        k = data['annotations']['image_name'== p]

        angle_theta.append(k['angle_theta'])

        angle_psi.append(k['angle_psi'])

        angle_phi.append(k['angle_phi'])

        linear_x.append(k['linear_x'])

        linear_y.append(k['linear_y'])

        linear_z.append(k['linear_z'])

        altitude.append(k['altitude'])



validation = np.hstack([angle_theta, angle_psi,linear_x,linear_y,linear_z])





with open("validation.pickle", "wb") as fp:

    pickle.dump(validation, fp)

    

with open('../input/auair2019annotations1/annotations.json') as json_file:

    data = json.load(json_file)

    for p in lines3:

        k = data['annotations']['image_name'== p]

        angle_theta.append(k['angle_theta'])

        angle_psi.append(k['angle_psi'])

        angle_phi.append(k['angle_phi'])

        linear_x.append(k['linear_x'])

        linear_y.append(k['linear_y'])

        linear_z.append(k['linear_z'])

        altitude.append(k['altitude'])

        

test = np.hstack([angle_theta, angle_psi,angle_phi,linear_x,linear_y,linear_z,altitude])





with open("test.pickle", "wb") as fp:

    pickle.dump(test, fp)





        

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Input

from tensorflow.keras.models import Model

from keras.models import model_from_json

from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

import pickle

import json

import numpy as np



def create_cnn(width, height, depth, filters=(16, 32, 64, 128)):

 inputShape = (height, width, depth)

 chanDim = -1

 inputs = Input(shape=inputShape)

 for (i, f) in enumerate(filters):

  if i == 0:

   x = inputs

  x = Conv2D(f, (3, 3), padding="same")(x)

  x = Activation("relu")(x)

  x = BatchNormalization(axis=chanDim)(x)

  x = MaxPooling2D(pool_size=(2, 2))(x)

  x = Flatten()(x)

  x = Dense(16)(x)

  x = Activation("relu")(x)

  x = BatchNormalization(axis=chanDim)(x)

  x = Dropout(0.5)(x)

  x = Dense(4)(x)

  x = Activation("relu")(x)

  x = Dense(1, activation="linear")(x)

  model = Model(inputs, x)

  return model



model = create_cnn(64, 64, 3)

opt = Adam(lr=1e-3, decay=1e-3 / 200)

model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

model.fit(x=trainImagesX, y=trainAttrX, validation_data=(validationImagesX, validationAttrX), epochs=200, batch_size=10)







def create_mlp(dim):

    model = Sequential()

    model.add(Dense(8, input_dim=dim, activation="relu"))

    model.add(Dense(4, activation="relu"))

    

    return model







mlp = models.create_mlp(trainx.shape[1])

cnn = models.create_cnn(64, 64, 3)

combinedInput = concatenate([mlp.output, cnn.output])

x = Dense(4, activation="relu")(combinedInput)

x = Dense(1, activation="linear")(x)

model = Model(inputs=[mlp.input, cnn.input], outputs=x)

opt = Adam(lr=1e-3, decay=1e-3 / 200)

model.compile(loss="mean_absolute_percentage_error", optimizer=opt)



with open("../input/pickle/train.pickle", 'rb') as train:

    train = pickle.load(train)

    

with open("../input/pickle/validation.pickle", 'rb') as validation:

    validation = pickle.load(validation)

    

with open("../input/inputs/trainImagesX.pickle", 'rb') as a:

    a = pickle.load(a)

    

with open("../input/inputs/validationImagesX.pickle", 'rb') as validationImagesX:

    validationImagesX = pickle.load(validationImagesX)

    

with open("../input/inputs/trainAttrX.pickle", 'rb') as trainAttrX:

    trainy = pickle.load(trainAttrX)

    

with open("../input/inputs/validationAttrX.pickle", 'rb') as validationAttrX:

    validationy = pickle.load(validationAttrX)

   



validation = np.asarray(validation)

validationy = np.asarray(validationAttrX)

validationImagesX = np.asarray(validationImagesX)



train = np.asarray(train)



trainy = np.asarray(trainAttrX)



a = np.asarray(a)







    

history = model.fit(x=[train, a], y=trainy,validation_data=([validation, validationImagesX], validationy),epochs=200, batch_size=8)



loss_train = history.history['loss']

loss_val = history.history['val_loss']

epochs = range(1,201)

plt.plot(epochs, loss_train, 'g', label='Training loss')

plt.plot(epochs, loss_val, 'b', label='validation loss')

plt.title('Training and Validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()





model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)



model.save_weights("model.h5")
from tensorflow.keras.models import Model

from keras.models import model_from_json

import matplotlib.pyplot as plt

import numpy as np

import pickle





datas = []



arrays = np.load("../input/inputs/images.pickle",allow_pickle=True)

list_altitudes = np.load("../input/inputs/list_altitudes.pickle", allow_pickle=True)





json_file = open('model.json', 'r')

model_json = json_file.read()

json_file.close()

model = model_from_json(model_json)



for f, b in zip(list_altitudes, arrays):

    print(f)

    print(b)

    f = np.array(f)

    b = np.array(b)

    preds = model.predict(b)

    diff = preds.flatten() - f

    diff = diff/1000

    PercentDiff = np.abs(diff)

    datas.append(absDiff)



with open("datas.txt", "wb") as fp:

  pickle.dump(datas, fp)



datas = np.load("datas.txt", allow_pickle=True)

plt.boxplot(datas[0], datas[1], datas[2],datas[3],datas[4],datas[5], names=c("2.5-7.5","7.5-12.5","12.5-17.5","17.5-22.5","22.5-27.5","27.5-32.5"))

plt.title('Estimation Error(Meter)')

plt.legend()

plt.ylabel('Meter')

plt.legend()

plt.show()

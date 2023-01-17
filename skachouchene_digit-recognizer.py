import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, Activation, MaxPool2D, BatchNormalization

from keras.utils import to_categorical

import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd

data = pd.read_csv("../input/digit-recognizer/train.csv")
train, val = train_test_split(data, test_size=0.10, random_state=4)

X = (train.drop("label", axis=1)/255.0).to_numpy().reshape(37800,28,28,1)

Y = train["label"].to_numpy()

X_val=(val.drop("label", axis=1)/255.0).to_numpy().reshape(4200,28,28,1) 

Y_val=val["label"].to_numpy()
#1st model

model0=Sequential()

model0.add(Conv2D(128, kernel_size=3, activation="relu", input_shape=(28,28,1)))

model0.add(MaxPool2D(strides=(2,2)))

model0.add(Dropout(0.25))

model0.add(Conv2D(32, kernel_size=3, activation="relu"))

model0.add(Conv2D(16, kernel_size=3, activation="relu"))

model0.add(Dropout(0.2))

model0.add(Flatten())

model0.add(Dense(128, activation = "relu"))

model0.add(Dropout(0.4))

model0.add(Dense(512, activation = "relu"))

model0.add(Dropout(0.4))



model0.add(Dense(10, activation="softmax"))
#2nd model: ResNet50 (Transfer Learning)



def create_resnet():

    #input_tensor = Input(shape=(28, 28, 3))

    base_model = ResNet50(weights=None, include_top=False, input_tensor=None)

    base_model.load_weights('../input/resnet50weightsfile/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

    x = GlobalAveragePooling2D()(base_model.output)

    x = BatchNormalization()(x)

    x = Dropout(0.4)(x)

    x = Dense(2048, activation=elu)(x)

    x = BatchNormalization()(x)

    x = Dropout(0.4)(x)

    x = Dense(1024, activation=elu)(x)

    x = BatchNormalization()(x)

    x = Dropout(0.3)(x)

    x = Dense(512, activation=elu)(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    output_layer = Dense(10, activation='softmax', name="Output_Layer")(x)

    model_resnet = Model(input_tensor, output_layer)



    return model_resnet
model = create_resnet()
model0.summary()
model0.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("X_train original shape", X.shape)

print("y_train original shape", Y.shape)

print("X_test original shape", X_val.shape)

print("y_test original shape", Y_val.shape)



#model0.fit(X, Y,validation_data=(X_val ,Y_val), batch_size=64, epochs=20)
model.fit(X, Y,validation_data=(X_val ,Y_val), batch_size=64, epochs=20)
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

sample_submission0 = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")



X_test = test.to_numpy().reshape(28000,28,28,1)
submission_result = model.predict(X_test)

submission_result0 = model0.predict(X_test)
submission_result0
submission_final0=[]

for i in range(28000):

  for j in range(10):

    if submission_result0[i][j]==1.0:

      submission_final0.append(j)

      break

    if j==9 and submission_result0[i][j]!=1.0:

      submission_final0.append(0)

    

submission_final=[]

for i in range(28000):

  for j in range(10):

    if submission_result[i][j]==1.0:

      submission_final.append(j)

      break

    if j==9 and submission_result[i][j]!=1.0:

      submission_final.append(0)
submission_final=np.array(submission_final)

submission_final0=np.array(submission_final0)
sample_submission['Label']=submission_final

sample_submission0['Label']=submission_final0

sample_submission0.head()
sample_submission.to_csv("submission_final.csv", index=False) #Convert DataFrame to .csv file

sample_submission0.to_csv("submission_final0.csv", index=False) #Convert DataFrame to .csv file

model0.save_weights('digit_recog_weights0')

model0.save('digit_recog_model0.h5')
model.save_weights('digit_recog_weights')

model.save('digit_recog_model.h5')
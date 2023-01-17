import numpy as np

from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

from tensorflow.python.keras.applications import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd

import os

import matplotlib.pyplot as plt
print(os.listdir("../input/aptos2019-blindness-detection"))

print(os.listdir("../input/resnet50"))
class_weights = {0 : 1, 1 : 4.87, 2 : 1.8, 3 : 9.35, 4 : 6.12}
filenames = os.listdir("../input/aptos2019-blindness-detection/test_images")

img_paths = ["../input/aptos2019-blindness-detection/test_images/" + filename for filename in filenames]



train = pd.read_csv("../input/aptos2019-blindness-detection/train.csv", sep = ",")

submission = pd.DataFrame(columns = ["id_code", "diagnosis"])



for i in range(0, len(train)):

	id_code = train.iloc[i, 0]

	id_code = id_code + ".png"

	train.iloc[i, 0] = id_code

	diagnosis = train.iloc[i, 1]

	diagnosis = str(diagnosis)

	train.iloc[i, 1] = diagnosis



	i += 1



print(train.head())
image_size = 224

num_classes = 5

batch_size = 32

pred_batch_size = 50

num_epochs = 20

steps_per_epoch = 64
data_generator = ImageDataGenerator(preprocessing_function = preprocess_input, validation_split = 0.3)





train_generator = data_generator.flow_from_dataframe(

		dataframe = train,

		directory = "../input/aptos2019-blindness-detection/train_images",

		x_col = "id_code",

		y_col = "diagnosis",

		target_size = (image_size, image_size),

		batch_size = batch_size,

		class_mode = "categorical",

		subset = "training")



validation_generator = data_generator.flow_from_dataframe(

		dataframe = train,

		directory = "../input/aptos2019-blindness-detection/train_images",

		x_col = "id_code",

		y_col = "diagnosis",

		target_size = (image_size, image_size),

		batch_size = batch_size,

		class_mode = "categorical",

		subset = "validation")
model = Sequential()

model.add(ResNet50(include_top = False, pooling = "avg",

                   weights = "../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"))

model.add(Dense(128, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(64, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation = "softmax"))



model.layers[0].trainable = True



model.compile(optimizer = "sgd", loss = "categorical_crossentropy", metrics = ["accuracy"])



print(model.summary())
history = model.fit_generator(

		train_generator,

		epochs = num_epochs,

		steps_per_epoch = steps_per_epoch,

		validation_data = validation_generator,

		validation_steps = 1,

		class_weight = class_weights)
def plot():

    plt.plot(history.history["acc"])

    plt.plot(history.history["val_acc"])

    plt.title("model accuracy (image_size = {}, batch_size = {},\nsteps_per_epoch = {}, num_epochs = {})".

             format(image_size, batch_size, steps_per_epoch, num_epochs))

    plt.ylabel("accuracy")

    plt.xlabel("epoch")

    plt.ylim(top = 1, bottom = 0.4)

    plt.legend(["train", "test"], loc = "upper left")

    plt.show()

    

    plt.plot(history.history["loss"])

    plt.plot(history.history["val_loss"])

    plt.title("loss")

    plt.ylabel("loss")

    plt.xlabel("epoch")

    plt.ylim(top = 3, bottom = 0)

    plt.legend(["train", "test"], loc = "upper left")

    plt.show()

    

plot()
def read_and_prep_images(img_paths, image_size):

	imgs = [load_img(img_path, target_size = (image_size, image_size)) for img_path in img_paths]

	img_array = np.array([img_to_array(img) for img in imgs])

	output = preprocess_input(img_array)



	return output
def make_preds(img_paths, preds):

    img_paths = img_paths

    print(len(img_paths))

    submission = pd.DataFrame(columns = ["id_code", "diagnosis"])

    for i, img_path in enumerate(img_paths):

        pred = preds[i]

        p0 = pred[0]

        p1 = pred[1]

        p2 = pred[2]

        p3 = pred[3]

        p4 = pred[4]



        if p0 > p1 and p0 > p2 and p0 > p3 and p0 > p4:

            p = 0

        elif p1 > p0 and p1 > p2 and p1 > p3 and p1 > p4:

            p = 1

        elif p2 > p0 and p2 > p1 and p2 > p3 and p2 > p4:

            p = 2

        elif p3 > p0 and p3 > p1 and p3 > p2 and p3 > p4:

            p = 3

        elif p4 > p0 and p4 > p1 and p4 > p2 and p4 > p3:

            p = 4



        img_path = img_path[51:-4]

        df = pd.DataFrame([[img_path, p]], columns = ["id_code", "diagnosis"])

        submission = submission.append(df, ignore_index = True)

    print(len(submission))

    #print(submission)

    return submission



complete_submission = pd.DataFrame(columns = ["id_code", "diagnosis"])

a = int(len(img_paths) / pred_batch_size)

for i in range(0, a + 1):

    img_paths_n = img_paths[i * pred_batch_size:(i + 1) * pred_batch_size]

    test_data = read_and_prep_images(img_paths_n, image_size)

    preds = model.predict(test_data)

    submission = make_preds(img_paths_n, preds)

    complete_submission = complete_submission.append(submission, ignore_index = True)

    print("Images {} to {} predicted and saved.".format(i * pred_batch_size, i * pred_batch_size + len(img_paths_n)))

    i += 1



print(len(complete_submission))

complete_submission.to_csv(("submission.csv"), sep = ",", index = False)

print("All done")
import pandas as pd

import numpy as np





from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

# load function from: https://www.kaggle.com/kmader/exploring-the-icebergs-with-skimage-and-keras

# b/c I didn't want to reinvent the wheel

def load_and_format(in_path):

	""" take the input data in .json format and return a df with the data and an np.array for the pictures """

	out_df = pd.read_json(in_path)

	out_images = out_df.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)

	out_images = np.stack(out_images).squeeze()

	return out_df, out_images





train_df, train_images = load_and_format('../input/train.json')



test_df, test_images = load_and_format('../input/test.json')


#also from https://www.kaggle.com/kmader/exploring-the-icebergs-with-skimage-and-keras

X_train, X_test, y_train, y_test = train_test_split(train_images,

		                                            to_categorical(train_df['is_iceberg']),

                                                    random_state = 42,

                                                    test_size = 0.5

                                                   )

print('Train', X_train.shape, y_train.shape)

print('Validation', X_test.shape, y_test.shape)

dummy_dat = np.zeros((802,75,75,1), dtype=np.float32)

fudge_X_train = np.concatenate((X_train, dummy_dat), axis = 3)
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest')



datagen.fit(fudge_X_train)



x_batches = fudge_X_train

y_batches = y_train


epochs = 5



for e in range(epochs):

	print('Epoch', e)

	batches = 0

	per_batch = 5

	for x_batch, y_batch in datagen.flow(fudge_X_train, y_train, batch_size=per_batch):

		x_batches = np.concatenate((x_batches, x_batch), axis = 0)

		y_batches = np.concatenate((y_batches, y_batch), axis = 0)

		batches += 1

		if batches >= len(fudge_X_train) / per_batch:

			# we need to break the loop by hand because

			# the generator loops indefinitely

			break



x_train_new = x_batches[:,:,:,:2]

x_train_new.shape
y_batches.shape
import matplotlib.pyplot as plt



plt.imshow(x_train_new[500,:,:,0])


plt.imshow(x_train_new[-32,:,:,0])

plt.show()

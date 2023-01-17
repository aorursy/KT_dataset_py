import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from random import randint



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
from keras.models import Sequential

from keras.layers import Dense

from keras.utils import np_utils



from matplotlib import pyplot

%matplotlib inline
data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")
data_train.head()
target = data_train["label"].copy()

data_train = data_train.drop("label",axis=1)
print(data_train.shape)

print(data_test.shape)
im = data_train.iloc[0,:].reshape([28,28])

pyplot.imshow(im)
L = 10

pyplot.figure(figsize=(13,13))

for i in range(0, L*L):

    pyplot.subplot(L,L,i+1)

    pyplot.imshow(data_train.iloc[randint(0,4200),:].reshape([28,28]))

    pyplot.axis('off')

        

plt.title('Random observation of images')

pyplot.show()
mean_img = np.mean(data_train, axis=0)

std_img = np.std(data_train, axis=0)



pyplot.figure(figsize=(10,5))

pyplot.subplot(1,2,1)

pyplot.imshow(mean_img.reshape([28,28]))

pyplot.title('Mean Img')

pyplot.axis('off')

pyplot.subplot(1,2,2)

pyplot.imshow(std_img.reshape([28,28]))

pyplot.title('STD Img')

pyplot.axis('off')

pyplot.show()
train = (data_train - mean_img)/std_img 
# fix random seed

seed = 7

np.random.seed(seed)

 

# normalize

train = np.array(data_train / 255)

test = np.array(data_test / 255)

# one hot encode outputs

y_train = np_utils.to_categorical(target)

print(train.shape)

print(y_train.shape)

print(test.shape)

num_classes = y_train.shape[1]

num_pixels = 784

print(num_classes)
def the_model():

	model = Sequential()

	model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))

	model.add(Dense(num_classes, init='normal', activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model
model = the_model()

model.fit(train, y_train, nb_epoch=10, batch_size=200, verbose=2)

scores = model.evaluate(train, y_train, verbose=0)

print("Baseline Error: %.2f%%" % (100-scores[1]*100))
predict = model.predict(test, verbose=0)
predict.shape
predict[0]
p = np_utils.categorical_probas_to_classes(predict)
submission = pd.DataFrame({'ImageId': [x+1 for x in range(test.shape[0])], 'Label': p})
submission.to_csv("submission101802.csv",index=False)
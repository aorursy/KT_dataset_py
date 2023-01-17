import numpy as np

# Load the data (you can use pandas, but I'm more comfortable working with numpy)

train = np.genfromtxt("../input/train.csv", delimiter=',', skip_header = 1)

test = np.genfromtxt("../input/test.csv", delimiter=',', skip_header = 1)
print(train.shape)

print(test.shape)
#Here we need to get our value for the labels in a seperate variable

#and these are specific labels, so let's seperate them

labels = train[:,0]

#and let us get our image information into another variable

images = train[:,1:]
print(len(labels))

print(len(images))

print()

print(labels.ndim)

print(images.ndim)

print()

print(labels.shape)

print(images.shape)
#Let's see our data values and distribution of the information

import seaborn as sns

sns.distplot(labels)
sns.kdeplot(labels, shade=True)
#Though here is an interesting issue

#It's better to make numbers more uniform for this approach or at least distributed more densly

#Say from 0 to 1

sns.kdeplot(images[666], shade=True)
print(np.amax(images))

print(np.amin(images))

#As we confirm - values vary between 0 and 255 (standard shading though)

#Let's create a new variable to address that

new_images = images/255

print(np.amax(new_images))

print(np.amin(new_images))
#Visually not much changed, but it will matter in general later

#As such distribution in theory should allow for a better learning practice and rate and accuracy and other fancy stuff

sns.kdeplot(new_images[666], shade=True)
#What we see is not much but it's something and we do recognize that at least we are on the right track

#Now let's come to building a simple neural network

from keras.models import Sequential

from keras.layers.core import Dense, Activation

from keras.optimizers import Adam
model = Sequential()

model.add(Dense(32, input_dim=new_images.shape[1], activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(10, activation='softmax')) #We will have 10 classes to predict of numbers, so we need a 10 at the output



optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

model.compile(loss='mean_squared_error', optimizer=optimizer, metrics =['accuracy'])

model.summary()
#And of course, before doing anything - dedicate a small sample of data to the validation set

valid_images = new_images[40000:,:]

valid_labels = labels[40000:]



train_images = new_images[:40000,:]

train_labels = labels[:40000]



print(valid_images.shape)

print(len(valid_images))

print(valid_labels.shape)

print(len(valid_labels))

print()

print(train_images.shape)

print(len(train_images))

print(train_labels.shape)

print(len(train_labels))
#And let us binarize our data for the predictions sake

from sklearn.preprocessing import LabelBinarizer

onehot = LabelBinarizer()

Y_train = onehot.fit_transform(train_labels)

Y_val   = onehot.transform(valid_labels)
print(Y_train[0])

print(train_labels[0])

print()

print(Y_val[50])

print(valid_labels[50])
#Comence the training of our model

#model.fit(train_images, Y_train, epochs=32, batch_size=8)
#We can make a short evaluation of how well our model seems to work

#model.evaluate(valid_images,Y_val)

#The second score is of accuracy, hence on validation set we are predicting roughly 96% of values correctly.

#That is 1920 out of 2000.

#After every itteration of committing the code the values can change a little bit...just go with the approx. average of 96%
#We convert our binarized values back into it's original form and can compare what is the situation for the first 20

#values and how they got predicted from our model

#temp_val = onehot.inverse_transform(Y_val)

#for zzz in range(0,20):

#    print(onehot.inverse_transform(model.predict(valid_images[zzz].reshape(1,-1)))==temp_val[zzz])

    

#reshaping is just a weird part of restructuring the matrix, so that the prediction would work

#though in reality it changes nothing...I cannot explain it actually properly, but otherwise the model would just not work

#with what we feed it with :(
#Naturally we do not have any labels, so we do not have to care about them, but will have to use our onehot type binarizing

#converted further, as it is what our model will be predicting.

#print(test.shape) #just to remember what are we working with
#I will do this in a most inefficient way for creating of a submission file

#final_list = [] #this is what we will populate with our data

#for number in range(test.shape[0]):

#    temp_holder = []

#    temp_var = int(onehot.inverse_transform(model.predict(test[number].reshape(1,-1)))) #int is added as otherwise value is predicted

                                                                                        #in an inconvenient array([value]) format

#    temp_holder.append(number+1)

#    temp_holder.append(temp_var)

#    final_list.append(temp_holder)
#here is the example of the "issue" that was addressed

#print(int(onehot.inverse_transform(model.predict(test[5].reshape(1,-1)))))

#print(onehot.inverse_transform(model.predict(test[5].reshape(1,-1))))
#Before saving it into a file, let's look up how the numbers are distributed in our list

#play_list = np.array(final_list)

#print(play_list.shape)

#and we want our values to be visualized so let's dedicate a variable to it for a moment

#play_vizual = play_list[:,1]

#print(play_vizual.shape)
#This will get us a rough idea

#sns.distplot(play_vizual)

#well, we can see that distribution looks...plausible, as no one value is dominated, so maybe it's actually ok

#let's save our file
#This is our train data. It's shape is 40000x784

train_images.shape
#If we make "pictures" out of this information instead of flat signals, we could utilize a Convolutional Neural Network (CNN)

#To not make any mistakes, the following structure could be utilized, though there are more elegant ways of solving this task also:

train_images_cnn = []

for value in range(train_images.shape[0]):

    temp_val = np.reshape(train_images[value], (28,28))

    train_images_cnn.append(temp_val)
#And now we have our pictures

np.array(train_images_cnn).shape

#save it

train_images_cnn_final = np.array(train_images_cnn)\

#and re-shape for 4 dimensional concept (I have no explanation for this, it just doesn't work without it)

X = train_images_cnn_final.reshape(train_images_cnn_final.shape[0], train_images_cnn_final.shape[1], train_images_cnn_final.shape[2], 1)

train_images_cnn_final = X
train_images_cnn_final.shape
#The structure for a cnn we can use as the following

from keras.models import Sequential

from keras.optimizers import Adam

from keras.layers import Dense, Conv2D, Flatten, Input, Conv3D, Conv1D, InputLayer, MaxPooling2D

from sklearn.preprocessing import LabelBinarizer



model = Sequential()

model.add(InputLayer(input_shape=(28, 28, 1)))

model.add(Conv2D(256, kernel_size=4, activation='relu'))

model.add(Conv2D(128, kernel_size=4, activation='relu'))

model.add(MaxPooling2D((2, 2), strides=2, padding='same'))

model.add(Conv2D(128, kernel_size=2, activation='relu'))

model.add(Conv2D(64, kernel_size=1, activation='relu'))

model.add(MaxPooling2D((2, 2), strides=2, padding='same'))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))

optimizer = Adam (lr = 0.0001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])



model.summary()
#Because of how it trains, there is no need even to go into more than 6-10 epochs. There is even a higher chance that then we will overfit the data.

model.fit(train_images_cnn_final, Y_train, epochs=10, batch_size=4)
#We can evaluate the performance of our model on the set aside validation data

#First we need to prepare it though. 

valid_images_cnn = []

for value in range(valid_images.shape[0]):

    temp_val = np.reshape(valid_images[value], (28,28))

    valid_images_cnn.append(temp_val)



valid_images_cnn_final = np.array(valid_images_cnn)

#and re-shape for 4 dimensional concept (I have no explanation for this, it just doesn't work without it)

Z = valid_images_cnn_final.reshape(valid_images_cnn_final.shape[0], valid_images_cnn_final.shape[1], valid_images_cnn_final.shape[2], 1)

valid_images_cnn_final = Z



model.evaluate(valid_images_cnn_final,Y_val)



#Accuracy between 98-99%...very nice. Let's move forward.
#We convert our binarized values back into it's original form and can compare what is the situation for the first 20 values and how they got predicted from our model

temp_val = onehot.inverse_transform(Y_val)

for zzz in range(0,20):

    print(onehot.inverse_transform(model.predict(valid_images_cnn_final[zzz].reshape(1, 28, 28, -1)))==temp_val[zzz])

    

#reshaping is just a weird part of restructuring the matrix, so that the prediction would work

#though in reality it changes nothing...I cannot explain it actually properly, wait...I already mentioned this earlier
print(test.shape) #just to remember what are we working with, and this time we need to again reshape the data
#First let's get our data in a presentable form to the model

#A horrible approach is used where the same variable is just recycled...don't do this in the real world

test_images = []

for value in range(test.shape[0]):

    temp_val = np.reshape(test[value], (28,28))

    test_images.append(temp_val)



test_images = np.array(test_images)

L = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)

test_images = L

#But hey...as long as it works...for now

print(test_images.shape)
#Let's make predictions

final_list = []

for number in range(test_images.shape[0]):

    temp_holder = []

    temp_var = int(onehot.inverse_transform(model.predict(test_images[number].reshape(1, 28, 28, -1)))) 

    temp_holder.append(number+1)

    temp_holder.append(temp_var)

    final_list.append(temp_holder)
#Before saving it into a file, let's look up how the numbers are distributed in our list

play_list = np.array(final_list)

print(play_list.shape)

#and we want our values to be visualized so let's dedicate a variable to it for a moment

play_vizual = play_list[:,1]

print(play_vizual.shape)

#This will get us a rough idea

sns.distplot(play_vizual)

#well, we can see that distribution looks...plausible, as no one value is dominated, so maybe it's actually ok

#let's save our file
#We can download the file and make few adjustments in excel ourselves if needed

#But this version should present the final file for submission as it is

np.savetxt('digits_prediction.csv', final_list, delimiter = ',', header = 'ImageId,Label', fmt='%1.f', comments='')
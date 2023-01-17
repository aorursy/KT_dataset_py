%matplotlib inline



# For simple vectorized calculations

import numpy as np



# Mainly data handling and representation

import pandas as pd



# Models

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation, Dropout

from keras.losses import categorical_crossentropy

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras import Model



# Data preparation

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA



# Plotting and display

from IPython.display import display

from matplotlib import pyplot as plt



np.random.seed(0)
# Path of the file to read.

train_file_path = '../input/train.csv'



# Read the file

digit_data_orig = pd.read_csv(train_file_path)



# The shape of the data

digit_data_orig.shape
# Separate the label from the data

y_orig = digit_data_orig.iloc[:, 0].values.reshape(-1,1)



m = y_orig.shape[0]

print("The number of images: m = {}".format(m))



# There are 42000 images so 42000 labels

print("The shape of y: {}".format(y_orig.shape))
# One-hot encode the categorical values

def one_hot_encode_categories(y):

    

    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)



    y_one_hot = pd.DataFrame(encoder.fit_transform(y), columns=encoder.get_feature_names())

        

    return y_one_hot, encoder
# One hot encode y

y_one_hot, encoder = one_hot_encode_categories(y_orig)



# Strip the category names

y_one_hot.columns = pd.DataFrame(y_one_hot.columns)[0].apply(lambda x: x[-1])



# The number of categories

n_y = y_one_hot.shape[1]

print("The number of categories: n_y = {}".format(n_y))



# A few examples

print(y_orig[0:10])

y_one_hot.head()
# Let's see how many examples are ther from each category

display(pd.DataFrame(y_one_hot.sum(axis=0)).transpose())



# Plot the values

plt.bar(y_one_hot.columns, y_one_hot.sum(axis=0))



# There is approximately the same number of examples there are from each category
# Separate the image data

X_orig = digit_data_orig.iloc[:, 1:].values.reshape(-1,1)



print("The shape of X without reshaping: {}".format(X_orig.shape))



# There are 42000 images and 64x64 pixel each image which is 32928000 total
# Let's reshape the images and view some

X_reshaped = X_orig.reshape(-1,28,28)



def plot_sample_images(X, y, images_to_show=10, random=True):



    fig = plt.figure(1)



    images_to_show = min(X.shape[0], images_to_show)



    # Set the canvas based on the numer of images

    fig.set_size_inches(18.5, images_to_show * 0.3)



    # Generate random integers (non repeating)

    if random == True:

        idx = np.random.choice(range(X.shape[0]), images_to_show, replace=False)

    else:

        idx = np.arange(images_to_show)

        

    # Print the images with labels

    for i in range(images_to_show):

        plt.subplot(images_to_show/10 + 1, 10, i+1)

        plt.title(str(y[idx[i]]))

        plt.imshow(X[idx[i], :, :], cmap='Greys')

        



# Choose how many images you would like to see

images_to_show = 30



plot_sample_images(X_reshaped, y_orig, images_to_show=images_to_show)
# The number of X features are 28*28 = 784

n_x = 28*28



print("The number of X features are: n_x = {}".format(n_x))
# Scale the image pixel values from 0-255 to 0-1 range so the neural net can to converge faster

X_scaled = X_reshaped / 255



print("Original scale: {} - {}".format(X_reshaped.min(), X_reshaped.max()))

print("New scale: {} - {}".format(X_scaled.min(), X_scaled.max()))
X = X_scaled.reshape(-1, 28, 28, 1)

y = y_one_hot
# We can train a model using directly the images, let's first do that
def model_definition():

    # Define a simple model in Keras

    model = Sequential()



    # Add layers to the model



    # Add convolutional layer

    model.add(Conv2D(50, kernel_size=(5,5), input_shape=(28,28,1)))



    # Add ReLu activation function

    model.add(Activation('relu'))



    # Add dropout layer for generalization

    model.add(Dropout(0.05))



    # Add maxpool layer

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))



    # Add batch normalization to help learning and avoid vanishing or exploding gradient

    model.add(BatchNormalization())



    # Add convolutional layer

    model.add(Conv2D(50, kernel_size=(3,3)))



    # Add ReLu activation function

    model.add(Activation('relu'))



    # Add dropout layer for generalization

    model.add(Dropout(0.05))



    # Maxpool layer

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))



    # Add batch normalization to help learning and avoid vanishing or exploding gradient

    model.add(BatchNormalization())



    # Add flatten layer to get 1d data for dense layer

    model.add(Flatten())



    # Dense layer

    model.add(Dense(100, input_dim=650))

    

    # Add ReLu activation function

    model.add(Activation('relu'))



    # Dense layer

    model.add(Dense(10))

    

    # Add sigmoid activation function to get values beteween 0-1

    model.add(Activation('softmax'))

    

    return model
model = model_definition()
# Define the hyperparameters



batch_size = 32

epochs = 100
# Define the loss function, this is a categorical cross entropy

loss = categorical_crossentropy
# Define the optimizer

optimizer = Adam(lr=0.0005)
# Compile the model

model.compile(loss=loss, optimizer=optimizer, metrics=["categorical_accuracy"])
# Let's see the model configuration

model.summary()
# Split the data into train and validation parts

train_X, val_X, train_y, val_y = train_test_split(X, y.values, random_state=1)



print("The train image shape: {}".format(train_X.shape))

print("The train label shape: {}".format(train_y.shape))
# Define the augmentation properties

generator = ImageDataGenerator(#featurewise_center=True,

                               #samplewise_center=True,

                               #featurewise_std_normalization=True,

                               #samplewise_std_normalization=True,

                               #zca_whitening=False,

                               #zca_epsilon=1e-06,

                               rotation_range=10,

                               width_shift_range=0.1,

                               height_shift_range=0.1,

                               #brightness_range=None,

                               shear_range=5,

                               zoom_range=0.1,

                               cval=0.0,)



# Fit the augmentation to the images

generator.fit(X)



X_augmented, y_augmented = generator.flow(train_X, train_y, batch_size=batch_size).next()



# Plot some augmented images

plot_sample_images(X_augmented[:10,:,:,0], encoder.inverse_transform(y_augmented)[:10,0], 10)
# Let's train the model

history = model.fit_generator(generator.flow(train_X, train_y, batch_size=batch_size),

                              steps_per_epoch=len(train_X) / batch_size,

                              validation_data=[val_X, val_y],

                              epochs=epochs)
def plot_history(history):# Plot the loss and accuracy

    # Format the train history

    history_df = pd.DataFrame(history.history, columns=history.history.keys())



    

    # Plot the accuracy

    fig = plt.figure()

    fig.set_size_inches(18.5, 10)

    ax = plt.subplot(211)

    ax.plot(history_df["categorical_accuracy"], label="categorical_accuracy")

    ax.plot(history_df["val_categorical_accuracy"], label="val_categorical_accuracy")

    ax.legend()

    plt.title('Score during training.')

    plt.xlabel('Training step')

    plt.ylabel('Accuracy')

    plt.grid(b=True, which='major', axis='both')

    

    # Plot the loss

    ax = plt.subplot(212)

    ax.plot(history_df["loss"], label="loss")

    ax.plot(history_df["val_loss"], label="val_loss")

    ax.legend()

    plt.title('Loss during training.')

    plt.xlabel('Training step')

    plt.ylabel('Loss')

    plt.grid(b=True, which='major', axis='both')

    

    plt.show()
plot_history(history)
# The result seems to be good, so let's train the data on the whole train dataset



# Reinitialize the model

#model = model_definition()



# Compile the model

model.compile(loss=loss, optimizer=optimizer, metrics=["categorical_accuracy"])



# Train the model on the full train data

final_history = model.fit_generator(generator.flow(X, y, batch_size=batch_size),

                                              steps_per_epoch=len(X) / batch_size,

                                              epochs=epochs)
def plot_history(history):# Plot the loss and accuracy

    # Format the train history

    history_df = pd.DataFrame(history.history, columns=history.history.keys())

    display(history_df)

    

    fig = plt.figure()

    ax = plt.subplot(111)

    for i in range(history_df.shape[1]):

        ax.plot(history_df.iloc[:,i], label=history_df.columns[i])

    ax.legend()

    plt.title('Score() and loss during training.')

    plt.xlabel('Training step')

    plt.ylabel('Accuracy and Loss')

    plt.grid(b=True, which='major', axis='both')

    plt.show()



plot_history(final_history)
# Path of the file to read.

train_file_path = '../input/test.csv'



# Read the file

test_data_orig = pd.read_csv(train_file_path)



# The shape of the data

test_data_orig.shape
# Let's reshape the images and view some

X_test_reshaped = test_data_orig.values.reshape(-1, 28, 28, 1)

print(X_test_reshaped.shape)

# Choose how many images you would like to see

images_to_show = 30



plot_sample_images(X_test_reshaped[:, :, :, 0], np.zeros((X_test_reshaped.shape[0])), images_to_show=images_to_show)
# Scale the image pixel values from 0-255 to 0-1 range so the neural net can to converge faster

X_test_scaled = X_test_reshaped / 255



# The final X_test

X_test = X_test_scaled
# Predict the labels

test_preds_unscaled = model.predict(X_test)
# Inverse transform the predictions to the original scale

test_preds = encoder.inverse_transform(test_preds_unscaled)[:,0]
# Save the predictions

output = pd.DataFrame({'ImageId': range(1, test_preds.shape[0] + 1),

                       'Label': test_preds})



output.to_csv('submission.csv', index=False)
model.summary()
# Get the output of the last activation function

layer_name = 'dense_4'



# Define an intermediate model

intermediate_layer_model = Model(inputs=model.input,

                                 outputs=model.get_layer(layer_name).output)



# Calculate the values of the intermediate model

intermediate_output = intermediate_layer_model.predict(X)
PCA_transformer = PCA()

        

# Fit and transform

activation_7_PCA = PCA_transformer.fit_transform(intermediate_output)
# %matplotlib inline

number_of_points = 10000



x1 = activation_7_PCA[:number_of_points, 0]

x2 = activation_7_PCA[:number_of_points, 1]



fig = plt.figure()

fig.set_size_inches(18.5, 10)



plt.scatter(x=x1,

            y=x2,

            c=(y_orig[:number_of_points])[:, 0],

            cmap="tab10")



ax = plt.subplot(111)



for i in range(number_of_points):

    if not i % 100:

        ax.annotate(str(y_orig[i]), (x1[i], x2[i]))



plt.title('Last layer visualization using PCA 2D')

plt.show()


from mpl_toolkits.mplot3d import Axes3D



number_of_points = 1000



x1 = activation_7_PCA[:number_of_points, 0]

x2 = activation_7_PCA[:number_of_points, 1]

x3 = activation_7_PCA[:number_of_points, 2]



fig = plt.figure()

fig.set_size_inches(10, 10)







ax = plt.subplot(111, projection='3d')



ax.scatter(xs=x1,

            ys=x2,

            zs=x3,

            c=(y_orig[:number_of_points])[:,0],

            cmap="tab10",)



for i in range(number_of_points):

    if not i % 10:

        ax.text(x1[i], x2[i], x3[i], str(y_orig[i]))



plt.title('Last layer visualization using PCA 3D')

plt.show()
augmented_data_batch = 100000

X_aug, y_aug_real_unscaled = generator.flow(X, y.values, batch_size=augmented_data_batch).next()



# Print examples when the model made bad decisions

y_aug_preds_unscaled = model.predict(X_aug)

# Inverse transform the predictions to the original scale

y_aug_preds = encoder.inverse_transform(y_aug_preds_unscaled)

y_aug_real = encoder.inverse_transform(y_aug_real_unscaled)
y_aug_all = pd.DataFrame([y_aug_preds[:,0], y_aug_real[:,0]]).transpose()

y_aug_all.columns = ["y predicted", "y real"]

y_aug_all.head(10)

plot_sample_images(X_aug[:10, :, :, 0], y_aug_real, 10, random=False)
pred_errors = y_aug_all[y_aug_all['y predicted'] != y_aug_all['y real']]

total_errors = pred_errors.shape[0]



print("The total number of errors from {} augmented image: {}".format(augmented_data_batch, total_errors))

print("Which is {0:.3f}%".format(total_errors/augmented_data_batch * 100))
errors_by_category = []

error_count_by_category = []



for i in range(n_y):

    

    errors_by_category.append(pred_errors[pred_errors["y real"] == i])



    error_count_by_category.append(errors_by_category[i].shape[0])



error_count_by_category_df = pd.DataFrame(error_count_by_category).transpose()



print("Number of errors by category: ")

display(error_count_by_category_df)



fig = plt.figure()

ax = plt.subplot(111)

plt.bar(x=error_count_by_category_df.columns, height=error_count_by_category_df.values[0]/total_errors * 100)



plt.title('Percentage of error by category')

plt.xlabel('Categories')

plt.ylabel('Percentage of error (%)')

#plt.xticks(range(n_y))

plt.show()
for errors in errors_by_category:

    plot_sample_images(X_aug[errors.index, :, :, 0], errors["y predicted"].values, 10, random=False)

    plt.show()
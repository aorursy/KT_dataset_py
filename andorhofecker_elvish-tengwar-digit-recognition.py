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



# For checking GPU backend

from keras import backend

from tensorflow.python.client import device_lib



# Data preparation

from sklearn.preprocessing import MinMaxScaler, PowerTransformer, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA



# Plotting and display

from IPython.display import display

from matplotlib import pyplot as plt



# Image manipulation

from PIL import Image



np.random.seed(0)
# Path of the file to read.

train_file_path = "../input/tengwar-digits-dataset/tengwar_data.csv"



# Read the file

digit_data_orig = pd.read_csv(train_file_path)



# The shape of the data

digit_data_orig.shape
# Separate the label from the data

y_orig = digit_data_orig.iloc[:, 0].values.reshape(-1,1)



m = y_orig.shape[0]

print("The number of images: m = {}".format(m))



# There are 308 images so 308 labels

print("The shape of y: {}".format(y_orig.shape))
# One-hot encode the categorical values

def one_hot_encode_categories(y):

    

    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)



    y_one_hot = pd.DataFrame(encoder.fit_transform(y), columns=encoder.get_feature_names())

        

    return y_one_hot, encoder
# One hot encode y

y_one_hot, encoder = one_hot_encode_categories(y_orig)



# Strip the category names

y_one_hot.columns = pd.DataFrame(y_one_hot.columns)[0].apply(lambda x: x[3:-2])



# The number of categories

n_y = y_one_hot.shape[1]

print("The number of categories: n_y = {}".format(n_y))



# A few examples

print(np.random.choice(y_orig.reshape(-1), 10))

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
# In case of square image we can calculate the edge size

edge_size = 64



# Let's reshape the images and view some

X_reshaped = X_orig.reshape(-1, edge_size, edge_size)



def plot_sample_images(X, y, images_to_show=10, random=True):



    fig = plt.figure(1)



    images_to_show = min(X.shape[0], images_to_show)



    # Set the canvas based on the numer of images

    fig.set_size_inches(18.5, max(3, images_to_show * 0.3))



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
# The number of X features are 64*64 = 4096

n_x = edge_size*edge_size



print("The number of X features are: n_x = {}".format(n_x))
# Scale the image pixel values from 0-255 to 0-1 range so the neural net can to converge faster

X_scaled = X_reshaped / 255



print("Original scale: {} - {}".format(X_reshaped.min(), X_reshaped.max()))

print("New scale: {} - {}".format(X_scaled.min(), X_scaled.max()))
X = X_scaled.reshape(-1, edge_size, edge_size, 1)

y = y_one_hot
# We can train a model using directly the images, let's first do that
def model_definition():

    # Define a simple model in Keras

    model = Sequential()



    # Add layers to the model



    # Add convolutional layer

    model.add(Conv2D(10, kernel_size=(5,5), input_shape=(edge_size, edge_size, 1)))



    # Add ReLu activation function

    model.add(Activation('relu'))



    # Add dropout layer for generalization

    model.add(Dropout(rate = 0.1))



    # Add maxpool layer

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))



    # Add batch normalization to help learning and avoid vanishing or exploding gradient

    model.add(BatchNormalization())



    # Add convolutional layer

    model.add(Conv2D(10, kernel_size=(3,3)))



    # Add ReLu activation function

    model.add(Activation('relu'))



    # Add dropout layer for generalization

    model.add(Dropout(rate = 0.1))



    # Maxpool layer

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))



    # Add batch normalization to help learning and avoid vanishing or exploding gradient

    model.add(BatchNormalization())



    # Add flatten layer to get 1d data for dense layer

    model.add(Flatten())



    # Dense layer

    model.add(Dense(10))#, input_dim=650))

    

    # Add ReLu activation function

    model.add(Activation('relu'))



    # Dense layer

    model.add(Dense(n_y))

    

    # Add sigmoid activation function to get values beteween 0-1

    model.add(Activation('softmax'))

    

    return model
model = model_definition()
# Define the hyperparameters



batch_size = 64

epochs = 300
# Define the loss function, this is a categorical cross entropy

loss = categorical_crossentropy
# Define the optimizer

optimizer = Adam(lr=0.0002)
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

                               shear_range=0.1,

                               zoom_range=0.1,

                               cval=0.0,)



# Fit the augmentation to the images

generator.fit(X)



X_augmented, y_augmented = generator.flow(train_X, train_y, batch_size=batch_size).next()



# Plot some augmented images

plot_sample_images(X_augmented[:10,:,:,0], encoder.inverse_transform(y_augmented)[:10,0], 10)
def kfold(X, y_one_hot, n_splits=5, seed=0):

    """

    This function randomly splits the input and output data equally probable by label to train and validation set.

    X: nd array of shape: (m, n_x, n_x, colors)

    y: pandas DataFrame with shape: (m, n_y) and column names as label names

    n_splits: the number of splits that kfold splits the data into

    

    m: number of samples

    n_h: height in pixels

    n_w: width in pixels

    colors: nr of color layers, colors >= 1

    n_y: number of label types, n_y >= 1

    """

    

    # Initialize the already chosen for validation because we do not want to choose a validation sample twice

    already_chosen_for_validation = []

    

    for i in range(n_splits):

        

        # The number of samples

        m = X.shape[0]



        # Define the validation and train size

        validation_size = int(m / n_splits)

        train_size = m - validation_size



        # The number of labels

        n_y = y_one_hot.shape[1]



        # Number of example by label

        n_label = int(train_size / n_y)



        # Initialize train set list

        X_train_by_label = []

        y_train_by_label = []



        # Initialize validation set list

        X_validation_by_label = []

        y_validation_by_label = []



        # choose equally randomly from each label type about equally

        for label in y_one_hot.columns:

            # Get the indexes from the y_one_hot where the "label" is the label variable

            indexes_by_label = y_one_hot[y_one_hot[label] == 1].index.values.tolist()



            # The number of samples in this type of label

            m_label = len(indexes_by_label)

            

            # Remove the already once chosen validation data

            available_indexes_by_label = list(set(indexes_by_label).difference(set(already_chosen_for_validation)))

            

            if i < n_splits - 1:

                # Choose from the indexes randomly

                val_indexes_by_label = np.random.choice(available_indexes_by_label, int(m_label / n_splits), replace=False)

            else:

                val_indexes_by_label = available_indexes_by_label

            

            # New set with elements in indexes_by_label but not in train_indexes_by_label is equals the indexes of the training set by label

            train_indexes_by_label = list(set(indexes_by_label).difference(set(val_indexes_by_label)))



            # Append the selected categories of the train data by label

            X_train_by_label.append(X[train_indexes_by_label, :, :, :])

            y_train_by_label.append(y_one_hot.iloc[train_indexes_by_label, :])



            # Append the selected categories of the validation data by label

            X_validation_by_label.append(X[val_indexes_by_label, :, :, :])

            y_validation_by_label.append(y_one_hot.iloc[val_indexes_by_label, :])

            

            # Extend the already_chosen_for_validation list with the newly chosen validation

            already_chosen_for_validation.extend(val_indexes_by_label)





        # Create final train data

        X_train = np.concatenate(X_train_by_label)

        y_train = np.concatenate(y_train_by_label)



        # Create final validation data

        X_validation = np.concatenate(X_validation_by_label)

        y_validation = np.concatenate(y_validation_by_label)

    

        yield X_train, X_validation, y_train, y_validation
# Train with cross validation

histories = []

scores = []

models = []



# Cross validation train

for X_train, X_val, y_train, y_val in kfold(X, y, n_splits=5):

    # Define model

    model = model_definition()

    

    # Compile model

    model.compile(loss=loss, optimizer=optimizer, metrics=["categorical_accuracy"])

    

    # Fit the model

    histories.append(model.fit_generator(generator.flow(X_train, y_train, batch_size=batch_size),

                                         steps_per_epoch=len(X_train) / batch_size,

                                         validation_data=[X_val, y_val],

                                         epochs=epochs,

                                         verbose=0))

    

    # Append models

    models.append(model)

    

    # Calculate the score of the model

    score = model.evaluate(X_val, y_val, verbose=0)

    print("{}: {}".format(model.metrics_names, score))

    scores.append(score)
for i in range(len(model.metrics_names)):

    print("Average {}: {}".format(model.metrics_names[i], np.mean(np.array(scores)[:, i])))
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

    

    #display(history_df)
for history in histories:

    plot_history(history)
final_model = model_definition()



# Compile the model

final_model.compile(loss=loss, optimizer=optimizer, metrics=["categorical_accuracy"])



# Train the model on the full train data

final_history = final_model.fit_generator(generator.flow(X, y, batch_size=batch_size),

                                          steps_per_epoch=len(X) / batch_size,

                                          epochs=epochs,

                                          verbose=0)
# Get the output of the last activation function

layer_name = 'dense_2'



# Define an intermediate model

intermediate_layer_model = Model(inputs=model.input,

                                 outputs=model.get_layer(layer_name).output)



# Calculate the values of the intermediate model

intermediate_output = intermediate_layer_model.predict(X)
PCA_transformer = PCA()

        

# Fit and transform

activation_7_PCA = PCA_transformer.fit_transform(intermediate_output)
%matplotlib inline

number_of_points = 308



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

    if not i % 10:

        ax.annotate(str(y_orig[i]), (x1[i], x2[i]))



plt.title('Last layer visualization using PCA 2D')

plt.show()
from mpl_toolkits.mplot3d import Axes3D



number_of_points = 308



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
augmented_data_batch = val_X.shape[0]

#X_aug, y_aug_real_unscaled = generator.flow(val_X, val_y, batch_size=augmented_data_batch).next()

X_aug = val_X

# Print examples when the model made bad decisions

y_aug_preds_unscaled = model.predict(X_aug)



# Inverse transform the predictions to the original scale

y_aug_preds = encoder.inverse_transform(y_aug_preds_unscaled)

y_aug_real = encoder.inverse_transform(val_y)
y_aug_all = pd.DataFrame([y_aug_preds[:,0], y_aug_real[:,0]]).transpose()

y_aug_all.columns = ["y predicted", "y real"]

print(y_aug_all.head(10))

plot_sample_images(X_aug[:10, :, :, 0], y_aug_real, 10, random=False)
pred_errors = y_aug_all[y_aug_all['y predicted'] != y_aug_all['y real']]

print(pred_errors)

total_errors = pred_errors.shape[0]

print(total_errors)



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
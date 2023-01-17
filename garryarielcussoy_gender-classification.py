# Import packages

import numpy as np

import pandas as pd



# Read the data

data_df = pd.read_csv("/kaggle/input/gender-classification/Transformed Data Set - Sheet1.csv")



# Take a look at some data examples

data_df.head(10)
# Describe the data

data_df.describe()
# Turn male into 1 and female 0

data_df['Gender'].replace(to_replace = 'F', value = 0, inplace = True)

data_df['Gender'].replace(to_replace = 'M', value = 1, inplace = True)
# Create one hot encoding

fav_color_df = pd.get_dummies(data_df[["Favorite Color"]], prefix = "color")

fav_music_df = pd.get_dummies(data_df[["Favorite Music Genre"]], prefix = "music")

fav_beverage_df = pd.get_dummies(data_df[["Favorite Beverage"]], prefix = "beverage")

fav_drink_df = pd.get_dummies(data_df[["Favorite Soft Drink"]], prefix = "drink")
# Merging one hot encoding and create new dataframe

transformed_df = pd.merge(fav_color_df, fav_music_df, left_index = True, right_index = True)

transformed_df = pd.merge(transformed_df, fav_beverage_df, left_index = True, right_index = True)

transformed_df = pd.merge(transformed_df, fav_drink_df, left_index = True, right_index = True)



# Take a look at some data examples

transformed_df.head(10)
# Choose feature (Manual)

feature = [

    "music_Electronic",

    "music_Hip hop",

    "music_Jazz/Blues",

    "music_Pop",

    "music_R&B and soul",

    "beverage_Vodka",

    "drink_Other"

]
# Choose all feature

# feature = []

# for col in transformed_df.columns:

#     feature.append(col)
# Choose feature (By rule)

# feature = []

# analyze_df = pd.merge(transformed_df, data_df["Gender"], left_index = True, right_index = True)

# for index, row in analyze_df.corr().iterrows():

#     if abs(row["Gender"]) > 0.08 and index != "Gender":

#         feature.append(index)
# Import packages related to training model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score



# Turn into numpy array

X = np.asarray(transformed_df[feature])

y = np.asarray(data_df['Gender'])



# Split dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# Preprocess train data

header = []

for col in transformed_df[feature].columns:

    header.append(col)

header = np.array(header)



x_df = pd.DataFrame(

    X_train,

    columns = header

)



y_df = pd.DataFrame(

    y_train,

    columns = ["gender"]

)



train_df = pd.merge(x_df, y_df, left_index = True, right_index = True)



# Look at the correlation

corr_df = train_df.corr()

corr_df.head(len(feature))
# Create logistic regression

LR = LogisticRegression().fit(X_train, y_train)
# Predict result

y_predict = LR.predict(X_test)



# Evaluate the accuracy

score = accuracy_score(y_predict, y_test)



# Print result

print("The accuracy is " + str(score))
# Using NN model

import tensorflow as tf

from tensorflow import keras



# Create callback

class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs = {}):

        if ((logs.get('val_accuracy') > 0.72 and logs.get('val_loss') <= 0.5931) or logs.get('val_accuracy') >= 0.9):

            self.model.stop_training = True

            print("Stop here")

callback = myCallback()



# Build model

tf.random.set_seed(42)

model = keras.Sequential([

    keras.layers.Dense(128, activation = 'relu', input_shape = [len(feature)]),

    keras.layers.Dropout(0.4),

    keras.layers.Dense(2, activation = 'softmax')

])



# Compile model

model.compile(

    loss = 'binary_crossentropy',

    optimizer = keras.optimizers.Adam(0.001),

    metrics = ['accuracy']

)



# Fit the model

model.fit(

    X_train, y_train,

    epochs = 200,

    batch_size = 1,

    verbose = 1,

    validation_split = 0.2,

    callbacks = [callback]

)
# Predict result (If the last layer using softmax)

y_predict = model.predict(X_test)

result = []

for index in range(len(y_predict)):

  each_result = np.argmax(y_predict[index])

  result.append(each_result)



# Formatting

result = np.array(result)

    

# Evaluate the accuracy

score = accuracy_score(result, y_test)



# Print result

print("The accuracy is " + str(score))
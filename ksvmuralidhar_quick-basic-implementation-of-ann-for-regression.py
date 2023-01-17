import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import tensorflow as tf

%matplotlib inline

pd.set_option("display.max_rows",None)
src = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",sep="\t",header=None,

                 names=["frequency","angle_of_attack","chord_length","fs_velocity","disp_thickness","sound_pressure"])

src.head()
src.describe()
src.isnull().sum() # Checking for missing values
src.plot(kind="box",figsize=(15,5)) # Plotting box plot for finding outliers
# Function to find outliers using IQR method / Box-whisker method

def find_outliers(x):

    q1 = x.quantile(0.25)

    q3 = x.quantile(0.75)

    iqr = q3 - q1

    minimum = q1 - (1.5 * iqr)

    maximum = q3 + (1.5 * iqr)

    return x[(x < minimum) | (x > maximum)]
#Dropping the outliers sice it is a quick and dirty way to bulid an ANN.

#Dropping the outliers is not the ideal way to do in practice.

for j in range(20):

    for col in src.columns:

        src.loc[src[col].isin(find_outliers(src[col])), col] = np.nan

    src.dropna(inplace=True)



#Box plot shows there are no outliers left

src.plot(kind="box",figsize=(15,5))
# Forming input and output arrays

X = src.iloc[:,:-1].values

y = src.iloc[:,-1].values
#Train, test split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=11)



print("Dataset shape: "+str(src.shape))

print("Training set shape: "+str(X_train.shape))

print("Test set shape: "+str(X_test.shape))
#It is prudent to apply the transforms separately on training and testing sets

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
np.random.seed(11) #setting seed to ensure model reproducibility

tf.random.set_seed(11)

#Building the ANN with some random structure

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=8,activation="relu")) # Hidden Layer 1 with 8 neurons and ReLU activation

ann.add(tf.keras.layers.Dense(units=8,activation="relu")) # Hidden Layer 2 with 8 neurons and ReLU activation

ann.add(tf.keras.layers.Dense(units=1)) # output layer

#since it is a regression problem, we need not use activation function in output layer



ann.compile(optimizer="adam",loss="mean_squared_error") #using MSE as loss function



#Fitting the model

model_log = ann.fit(X_train,y_train,epochs=500,batch_size=32) #model log helps to track the steps in model fitting process
#Plotting the loss function during the fitting process

pd.Series(model_log.history["loss"]).plot()

plt.title("Plotting the loss function per epoch")

plt.ylabel("loss")

plt.xlabel("epoch")
#Applying scaler to X_test

np.random.seed(11) #setting seed to ensure model reproducibility

tf.random.set_seed(11)

X_test = scaler.transform(X_test)

pred = ann.predict(X_test,batch_size=32)

pred = pred.flatten()

pred_df = pd.DataFrame({"Actual":y_test,"Prediction":pred}) # dataframe showing actual vs predictions

pred_df
pred_df.plot(kind="kde")
mean_squared_error(y_test,pred) #mean squared error of model on test set. Can alsp use model.evaluate()
np.random.seed(11) #setting seed to ensure model reproducibility

tf.random.set_seed(11)

mean_squared_error(y_train,ann.predict(X_train)) #mean squared error of model on training set 
#Since the error on training set and test set is close. It ensures the model is not overfit.

#This model was built in less than an hour. The model accuracy can be improved by tuning the hyperparameters and trying different model architectures
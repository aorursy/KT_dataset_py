#!/usr/bin/env python3 
# Author: Mbonu Chinedum 

# Description: Breast Cancer Analysis

# University: Nnamdi Azikiwe University 

# Date Created: 22/04/2020

# Date Modified: None 
# Importing the necessary modules 

import numpy as np 

import pandas as pd 

import seaborn as sns 

from collections import Counter

import matplotlib.pyplot as plt 

from xgboost import XGBClassifier 

from sklearn.utils import resample

from sklearn.decomposition import PCA

from tensorflow.keras.layers import Dense 

from sklearn.preprocessing import StandardScaler 

from tensorflow.keras.models import Sequential 

from sklearn.model_selection import StratifiedKFold 

from sklearn.model_selection import cross_val_score

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split
# Loading the dataset into memory using the pandas read_csv method 

# Specifying the path to the dataset 

dataset = "/kaggle/input/breast-cancer-wisconsin-data/data.csv"



# Reading the dataset into memory 

df = pd.read_csv(dataset, delimiter=",")

# Dropping the unnecessary columns 

df = df.drop("Unnamed: 32", axis = 1)



# Setting some basic parameters for the analysis 

%matplotlib inline 

plt.style.use('ggplot')



# Creating a new columns from the previous loaded dataframe 

df = df[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", 

        "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", 

        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", 

        "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",

        "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",

        "symmetry_worst", "fractal_dimension_worst", "diagnosis"]]



# Splitting the loaded dataframe into Input(X) and Output(y) features

# Splitting into input features and assigning it a variable X

X = df[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", 

        "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", 

        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", 

        "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",

        "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",

        "symmetry_worst", "fractal_dimension_worst"]]



# Splitting into output feature and assigning it a label variable y 

y = df["diagnosis"]



# Viewing the head of the filtered dataframe 

df.head() 
# Displaying the names of the columns 

# Converting the names of the columns for the loaded dataframe into a numpy array

ColumnNames = df.columns.values



# Displaying the names of the columns 

print(ColumnNames)
# Describing the head of the loaded dataframe 

df.describe()
# Displaying the shape of the input and output dataset 

print('Input Shape: {}'.format(X.shape))

print('Output Shape: {}'.format(y.shape))
# Getting the value count for the two labels in the diagnosis column

value_count = df['diagnosis'].value_counts() 



# Setting the figure size of the plot 

plt.figure(figsize = (18, 7))



# Plotting the Count for the value counts in the diagnosis column

value_count.plot(kind = "bar", color = "brown", rot=0)

plt.ylabel("Counts")

plt.title("A bar chart showing the count of Benign and Malignant Labels")

plt.grid(True)

plt.show() 



# Plotting a pie chart of the imbalanced dataset 

value_count.plot(kind = "pie", figsize=(18, 7))

plt.title("A Pie Chart showing the count of Benign and Malignant Labels")

plt.show() 



# Printing the number of counts for the values of the labels in the diagnosis column 

B, M = value_count 

print("Number of Benign: {}".format(B))

print("Number of Malignant: {}".format(M))
# Displaying a violin plot for the first ten features 

labels = y 

input_features = X



# Normalizing the dataframe 

dataN2 = (input_features - input_features.mean()) / (input_features.std())

input_features = pd.concat([y, dataN2.iloc[:, 0:10]], axis = 1)



data = pd.melt(input_features, id_vars = "diagnosis", var_name = "features", 

              value_name = "value")



# Plotting the first Ten feature

plt.figure(figsize = (18, 7))

sns.violinplot(x = "features", y = "value", hue = "diagnosis", data = data, split = True, 

              inner = "quart")

plt.xticks(rotation = 90)

plt.grid(True)

plt.show() 
# Displaying a violin plot for the second 10 features 

data = pd.concat([y, dataN2.iloc[:,10:20]], axis=1)

data = pd.melt(data, id_vars = "diagnosis",

                    var_name = "features",

                    value_name = 'value')



# Plotting the Violin plot 

plt.figure(figsize=(18, 7))

sns.violinplot(x = "features", y = "value", hue = "diagnosis", data = data, split = True, inner = "quart")

plt.xticks(rotation=90)

plt.grid(True)

plt.show() 
# Displaying a violin plot for the third Ten features 

data = pd.concat([y, dataN2.iloc[:,20:31]], axis = 1)

data = pd.melt(data, id_vars = "diagnosis",

                    var_name = "features",

                    value_name = 'value')



# Plotting the violin plot for the third ten features

plt.figure(figsize=(18, 7))

sns.violinplot(x = "features", y = "value", hue = "diagnosis", data = data, split = True, inner = "quart")

plt.xticks(rotation=90)

plt.grid(True)

plt.show() 
# Displaying a Correlation matrix of all the features of the breast cancer dataset.

f,ax = plt.subplots(figsize=(20, 18))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
# Plotting a Boxplot for the loaded dataframe 

boxplot = df.boxplot(by="diagnosis", figsize=(19, 9))

plt.show() 
# Plotting a boxplot of the "Area_mean" and "Area_worst"

boxplot = df.boxplot(column = ["area_mean", "area_worst"], by="diagnosis", 

                    layout = (2, 1), figsize=(19, 9))

# Showing the boxplot

plt.show() 
# Setting the figure of the Graph 

plt.figure(figsize=(18, 7))



# Plotting a scatter plot of diagnosis against area_mean 

plt.scatter(df['diagnosis'], df['area_mean'])



# Displaying the graph 

plt.show() 
# Setting the figure of the Graph 

plt.figure(figsize=(18, 7))



# Plotting a scatter plot of diagnosis against area_worst

plt.scatter(df['diagnosis'], df['area_worst'])



# Displaying the graph 

plt.show() 
# Setting the figure of the Graph 

plt.figure(figsize=(18, 7))



# Plotting a scatter plot of diagnosis against radius_mean

plt.scatter(df['diagnosis'], df['radius_mean'])



# Displaying the graph 

plt.show()
# Transforming the diagnosis column into a binary classification column 

# where by 0 == "B"(Benign), and 1 == "M"(Malignant)

df["diagnosis"] = [1 if b == "M" else 0 for b in df["diagnosis"]]



# Checking the value counts for the Diagnosis column 

df["diagnosis"].value_counts() 



# Remember that 1 == Malignant, And 0 == Benign Type of Cancer 
# Balancing the imbalanced dataset 

# Separate majority and minority classes 

df_majority = df[df["diagnosis"] == 0]

df_minority = df[df["diagnosis"] == 1]



# Upsample the minority class sample with replacement to 

# Match the majority class reproducible results 

df_minority_upsampled = resample(df_minority, 

                                replace = True, 

                                n_samples = 357, 

                                random_state = 123)



# Combine the majority class with upsampled minority class 

Balanced_data = pd.concat([df_majority, df_minority_upsampled])



# Displaying the new class distribution 

counter = Counter(Balanced_data["diagnosis"])

print(counter)
# Getting the value counts for the values in the diagnosis columns for the 

# Balanced dataset 

value_count = Balanced_data['diagnosis'].value_counts() 



# Setting the figure size of the plot 

plt.figure(figsize = (18, 7))



# Plotting the Count for the value counts in the diagnosis column of the balanced data

value_count.plot(kind = "bar", color = "brown")

plt.ylabel("Counts")

plt.title("A bar chart showing the count of Benign and Malignant Labels")

plt.grid(True)

plt.show() 



# Plotting a pie chart of the imbalanced dataset 

value_count.plot(kind = "pie", figsize=(18, 7))

plt.title("A Pie Chart showing the count of Benign and Malignant Labels")

plt.show() 





# Printing the number of counts for the values of the labels in the diagnosis column 

B, M = value_count 

print("Number of Benign: {}".format(B))

print("Number of Malignant: {}".format(M))
# Splitting the Balanced dataframe into input and output features 

X = Balanced_data[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", 

        "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", 

        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", 

        "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",

        "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",

        "symmetry_worst", "fractal_dimension_worst"]]



# Splitting into output feature and assigning it a label variable y 

y = Balanced_data["diagnosis"]
# Split the Balanced dataset into train test split 

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                   test_size = 0.3, 

                                                   random_state = 42)



# Performing Normalization the the Training and Test set 

X_train_normalized = (X_train - X_train.mean()) / (X_train.max() - X_train.min())

X_test_normalized = (X_test - X_test.mean()) / (X_test.max() - X_test.min())



# Creating the Principal Component Class model 

pca_model = PCA() 

pca_model.fit(X_train_normalized)



# Plotting the Features Extracted 

plt.figure(1, figsize=(18, 7))

plt.clf()

plt.axes([.2, .2, .7, .7])

plt.plot(pca_model.explained_variance_ratio_, linewidth=2)

plt.axis('tight')

plt.xlabel('n_components')

plt.ylabel('explained_variance_ratio_')
### MORE ON PRINCIPAL COMPONENT ANALYSIS

# Creating a standard scaler class

scaler = StandardScaler() 



# Fitting the balanced dataframe on the scaler class 

scaler.fit(df)



# Transforming the scaled data 

scaled_data = scaler.transform(Balanced_data)



# Building and fitting the PCA model 

pca = PCA(n_components = 2) 

pca.fit(scaled_data)
# Transforming the data into its first 2 principal components 

x_pca = pca.transform(scaled_data)



# Displaying the shape of the scaled data 

print(scaled_data.shape)



# Displaying the shape of the X_pca data 

print(x_pca.shape)
# Plotting the Graph of the first and second principal components

plt.figure(figsize=(18, 8))

plt.scatter(x_pca[:,0],x_pca[:,1], c=Balanced_data['diagnosis'],cmap='rainbow')

plt.xlabel('First principal component')

plt.ylabel('Second Principal Component')
# Displaying the shape of the data 

print("Input Shape: {}, {}".format(X_train.shape, X_test.shape))

print("Output Shape: {}, {}".format(y_train.shape, y_test.shape))
# Fitting the model on training data 

model = XGBClassifier() 

model.fit(X_train, y_train)
# Evualting the model to find how accurate it performs on it predictons 

predictions = model.score(X_test, y_test)

print("The XGBoost Model is: {:.2f} accurate".format(predictions * 100))
# Creating a function to create the model, required for the KerasClassifier 

def create_model():

    # Creating the model 

    model = Sequential()

    model.add(Dense(120, input_dim=X_train.shape[1], activation='relu'))

    model.add(Dense(32, activation="relu"))

    model.add(Dense(8, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))

    # Compiling the model 

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Returning the model 

    return model 



# Creating the model from the defined function 

model = create_model()
# Training the model 

history = model.fit(X_train, y_train, epochs=200, batch_size=4, validation_data=(X_test, y_test))
# Displaying the accuracy of the model 

Accuracy = model.evaluate(X_test, y_test)[1] * 100

Accuracy = str(Accuracy)[:5]

!echo 

print('The model is {}% Accurate.'.format(Accuracy[:5]))
# showing a plot of how Accurate the model is against the 

# test dataset

plt.figure(figsize=(18, 7))

plt.grid(True)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()
# showing a plot of the loss with respect to the number of epochs 

plt.figure(figsize=(18, 7))

plt.grid(True)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper right')

plt.show() 
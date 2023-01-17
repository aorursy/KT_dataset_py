import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sns

%matplotlib inline
def load_dataset(path):

    '''

        Loading Dataset

    '''

    return pd.read_csv(path)
def preprocess(dataset):

    '''

        Preprocssing the Dataset

    '''

    Y = dataset["Type"]

    X = dataset.drop(['Type'],axis=1)

    X = X.values.reshape(-1,151,332,1)

    X = X/255.0

    

    return X,Y
def display(n,label):

    '''

        Displaying images in grid of 1xn

    '''

    fig = plt.figure(figsize=(20,20))

    label_index = np.where(np.array(Y) == label)

    for index in range(n):

        i = label_index[0][index]

        ax = fig.add_subplot(1, n, index+1, xticks=[], yticks=[])

        ax.imshow(X[i].reshape(151,332), cmap='gray')

        ax.set_title(str(Y[i]))
dataset_path = "../input/eye-disorder-dataset/eye_dataset.csv"

dataset = load_dataset(dataset_path)

X,Y = preprocess(dataset)
print("Shape of X : ",X.shape)

print("Shape of Y : ",Y.shape)

print("Shape of Image : ", X[0].shape)
sns.countplot(Y)
display(5,"cat")

display(5,"crossed")

display(5,"bulk")
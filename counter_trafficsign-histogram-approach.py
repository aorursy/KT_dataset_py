# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input/germantrafficsigndataset/GermanTrafficSignDatasetWithBackdoorImages/CleanDataset/train"))

from PIL import Image

import random

import matplotlib.pyplot as plot

# Any results you write to the current directory are saved as output.
def GenerateDF(path, dummy_classes=['Pedestrian', 'Parking', 'SpeedLimit', 'DoNotEnter', 'GiveWay', 'Stop', 'TurnRight']):

    classes = dummy_classes

    

    class_list = []

    path_list = []

    for c in classes:

        for file in os.listdir(os.path.join(path, c)):

            total_path = os.path.join(os.path.join(path, c), file)

            path_list.append(total_path)

            class_list.append(c)

            

    class_list = pd.Series(class_list)

    class_list = pd.get_dummies(class_list, columns=dummy_classes)

    path_list = pd.Series(path_list).rename("path")

    

    return pd.concat([path_list, class_list], 1)
train_data = GenerateDF("../input/germantrafficsigndataset/GermanTrafficSignDatasetWithBackdoorImages/CleanDataset/train").sample(frac=1).reset_index().drop("index", 1)

print(train_data.shape)

train_data.head()
#val_data = train_data.sample(frac=0.2)

#train_data = train_data.drop(val_data.index)

#print(train_data.shape)
test_data = GenerateDF("../input/germantrafficsigndataset/GermanTrafficSignDatasetWithBackdoorImages/CleanDataset/test").sample(frac=1).reset_index().drop("index", 1)

print(test_data.shape)

test_data.head()
def LoadImage(path, augmentation=False):

    img = Image.open(path).resize((35,35))

    

    if augmentation:

        angle = (random.random()*2-1)*15.0

        img = img.rotate(angle)

    img = np.array(img)

    

    img = img-np.min(img)

    img = img/np.max(img)*255

    

    

    return np.round(img,0).astype("int")



img = LoadImage(train_data.sample(1).values[0][0])

print(img.shape)

plot.imshow(img)
def CalcHist(img, steps=2):

    hist, bins = np.histogram(img.ravel(), np.arange(0,256,steps),[0,256])

    hist = hist-np.min(hist)

    hist = hist/np.max(hist)

    return hist

    

plot.plot(CalcHist(img))
def MapHistogram(path):

    img = LoadImage(path)

    return CalcHist(img)
train_data = pd.concat([train_data, train_data.path.map(MapHistogram).rename("hist")], 1)

print(1)

test_data = pd.concat([test_data, test_data.path.map(MapHistogram).rename("hist")], 1)
def PrepareForTraining(df):

    arrs = []

    sols = []

    for i, row in df.iterrows():

        arrs.append(row["hist"])

        sols.append(np.argmax(row.drop(["path", "hist"]).values))

    return np.array(arrs), np.array(sols)



X_train, Y_train = PrepareForTraining(train_data)

print(X_train.shape)

print(Y_train.shape)



X_test, Y_test = PrepareForTraining(test_data)

print(X_test.shape)

print(Y_test.shape)

from sklearn.metrics import confusion_matrix, accuracy_score

import seaborn as sns



def Evaluate(model, name):

    model.fit(X_train, Y_train)

    

    p = model.predict(X_test)

    

    acc = accuracy_score(Y_test, p)

    M = confusion_matrix(Y_test, p)

    

    print("Accuracy: "+str(acc))

    

    

    plot.figure(figsize=(8,6))

    plot.title(name)

    sns.heatmap(M, annot=True, fmt='g')

    plot.xticks(np.arange(M.shape[0]), ['Parking', 'SpeedLimit', 'Pedestrian', 'TurnRight', 'Stop', 'GiveWay', 'DoNotEnter', 'CycleTrack'], rotation=90)

    plot.yticks(np.arange(M.shape[0]), ['Parking', 'SpeedLimit', 'Pedestrian', 'TurnRight', 'Stop', 'GiveWay', 'DoNotEnter', 'CycleTrack'], rotation=0)

    plot.xlabel("Predicted")

    plot.ylabel("True")

    plot.savefig("confusion_"+str(name)+".png")

    plot.show()

    
from sklearn.linear_model import LogisticRegression



lreg = LogisticRegression()

Evaluate(lreg, "LogisticRegression")
from sklearn.ensemble import RandomForestClassifier



RF = RandomForestClassifier()

Evaluate(RF, "RandomForest")
from sklearn.neighbors import KNeighborsClassifier as KNN



knn = KNN(5)

Evaluate(knn, "KNN")
from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation="relu")

Evaluate(mlp, "MLP")
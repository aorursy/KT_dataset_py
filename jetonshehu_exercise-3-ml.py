# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import matplotlib.pyplot as plot

import os

from PIL import Image

import random

import matplotlib.pyplot as plot

import cv2

from sklearn.cluster import MiniBatchKMeans

import time



print(os.listdir("../input/germantrafficsigndatasetwithbackdoorimages/GermanTrafficSignDatasetWithBackdoorImages/CleanDataset/train"))



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
train_data = GenerateDF("../input/germantrafficsigndatasetwithbackdoorimages/GermanTrafficSignDatasetWithBackdoorImages/CleanDataset/train").sample(frac=1).reset_index().drop("index", 1)

print(train_data.shape)

train_data.head()
test_data = GenerateDF("../input/germantrafficsigndatasetwithbackdoorimages/GermanTrafficSignDatasetWithBackdoorImages/CleanDataset/test").sample(frac=1).reset_index().drop("index", 1)

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
#Concatenate histogram to the train and test data

train_data_hist = pd.concat([train_data, train_data.path.map(MapHistogram).rename("hist")], 1)

test_data_hist = pd.concat([test_data, test_data.path.map(MapHistogram).rename("hist")], 1)
def PrepareForTrainingHist(df):

    arrs = []

    sols = []

    for i, row in df.iterrows():

        arrs.append(row["hist"])

        sols.append(np.argmax(row.drop(["path", "hist"]).values))

    return np.array(arrs), np.array(sols)
#Split Predictors and Classes

X_train, Y_train = PrepareForTrainingHist(train_data_hist)

print(X_train.shape)

print(Y_train.shape)



X_test, Y_test = PrepareForTrainingHist(test_data_hist)

print(X_test.shape)

print(Y_test.shape)
#Create df out of cm

def cm2df(cm, labels):

    df = pd.DataFrame()

    # rows

    for i, row_label in enumerate(labels):

        rowdata={}

        # columns

        for j, col_label in enumerate(labels): 

            rowdata[col_label]=cm[i,j]

        df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))

    return df[labels]
from sklearn.metrics import confusion_matrix, accuracy_score

import seaborn as sns



def Evaluate(model, name):

    model.fit(X_train, Y_train)

    

    p = model.predict(X_test)

    

    acc = accuracy_score(Y_test, p)

    M = confusion_matrix(Y_test, p)

    cm_as_df=cm2df(M,['Parking', 'SpeedLimit', 'Pedestrian', 'TurnRight', 'Stop', 'GiveWay', 'DoNotEnter'])

    

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

    

    return cm_as_df
from sklearn.linear_model import LogisticRegression



lreg = LogisticRegression()

cm_le_hist = Evaluate(lreg, "LogisticRegression_HIST")
from sklearn.ensemble import RandomForestClassifier



RF = RandomForestClassifier()

cm_rf_hist = Evaluate(RF, "RandomForest_HIST")
from sklearn.neighbors import KNeighborsClassifier as KNN



knn = KNN(5)

cm_knn_hist = Evaluate(knn, "KNN_HIST")
from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation="relu")

cm_mlp_hist = Evaluate(mlp, "MLP_HIST")
def LoadImageKA(path):

    

    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img,(35,35),interpolation=cv2.INTER_AREA)



    img = img-np.min(img)

    img = img/np.max(img)*255

    img = img.astype(np.uint8)

    

    



    return img
def extract_features(img, vector_size=35):

    # Using Kaze

    alg = cv2.KAZE_create()

    

    desc_size = []

    kps_size = []

    

    #Getting image keypoints

    kps = alg.detect(img)

    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]

    

    kps, dsc = alg.compute(img, kps)

    #Check if image has descriptors

    if dsc is None:

        return None

    else:

        dsc = dsc.flatten()

        needed_size = (vector_size * 64)

        if dsc.size < needed_size:



            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])

    

    

        return dsc
def map_descriptors(path):

    img = LoadImageKA(path)

    return extract_features(img)
display("Train Data")

train_data_ka = pd.concat([train_data, train_data.path.map(map_descriptors).rename("kaze")], 1)

train_data_ka.dropna(inplace=True)

display(train_data_ka.head())

display("Test Data")

test_data_ka = pd.concat([test_data, test_data.path.map(map_descriptors).rename("kaze")], 1)

test_data_ka.dropna(inplace=True)

test_data_ka.head()
def PrepareForTrainingKA(df):

    arrs = []

    sols = []

    for i, row in df.iterrows():

        arrs.append(row["kaze"])

        sols.append(np.argmax(row.drop(["path", "kaze"]).values))

    return np.array(arrs), np.array(sols)
X_train, Y_train = PrepareForTrainingKA(train_data_ka)

print(X_train.shape)

print(Y_train.shape)



X_test, Y_test = PrepareForTrainingKA(test_data_ka)

print(X_test.shape)

print(Y_test.shape)
from sklearn.linear_model import LogisticRegression



lreg = LogisticRegression()

cm_lr_ka = Evaluate(lreg, "LogisticRegression_FE")
from sklearn.ensemble import RandomForestClassifier



RF = RandomForestClassifier()

cm_rf_ka = Evaluate(RF, "RandomForest_FE")
from sklearn.neighbors import KNeighborsClassifier as KNN



knn = KNN(5)

cm_knn_ka = Evaluate(knn, "KNN_FE")
from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation="relu")

cm_mlp_ka = Evaluate(mlp, "MLP_FE")
def read_and_clusterize(file_images, num_cluster):



    kaze_keypoints = []



    for path in file_images.path:

        #read image

        img = cv2.imread(path,1)

        # Convert them to grayscale

        img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img,(35,35),interpolation=cv2.INTER_AREA)



        img = img-np.min(img)

        img = img/np.max(img)*255

        img = img.astype(np.uint8)

        # KAZE extraction

        kaze = cv2.KAZE_create()

        kp, descriptors = kaze.detectAndCompute(img,None)

        #append the descriptors to a list of descriptors

        if descriptors is not None:

            kaze_keypoints.append(descriptors)



    kaze_keypoints=np.asarray(kaze_keypoints)

    kaze_keypoints=np.concatenate(kaze_keypoints, axis=0)

    #with the descriptors detected,we create clusters

    kmeans = MiniBatchKMeans(n_clusters=num_cluster, random_state=0).fit(kaze_keypoints)



    return kmeans
def build_histogram(descriptor_list, cluster_alg):

    histogram = np.zeros(len(cluster_alg.cluster_centers_))

    cluster_result =  cluster_alg.predict(descriptor_list)

    for i in cluster_result:

        histogram[i] += 1.0

    return histogram
def calculate_centroids_histogram(file_images, model, num_clusters):



    feature_vectors=[]

    class_vectors=[]





    for i, row in file_images.iterrows():

        #read image

        image = cv2.imread(row.path,1)

        #Convert them to grayscale

        image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image,(35,35),interpolation=cv2.INTER_AREA)

        #Kaze extraction

        kaze = cv2.KAZE_create()

        kp, descriptors = kaze.detectAndCompute(image, None)

        if descriptors is not None:

            #classification of all descriptors in the model

            predict_kmeans=model.predict(descriptors)

            #calculates the histogram

            hist = build_histogram(descriptors, model)

            #histogram is the feature vector

            feature_vectors.append(hist)

            #define the class of the image

            class_sample = np.argmax(row.drop(["path", "kaze"]).values)

            class_vectors.append(class_sample)



    feature_vectors=np.asarray(feature_vectors)

    class_vectors=np.asarray(class_vectors)

    #return vectors and classes we want to classify

    return class_vectors, feature_vectors
def create_bovw(num_clusters):

    model= read_and_clusterize(train_data_ka, num_clusters)

    [train_class,train_featvec] = calculate_centroids_histogram(train_data_ka,model, num_clusters)

    [test_class,test_featvec] = calculate_centroids_histogram(test_data_ka,model, num_clusters)

    

    return train_featvec,train_class, test_featvec, test_class, 
start = time.time()

X_train, Y_train, X_test, Y_test = create_bovw(800)

print("Training time: "+str(time.time()-start))
from sklearn.linear_model import LogisticRegression



start = time.time()

lreg = LogisticRegression()

cm_lr_bovw = Evaluate(lreg, "LogisticRegression_BOVW")

print("Logistic Regression Prediction time: "+str(time.time()-start))
from sklearn.ensemble import RandomForestClassifier



start = time.time()

RF = RandomForestClassifier()

cm_rf_bovw = Evaluate(RF, "RandomForest_BOVW")

print("Random Forest Prediction time: "+str(time.time()-start))
from sklearn.neighbors import KNeighborsClassifier as KNN



start = time.time()

knn = KNN(5)

cm_knn_bovw = Evaluate(knn, "KNN_BOVW")

print("KNN Prediction time: "+str(time.time()-start))
from sklearn.neural_network import MLPClassifier



start = time.time()

mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation="relu")

cm_mlp_bovw = Evaluate(mlp, "MLP_BOVW")

print("MLP Prediction time: "+str(time.time()-start))
val_data = train_data.sample(frac=0.2)

train_data_dl = train_data.drop(val_data.index)

print(train_data_dl.shape)
def LoadImageDL(path, augmentation=True):

    img = Image.open(path).resize((35,35))

    

    if augmentation:

        angle = (random.random()*2-1)*15.0

        img = img.rotate(angle)

    img = np.array(img)

    img = img-np.min(img)

    img = img/np.max(img)

    

    return img



img = LoadImageDL(train_data_dl.sample(1).values[0][0])

print(img.shape)

plot.imshow(img)



def Generate(data, batch_size=15):

    while True:

        batch = data.sample(batch_size)

        imgs = []

        classes = []

        for i, row in batch.iterrows():

            try:

                img = LoadImageDL(row["path"]).astype("float")

                c = row.drop("path").values.astype("float")



                imgs.append(img)

                classes.append(c)

            except Exception as e:

                print(e)

        yield np.array(imgs), np.array(classes)

        

gen = Generate(train_data_dl)

imgs, sols = next(gen)

print(imgs.shape)

print(sols.shape)

print(sols)
from keras.models import Model, load_model

from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, LeakyReLU, Dropout

from keras.optimizers import SGD, Adam

from keras.callbacks import ModelCheckpoint



inp = Input(shape=(35,35,3))

x = BatchNormalization()(inp)

x = Conv2D(256, 5, activation="relu")(x)

x = MaxPooling2D()(x)

x = LeakyReLU()(x)

x = Conv2D(128, 3, activation="relu")(x)

x = MaxPooling2D()(x)

x = Conv2D(64, 3, activation="relu")(x)

x = Flatten()(x)

x = Dropout(0.5)(x)

x = Dense(7, activation="softmax")(x)







model = Model(inp, x)

model.compile(optimizer=SGD(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])



model.summary()
from keras.utils import plot_model

plot_model(model, to_file='model.png')
train_gen = Generate(train_data_dl)

val_gen = Generate(val_data)
clb = [ModelCheckpoint("best.h5", save_best_only=True, verbose=0)]

start = time.time()

h = model.fit_generator(train_gen, epochs=200, steps_per_epoch=10, validation_data=val_gen, validation_steps=5, verbose=1, callbacks=clb)
print("Training time: "+str(time.time()-start))
plot.figure(figsize=(8,6))

plot.plot(h.history["loss"], label="loss")

plot.plot(h.history["val_loss"], label="val. loss")

plot.xlabel("Epoch")

plot.ylabel("Loss")

plot.tight_layout()

plot.savefig("loss.png")

plot.show()
imgs = []

classes = []

for i, row in test_data.iterrows():

    img = LoadImage(row["path"]).astype("float")

    c = row.drop("path").values.astype("float")

    print(c)



    imgs.append(img)

    classes.append(c)



imgs = np.array(imgs)

classes = np.array(classes)

print(imgs.shape)

print(classes.shape)

print(np.sum(classes,0))
start = time.time()

p = model.predict(imgs)

print("Prediction time: "+str(time.time()-start))

p = np.argmax(p, axis=1)



Y_true = np.argmax(classes, axis=1)

print(Y_true)
cols = test_data.columns[1:]
from sklearn.metrics import confusion_matrix

import seaborn as sns

M = confusion_matrix(Y_true, p)

print(M)



plot.figure(figsize=(8,6))

sns.heatmap(M, annot=True, fmt='g')

plot.xticks(np.arange(M.shape[0]), cols, rotation=90)

plot.yticks(np.arange(M.shape[0]), cols, rotation=0)

plot.xlabel("Predicted")

plot.ylabel("True")

plot.savefig("confusion.png")

plot.show()
from sklearn.metrics import accuracy_score as acc



print(acc(Y_true, p))
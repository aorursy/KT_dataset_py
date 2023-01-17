#Python 3 environment
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import copy

#Data visualization
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import statsmodels as sm
import seaborn as sns
import tensorflow as ts
import keras
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.model_selection import train_test_split


#Added from external 
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error

#For Neural Nets
import tensorflow 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten

import os
print(os.listdir("../input"))
#Load Data
with                 open ('../input/HAM10000_metadata.csv') as metadata,   \
                     open ('../input/hmnist_28_28_L.csv')    as images,     \
                     open ('../input/hmnist_28_28_RGB.csv')  as colorImages:
    metadataDF = pd.read_csv(metadata)
    imageDF = pd.read_csv(images)  
    colorImageDF = pd.read_csv(colorImages)

#Drop empty metadata
colorImageDF = colorImageDF[metadataDF.sex != 'unknown']
colorImageDF = colorImageDF[metadataDF.age != 0]
colorImageDF = colorImageDF[metadataDF.localization != 'unknown']
colorImageDF = colorImageDF.dropna(axis=0)

imageDF = imageDF[metadataDF.sex != 'unknown']
imageDF = imageDF[metadataDF.age != 0]
imageDF = imageDF[metadataDF.localization != 'unknown']
imageDF = imageDF.dropna(axis=0)


metadataDF = metadataDF[metadataDF.sex != 'unknown']
metadataDF = metadataDF[metadataDF.age != 0]
metadataDF = metadataDF[metadataDF.localization != 'unknown']
metadataDF = metadataDF.dropna(axis=0)

cancerTypes = {"akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "nv": "Melanocytic Nevi",
    "mel": "Melanoma",
    "vasc": "Vascular Skin Lesions"}

countTypes = Counter(metadataDF['dx'])
plt.title("Count of Skin Lesions in Data (before)")
plt.bar([cancerTypes[x] for x in countTypes.keys()], countTypes.values())
plt.xticks(rotation='vertical')
plt.show()
colorImageBalancedDF = colorImageDF.drop(colorImageDF[colorImageDF.label == 4].iloc[:5000].index)
imageBalancedDF = imageDF.drop(imageDF[imageDF.label == 4].iloc[:5000].index)
metadataBalancedDF = metadataDF.drop(metadataDF[metadataDF.dx == 'nv'].iloc[:5000].index)
countTypes = Counter(metadataBalancedDF['dx'])
plt.title("Count of Skin Lesions in Data (after)")
plt.bar([cancerTypes[x] for x in countTypes.keys()], countTypes.values())
plt.xticks(rotation='vertical')
plt.show()
#Data Visualization of Color
colorResponse = colorImageDF.label
colorPixels = colorImageDF.drop(['label'],axis=1)
colorPixels = colorPixels.values.reshape((7631456,3))
reds,greens,blues = map(list, zip(*colorPixels))
#Visualize the distributions
alpha = .3
colors = [(reds,'red',"Reds"),(greens,'green',"Greens"),(blues,'blue',"Blues")]

for pixels,color,label in colors:
    plt.hist(pixels,color=color,label=label,alpha=alpha)

plt.title("Distribution of Color Pixels in RGB Lesion Images")

plt.legend()
plt.show()
#Show just reds
newReds = [[]]*7
for i,r in enumerate(colorResponse): 
    newReds[r].append(reds[i])
plt.hist(newReds[2],color='red',label="ok",alpha=alpha)
plt.show()
#Means is the mean of reds and greens for each image
cancerType = colorImageBalancedDF['label']
colorXValues = colorImageBalancedDF.drop(['label'], axis=1)

meansRed = []
meansGreen = []
meansBlue = []
for row in colorXValues.values:
    rgb = np.array(row.reshape((784,3)))
    meansRed.append(np.mean(rgb[0]))
    meansGreen.append(np.mean(rgb[1]))
    meansBlue.append(np.mean(rgb[2]))
plt.figure(figsize=(15,8))
sns.scatterplot(meansRed, meansGreen, hue=cancerType)
plt.title('Means of Red and Green Pixel Values')
plt.show()
#Create lists for later use
diagnosis = {'bkl':0, 'nv':1, 'df':2, 'mel':3, 'vasc':4, 'bcc':5, 'akiec':6}
diagnosisColumns = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']
sex = {'male': 1,'female': 0}
localization = {'scalp': 0,'ear': 1,'face': 2,'back': 3,'trunk': 4,
                'chest': 5,'upper extremity': 6,'abdomen': 7,'lower extremity': 8,
                'genital': 9,'neck': 10,'hand': 11,'foot': 12,'acral': 13}

#split data by response variable
responseColor = colorImageBalancedDF['label']
colorImageBalancedValuesDF = colorImageBalancedDF.drop(['label'], axis=1)
x_train_color, x_test_color, y_train_color, y_test_color = train_test_split(colorImageBalancedValuesDF,
                                                    responseColor,
                                                    test_size=.3, 
                                                    random_state=1, 
                                                    stratify=responseColor)
Counter(y_train_color)
#split data by response variable
response = metadataBalancedDF['dx']
# we choose a test size of 0.2
x_train, x_test, y_train, y_test = train_test_split(metadataBalancedDF,
                                                    response,
                                                    test_size=.3, 
                                                    random_state=1, 
                                                    stratify=metadataBalancedDF['dx'])
#Create dummy variables -> One Hot Vector
x_train = pd.get_dummies(x_train.sex).join(pd.get_dummies(x_train.localization)).join(x_train.age)
x_test = pd.get_dummies(x_test.sex).join(pd.get_dummies(x_test.localization)).join(x_test.age)
x_train.shape
# take out acral localization
metadataBalancedDF = metadataBalancedDF[metadataBalancedDF.localization != 'acral']
sex_columns = ['male','female']
localization_columns = ['scalp','ear','face','back','trunk',
                        'chest','upper extremity','abdomen','lower extremity',
                        'genital','neck','hand','foot']
metadata_columns = ['lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization']
columns = list(sex_columns)
columns.extend(localization_columns)
columns.append('age')

x_train_log = x_train[columns]
x_test_log = x_test[columns]

model = LogisticRegressionCV(
                        Cs=list(np.power(10.0, np.arange(-10, 10))),
                        max_iter=10000,
                        fit_intercept=True,
                        cv=5
)
model.fit(x_train_log,y_train)

#Predict
y_pred_train = model.predict(x_train_log)
y_pred_test = model.predict(x_test_log)

#Perfromance Evaluation
train_score = accuracy_score(y_train, y_pred_train)*100
test_score = accuracy_score(y_test, y_pred_test)*100

print("Training Set Accuracy:",str(train_score)+'%')
print("Testing Set Accuracy:",str(test_score)+'%')
#Show the coefficients
coefs = pd.DataFrame(model.coef_.T, columns=diagnosisColumns)
rows = list(sex_columns)
rows.extend(localization_columns)
rows.append('age')
coefs['Predictor'] = rows
print("Coefficients from model:")
display(coefs)
#Visualizing image greyscale 
testImage = np.array([imageDF.iloc[0][:784]])
testImage = testImage.reshape((28,28))
testImage.shape
plt.imshow(testImage)
plt.show()
#Visualizing RGB image with python plt
testImage = np.array([colorImageDF.iloc[0]])
testImage = testImage[0][:2352]
testImage = testImage.reshape((28,28,3))
plt.imshow(testImage)
plt.show()
#Visualizing full image from file
filepath = "../input/ham10000_images_part_1/ISIC_0027419.jpg"
image = mpimg.imread(filepath)
plt.imshow(image)
plt.show()
class KMeansClassifier(object):
    """ K-Means Classifier Class """

    def __init__(self, K, centers, points):
        """
        Initialize classifier

        :param K: number of clusters
        :param centers: np list of centers (length K)
        :param points: np list of points
        """
        self.K = K
        self.centers = centers
        self.points = points
        self.count = 0

        # clusters[i] contains all points assigned to it
        self.clusters = [[] for _ in range(K)]

    def fit(self):
        """
        Fit the points to the K clusters
        """
        i = 0
        while True:
            prev_centers = copy.deepcopy(self.centers)
            self.clusters = [[] for _ in range(self.K)]

            self.assign_clusters()
            self.update_centers()

            # Stop once we have converged
            if prev_centers == self.centers:
                break
            i += 1
            self.count += 1
            if self.count == 30: 
                break

    def assign_clusters(self):
        """
        TODO

        Assign each point to the closest cluster, based on
        the center position of that cluster.

        Hints: Look into np.linalg.norm and np.subtract
        """
        for point in self.points: 
            minDistance = float("Inf")
            centerIndex = None
            for i,center in enumerate(self.centers): 
                distance = np.linalg.norm(np.subtract(point,center))
                if distance < minDistance: 
                    minDistance = distance
                    centerIndex = i
            self.clusters[centerIndex].append(point)
        
    def update_centers(self):
        """
        TODO

        Update the cluster centers based on the mean of all
        points assigned to it.
        """
        for i,cluster in enumerate(self.clusters): 
            first = [c[0] for c in cluster]
            second = [c[1] for c in cluster]
            #third = [c[2] for c in cluster]
            
            self.centers[i] = [np.mean(first), np.mean(second)] #,np.mean(third)]
        
    def classify(self, point):
        pass
    def error(self):
        """
        TODO

        Implement the K-Means error function, sum of squared distance between
        all points and their assigned center.

        Hints: Like assign_cluster, look into np.linalg.norm and np.subtract.
               Make sure to square each distance!
        """
        error = 0
        for i,cluster in enumerate(self.clusters): 
            for point in cluster: 
                center = self.centers[i]
                distanceSquared = np.linalg.norm(np.subtract(point,center))**2
                error += distanceSquared
        return error

def makeMeans(k,points,centers):
    print("computing kmeans for k=", k);

    kmeans = KMeansClassifier(k, centers, points)
    kmeans.fit()
    print('Error: {}'.format(kmeans.error()))
    plt.figure(figsize=(17,12))
    for cluster in kmeans.clusters:
        try:
            plt.plot(list(zip(*cluster))[0], list(zip(*cluster))[1], 'o')
        except:
            continue
    plt.plot(list(zip(*kmeans.centers))[0], list(zip(*kmeans.centers))[1], 'x')
    plt.title('KMeans with k=' + str(k))
    plt.savefig('kmeans.png')
    plt.show()
    print(kmeans.centers)
meansTrain = []
for row in x_train_color.values:
    rgb = np.array(row.reshape((784,3)))
    meansTrain.append([np.mean(rgb[0]), np.mean(rgb[1])])

meansTest = []
for row in x_test_color.values:
    rgb = np.array(row.reshape((784,3)))
    meansTest.append([np.mean(rgb[0]), np.mean(rgb[1])])


#Means is the mean of reds and greens for each image
# initialize 7 reasonably random centroids
centers = [[10,10],[250,250],[174,174],[120,130],[75,75],[123,123],[215,123]]
meansTrain = np.array(meansTrain)
makeMeans(7,meansTrain,centers)
plt.figure(figsize=(17,12))
sns.scatterplot(meansTrain[:,0], meansTrain[:,1], hue=y_train_color)
plt.show()
#split data by response variable
response = imageBalancedDF['label']
predictors = imageBalancedDF.drop(['label'], axis=1)
# we use a test size of 0.5 as training takes much longer
X_train, X_test, Y_train, Y_test = train_test_split(predictors,
                                                    response,
                                                    test_size=.5, 
                                                    random_state=1, 
                                                    stratify=response)
Y_train = keras.utils.to_categorical(Y_train)
Y_test = keras.utils.to_categorical(Y_test)
X_train_normal = (X_train.values - np.mean(X_train.values))/np.std(X_train.values)
X_test_normal = (X_test.values - np.mean(X_test.values))/np.std(X_test.values)
from keras import regularizers
#nodes in the layers: 30 to 30 to 30 to 1
input_dim = X_train_normal.shape[1] # input dimension: just x
model = Sequential()

# our first hidden layer
model.add(Dense(40, input_dim=input_dim, 
                activation='relu',  kernel_regularizer=regularizers.l2(0.01))) 

# our first hidden layer
model.add(Dense(40,
                activation='relu',  kernel_regularizer=regularizers.l2(0.01)),  ) 

# our third hidden layer
model.add(Dense(40,activation='relu',  kernel_regularizer=regularizers.l2(0.01)))

# Our output layer
model.add(Dense(7, kernel_initializer='normal', activation='softmax',input_dim=40)) 

# compile it 
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Summary 
model.summary()
batch_size = 32
epochs = 60

# fit the model
model_history = model.fit(X_train_normal, Y_train, 
                          batch_size=batch_size, 
                          validation_data=(X_test_normal, Y_test), 
                          epochs=epochs, 
                          verbose=1)
plt.figure(figsize=(13,8))

# Plotting the train and validation errors
plt.plot(model_history.history['loss'], label = 'Training Error')
plt.plot(model_history.history['val_loss'], label = 'testing Error')
plt.title("Error over epochs")
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend();
plt.savefig('error.png')
print(len(model_history.history['loss']))
print("Loss: train={:.3f}, test={:.3f}".format(
    model_history.history['loss'][-1],
    model_history.history['val_loss'][-1]))
plt.figure(figsize=(13,8))

# Plotting the train and validation errors
plt.plot(model_history.history['acc'], label = 'Training Accuracy')
plt.plot(model_history.history['val_acc'], label = 'testing Accuracy')
plt.title("Accuracy over epochs")
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend();
plt.savefig('accuracy.png')
print(len(model_history.history['acc']))
print("Accuracy: train={:.3f}, test={:.3f}".format(
    model_history.history['acc'][-1],
    model_history.history['val_acc'][-1]))
predictions = model.predict(X_train)
Counter(colorImageBalancedDF['label'])
pd.DataFrame(predictions).sample(5)
pd.DataFrame(Y_train).sample(5)
#split data by response variable
response = colorImageBalancedDF['label']
predictors = colorImageBalancedDF.drop(['label'], axis=1)
# test size of 0.5 again
X_train, X_test, Y_train, Y_test = train_test_split(predictors,
                                                    response,
                                                    test_size=.5, 
                                                    random_state=1, 
                                                    stratify=response)
Y_train = keras.utils.to_categorical(Y_train)
Y_test = keras.utils.to_categorical(Y_test)
X_train_normal = (X_train.values - np.mean(X_train.values))/np.std(X_train.values)
X_test_normal = (X_test.values - np.mean(X_test.values))/np.std(X_test.values)
TRAIN_X_SHAPE = X_train_normal.reshape(2367, 28,28,3)
TEST_X_SHAPE = X_test_normal.reshape(2367, 28,28,3)
import keras.backend as K
K.set_image_dim_ordering('tf')
model = Sequential()
# add layers of CNN (we can play around with these numbers to get better accuracy)
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(28,28,3), kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3,3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(0.3))
model.add(Dense(7, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()
batch_size = 32
epochs = 20

# fit the model
model_history = model.fit(TRAIN_X_SHAPE, Y_train, 
                          batch_size=batch_size, 
                          validation_data=(TEST_X_SHAPE, Y_test), 
                          epochs=epochs, 
                          verbose=1)
plt.figure(figsize=(13,8))

# Plotting the train and validation errors
plt.plot(model_history.history['loss'], label = 'Training Error')
plt.plot(model_history.history['val_loss'], label = 'testing Error')
plt.title("Error over epochs")
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend();
plt.savefig('error.png')
print(len(model_history.history['loss']))
print("Loss: train={:.3f}, test={:.3f}".format(
    model_history.history['loss'][-1],
    model_history.history['val_loss'][-1]))
plt.figure(figsize=(13,8))

# Plotting the train and validation errors
plt.plot(model_history.history['acc'], label = 'Training Accuracy')
plt.plot(model_history.history['val_acc'], label = 'testing Accuracy')
plt.title("Accuracy over epochs")
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend();
plt.savefig('accuracy.png')
print(len(model_history.history['acc']))
print("Accuracy: train={:.3f}, test={:.3f}".format(
    model_history.history['acc'][-1],
    model_history.history['val_acc'][-1]))
predictions = model.predict(TEST_X_SHAPE)
pd.DataFrame(predictions).sample(5)
pd.DataFrame(Y_train).sample(5)
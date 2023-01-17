#system libraries
import os
import glob
import gc
import time
import datetime
import sys
from pathlib import Path
#Data manipulation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#Plotting
import cv2 as cv2
from skimage import feature
#Scientific
from scipy import stats
#Machine learning
import multiprocessing
#import joblib
from sklearn.externals.joblib import parallel_backend
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import svm, datasets
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
#cnn
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from keras import callbacks
from keras import preprocessing
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.engine.training import Model
from keras.utils import plot_model
def process_image(image_BGR, label, show):
    binary_mask = cv2.split(image_BGR)[0]
    ret, binary_mask = cv2.threshold(binary_mask, 10, 255, cv2.THRESH_BINARY)
    ret, binary_green = cv2.threshold(cv2.split(image_BGR)[1], 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    binary_mask = cv2.erode(binary_mask, kernel, iterations = 1)
    green_diff = cv2.subtract(binary_mask, binary_green)
    green_diff_canny = cv2.Canny(green_diff, 40, 40)
    green = cv2.split(image_BGR)[1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    green_contrast = clahe.apply(green)

    ########### PLOTTING ###########
    def plot_subfigures(rows, columns, index, image, isColor, title):
        plt.subplot(rows, columns, index)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        if isColor:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else :
            plt.imshow(image, cmap='gray')
        plt.title('{}'.format(title))
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    if show:
        plt.figure(1, figsize=(10, 5))
        plt.suptitle("Preprocessing Uninfected cell" if label == 0 else "Preprocessing Infected cell")
        #Sub figures
        plot_subfigures(2, 4, 1, image_BGR, True, "Original")
        plot_subfigures(2, 4, 2, binary_mask, False, "Binary Mask")
        plot_subfigures(2, 4, 3, binary_green, False, "Binary Green")
        plot_subfigures(2, 4, 4, green_diff, False, "Green diff")
        plot_subfigures(2, 4, 5, green_diff_canny, False, "Green diff Canny")
        plot_subfigures(2, 4, 6, green_contrast, False, "Green Contrast")
        plt.savefig(figure_directory + "cells_processing.pdf")
        plt.show()
    return (green_diff, green_diff_canny, green_contrast)
############# INPUT CONFIGURATION ###########
print("########################## MALARIA CELLS (Luca Giulianini) #########################")
start = time.time()
IN_COLAB = 'google.colab' in sys.modules
dataset_base_directory = "input/cell-images-for-detecting-malaria/cell_images/"
if not IN_COLAB: dataset_base_directory = "../" + dataset_base_directory
cells_folders = os.listdir(dataset_base_directory)
print("Cells folders:", cells_folders)
infected_folder = dataset_base_directory + "/Parasitized"
uninfected_folder = dataset_base_directory + "/Uninfected"
infected_images = glob.glob(infected_folder + "/*.png")
uninfected_images = glob.glob(uninfected_folder + "/*.png")
infected_number = len(infected_images)
uninfected_number = len(uninfected_images)
print('# Infected fotos: ', infected_number)
print('# Infected fotos: ', uninfected_number)

############# OUTPUT CONFIGURATION ###########
def create_new_folder(directory):
    if not os.path.exists(directory):
        try:
            os.mkdir(directory)
        except:
            print("Could not create {} directory".format(directory))
output_directory = r"output/"
figure_directory = "output/figures/"
models_directory = output_directory + r"models/"
classifiers_directory = output_directory + r"classifiers/"
logs_directory = output_directory + r"logs/"
model_directory = models_directory + time.strftime('%Y-%m-%d %H-%M-%S') + "/"
log_directory = logs_directory + time.strftime('%Y-%m-%d %H-%M-%S') + "/"

create_new_folder(output_directory)
create_new_folder(figure_directory)
create_new_folder(models_directory)
create_new_folder(logs_directory)
create_new_folder(model_directory)
create_new_folder(classifiers_directory)
create_new_folder(log_directory)
print("Output folders created under:", output_directory)
############ LOAD IMAGES AND LABELS ###############
number_of_cells = infected_number
cell_dim = 100
channels = 5
n_classes = 2
n_epochs = 25
images = []
labels = []

def load_images(imagePaths, label):
    for index, imagePath in enumerate(imagePaths):
        image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        image_BGR = cv2.resize(image, (cell_dim, cell_dim))
        images.append(image_BGR)
        labels.append(label)
        if index == number_of_cells - 1:
            break
        
load_images(infected_images, 1)
load_images(uninfected_images, 0)
print("Loaded {} images".format(len(images)))
################ SPLIT TRAIN VALID TEST ##################
images, labels = shuffle(images, labels) 
x_train_image, x_test_image, y_train_label, y_test_label = train_test_split(
    images, labels, test_size=0.2, random_state=23)

x_train_image, x_valid_image, y_train_label, y_valid_label = train_test_split(
    x_train_image, y_train_label, test_size=0.2, random_state=23)

x_train = []
y_train = []
x_valid = []
y_valid = []
x_test = []
y_test = []
################ STACKING ##################
print("Stacking images")
def populateSet(x_in, y_in, x_out, y_out):
    for image, label in zip(x_in, y_in):
        green_diff, green_diff_canny, green_contrast = process_image(image, label, show=False)
        image_out = np.dstack((image, green_diff))
        image_out = np.dstack((image_out, green_diff_canny))
        x_out.append(image_out)
        y_out.append(label)

populateSet(x_train_image, y_train_label, x_train, y_train)
populateSet(x_valid_image, y_valid_label, x_valid, y_valid)
populateSet(x_test_image, y_test_label, x_test, y_test)

#SHOW PROCESSING
for index, label in enumerate(y_train_label): 
    if label == 1:
        process_image(x_train_image[index], label, show=True)
        break
############# CREATING NP ARRAY ###########
x_train = np.array(x_train)
y_train = np.array(y_train)
x_valid = np.array(x_valid)
y_valid = np.array(y_valid)
x_test = np.array(x_test)
y_test = np.array(y_test)

del x_train_image
del x_valid_image
del x_test_image
del y_train_label
del y_valid_label
del y_test_label
del images
del infected_images
del uninfected_images
gc.collect()

########### PLOTTING ###########
plt.figure(1, figsize=(10, 5))
plt.title('Random cells visualisation')
for i in range(20):
    r = np.random.randint(0, x_train.shape[0], 1)
    plt.subplot(4, 5, i + 1)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.imshow(cv2.cvtColor(x_train[r[0]][:, :, :3], cv2.COLOR_BGR2RGB))
    plt.title('{} : {}'.format(
        'Infected' if labels[r[0]] == 1 else 'Unifected', labels[r[0]]))
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.savefig(figure_directory + "random_cells.pdf")
plt.show()

############# NORMALIZING ################
def normalize(dataset):
    dataset = dataset / 255
    dataset = dataset.reshape(dataset.shape[0], dataset.shape[1], dataset.shape[2], channels)

normalize(x_train)
normalize(x_valid)
normalize(x_test)

print(f'Shape of training image : {x_train.shape}')
print(f'Shape of validation image : {x_valid.shape}')
print(f'Shape of testing image : {x_test.shape}')
print(f'Shape of training labels : {y_train.shape}')
print(f'Shape of validation labels : {y_valid.shape}')
print(f'Shape of testing labels : {y_test.shape}')
# DATA GENERATOR
datagen = ImageDataGenerator(
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

# CREATING SEQUENTIAL MODEL
def CNNbuild(height, width, classes, channels):
    model=Sequential()
    model.add(Conv2D(16, (5,5), activation = 'relu', input_shape = (height, width, channels)))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization(axis =-1))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization(axis = -1))
    
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
        
    model.add(Dense(512, activation = 'relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

model = CNNbuild(height = cell_dim, width = cell_dim, classes = n_classes, channels = channels)
adam = optimizers.Adam(learning_rate = 0.001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()
from IPython.display import SVG
from keras.utils import model_to_dot
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
plot_model(model, to_file=figure_directory + "model.pdf", show_shapes=True, show_layer_names=True, dpi=300)
##################### SETTINGS CALLBACKS ####################
model_file = model_directory + "{epoch:02d}-val_accuracy-{val_accuracy:.2f}-val_loss-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(
    model_file, 
    monitor='val_accuracy', 
    save_best_only=True)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=6,
    verbose=1,
    restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.6,
    patience=1,
    verbose=1)

callbacks = [checkpoint, reduce_lr, early_stopping]

#history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), steps_per_epoch=len(x_train) / 32, validation_data=(x_valid, y_valid), epochs= n_epochs, verbose=1, shuffle=True, callbacks=callbacks)
history = model.fit(x_train, y_train, batch_size=32, validation_data=(x_valid, y_valid), epochs=n_epochs, verbose=1, shuffle=True, callbacks=callbacks)

################## EVALUATE ####################
plt.figure(figsize=(10, 6))
epochs = len(history.history['accuracy'])
plt.plot(range(epochs), history.history['val_accuracy'], label='Training Accuracy')
plt.plot(range(epochs), history.history['val_loss'], label='Taining Loss')
plt.xlabel("Number of Epoch's")
plt.ylabel('Accuracy/Loss Value')
plt.title('Training Accuracy and Training Loss')
plt.legend(loc="best")
plt.savefig(figure_directory + "accuracy_loss_graph.pdf")
plt.show()

################ METRICS ###################
metrics = model.evaluate(x_test, y_test, verbose=1)
cnn_predicted = model.predict(x_test)
cnn_predicted = cnn_predicted.reshape(cnn_predicted.shape[0])
cnn_predicted = np.round(cnn_predicted)
cnn_predicted = cnn_predicted.astype(int)
print()
print(f'LOSS : {metrics[0]}')
print(f'ACCURACY : {metrics[1]}')
#GRID SEARCH THOUGH CROSS VALIDATION (By Gianluca Aguzzi)
def best_performance_of(classifier, params, x_train, y_train, x_test, y_test, cross_validation = 10, save= False, feature_type = "neural_feature"):
    print("Grid search for: ", classifier)
    clf = GridSearchCV(classifier, params, n_jobs=multiprocessing.cpu_count(), cv=cross_validation, verbose=3)
    #Fit and search best param
    #with parallel_backend('threading'):
      #clf.fit(x_train, y_train)
    clf.fit(x_train, y_train)
    print("\nBest parameters set:")
    print(clf.best_params_)
    #used to print confusion matrix
    y_predict=clf.best_estimator_.predict(x_test)
    
    confusion_m = confusion_matrix(y_test, y_predict)
    index = ['bad','good']  
    columns = ['bad','good']
    cm_df = pd.DataFrame(confusion_m,columns,index)                      
    plt.figure(figsize=(10,6))
    heatmap = sns.heatmap(cm_df, annot=True, cmap="Blues")
    fig = heatmap.get_figure()
    fig.savefig(figure_directory + type(classifier).__name__ + "_" + feature_type + "_" + "matrix.pdf")
    plt.show()
    print("confusion matrix:")
    print(confusion_m)
    #Same information
    print("\nClassification report:")
    print(classification_report(y_test, y_predict))
    print("Best estimator score: ", clf.best_estimator_.score(x_test, y_test))
    #Store classifier if save is true
    if(save):
        score = int(clf.best_estimator_.score(x_test, y_test) * 100)
        print("store:" + "max" + str(score) +  ".sav")
        joblib.dump(clf.best_estimator_, classifiers_directory + "max" + str(score) +  ".sav")
################ NEURAL FEATURE EXTRACTION #############
extract = Model(model.inputs, model.layers[-5].output)
x_train_neural_features = extract.predict(x_train)
x_valid_neural_features = extract.predict(x_valid)
x_test_neural_features = extract.predict(x_test)
print("Extracted features from flatten layer, shape:", x_train_neural_features.shape)

################# TRAIN CLASSIFIERS ON NEURAL FEATURES ################
grid_search = False
if grid_search:
    #### SVM
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                        'C': [1, 10, 100]}]

    best_performance_of(svm.SVC(), tuned_parameters, 
                        x_train = x_train_neural_features, y_train = y_train, x_test = x_test_neural_features, y_test = y_test)

    #### RANDOM FOREST
    tuned_parameters = {"max_depth":[11,13,15,17,20]}
    best_performance_of(RandomForestClassifier(random_state=0), tuned_parameters, 
                        x_train = x_train_neural_features, y_train = y_train, x_test = x_test_neural_features, y_test = y_test)
    
    #### KNN
    tuned_parameters = {"n_neighbors": [3,5,9,11,15,17,19],}
    best_performance_of(KNeighborsClassifier(), tuned_parameters,
                        x_train = x_train_neural_features, y_train = y_train, x_test = x_test_neural_features, y_test = y_test)

####### SVM
svm_neural = svm.SVC(gamma=0.001, C=1, kernel='rbf') # [10, 0.001] V [1, 0.001]
####### RANDOM FOREST
rf_neural = RandomForestClassifier(max_depth=16, random_state=0)
####### KNN
knn_neural = KNeighborsClassifier(n_neighbors = 5) #5

################### MAJORITY NEURAL CLASSIFIER ####################
print("SVM/RF/KNN training on neural features...")
multi_neural = VotingClassifier(estimators=[
       ('svm', svm_neural), ('rf', rf_neural), ('knn', knn_neural)],
       voting='hard', weights=[1.2,1,1],
       flatten_transform=True, n_jobs=-1)
multi_neural = multi_neural.fit(x_train_neural_features, y_train)
multi_neural_predicted = multi_neural.predict(x_test_neural_features)
print("SVM/RF/KNN neural features accuracy score:", accuracy_score(y_test, multi_neural_predicted))
################ GRAYSCALE IMAGES #############
x_train_gray_diff = x_train[:, :, :, 3:4]
x_test_gray_diff = x_test[:, :, :, 3:4]

x_train_gray_diff = x_train_gray_diff.reshape(x_train_gray_diff.shape[0], x_train_gray_diff.shape[1], x_train_gray_diff.shape[2])
x_test_gray_diff = x_test_gray_diff.reshape(x_test_gray_diff.shape[0], x_test_gray_diff.shape[1], x_test_gray_diff.shape[2])
print("Green difference images shape:", x_train_gray_diff.shape)

plt.title("Green difference image")
plt.imshow(cv2.cvtColor(x_train_gray_diff[0], cv2.COLOR_BGR2RGB))
plt.show()

################# LBP FEATURE EXTRACTION ############
class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		# return the histogram of Local Binary Patterns
		return hist

x_train_lbp_features = []
x_test_lbp_features = []

desc = LocalBinaryPatterns(24, 8)

print("Computing LBP descriptors...")
for gray in x_train_gray_diff:
    hist = desc.describe(gray)
    x_train_lbp_features.append(hist)
for gray in x_test_gray_diff:
    hist = desc.describe(gray)
    x_test_lbp_features.append(hist)

x_train_lbp_features = np.array(x_train_lbp_features)
x_test_lbp_features = np.array(x_test_lbp_features)
print("lbp features shape:", x_train_lbp_features.shape)
################# TRAIN CLASSIFIERS ON LBP FEATURES ################
grid_search = False
if grid_search:
    #### SVM
    tuned_parameters = [{'kernel': ['linear'], 'C': [100, 1000, 10000]}]

    best_performance_of(svm.SVC(), tuned_parameters, 
                        x_train = x_train_lbp_features, y_train = y_train, x_test = x_test_lbp_features, y_test = y_test)

    #### RANDOM FOREST
    tuned_parameters = {"max_depth":[5,7,9,11,13,15]}
    best_performance_of(RandomForestClassifier(random_state=0), tuned_parameters, 
                        x_train = x_train_lbp_features, y_train = y_train, x_test = x_test_lbp_features, y_test = y_test)
    
    #### KNN
    tuned_parameters = {"n_neighbors": [3,5,9,11,15,17,19],}
    best_performance_of(KNeighborsClassifier(), tuned_parameters,
                        x_train = x_train_lbp_features, y_train = y_train, x_test = x_test_lbp_features, y_test = y_test)

####### SVM
svm_lbp = svm.SVC(C=10000, kernel='linear') #10000
####### RANDOM FOREST
rf_lbp = RandomForestClassifier(max_depth=7, random_state=0) #7
####### KNN
knn_lbp = KNeighborsClassifier(n_neighbors = 9) #9

################### MAJORITY LBP CLASSIFIER ####################
print("SVM/RF/KNN training on LBP features...")
multi_lbp = VotingClassifier(estimators=[
       ('svm', svm_lbp), ('rf', rf_lbp), ('knn', knn_lbp)],
       voting='hard', weights=[1.2,1,1],
       flatten_transform=True, n_jobs=-1)
multi_lbp = multi_lbp.fit(x_train_lbp_features, y_train)
multi_lbp_predicted = multi_lbp.predict(x_test_lbp_features)
print("SVM/RF/KNN LBP features accuracy score:", accuracy_score(y_test, multi_lbp_predicted))
################### FINAL EVALUATION ########################
print("\nFINAL EVALUATION")

cnn_neural_lbp_predicted = np.column_stack((cnn_predicted, multi_neural_predicted))
cnn_neural_lbp_predicted = np.column_stack((cnn_neural_lbp_predicted, multi_lbp_predicted))
cnn_neural_lbp_predicted = stats.mode(cnn_neural_lbp_predicted, axis=1)[0]
print("\nFINAL CNN/Neural/LBP accuracy score:", accuracy_score(y_test, cnn_neural_lbp_predicted))
################### PCA T-SNE AND PLOTTING ################# 
print("Features visualisation: PDA -> T-SNE")
feature_number = 10000
x_visual_features = x_train_neural_features[:feature_number]
y_visual_features = y_train[:feature_number]
#features_columns = ['pixel' + str(i) for i in range(x_train_neural_features.shape[1])]
df = pd.DataFrame(x_visual_features) #, columns=features_columns)
df['y'] = y_visual_features
#df['label'] = df['y'].apply(lambda i: str(i))
print('Dataframe size:', df.shape)

pca = PCA(n_components=50)
pca_result = pca.fit_transform(df)
print("PCA shape:", pca_result.shape)

tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_result = tsne.fit_transform(df)
print("T-SNE shape:", tsne_pca_result.shape)

df['dim-1'] = tsne_pca_result[:,0]
df['dim-2'] = tsne_pca_result[:,1] 

plt.figure(figsize=(10,6))
plt.title("T-SNE 2D visualisation")
sns.scatterplot(
    x="dim-1", y="dim-2",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=df,
    legend="full",
    alpha=1
)
plt.savefig(figure_directory + "cells_visualisation_2D.pdf")
plt.show()

######################## COMPUTE ELAPSED TIME ###########################
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("ELAPSED TIME:", "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

# WAIT AND DESTROY
cv2.waitKey()
cv2.destroyAllWindows()
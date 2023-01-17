import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, KFold ,cross_val_score

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize 
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  

import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib as mpl
%matplotlib inline
mpl.style.use( 'ggplot' )
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import matplotlib.pylab as pylab
import seaborn as sns
sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)
train = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
test = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
print(f'Train:\t {train.shape} \nTest:\t {test.shape}')
      
# check for null values
print(f'Are there null values? {train.isnull().any().sum()>0 | test.isnull().any().sum()>0}')
# verifying the data is balanced
train['label'].value_counts()
test['label'].value_counts()
#defining the list for labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
# Visualize some random pictures and their labels
fig, axes = plt.subplots(5, 5, figsize = (6,6))
for row in axes:
    for axe in row:
        index = np.random.randint(60000)
        img = train.drop('label', axis=1).values[index].reshape(28,28)
        cloths = train['label'][index]
        axe.imshow(img, cmap='gray')
        axe.set_title(class_names[cloths])
        axe.set_axis_off()
x=train.drop('label', axis=1)
# finding pixels that are always black
temp_dic ={}
for col in x.columns:
    count = 0
    for val in x[col]:
        if val > 0 :
            count+=1
    temp_dic[col]=count
sort_dic = sorted(temp_dic.items(), key=lambda x: x[1])

# present the least used pixels
print('The least used pixels are:')
for i in range(10):
    print(f'{sort_dic[i][0]:<10} {sort_dic[i][1]:>10}')  
#presenting heatmaps for pixels usage per category
fig, axes = plt.subplots(3,4, figsize = (6,6))
i = 0
for row in axes:
    for axe in row:
        if i<10:
            selected_category = train.loc[train['label'] == i].drop('label', axis=1)
            sns.heatmap(pd.DataFrame(selected_category).mean().values.reshape(28, 28), cmap='gray_r',cbar=False,ax=axe)
            axe.set_title(class_names[i])
            axe.set_axis_off()
        else:
            axe.axis('off')
        i+=1
df = train.copy()
df_test = test.copy()
df.label.unique()
X_train = df.iloc[:,1:]
y_train = df.iloc[:,0]
pca = PCA(n_components=2)
X_r = pca.fit(X_train).transform(X_train)
plt.style.use('fivethirtyeight')
fig, axarr = plt.subplots(1, 2, figsize=(8, 4))
sns.heatmap(pca.components_[0, :].reshape(28, 28), ax=axarr[0], cmap='gray_r')
sns.heatmap(pca.components_[1, :].reshape(28, 28), ax=axarr[1], cmap='gray_r')
axarr[0].set_title("{0:.2f}% Explained Variance".format(pca.explained_variance_ratio_[0]*100),fontsize=12)
axarr[1].set_title("{0:.2f}% Explained Variance".format(pca.explained_variance_ratio_[1]*100),fontsize=12)
axarr[0].set_aspect('equal')
axarr[1].set_aspect('equal')
plt.suptitle('2-Component PCA');
# PCA
label = 2
X_train_label = X_train[y_train==label]

class PCAOutlierRemover:
    def __init__(self, n_components=10, k_outliers=5):
        self.n_components = n_components
        self.k_outliers = k_outliers

    def fit(self, X, y=None):
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X) 

        self.errors_sorted_ = {}
        for label in range(10):
            X_label = X[y_train==label]
            X_reduced = self.pca.transform(X_label)                                    # NumPy array [N x n_components]
            X_reconstructed = self.pca.inverse_transform(X_reduced)                    # NumPy array [N x k_outliers]
            X_reconstructed_df = pd.DataFrame(X_reconstructed,
                                            index=X_label.index)                       # pandas DataFrame [N x k_outliers]
            errors = ((X_label - X_reconstructed)**2).sum(axis=1)                      # pandas Series [N x 1, indexed by X.index]
            self.errors_sorted_[label] = errors.iloc[errors.argsort()]                 # pandas Series [N x 1, indexed by X.index] 
        return self

    def transoform(self, X):
        X_without_outliers = []
        for label in range(10):
            X_label = X[y_train==label]
            self.without_k_worst_ix = self.errors_sorted_[label][:-self.k_outliers].index.values        # NumPy array [k_outliers x 1]
            X_without_outliers.append(X_label.loc[self.without_k_worst_ix, :])
        return pd.concat(X_without_outliers, axis=0)
    
    
my_pca_remover = PCAOutlierRemover(n_components=10, k_outliers=5)
my_pca_remover.fit(X_train)
X_train_without_pca_outliers = my_pca_remover.transoform(X_train)
X_train_without_pca_outliers.shape
label = 3 #selected category number
X_train_label = X_train[y_train==label]
y_train_kb = y_train==label

class SKBOutlierRemover:
    def __init__(self, score_func=f_classif ,k=20 , y=y_train_kb, k_outliers=5):
        self.score_func = score_func
        self.k_outliers = k_outliers
        self.k= k
        self.y= y

    def fit(self, X, y=None):
        self.kbest = SelectKBest(score_func=self.score_func , k=self.k)             #commented out the [y=self.y] that was passed to the SelectKBest - not needed
        self.kbest.fit(X, y)

        self.errors_sorted_ = {}
        for label in range(10):
            X_label = X[y_train==label]
            X_reduced = self.kbest.transform(X_label)                                    # NumPy array [N x n_components]
            X_reconstructed = self.kbest.inverse_transform(X_reduced)                    # NumPy array [N x k_outliers]
            X_reconstructed_df = pd.DataFrame(X_reconstructed,
                                            index=X_label.index)                       # pandas DataFrame [N x k_outliers]
            errors = ((X_label - X_reconstructed)**2).sum(axis=1)                      # pandas Series [N x 1, indexed by X.index]
            self.errors_sorted_[label] = errors.iloc[errors.argsort()]                 # pandas Series [N x 1, indexed by X.index]
        return self

    def transform(self, X):
        X_without_outliers = []
        for label in range(10):
            X_label = X[y_train==label]
            self.without_k_worst_ix = self.errors_sorted_[label][:-self.k_outliers].index.values        # NumPy array [k_outliers x 1]
            X_without_outliers.append(X_label.loc[self.without_k_worst_ix, :])
        return pd.concat(X_without_outliers, axis=0)
   
my_skb_remover = SKBOutlierRemover(score_func=f_classif, k=20, y=y_train.loc[X_train_without_pca_outliers.index])
my_skb_remover.fit(X=X_train_without_pca_outliers, y=y_train.loc[X_train_without_pca_outliers.index])
X_train_without_skb_outliers = my_skb_remover.transform(X_train_without_pca_outliers)
X_train = X_train_without_skb_outliers.copy()
y_train = y_train.loc[X_train.index]
X_train.shape
#normalization
X_norm = normalize(X_train.values)
y_norm = y_train
# reshaping
X_train = np.array(X_norm).reshape(-1, 28, 28, 1)

#categorizing
num_classes = 10
y_train = keras.utils.np_utils.to_categorical(y_norm, num_classes)
# # Model CNN 3 with dropout layers to reduce overfitting:
# img_rows, img_cols = 28, 28
# input_shape = (img_rows, img_cols, 1)
# model3 = Sequential()
# model3.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', input_shape=input_shape))
# model3.add(MaxPooling2D((2, 2)))
# model3.add(Dropout(0.25))
# model3.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model3.add(MaxPooling2D(pool_size=(2, 2)))
# model3.add(Dropout(0.25))
# model3.add(Conv2D(128, (3, 3), activation='relu'))
# model3.add(Dropout(0.4))
# model3.add(Flatten())
# model3.add(Dense(128, activation='relu'))
# model3.add(Dropout(0.3))
# model3.add(Dense(num_classes, activation='softmax'))
# model3.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
# model3.summary()
# X_train_m3, X_val_m3, y_train_m3, y_val_m3 = train_test_split(X_train, y_train, test_size=0.2)
# history = model3.fit(X_train_m3, y_train_m3, batch_size=256, epochs=20,validation_data=(X_val_m3,y_val_m3))
# print(history.history.keys())
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
# # Model 4:
# img_rows, img_cols = 28, 28
# input_shape = (img_rows, img_cols, 1)
# model4 = Sequential()
# model4.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last', input_shape=input_shape))
# model4.add(BatchNormalization())
# model4.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
# model4.add(BatchNormalization())
# model4.add(Dropout(0.25))
# model4.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
# model4.add(MaxPooling2D(pool_size=(2, 2)))
# model4.add(Dropout(0.25))
# model4.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
# model4.add(BatchNormalization())
# model4.add(Dropout(0.25))
# model4.add(Flatten())
# model4.add(Dense(512, activation='relu'))
# model4.add(BatchNormalization())
# model4.add(Dropout(0.5))
# model4.add(Dense(128, activation='relu'))
# model4.add(BatchNormalization())
# model4.add(Dropout(0.5))
# model4.add(Dense(num_classes, activation='softmax'))
# optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999 )
# model4.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
# model4.summary()
# X_train_m4, X_val_m4, y_train_m4, y_val_m4 = train_test_split(X_train, y_train, test_size=0.2)
# history = model4.fit(X_train_m4, y_train_m4, batch_size=128, epochs=20,validation_data=(X_val_m4,y_val_m4))
# print(history.history.keys())
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
data= np.array(X_train).reshape(-1, 28, 28) # Transformation for Resnet50:
data = np.stack([data, data, data], axis=-1) #convert to 3 channel RGB
data.shape
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
# Resnet50:
num_classes = 10
def create_model():
        model_rn50 = Sequential()
        model_rn50.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path)) # , input_shape=(28, 28, 1, 3)
        # model_rn50.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
        model_rn50.add(Dropout(0.50))
        model_rn50.add(Dense(num_classes, activation='softmax'))
        model_rn50.layers[0].trainable = True # Indicate whether the first layer should be trained/changed or not.
        # model.layers[0].trainable = False
        model_rn50.summary()
        model_rn50.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model_rn50
X_train_rn50, X_val_rn50, y_train_rn50, y_val_rn50 = train_test_split(data, y_train, test_size=0.2)
# history = create_model().fit(X_train_rn50, y_train_rn50, batch_size=128, epochs=20,validation_data=(X_val_rn50,y_val_rn50),validation_split=0.3)
#grid search
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

model = KerasClassifier(build_fn=create_model)
param_grid = {'epochs':[10,20,25],'batch_size':[128, 256]}
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='roc_auc', verbose=1)
grid_result = grid.fit(X_train_rn50, y_train_rn50)
grid.estimators

history=grid_result
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
X_test_rn50 = df_test.iloc[:,1:]
y_test_rn50 = df_test.iloc[:,0]
X_test_norm = normalize(X_test_rn50.values)
test_data_rn50= np.array(X_test_norm).reshape(-1, 28, 28)
test_data_rn50 = X_rgb = np.stack([test_data_rn50, test_data_rn50, test_data_rn50], axis=-1)
test_data_rn50.shape

predicted_classes = model_rn50.predict_classes(test_data_rn50)
unique_elements, counts_elements = np.unique(predicted_classes, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))
p = pd.Series(predicted_classes[:10000])
y = y_test_rn50
correct = (p==y)
correct = correct.index[correct]
incorrect = (p!=y)
incorrect = incorrect.index[incorrect]
print("Correct predicted classes:",correct.shape[0])
print("Incorrect predicted classes:",incorrect.shape[0])
target_names = ["Class {} ({}) :".format(i,labels[i]) for i in range(num_classes)]
print(classification_report(y_test_rn50, predicted_classes, target_names=target_names))
img_rows, img_cols = 28, 28
def plot_images(data_index,cmap="Blues"):   # Plot the sample images now
    f, ax = plt.subplots(3,3, figsize=(6,6))
    for i, indx in enumerate(data_index[:9]):
        ax[i//3, i%3].imshow(test.drop('label', axis=1).values[indx].reshape(img_rows,img_cols), cmap=cmap)
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title("True:{}  Pred:{}".format(labels[y_test_rn50[indx]],labels[predicted_classes[indx]]),fontsize=9)
    plt.show()    
plot_images(pd.Series(correct), "Greens")

plot_images(incorrect, "Reds")
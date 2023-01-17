import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
!pip install --upgrade --quiet tensorflow-gpu
import glob
import cv2
submission = pd.read_csv("../input/jovian-pytorch-z2g/submission.csv")
submission.sample(10)
train_files = glob.glob("../input/jovian-pytorch-z2g/Human protein atlas/train/*.png")
test_files = glob.glob("../input/jovian-pytorch-z2g/Human protein atlas/test/*.png")
len(train_files),len(test_files)
train_files_map = {int(train_file.split(".")[2].split("/")[-1]) : train_file for train_file in train_files}
train_labels = pd.read_csv("../input/jovian-pytorch-z2g/Human protein atlas/train.csv")
train_labels['File_path'] = train_labels['Image'].map(train_files_map)
train_labels.head()
sample_data = train_labels.groupby('Label', group_keys=False)[['File_path']].apply(lambda x : x.sample(5, replace=True))
sample_data = sample_data.reset_index()
sample_data.columns = ['Label','Image','File_path']
sample_data.head(10)
from collections import Counter
label_counter = Counter(train_labels['Label'].values)
label,counter = zip(*list(label_counter.most_common()))
label_df = pd.DataFrame()
label_df['label'],label_df['count'] = label,counter
label_df['label_num'] = label_df['label'].apply(lambda x: len(x.split(" ")))
label_df.groupby('label_num').sum()[['count']].plot(kind='bar',title='Total Imager for Each Number of Labels')
label_df.groupby('label_num').count()[['count']].plot(kind='bar',title='Unique Label Value present in Each Label Class')
unique_labels = set([int(l) for label in label_df['label'].unique() for l in label.split()])
unique_labels
fig,ax = plt.subplots(5,5,figsize=(15,15),sharex=True)
fig.suptitle('Sample Images Belonging to Each Class', fontweight = 'bold',fontsize = 20)
#plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)
np.vectorize(lambda ax:ax.axis('off'))(ax) #turns off x and y axis for all figures

for row,i in enumerate(np.random.choice(sample_data['Label'].unique(),5)):
    for j in range(5):
        ax[row,j].imshow(cv2.imread(sample_data[sample_data['Label'] == i]['File_path'].values[j]))
        ax[row,j].set_title(f"Multi-Label : {i}")
150/19236
imbalanced_data = train_labels.groupby('Label').count()[['Image']].sort_values('Image')
imbalanced_data['percentage'] = imbalanced_data['Image']/imbalanced_data['Image'].sum() * 100
imbalanced_data.head()
imbalanced_data.sort_values("Image",ascending=False).head()
model_data = imbalanced_data[imbalanced_data['Image'] > 100]
img_idx = model_data.index.values
dataset = train_labels[train_labels['Label'].isin(img_idx)]
dataset.head()
dataset.groupby('Label').count()[['Image']].sort_values('Image').plot(kind='bar', figsize=(8,6),title='Image Count for Classes with > 1% Images')
lbl = [list([int(l) for l in label.split()]) for label in dataset.Label]
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb_labels = mlb.fit_transform(lbl)
mlb_labels.shape
mlb.classes_
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator
unique_label = { val : idx for idx,val in enumerate(dataset['Label'].unique())}
dataset['unique_label'] = dataset['Label'].map(unique_label)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(dataset[['File_path','Label']],dataset['unique_label'],stratify=dataset['unique_label'],test_size=0.2,shuffle=True)
#X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,stratify=y_train,test_size=0.2,shuffle=True)

#train,test,valid = X_train,X_test,X_val
train,test = X_train,X_test
train['y'] = y_train

#X_train.shape,X_test.shape,X_val.shape
X_train.shape,X_test.shape
datagen=ImageDataGenerator(rescale=1./255., validation_split=0.25)
test_datagen=ImageDataGenerator(rescale=1./255.)
train_generator = datagen.flow_from_dataframe( dataframe=train, x_col="File_path", y_col="Label",
                subset="training", batch_size=32, seed=42, shuffle=True, class_mode="categorical", target_size=(128,128))

valid_generator = datagen.flow_from_dataframe( dataframe=train, x_col="File_path", y_col="Label",
                subset="validation", batch_size=32, seed=42, shuffle=True, class_mode="categorical", target_size=(128,128))

test_generator = test_datagen.flow_from_dataframe( dataframe=test,  x_col="File_path", y_col=None,
                batch_size=32, seed=42, shuffle=False, class_mode=None, target_size=(128,128))
import tensorflow
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(Conv2D(32, (3, 3),input_shape=(128,128,3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(29, activation='softmax'))
model.compile(optimizer = 'adam',loss="categorical_crossentropy",metrics=["accuracy", "categorical_accuracy"])
model.summary()
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10)
model.save("conv2d_2_layers_2_5_mil_param.h5")
#submission
test_submission = pd.DataFrame()
test_submission['Files_path'] = test_files
test_files_map = { test_file : int(test_file.split(".")[2].split("/")[-1]) for test_file in test_files}
test_submission['Id'] = test_submission['Files_path'].map(test_files_map)
test_submission.head()
from tqdm import tqdm
predictions = []
for file_path in tqdm(test_submission.Files_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (128,128))
    predictions.append(model.predict(img.reshape(1,128,128,3)))
pred_index = [ np.argmax(pred[0]) for pred in predictions]
test_submission['predict_index'] = pred_index 
test_submission.head()
train_generator_reverse = { value : key for key,value in train_generator.class_indices.items() }
test_submission['Label'] = test_submission['predict_index'].map(train_generator_reverse)
test_submission.head()
test_submission = test_submission[['Id','Label']]
test_submission.columns = ['Image','Label']
test_submission.head()
test_submission.to_csv("submission_CNN_4_Layers_2_Million_Param_29_acc.csv",index=False)
test_submission.head()

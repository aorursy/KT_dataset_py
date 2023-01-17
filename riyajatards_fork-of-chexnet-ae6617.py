import os
os.makedirs("/kaggle/working/dataset")
# Necessary Dependencies
import numpy as np 
import pandas as pd 
from glob import glob
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from itertools import chain
from datetime import datetime
import statistics

# Disease Names 
disease_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion','Pneumonia', 'Pleural_Thickening',
'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

# Load Stanford Images Distribution Files


labels_train_val = pd.read_csv('/kaggle/input/txt-file/train_val_list.txt')
labels_train_val.columns = ['Image_Index']

labels_test = pd.read_csv('/kaggle/input/txt-file/test_list.txt')
labels_test.columns = ['Image_Index']


print(labels_test['Image_Index'])




# NIH Dataset Labels CSV File 

labels_df = pd.read_csv('/kaggle/input/data/Data_Entry_2017.csv')

labels_df.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                  'Patient_Age', 'Patient_Gender', 'View_Position',
                  'Original_Image_Width', 'Original_Image_Height',
                  'Original_Image_Pixel_Spacing_X',
                  'Original_Image_Pixel_Spacing_Y', 'dfd']


labels_df['Finding_Labels'] = labels_df['Finding_Labels'].map(lambda x: x.replace('No Finding', str(None)))
labels_df['Finding_Labels'] = labels_df['Finding_Labels'].replace('None', None)
# Binarizes Each Disease Class in their own column
from tqdm import tqdm

for diseases in tqdm(disease_labels): #TQDM is a progress bar setting
    labels_df[diseases] = labels_df['Finding_Labels'].map(lambda result: 1 if diseases in result else 0)
#     print(labels_df['Finding_Labels'].map(lambda result: 1 if diseases in result else 0))

train_val_merge = pd.merge(left=labels_train_val, right=labels_df, left_on='Image_Index', right_on='Image_Index')

test_merge = pd.merge(left=labels_test, right=labels_df, left_on='Image_Index', right_on='Image_Index')



print(train_val_merge,test_merge)


# Splitting Finding Labels
train_val_merge['Finding_Labels'] = train_val_merge['Finding_Labels'].apply(lambda s: [l for l in str(s).split('|')])

test_merge['Finding_Labels'] = test_merge['Finding_Labels'].apply(lambda s: [l for l in str(s).split('|')])

print(test_merge)



num_glob = glob('/kaggle/input/data/*/images/*.png')
img_path = {os.path.basename(x): x for x in num_glob}


train_val_merge['Paths'] = train_val_merge['Image_Index'].map(img_path.get)
test_merge['Paths'] = test_merge['Image_Index'].map(img_path.get)

from tqdm import tqdm
import os,cv2


# No Overlap in patients between the Train and Validation Data Sets
patients = np.unique(train_val_merge['Patient_ID'])
test_patients = np.unique(test_merge['Patient_ID'])

print('Number of Patients Between Train-Val Overall: ', len(patients))
print('Number of Patients Between Train-Val Overall: ', len(test_patients))


# Train-Validation Split 
train_df, val_df = train_test_split(patients,
                                   test_size = 0.0669,
                                   random_state = 2019,
                                    shuffle= True)  



labels_df

print('No. of Unique Patients in Train dataset : ',len(train_df))
train_df = train_val_merge[train_val_merge['Patient_ID'].isin(train_df)]
print('Training Dataframe   : ', train_df.shape[0],' images',)

print('\nNo. of Unique Patients in Validtion dataset : ',len(val_df))
val_df = train_val_merge[train_val_merge['Patient_ID'].isin(val_df)]
print('Validation Dataframe   : ', val_df.shape[0],' images')

print('\nNo. of Unique Patients in Testing dataset : ',len(test_patients))
test_df = test_merge[test_merge['Patient_ID'].isin(test_patients)]
print('Testing Dataframe   : ', test_df.shape[0],' images')


train_df1 = train_df.drop(['Finding_Labels','Follow_Up_#','Patient_Age','Patient_Gender','View_Position','Original_Image_Width','Original_Image_Height','Original_Image_Pixel_Spacing_X','Original_Image_Pixel_Spacing_Y','Paths'],axis=1)
train_df1.to_csv("train.csv")

val_df1= val_df.drop(['Finding_Labels','Follow_Up_#','Patient_Age','Patient_Gender','View_Position','Original_Image_Width','Original_Image_Height','Original_Image_Pixel_Spacing_X','Original_Image_Pixel_Spacing_Y','Paths'],axis=1)
val_df1.to_csv("valid.csv")

test_df1= test_df.drop(['Finding_Labels','Follow_Up_#','Patient_Age','Patient_Gender','View_Position','Original_Image_Width','Original_Image_Height','Original_Image_Pixel_Spacing_X','Original_Image_Pixel_Spacing_Y','Paths'],axis=1)
test_df1.to_csv("test.csv")


#
# for value in tqdm(img_path.values()):
#     I = cv2.cvtColor(cv2.imread(value),cv2.COLOR_BGR2RGB)
#     I = cv2.resize(I,(320,320))
#     filename = os.path.splitext(os.path.basename(value))[0]
# #     if not os.path.exists("/kaggle/working/data/"+filename+".jpg"):
#     cv2.imwrite("/kaggle/working/dataset/"+filename+".png",I)
# #     print(value)

train_df1
# No Overlap in patients between the Train and Validation Data Sets
patients = np.unique(train_val_merge['Patient_ID'])
test_patients = np.unique(test_merge['Patient_ID'])

print('Number of Patients Between Train-Val Overall: ', len(patients))
print('Number of Patients Between Train-Val Overall: ', len(test_patients))
# Train-Validation Split 
train_df, val_df = train_test_split(patients,
                                   test_size = 0.0669,
                                   random_state = 2019,
                                    shuffle= True)  


print('No. of Unique Patients in Train dataset : ',len(train_df))
train_df = train_val_merge[train_val_merge['Patient_ID'].isin(train_df)]
print('Training Dataframe   : ', train_df.shape[0],' images')

print('\nNo. of Unique Patients in Validtion dataset : ',len(val_df))
val_df = train_val_merge[train_val_merge['Patient_ID'].isin(val_df)]
print('Validation Dataframe   : ', val_df.shape[0],' images')

print('\nNo. of Unique Patients in Testing dataset : ',len(test_patients))
test_df = test_merge[test_merge['Patient_ID'].isin(test_patients)]
print('Testing Dataframe   : ', test_df.shape[0],' images')
X_train = train_df['Paths']
y_train = train_df['Finding_Labels']

X_val = val_df['Paths']
y_val = val_df['Finding_Labels']

X_test = test_df['Paths']
y_test = test_df['Finding_Labels']
y_train
# Binarizing Labels 
from sklearn.preprocessing import MultiLabelBinarizer

print("Labels - ")
mlb = MultiLabelBinarizer()
mlb.fit(y_train)
X_train = train_df['Paths']
y_train = train_df['Finding_Labels']

X_val = val_df['Paths']
y_val = val_df['Finding_Labels']

X_test = test_df['Paths']
y_test = test_df['Finding_Labels']
y_train_bin = mlb.transform(y_train)
y_val_bin = mlb.transform(y_val)
y_test_bin = mlb.transform(y_test)
# Necessary Dependencies
import numpy as np 
import pandas as pd 
from glob import glob
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from itertools import chain
from datetime import datetime
import statistics

# Disease Names 
disease_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion','Pneumonia', 'Pleural_Thickening',
'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

# Load Stanford Images Distribution Files


labels_train_val = pd.read_csv('/kaggle/input/txt-file/train_val_list.txt')
labels_train_val.columns = ['Image_Index']

labels_test = pd.read_csv('/kaggle/input/txt-file/test_list.txt')
labels_test.columns = ['Image_Index']


print(labels_test['Image_Index'])




# NIH Dataset Labels CSV File 

labels_df = pd.read_csv('/kaggle/input/data/Data_Entry_2017.csv')

labels_df.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                  'Patient_Age', 'Patient_Gender', 'View_Position',
                  'Original_Image_Width', 'Original_Image_Height',
                  'Original_Image_Pixel_Spacing_X',
                  'Original_Image_Pixel_Spacing_Y', 'dfd']





labels_df['Finding_Labels'] = labels_df['Finding_Labels'].map(lambda x: x.replace('No Finding', str(None)))
labels_df['Finding_Labels'] = labels_df['Finding_Labels'].replace('None', None)
# Binarizes Each Disease Class in their own column
from tqdm import tqdm

for diseases in tqdm(disease_labels): #TQDM is a progress bar setting
    labels_df[diseases] = labels_df['Finding_Labels'].map(lambda result: 1 if diseases in result else 0)
#     print(labels_df['Finding_Labels'].map(lambda result: 1 if diseases in result else 0))

train_val_merge = pd.merge(left=labels_train_val, right=labels_df, left_on='Image_Index', right_on='Image_Index')

test_merge = pd.merge(left=labels_test, right=labels_df, left_on='Image_Index', right_on='Image_Index')



print(train_val_merge,test_merge)


# Splitting Finding Labels
train_val_merge['Finding_Labels'] = train_val_merge['Finding_Labels'].apply(lambda s: [l for l in str(s).split('|')])

test_merge['Finding_Labels'] = test_merge['Finding_Labels'].apply(lambda s: [l for l in str(s).split('|')])

print(test_merge)



num_glob = glob('/kaggle/input/data/*/images/*.png')
img_path = {os.path.basename(x): x for x in num_glob}


train_val_merge['Paths'] = train_val_merge['Image_Index'].map(img_path.get)
test_merge['Paths'] = test_merge['Image_Index'].map(img_path.get)

from tqdm import tqdm
import os,cv2


# No Overlap in patients between the Train and Validation Data Sets
patients = np.unique(train_val_merge['Patient_ID'])
test_patients = np.unique(test_merge['Patient_ID'])

print('Number of Patients Between Train-Val Overall: ', len(patients))
print('Number of Patients Between Train-Val Overall: ', len(test_patients))


# Train-Validation Split 
train_df, val_df = train_test_split(patients,
                                   test_size = 0.0669,
                                   random_state = 2019,
                                    shuffle= True)  





print('No. of Unique Patients in Train dataset : ',len(train_df))
train_df = train_val_merge[train_val_merge['Patient_ID'].isin(train_df)]
print('Training Dataframe   : ', train_df.shape[0],' images',)

train_df.to_csv("train_f.csv")

print('\nNo. of Unique Patients in Validtion dataset : ',len(val_df))
val_df = train_val_merge[train_val_merge['Patient_ID'].isin(val_df)]
print('Validation Dataframe   : ', val_df.shape[0],' images')

print('\nNo. of Unique Patients in Testing dataset : ',len(test_patients))
test_df = test_merge[test_merge['Patient_ID'].isin(test_patients)]
print('Testing Dataframe   : ', test_df.shape[0],' images')


# train_df1 = train_df.drop(['Finding_Labels','Follow_Up_#','Patient_Age','Patient_Gender','View_Position','Original_Image_Width','Original_Image_Height','Original_Image_Pixel_Spacing_X','Original_Image_Pixel_Spacing_Y','Paths'],axis=1)
# train_df1.to_csv("train.csv")

# val_df1= val_df.drop(['Finding_Labels','Follow_Up_#','Patient_Age','Patient_Gender','View_Position','Original_Image_Width','Original_Image_Height','Original_Image_Pixel_Spacing_X','Original_Image_Pixel_Spacing_Y','Paths'],axis=1)
# val_df1.to_csv("valid.csv")

# test_df1= test_df.drop(['Finding_Labels','Follow_Up_#','Patient_Age','Patient_Gender','View_Position','Original_Image_Width','Original_Image_Height','Original_Image_Pixel_Spacing_X','Original_Image_Pixel_Spacing_Y','Paths'],axis=1)
# test_df1.to_csv("test.csv")



X_train = train_df['Paths']
y_train = train_df['Finding_Labels']

X_val = val_df['Paths']
y_val = val_df['Finding_Labels']

X_test = test_df['Paths']
y_test = test_df['Finding_Labels']


# Binarizing Labels 
from sklearn.preprocessing import MultiLabelBinarizer

print("Labels - ")
mlb = MultiLabelBinarizer()
mlb.fit(y_train)


y_train_bin = mlb.transform(y_train)
y_val_bin = mlb.transform(y_val)
y_test_bin = mlb.transform(y_test)

def data_generator_variable_input_img(train_df,y_train_bin,batch_size = 1):
    
    while True:
        
        
        np.random.seed(2020)
        
        img_list = train_df["Paths"].values
        
        indx = np.random.randint(len(img_list),size=batch_size)
        
        label = y_train_bin
        
        Y = []
        
        X  = []

        for i in indx:

            I = cv2.cvtColor(cv2.imread(img_list[i]),cv2.COLOR_BGR2RGB)
            label1 = list(label[i])
            i224 = cv2.resize(I,(224,224))   
            X.append(i224)
            Y.append(label1)


        yield([np.array(X)],np.array(Y).astype("float32"))
                           
                   
# Section of Code written by brucechou1983 - https://github.com/brucechou1983/CheXNet-Keras
# I have no experience with class weighting, brucechou1983 provided a very thorough explanation of the topic with example code
CLASS_NAMES = disease_labels
def get_class_weights(total_counts, class_positive_counts, multiply):
    """
    Calculate class_weight used in training
    Arguments:
    total_counts - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    multiply - int, positve weighting multiply
    use_class_balancing - boolean 
    Returns:
    class_weight - dict of dict, ex: {"Effusion": { 0: 0.01, 1: 0.99 }, ... }
    """
    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }

    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = []
    for i, class_name in enumerate(class_names):
        class_weights.append(get_single_class_weight(label_counts[i], total_counts))

    return class_weights

def get_sample_counts(dataset, class_names):
    """
    Get total and class-wise positive sample count of a dataset
    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test
    class_names - list of str, target classes
    Returns:
    total_count - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    """
    df = pd.read_csv("train_f.csv")
    total_count = df.shape[0]
    labels = df[class_names].as_matrix()
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))
    return total_count, class_positive_counts

newfds = 'newfds'
train_counts, train_pos_counts = get_sample_counts("train", CLASS_NAMES)
class_weights = get_class_weights(train_counts, train_pos_counts, multiply=1)



f = data_generator_variable_input_img(train_df,y_train_bin,batch_size = 1)
f.__next__()
# DenseNet Dependencies
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense,Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC, BinaryAccuracy, FalsePositives
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras import backend as K



# Hyperparameters

IMG_SHAPE = (224,224,3)

EPOCHS = 100

STEPS_PER_EPOCH = 10

OPTIMIZER = Adam(learning_rate=0.001, ####### Modified
                 beta_1=0.9,
                 beta_2=0.999)

LOSS = BinaryCrossentropy() # Not un-weighted 

METRICS = ['BinaryAccuracy']

from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping

from tensorflow.keras.models import load_model

callback = [
        ReduceLROnPlateau(patience=5, verbose=1),
        ModelCheckpoint("chest_xray_model_v2_weights.h5",
                        save_best_only=True,
                        save_weights_only=False),ModelCheckpoint("chest_xray_model_v2.h5",
                        save_best_only=True,
                        save_weights_only=True)]

# with tf.device('/GPU:0'):

base_model = DenseNet121(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
                                               #pooling="avg")

base_model.trainable = True

x = base_model.output

x = GlobalAveragePooling2D()(x)
predictions = Dense(14, activation='sigmoid',name='Final')(x)

model = Model(inputs=base_model.input, outputs=predictions)

#model.summary()

#latest = tf.train.latest_checkpoint(pre_weights)
#print('Loading Weights from file: ', latest)
#weights_DIR = '/kaggle/input/bruce-weights/brucechou1983_CheXNet_Keras_0.3.0_weights.h5'
#model.load_weights(weights_DIR)

model.compile(loss = LOSS,
              optimizer=OPTIMIZER,
              metrics=METRICS
                             )


# history = model.fit(
#                     train_ds,
#                     steps_per_epoch = STEPS_PER_EPOCH,
#                     validation_data= create_dataset(X_val, y_val_bin.astype(np.float32)),
#                     validation_steps = STEPS_PER_EPOCH / 10, 
#                     epochs=EPOCHS,
#                     class_weight = class_weights,
#                     callbacks=callback)len(y_train_bin)//1

history = model.fit_generator(data_generator_variable_input_img(train_df,y_train_bin,batch_size = 32),validation_data=data_generator_variable_input_img(val_df,y_val_bin,batch_size = 32),steps_per_epoch = 100,epochs=45 ,validation_steps=25,callbacks = callback,class_weight = class_weights)


F = data_generator_variable_input_img(train_df,y_train_bin,batch_size = 1)
F.__next__()[1].shape





# Necessary Dependencies
import numpy as np 
import pandas as pd 
from glob import glob
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from itertools import chain
from datetime import datetime
import statistics

print('Started')
print(os.listdir("/kaggle/input/txt-file"))
# Establish Directories 

if not os.path.exists('logs'):
    os.makedirs('logs')
    
if not os.path.exists('callbacks'):
    os.makedirs('callbacks')
    
CALLBACKS_DIR = '/kaggle/output/callbacks'
# Disease Names 
disease_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']
# Load Stanford Images Distribution Files

labels_train_val = pd.read_csv('/kaggle/input/txt-file/train_val_list.txt')
labels_train_val.columns = ['Image_Index']

labels_test = pd.read_csv('/kaggle/input/txt-file/test_list.txt')
labels_test.columns = ['Image_Index']


print(labels_train_val,labels_test)
print(len(labels_train_val),len(labels_test))

# NIH Dataset Labels CSV File 

labels_df = pd.read_csv('/kaggle/input/data/Data_Entry_2017.csv')

labels_df.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                  'Patient_Age', 'Patient_Gender', 'View_Position',
                  'Original_Image_Width', 'Original_Image_Height',
                  'Original_Image_Pixel_Spacing_X',
                  'Original_Image_Pixel_Spacing_Y', 'dfd']
print(labels_df)
labels_df['Finding_Labels'] = labels_df['Finding_Labels'].map(lambda x: x.replace('No Finding', str(None)))
labels_df['Finding_Labels'] = labels_df['Finding_Labels'].replace('None', None)
# Binarizes Each Disease Class in their own column
from tqdm import tqdm

for diseases in tqdm(disease_labels): #TQDM is a progress bar setting
    labels_df[diseases] = labels_df['Finding_Labels'].map(lambda result: 1 if diseases in result else 0)
#     print(labels_df['Finding_Labels'].map(lambda result: 1 if diseases in result else 0))
labels_df[diseases]
train_val_merge = pd.merge(left=labels_train_val, right=labels_df, left_on='Image_Index', right_on='Image_Index')

test_merge = pd.merge(left=labels_test, right=labels_df, left_on='Image_Index', right_on='Image_Index')



print(train_val_merge,test_merge)
# Splitting Finding Labels
train_val_merge['Finding_Labels'] = train_val_merge['Finding_Labels'].apply(lambda s: [l for l in str(s).split('|')])

test_merge['Finding_Labels'] = test_merge['Finding_Labels'].apply(lambda s: [l for l in str(s).split('|')])

print(test_merge)
# Mapping Images
num_glob = glob('/kaggle/input/data/*/images/*.png')
img_path = {os.path.basename(x): x for x in num_glob}

train_val_merge['Paths'] = train_val_merge['Image_Index'].map(img_path.get)
test_merge['Paths'] = test_merge['Image_Index'].map(img_path.get)
print(len(num_glob))
# Delete No Finding Class

#train_val_merge = train_val_merge.drop(train_val_merge[train_val_merge.Del == 1].index)
#test_merge = test_merge.drop(test_merge[test_merge.Del == 1].index)
# No Overlap in patients between the Train and Validation Data Sets
patients = np.unique(train_val_merge['Patient_ID'])
test_patients = np.unique(test_merge['Patient_ID'])

print('Number of Patients Between Train-Val Overall: ', len(patients))
print('Number of Patients Between Train-Val Overall: ', len(test_patients))
# Train-Validation Split 
train_df, val_df = train_test_split(patients,
                                   test_size = 0.0669,
                                   random_state = 2019,
                                    shuffle= True)  


print('No. of Unique Patients in Train dataset : ',len(train_df))
train_df = train_val_merge[train_val_merge['Patient_ID'].isin(train_df)]
print('Training Dataframe   : ', train_df.shape[0],' images')

print('\nNo. of Unique Patients in Validtion dataset : ',len(val_df))
val_df = train_val_merge[train_val_merge['Patient_ID'].isin(val_df)]
print('Validation Dataframe   : ', val_df.shape[0],' images')

print('\nNo. of Unique Patients in Testing dataset : ',len(test_patients))
test_df = test_merge[test_merge['Patient_ID'].isin(test_patients)]
print('Testing Dataframe   : ', test_df.shape[0],' images')
X_train = train_df['Paths']
y_train = train_df['Finding_Labels']

X_val = val_df['Paths']
y_val = val_df['Finding_Labels']

X_test = test_df['Paths']
y_test = test_df['Finding_Labels']
print(y_train)
XX_train = []
for i in X_train[:]:
    XX_train.append(i)
print(len(XX_train))

X_train = XX_train
# Binarizing Labels 
from sklearn.preprocessing import MultiLabelBinarizer

print("Labels - ")
mlb = MultiLabelBinarizer()
mlb.fit(y_train)

N_LABELS = len(mlb.classes_)
print(N_LABELS,mlb.classes_)
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))
train_df.to_csv("train_f.csv")
IMG_IND = 224
BATCH_SIZE = 32

# Print a link to Tensorflow Data 
import tensorflow as tf

# Checkout Tensorflow.Data for more elaboration
def parse_function(filename, label):

    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_png(image_string, channels=3) # channels=3
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_IND, IMG_IND])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    
    return image_normalized, label
AUTOTUNE = tf.data.experimental.AUTOTUNE 
SHUFFLE_BUFFER_SIZE = 1280 
def create_dataset(filenames, labels):

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    

    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset
# os.makedirs("/kaggle/working/"+"images_001"+"/"+"images")
y_train_bin = mlb.transform(y_train)
def data_generator_variable_input_img(img_list,y_train_bin,batch_size = 1):
    
    while True:
        
        Y = []
        
        np.random.seed(2019)
        paths = np.random.choice(img_list,batch_size)
        np.random.seed(2019)
        labels =  np.random.choice(y_train_bin,batch_size)
        
        for img in paths:


            I = cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2RGB)
            
#             i128 = cv2.resize(I,(128,128))
            
            i512 = cv2.resize(I,(512,512))
        
            im = Image.fromarray(i512)
            # rotating a image 90 deg counter clockwise 
            im1 = im.rotate(90) 
            im2 = im.rotate(180) 
            im3 = im.rotate(270) 
#             i608 = cv2.resize(I,(608,608))
            
            label1 = int(LABEL_DICT[img.split("/")[-2]])
            
            label = keras.utils.to_categorical(label1, num_classes=6, dtype='float32')
            
#             label = keras.utils.to_categorical(label1, num_classes=6, dtype='float32')
#             label = np.zeros((1,6))
#             j = label1+1
#             label[0,:j] = 1
            
#             I128.append(i128)
#             I256.append(i256)

            I512_0.append(i512)
            I512_90.append(np.array(im1))
            I512_180.append(np.array(im2))
            I512_270.append(np.array(im3))
        

            Y.append(label)
            
#             print(label)
            
            y_smooth = smooth_labeld_array(np.array(Y),num_classes = 6)

        yield([np.array(I512_0)/255.,np.array(I512_90)/255.,np.array(I512_180)/255.,np.array(I512_270)  /255.],y_smooth.astype("float32"))
                           
                 
y_train_bin
# transform the targets of the training and test sets
y_train_bin = mlb.transform(y_train)
# for k in range(len(y_train_bin)):
    
    
#     path_name = X_train[k].split("/")[-3]
#     foldr_name = X_train[k].split("/")[-2]
#     file_name = X_train[k].split("/")[-1]
    
#     if not os.path.exists("/kaggle/working/"+str(path_name)):
#         os.makedirs("/kaggle/working/"+str(path_name))
#     if not os.path.exists("/kaggle/working/"+str(path_name)+"/"+str(foldr_name)):
#         os.makedirs("/kaggle/working/"+str(path_name)+"/"+str(foldr_name))
# #     print(path_name)
#     if os.path.exists("/kaggle/working/"+str(path_name)) and os.path.exists("/kaggle/working/"+str(path_name)+"/"+str(foldr_name)) :
#         im = cv2.imread(X_train[k])
#         cv2.imwrite("/kaggle/working/"+str(path_name)+"/"+str(foldr_name)+"/"+file_name,im)
#         print(y_train_bin[k],X_train[k])
        
#     cv2.imwrite("/kaggle/working/save_img/x.jpg"
# print(y_train_bin)
y_val_bin = mlb.transform(y_val)
y_test_bin = mlb.transform(y_test)
import glob

print(len(glob.glob("/kaggle/input/data/*/*/*/*.jpg")))
train_ds = create_dataset(X_train, y_train_bin.astype(np.float32))
val_ds = create_dataset(X_val, y_val_bin.astype(np.float32))
test_ds = create_dataset(X_test, y_test_bin.astype(np.float32))
# Load in Stanford Distribution charts of images 
# Print # of diseases in each Train-Val-Test
### Turn this into a function ###

print('# of Diseases in each Class - Training')
for i in disease_labels:
    print(i,int(train_df[i].sum()))

print('\n')

print('# of Diseases in each Class - Validation')
for i in disease_labels:
    print(i,int(val_df[i].sum()))

print('\n')

print('# of Diseases in each Class - Testing')
for i in disease_labels:
    print(i,int(test_df[i].sum()))

print('\n')
# Visualize a couple of images 
import matplotlib.style as style
from PIL import Image


nobs = 6 # Maximum number of images to display
ncols = 3 # Number of columns in display
nrows = nobs//ncols # Number of rows in display
'''
style.use("default")
plt.figure(figsize=(8,4*nrows))
for i in range(nrows*ncols):
    ax = plt.subplot(nrows, ncols, i+1)
    plt.imshow(Image.open(X_train[i]))
    plt.title(y_train[i], size=10)
    plt.axis('off')
    '''
# Show Original vs Downsampled Images
# Show Random Horizontal Flip

for i in range(4):
    #print(X_train[i], y_train_bin[i])
    #print(X_val[i], y_val_bin[i])
    print(X_test[i], y_test_bin[i])
    
# DenseNet Dependencies
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense,Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC, BinaryAccuracy, FalsePositives
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras import backend as K
print("TensorFlow version: ", tf.__version__)
# Hyperparameters

IMG_SHAPE = (320,320,3)

EPOCHS = 100

STEPS_PER_EPOCH = 10

OPTIMIZER = Adam(learning_rate=0.001, ####### Modified
                 beta_1=0.9,
                 beta_2=0.999)

LOSS = BinaryCrossentropy() # Not un-weighted 

METRICS = ['BinaryAccuracy']
# Test if GPU present
print("Num GPUs Used: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Section of Code written by brucechou1983 - https://github.com/brucechou1983/CheXNet-Keras
# I have no experience with class weighting, brucechou1983 provided a very thorough explanation of the topic with example code
CLASS_NAMES = disease_labels
def get_class_weights(total_counts, class_positive_counts, multiply):
    """
    Calculate class_weight used in training
    Arguments:
    total_counts - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    multiply - int, positve weighting multiply
    use_class_balancing - boolean 
    Returns:
    class_weight - dict of dict, ex: {"Effusion": { 0: 0.01, 1: 0.99 }, ... }
    """
    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }

    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = []
    for i, class_name in enumerate(class_names):
        class_weights.append(get_single_class_weight(label_counts[i], total_counts))

    return class_weights

def get_sample_counts(dataset, class_names):
    """
    Get total and class-wise positive sample count of a dataset
    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test
    class_names - list of str, target classes
    Returns:
    total_count - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    """
    df = pd.read_csv("train_f.csv")
    total_count = df.shape[0]
    labels = df[class_names].as_matrix()
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))
    return total_count, class_positive_counts

newfds = 'newfds'
train_counts, train_pos_counts = get_sample_counts("train", CLASS_NAMES)
class_weights = get_class_weights(train_counts, train_pos_counts, multiply=1)

train_pos_counts
train_counts
class_weights
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # total number of patients (rows)
    N = labels.shape[0]
    
    positive_frequencies = np.sum(labels==1,axis=0)/N
    negative_frequencies = np.sum(labels==0,axis=0)/N

    ### END CODE HERE ###
    return positive_frequencies, negative_frequencies

# freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
# freq_pos


# pos_weights = freq_neg
# neg_weights = freq_pos
# Saves weights every 5 Epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=CALLBACKS_DIR, 
    verbose=1, 
    save_weights_only=True,
    period=1)
cp_callback1 = tf.keras.callbacks.ModelCheckpoint(
    filepath=CALLBACKS_DIR, 
    verbose=1, 
    save_weights_only=False,save_best_only = True,
    period=1)

tensorboard_callback = tf.keras.callbacks.TensorBoard("logs")

reduced_lr = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=.1,
                        patience=5,
                        verbose=1,
                        mode='min',
                        cooldown=0,
                        min_lr=1e-8 
                        )
with tf.device('/GPU:0'):

    base_model = tf.keras.applications.DenseNet121(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
                                                   #pooling="avg")

    base_model.trainable = True

    x = base_model.output

    x = GlobalAveragePooling2D()(x)
    predictions = Dense(14, activation='sigmoid',name='Final')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    #model.summary()
    
    #latest = tf.train.latest_checkpoint(pre_weights)
    #print('Loading Weights from file: ', latest)
    #weights_DIR = '/kaggle/input/bruce-weights/brucechou1983_CheXNet_Keras_0.3.0_weights.h5'
    #model.load_weights(weights_DIR)

    model.compile(loss = LOSS,
                  optimizer=OPTIMIZER,
                  metrics=METRICS
                                 )


history = model.fit(
                    train_ds,
                    steps_per_epoch = STEPS_PER_EPOCH,
                    validation_data= create_dataset(X_val, y_val_bin.astype(np.float32)),
                    validation_steps = STEPS_PER_EPOCH / 10, 
                    epochs=EPOCHS,
                    #use_multiprocessing=True,
                    class_weight = class_weights,
                    callbacks=[reduced_lr, tensorboard_callback, cp_callback,cp_callback1]
                    )
    
import pandas as pd
pd.__version__
model.save_weights("model_4may.h5")
os.getcwd()
# Graph of Binary Accuracy 

acc = history.history['BinaryAccuracy']
val_acc = history.history['val_BinaryAccuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(40, 10))
plt.subplot(1, 2, 1)
plt.grid()
plt.plot(epochs_range, acc, label='Training Binary Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Binary Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Binary Accuracy', color='Green')
#fig.savefig('TrainingValidationAccuracy.png')
# Graph of Loss
plt.figure(figsize=(40, 10))
plt.subplot(1, 2, 2)
plt.grid()
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss', color='red')
plt.show()
# Load Model
#model.load_weights(weights_DIR)
# Loading Latest Weights 
#latest = tf.train.latest_checkpoint(checkpoint_dir)
#print('Loading Weights from file: ', latest)
STEPS = len(y_test_bin)
print(STEPS)
# Predict
pred = model.predict(test_ds,steps=STEPS,verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)
y_test = y_test_bin
pred = pred
# Print AUC scores
labels = []
scores = []
for i in range(14):
    
    try:
        print('{0:<20}\t{1:<1.6f}'.format(disease_labels[i], roc_auc_score(y_test[:,i], pred[:,i])))
        labels.append(disease_labels[i])
        scores.append(roc_auc_score(y_test[:,i], pred[:,i]))
        
        
    except:
        print('Not Working')
    
print('done') 
# True Postive Vs ...
from sklearn.metrics import roc_curve, auc, roc_auc_score
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(disease_labels):
    fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), pred[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('Completed_Training_CheXNet.png')
# Bar Graph
import matplotlib
my_mean_aur = round(statistics.mean(scores), 3) * 100 
print(f'My AUROC mean score: {my_mean_aur}% ')

Stanford_scores = [0.8094, 0.7901, 0.7345, 0.8887, 0.8878, 0.9371, 0.8047, 0.8638, 0.7680, 0.8062, 0.9248, 0.7802, 0.8676, 0.9164]
stanford_mean_aur = round(statistics.mean(Stanford_scores), 3) * 100 
print(f'Stanford Auroc mean score: {stanford_mean_aur}% ')

labels = ['Stanford Mean AUROC', 'My Mean AUROC']
mean_scores = [stanford_mean_aur, my_mean_aur]

matplotlib.pyplot.bar(labels, mean_scores)


plt.show()
# Stanford MEAN AUROC Scores
stanford_labels = disease_labels
print(stanford_labels)
stanford_scores = Stanford_scores
print(stanford_scores)


# Disease Class Scores

labels = stanford_labels

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(40,20))
rects1 = ax.bar(x - width/2, stanford_scores, width, label='Stanford AUROC', color='green')
rects2 = ax.bar(x + width/2, scores, width, label='MY CHEXNET',color='orange')

ax.set_ylabel('Scores', color='grey', fontsize= 30)
ax.set_title('Comparing AUROC Scores')
ax.set_xticks(x)
ax.set_xlabel('Disease Classes', color='grey', fontsize= 30)
ax.set_xticklabels(labels, color='grey', fontsize= 17)
ax.legend()


plt.show()
fig.savefig('ComparisonAUROC.png')
# Notes



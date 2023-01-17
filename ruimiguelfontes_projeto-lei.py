from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense,MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
from sklearn.preprocessing import LabelEncoder
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from pandas.plotting import scatter_matrix
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from sklearn import preprocessing
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.models import Model
from keras import optimizers
from os.path import join
import tensorflow as tf
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import glob
import os
print(tf.executing_eagerly())
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.random.set_seed(666)
tf.keras.backend.clear_session()
# Ler os datasets
# Dataset de metadata
dfMetadata = pd.read_csv("../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")

# Tornar o id da lesão no índice
dfMetadata = dfMetadata.set_index('lesion_id')
dfMetadata
# Ver os ficheiros
print(os.listdir("../input/skin-cancer-mnist-ham10000/ham10000_images_part_2"))
image_height = 100 #150
image_width = 100 #150
batch_size = 32 #10
epochs  = 40 #10
# Tratamento dos dados

# Verificar o tipo dos dados no dataset
print(dfMetadata.dtypes)

# Verificar a existência de valores em falta
print(dfMetadata.isnull().sum())

dfMetadata.columns # ['image_id', 'dx', 'dx_type', 'age', 'sex', 'localization']
# Valores únicos
dfMetadata.nunique()

# Ocorrências dos diferentes valores nas colunas
# dfMetadata['sex'].value_counts()
#--------------------------------------------------------------------------------------------#
# age        | localization            | dx            | dx_type           | sex             |
#------------|-------------------------|---------------|-------------------|-----------------|
# 45    1299 | back               2192 | nv       6705 | histo        5340 | male       5406 |
# 50    1187 | lower extremity    2077 | mel      1113 | follow_up    3704 | female     4552 |
# 55    1009 | trunk              1404 | bkl      1099 | consensus     902 | unknown      57 |
# 40     985 | upper extremity    1118 | bcc       514 | confocal       69 |-----------------|
# 60     803 | abdomen            1022 | akiec     327 |-------------------|
# 70     756 | face                745 | vasc      142 |
# 35     753 | chest               407 | df        115 |
# 65     731 | foot                319 |---------------|
# 75     618 | unknown             234 |
# 30     464 | neck                168 |
# 80     404 | scalp               128 |
# 85     290 | hand                 90 |
# 25     247 | ear                  56 |
# 20     169 | genital              48 |
# 5       86 | acral                 7 |
# 15      77 |-------------------------|
# 10      41 |
# 0       39 |
#------------|

# Eliminar os valores em falta
dfMetadata = dfMetadata[dfMetadata['localization'] != 'unknown']
dfMetadata = dfMetadata[dfMetadata['sex'] != 'unknown']
# for x in dfMetadata['age']:
#     if x < 5:
#         dfMetadata = dfMetadata[dfMetadata['age'] != x]

dfMetadata['age'].value_counts()

# Ocorrências dos diferentes valores nas colunas depois de eliminados os valores em falta
#--------------------------------------------------------------------------------------------#
# age        | localization            | dx            | dx_type           | sex             |
#------------|-------------------------|---------------|-------------------|-----------------|
# 45    1276 | back               2190 | nv       6499 | histo        5282 | male       5314 |
# 50    1182 | lower extremity    2077 | mel      1103 | follow_up    3694 | female     4457 |
# 55     991 | trunk              1401 | bkl      1076 | consensus     728 |-----------------| 
# 40     922 | upper extremity    1118 | bcc       509 | confocal       67 |
# 60     786 | abdomen            1020 | akiec     327 |-------------------|
# 70     749 | face                745 | vasc      142 |
# 35     738 | chest               407 | df        115 |
# 65     730 | foot                316 |---------------|
# 75     611 | neck                168 | 
# 30     460 | scalp               128 |
# 80     399 | hand                 90 |
# 85     286 | ear                  56 |
# 25     223 | genital              48 |
# 20     169 | acral                 7 |
# 5       86 |-------------------------|
# 15      77 |
# 10      39 |
# 0       37 |
#------------|

# Corrigir a coluna age
encodeAge = {"age":{0.0: 10.0, 5.0: 10.0, 10.0: 10.0}}
dfMetadata.replace(encodeAge, inplace=True)
dfMetadata.head()
dfMetadata['age'].value_counts()

# Fazer label encoding 
# Encoding da coluna localization, dx, dx_type e sex
encode = {"localization":{"back": 0, "lower extremity": 1, "trunk": 2, "upper extremity": 3, "abdomen": 4, "face": 5, "chest": 6, "foot": 7, "neck": 8, "scalp": 9, "hand": 10, "ear": 11, "genital": 12, "acral": 13},
            "dx":{"nv": 0, "mel": 1, "bkl": 2, "bcc": 3, "akiec": 4, "vasc": 5, "df": 6},
            "dx_type":{"histo": 0, "follow_up": 1, "consensus": 2, "confocal": 3},
            "sex":{"male": 0, "female": 1}}
dfMetadata.replace(encode, inplace=True)
dfMetadata.head()

# Representação dos dados para verificar se existem outliers
dfMetadata.copy().plot(kind='box', subplots=True, layout=(2,7), sharex=False, sharey=False)
#plt.savefig('outliers.png', bbox_inches='tight', pad_inches=0.0)
plt.show()

# Correlação entre os dados
corrMatrix = dfMetadata.corr(method='pearson')
sns.heatmap(corrMatrix, annot=True)
plt.savefig('corr.png', bbox_inches='tight', pad_inches=0.0)
plt.show()

# drop dx_type dos dados
del dfMetadata['dx_type']

base_dir = '..'
image_extension_pattern = '*.jpg'
image_paths = sorted((y for x in os.walk(base_dir) for y in
                          glob.glob(join(x[0], image_extension_pattern))))

combined_img_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in image_paths}


dfMetadata['path_of_image'] = dfMetadata['image_id'].map(combined_img_dict.get)

#abrir imagem
dfMetadata['image']=dfMetadata['path_of_image'].map(lambda x: np.asarray(Image.open(x).resize((100,100))))

#remover colunas e vairaveis nao utilizadas a partir de agora
del dfMetadata['image_id']
del dfMetadata['path_of_image']
del image_paths
del image_extension_pattern
del combined_img_dict
del base_dir

# divisao entre treino e teste
features=dfMetadata.drop(columns=['dx'],axis=1)
target=dfMetadata['dx']
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20,random_state=666)
del dfMetadata
del features
del target

#Normalization
x_train_image = np.asarray(x_train['image'].tolist())
x_test_image = np.asarray(x_test['image'].tolist())

x_train_mean = np.mean(x_train_image)
x_train_std = np.std(x_train_image)

x_test_mean = np.mean(x_test_image)
x_test_std = np.std(x_test_image)

x_train_image = (x_train_image - x_train_mean)/x_train_std
x_test_image = (x_test_image - x_test_mean)/x_test_std

del x_train['image']
del x_test['image']

#one hot encode
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
x_train_image
#dfMetadata = dfMetadata.drop(dfMetadata[dfMetadata.cell_type_idx == 4].iloc[:5000].index)

#fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
#dfMetadata['cell_type'].value_counts().plot(kind='bar', ax=ax1)
# Data augmentation
# Rotação e rescaling
# train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15)
# test_datagen = ImageDataGenerator(rescale=1./255)
# #val_datagen = ImageDataGenerator(rescale=1./255)

# #######################################################

# training_set = train_datagen.flow_from_directory(dfMetadata['image'],
#                                                  target_size=(image_width, image_height),
#                                                  batch_size=batch_size,
#                                                  class_mode='categorical')
####################################################
# test_set = test_datagen.flow_from_directory(base_skin_dir+'/test',
#                                             target_size=(image_width, image_height),
#                                             batch_size=batch_size,
#                                             class_mode='categorical')
#inputs do modelo

image_input = Input((100, 100, 3),name='image_input')
vector_input = Input((3,),name='vector_input')

#model CNN
conv_layer_1 = Conv2D(32, (3,3),padding='same',activation='relu')(image_input)
conv_layer_2 = Conv2D(32, (3,3),padding='same',activation='relu')(conv_layer_1)

maxpool_1= MaxPooling2D((2, 2))(conv_layer_2)
drop1=Dropout(0.2)(maxpool_1)

conv_layer_3 = Conv2D(32, (3,3),padding='same',activation='relu')(drop1)
conv_layer_4 = Conv2D(32, (3,3),padding='same',activation='relu')(conv_layer_3)

maxpool_2= MaxPooling2D((2, 2))(conv_layer_4)
drop2=Dropout(0.3)(maxpool_2)

flatten = Flatten()(drop2)

#model MLP
mlp_1= Dense(3,activation='relu')(vector_input)
mlp_2= Dense(10,activation='relu')(mlp_1)
mlp_3= Dense(10,activation='relu')(mlp_2)

#concatenate models anteriores
concat_layer= Concatenate()([flatten, mlp_3])
mlp_4=Dense(1028,activation='relu')(concat_layer)
drop3=Dropout(0.8)(mlp_4)

#mlp_6=Dense(200)(drop)

out=Dense(7,activation='softmax')(drop3)

model = Model(inputs=[image_input, vector_input], outputs=out)
###### Pa ver coisas bonitas
def print_history_accuracy(history):
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def print_history_loss(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

early_stopping_monitor = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=20,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

model.summary()

####################################################################
# nv       6499 0,6499%
# mel      1103 0,1103%
# bkl      1076 0,1076
# bcc       509 0,0509
# akiec     327 0,0327
# vasc      142 0,0142
# df        115 0,0115
# 9771

weight_for_0 = 1/7
weight_for_1 = 3/7
weight_for_2 = 3/7
weight_for_3 = 5/7
weight_for_4 = 5/7
weight_for_5 = 6/7
weight_for_6 = 6/7

class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2, 3: weight_for_3, 4: weight_for_4, 5: weight_for_5, 6: weight_for_6}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))
print('Weight for class 2: {:.2f}'.format(weight_for_2))
print('Weight for class 3: {:.2f}'.format(weight_for_3))
print('Weight for class 4: {:.2f}'.format(weight_for_4))
print('Weight for class 5: {:.2f}'.format(weight_for_5))
print('Weight for class 6: {:.2f}'.format(weight_for_6))
####################################################################

# Criar o modelo
opt= optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer = opt ,loss = "categorical_crossentropy", metrics=["accuracy"])

# Treinar o modelo
history=model.fit({'image_input':x_train_image,'vector_input':x_train},y_train,validation_data=([x_test_image,x_test], y_test),
                  epochs=epochs,
                  batch_size=batch_size, 
                  class_weight=class_weight,
                  callbacks=[early_stopping_monitor])
model.save('my_model.h5') 
print_history_accuracy(history)
print_history_loss(history)
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
y_pred=model.predict([x_test_image,x_test])
y_pred=np.round(y_pred)
y_pred
y_test
cm=confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
df_cm = pd.DataFrame(cm, range(7), range(7))
sns.set(font_scale=1.4) # for label size
plt.figure(figsize=(16,5))
sns.heatmap(df_cm,annot=True, annot_kws={"size": 15}) # font size
plt.show()

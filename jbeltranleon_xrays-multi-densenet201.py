import os

try:

    inpath = "../input/data/" #Kaggle

    print(os.listdir(inpath))

except FileNotFoundError:

    inpath = "./" #Local

    print(os.listdir(inpath))
import pandas as pd

data = pd.read_csv(inpath + 'Data_Entry_2017.csv')

print(f"Las dimensiones del conjunto de datos son: {data.shape}")



data.head()
data = data[data['Patient Age']<100]



print(f"Las dimensiones del conjunto de datos son: {data.shape}")
data = data[['Image Index', 'Finding Labels']]



print(f"Las dimensiones del conjunto de datos son: {data.shape}")
"""

Leemos todas las rutas de las imagenes

"""

from glob import glob

all_image_paths = {os.path.basename(x): x for x in 

                   glob(os.path.join('..', 'input','nih-chest-xrays-224-gray', 'images*', '*.png'))}

print('Imágenes encontradas:', len(all_image_paths))



"""

Agregamos la columna path al conjunto de datos

"""

data['Path'] = data['Image Index'].map(all_image_paths.get)



data.sample(5, random_state=3)
"""

Create a np array with all the single deseases

"""

import numpy as np

from itertools import chain

all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))



all_labels
all_labels = np.delete(all_labels, np.where(all_labels == 'No Finding'))

print(f'Tipo actual: {type(all_labels)}')



all_labels = [x for x in all_labels]

print(f'Tipo final: {type(all_labels)}')



print(f'Enfermedades: ({len(all_labels)}): {all_labels}')
"""

Agregamos una columna, por cada enfermedad

"""

for c_label in all_labels:

    if len(c_label)>1: # leave out empty labels

        # Add a column for each desease

        data[c_label] = data['Finding Labels'].map(lambda finding: 1 if c_label in finding else 0)

        

print(f"Las dimensiones del conjunto de datos son: {data.shape}")

data.head()
label_counts = data['Finding Labels'].value_counts()

label_counts
data = data.groupby('Finding Labels').filter(lambda x : len(x)>11)
label_counts = data['Finding Labels'].value_counts()

print(label_counts.shape)

label_counts
from sklearn.model_selection import train_test_split



train_and_valid_df, test_df = train_test_split(data,

                                               test_size = 0.30,

                                               random_state = 2018,

                                              )



train_df, valid_df = train_test_split(train_and_valid_df,

                                      test_size=0.30,

                                      random_state=2018,

                                     )



print(f'Entrenamiento {train_df.shape[0]} Validación {valid_df.shape[0]} Prueba: {test_df.shape[0]}')
from keras_preprocessing.image import ImageDataGenerator

base_generator = ImageDataGenerator(rescale=1./255)
IMG_SIZE = (224, 224)

def flow_from_dataframe(image_generator, dataframe, batch_size):



    df_gen = image_generator.flow_from_dataframe(dataframe,

                                                 x_col='Path',

                                                 y_col=all_labels,

                                                 target_size=IMG_SIZE,

                                                 classes=all_labels,

                                                 color_mode='rgb',

                                                 class_mode='raw',

                                                 shuffle=False,

                                                 batch_size=batch_size)

    

    return df_gen
train_gen = flow_from_dataframe(image_generator=base_generator, 

                                dataframe= train_df,

                                batch_size = 32)



valid_gen = flow_from_dataframe(image_generator=base_generator, 

                                dataframe=valid_df,

                                batch_size = 32)



test_gen = flow_from_dataframe(image_generator=base_generator, 

                               dataframe=test_df,

                               batch_size = 32)
train_x, train_y = next(train_gen)

print(f"Dimensiones de la imagen: {train_x[1].shape}")

print(f"Vector de enfermedades: {train_y[1]}")
from keras.layers import Input

from keras.applications.densenet import Dense201

from keras.layers.core import Dense

from keras.models import Model



input_shape=(224, 224, 3)

img_input = Input(shape=input_shape)



base_model = Dense201(include_top=False, input_tensor=img_input, input_shape=input_shape, 

                         pooling="avg", weights='imagenet')

x = base_model.output

predictions = Dense(len(all_labels), activation="sigmoid", name="predictions")(x)

model = Model(inputs=img_input, outputs=predictions)
from contextlib import redirect_stdout



with open('model_summary.txt', 'w') as f:

    with redirect_stdout(f):

        model.summary()
from keras.callbacks import ModelCheckpoint

model_train = model

output_weights_name='weights.h5'

checkpoint = ModelCheckpoint(

             output_weights_name,

             save_weights_only=True,

             save_best_only=True,

             verbose=1,

            )
import keras.backend as kb

from keras.callbacks import Callback

from sklearn.metrics import roc_auc_score

import shutil

import warnings

import json



class MultipleClassAUROC(Callback):

    """

    Monitor mean AUROC and update model

    """

    def __init__(self, generator, class_names, weights_path, stats=None):

        super(Callback, self).__init__()

        self.generator = generator

        self.class_names = class_names

        self.weights_path = weights_path

        self.best_weights_path = os.path.join(

            os.path.split(weights_path)[0],

            f"best_{os.path.split(weights_path)[1]}",

        )

        self.best_auroc_log_path = os.path.join(

            os.path.split(weights_path)[0],

            "best_auroc.log",

        )

        self.stats_output_path = os.path.join(

            os.path.split(weights_path)[0],

            ".training_stats.json"

        )

        # for resuming previous training

        if stats:

            self.stats = stats

        else:

            self.stats = {"best_mean_auroc": 0}



        # aurocs log

        self.aurocs = {}

        for c in self.class_names:

            self.aurocs[c] = []



    def on_epoch_end(self, epoch, logs={}):

        """

        Calcula el promedio de las Curvas ROC y guarda el mejor grupo de pesos

        de acuerdo a esta metrica

        """

        print("\n*********************************")

        self.stats["lr"] = float(kb.eval(self.model.optimizer.lr))

        print(f"Learning Rate actual: {self.stats['lr']}")



        """

        y_hat shape: (#ejemplos, len(etiquetas))

        y: [(#ejemplos, 1), (#ejemplos, 1) ... (#ejemplos, 1)]

        """

        y_hat = self.model.predict_generator(self.generator,steps=self.generator.n/self.generator.batch_size)

        y = self.generator.labels



        print(f"*** epoch#{epoch + 1} Curvas ROC Fase Entrenamiento ***")

        current_auroc = []

        for i in range(len(self.class_names)):

            try:

                score = roc_auc_score(y[:, i], y_hat[:, i])

            except ValueError:

                score = 0

            self.aurocs[self.class_names[i]].append(score)

            current_auroc.append(score)

            print(f"{i+1}. {self.class_names[i]}: {score}")

        print("*********************************")



        mean_auroc = np.mean(current_auroc)

        print(f"Promedio Curvas ROC: {mean_auroc}")

        if mean_auroc > self.stats["best_mean_auroc"]:

            print(f"Actualización del resultado de las Curvas de ROC de: {self.stats['best_mean_auroc']} a {mean_auroc}")



            # 1. copy best model

            shutil.copy(self.weights_path, self.best_weights_path)



            # 2. update log file

            print(f"Actualización del archivo de logs: {self.best_auroc_log_path}")

            with open(self.best_auroc_log_path, "a") as f:

                f.write(f"(epoch#{epoch + 1}) auroc: {mean_auroc}, lr: {self.stats['lr']}\n")



            # 3. write stats output, this is used for resuming the training

            with open(self.stats_output_path, 'w') as f:

                json.dump(self.stats, f)



            print(f"Actualización del grupo de pesos: {self.weights_path} -> {self.best_weights_path}")

            self.stats["best_mean_auroc"] = mean_auroc

            print("*********************************")

        return
training_stats = {}

auroc = MultipleClassAUROC(

    generator=valid_gen,

    class_names=all_labels,

    weights_path=output_weights_name,

    stats=training_stats

)
from keras.optimizers import Adam

initial_learning_rate=1e-3

optimizer = Adam(lr=initial_learning_rate)

model_train.compile(optimizer=optimizer, loss="binary_crossentropy")
from keras.callbacks import TensorBoard, ReduceLROnPlateau

#TODO - VALIDATE THE LOGS OUTPUT

logs_base_dir = '../working/'

patience_reduce_lr=2

min_lr=1e-8

callbacks = [

            checkpoint,

            TensorBoard(log_dir=os.path.join(logs_base_dir, "logs"), batch_size=train_gen.batch_size),

            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_reduce_lr,

                              verbose=1, mode="min", min_lr=min_lr),

            auroc,

        ]
epochs=20

fit_history = model.fit_generator(

    generator=train_gen,

    steps_per_epoch=train_gen.n/train_gen.batch_size,

    epochs=epochs,

    validation_data=valid_gen,

    validation_steps=valid_gen.n/valid_gen.batch_size,

    callbacks=callbacks,

    shuffle=False

)
import matplotlib.pyplot as plt



plt.figure(1, figsize = (15,8)) 

    

plt.subplot(222)  

plt.plot(fit_history.history['loss'])  

plt.plot(fit_history.history['val_loss'])  

plt.title('model loss')  

plt.ylabel('loss')  

plt.xlabel('epoch')  

plt.legend(['train', 'valid']) 



plt.show()
model.load_weights('weights.h5')
pred_y = model.predict_generator(test_gen, steps=test_gen.n/test_gen.batch_size, verbose = True)
test_gen.reset()

test_x, test_y = next(test_gen)

print(f"Vector de enfermedades: {test_y[1]}")

print(f"Vector de enfermedades producto de la predicción: {pred_y[2]}")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

test_gen.reset()

test_x, test_y = next(test_gen)

# Space

fig, c_ax = plt.subplots(1,1, figsize = (9, 9))

for (idx, c_label) in enumerate(all_labels):

    #Points to graph

    fpr, tpr, thresholds = roc_curve(test_gen.labels[:,idx].astype(int), pred_y[:,idx])

    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))

    

#convention

c_ax.legend()



#Labels

c_ax.set_xlabel('False Positive Rate')

c_ax.set_ylabel('True Positive Rate')



# Save as a png

fig.savefig('barely_trained_net.png')
from sklearn.metrics import roc_auc_score

# ROC AUC

auc = roc_auc_score(test_gen.labels, pred_y)

print('ROC AUC: %f' % auc)
sickest_idx = np.argsort(np.sum(test_y, 1)<1)



#Space of images

fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))



# Padding

fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

counter = 0



for (idx, c_ax) in zip(sickest_idx, m_axs.flatten()):

    

    # Image show

    c_ax.imshow(test_x[idx, :,:,0], cmap = 'bone')

    

    stat_str = [n_class[:4] for n_class, n_score in zip(all_labels, test_y[idx]) if n_score>0.5]

        

    # Building the labels

    pred_str = [f'{n_class[:4]}:{p_score*100:.2f}%'

                for n_class, n_score, p_score 

                in zip(all_labels,test_y[idx],pred_y[idx]) 

                if (n_score>0.5) or (p_score>0.5)]

    

    c_ax.set_title(f'Index {idx}, Labels: '+', '.join(stat_str)+'\n Pred: '+', '.join(pred_str))

    c_ax.axis('off')

fig.savefig('trained_img_predictions.png')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import pickle



import matplotlib.pylab as plt

import time

import json

import keras

from keras.models import Model, Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D

from keras.layers import Input, Concatenate, Flatten, UpSampling2D

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2

from keras.preprocessing.image import load_img

import keras.backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam, SGD

from keras.utils import np_utils

from keras.utils import plot_model

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint

from keras_applications.resnext import ResNeXt50

from keras_applications.resnet import ResNet50

from keras.initializers import VarianceScaling

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

from keras.models import load_model

import gc

""""""

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Préparation des données et augmentation, création du modèle (questions 1->8 partie A)

def get_data(trainPath, testPath):

    """Cette fonction retourne les données d'apprentissage et de text

    params:

        ---> trainPath : chemin de la directory des images d'apprentissage

        ---> trainPath : chemin de la directory des images de test

    retour :

        ---> trainGen : générateur d'image d'apprentissage

        ---> trainGen : générateur d'image de test

        ---> train_x : tableau d'image d'apprentissage

        ---> train_y: tableau des classes d'apprentissage

        ---> test_x : tableau d'image de test

        ---> test_y : tableau des classes de test

    """

    

    # instancier un objet ImageDataGenerator pou l'augmentation des donnees train

    trainAug = ImageDataGenerator(rescale = 1./255, horizontal_flip=True,fill_mode="nearest")

    testAug = ImageDataGenerator(rescale = 1./255)

    

    # definir la moyenne des images ImageNet par plan RGB pour normaliser les images de la base AFF20

    mean = np.array([123.68, 116.779, 103.939], dtype="float32")/255

    trainAug.mean = mean

    testAug.mean = mean



    # initialiser le generateur de train

    trainGen = trainAug.flow_from_directory(

    trainPath,

    class_mode="categorical",

    target_size=(224, 224),

    color_mode="rgb",

    shuffle=True,

    batch_size=16)



    # initialiser le generateur de test

    testGen = testAug.flow_from_directory(

    testPath,

    class_mode="categorical",

    target_size=(224, 224),

    color_mode="rgb",

    shuffle=False,

    batch_size=16)

    

    

    #Lire les données sous forme de tableaux numpy, pour l'évalusation

    #puisque la fonction fit de la class gridsearchcv prend en paramétre des

    #tableaux et non pas des générateur.

    

    #pour cette partie on peut bien lire la base de données manuelement (des boucle for)

    #mais dans ce cas on fera l'évaluation avec des données non augmenter, et l'apprentissage

    #avec des données augmenter. pour cela on extrait les tableaux à partir des générateur eux même.

    #c'est aussi plus rapide que d'utiliser des boucles.

    

    #les dimension des deux bases

    n_train = trainGen.samples

    n_test = testGen.samples

    

    # initialiser le generateur de train

    trainGen_tmp = trainAug.flow_from_directory(

    trainPath,

    class_mode="categorical",

    target_size=(224, 224),

    color_mode="rgb",

    shuffle=True,

    batch_size=n_train)



    # initialiser le generateur de test

    testGen_tmp = testAug.flow_from_directory(

    testPath,

    class_mode="categorical",

    target_size=(224, 224),

    color_mode="rgb",

    shuffle=False,

    batch_size=n_test)

    

    

    train_x = trainGen_tmp.next()[0]

    train_y = trainGen_tmp.next()[1]

    

    test_x = testGen_tmp.next()[0]

    test_y = testGen_tmp.next()[1]

    

    print("x_train_shape:",train_x.shape)

    print("y_train_shape:",test_y.shape)

    

    print("x_test_shape:",test_x.shape)

    print("y_test_shape:",test_y.shape)

    

    return trainGen,testGen, train_x, train_y, test_x, test_y





def DB_layer(x, bl_number, layer_number, nb_filter = 16, dropout_rate = 0.2, weight_decay=1E-4):

    """une couche du block DB comme décrit dans l'énoncé"""

    

    bl_number = str(bl_number)

    layer_number = str(layer_number)

    x = BatchNormalization(axis=-1, name = "BatchNorm_block"+bl_number+"_layer"+layer_number)(x)#, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay)

    x = Activation('relu', name="Relu_block"+bl_number+"_layer"+layer_number)(x)

    x = Conv2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,name="Conv2D_block"+bl_number+"_layer"+layer_number)(x)#, kernel_regularizer=l2(weight_decay)

    x = Dropout(dropout_rate,name="Dropout_block"+bl_number+"_layer"+layer_number)(x)

    return x





def TD_block(x, dropout_rate=0.2, weight_decay=1E-4):

    """un block TD comme décrit dans l'énoncé"""

    

    #le block TD dévise le nombre de chanels par la moitié

    nb_filter = int(x.shape[-1]/2)

    

    x = BatchNormalization(axis=-1,name="BatchNorm_TD1")(x)

    x = Activation('relu' , name="Relu_TD1")(x)

    x = Conv2D(nb_filter, (1, 1),kernel_initializer="he_uniform",padding="same",use_bias=False, name="Conv2D_TD1")(x)#,kernel_regularizer=l2(weight_decay)

    x = Dropout(dropout_rate, name="Dropout_TD1")(x)

    x = MaxPooling2D((2, 2), strides=(2,2), name="MaxPool_TD1")(x)



    return x





def DB_block(x, bl_number, nb_layers = 4, nb_filter = 16, growth_rate = 16):

    """un block DB comme décrit dans l'énoncé"""

    list_feat = [x]

    for i in range(nb_layers):

        x = DB_layer(x,bl_number,i+1)

        list_feat.append(x)

        x = Concatenate(axis=-1)(list_feat)

    return x



def DenseNet():

    """Définir le modele Mini DensNet comme décrit dans l'énoncé"""

    growth_rate = 16

    nb_filter = 16

    dropout_rate=0.2

    weight_decay=1E-4

    model_input = Input(shape=img_dim)

    

    # convolution initial de 48 filtre

    x = Conv2D(48, (3, 3),kernel_initializer="he_uniform",padding="same", name="initial_conv2D", use_bias=False)(model_input)#,kernel_regularizer=l2(weight_decay)



    # DB bloc 1

    x = DB_block(x,bl_number=1)

    

    

    # bloc de  transition

    x = TD_block(x)

    

    # DB bloc 2

    x = DB_block(x,bl_number=2)

    

    #la sortie aprés le deuxiem block DB a une dimension de (112,112,120)

    #d'une part (112,112,120) = 1 505 280 > (224,244,3) = 150 528, 

    #alors le vecteur descripteur et plus grand 10 fois que l'image initial se qui ne semble pas normale

    #d'autre part passer d'une couche de 1505280 neurons à une avec 20 neurons (20 classes) semble trops dégrader

    

    

    #cette couche représente les vecteurs déscripteurs

    x = Flatten()(x)

    # couche FC

    x = Dense(nb_classes, activation='softmax', name="Finale_dense")(x)



    densenet = Model(inputs=[model_input], outputs=[x], name="DenseNet")

    

    # optimizer

    opt = SGD(learning_rate=0.001, momentum=0.0, decay=decay_learning_rate)

    densenet.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    

    return densenet





#Evaluation du modèle miniDenseNet (question 1)



#question a

def GridSearchCV_on_epoches(define_model, epoch_nbrs = [10, 20] ):

    """appliquer l'évaluation sur le nombre d'epoche et afficher le résultat,

    

    Ps : Nous avons généralisé la fonction pour la réutiliser

    pour les modele suivant(Transfer-learning)

    

    params :

        --->define_model: fonction qui retourne un modele keras

    retour :

        --->grid_result : résultat de l'évaluation

    """

    param_grid_epoches = dict(epochs=epoch_nbrs)

    

    model = KerasClassifier(build_fn=define_model, batch_size=batch_size)

    

    grid_epoche = GridSearchCV(estimator=model, param_grid = param_grid_epoches, cv=3)

    grid_result = grid_epoche.fit(train_x_resized, train_y_resized)

    

    # afficher les resultats

    print("Best: %f with %s" % (grid_result.best_score_, grid_result.best_params_))

    # afficher les resultats detailles

    means = grid_result.cv_results_['mean_test_score']

    stds = grid_result.cv_results_['std_test_score']

    params = grid_result.cv_results_['params']

    

    for mean, stdev, param in zip(means, stds, params):

        print("mean (+/- std) = %f (%f) with: %r" % (mean, stdev, param))

    

    return grid_result



#question b

def GridSearchCV_on_batches(define_model, batch_sizes = [16, 32, 64]):

    """appliquer l'évaluation sur la taille des batchs et afficher le résultat,

    

    Ps : Nous avons généralisé la fonction pour la réutiliser

    pour les modele suivant(Transfer-learning)

    

    params :

        --->define_model: fonction qui retourne un modele keras

    retour :

        --->grid_result : résultat de l'évaluation

        """

    param_grid_batch_sizes = dict(batch_size=batch_sizes)

    

    model = KerasClassifier(build_fn=define_model, epochs=epochs)

    

    grid_batches = GridSearchCV(estimator=model, param_grid = param_grid_batch_sizes, cv=3)

    grid_result = grid_batches.fit(train_x_resized, train_y_resized)

    

    # afficher les resultats

    print("Best: %f with %s" % (grid_result.best_score_, grid_result.best_params_))

    # afficher les resultats detailles

    means = grid_result.cv_results_['mean_test_score']

    stds = grid_result.cv_results_['std_test_score']

    params = grid_result.cv_results_['params']

    

    for mean, stdev, param in zip(means, stds, params):

        print("mean (+/- std) = %f (%f) with: %r" % (mean, stdev, param))

    

    return grid_result



#question c



#fonctions de dégradations du taux d'apprentissage 

def standard_decay(curr_epoch):

    """Dégradation standard"""

    init_eta = learning_rate

    decay = decay_learning_rate

    nb_batches = int( train_shape / batch_size)

    eta = init_eta / (1+decay*(nb_batches*curr_epoch))

    return eta



def step_decay(curr_epoch):

    """Dégradation par pas"""

    init_eta = learning_rate

    drop_factor = 0.15

    drop_every = 10

    decay = np.floor((1 + curr_epoch) / drop_every)

    eta = init_eta * drop_factor ** decay

    return eta



def poly_decay(curr_epoch):

    """Dégradation polynômiale"""

    total_epochs = epochs

    init_eta = learning_rate

    order = 1.0

    decay = (1 - (curr_epoch / float(total_epochs))) ** order

    eta = init_eta * decay

    return eta



def GridSearchCV_on_decay(define_model):

    """appliquer l'évaluation sur les fonctions de dégradation

    de taux d'apprentissage et afficher le résultat,

    

    Ps : Nous avons généralisé la fonction pour la réutiliser

    pour les modele suivant(Transfer-learning)

    

    params :

        --->define_model: fonction qui retourne un modele keras

    retour :

        --->grid_result : résultat de l'évaluation

        """

    lrate_std_decay=LearningRateScheduler(standard_decay)

    lrate_step_decay=LearningRateScheduler(step_decay)

    lrate_poly_decay=LearningRateScheduler(poly_decay)

    

    decays = [[lrate_step_decay], [lrate_poly_decay], [lrate_std_decay]]

    param_grid_decays = dict(callbacks=decays)

    

    model = KerasClassifier(build_fn=define_model, epochs=epochs,batch_size=batch_size)

    

    grid_decay = GridSearchCV(estimator=model, param_grid = param_grid_decays, cv=3)

    grid_result = grid_decay.fit(train_x_resized, train_y_resized)

    

    # afficher les resultats

    #pour afficher les résultat pour cette partie nous affichons le doc_string de la fonction utilisé,

    #sinon ça va étre que l'adresse de la fonction.

    print("Best: %f with %s" % (grid_result.best_score_, grid_result.best_params_["callbacks"][0].schedule.__doc__))

    # afficher les resultats detailles

    means = grid_result.cv_results_['mean_test_score']

    stds = grid_result.cv_results_['std_test_score']

    params = grid_result.cv_results_['params']

    

    for mean, stdev, param in zip(means, stds, params):

        print("mean (+/- std) = %f (%f) with: %r" % (mean, stdev, param["callbacks"][0].schedule.__doc__))#

    

    return grid_result





#question 2 - entrainer le modele et le sauvgarder

def fit_and_save(model, modelname, nb_epoch, learning_rate, weight_decay, x_gen):

    """Cette fonction entraine le modele et le sauvgarde

    

    Ps : Nous avons généralisé la fonction pour la réutiliser

    pour les modele suivant(Transfer-learning)

    

    params:

        --->model : modele à entainer

        --->modelname : le nom du modele

        --->nb_epoch : nb_epoch

        --->learning_rate : learning_rate

        --->weight_decay : weight_decay

    """



    # Model summary

    model.summary()

    

    #plot model

    plot_model(model, to_file='Architecture_'+modelname+'.png')

    

    #Callbacks

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=False)

    

    filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto') 

    print("Training")

    

    callbacks = [checkpoint]

    

    H = model.fit_generator(x_gen,epochs=nb_epoch)

    

    model.save(modelname+'.h5')

    

    return model,H



#question 2 - Sauvegarder les valeurs des mesures de performances et l'historique d'entrainement

def save_history_metrics(model, H,modele_name,test_x,test_y):

    """Cette fonction sauvgarde l'historique 

    

    Ps : Nous avons généralisé la fonction pour la réutiliser

    pour les modele suivant(Transfer-learning)

    """

    with open('best_history_for_'+modele_name+'.pickle', 'wb') as f:

        pickle.dump(H, f)

    

    y = model.predict(test_x)

    

    y_predicted = [x.argmax() for x in y]

    y_result = [x.argmax() for x in test_y]

    

    acc = accuracy_score(y_result, y_predicted)

    precision = precision_score(y_result, y_predicted, average = "macro" )

    f1 = f1_score(y_result, y_predicted, average = "macro")

    recall = recall_score(y_result, y_predicted , average = "macro")

    

    model_metrics = dict(accuracy_score= acc ,  precision_score = precision, f1_score = f1 ,recall_score = recall)

    

    with open('best_metrics_for_'+modele_name+'.pickle', 'wb') as handle:

        pickle.dump(model_metrics, handle)

        

    print("accuracy_score : ",acc)

    print("precision_score : ",precision)

    print("f1_score : ",f1)

    print("recall_score : ",recall)

    

    epochs = len(history['accuracy'])

    N = np.arange(0, epochs)

    plt.style.use("ggplot")

    plt.figure()

    plt.plot(N, history["loss"], label="train_loss")

    plt.plot(N, history["accuracy"], label="train_acc")

    plt.title("Training Loss and Accuracy"+modele_name)

    plt.xlabel("Epoch #")

    plt.ylabel("Loss/Accuracy")

    plt.legend()

    plt.savefig("Loss_Acc_for"+modele_name) 

    

#question 3

def extract_carts(model_path,modelname, X, Y,basename):

    """Cette fonction extraire les vecteurs descripteurs des images

    des quatre base de données (crops_train,resized_train,crops_test,resized_test)

    en utilisant un modele keras et les sauvgardes sous forme de 4 fichiers .csv

    

    Ps : Nous avons généralisé la fonction pour la réutiliser

    pour les modele suivant(Transfer-learning)

    

    params:

        --->model_path : chemin du modele dans le disk

        --->modelname : nom du modele

        --->X : ensemble de données

        --->basename : nom de la base

    """

    model = load_model(model_path)

    model.layers.pop()

    model.summary()

    

    model2 = Model(input=model.input, output=[model.layers[-1].output])

    

    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay = 1E-4)



    model2.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    del model

    gc.collect()

    

    vec_cara = model2.predict(X)

    pd.DataFrame(vec_cara).to_csv('Vecteurs_caractertique_'+basename+'_using_'+modelname+'.csv')

    pd.DataFrame(Y).to_csv('Labels_'+basename+'_using_'+modelname+'.csv')

    



#_______________________________________________Main______________________________________________________



#-->Préparer les données

#les chemin de nos base de données

trainPath_crops = "/kaggle/input/db-inf907/AFF20_crops/AFF20_crops/AFF20_crops_train"

testPath_crops = "/kaggle/input/db-inf907/AFF20_crops/AFF20_crops/AFF20_crops_test"



trainPath_resized = "/kaggle/input/db-inf907/AFF20_resized/AFF20_resized/AFF20_resized_train"

testPath_resized = "/kaggle/input/db-inf907/AFF20_resized/AFF20_resized/AFF20_resized_test"



trainGen_crops, testGen_crops, train_x_crops, train_y_crops, test_x_crops, test_y_crops = get_data(trainPath_crops, testPath_crops)



trainGen_resized,testGen_resized, train_x_resized, train_y_resized, test_x_resized, test_y_resized = get_data(trainPath_resized, testPath_resized)



#la dimension de chaque image (244,244,3)

img_dim = (train_x_crops.shape[1:])



#le nombre de classes 20

nb_classes = train_y_crops.shape[1]



#la taille de l'emsemble d'entrainement crops

train_shape = train_x_crops.shape[0]



batch_size = 16

epochs = 50

learning_rate = 0.01

decay_learning_rate = 1E-4

"""

#-->Créer le modele MiniDenseNet, et afficher sa structure

MiniDenseNet = DenseNet()

#MiniDenseNet.summary()



#Lancer l'évaluation sur les nombres d'epochs

GridSearchCV_on_epoches(DenseNet,epoch_nbrs=[20,30,50])



#Lancer l'évaluation sur les tailles des batches

GridSearchCV_on_batches(DenseNet, batch_sizes=[8,16,32])



#Lancer l'évaluation sur les fonction de dégradation du taux d'apprentissage

GridSearchCV_on_decay(DenseNet)



#Nous avons trouvé que les paramétre optimaux sont:

#--> epochs = 50

#--> batchsize = 32

#--> Dégradation standard

#NB : Par rapport à la taille du batches, la mémoire nous a pas permis de tester de grande valeurs





#lancer l'apprentissage avec les paramétres trouvés avec la base resized

MiniDenseNet_onR, H = fit_and_save(MiniDenseNet,"DenseNet_onResized",epochs,learning_rate,decay_learning_rate,trainGen_resized)



#sauvgarder l'historique d'apprentissage, les diférant métrics et le graphe de ACC/LOSS

save_history_metrics(MiniDenseNet_onR, H.history, "DenseNet_onResized",test_x_resized,test_y_resized)





#lancer l'apprentissage avec les paramétres trouvés avec la base crops

MiniDenseNet = DenseNet()

MiniDenseNet_onC, H = fit_and_save(MiniDenseNet,"DenseNet_onCrops",epochs,learning_rate,decay_learning_rate,trainGen_crops)



#sauvgarder l'historique d'apprentissage, les diférant métrics et le graphe de ACC/LOSS

save_history_metrics(MiniDenseNet_onC, H.history, "DenseNet_onCrops",test_x_crops,test_y_crops)





#Utiliser les modeles entrainer pour extraire les caractéristiques

model_path_R = "/kaggle/input/models/DenseNet_onResized.h5"

model_path_C = "/kaggle/input/models/DenseNet_onCrops.h5"



extract_carts(model_path_R,"DenseNet_onResized",train_x_resized[:200] ,train_y_resized[:200],"train_resized")

extract_carts(model_path_R,"DenseNet_onResized", test_x_resized[:200],test_y_resized[:200],"test_resized")

extract_carts(model_path_C,"DenseNet_onCrops", train_x_crops[:200],train_y_crops[:200],"train_crops")

extract_carts(model_path_C,"DenseNet_onCrops",test_x_crops[:200] ,test_y_crops[:200],"test_crops")



"""
model_path = "/kaggle/input/models/DenseNet.h5"



extract_carts(model_path,"DenseNet_onCrops", train_x_crops[:200],train_y_crops[:200],"train_crops")
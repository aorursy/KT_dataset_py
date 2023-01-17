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

"""

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

"""
#Préparation des données et augmentation, création du modèle (questions 1, 2 partie B)

def get_data(trainPath, testPath):

    """Cette fonction retourne les données d'apprentissage et de test

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



#question 3, 4, 5 et 6

def resnext50_model(nombre_class=20, show_models = True):

    

    #charger le modele

    model = ResNeXt50(include_top = True,

                  weights='imagenet',

                  backend=keras.backend,

                  layers=keras.layers,

                  models=keras.models,

                  utils=keras.utils 

            )

    #afficher le modéle avant la suppression

    if show_models:

        model.summary()

    

    #afficher les differents paramétres de la dérniere couche Dense du modele, pour définir une similaire à elle

    

    activation_ = model.layers[-1].activation

    use_bias_ = model.layers[-1].use_bias

    

    kernel_init_ = model.layers[-1].kernel_initializer

    kernel_initializer_scale = model.layers[-1].kernel_initializer.scale

    kernel_initializer_mode = model.layers[-1].kernel_initializer.mode

    kernel_initializer_distribution = model.layers[-1].kernel_initializer.distribution

    kernel_initializer_seed = model.layers[-1].kernel_initializer.seed

    

    bias_initializer = model.layers[-1].bias_initializer

    kernel_regularizer = model.layers[-1].kernel_regularizer

    bias_regularizer = model.layers[-1].bias_regularizer

    activity_regularizer = model.layers[-1].activity_regularizer

    kernel_constraint = model.layers[-1].kernel_constraint

    bias_constraint = model.layers[-1].bias_constraint

    if show_models:

        print("Paramétre du dérnier couche:")

        print("fonction d'activation",activation_)

        print("use_bias_",use_bias_)

        print("kernel_init",kernel_init_,"avec scale=",kernel_initializer_scale,"mode=",kernel_initializer_mode,"distribution=",kernel_initializer_distribution,"et seed=",kernel_initializer_seed)

        print("bias_initializer",bias_initializer)

        print("kernel_regularizer",kernel_regularizer)

        print("bias_regularizer",bias_regularizer)

        print("activity_regularizer",activity_regularizer)

        print("kernel_constraint",kernel_constraint)

        print("bias_constraint",bias_constraint)

    

    #supprimer la deniere couche

    model.layers.pop()

    

    if show_models:

        #afficher le modéle aprés la suppression

        print("le modele sans la dérniere couche:")

        model.summary()

    

    #Définir notre nouvel couche, avec les méme caractéristique que la couche originale

    #Définir l'initialiseur pour cette couche

    kernel_initial = VarianceScaling(scale=kernel_initializer_scale, mode=kernel_initializer_mode, distribution=kernel_initializer_distribution, seed=kernel_initializer_seed)

    # couche FC

    o = Dense(nombre_class, activation='softmax', use_bias=use_bias_, kernel_initializer=kernel_initial, bias_initializer='zeros',

              kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,

              kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(model.layers[-1].output)



      

    model2 = Model(inputs=model.inputs, outputs=[o])

    

    if show_models:

        print("le nouveau modéle")

        model2.summary()

    

    return model2





#Question 7

def frezz_model(model, show):

    """

    Ici touts les couches sont à priori toute gelées sauf la FC

    """

    for layer in model.layers[:-1]:

        layer.trainable = False

    if show:

        model.summary()

        for layer in model.layers:

            print(layer.name , layer.trainable)

    

    opt = SGD(learning_rate=0.001, momentum=0.0, decay=decay_learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    return model



#Question 8 a et b

def frezz_blocks(model, Block_to_disable, show= True ):

    """Cette fonction prend en paramétre un model, le nombre de block à geler

    et il retourne le modele geler

    

    NB : dans le cas du modele resnet/resnext nous avon géler par stage, un stage est composé de blocks résiduel

    On a fait ce choix en se basent sur les architecture des modéles (ces architecture sont composé de stages)

    

    params:

        --->model : un model keras à geler

        --->Block_to_disable : 1 ou 2

        --->show : pour afficher les couches

    """

    #l'indice de la dernier couche du stage 1

    Resnext_last_layer_stage1 = 49

    

    ##l'indice de la dernier couche du stage 2

    Resnext_last_layer_stage2 = 107

    

    #le nombre de couches

    total_layer = len(model.layers)

    

    if Block_to_disable == 1:

        for layer in model.layers:

            layer.trainable = True

        

        for layer in model.layers[:Resnext_last_layer_stage1]:

            layer.trainable = False

        

    elif Block_to_disable == 2:

        for layer in model.layers:

            layer.trainable = True

        

        for layer in model.layers[:Resnext_last_layer_stage2]:

            layer.trainable = False

            

        

    opt = SGD(learning_rate=0.001, momentum=0.0, decay=decay_learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    

    if show:

        model.summary()

        for layer in model.layers:

            print(layer.name , layer.trainable)

    return model



def define_model(model,frezzing_options):

    """Cette fonction prend en paramétre le modéle à définir,

    les options pour geler les couche. Et elle retourn le modele

    params:

        --->modelname : model keras

        --->frezzing_options: 0 pour un modéle gelée sauf la FC,

                        1 pour geler le premier block, 

                        2 pour geler le 2eme block"""

    

    if frezzing_options == 0:

        returned_model = frezz_model(model, show= False)

    elif frezzing_options == 1:

        returned_model = frezz_blocks(model,Block_to_disable=1, show= False)

    elif frezzing_options == 2:

        returned_model = frezz_blocks(model,Block_to_disable=2, show= False)

            

    return returned_model



def get_model_resnext_freezed():

    """fonction qui retourne un modele avec que la couche FC entrainable"""

    modele_resnext = resnext50_model(show_models = False)

    return define_model(modele_resnext,0)



def get_model_resnext_freezed_1block():

    """fonction qui retourne un modele avec un block gelé"""

    modele_adapter = load_model("/kaggle/input/t-l-resnext/ResNext_freezed.h5")

    return define_model(modele_adapter,1)



def get_model_resnext_freezed_2block():

    """fonction qui retourne un modele avec deux blocks gelés"""

    modele_adapter = load_model("/kaggle/input/t-l-resnext/ResNext_freezed.h5")

    return define_model(modele_adapter,2)



#Evaluation du modèle ResNext (question 1)



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

        --->x_gen : generateur d'entrainement

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

def save_history_metrics(model, history,modele_name,test_x,test_y):

    """Cette fonction sauvgarde l'historique 

    

    Ps : Nous avons généralisé la fonction pour la réutiliser

    pour les modele suivant(Transfer-learning)

    """

    with open('best_history_for_'+modele_name+'.pickle', 'wb') as f:

        pickle.dump(history, f)

    

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

    print(vec_cara.shape)

    print(X.shape)

    

    pd.DataFrame(vec_cara).to_csv('Vecteurs_caractertique_'+basename+'_using_'+modelname+'.csv')

    pd.DataFrame(Y).to_csv('Labels_'+basename+'_using_'+modelname+'.csv')



#_______________________________________________Main______________________________________________________



#-->Préparer les données

#les chemin de nos base de données

trainPath_crops = "/kaggle/input/db-inf907/AFF20_crops/AFF20_crops/AFF20_crops_train"

testPath_crops = "/kaggle/input/db-inf907/AFF20_crops/AFF20_crops/AFF20_crops_test"



#--->Question 1 et 2

trainGen_crops, testGen_crops, train_x_crops, train_y_crops, test_x_crops, test_y_crops = get_data(trainPath_crops, testPath_crops)



#Initialisation des paramétres

#la dimension de chaque image (244,244,3)

img_dim = (train_x_crops.shape[1:])



#le nombre de classes 20

nb_classes = train_y_crops.shape[1]



#la taille de l'emsemble d'entrainement, pour la Dégradation standard

train_shape = train_x_crops.shape[0]



batch_size = 16

epochs = 25

learning_rate = 0.01

decay_learning_rate = 1E-4

"""

#--->Question 3 à 6

#-->Créer le modele ResNext avec la nouvel FC et les autres couches sont toute gelées. 

#et afficher sa structure

ResNext_freezed = get_model_resnext_freezed()

ResNext_freezed.summary()



#--->Question 7: évaluation et apprentissage du modele ResNext_freezed



#évaluation du modéle ResNext avec tous les couche gelés (sauf FC)

#Lancer l'évaluation sur les nombres d'epochs

#GridSearchCV_on_epoches(get_model_resnext_freezed,epoch_nbrs=[20,30,50])



#Lancer l'évaluation sur les tailles des batches

#GridSearchCV_on_batches(get_model_resnext_freezed, batch_sizes=[8,16,32])



#Lancer l'évaluation sur les fonction de dégradation du taux d'apprentissage

#GridSearchCV_on_decay(get_model_resnext_freezed)



#Nous avons trouvé que les paramétre optimaux sont:

#--> epochs = 50

#--> batchsize = 16

#--> Dégradation standard



#Faire un fine-tuning sur la nouvelle FC pour un certain nombre d’epoch, afin

#d’apprendre des poids adaptés à la nouvelle base. Ici les autres couches sont à priori

#toute gelées. 

ResNext_freezed, H = fit_and_save(ResNext_freezed,"ResNext_freezed",epochs,learning_rate,decay_learning_rate,trainGen_crops)



#sauvgarder l'historique d'apprentissage, les diférant métrics et le graphe de ACC/LOSS

save_history_metrics(ResNext_freezed, H.history, "ResNext_freezed",test_x_crops,test_y_crops)





#--->Question 8-a: évaluation et apprentissage du modele resnext_freezed_2block



#Geler le premièr stage et faire un fine-tuning sur les couches supérieures.. 

#et afficher sa structure

resnext_freezed_1block = get_model_resnext_freezed_1block()

resnext_freezed_1block.summary()



#évaluation du modéle ResNext avec 1 block gelé



#Lancer l'évaluation sur les nombres d'epochs

#GridSearchCV_on_epoches(get_model_resnext_freezed_1block,epoch_nbrs=[20,30,50])



#Lancer l'évaluation sur les tailles des batches

#GridSearchCV_on_batches(get_model_resnext_freezed_1block, batch_sizes=[8,16,32])



#Lancer l'évaluation sur les fonction de dégradation du taux d'apprentissage

#GridSearchCV_on_decay(get_model_resnext_freezed_1block)



#Nous avons trouvé que les paramétre optimaux sont:

#--> epochs = 25

#--> batchsize = 16

#--> Dégradation standard



#Faire un fine-tuning.

resnext_freezed_1block, H = fit_and_save(resnext_freezed_1block,"resnext_freezed_1block",epochs,learning_rate,decay_learning_rate,trainGen_crops)



#sauvgarder l'historique d'apprentissage, les diférant métrics et le graphe de ACC/LOSS

save_history_metrics(resnext_freezed_1block, H.history, "resnext_freezed_1block",test_x_crops,test_y_crops)



#--->Question 8-b: évaluation et apprentissage du modele resnext_freezed_1block



#-->Geler les deux premièrs stages et faire un fine-tuning sur les couches supérieures.. 

#et afficher sa structure

resnext_freezed_2block = get_model_resnext_freezed_2block()

resnext_freezed_2block.summary()



#évaluation du modéle ResNext avec 1 block gelé



#Lancer l'évaluation sur les nombres d'epochs

#GridSearchCV_on_epoches(get_model_resnext_freezed_2block,epoch_nbrs=[20,30,50])



#Lancer l'évaluation sur les tailles des batches

#GridSearchCV_on_batches(get_model_resnext_freezed_2block, batch_sizes=[8,16,32])



#Lancer l'évaluation sur les fonction de dégradation du taux d'apprentissage

#GridSearchCV_on_decay(get_model_resnext_freezed_2block)



#Nous avons trouvé que les paramétre optimaux sont:

#--> epochs = 50

#--> batchsize = 16

#--> Dégradation standard



#Faire un fine-tuning sur la nouvelle FC pour un certain nombre d’epoch, afin

#d’apprendre des poids adaptés à la nouvelle base. Ici les autres couches sont à priori

#toute gelées. 

resnext_freezed_2block, H = fit_and_save(resnext_freezed_2block,"resnext_freezed_1block",epochs,learning_rate,decay_learning_rate,trainGen_crops)



#sauvgarder l'historique d'apprentissage, les diférant métrics et le graphe de ACC/LOSS

save_history_metrics(resnext_freezed_2block, H.history, "resnext_freezed_1block",test_x_crops,test_y_crops)

"""





#Utiliser le modele entrainer pour extraire les caractéristique



model_path_C = "/kaggle/input/t-l-resnext/resnext_freezed_1block.h5"

extract_carts(model_path_C,"ResNext_freezed_1block",train_x_crops ,train_y_crops,"train_crops")

extract_carts(model_path_C,"ResNext_freezed_1block",test_x_crops ,test_y_crops,"test_crops")
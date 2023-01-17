import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import sys

import os

from keras.applications.vgg16 import VGG16

import keras

from numpy import load

from matplotlib import pyplot

from sklearn.model_selection import train_test_split

from keras import backend

from keras.layers import Dense

from keras.layers import Flatten

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D

from keras.optimizers import SGD

from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.layers import Dropout

from keras.layers.normalization import BatchNormalization
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

traindir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train"

validdir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid"

testdir = "../input/new-plant-diseases-dataset/test/test"



train_datagen = ImageDataGenerator(rescale=1./255,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   fill_mode='nearest')



valid_datagen = ImageDataGenerator(rescale=1./255)



batch_size = 128

training_set = train_datagen.flow_from_directory(traindir,

                                                 target_size=(224, 224),

                                                 batch_size=batch_size,

                                                 class_mode='categorical')



valid_set = valid_datagen.flow_from_directory(validdir,

                                            target_size=(224, 224),

                                            batch_size=batch_size,

                                            class_mode='categorical')

class_dict = training_set.class_indices

print(class_dict)
li = list(class_dict.keys())

print(li)
train_num = training_set.samples

valid_num = valid_set.samples

print("train_num is:",train_num)

print("valid_num is:",valid_num)
# base_model=VGG16(include_top=False,input_shape=(224,224,3))

# base_model.trainable=False
# classifier=keras.models.Sequential()

# classifier.add(base_model)

# classifier.add(Flatten())

# classifier.add(Dense(38,activation='softmax'))

# classifier.summary()
# classifier.compile(optimizer='adam',

#               loss='categorical_crossentropy',

#               metrics=['accuracy'])
# #fitting images to CNN

# history = classifier.fit(training_set,

#                          steps_per_epoch=train_num//batch_size,

#                          validation_data=valid_set,

#                          epochs=5,

#                          validation_steps=valid_num//batch_size,

#                          )
# #Saving our model

# filepath="Mymodel.h5"

# classifier.save(filepath)
# import matplotlib.pyplot as plt

# import seaborn as sns

# sns.set()



# acc = history.history['accuracy']

# val_acc = history.history['val_accuracy']

# loss = history.history['loss']

# val_loss = history.history['val_loss']

# epochs = range(1, len(loss) + 1)



# #accuracy plot

# plt.plot(epochs, acc, color='green', label='Training Accuracy')

# plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')

# plt.title('Training and Validation Accuracy')

# plt.ylabel('Accuracy')

# plt.xlabel('Epoch')

# plt.legend()



# plt.figure()

# #loss plot

# plt.plot(epochs, loss, color='pink', label='Training Loss')

# plt.plot(epochs, val_loss, color='red', label='Validation Loss')

# plt.title('Training and Validation Loss')

# plt.xlabel('Epoch')

# plt.ylabel('Loss')

# plt.legend()



# plt.show()
# from tensorflow import keras

# classifier = keras.models.load_model("../input/modelvgg16/Mymodel.h5")
# # predicting an image

# import matplotlib.pyplot as plt

# from keras.preprocessing import image

# import numpy as np

# image_path = "../input/new-plant-diseases-dataset/test/test/TomatoEarlyBlight1.JPG"

# new_img = image.load_img(image_path, target_size=(224, 224))

# img = image.img_to_array(new_img)

# img = np.expand_dims(img, axis=0)

# img = img/255



# print("Following is our prediction:")

# prediction = classifier.predict(img)

# # decode the results into a list of tuples (class, description, probability)

# # (one such list for each sample in the batch)

# d = prediction.flatten()

# j = d.max()

# for index,item in enumerate(d):

#     if item == j:

#         class_name = li[index]



     

        

# ##Another way

# img_class = classifier.predict_classes(img)

# img_prob = classifier.predict_proba(img)

# #ploting image with predicted class name        

# plt.figure(figsize = (4,4))

# plt.imshow(new_img)

# plt.axis('off')

# plt.title(class_name)

# plt.show()
# #Confution Matrix and Classification Report

# from sklearn.metrics import classification_report, confusion_matrix

# Y_pred = classifier.predict_generator(valid_set, valid_num//batch_size+1)
# class_dict = valid_set.class_indices

# li = list(class_dict.keys())

# print(li)
# y_pred = np.argmax(Y_pred, axis=1)

# print('Confusion Matrix')

# print(confusion_matrix(valid_set.classes, y_pred))

# print('Classification Report')

# target_names =li ## ['Peach___Bacterial_spot', 'Grape___Black_rot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

# print(classification_report(valid_set.classes, y_pred, target_names=target_names))
from tensorflow import keras

modelvgg16 = keras.models.load_model("../input/modelvgg16/Mymodel.h5")

AlexNetModel = keras.models.load_model("../input/modelalexnet-and-best-weights-9/AlexNetModel.hdf5")

InceptionV3Model = keras.models.load_model("../input/inceptionv3model/InceptionV3.h5")

CNNModel = keras.models.load_model("../input/cnn-model/cnn_model.h5")

MobileNetModel = keras.models.load_model("../input/leaf-cnn-mobelnet/leaf-cnn.h5")
#no one of models is regressor or classifier



from sklearn.base import is_classifier, is_regressor

models_names=['InceptionV3','AlexNetModel','vgg16model','CNNModel','MobileNetModel']

print("model name\t estimator name\t is_regressor\t is_classifier")    

for estimator , model_name in zip([InceptionV3Model,AlexNetModel,modelvgg16,MobileNetModel],models_names):

        print("{}\t {}   \t {}    \t {}".format(model_name,estimator.__class__.__name__,

                                              is_regressor(estimator),

                                              is_classifier(estimator)

                                              ))

      
X, Y = training_set.next()    

#Y=np.zeros(X.shape[0])    

print("X.shape =",X.shape)

print("Y.shape =",Y.shape)
import pandas

from sklearn import model_selection

from sklearn.ensemble import VotingClassifier

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

import numpy



#X, Y =  make_classification()

X, Y = training_set.next()    

Y=np.zeros(X.shape[0])



print("X.shape =",X.shape)

print("Y.shape =",Y.shape)



# Initializing _estimator_typefir each model

for model in [modelvgg16, AlexNetModel, InceptionV3Model]:

    model._estimator_type = "classifier"

    #print(model._estimator_type)



estimators=[

            ('modelvgg16', modelvgg16), 

            ('AlexNetModel', AlexNetModel),

            ('InceptionV3Model', InceptionV3Model)

           ]



print("The names and types of the model in ensemble:")

for e in estimators:

    print("{0:s} : {1:s}".format( e[0], type(e[1]).__name__))

    

ensemble_model = VotingClassifier(estimators=estimators,voting='hard')     # create the ensemble model



######################### Split train+test #######################################

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,random_state=2)



#Whole Wine Classifier

ensemble_model.fit(x_train, y_train)

y_pred = ensemble_model.predict(x_test)

from sklearn.metrics import accuracy_score

print("accuracy : ",accuracy_score(y_test,y_pred))
import pandas

from sklearn import model_selection

from sklearn.ensemble import VotingClassifier

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

import numpy



#X, Y =  make_classification()

X, Y = training_set.next()    

Y=np.zeros(X.shape[0])

print("X.shape =",X.shape)

print("Y.shape =",Y.shape)



# Initializing _estimator_typefir each model

for model in [modelvgg16, AlexNetModel,InceptionV3Model]:

    model._estimator_type = "classifier"

    #print(model._estimator_type)

    

estimators=[

            ('modelvgg16', modelvgg16), 

            ('AlexNetModel', AlexNetModel),

            ('InceptionV3Model', InceptionV3Model)

           ]



print("The names and types of the model in ensemble:")

for e in estimators:

    print("{0:s} : {1:s}".format( e[0], type(e[1]).__name__))

    

ensemble_model = VotingClassifier(estimators=estimators,voting='hard')     # create the ensemble model



######################### Split train+test #######################################

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,random_state=2)



#Whole Wine Classifier

ensemble_model.fit(x_train, y_train)

y_pred = ensemble_model.predict(x_test)

from sklearn.metrics import accuracy_score

print("accuracy : ",accuracy_score(y_test,y_pred))
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import itertools

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from mlxtend.classifier import EnsembleVoteClassifier

from mlxtend.data import iris_data

from mlxtend.plotting import plot_decision_regions



# Initializing Classifiers

clf1 = modelvgg16

clf2 = AlexNetModel

clf3=InceptionV3Model

ensemble_model = EnsembleVoteClassifier(clfs=[clf1, clf2,clf3],weights=[2, 1, 1], voting='soft')



X, Y = training_set.next()    

Y=np.zeros(X.shape[0])    

print("X.shape =",X.shape)

print("Y.shape =",Y.shape)



# Initializing _estimator_typefir each model

for model in [modelvgg16, AlexNetModel,InceptionV3Model]:

    model._estimator_type = "classifier"

    #print(model._estimator_type)







######################### Split train+test #######################################

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,random_state=2)



#Whole Wine Classifier

ensemble_model.fit(x_train, y_train)

y_pred = ensemble_model.predict(x_test)

from sklearn.metrics import accuracy_score

print("accuracy : ",accuracy_score(y_test,y_pred))
# def clone(estimator, safe=False):

#     """Constructs a new estimator with the same parameters.



#     Clone does a deep copy of the model in an estimator

#     without actually copying attached data. It yields a new estimator

#     with the same parameters that has not been fit on any data.



#     Parameters

#     ----------

#     estimator : estimator object, or list, tuple or set of objects

#         The estimator or group of estimators to be cloned



#     safe : boolean, optional

#         If safe is false, clone will fall back to a deep copy on objects

#         that are not estimators.



#     """

#     estimator_type = type(estimator)

#     # XXX: not handling dictionaries



#     klass = estimator.__class__

#     new_object_params = estimator.get_params(deep=False)

#     for name, param in new_object_params.items():

#         new_object_params[name] = clone(param, safe=False)

#     new_object = klass(**new_object_params)

#     params_set = new_object.get_params(deep=False)



#     # quick sanity check of the parameters of the clone

#     for name in new_object_params:

#         param1 = new_object_params[name]

#         param2 = params_set[name]

#         if param1 is not param2:

#             raise RuntimeError('Cannot clone object %s, as the constructor '

#                                'either does not set or modifies parameter %s' %

#                                (estimator, name))

#     return new_object





# clone(modelvgg16)
# predicting an image

import matplotlib.pyplot as plt

from keras.preprocessing import image

import numpy as np



class votingClassifer:

    'votingClassifer class'





    def __init__(self, estimators,mode="hard",weight=None,show_info="percent"):

        

        if mode not in ["hard","soft"]:

            raise Exception("the mode should be 'hard' or 'soft'")

            

        self.estimators=estimators

        self.mode=mode

        self.weight=weight

        self.show_info=show_info  

            

    def predict(self,x_test):        

        if self.mode=="hard":

            return self.votingClassifer_hard(self.estimators, x_test, show_info=self.show_info)

        else:

            return self.votingClassifer_soft(self.estimators, x_test, weight=self.weight, show_info=self.show_info)



    def getNumberDiff(self, index_classes, n):  

        

        for x in index_classes:

            if index_class_prefer != x:

                return x    

    

    def getNumberElse(self, index_classes, n):



        indices = [i for i, value in enumerate(index_classes) if value != n]

        counts = np.bincount(indices)

        ind=np.argmax(counts)

        n1=(indices == ind).sum()

 

                

        return n1

    

    

    def votingClassifer_hard(self,estimators,x_test,show_info='percent'):

        

        if show_info not in ["info","percent","nothing"]:

            raise Exception("the attribut 'show_info' should be 'info' or 'percent','nothing'")            

            

        

        cpt=0

        index_classes_glob, class_names_glob, probs_glob=[],[],[]

        N=len(x_test)

        for x in x_test:

            index_classes, class_names, probs=[],[],[]

            for model in estimators:

                img = np.expand_dims(x, axis=0)

                # make a prediction

                y_prob = model.predict(img)[0]

                probabilty = y_prob.flatten()

                max_prob = probabilty.max()

                y_classes = y_prob.argmax(axis=-1)

                index_class, class_name, prob = y_classes,li[y_classes],max_prob

                index_classes.append(index_class)

                class_names.append(class_name)

                probs.append(prob)

            index_classes, class_names, probs = np.array(index_classes), np.array(class_names), np.array(probs)    

            counts = np.bincount(index_classes)

            index_class_prefer=np.argmax(counts)

 

            n1=(index_classes == index_class_prefer).sum()

    

            if n1 == 1:

                print("\n Each estimator predict a different class")

                prob = probs.max()

                indice = [i for i, value in enumerate(probs) if value == prob][0]

                class_name = class_names[indice]



                

            elif n1 == len(estimators)/2 and len(estimators)/2 == self.getNumberElse(index_classes, index_class_prefer):

                

                print("\n the half-estimators predict a class and the other estimators predict a different class")

                

                indices1 = [i for i, value in enumerate(index_classes) if value == index_class_prefer]

                sum2=0

                for ind in indices1:

                    sum2+=probs[ind]

                

                prob1=sum2/len(indices1)

                

                n2=self.getNumberDiff(index_classes, index_class_prefer)      

                

                indices2 = [i for i, value in enumerate(index_classes) if value == n2]

                sum2=0

                for ind in indices2:

                    sum2+=probs[ind]

                

                prob2=sum2/len(indices2)

                

                if prob1 < prob2:

                    prob=prob2

                    indice = [i for i, value in enumerate(index_classes) if value == n2][0]

                    class_name = class_names[indice]

                    

                else:

                    prob=prob1

                    indice = [i for i, value in enumerate(index_classes) if value == index_class_prefer][0]

                    class_name = class_names[indice]

                

            else:

                

                sum1=0

                nbr=0

                for i, index in zip(range(len(probs)),index_classes):

                    if index_class_prefer== index:

                        sum1+=probs[i]

                        nbr+=1

                        

                prob=sum1/nbr

                indice = index_class_prefer

                class_name = li[index_class_prefer]



            

            if show_info=="info":

                cpt+=1

                print("\rpercent: {:.2f}%, li[{}]:{} --> {}".format(cpt*100/N,index_class_prefer, li[index_class_prefer], prob), end='')   

            elif show_info=="percent":

                cpt+=1

                print("\rpercent: {:.2f}%".format(cpt*100/N), end='')

          

                

            index_classes_glob.append(indice)

            class_names_glob.append(class_name)

            probs_glob.append(prob)



        return np.array(index_classes_glob), np.array(class_names_glob), np.array(probs_glob) 

    

    

    def votingClassifer_soft(self, estimators, x_test, weight=None, show_info="percent"):

        

        if show_info not in ["info","percent","nothing"]:

            raise Exception("the attribut 'show_info' should be 'info' or 'percent','nothing'")  



        if weight is None :

            weight=np.ones(len(estimators))



        if len(weight) != len(estimators):

            raise Exception("number of models and wheight should be equals")    



        cpt=0 

        # get number of classes 

        x = image.img_to_array(x_test[0])

        x = np.expand_dims(x, axis=0)

        y_prob = estimators[0].predict(x) 

        num_classes =  y_prob.shape[1]



        Tab=np.zeros(num_classes)# num_classes

        index_classes_glob, class_names_glob, probs_glob=[],[],[]

        N=len(x_test)

        for x in x_test:

            index_classes, class_names, probs= [], [], []

            for model in estimators:

                img = np.expand_dims(x, axis=0)

                # make a prediction

                y_prob = model.predict(img)[0]

                idxs = np.argsort(y_prob)

                # loop over the indexes of the high confidence class labels

                for (index, value) in enumerate(idxs):

                    # build the label and draw the label on the image

                    #label = "{}) {}[{}]: {:.2f}%".format(index,li[value],value, y_prob[value] * 100)

                    #print(label)

                    Tab[value]=y_prob[value]



                probs.append(Tab)



            probs=np.array(probs) # probs.shape: (3, 38)



            proba=[]   



            div=sum(weight)

            for i in range(len(probs[0])):

                s=0

                for j in range(len(weight)):

                    s+= probs[j][i] * weight[j] 

                s=s/div

                proba.append(s)



            proba=np.array(proba)

            max_prob=max(proba)



            indices=[i for i, value in enumerate(proba) if value == max_prob]



            index_class, class_name, prob=indices[0],li[indices[0]],max_prob 



            index_classes_glob.append(index_class)

            class_names_glob.append(class_name)

            probs_glob.append(prob) 



            if show_info=="info":

                cpt+=1

                print("percent: {:.2f}%, li[{}]:{} --> {}%".format(cpt*100/N,index_class, li[index_class], prob))   

            elif show_info=="percent":

                cpt+=1

                print("\rpercent: {:.2f}%".format(cpt*100/N), end='')







        return np.array(index_classes_glob), np.array(class_names_glob), np.array(probs_glob)  

       

            

# valid_set

x_test, y_test = valid_set.next() 
from sklearn.metrics import accuracy_score

import numpy as np

models=[modelvgg16, AlexNetModel,InceptionV3Model,CNNModel,MobileNetModel]

model_names=["vgg16Model", "AlexNetModel","InceptionV3Model",'CNNModel','MobileNetModel']

for model,model_name in zip(models,model_names):

    y_prob = model.predict(x_test)

    y_pred1 = y_prob.argmax(axis=-1)

    y_test1=np.argmax(y_test, axis=1)

    # accuracy

    print(model_name+" accuracy: ",accuracy_score(y_test1,y_pred1))

    del model , y_pred1, y_test1
# valid_set

x_test, y_test = valid_set.next() 



#VotingClassifier(hard)

estimators = [modelvgg16, AlexNetModel,CNNModel,MobileNetModel]

vc=votingClassifer(estimators=estimators,mode="hard",show_info="percent")

index_classes, class_names, probs=vc.predict(x_test)
# inverse_to_categorical inverser format binary to format indexation

# datagenerator use the methode to_categorical for labelsation to frmat binary 

# the method inverce of to_categorical is argmax

import numpy as np

y_test1=np.argmax(y_test, axis=1)

y_pred1=index_classes



# accuracy

from sklearn.metrics import accuracy_score

print("votingClassifer(hard) accuracy : ",accuracy_score(y_test1,y_pred1))
# confusion_matrix

print("confusion_matrix")

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

print(confusion_matrix(y_test1,y_pred1))
# valid_set

x_test, y_test = valid_set.next() 
import numpy as np

from sklearn.metrics import accuracy_score

def getWeight(models):

    tab=[]

    cpt=0

    nbr=len(models)

    for model in models:

        y_prob = model.predict(x_test)

        y_pred1 = y_prob.argmax(axis=-1)

        y_test1=np.argmax(y_test, axis=1)

        tab.append(accuracy_score(y_test1,y_pred1))

        cpt+=1

        print("\rpercent: {:.2f}%".format(cpt*100/nbr), end='')

        

    tab=np.array(tab)    

    max_value=max(tab) 

    weight=[(x*nbr/max_value) for x in tab]

    weight=getmin_max_indice(np.array(weight))

    return weight



def getmin_max_indice(weight):

    d=sorted(weight, reverse=True)

    a=sorted(weight, reverse=False)

    cpt=0

    for ma,mi in zip(d,a):

        max_indice=[i for i, value in enumerate(weight) if value == ma]

        min_indice=[i for i, value in enumerate(weight) if value == mi]

        c=weight[max_indice]

        weight[max_indice]=weight[min_indice]

        weight[min_indice]=c

        cpt+=1

        if cpt==int(len(weight)/2):

            break

    return weight



models=[modelvgg16, AlexNetModel,CNNModel,MobileNetModel]

weight=getWeight(models)

print("\nweight=",weight)  
#VotingClassifier(soft)

estimators = [modelvgg16, AlexNetModel,CNNModel,MobileNetModel]

vc=votingClassifer(estimators=estimators,mode="soft",weight=weight,show_info="percent")



index_classes, class_names, probs=vc.predict(x_test)
# inverse_to_categorical inverser format binary to format indexation

# datagenerator use the methode to_categorical for labelsation to frmat binary 

# the method inverce of to_categorical is argmax

import numpy as np

y_test1=np.argmax(y_test, axis=1)

y_pred1=index_classes



# accuracy

from sklearn.metrics import accuracy_score

print("votingClassifer(soft) accuracy : ",accuracy_score(y_test1,y_pred1))
# confusion_matrix

print("confusion_matrix")

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

print(confusion_matrix(y_test1,y_pred1))


# predicting an image

import matplotlib.pyplot as plt

from keras.preprocessing import image

import numpy as np

directory="../input/new-plant-diseases-dataset/test/test"

files = [os.path.join(directory,p) for p in sorted(os.listdir(directory))]



models=[modelvgg16, AlexNetModel, CNNModel, MobileNetModel]

model_names=["modelvgg16", "AlexNetModel", "CNNModel", "MobileNetModel"]

for model,model_name in zip(models,model_names):

    for i in range(0,10):

        image_path = files[i]

        new_img = image.load_img(image_path, target_size=(224, 224))

        img = image.img_to_array(new_img)

        img = np.expand_dims(img, axis=0)

        img = img/255

        prediction = model.predict(img)

        probabilty = prediction.flatten()

        max_prob = probabilty.max()

        index=prediction.argmax(axis=-1)[0]

        class_name = li[index]

        #ploting image with predicted class name        

        plt.figure(figsize = (4,4))

        plt.imshow(new_img)

        plt.axis('off')

        plt.title(model_name+": "+class_name+" "+ str(max_prob)[0:4]+"%")

        plt.show()

        
import matplotlib.pyplot as plt

from keras.preprocessing import image

import numpy as np

directory="../input/new-plant-diseases-dataset/test/test"

files = [os.path.join(directory,p) for p in sorted(os.listdir(directory))]



for i in range(0,10):

    image_path = files[i]

    new_img = image.load_img(image_path, target_size=(224, 224))

    img = image.img_to_array(new_img)

    img = np.expand_dims(img, axis=0)

    img = img/255

    model_name="votingClassifer_soft"

    vc=votingClassifer(estimators=estimators,mode="soft",weight=weight, show_info="nothing")

    index_classes, class_names, probs=vc.predict(img)

    #ploting image with predicted class name        

    plt.figure(figsize = (4,4))

    plt.imshow(new_img)

    plt.axis('off')

    plt.title(model_name+": "+class_names[0]+" "+str(probs[0])[0:4]+"%")

    plt.show()

    model_name="votingClassifer_hard"

    vc=votingClassifer(estimators=estimators,mode="hard", show_info="nothing")

    index_classes, class_names, probs=vc.predict(img)    

    #ploting image with predicted class name        

    plt.figure(figsize = (4,4))

    plt.imshow(new_img)

    plt.axis('off')

    plt.title(model_name+": "+class_names[0]+" "+str(probs[0])[0:4]+"%")

    plt.show()
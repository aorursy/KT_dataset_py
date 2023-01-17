import numpy as np 

import pandas as pd 

import os
from keras.models import Sequential

from keras.models import model_from_json

from keras_preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D

from keras import regularizers, optimizers

from keras.regularizers import l2

from sklearn.metrics import accuracy_score,recall_score,precision_score

from sklearn.metrics import f1_score

import pandas as pd

import numpy as np

from collections import Counter

import collections
def append_ext(fn):

    return fn+".jpeg"
traindf=pd.read_csv("/kaggle/input/melanoma/melanoma dataset withCSV file/train.csv",dtype=str)

testdf=pd.read_csv("/kaggle/input/melanoma/melanoma dataset withCSV file/testlabel.csv",dtype="category")
data=dict( enumerate(testdf['label'].cat.categories ) )

print(data)
value=testdf['label'] = testdf.label.astype('category').cat.codes

print(value.head())
c = testdf.label.astype('category')

print(c.head())
traindf["id"]=traindf["id"].apply(append_ext)

testdf["id"]=testdf["id"].apply(append_ext)
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.20)
train_generator=datagen.flow_from_dataframe(

dataframe=traindf,

directory="/kaggle/input/melanoma/melanoma dataset withCSV file/train",

x_col="id",

y_col="label",

subset="training",

batch_size=5,

seed=30,

shuffle=True,

class_mode="categorical",

target_size=(32,32))
valid_generator=datagen.flow_from_dataframe(

dataframe=traindf,

directory="/kaggle/input/melanoma/melanoma dataset withCSV file/train",

x_col="id",

y_col="label",

subset="validation",

batch_size=5,

seed=30,

shuffle=True,

class_mode="categorical",

target_size=(32,32))
test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(

dataframe=testdf,

directory="/kaggle/input/melanoma/melanoma dataset withCSV file/test",

x_col="id",

y_col=None,

batch_size=5,

seed=30,

shuffle=False,

class_mode=None,

target_size=(32,32))
model=Sequential()

#Layer 1

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32,32,3)))

model.add(Activation('relu'))

#Layer 2

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

# Layer 3

model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

# Layer 4

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

# Layer 5

model.add(Flatten())

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.5))

# Layer 6

model.add(Dense(2, activation='softmax'))

model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
#Fitting the model:

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

print(STEP_SIZE_TRAIN)

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

print(STEP_SIZE_VALID)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

print(STEP_SIZE_TEST)
model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=50)
def Model_metrics(loaded_model):

    VALIDATION_score = loaded_model.evaluate_generator(generator=valid_generator,steps=STEP_SIZE_TEST)

    print("VALIDATION_SCORE:",score[1])

    train_score = loaded_model.evaluate_generator(generator=train_generator,steps=STEP_SIZE_TRAIN,verbose=1)

    print("TRAIN_SCORE:",train_score[1])
#SAVE ND LOAD 

model_json = model.to_json()

with open("Model.json", "w") as json_file:

    json_file.write(model_json)



# serialize weights to HDF5

model.save_weights("Model.h5")

print("Saved model to disk")



# load weights into new model

loaded_model.load_weights("Model.h5")

print("Loaded model from disk")
Model_metrics(loaded_model)
#Evaluate the model

VALIDATION_score=model.evaluate_generator(generator=valid_generator,steps=STEP_SIZE_TEST)

print("VALIDATION_SCORE:",score[1])

train_score=model.evaluate_generator(generator=train_generator,steps=STEP_SIZE_TRAIN,verbose=1)

print("TRAIN_SCORE:",train_score[1])
test_generator.reset()

pred=model.predict_generator(test_generator,

steps=STEP_SIZE_TEST,verbose=1)

print(pred)

print(len(pred))

#Before prediction test arrayset

codes, cats = pd.factorize(testdf.label)

print("Before prediction testdataset:\n",codes)

#length of array

print("Length of testdataset before prediction:",len(codes))

print("----------------------------------------------------------------------")
#After prediction array set

predicted_class_indices=np.argmax(pred,axis=1)

predicted=predicted_class_indices

print("After prediction testdataset:\n",predicted)

print("Length of testdataset after prediction:",len(predicted))

print("---------------------------------------------------------------------")
#No. of images predicted perfectly 

similar=sum(a == b for a,b in zip(codes, predicted))

print("Total no.of images predicted correctly: ",similar)

print("---------------------------------------------------------------------")

labels = (train_generator.class_indices)

#print(labels)

labels = dict((v,k) for k,v in labels.items())

print(labels)
predictions = [labels[k] for k in predicted_class_indices]

print("After prediction:\n",predictions)
#SPLIT DATASET IN MELANOMA AND NON-MELANOMA

def split_list(a_list):

   return a_list[0:31], a_list[31:]
B, C = split_list(predicted)

D, E = split_list(predictions)

F,G=split_list(codes)
#Precision tp/tp+fp



#Recall tp/tp+fn



print("MELANOMA TEST DATASET ")

print("Length of dataset:",len(F))

print("Before prediction [0: melanoma , 1: non-melanoma]:\n",F)

my_dict = dict(Counter(F))

print("Count of melanoma predicted correctly which are only 0's:\n",my_dict)

print("After prediction [0: melanoma , 1: non-melanoma]:\n",B)

my_dict = dict(Counter(B))

print("Count of melanoma predicted correctly which are only 0's:\n",my_dict)

print("---------------------------------------------------------------------")
print("NON-MELANOMA TEST DATASET")

print("Length of dataset:",len(G))

print("Before prediction [0: melanoma , 1: non-melanoma]:\n",G)

my_dict = dict(Counter(G))

print("Count of melanoma predicted correctly which are only 0's:\n",my_dict)

print("After prediction [0: melanoma , 1: non-melanoma]:\n",C)

my_dict = dict(Counter(C))

print("Count of non-melanoma predicted correctly which are only 1's:\n",my_dict)

print("---------------------------------------------------------------------")
#Precision tp/tp+fp



#Recall tp/tp+fn



#print("MELANOMA dataset after prediction:\n",D)

#print("NON-MELANOMA dataset after prediction:\n",E)

print("---------------------------------------------------------------------")

print("Total no.of images predicted correctly: ",similar)

print("---------------------------------------------------------------------")

print("Accuracy Score:",accuracy_score(codes,predicted))

#ASSUME MELANOMA POSITIVE NONMELANOMA NEGATIVE

#melanoma predicted true positive melanoma:0

print("---------------------------------------------------------------------")

print("MELANOMA PRECISION RECALL F1")

TP=collections.Counter(B)[0]

print("Melanoma_truepositive",TP)

#melanoma predicted false negative nonmleanoma:1

FN=collections.Counter(B)[1]

print("Melanoma_false negative",FN)

#nonmelanoma predicted false positive melanoma:0

FP=collections.Counter(C)[0]

print("Non melanoma_false positive",FP)

#nonmelanoma predicted TRUE negative nonmelanoma:1

TN=collections.Counter(C)[1]

print("Non melanoma_True negative",TN)

#PRECISION tp/tp+fp

TRUE_POS_NEG=TP+FP

#print(TRUE_POS_NEG)

PRECISION_SCORE=TP/TRUE_POS_NEG

print("Melanoma_Precision_score",PRECISION_SCORE)

#RECALL tp/tp+fn

TRUE_POS_FALSE_NEG=TP+FN

RECALL_SCORE=TP/TRUE_POS_FALSE_NEG

print("Melanoma_Recall_score",RECALL_SCORE)

print("------------------------------------------------------")





#NONMELANOMA PRECISION ND RECALL

#nonmelanoma predicted true positive nonmleanoma:1

print("NONMELANOMA PRECISION RECALL F1")

Non_TP=collections.Counter(C)[1]

print("Non melanoma_truepositive",Non_TP)

#nonmelanoma predicted false negative nonmleanoma:1

Non_FN=collections.Counter(C)[0]

print("Non melanoma_false negative",Non_FN)

#melanoma predicted false positive melanoma:0

Non_FP=collections.Counter(B)[1]

print("Melanoma_false positive",Non_FP)

#melanoma predicted TRUE negative melanoma:1

Non_TN=collections.Counter(B)[0]

print("Melanoma_True negative",Non_TN)

#PRECISION tp/tp+fp

TRUE_POS_NEG=Non_TP+Non_FP

#print(TRUE_POS_NEG)

PRECISION_SCORE1=Non_TP/TRUE_POS_NEG

print("Non melanoma Precision_score",PRECISION_SCORE1)

#RECALL tp/tp+fn

TRUE_POS_FALSE_NEG=Non_TP+Non_FN

RECALL_SCORE1=Non_TP/TRUE_POS_FALSE_NEG

print("Non melanoma Recall_score",RECALL_SCORE1)







#ACCURACY TP+TN/TP+TN+FP+FN

total_predictions=TP+TN+FP+FN

#print("total_predictions",total_predictions)

Noofpredicitions=TP+TN

ACCURACY=Noofpredicitions/total_predictions

print("Accuracy:",ACCURACY)

#F1_SCORE 

precirecall=2*PRECISION_SCORE*RECALL_SCORE

#print(precirecall)

precisirecall=PRECISION_SCORE+RECALL_SCORE

#print(precisirecall)

f1score=precirecall/precisirecall

print("melanoma_f1_score",f1score)





precirecall1=2*PRECISION_SCORE1*RECALL_SCORE1

#print(precirecall)

precisirecall1=PRECISION_SCORE1+RECALL_SCORE1

#print(precisirecall)

f1score1=precirecall1/precisirecall1

print("Non melanoma_f1_score",f1score1)



#Finally, save the results to a CSV file.

filenames=test_generator.filenames

#print("Testdata images with .JPEG:\n",filenames)

a={'Filename':filenames,'Predictions':predictions}

#print(a)

results=pd.DataFrame.from_dict(a, orient='index')

print(results.transpose())

results.to_csv("PREDICTED_FILE.csv",index=False)

#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
#loaded weights

json_file = open('Model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)
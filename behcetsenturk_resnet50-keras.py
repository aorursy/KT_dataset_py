# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import cv2

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# We get csv files as dataframe

# csv dosyasını pandas dataframe olarak alıyoruz

dataFrame = pd.read_csv("../input/humpback-whale-identification/train.csv")

dataFrame.tail()
# We won't use new_whale images

# new_whale sınıfında ki resimleri kullanmıyoruz

dataFrame = dataFrame[dataFrame.Id != "new_whale"]
# We turn and split train set inputs as numpy array.

# Eğitim setinin girdilerini numpy array olarak ayırıyoruz

trainImageNames = dataFrame.Image.values
classSampleCountMinLimit = 3

# We find nıt unique classes and create a list with them. Our minimum instance limit is classSampleCountMinLimit.

# Burda classSampleCountMinLimit'dan fazla örneği bulunan sınıfların listesini oluşturuyoruz.

labelsCountList = dataFrame.Id.value_counts()

nonUniqueLabelsList = []

for i in range(len(labelsCountList)):

    if labelsCountList[i] >= classSampleCountMinLimit and labelsCountList.index[i] not in nonUniqueLabelsList:

        nonUniqueLabelsList.append(labelsCountList.index[i])
# We won't use train_test_split because we must use nonUniqueLabelsList when we split

# train_test_split yerine validation setimizi kendimiz ayırıyoruz.



count = 0

X_train_names = [] 

X_val_names = []



for i in range(len(trainImageNames)):

    if dataFrame.Id.values[i] in nonUniqueLabelsList and count < 1500:

        X_val_names.append(trainImageNames[i])

        count = count + 1

    else:

        X_train_names.append(trainImageNames[i])

        

X_train_names = np.array(X_train_names)

X_val_names = np.array(X_val_names)
#from sklearn.model_selection import train_test_split

#X_train_names, X_val_names, y_train, y_val = train_test_split(trainImageNames, labelNames, test_size = 0.1, random_state = 11)
print("Shapes ->\nX_train = %s\nX_val = %s" % (X_train_names.shape, X_val_names.shape))
# We split outputs.

count = 0

y_train_names = []

y_val_names = []



for i in range(len(dataFrame.Id.values)):

    if dataFrame.Id.values[i] in nonUniqueLabelsList and count < 1500:

        y_val_names.append(dataFrame.Id.values[i])

        count = count + 1

    else:

        y_train_names.append(dataFrame.Id.values[i])



y_train_names = np.array(y_train_names)

y_val_names = np.array(y_val_names)
print("The classes found in validation set(Top 10)")

pd.value_counts(pd.Series(y_val_names))[:10]
#y_train_names, y_val_names = train_test_split(dataFrame.Id.values, test_size = 0.1, random_state = 11)
# To use data flow from directory we need special directory structure

# We need a directory for every class.



# Keras'ın data_generatörü için gereken dizin yapısını oluşturuyoruz

# Örnek dizin yapısında her sınıf için bir klasör gerekli



#data/

#    train/

#        dogs/

#            dog001.jpg

#            dog002.jpg

#            ...

#        cats/

#            cat001.jpg

#            cat002.jpg

#            ...

#    validation/

#        dogs/

#            dog001.jpg

#            dog002.jpg

#            ...

#        cats/

#            cat001.jpg

#            cat002.jpg

#            ...



os.mkdir("./data")



os.mkdir("./data/validation")

for i in y_train_names:

    if not os.path.exists("./data/validation/"+i):

        os.mkdir("./data/validation/"+i)

    

os.mkdir("./data/train")

for i in y_train_names:

    if not os.path.exists("./data/train/"+i):

        os.mkdir("./data/train/"+i)
# Copying our images to their new places.

# Oluşturduğumuz dizin yapısına resimlerimizi kopyalıyoruz



import shutil

for i in range(len(X_val_names)):

    shutil.copy("../input/humpback-whale-identification/train/"+X_val_names[i], "./data/validation/"+y_val_names[i]+"/"+X_val_names[i])

for i in range(len(X_train_names)):    

    shutil.copy("../input/humpback-whale-identification/train/"+X_train_names[i], "./data/train/"+y_train_names[i]+"/"+X_train_names[i])
# We get pretrained resnet50 and delete last layer.

# PreTrained resnet50'yi alıp son katmanı atıyoruz



from tensorflow.python.keras.applications import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout



num_classes = np.unique(y_train_names).shape[0]

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



model = Sequential()

model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

#model.add(Dropout(0.25))

model.add(Dense(num_classes, activation='softmax'))



# We need to write this to say keras that don't train the first layers of resnet50

# Resnet'in ilk katmanları zaten eğitilmiş olduğu için bunu belirtmemiz lazım

model.layers[0].trainable = False
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# https://stackoverflow.com/a/46709583

# We need some samples to fit data_image_generator. Click above for more info



# Sample veriye ihtiyacımız var data_augmention generator'un std normalizasyonu için.

X_sample = []

trainHeight = 224

trainWidth = 224

for i in os.listdir('../input/humpback-whale-identification/train/')[:500]:

    img = cv2.imread('../input/humpback-whale-identification/train/'+i) # Resimleri tek tek alıyoruz

    img = cv2.resize(img, dsize=(trainWidth, trainHeight), interpolation=cv2.INTER_LINEAR) # Resize ediyoruz INTER_LINEAR yerine başka alogritmalar ilede resim küçültme yapabiliriz

    X_sample.append(img)
from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator



# resNet's resolution

# resNet 224x224 resolution resimler ile eğitilmiş

image_size = 224



# Data augmention

# burda data augmention yapıyoruz

data_generator = ImageDataGenerator(

        preprocessing_function=preprocess_input,

        #rotation_range=40,

        #width_shift_range=0.2,

        #height_shift_range=0.2,

        #shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        vertical_flip=True,

        featurewise_center=True, 

        featurewise_std_normalization=True,

        #rescale=1./255,

        fill_mode='nearest')



data_generator.fit(X_sample)



# train ve validation verilerimizi çekiyoruz

train_generator = data_generator.flow_from_directory(

        'data/train',

        target_size=(image_size, image_size),

        batch_size=64,

        class_mode='categorical')



validation_generator = data_generator.flow_from_directory(

        'data/validation',

        target_size=(image_size, image_size),

        batch_size=64,

        class_mode='categorical')



# We fit model here

# modeli eğitiyoruz

model.fit_generator(

        train_generator,

        epochs = 20,

        steps_per_epoch=X_train_names.shape[0] // 64,

        validation_data=validation_generator,

        validation_steps=1)
# We get sample submisson file. We use it as a template.

# Sample submisson dosyasını çekiyoruz predictionları o tablo üzerine yeniden dolduracağız

sample_df = pd.read_csv("../input/humpback-whale-identification/sample_submission.csv")

testImagesNames = list(sample_df.Image)
# We get test images with openCV

X_test = []

for i in testImagesNames:

    img = cv2.imread('../input/humpback-whale-identification/test/'+i) # Resimleri tek tek alıyoruz

    img = cv2.resize(img, dsize=(trainWidth, trainHeight), interpolation=cv2.INTER_LINEAR) # Resize ediyoruz INTER_LINEAR yerine başka alogritmalar ilede resim küçültme yapabiliriz

    X_test.append(img)
# We normalize test images.



X_test = np.array(X_test)



k = X_test.shape[0]//1000

for i in range(k):

    X_test[i*1000:i*1000+1000] = X_test[i*1000:i*1000+1000] / 255.0

    if i == k-1:

        X_test[i*1000:] = X_test[i*1000:] / 255.0
# We predict test images

predictions = model.predict(X_test, verbose=1)
# If a pediction is higher than 0.4 we will get it's index num.

# Eğer 0.4'ten daha yukarıda bir sınıfı tahmin etmişsek onun index numarasını alıyoruz.

nonNewWhalePredIndexList = [indx for indx,i in enumerate(predictions) for j in i if j > 0.4]
print("Non new_whale class image count =",np.unique(np.array(nonNewWhalePredIndexList)).shape[0])
# We add 1.0 to begin of prediction array if it's in new_whale class

# Prediction listesinin başına 1.0 ekliyoruz ki new_whale class'ı olduğu kesinleşsin.

pred2 = []

for indx in range(len(predictions)):

    if indx not in nonNewWhalePredIndexList:

        newPredItem = np.insert(predictions[indx], 0, 1.0)

        pred2.append(newPredItem)

    else:

        pred2.append(predictions[indx])

pred2 = np.array(pred2)
# We must delete our files to get rid of kaggle commit errors.

shutil.rmtree("./data/")
# In here we create our submission file.

labels_list = list(np.unique(y_train_names))

labels_list = ['new_whale'] + labels_list

pred_list = [[labels_list[i] for i in p.argsort()[-5:][::-1]] for p in pred2]

pred_dic = dict((key, value) for (key, value) in zip(testImagesNames,pred_list))

pred_list_cor = [' '.join(pred_dic[id]) for id in testImagesNames]

df = pd.DataFrame({'Image':testImagesNames,'Id': pred_list_cor})

df.to_csv('submission.csv', header=True, index=False)

df.head()
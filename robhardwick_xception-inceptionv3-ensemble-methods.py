import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



from keras.preprocessing.image import ImageDataGenerator

from keras.applications.xception import Xception

from keras.applications.inception_v3 import InceptionV3



from keras.applications.xception import preprocess_input as xception_preprocessor

from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor
LABELS = "../input/dog-breed-identification/labels.csv"



train_df = pd.read_csv(LABELS)

#return top 16 value counts and convert into list

plt.figure(figsize=(13, 6))

train_df['breed'].value_counts().plot(kind='bar')

plt.show()



top_breeds = sorted(list(train_df['breed'].value_counts().head(16).index))

train_df = train_df[train_df['breed'].isin(top_breeds)]



print(top_breeds)
from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

from sklearn.model_selection import train_test_split



SEED = 1234



TRAIN_FOLDER = "../input/dog-breed-identification/train/"

TEST_FOLDER = "../input/dog-breed-identification/test/"



DIM = 299



train_df['image_path'] = train_df.apply( lambda x: ( TRAIN_FOLDER + x["id"] + ".jpg" ), axis=1 )



train_data = np.array([ img_to_array(load_img(img, target_size=(DIM, DIM))) for img in train_df['image_path'].values.tolist()]).astype('float32')

train_labels = train_df['breed']





x_train, x_validation, y_train, y_validation = train_test_split(train_data, train_labels, test_size=0.2, stratify=np.array(train_labels), random_state=SEED)



#calculate the value counts for train and validation data

data = y_train.value_counts().sort_index().to_frame()

data.columns = ['train']

data['validation'] = y_validation.value_counts().sort_index().to_frame()



new_plot = data[['train','validation']].sort_values(['train']+['validation'], ascending=False)

new_plot.plot(kind='bar', stacked=True)

plt.show()
# Let's convert our labels into one hot encoded format



y_train = pd.get_dummies(y_train.reset_index(drop=True), columns=top_breeds).as_matrix()

y_validation = pd.get_dummies(y_validation.reset_index(drop=True), columns=top_breeds).as_matrix()



print(y_train[0])
plt.subplot(1, 2, 1)

plt.title(top_breeds[np.where(y_train[5]==1)[0][0]])

plt.axis('off')

plt.imshow(x_train[5].astype(np.uint8))



plt.subplot(1, 2, 2)

plt.title(top_breeds[np.where(y_train[7]==1)[0][0]])

plt.axis('off')

plt.imshow(x_train[7].astype(np.uint8))

plt.show()
from os import makedirs

from os.path import expanduser, exists, join



!ls ../input/keras-pretrained-models/



cache_dir = expanduser(join('~', '.keras'))

if not exists(cache_dir):

    makedirs(cache_dir)

models_dir = join(cache_dir, 'models')

if not exists(models_dir):

    makedirs(models_dir)

    

!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/

!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/

!cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss, accuracy_score



batch_size = 32

epochs = 30

num_classes = len(top_breeds)



def generate_features(model_info, data, labels, datagen):

    print("generating features...")

    datagen.preprocessing_function = model_info["preprocessor"]

    generator = datagen.flow(data, labels, shuffle=False, batch_size=batch_size, seed=model_info["seed"])

    bottleneck_model = model_info["model"](weights='imagenet', include_top=False, input_shape=model_info["input_shape"], pooling=model_info["pooling"])

    return bottleneck_model.predict_generator(generator)

import time



models = {

    "InceptionV3": {

        "model": InceptionV3,

        "preprocessor": inception_v3_preprocessor,

        "input_shape": (299,299,3),

        "seed": 1234,

        "pooling": "avg"

    },

    "Xception": {

        "model": Xception,

        "preprocessor": xception_preprocessor,

        "input_shape": (299,299,3),

        "seed": 5512,

        "pooling": "avg"

    }

}



for model_name, model in models.items():

    print("Predicting : {}".format(model_name))

    filename = model_name + '_features.npy'

    validfilename = model_name + '_validfeatures.npy'

    if exists(filename):

        features = np.load(filename)

        validation_features = np.load(validfilename)

    else:

        train_datagen = ImageDataGenerator(

                zoom_range = 0.3,

                width_shift_range=0.1,

                height_shift_range=0.1)

        validation_datagen = ImageDataGenerator()

        features = generate_features( model, x_train, y_train, train_datagen)

        validation_features = generate_features(model, x_validation, y_validation, validation_datagen)

        np.save(filename, features)

        np.save(validfilename, validation_features)

    

    # Now that we have created or loaded the features  we need to do some predictions.

    start_time = time.time()

    

    logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)

    logreg.fit(features, (y_train * range(num_classes)).sum(axis=1))



    model["predict_proba"] = logreg.predict_proba(validation_features)

    end_time = time.time()

    print('Training time : {} {}'.format(np.round((end_time-start_time)/60, 2),' minutes'))
probas = [ model["predict_proba"] for model_name, model in models.items() ]



avgprobas = np.average(probas, axis=0, weights=[1,1])



print('ensemble validation logLoss : {}'.format(log_loss(y_validation, avgprobas)))
features  = np.hstack( [ np.load(model_name + '_features.npy') for model_name, model in models.items() ])

validation = np.hstack( [ np.load(model_name + '_validfeatures.npy') for model_name, model in models.items() ])



logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)

logreg.fit(features, (y_train * range(num_classes)).sum(axis=1))



predict_probs = logreg.predict_proba(validation)



print('ensemble of features va logLoss : {}'.format(log_loss(y_validation, predict_probs)))
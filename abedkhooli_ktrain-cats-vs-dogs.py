import os

os.environ["DISABLE_V2_BEHAVIOR"]="1" 
!pip install -q -U pip

!pip install -q "tensorflow_gpu>=2.0.0"  # cpu: pip3 install "tensorflo

!pip install -q transformers

!pip install -q ktrain 
!pip install -q git+https://github.com/amaiya/eli5@tfkeras_0_10_1 

!pip install -q git+https://github.com/amaiya/stellargraph@no_tf_dep_082
import ktrain

import logging

from ktrain import vision as vis

import stellargraph, eli5, tensorflow

eli5.__version__, stellargraph.__version__, ktrain.__version__, tensorflow.__version__
tensorflow.get_logger().setLevel(logging.WARNING) # ERROR
!cp /kaggle/input/dogs-vs-cats/train.zip train.zip

!cp /kaggle/input/dogs-vs-cats/test1.zip test1.zip

!unzip -q test1.zip

!unzip -q train.zip
!mkdir /kaggle/working/dc

!cp -r train /kaggle/working/dc/

!mkdir /kaggle/working/dc/valid

!mv /kaggle/working/dc/train/cat.1*.jpg /kaggle/working/dc/valid/

!mv /kaggle/working/dc/train/dog.1*.jpg /kaggle/working/dc/valid/



!mkdir /kaggle/working/dc/train/dogs 

!mkdir /kaggle/working/dc/train/cats

!mkdir /kaggle/working/dc/valid/dogs 

!mkdir /kaggle/working/dc/valid/cats

!mv /kaggle/working/dc/train/dog.*.jpg /kaggle/working/dc/train/dogs/

!mv /kaggle/working/dc/train/cat.*.jpg /kaggle/working/dc/train/cats/

!mv /kaggle/working/dc/valid/dog.*.jpg /kaggle/working/dc/valid/dogs/

!mv /kaggle/working/dc/valid/cat.*.jpg /kaggle/working/dc/valid/cats/
DATADIR = '/kaggle/working/dc'

(train_data, val_data, preproc) = vis.images_from_folder(

                                              datadir=DATADIR,

                                              data_aug = vis.get_data_aug(horizontal_flip=True),

                                              train_test_names=['train', 'valid'], 

                                              target_size=(224,224), color_mode='rgb')

model = vis.image_classifier('pretrained_resnet50', train_data, val_data, freeze_layers=15)
learner = ktrain.get_learner(model=model, train_data=train_data, val_data=val_data, 

                             workers=8, use_multiprocessing=False, batch_size=64)
learner.lr_find(max_epochs=2 , show_plot=True)
import gc , torch

gc.collect()

torch.cuda.empty_cache()
learner.fit_onecycle(1e-5, 3)
loss, acc = learner.model.evaluate_generator(learner.val_data, 

                                             steps=len(learner.val_data))

print(f'final loss:{loss}, final accuracy:{acc}')
learner.view_top_losses(n=3, preproc=preproc)
predictor = ktrain.get_predictor(learner.model, preproc)

predictor.predict_filename('/kaggle/working/dc/valid/cats/cat.11724.jpg')
# forked copies seem to fail to install on docker (need pip3)

predictor.explain('/kaggle/working/dc/valid/cats/cat.11724.jpg')
#=============== working with eli5 directly (may not work due to pip3 absence?)

import keras

import numpy as np

dims = learner.model.input_shape[1:3]

image_uri = '/kaggle/working/dc/valid/cats/cat.11724.jpg'

im = keras.preprocessing.image.load_img(image_uri, target_size=dims)

doc = keras.preprocessing.image.img_to_array(im)



doc = np.expand_dims(doc, axis=0)

image = keras.preprocessing.image.array_to_img(doc[0])

ex = eli5.explain_prediction(learner.model, doc)

eli5.show_prediction(model, doc)
ex.error
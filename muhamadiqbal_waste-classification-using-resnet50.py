from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
num_class = 6



model = Sequential([ResNet50(include_top=False,

                             weights='imagenet',

                             pooling='avg'),

                    Dense(num_class, activation='softmax')])



model.layers[0].trainable = False
from tensorflow.keras.optimizers import SGD
lr = 0.01

momentum = 0.001

opt = SGD(learning_rate=lr, momentum=momentum)



model.compile(optimizer=opt,

              loss='categorical_crossentropy',

              metrics=['accuracy'])



model.summary()
data_path = "../input/split-garbage-dataset/split-garbage-dataset"



data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)



img_shape = 224

train_gen = data_generator.flow_from_directory(data_path + '/train',

                                               target_size=(img_shape, img_shape),

                                               batch_size=64,

                                               class_mode='categorical')



val_gen = data_generator.flow_from_directory(data_path + '/valid',

                                             target_size=(img_shape, img_shape),

                                             batch_size=1,

                                             class_mode='categorical',

                                             shuffle=False)
from tensorflow.keras.callbacks import ModelCheckpoint
n_epoch = 50



model_name = 'resnet50_batch64_sgd01m001'

checkpoint = ModelCheckpoint('./' +  model_name + '.h5',

                             monitor='val_loss',

                             save_best_only=True,

                             verbose=1)



history = model.fit_generator(train_gen,

                              steps_per_epoch=train_gen.samples/train_gen.batch_size,

                              validation_data=val_gen,

                              validation_steps=val_gen.samples/val_gen.batch_size,

                              epochs=n_epoch,

                              callbacks=[checkpoint])
import matplotlib.pyplot as plt

import seaborn as sns
val_acc = history.history['val_accuracy']

val_loss = history.history['val_loss']

acc = history.history['accuracy']

loss = history.history['loss']
plt.plot(range(n_epoch), acc, 'b*-', label = 'Training accuracy')

plt.plot(range(n_epoch), val_acc, 'r', label = 'Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()
plt.plot(range(n_epoch), loss, 'b*-', label = 'Training loss')

plt.plot(range(n_epoch), val_loss, 'r', label = 'Validation loss')

plt.title('Training and validation loss')

plt.legend()
data_path = "../input/split-garbage-dataset/split-garbage-dataset"



data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)



test_gen = data_generator.flow_from_directory(data_path + '/test',

                                              target_size=(img_shape, img_shape),

                                              batch_size=1,

                                              class_mode='categorical',

                                              shuffle=False)
from tensorflow.keras.models import load_model

import numpy as np
eval_model = load_model('./' + model_name + '.h5')

eval_model.summary()
y_pred = eval_model.predict_generator(test_gen)

y_pred = np.argmax(y_pred, axis=1)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(test_gen.classes, y_pred))
cf_matrix = confusion_matrix(test_gen.classes, y_pred)



plt.figure(figsize=(8,5))

heatmap = sns.heatmap(cf_matrix, annot=True, fmt='d', color='blue')

plt.xlabel('Predicted class')

plt.ylabel('True class')

plt.title('Confusion matrix of model')
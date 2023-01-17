import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

import cv2



from tensorflow.keras.applications import vgg16

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers, models, optimizers, callbacks



from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split



from sklearn.experimental import enable_hist_gradient_boosting  # noqa

from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier, RandomForestClassifier

from sklearn.svm import SVC



from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline, make_pipeline



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import RandomizedSearchCV



from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



import warnings

warnings.filterwarnings("ignore")
!ls ../input/identify-the-dance-form/
train_labels = pd.read_csv("../input/identify-the-dance-form/train.csv")

train_labels.head()
img = cv2.imread('/kaggle/input/identify-the-dance-form/train/404.jpg')

plt.imshow(img)

plt.title(train_labels[train_labels.Image=="404.jpg"].target.values[0])

plt.show()
def load_data(df, path):

    images = []

    labels = []

    for i in zip(df.values):

        file = i[0][0]

        label = i[0][1]

        image = cv2.resize(cv2.imread(path+file), 

                           (224,224))

        image = vgg16.preprocess_input(image)

        images.append(image)

        labels.append(label)

    return np.array(images), np.array(labels)



image_path = "/kaggle/input/identify-the-dance-form/train/"



X, y = load_data(train_labels, image_path)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)



encoder = OneHotEncoder(sparse=False)

y_train = encoder.fit_transform(y_train.reshape(-1,1))

y_val = encoder.transform(y_val.reshape(-1,1))
data_gen = ImageDataGenerator(rotation_range=30,

                              width_shift_range=0.25,

                              height_shift_range=0.25,

                              shear_range=0.2,

                              zoom_range=0.3,

                              horizontal_flip=True,

                              fill_mode='nearest')



data_gen.fit(X_train)
num_classes = len(np.unique(y))



model = vgg16.VGG16(include_top=False, weights='imagenet', 

                    pooling='avg', input_shape = (224, 224, 3))







for layer in model.layers[:17]:

    layer.trainable = False







x = layers.Dense(1024, activation='relu')(model.output)

x = layers.Dense(512, activation='relu')(x)

output = layers.Dense(num_classes, activation='softmax')(x)



model = models.Model(model.input, output)



model.compile(loss="categorical_crossentropy",

              optimizer=optimizers.RMSprop(lr = 0.001),

              metrics=['accuracy'])
lr_reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', 

                                        patience=2, 

                                        verbose=1, 

                                        factor=0.5, 

                                        min_lr=0.00001)



hist = model.fit(data_gen.flow(X_train, y_train, batch_size=64),

                 steps_per_epoch=len(X_train)/64, epochs=50,

                 validation_data=(X_val, y_val),

                 callbacks = [lr_reduce])
plt.figure(figsize=(10,7))

plt.subplot(121)

plt.plot(hist.history["loss"])

plt.plot(hist.history["val_loss"])

plt.legend(["train","val"])



plt.subplot(122)

plt.plot(hist.history["accuracy"])

plt.plot(hist.history["val_accuracy"])

plt.legend(["train","val"])



plt.show()
extractor = models.Model(model.input, model.layers[-3].output)
X_new_train = extractor.predict(X_train)

X_new_val = extractor.predict(X_val)



scaler = StandardScaler()

X_new_train = scaler.fit_transform(X_new_train)

X_new_val = scaler.transform(X_new_val)



print("Train Shape:", X_new_train.shape, ", Val Shape:", X_new_val.shape)
y_new_train = encoder.inverse_transform(y_train).reshape(-1)

y_new_val = encoder.inverse_transform(y_val).reshape(-1)
learning_rate = np.linspace(1e-3, 1, num=500)



max_iter = list(range(50,301))



l2_regularization = np.linspace(0, 5, num=500)



max_leaf_nodes = list(range(10, 51))



max_depth = list(range(3, 21))



min_samples_leaf = list(range(10,51))



# Create the random grid

random_grid = {'learning_rate': learning_rate,

               'max_iter': max_iter,

               'max_leaf_nodes': max_leaf_nodes,

               'l2_regularization': l2_regularization,

               'max_depth': max_depth,

               'min_samples_leaf':min_samples_leaf}



random_grid = {'histgradientboostingclassifier__' + key: random_grid[key] for key in random_grid}



smote_par = {"mohiniyattam":60, "odissi":60, "bharatanatyam":60,

             "kathakali":60, "sattriya":60, "kathak":60,

             "kuchipudi":60, "manipuri":60}

clf = make_pipeline(SMOTE(smote_par),

                    HistGradientBoostingClassifier(loss="categorical_crossentropy"))



gb_random = RandomizedSearchCV(clf, random_grid, n_iter=50, cv=3,

                              random_state=1, n_jobs=-1, refit=True)

    

gb_random.fit(X_new_train, y_new_train)
print(gb_random.best_score_)

gb_random.best_estimator_
y_pred = gb_random.best_estimator_.predict(X_new_val)



print(classification_report(y_new_val,y_pred,

      labels=np.unique(y_new_val)))



sns.heatmap(confusion_matrix(y_new_val,y_pred), 

            xticklabels=np.unique(y_new_val),

            yticklabels=np.unique(y_new_val),

            annot=True, cbar=False, cmap="Blues")



plt.show()
random_grid = {'C': [0.1, 1, 5, 10, 100], 

              'gamma': [1,0.1,0.01,0.001, 0.0005],

              'degree': [2, 3, 4],

              'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}



random_grid = {'svc__' + key: random_grid[key] for key in random_grid}



smote_par = {"mohiniyattam":60, "odissi":60, "bharatanatyam":60,

             "kathakali":60, "sattriya":60, "kathak":60,

             "kuchipudi":60, "manipuri":60}

clf = make_pipeline(SMOTE(smote_par),

                    SVC())



svm_random = RandomizedSearchCV(clf, random_grid, n_iter=50, cv=3,

                              random_state=1, n_jobs=-1, refit=True)

    

svm_random.fit(X_new_train, y_new_train)
print(svm_random.best_score_)

svm_random.best_estimator_
y_pred = svm_random.best_estimator_.predict(X_new_val)



print(classification_report(y_new_val,y_pred,

      labels=np.unique(y_new_val)))



sns.heatmap(confusion_matrix(y_new_val,y_pred), 

            xticklabels=np.unique(y_new_val),

            yticklabels=np.unique(y_new_val),

            annot=True, cbar=False, cmap="Blues")



plt.show()
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 300, num = 11)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']



# Maximum depth of the trees

max_depth = [int(x) for x in np.linspace(10, 50, num = 10)]



# Minimum number of samples required for the split

min_samples_split = [2, 5, 10]



# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4, 6, 8, 10]



# Method of selecting samples for training each tree

bootstrap = [True, False]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



random_grid = {'randomforestclassifier__' + key: random_grid[key] for key in random_grid}



smote_par = {"mohiniyattam":60, "odissi":60, "bharatanatyam":60,

             "kathakali":60, "sattriya":60, "kathak":60,

             "kuchipudi":60, "manipuri":60}



clf = make_pipeline(SMOTE(smote_par),

                    RandomForestClassifier())



rf_random = RandomizedSearchCV(clf, random_grid, n_iter=50, cv=3,

                              random_state=1, n_jobs=-1, refit=True)

    

rf_random.fit(X_new_train, y_new_train)
print(rf_random.best_score_)

rf_random.best_estimator_
y_pred = rf_random.best_estimator_.predict(X_new_val)



print(classification_report(y_new_val,y_pred,

      labels=np.unique(y_new_val)))



sns.heatmap(confusion_matrix(y_new_val,y_pred), 

            xticklabels=np.unique(y_new_val),

            yticklabels=np.unique(y_new_val),

            annot=True, cbar=False, cmap="Blues")



plt.show()
ensemble = VotingClassifier([("hgb",gb_random.best_estimator_["histgradientboostingclassifier"]),

                             ("svm",svm_random.best_estimator_["svc"]),

                             ("rf",rf_random.best_estimator_["randomforestclassifier"])])



clf = make_pipeline(SMOTE(smote_par),

                    ensemble)

clf.fit(X_new_train, y_new_train)



y_pred = clf.predict(X_new_val)



print(classification_report(y_new_val,y_pred,

      labels=np.unique(y_new_val)))
sns.heatmap(confusion_matrix(y_new_val,y_pred), 

            xticklabels=np.unique(y_new_val),

            yticklabels=np.unique(y_new_val),

            annot=True, cbar=False, cmap="Blues")



plt.show()
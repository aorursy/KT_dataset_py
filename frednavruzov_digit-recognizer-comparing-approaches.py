import pandas as pd

import numpy as np



# vanilla ML methods ------------------------------------------------

# linear algorithms

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC, SVC

# metric algorithms

from sklearn.neighbors import KNeighborsClassifier

# ensemble algorithms

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



# deep learning -----------------------------------------------------

# to build own CNN from scratch

from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, BatchNormalization, Activation, Flatten

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical



# visualize results / dimensionality reduction / manifold learning (TODO)

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

%matplotlib inline



# utilities / preprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold

from sklearn.pipeline import Pipeline
train = pd.read_csv('../input/train.csv', engine='c', sep=',')

test = pd.read_csv('../input/test.csv', engine='c', sep=',')



picture_size = (28, 28)

print(train.shape, test.shape)

assert (train.shape[1] - 1) == picture_size[0]*picture_size[1] # to test whether we have correct picture sizes (784px)

assert (test.shape[1]) == picture_size[0]*picture_size[1] # to test whether we have correct picture sizes (784px)

train.head()
# explore class balance and visualize some examples

print('Train label distribution\n{}'.format(train.label.value_counts(normalize=True)))
# visualize pictures

examples = [train[train.label == k].sample(1, random_state=42).values for k in range(10)]



f, axarr = plt.subplots(1, 10, squeeze=False, figsize=(12, 1.2))



for i,e in enumerate(examples):

    # reshape 1D to 2D array - a picture

    img = e[:, 1:].reshape(picture_size).astype(float)

    # then draw it

    axarr[0, i].set_title(i)

    axarr[0, i].imshow(img, cmap='gray', interpolation='bicubic')
# extract X, y from data

X, y = train.drop('label', axis=1).values, train['label'].values



# Scale features to [0,1], some models are sensitive to non-scaled data

sc = MinMaxScaler()

X = sc.fit_transform(X)





X_train, X_holdout, y_train, y_holdout = train_test_split(

                                            X,

                                            y,

                                            test_size=0.15,

                                            random_state=42,

                                            # to preserve initial class balance

                                            stratify=train[train.columns[0]].values, 

                                                         )
%%time

# Logistic regression

clf = LogisticRegression(n_jobs=-1, 

                         C=1000,

                         fit_intercept=False,

                         class_weight='balanced', 

                         random_state=42,

                         solver='lbfgs',

                         multi_class='multinomial',

                         verbose=3)





'''

clf.fit(X_train, y_train)

acc = accuracy_score(clf.predict(X_holdout), y_holdout)

print(acc)



# 0.908571428571 Wall time: 31.6 s

'''
%%time

# Logistic regression on PCA vectors - faster, higher accuracy

pca = PCA(n_components=150, random_state=42)

X_transformed = pca.fit_transform(X_train)



'''

clf.fit(X_transformed, y_train)

acc = accuracy_score(clf.predict(pca.transform(X_holdout)), y_holdout)

print(acc)



# 0.913968253968 Wall time: 15.2 s

'''
%%time

# Linear SVC

clf = LinearSVC(C=0.05, 

                fit_intercept=True, 

                class_weight='balanced',

                multi_class='crammer_singer',

                dual=False,

                random_state=42)



'''

clf.fit(X_train, y_train)

acc = accuracy_score(clf.predict(X_holdout), y_holdout)

print(acc)



# 0.921904761905 Wall time: 5.85 s

'''
%%time

# SVC on PCA vectors - slower, and lower accuracy

pca = PCA(n_components=150, random_state=42)

X_transformed = pca.fit_transform(X_train)



'''

clf.fit(X_transformed, y_train)

acc = accuracy_score(clf.predict(pca.transform(X_holdout)), y_holdout)

print(acc)



# 0.919206349206 Wall time: 11.2 s

'''
%%time

# SVM with non-linear kernel - nice!

clf = SVC(C=12,

          kernel='rbf',

          class_weight='balanced',

          random_state=42

         )



pca = PCA(n_components=40, random_state=42)

X_transformed = pca.fit_transform(X_train)



'''

clf.fit(X_transformed, y_train)

acc = accuracy_score(clf.predict(pca.transform(X_holdout)), y_holdout)

print(acc)



# 0.98253968254 Wall time: 15.6 s

'''
%%time

# KNN

clf = KNeighborsClassifier(n_jobs=-1, 

                           n_neighbors=10,

                           weights='distance', 

                           p=2

                          )



'''

clf.fit(X_train, y_train)

acc = accuracy_score(clf.predict(X_holdout), y_holdout)

print(acc)



# 0.965555555556 Wall time: 1min 7s

'''
%%time

# KNN on PCA vectors - faster + better accuracy

pca = PCA(n_components=40, random_state=42)

X_transformed = pca.fit_transform(X_train)



'''

clf.fit(X_transformed, y_train)

acc = accuracy_score(clf.predict(pca.transform(X_holdout)), y_holdout)

print(acc)



# 0.97380952381 Wall time: 7.32 s

'''
%%time

# Random Forest

clf = RandomForestClassifier(n_estimators=250, 

                             n_jobs=-1, 

                             random_state=42, 

                             class_weight='balanced',

                             criterion='gini'

                             

)



'''

clf.fit(X_train, y_train)

acc = accuracy_score(clf.predict(X_holdout), y_holdout)

print(acc)



# 0.967301587302 Wall time: 11.6 s

'''
%%time

# Gradient Boosting

clf = GradientBoostingClassifier(n_estimators=90, 

                                 random_state=42, 

                                 verbose=0,

                                 max_features='sqrt',

                                 subsample=0.7,

                                 learning_rate=0.1, 

                                 max_depth=5

                                

)



'''

clf.fit(X_train, y_train)

acc = accuracy_score(clf.predict(X_holdout), y_holdout)

print(acc)



# 0.957142857143 Wall time: 1min 36s

'''
# uncomment it to choose best hyperparameters



# best baseline = support vector machines

clf = SVC(

          kernel='rbf',

          class_weight='balanced',

          random_state=42

         )



# features

features = PCA()



# pipeline

model = Pipeline(

    [

        ('sc', sc), # scaling

        ('f', features), # feature extraction

        ('clf', clf) # classifier

    ]

)



# parameters to choose from

params = {

    # feature-based

    'f__n_components': [40], # [10, 40, 100],

    # model-based

    'clf__C': [12],  # [11, 12, 13],

    'clf__class_weight': ['balanced']

}



# cv split object (stratified = preserve class balance in train/validation)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)



# grid search object

gs = GridSearchCV(

    model, 

    param_grid=params, 

    scoring='accuracy', 

    n_jobs=-1, 

    cv=cv, 

    verbose=2, 

    error_score=0

)



# fit it

gs.fit(X_train, y_train)



# best model

clf = gs.best_estimator_

print(gs.best_score_)

print(clf)



# check if it's valid on holdout

acc = accuracy_score(clf.predict(X_holdout), y_holdout)

print(acc)
%%time



# now use the whole dataset to train

clf.fit(X, y) 



# prediction

y_pred = clf.predict(test.values)
# create a dataframe

subm = pd.DataFrame({'ImageId':[x for x in range(1, len(test) + 1)], 'Label':y_pred})

subm.head()
# save to csv

# subm.to_csv('../input/subm_svm.csv', index=False, encoding='utf-8')
# let's build simple convolutional network from scratch

model = Sequential()



# 1st set of layers

model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1)))

# Normalize the activations of the previous layer at each batch - apply a transformation 

# that maintains the mean activation close to 0 and the activation standard deviation close to 1.

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPool2D())

model.add(Dropout(0.25)) # regularization



# 2nd set of layers

model.add(Conv2D(32, (3, 3)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPool2D())

model.add(Dropout(0.25))



# fully-connected layers, flatten previous output and connect n x m

model.add(Flatten())

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.5))



# 10 digits -> 10 classes

model.add(Dense(10))

model.add(Activation('softmax')) # probabilistic output



# compile model

model.compile(loss='categorical_crossentropy', 

              optimizer='adamax', 

              metrics=['accuracy']

             )



# let's explore model structure

model.summary()
# preprocess initial data to feed itto CNN

train = pd.read_csv('../input/train.csv', engine='c', sep=',')

test = pd.read_csv('../input/test.csv', engine='c', sep=',')



train = train.values



train_x = train[:,1:].reshape(train.shape[0], 28, 28, 1).astype('float32') / 255.

train_y = to_categorical(train[:, 0], 10) # 3 -> [0,0,0,1,0,0,0,0,0,0]





test = test.values

test_x = test.reshape(test.shape[0], 28, 28, 1).astype('float32') / 255.
# augment dataset with random shifts, zooms, rotations or use dataset as is (choose below)

datagen = ImageDataGenerator(

            width_shift_range=0.1,

            height_shift_range=0.1,

            zoom_range=0.1, 

            rotation_range=10 # degrees

    )
# train model (uncomment selected case)

'''

model.fit_generator( 

    datagen.flow(train_x, train_y, batch_size=64), 

    len(train_x), 

    nb_epoch=3,

)

'''



# uncomment if you have CUDA-optimized GPU, accuracy > 99.5%

# model.fit(train_x, train_y, batch_size=64, epochs=10)



# use epochs = 10 to get correct results

model.fit(train_x, train_y, batch_size=128, epochs=1)


# save to csv

# subm.to_csv('../input/subm_cnn.csv', index=False, encoding='utf-8')# predict classes

y = model.predict_classes(test_x)



# create a dataframe

subm = pd.DataFrame({'ImageId':[x for x in range(1, len(test_x) + 1)], 'Label':y})

subm.head()
# save to csv

# subm.to_csv('../input/subm_cnn.csv', index=False, encoding='utf-8')
import numpy as np

import pandas as pd

import time

import re_text_filter

import vectorization

import nltk



from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.decomposition import PCA, TruncatedSVD



from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.svm import SVC 

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

import xgboost



from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report



import matplotlib

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches



"Show matplotlib plots in nice, high quality, high resolution."

plt.style.use('ggplot')

%matplotlib inline

matplotlib.rcParams['figure.dpi'] = 300



from sklearn.tree import plot_tree



import tensorflow as tf
df_train = pd.read_csv("../input/nlp-getting-started/train.csv")

df_train.head(10)
train_set_size = len(df_train)

train_set_positive_size = len(df_train[df_train.target==1])

train_set_negative_size = len(df_train[df_train.target==0])

print(f'{train_set_size} total observations; {train_set_positive_size} positive, {train_set_negative_size} negative.')
X = df_train['text']; y = df_train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train
"Names of regular-expression based filtering transformations applied to the input texts."

re_text_filter.all_default_transformations
sparse_embedded_X = re_text_filter.ReTextFilter().fit_transform(X_train)

sparse_embedded_X = CountVectorizer(

                        min_df=2, 

                        stop_words=stopwords).fit_transform(sparse_embedded_X)

sparse_embedded_X = TfidfTransformer().fit_transform(sparse_embedded_X)
"""

Visualize the embeddings.

Credit due to https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert

"""

lsa = TruncatedSVD(n_components=2)

lsa_scores = lsa.fit_transform(sparse_embedded_X)

colors = ['red', 'blue']

color_labels = [colors[i] for i in y_train]

plt.scatter(lsa_scores[:, 0],lsa_scores[:, 1],

           s=2, alpha=0.8, c=color_labels, cmap=matplotlib.colors.ListedColormap(colors)

           )

orange_patch = mpatches.Patch(color='red', label='Not real')

blue_patch = mpatches.Patch(color='blue', label='Real')

plt.legend(handles=[orange_patch, blue_patch], prop={'size': 10})

plt.gca()

fig = plt.figure(figsize=(4, 5), dpi=300)  

plt.show()
tree_clf = Pipeline([

    ('filter', re_text_filter.ReTextFilter()),

    ('vect', CountVectorizer(min_df=3, stop_words=stopwords)),

    ('tfidf', TfidfTransformer(use_idf=False)),

    ('clf', DecisionTreeClassifier(max_depth=30))

])

tree_clf.fit(X_train, y_train)

predictions = tree_clf.predict(X_test)

print(classification_report(y_test, predictions))
plt.figure(figsize=(20,20))

plt.suptitle("Decision surface of a decision tree")

plt.legend(loc='lower right', borderpad=0, handletextpad=0)

plt.axis("tight")



plot_tree(tree_clf['clf'], 

          impurity=False,

          proportion=False,

          label='root',

          filled=True, 

          fontsize=12, 

          rounded=True,

          feature_names=tree_clf['vect'].get_feature_names())



fig = plt.gcf()



plt.show()
"It's a bit of a mess, but we can start so see how the tree makes its predictions."

tree_clf.predict(['suicide california hiroshima japan migrants'])
gb_rf_clf = Pipeline([

    ('filter', re_text_filter.ReTextFilter()),

    ('vect', TfidfVectorizer(min_df=2, stop_words=stopwords, use_idf=False)),

    ('clf', GradientBoostingClassifier()

    )

])

begin_time = time.time()

gb_rf_clf.fit(X_train, y_train)

train_time_seconds = time.time() - begin_time

print(f'Training time took {train_time_seconds}')



predictions = gb_rf_clf.predict(X_test)

print(classification_report(y_test, predictions))
rndf_clf = Pipeline([

    ('filter', re_text_filter.ReTextFilter()),

    ('vect', TfidfVectorizer(

        use_idf=False,

        min_df=2, 

        stop_words=stopwords)),

    ('clf', RandomForestClassifier())

])

begin_time = time.time()

rndf_clf.fit(X_train, y_train)

train_time_seconds = time.time() - begin_time

print(f'Training time took {train_time_seconds}')



predictions = rndf_clf.predict(X_test)

print(classification_report(y_test, predictions))
ada_rf_clf = Pipeline([

    ('filter', re_text_filter.ReTextFilter()),

    ('vect', TfidfVectorizer(min_df=2, stop_words=stopwords)),

    ('clf', AdaBoostClassifier(

                RandomForestClassifier(max_depth=20), 

                n_estimators=100, 

                algorithm="SAMME.R", 

            learning_rate=0.5

        )

    )

])

begin_time = time.time()

ada_rf_clf.fit(X_train, y_train)

train_time_seconds = time.time() - begin_time

print(f'Training time took {train_time_seconds}')



predictions = ada_rf_clf.predict(X_test)

print(classification_report(y_test, predictions))
xgb_clf = Pipeline([

    ('filter', re_text_filter.ReTextFilter()),

    ('vect', TfidfVectorizer(min_df=2, stop_words=stopwords, use_idf=False)),

    ('clf', xgboost.XGBClassifier())

])



xgb_clf.fit(X_train, y_train)

predictions = xgb_clf.predict(X_test)

print(classification_report(y_test, predictions))
svm_clf = Pipeline([

    ('filter', re_text_filter.ReTextFilter()),

    ('vect', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('clf', SVC())

])

begin_time = time.time()

svm_clf = svm_clf.fit(X_train, y_train)

train_time_seconds = time.time() - begin_time

print(f'Training time took {train_time_seconds}')

predictions = svm_clf.predict(X_test)

print(classification_report(y_test, predictions))
glv_vectorizer_100d = vectorization.GloVePaddedVectorizer(

                        embeddings_file='../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt',

                        dimensions=100,

                        pad_len = 50

)

X_train_filtered = re_text_filter.ReTextFilter().fit_transform(X_train)

X_test_filtered = re_text_filter.ReTextFilter().fit_transform(X_test)

glv_vectorizer_100d.fit(X_train_filtered)

glv_vectorizer_100d.transform(["What does this text look like in vectorspace?"])
glv_vectorizer_avg = vectorization.GloVeAverageVectorizer(

                        embeddings_file='../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt')

glv_vectorizer_avg.fit(X_train)

glv_embedded_X = glv_vectorizer_avg.transform(X_train)



lsa = TruncatedSVD(n_components=2)

lsa_scores = lsa.fit_transform(glv_embedded_X)

colors = ['red', 'blue']

color_labels = [colors[i] for i in y_train]

plt.scatter(lsa_scores[:, 0],lsa_scores[:, 1],

           s=2, alpha=0.8, c=color_labels, cmap=matplotlib.colors.ListedColormap(colors)

           )

orange_patch = mpatches.Patch(color='red', label='Not real')

blue_patch = mpatches.Patch(color='blue', label='Real')

plt.legend(handles=[orange_patch, blue_patch], prop={'size': 10})

plt.gca()

fig = plt.figure(figsize=(3, 4), dpi=300)  

plt.show()
X_train_glove_vect = glv_vectorizer_100d.transform(X_train_filtered)

X_train_glove_vect = np.expand_dims(X_train_glove_vect, axis=3)

X_test_glove_vect = glv_vectorizer_100d.transform(X_test_filtered)

X_test_glove_vect = np.expand_dims(X_test_glove_vect, axis=3)

X_train_glove_vect.shape

y_train_binary = tf.keras.utils.to_categorical(y_train)

y_test_binary = tf.keras.utils.to_categorical(y_test)



vector_shape = glv_vectorizer_100d.transform(["Sample input sequence"]).shape

input_shape = np.array([vector_shape[1], vector_shape[2]])

input_shape_x, input_shape_y = input_shape



conv_nn_model_100d = tf.keras.models.Sequential()

conv_nn_model_100d.add(tf.keras.layers.Conv2D(

        100, kernel_size=(15, 100), 

        activation='elu', 

        input_shape=(*input_shape, 1),

         padding='same'

    )

)

conv_nn_model_100d.add(tf.keras.layers.BatchNormalization())

conv_nn_model_100d.add(tf.keras.layers.Conv2D(

        50, kernel_size=(10, 100), 

        activation='elu', 

        input_shape=(*input_shape, 1),

         padding='same'

    )

)

conv_nn_model_100d.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

conv_nn_model_100d.add(tf.keras.layers.Dropout(0.2))

conv_nn_model_100d.add(tf.keras.layers.Conv2D(

        50, 

        kernel_size=(10, 75), 

        activation='elu', 

        input_shape=(*input_shape / 2, 1),

        padding='same'

    )

)

conv_nn_model_100d.add(tf.keras.layers.Dropout(0.2))

conv_nn_model_100d.add(tf.keras.layers.Conv2D(

        25, 

        kernel_size=(10, 50), 

        activation='elu', 

        input_shape=(*input_shape / 2, 1),

        padding='same'

    )

)

conv_nn_model_100d.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

conv_nn_model_100d.add(tf.keras.layers.Dropout(0.2))

conv_nn_model_100d.add(tf.keras.layers.Conv2D(

        1, 

        kernel_size=(10, 25), 

        activation='elu', 

        input_shape=(*input_shape / 2, 1),

        padding='same'

    )

)

conv_nn_model_100d.add(tf.keras.layers.BatchNormalization())

conv_nn_model_100d.add(tf.keras.layers.MaxPooling2D(pool_size=(5,5)))

conv_nn_model_100d.add(tf.keras.layers.Flatten())

conv_nn_model_100d.add(tf.keras.layers.Dense(input_shape_x*input_shape_y, activation='elu'))

conv_nn_model_100d.add(tf.keras.layers.Dropout(0.1))

conv_nn_model_100d.add(tf.keras.layers.Dense(2, activation='softmax'))

conv_nn_model_100d.compile(loss=tf.keras.losses.categorical_crossentropy,

              optimizer=tf.keras.optimizers.Adam(

                  learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False),

              metrics=['accuracy'])



conv_nn_model_100d.fit(

    X_train_glove_vect, y_train_binary,

    batch_size=32,

    epochs=7,

    validation_data=(X_test_glove_vect, y_test_binary)

)

score = conv_nn_model_100d.evaluate(X_test_glove_vect, y_test_binary, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
"This wrapper class is to use Keras models in a way that plays nice with the voting classifier class defined below."

class TrainedKerasModelWrapper:

    def __init__(self, filterer, vectorizer, model):

        self.filterer = filterer

        self.vectorizer = vectorizer

        self.model = model

        

    def fit(self, X, y, batch_size, epochs):

        X = self.filterer.transform(X)

        X = self.vectorizer.transform(X)

        X = np.expand_dims(X, axis=3)

        y = tf.keras.utils.to_categorical(y)

        self.model.fit(X, y, batch_size=batch_size, epochs=epochs)

        

    def predict(self, X):

        X = self.filterer.transform(X)

        X = self.vectorizer.transform(X)

        X = np.expand_dims(X, axis=3)

        return self.model.predict_classes(X)
convnet_clf_100d = TrainedKerasModelWrapper(

    re_text_filter.ReTextFilter(),

    glv_vectorizer_100d,

    conv_nn_model_100d

)
"""

Source:

https://stackoverflow.com/questions/42920148/using-sklearn-voting-ensemble-with-partial-fit

<3

"""

class VotingClassifier(object):

    """ Implements a voting classifier for pre-trained classifiers"""

    def __init__(self, estimators):

        self.estimators = estimators



    def predict(self, X):

        Y = np.zeros([X.shape[0], len(self.estimators)], dtype=int)

        

        for i, clf in enumerate(self.estimators):

            Y[:, i] = clf.predict(X)

        # apply voting 

        y = np.zeros(X.shape[0])

        for i in range(X.shape[0]):

            y[i] = np.argmax(np.bincount(Y[i,:]))

        return y.astype(int)
estimators = [svm_clf, 

              convnet_clf_100d,

              rndf_clf,

              ada_rf_clf]

voting_clf = VotingClassifier(estimators)

predictions = voting_clf.predict(X_test)

print(classification_report(y_test, predictions))
ada_rf_clf.fit(X, y)

svm_clf.fit(X, y)

xgb_clf.fit(X, y) 

rndf_clf.fit(X, y)

convnet_clf_100d.fit(X, y, batch_size=32, epochs=7)
df_test = pd.read_csv("../input/nlp-getting-started/test.csv")

df_test.head()
X_test = df_test['text']

predictions_test = VotingClassifier(estimators).predict(X_test)
submission = pd.DataFrame({'id': df_test['id'], 'target': predictions_test})

submission.to_csv('submission.csv', index=False, header=True)

submission.sample(30)
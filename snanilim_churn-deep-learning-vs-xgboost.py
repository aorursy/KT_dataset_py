import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

dataset = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')

dataset.head()
plt.figure(figsize=(20,20))

churn_corr = dataset.corr()

churn_corr_top = churn_corr.index

sns.heatmap(dataset[churn_corr_top].corr(), annot=True)
X = dataset.iloc[:, 3:13]

y = dataset.iloc[:, 13]
X.head()
geography = pd.get_dummies(dataset['Geography'], drop_first=True)

gender = pd.get_dummies(dataset['Gender'], drop_first=True)

gender.head()
X = pd.concat([X, geography, gender], axis=1)
X = X.drop(['Geography', 'Gender'], axis=1)
X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
SC = StandardScaler()

X_train = SC.fit_transform(X_train)

X_test = SC.transform(X_test)
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LeakyReLU, PReLU, ELU

from keras.layers import Dropout
classifier = Sequential()
# Adding the input layer and the first hidden layer

classifier.add(Dense(6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 11))

classifier.add(Dropout(0.1))
# Adding the second hidden layer

classifier.add(Dense(6, kernel_initializer = 'he_uniform',activation='relu'))

classifier.add(Dropout(0.1))
# Adding the output layer

classifier.add(Dense(1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
# Compiling the ANN

classifier.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['accuracy'])
model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size = 10, epochs = 100)
print(model_history.history.keys())
# summarize history for accuracy

plt.figure(figsize=(10, 7))

plt.plot(model_history.history['accuracy'])

plt.plot(model_history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for loss

plt.figure(figsize=(10, 7))

plt.plot(model_history.history['loss'])

plt.plot(model_history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

y_pred
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
# Calculate the Accuracy

from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, y_pred)
score
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout

from keras.activations import relu, sigmoid
def create_model(layers, activation):

    model = Sequential()

    for i, nodes in enumerate(layers):

        if i == 0:

            model.add(Dense(nodes, kernel_initializer = 'he_uniform',activation=activation,input_dim = X_train.shape[1]))

            model.add(Dropout(0.1))

        else:

            model.add(Dense(nodes, kernel_initializer = 'he_uniform',activation=activation))

            model.add(Dropout(0.1))



    # Adding the output layer

    model.add(Dense(1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

                           

    # Compiling the ANN

    model.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['accuracy'])

    return model
model = KerasClassifier(build_fn=create_model, verbose=0)

layers = [(6, 3, 3), (10, 10), (45, 30, 15)]

activations = ['sigmoid', 'relu']

param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[30])

grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)

grid_result = grid.fit(X_train, y_train)
[grid_result.best_score_,grid_result.best_params_]
X.head()
y.head()
params = {

    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],

    "learning_rate": [0.5, 0.10, 0.15, 0.20, 0.25, 0.30],

    "min_child_weight": [1, 3, 5, 7],

    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],

    "colsample_bytree": [0.3, 0.4, 0.5, 0.7]

}
## Hyperparameter optimization using RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV

import xgboost
xgb_init = xgboost.XGBClassifier()
random_cv = RandomizedSearchCV(xgb_init, param_distributions=params, n_iter=5, scoring="roc_auc", n_jobs=1, cv=5, verbose=3)
random_cv.fit(X, y)
random_cv.best_estimator_
random_cv.best_params_
xgb_classifier = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.4, gamma=0.2, gpu_id=-1,

              importance_type='gain', interaction_constraints='',

              learning_rate=0.3, max_delta_step=0, max_depth=3,

              min_child_weight=1, missing=None, monotone_constraints='()',

              n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

              tree_method='exact', validate_parameters=1, verbosity=None)
from sklearn.model_selection import cross_val_score
predict = cross_val_score(xgb_classifier, X, y, cv=10)

predict
predict = predict.mean()

predict
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import HTML
th_props = [

  ('font-size', '28px'),

  ('text-align', 'center'),

  ('font-weight', 'bold'),

  ('color', '#021755'),

  ('background-color', '#fbe4e5')

  ]



# Set CSS properties for td elements in dataframe

td_props = [

  ('font-size', '25px'),

  ('background-color', '#f7f2ed')

  ]



# Set table styles

styles = [

  dict(selector="th", props=th_props),

  dict(selector="td", props=td_props)

  ]
d = {'ML Technique': ["Deep-Learning", "XGBoost"], 'Score': [score, predict]}

df = pd.DataFrame(data=d)

cm = sns.light_palette("red", as_cmap=True)



(df.style

  .set_caption('Deep-Learning VS XGBoost.')

  .format({'total_amt_usd_pct_diff': "{:.2%}"})

  .set_table_styles(styles))
import numpy as np
import pandas as pd
titanic = pd.read_csv('../input/train.csv')

# some preprocessing is applied for simplicity
titanic = titanic[['Sex', 'Pclass', 'Age', 'SibSp', 'Fare', 'Survived']]  # use subset of columns
titanic = titanic.dropna()  # drop rows with missing values

display(titanic.head())
# use only numerical values
Xy = titanic[['Pclass', 'Age', 'SibSp', 'Fare', 'Survived']].values

# separate inputs and outputs
X = Xy[:, :-1]
y = Xy[:, -1]
# using Kernel SVM is easy in sklearn
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# do the usual splitting into training / testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
for C in [0.01, 0.1, 1, 10, 100]:
    for gamma in [0.01, 0.1, 1.0, 10, 100]:
        score = 0
        # Task: fill in here proper training and scoring of SVC.        
        print(C, gamma, score)
# using Kernel SVM is easy in sklearn
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# do the usual splitting into training / testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# two arguments are necessary: sklearn "estimator" class,
# and range of parameters for this estimator. 
model = GridSearchCV(
    estimator=make_pipeline(StandardScaler(), SVC()),
    param_grid={  # format: lowercase_estimator_name__param_name
        'svc__C': [0.01, 1, 100], 
        'svc__gamma': [0.01, 1, 100],
    },
    cv=5,  # number of validation folds
    n_jobs=1  # number of parallel jobs
)
# Q: how does the grid search scale computationally?

model.fit(X_train, y_train)
print(model.score(X_test, y_test))

eg_inputs = X_test[:5]
print(eg_inputs)
print(model.predict(eg_inputs))
from skopt import BayesSearchCV

# include below until https://github.com/scikit-optimize/scikit-optimize/issues/718 is resolved
class BayesSearchCV(BayesSearchCV):
    def _run_search(self, x): raise BaseException('Use newer skopt')
model = BayesSearchCV(
    estimator=make_pipeline(StandardScaler(), SVC()),
    search_spaces={
        'svc__C': (0.01, 100.0, 'log-uniform'),  # specify ranges instead of discrete values
        'svc__gamma': (0.01, 100.0, 'log-uniform'),
    },
    cv=5,
    n_iter=16,  # fixed number of parameter configuration trials!
    n_jobs=4,  # it runs evaluations in parallel too!
    verbose=1
)

model.fit(X_train, y_train)
print(model.best_params_)
print(model.score(X_test, y_test))
from keras.layers import Input, Dense, LeakyReLU, Softmax
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

# define single input
inp = Input(shape=(4,))
h = inp

# Task: add a layer with 1 neuron
h = Dense(64)(h)  # linear transformation
h = LeakyReLU()(h)  # activation
h = Dense(2)(h)  # final linear layer
h = Softmax()(h)  # softmax activation

# create an ANN model definition
model = Model(inputs=[inp], outputs=[h])

# Task: set learning rate to 100.0, 0.0000001
# this creates a C program that is called from python
model.compile(Adam(), sparse_categorical_crossentropy, ['accuracy'])

# fit the model
model.fit(X_train, y_train, epochs=10)

# evaluate the model
loss, score = model.evaluate(X_test, y_test)
print(score)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def make_net(n_neurons=128):
    """Defines architecture of ANN and compiles it."""
    inp = Input(shape=(4,))
    h = Dense(n_neurons)(inp)  # linear transformation
    h = LeakyReLU()(h)  # activation
    h = Dense(2)(h)  # final linear layer
    h = Softmax()(h)  # softmax activation

    model = Model(inputs=[inp], outputs=[h])
    model.compile(Adam(), sparse_categorical_crossentropy, ['accuracy'])
    return model    

# can be used as part of pipeline, and in *SearchCV
# Task: wrap in pipeline, and add scaling of feature ranges
model = KerasClassifier(make_net, n_neurons=256)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)
# Obtain example outputs. Remember, columns are:
# Pclass, Age, SibSp, Fare
# Task: what leads to increase of survival likelihood?
my_input= np.array([
    [3, 22.0, 1, 7.2500],
    [3, 20.0, 0, 10.0],
])
print(model.predict_proba(my_input))
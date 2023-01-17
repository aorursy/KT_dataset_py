# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import keras
url = "../input/pima-indians-diabetes-database/diabetes.csv"

df=pd.read_csv(url)

df
#no missing data

df.describe().transpose()
#duplicate data should be removed to avoid overfitting

print(f"Number of duplicates in the training data are {df.duplicated().sum()} of {len(df)}, ie {(100* df.duplicated().sum()/len(df)).round(2)} % of data duplicated")

df.drop_duplicates(inplace=True)
#the dataset is not unbalanced

positive=df[df['Outcome']==1]

negative=df[df['Outcome']==0]

print(positive.shape)

print(negative.shape)

#plotting the Outcome

sns.distplot(df['Outcome'], bins=10);
#bloodpressure, bmi, glucose, insulin, skinthickness

df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);
df_copy = df.copy(deep = True)

df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace = True)

df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace = True)

df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace = True)

df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace = True)

df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace = True)
df_copy.describe().transpose()
#the data looks better defined now

df_copy.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);
df_num_corr = df_copy.corr()['Outcome'][:-1]

df_num_corr
for i in range(0, len(df_copy.columns), 5):

    sns.pairplot(data=df_copy,

                x_vars=df_copy.columns[i:i+5],

                y_vars=['Outcome'])
X=df_copy.iloc[:,:-1]

y=df_copy['Outcome']

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X= sc.fit_transform(X)
from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from pandas.plotting import scatter_matrix

from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import GradientBoostingClassifier



#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = df.Outcome, random_state=0)



models = []

models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))

models.append(('SVM', SVC(gamma='auto')))

models.append(('GNB', GaussianNB()))

models.append(('LR', LogisticRegression(solver='liblinear',random_state=0)))

models.append(('decisiontree', tree.DecisionTreeClassifier()))

models.append(('randomforest', RandomForestClassifier(max_depth=2, random_state=10, n_estimators=10)))

models.append(('GB', GradientBoostingClassifier()))



scores = []

names = []

        

for name, model in models:

    

    score = cross_val_score(model, X, y, scoring='accuracy', cv=5).mean()

    

    names.append(name)

    scores.append(score)

kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})

print(kf_cross_val)



axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)

axis.set(xlabel='Classifier', ylabel='Accuracy')

for p in axis.patches:

    height = p.get_height()

    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 

    

plt.show()





from sklearn.model_selection import GridSearchCV, KFold

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.optimizers import Adam
def create_model():

    # create model

    model = Sequential()

    model.add(Dense(8, input_dim = 8, kernel_initializer='normal', activation='relu'))

    model.add(Dense(4, input_dim = 8, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    

    # compile the model

    adam = Adam(lr = 0.01)

    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

    return model



model = create_model()

print(model.summary())
model = KerasClassifier(build_fn = create_model, verbose = 1)

model.fit(X,y)
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.optimizers import Adam



# Define a random seed

seed = 6

np.random.seed(seed)



# Start defining the model

def create_model():

    # create model

    model = Sequential()

    model.add(Dense(8, input_dim = 8, kernel_initializer='normal', activation='relu'))

    model.add(Dense(4, input_dim = 8, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    

    # compile the model

    adam = Adam(lr = 0.01)

    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

    return model



# create the model

model = KerasClassifier(build_fn = create_model, verbose = 1)



# define the grid search parameters

batch_size = [10,15, 20]

epochs = [10, 50, 100]



# make a dictionary of the grid search parameters

param_grid = dict(batch_size=batch_size, epochs=epochs)



# build and fit the GridSearchCV

grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), verbose = 10)

grid_results = grid.fit(X, y)



# summarize the results

print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))

means = grid_results.cv_results_['mean_test_score']

stds = grid_results.cv_results_['std_test_score']

params = grid_results.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print('{0} ({1}) with: {2}'.format(mean, stdev, param))
from keras.layers import Dropout



# Define a random seed

seed = 6

np.random.seed(seed)



# Start defining the model

def create_model(learn_rate, dropout_rate):

    # create model

    model = Sequential()

    model.add(Dense(8, input_dim = 8, kernel_initializer='normal', activation='relu'))

    model.add(Dropout(dropout_rate))

    model.add(Dense(4, input_dim = 8, kernel_initializer='normal', activation='relu'))

    model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    

    # compile the model

    adam = Adam(lr = learn_rate)

    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

    return model



# create the model

model = KerasClassifier(build_fn = create_model, epochs = 10, batch_size = 10, verbose = 0)



# define the grid search parameters

learn_rate = [0.001, 0.01, 0.1]

dropout_rate = [0.0, 0.1, 0.2]



# make a dictionary of the grid search parameters

param_grid = dict(learn_rate=learn_rate, dropout_rate=dropout_rate)



# build and fit the GridSearchCV

grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), verbose = 10)

grid_results = grid.fit(X, y)



# summarize the results

print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))

means = grid_results.cv_results_['mean_test_score']

stds = grid_results.cv_results_['std_test_score']

params = grid_results.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print('{0} ({1}) with: {2}'.format(mean, stdev, param))
seed = 6

np.random.seed(seed)



# Start defining the model

def create_model(activation, init):

    # create model

    model = Sequential()

    model.add(Dense(8, input_dim = 8, kernel_initializer= init, activation= activation))

    model.add(Dense(4, input_dim = 8, kernel_initializer= init, activation= activation))

    model.add(Dense(1, activation='sigmoid'))

    

    # compile the model

    adam = Adam(lr = 0.001)

    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

    return model



# create the model

model = KerasClassifier(build_fn = create_model, epochs = 10, batch_size = 10, verbose = 0)



# define the grid search parameters

activation = ['softmax', 'relu', 'tanh', 'linear']

init = ['uniform', 'normal', 'zero']



# make a dictionary of the grid search parameters

param_grid = dict(activation = activation, init = init)



# build and fit the GridSearchCV

grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), verbose = 10)

grid_results = grid.fit(X, y)



# summarize the results

print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))

means = grid_results.cv_results_['mean_test_score']

stds = grid_results.cv_results_['std_test_score']

params = grid_results.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


seed = 6

np.random.seed(seed)

from keras.layers import Dropout

# Start defining the model

def create_model(neuron1, neuron2):

    # create model

    model = Sequential()

    model.add(Dense(neuron1, input_dim = 8, kernel_initializer= 'normal', activation= 'relu'))

    model.add(Dense(neuron2, input_dim = neuron1, kernel_initializer= 'normal', activation= 'linear'))

    model.add(Dense(1, activation='sigmoid'))

    

    # compile the model

    adam = Adam(lr = 0.001)

    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

    return model



# create the model

model = KerasClassifier(build_fn = create_model, epochs = 10, batch_size = 10, verbose = 0)



# define the grid search parameters

neuron1 = [4, 8, 16]

neuron2 = [2, 4, 8]



# make a dictionary of the grid search parameters

param_grid = dict(neuron1 = neuron1, neuron2 = neuron2)



# build and fit the GridSearchCV

grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), refit = True, verbose = 10)

grid_results = grid.fit(X, y)



# summarize the results

print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))

means = grid_results.cv_results_['mean_test_score']

stds = grid_results.cv_results_['std_test_score']

params = grid_results.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print('{0} ({1}) with: {2}'.format(mean, stdev, param))
y_pred = grid.predict(X)

from sklearn.metrics import classification_report, accuracy_score



print(accuracy_score(y, y_pred))

print(classification_report(y, y_pred))
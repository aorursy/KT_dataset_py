import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import tensorflow.keras

from keras.models import Sequential

from keras.layers import Dense, Dropout



from sklearn.tree import DecisionTreeRegressor



from sklearn.neighbors import KNeighborsRegressor



from sklearn.linear_model import Lasso



from sklearn.linear_model import Ridge



from keras.callbacks import EarlyStopping

from keras.layers.advanced_activations import LeakyReLU



from sklearn.model_selection import KFold



from sklearn.model_selection import train_test_split



from sklearn.model_selection import cross_val_score 

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import RepeatedKFold
data =  pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv', error_bad_lines = False, encoding='latin-1')

#saving target value into different dataframe

answer = data.drop(['fbs','restecg','exang','oldpeak', 'ca', 'thal', 'slope', 'age', 'sex', 'cp', 'trestbps', 'chol', 'thalach'], axis=1)

#dropping extraneous data

data = data.drop(['fbs','restecg','exang','oldpeak', 'ca', 'thal', 'slope', 'target'], axis=1)

#Renaming columns for clarity

data.columns = ['Age', 'Gender','Chest Pain','Resting Blood Pressure', 'Cholesterol', 'Max HR']
data.head()
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor



def buildmodel():

    model= Sequential([

        Dense(256, input_dim=6),

        LeakyReLU(alpha=0.3),

        Dropout(0.1),

        Dense(256),

        LeakyReLU(alpha=0.3),

        Dropout(0.1),

        Dense(256),

        LeakyReLU(alpha=0.3),

        Dropout(0.1),

        Dense(256),

        LeakyReLU(alpha=0.3),

        Dropout(0.1),

        Dense(1)

    ])

    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])

    return(model)



estimator= KerasRegressor(build_fn=buildmodel, epochs=100, batch_size=2000, verbose=0)

kfold= RepeatedKFold(n_splits=5, n_repeats=100)

results = cross_val_score(estimator, data, answer, cv=kfold)

score0 = abs(results.mean())
regr_1 = DecisionTreeRegressor(max_depth=3)
regr_1.fit(data, answer)
score1 = abs(cross_val_score(estimator = regr_1, X = data, y = answer, cv=kfold).mean())



print(score1)
regr_2 = KNeighborsRegressor(n_neighbors=3,weights='distance', algorithm='brute')
regr_2.fit(data, answer)
score2 = abs(cross_val_score(estimator = regr_2, X = data, y = answer, cv=kfold).mean())



print(score2)
regr_3 = Lasso(alpha=0.001)
regr_3.fit(data, answer)
score3 = abs(cross_val_score(estimator = regr_3, X = data, y = answer, cv=kfold).mean())



print(score3)
regr_4 = Ridge(alpha=0.001)
regr_4.fit(data, answer)
score4 = abs(cross_val_score(estimator = regr_4, X = data, y = answer, cv=kfold).mean())



print(score4)
plt.style.use('ggplot')



x = ['ANN', 'Regression Tree', 'KNN', 'Lasso', 'Ridge']

energy = [score0, score1, score2, score3, score4]



x_pos = [i for i, _ in enumerate(x)]



plt.bar(x_pos, energy)

plt.xlabel("Model")

plt.ylabel("Loss")

plt.title("K Fold Cross Validation Scores")



plt.xticks(x_pos, x)



plt.show()
import pandas as pd

import numpy as np



df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

df.head()
print ('Number of Rows :', df.shape[0])

print ('Number of Columns :', df.shape[1])

print ('Number of Patients with outcome 1 :', df.Outcome.sum())

print ('Event Rate :', round(df.Outcome.mean()*100,2) ,'%')
df.describe()
from sklearn.preprocessing import normalize



X = df.to_numpy()[:,0:8] 

Y = df.to_numpy()[:,8]



X_norm = normalize(X)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier



SEED = 42

np.random.seed(SEED)



def create_model(optimizer = 'adam'):

    

    model = Sequential()

    model.add(Dense(12, activation = 'relu', input_dim=(8)))

    model.add(Dense( 8, activation = 'relu'))

    model.add(Dense( 1, activation = 'sigmoid'))

    

    model.compile(loss = 'binary_crossentropy', optimizer= optimizer, metrics = ['accuracy'])

    

    return model



model = KerasClassifier(build_fn = create_model, epochs = 150, batch_size = 8, verbose = 0)
from sklearn.model_selection import StratifiedKFold, cross_val_score

kfold = StratifiedKFold(n_splits = 8, shuffle = True, random_state = SEED)



%time results = cross_val_score(model, X_norm, Y, cv=kfold)

print ('Accuracy',round(results.mean()*100,2), '%')
param_grid = {

    'optimizer'  : ['rmsprop','adam','sgd'],

    'epochs'     : [100, 150, 200],

    'batch_size' : [8, 16, 32],

}
%%time 

from sklearn.model_selection import GridSearchCV



model = KerasClassifier(build_fn = create_model, verbose = 0)

grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3)

grid_result = grid.fit(X_norm, Y)



print (f'With {grid_result.best_params_} got {round(grid_result.best_score_*100,2)} as best score!!')
print ('---- GRID SEARCH RESULTS ----')

for p,s in zip(grid_result.cv_results_['params'],grid_result.cv_results_['mean_test_score']):

    print (f' Accuracy : {round(s*100,2)} % | Param : {p}')
import pandas as pd

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
df = pd.read_csv('../input/entradas-breast.csv')
target = pd.read_csv('../input/saidas-breast.csv')
def RN(optimizer,loss,kernel_initializer,activation,neurons):

    seq = Sequential()

    seq.add(Dense(units=neurons, activation=activation,kernel_initializer=kernel_initializer,input_dim=30))

    seq.add(Dropout(0.1))

    seq.add(Dense(units=neurons, activation=activation,kernel_initializer=kernel_initializer))

    seq.add(Dense(units=1,activation='sigmoid'))

    seq.compile(optimizer=optimizer,loss=loss,metrics=['binary_accuracy'])

    return seq
seq = KerasClassifier(build_fn=RN)
param = {'batch_size':[10],'epochs':[20],'optimizer':['adam'],'loss':['binary_crossentropy','hinge'],'kernel_initializer':['random_uniform','normal'],'activation':['relu'],'neurons':[16]}
grid_search = GridSearchCV(estimator=seq,param_grid=param,scoring='f1',cv=5)
grid_search = grid_search.fit(df,target)
best_param = grid_search.best_params_
best_score = grid_search.best_score_
best_param
best_score
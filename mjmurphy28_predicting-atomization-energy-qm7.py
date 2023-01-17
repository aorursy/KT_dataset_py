import pandas as pd

import scipy.io

import numpy as np

from scipy.spatial.distance import pdist, squareform

import networkx as nx

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import tensorflow as tf

import time

rand_state = 42

tf.set_random_seed(rand_state)

np.random.seed(rand_state)



import warnings

warnings.simplefilter('ignore')
qm7 = scipy.io.loadmat('../input/qm7.mat')

# compute the Eigenvectors of the pairwise distance matrix?

R = qm7['R']



y = np.transpose(qm7['T']).reshape((7165,))

y_scaling_factor = 2000.

y_scaled = y / y_scaling_factor



# k=0 # 0 = include diagnol, 1 = do not include diagnol



num_atoms = 23

iu = np.triu_indices(num_atoms,k=0) 

iu_dist = np.triu_indices(num_atoms,k=1) # for the pairwise distance matrix, all diagonol entries will be 0 





CM = np.zeros((qm7['X'].shape[0], num_atoms*(num_atoms+1)//2), dtype=float)

eigs = np.zeros((qm7['X'].shape[0], num_atoms), dtype=float)

centralities = np.zeros((qm7['X'].shape[0], num_atoms), dtype=float)

interatomic_dist = np.zeros((qm7['X'].shape[0], ((num_atoms*num_atoms)-num_atoms)//2), dtype=float) 



verbose=True



for i, cm in enumerate(qm7['X']):

    coulomb_vector = cm[iu]

    # Sort elements by decreasing order

    shuffle = np.argsort(-coulomb_vector)

    CM[i] = coulomb_vector[shuffle]

    dist = squareform(pdist(R[i]))

    # we can extract the upper triangle of the distance matri: return vector of dimension (1,num_atoms)

    dist_vector = dist[iu_dist]

    shuffle = np.argsort(-dist_vector)

    interatomic_dist[i] = dist_vector[shuffle]

    

    w,v = np.linalg.eig((dist))

    eigs[i] = w[np.argsort(-w)]

    centralities[i] = np.array(list(nx.eigenvector_centrality(nx.Graph(dist)).values()))

    

    if verbose and i % 500 == 0:

        print("Processed {} molecules".format(i))

    

X = np.concatenate((CM, eigs, centralities, interatomic_dist), axis=1)

X.shape
def mean_dist(x):

    x[x == 0] = np.nan

    return np.nanmean(x, axis=0)







mean_dists = np.apply_along_axis(mean_dist, axis=1, arr=interatomic_dist)
plt.figure(figsize=(8,6))

sns.distplot(mean_dists)

plt.xlabel('Interatomic Distance')

plt.ylabel('Probability')

plt.title('Distribution of interatomic distances')

plt.show()



plt.figure(figsize=(8,6))

sns.distplot(y)

plt.xlabel('Energy (kcal/mol)')

plt.ylabel('Probability')

plt.title('Distribution of Atomization Energy')

plt.show()
from sklearn.decomposition import KernelPCA

from matplotlib import cm



# scale Coulomb Matrices, divide by 370

CM = qm7['X'].reshape((7165, 529))





start_time = time.time()

kpca = KernelPCA(n_components=2, kernel="rbf")

CM_reduced = kpca.fit_transform(CM)

print("--- %s seconds ---" % (time.time() - start_time))

explained_variance = np.var(CM_reduced, axis=0)

explained_variance_ratio = explained_variance / np.sum(explained_variance)

print("Variance Explained: ", np.sum(explained_variance_ratio))



fig = plt.figure(figsize=(14,10))

ax  = fig.add_subplot(111)



scatter = ax.scatter(CM_reduced[:,0], CM_reduced[:,1], c=y, s=60, edgecolors='black', cmap=cm.jet_r)

colorbar = fig.colorbar(scatter, ax=ax, label = "(kcal/mol)")

plt.xlabel(r'$k-PCA_1$')

plt.ylabel(r'$k-PCA_1$')

plt.title("Coulomb Matrix: Kernel PCA (RBF)")

sns.despine()

plt.show()
from sklearn.manifold import TSNE



tsne = TSNE(n_components=2, random_state=rand_state, perplexity=50)

X_tsne = tsne.fit_transform(CM)



fig = plt.figure(figsize=(14,10))

ax  = fig.add_subplot(111)



scatter = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=y, s=45, edgecolors='green', cmap=cm.jet_r, alpha=0.5)

colorbar = fig.colorbar(scatter, ax=ax, label = "E (kcal/mol) ")

plt.xlabel(r'$Z_1$')

plt.ylabel(r'$Z_2$')

plt.title('Coulomb Matrix: t-SNE (perplexity = 50)')

sns.despine()

plt.show()
start_time = time.time()

kpca = KernelPCA(n_components=2, kernel="rbf")

X_reduced = kpca.fit_transform(X)

print("--- %s seconds ---" % (time.time() - start_time))

explained_variance = np.var(X_reduced, axis=0)

explained_variance_ratio = explained_variance / np.sum(explained_variance)

print("Variance Explained: ", np.sum(explained_variance_ratio))



fig = plt.figure(figsize=(14,10))

ax  = fig.add_subplot(111)



scatter = ax.scatter(X_reduced[:,0], X_reduced[:,1], c=y, s=60, edgecolors='black', cmap=cm.jet_r)

colorbar = fig.colorbar(scatter, ax=ax, label = "E (kcal/mol)")

plt.xlabel(r'$k-PCA_1$')

plt.ylabel(r'$k-PCA_1$')

plt.title("Visualizing All Features: Kernel PCA (RBF)")

plt.tight_layout()

sns.despine()

plt.show()
tsne = TSNE(n_components=2, random_state=rand_state, perplexity=50)

X_tsne = tsne.fit_transform(X)



fig = plt.figure(figsize=(14,10))

ax  = fig.add_subplot(111)



scatter = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=y, s=45, edgecolors='green', cmap=cm.jet_r, alpha=0.5)

colorbar = fig.colorbar(scatter, ax=ax, label = "E (kcal/mol) ")

plt.xlabel(r'$Z_1$')

plt.ylabel(r'$Z_2$')

plt.title('All Features: t-SNE (perplexity = 50)')

sns.despine()

plt.show()
def get_category(x, total_range, num_bins):

    bin_size = total_range/num_bins

    total_range = 1787.119995

    bin_size = total_range/num_bins



    return int(np.floor(np.abs(x/bin_size)))



total_range = 1787.119995

num_bins = 10



y_class = pd.Series(y).apply(lambda x: get_category(x, total_range, num_bins))
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score



X_2, X_val, y_class_2, y_class_val = train_test_split(X, y_class.values, 

                                                    test_size=0.15, 

                                                    random_state=rand_state)



# with cross validation no need to further split the data. if not using cross validation you should do one...

X_train, X_dev, y_class_train, y_class_dev = train_test_split(X_2, y_class_2, 

                                                    test_size=0.18, 

                                                    random_state=rand_state)



clf = SVC(kernel='linear', random_state=rand_state)

clf.fit(X_train, y_class_train)

print("Train score: ", clf.score(X_train, y_class_train))

print("Test score: ", clf.score(X_dev, y_class_dev))



print('--------\nEVALUATE on validation set\n--------')

class_preds = clf.predict(X_val)

print("Validation score: ", accuracy_score(y_class_val, class_preds))

print(classification_report(y_class_val, class_preds))
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



X = np.concatenate((CM, eigs, centralities), axis=1)





# with cross validation no need to further split the data. if not using cross validation you should do one...

X_2, X_val, y_2, y_val = train_test_split(X, y, 

                                          test_size=0.15, 

                                          random_state=rand_state)



# with cross validation no need to further split the data. if not using cross validation you should do one...

X_train, X_dev, y_train, y_dev = train_test_split(X_2, y_2, 

                                                  test_size=0.18, 

                                                  random_state=rand_state)





n_folds = 5

early_stopping = 50





start_time = time.time()

xg_train = xgb.DMatrix(X_train, label=y_train)



num_iters = 300



params = {"objective":"reg:linear", 

          'booster': 'gbtree', 

          'eval_metric': 'mae',

          'subsample': 0.9,

          'colsample_bytree':0.2,

          'learning_rate': 0.05,

          'max_depth': 6, 

          'reg_lambda': .9, 

          'reg_alpha': .01,

          'seed': rand_state}







cv = xgb.cv(params,

            xg_train, 

            num_boost_round=num_iters, 

            nfold=n_folds, 

            early_stopping_rounds=early_stopping, 

            verbose_eval = 0, 

            seed=rand_state,

            as_pandas=False)



print("--- %s seconds ---" % (time.time() - start_time))



plt.figure(figsize=(8,8))

plt.plot(cv['train-mae-mean'][100:], label='Train loss: ' + str(np.min(cv['train-mae-mean'])))

plt.plot(cv['test-mae-mean'][100:], label='Test loss: ' + str(np.min(cv['test-mae-mean'])))

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Mean absolute error')

plt.show()



model_xgb = xgb.XGBRegressor(**params, random_state=rand_state, n_estimators=num_iters)

model_xgb.fit(X_train, y_train, 

              early_stopping_rounds=early_stopping, 

              eval_metric='mae', 

              eval_set=[(X_dev, y_dev)], 

              verbose=False)



y_dev_pred = model_xgb.predict(X_dev)

print('Dev mean absoulte error: ', mean_absolute_error(y_dev, y_dev_pred))



y_val_pred = model_xgb.predict(X_val)

print('Validation mean absoulte error: ', mean_absolute_error(y_val, y_val_pred))
from yellowbrick.regressor import PredictionError



# Instantiate the linear model and visualizer

visualizer = PredictionError(xgb.XGBRegressor(**params, n_estimators=num_iters, random_state=rand_state))



visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_val, y_val)  # Evaluate the model on the test data

g = visualizer.poof()       
from yellowbrick.regressor import ResidualsPlot



# Instantiate the linear model and visualizer

visualizer = ResidualsPlot(xgb.XGBRegressor(**params, n_estimators=num_iters, random_state=rand_state))



visualizer.fit(X_train, y_train)  # Fit the training data to the model

visualizer.score(X_val, y_val)  # Evaluate the model on the test data

visualizer.poof()                 # Draw/show/poof the data
CM_scaled = (qm7['X'] / 370.0).reshape((7165, 529, 1))



# now pull out 10% of the data for validation

x_2, x_val, y_2, y_val = train_test_split(CM_scaled, y_scaled,  

                                                    test_size=.15, 

                                                    random_state=rand_state)



# the remaining 90% of the data will be used to build/test our model

x_train, x_dev, y_train, y_dev = train_test_split(x_2, y_2,  

                                                    test_size=.18, 

                                                    random_state=rand_state)
from keras.models import Sequential

from keras.layers import Dense, Conv1D, Flatten

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras import optimizers

import math



model_1 = Sequential()

# our input is now a (276, 1) 1D Tensor

kernel_initializer='he_normal'



model_1.add(Conv1D(32, 10, activation='relu', kernel_initializer=kernel_initializer))

model_1.add(Conv1D(128, 7, activation='relu', kernel_initializer=kernel_initializer))

model_1.add(Conv1D(128, 5, activation='relu', kernel_initializer=kernel_initializer))

model_1.add(Conv1D(256, 3, activation='relu', kernel_initializer=kernel_initializer))

model_1.add(Flatten())

model_1.add(Dense(1, activation='linear'))



model_1.compile(loss='mae',

              optimizer=optimizers.Adam(.001),

              metrics=['mae'])







start = time.time()



estop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

mcp_save = ModelCheckpoint('cnn_1d.hdf5', save_best_only=True, monitor='val_loss', mode='min')

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, epsilon=1e-4, mode='min')



history_1 = model_1.fit(x_train, y_train,

                      batch_size=4,

                      epochs=10, 

                      callbacks=[estop, mcp_save, reduce_lr_loss],

                      verbose=1,

                      validation_data=(x_dev, y_dev))



model_1.summary()



end = time.time()

print('Execution time: ', end-start)

print("Epochs: ", len(history_1.history['val_loss']))

print('Train loss: ', y_scaling_factor*np.min(history_1.history['loss']))

print('Test loss: ', y_scaling_factor*np.min(history_1.history['val_loss']))

y_preds = model_1.predict(x_val)

print("Validation error: ", y_scaling_factor*mean_absolute_error(y_val, y_preds))
plt.figure(figsize=(8,8))

train_label = "Train loss: {}".format(np.round(np.min(history_1.history['loss'])*y_scaling_factor, 3))

test_label = "Test loss: {}".format(np.round(np.min(history_1.history['val_loss'])*y_scaling_factor, 3))

(pd.Series(history_1.history['loss']).ewm(alpha=.1).mean()*y_scaling_factor).plot(label=train_label)

(pd.Series(history_1.history['val_loss']).ewm(alpha=.1).mean()*y_scaling_factor).plot(label=test_label)

plt.legend()

plt.title('MAE Loss for 1D CNN')

plt.xlabel('Epoch')

plt.ylabel('E (kcal/mol)')

plt.show()
from keras.layers import Conv2D, MaxPooling2D, Dropout





'''

convnets require specific input dimensions, in this case every example should be a (23, 23, 1) Coulom Matrix



'''

CM_scaled = qm7['X'].reshape((7165, 23, 23, 1)) / 370.0



X_cm_train, X_cm_test, y_train, y_test  = train_test_split(CM_scaled, y_scaled, 

                                                           test_size=.2, random_state=rand_state)



kernel_initializer='he_normal'

model_2 = Sequential()

model_2.add(Conv2D(32, kernel_size=(7,7), padding='same',

                 activation='relu',

                 kernel_initializer=kernel_initializer,

                 input_shape=(23, 23, 1)))

model_2.add(Conv2D(64, kernel_size=(5,5), kernel_initializer=kernel_initializer, activation='relu'))

model_2.add(Conv2D(64, kernel_size=(3,3), kernel_initializer=kernel_initializer, activation='relu'))

model_2.add(Conv2D(64, kernel_size=(3,3), kernel_initializer=kernel_initializer, activation='relu'))

model_2.add(Conv2D(128, kernel_size=(3,3), kernel_initializer=kernel_initializer, activation='relu'))

model_2.add(Conv2D(128, kernel_size=(3,3), kernel_initializer=kernel_initializer, activation='relu'))

model_2.add(Flatten())

model_2.add(Dense(1, activation='linear'))



model_2.compile(loss='mae',

              optimizer=optimizers.Adam(lr=.001),

              metrics=['mae'])



start = time.time()





estop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

mcp_save = ModelCheckpoint('cnn_2d.hdf5', save_best_only=True, monitor='val_loss', mode='min')

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, epsilon=1e-4, mode='min')



history_2 = model_2.fit(X_cm_train, y_train,

                      batch_size=4,

                      epochs=10, 

                      callbacks=[estop, mcp_save, reduce_lr_loss],

                      verbose=1,

                      validation_data=(X_cm_test, y_test))



end = time.time()



print(model_2.summary())



print('Execution time: ', end-start)

print("Epochs: ", len(history_2.history['val_loss']))

print('Train loss: ', y_scaling_factor*np.min(history_2.history['loss']))

print('Test loss: ', y_scaling_factor*np.min(history_2.history['val_loss']))
plt.figure(figsize=(8,8))

train_label = "Train loss: {}".format(np.round(np.min(history_2.history['loss'])*y_scaling_factor, 3))

test_label = "Test loss: {}".format(np.round(np.min(history_2.history['val_loss'])*y_scaling_factor, 3))

(pd.Series(history_2.history['loss']).ewm(alpha=.1).mean()*y_scaling_factor).plot(label=train_label)

(pd.Series(history_2.history['val_loss']).ewm(alpha=.1).mean()*y_scaling_factor).plot(label=test_label)

plt.legend()

plt.title('MAE Loss for 2D CNN')

plt.xlabel('Epoch')

plt.ylabel('E (kcal/mol)')

plt.show()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from sklearn.model_selection   import train_test_split, StratifiedKFold

from sklearn.ensemble          import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics           import confusion_matrix, accuracy_score, f1_score, roc_curve

from sklearn.utils             import resample

from keras.utils               import to_categorical

from sklearn.cross_validation  import KFold

from keras.models              import Sequential

from keras.layers.core         import Dense, Dropout

from keras.callbacks           import EarlyStopping



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Functions

def find_optimal_cutoff( target, predicted ):

    fpr, tpr, threshold = roc_curve(target, predicted)

    i = np.arange( len( tpr ) )

    roc=pd.DataFrame({'tf':pd.Series(tpr-(1-fpr), index=i), 'threshold':pd.Series(threshold, index=i)})

    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])



# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/HR_comma_sep.csv')

dataset = data.rename( columns = {'left':'class'} )



# Unbalanced dataset

# Upsampling

minority = dataset[ dataset['class'] == 1 ]

minority_upsampled = resample( minority, replace = True, n_samples = 11428, random_state = 123 )



# Downsampling

majority = dataset[ dataset['class'] == 0 ]

majority_downsampled = resample( majority, replace = False, n_samples = 3571, random_state = 123 )



dataset = pd.concat( [minority, majority_downsampled] )



# Transform features

dataset = pd.get_dummies( dataset, columns = ['sales', 'salary'] )



# Selection features

features = dataset.drop( 'class', axis = 1 )

labels = dataset[['class']]



rf = RandomForestClassifier( n_estimators = 100, criterion = 'entropy', max_depth = 15,

                             min_samples_leaf = 50, min_samples_split = 100, random_state = 10 )

# Train the selector

rf.fit( features, labels.values.ravel() )

features_imp = pd.Series(rf.feature_importances_,index=features.columns).sort_values(ascending=False)

#print( 'features importance:\n', features_imp )



criteria = rf.feature_importances_ >  0.155

features = features.iloc[:, criteria ]

print( pd.DataFrame({'Main Features': features.columns} ) )



# Split dataset - Train and Test dataset

trainX, testX, trainY, testY = train_test_split( features, labels, test_size = 0.2 )



trainX = trainX.as_matrix()

trainY = to_categorical( trainY, num_classes = 2 )



testX = testX.as_matrix()

testY = to_categorical( testY, num_classes = 2 )



## ==================================================================

## Cross-Validation

#results = []

#i = 0

#n_folds = 10

#cv = KFold( len( trainX ), n_folds = n_folds )

#callbacks = [EarlyStopping( monitor = 'val_loss', patience = 2 )]

#for traincv, testcv in cv:

#    print( '===> running fold', i+1, '/', n_folds )

#    # Train

#    mlp = Sequential()

#    mlp.add(Dense(100, input_dim=trainX.shape[1], kernel_initializer='random_uniform',bias_initializer='random_uniform', activation='relu'))

#    mlp.add( Dense( 2, kernel_initializer='random_uniform', bias_initializer = 'random_uniform', activation='sigmoid' ) )

#    mlp.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

#    mlp.fit( trainX[traincv], trainY[traincv], batch_size=64, epochs=50, verbose=0, validation_split=0.3, callbacks=callbacks )

#    # Test

#    predicted = mlp.predict_proba( trainX[testcv] )

#    predicted = np.where( predicted[:,0] > 0.5, 1, 0 )

#    acc = accuracy_score( trainY[testcv,0], predicted )

#    print( '\naccuracy:', acc )

#    results.append( acc )

#    i += 1

#print( 'Results:', str( 100*np.array( results ).mean() ), '+/-', str( np.array( results ).std() ) )

# ==================================================================

# Training model - MLP

mlp = Sequential()

mlp.add( Dense( 100, input_dim=trainX.shape[1], kernel_initializer='random_uniform', bias_initializer='random_uniform', activation='relu') )

mlp.add( Dense( 2, kernel_initializer='random_uniform', bias_initializer = 'random_uniform', activation='sigmoid' ) ) # Output Layer

mlp.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )



# Training

callbacks = [EarlyStopping( monitor = 'val_loss', patience = 2 )]

mlp_info = mlp.fit( trainX, trainY, batch_size=64, epochs=50, verbose=2, validation_split=0.3, callbacks = callbacks )



# Training Error

fig, axs = plt.subplots( figsize = ( 15, 5 ) )

axs.plot( range( 1, len( mlp_info.history['loss'] ) + 1 ), mlp_info.history['loss'] )

axs.plot( range( 1, len( mlp_info.history['val_loss'] ) + 1 ), mlp_info.history['val_loss'] )

axs.set_title( 'Model Loss' )

axs.set_title( 'Loss' )

axs.set_title( 'Epoch' )

axs.set_xticks( np.arange( 1,len(mlp_info.history['loss'] )+1), len(mlp_info.history['loss'])/10 )

axs.legend( ['train', 'val'], loc = 'best' )

plt.show()



# Performance model over Test Dataset

predicted = mlp.predict_proba( testX )

threshold = find_optimal_cutoff( testY[:,0], predicted[:,0] )

print( '\nthreshold:', threshold[0] )

predicted = np.where( predicted[:,0] > threshold, 1, 0 )



# Metrics

# Accuracy

acc = accuracy_score( testY[:,0], predicted )

print( 'accuracy:', acc )



# Confusion Matrix

cm = confusion_matrix( testY[:,0], predicted )

print( cm )



# F1-score

f1 = f1_score( testY[:,0], predicted )

print( 'f1-score:', f1 )



print( '===> well done' )
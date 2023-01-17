# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

plt.style.use('bmh')

plt.rcParams['figure.figsize'] = (10, 10)

title_config = {'fontsize': 20, 'y': 1.05}

train = pd.read_csv('../input/train.csv')

train.head()
X = train.iloc[:, 2:].values.astype('float64')

Y = train['target'].values



train['target'].unique()
#pd.DataFrame(X[Y == 0]).plot.kde(ind=100, legend=False, figsize=(10, 10))

#plt.title('Likelihood KDE Plots for the Negative Class (y = 0)', fontsize=20, y=1.05);
#pd.DataFrame(X[Y == 1]).plot.kde(ind=100, legend=False, figsize=(10, 10))

#plt.title('Likelihood KDE Plots for the Positive Class (y = 1)', fontsize=20, y=1.05);
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

scaled = pd.DataFrame(scaler.transform(X))

print("Done")
#scaled[Y == 0].plot.kde(ind=100, legend=False, figsize=(10, 10))

#plt.title('Likelihood KDE Plots for the Negative Class (y = 0) after Standardization', fontsize=20, y=1.05);
#scaled[Y == 1].plot.kde(ind=100, legend=False, figsize=(10, 10))

#plt.title('Likelihood KDE Plots for the Positive Class (y = 1) after Standardization', fontsize=20, y=1.05);
from sklearn.preprocessing import QuantileTransformer

quantileScaler = QuantileTransformer(output_distribution='normal')

quantileScaler.fit(X)

transformed = pd.DataFrame(quantileScaler.transform(X))

print('done')
#transformed[Y == 0].plot.kde(ind=100, legend=False, figsize=(10, 10))

#plt.title('Likelihood KDE Plots for the Negative Class (y = 0) after Quantile Transformation', fontsize=20, y=1.05);
#transformed[Y == 1].plot.kde(ind=100, legend=False, figsize=(10, 10));

#plt.title('Likelihood KDE Plots for the Positive Class (y = 1) after Quantile Transformation', fontsize=20, y=1.05);
#plt.figure(figsize=(10, 10))

#plt.imshow(transformed.corr())

#plt.colorbar()

#plt.title('Correlation Matrix Plot of the Features', fontsize=20, y=1.05);
def GetModel(init_mode='uniform'):

    from keras.models import Sequential

    from keras.layers import Dense, Activation, Dropout

    from keras.callbacks import EarlyStopping

    from keras import regularizers

    model = Sequential()

    model.add(Dense(128,activation='relu', kernel_initializer=init_mode,input_dim=200))

    model.add(Dropout(0.5))

    model.add(Dense(64,activation='relu',kernel_initializer=init_mode))

    model.add(Dropout(0.5))

    model.add(Dense(32,activation='relu',kernel_initializer=init_mode))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid',kernel_initializer=init_mode))



    # For a binary classification problem

    model.compile(optimizer='adam',

                  loss='binary_crossentropy',

                  metrics=['accuracy'])

    print(model.summary())

    return model



model = GetModel()
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



X = transformed.values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)



classifier = KerasClassifier(build_fn=GetModel, verbose=1)

batch_size = [128]

epochs = [15]



#param_grid = dict(batch_size=batch_size, epochs=epochs)



init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

param_grid = dict()

param_grid['init_mode'] =init_mode

param_grid['batch_size'] =batch_size

param_grid['epochs'] =epochs



score = 'precision'

grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1,cv=5,scoring='%s_macro' % score)

print('start')

grid_result = grid.fit(X_test, Y_test)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold

from keras.callbacks import EarlyStopping

X = transformed.values



model = GetModel('he_normal')



batch_size = 128

folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, Y)):

    print("Fold {}".format(fold_))

    X_fold_train, X_fold_test = X[trn_idx], X[val_idx]

    Y_fold_train, Y_fold_test = Y[trn_idx], Y[val_idx]

    model.fit(X_fold_train, Y_fold_train, epochs = 15, batch_size=batch_size, verbose = 2,validation_data=(X_fold_test, Y_fold_test), callbacks = [EarlyStopping(monitor='val_acc', patience=4)])

    



score,acc = model.evaluate(X, Y, verbose = 1, batch_size = batch_size)

print("loss: %.2f" % (score))

print("acc: %.2f" % (acc))



from sklearn.metrics import roc_curve, auc

Y_test_pridict = model.predict(X,batch_size=1,verbose = 1)

fpr, tpr, thr = roc_curve(Y, Y_test_pridict)

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic Plot', **title_config)

auc(fpr, tpr)
#print(len(XTemp))

#XTemp = X

#X = transformed.values

#X = scaled.values

#print('done')
#print(X[0])

#print(transformed[0])

#print(X.shape)

#print(transformed.shape)

#print(type(X))

#print(type(transformed))

#print('done')
#from sklearn.model_selection import train_test_split

#X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
#print(X_train.shape,Y_train.shape)

#print(X_test.shape,Y_test.shape)

#print(type(X_train))


#validation_size = int(len(X_test) * 0.3)

#print(validation_size)





#X_validate = X_test[-validation_size:]

#Y_validate = [Y_test[-validation_size:]]

#X_test = X_test[:-validation_size]

#Y_test = [Y_test[:-validation_size]]
#print(len(Y_test[0]))
#print(len(X_test))

#batch_size = 128

#model.fit(X_train, Y_train, epochs = 15, batch_size=batch_size, verbose = 1,validation_data=(X_validate, Y_validate), callbacks = [EarlyStopping(monitor='val_acc', patience=3)])
#score,acc = model.evaluate(X_test, Y_test, verbose = 1, batch_size = batch_size)

#print("loss: %.2f" % (score))

#print("acc: %.2f" % (acc))
def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    from sklearn.utils.multiclass import unique_labels

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
'''from sklearn.metrics import confusion_matrix

Y_test_pridict = model.predict(X_test,batch_size=1,verbose = 1)

print(Y_test_pridict[0])

#Y_test_pridict = np.argmax(Y_test_pridict,axis=1)

print(Y_test_pridict[0])

#cm = confusion_matrix(Y_test[0], Y_test_pridict)

plot_confusion_matrix(Y_test[0], Y_test_pridict,np.array(['1','0']))'''
'''from sklearn.metrics import roc_curve, auc

Y_test_pridict = model.predict(X,batch_size=1,verbose = 1)

fpr, tpr, thr = roc_curve(Y, Y_test_pridict)

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic Plot', **title_config)

auc(fpr, tpr)'''
testData = pd.read_csv("../input/test.csv")

testData.head()
#print(testData.head())

Y_Id_code = testData['ID_code'].values

testData = testData.iloc[:,1:].values.astype('float64')

testData.shape
#print(type(testData))

tData = pd.DataFrame(quantileScaler.transform(testData))

#tData = pd.DataFrame(scaler.transform(testData))

tData.head()
tData = tData.values

print(len(tData))

result = model.predict(tData,batch_size=1,verbose = 1)

#print(len(result))
print(Y_Id_code.shape)

print(result.shape)

dfResult = pd.DataFrame({"ID_code": Y_Id_code[:], "target": result[:,0]})

dfResult.head()

    
dfResult.to_csv("submission10.csv", index=False)
print(os.listdir())

#!pip install kaggle

#!kaggle competitions submit -c santander-customer-transaction-prediction -f submission.csv -m "Message"
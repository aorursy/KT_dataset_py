import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
%matplotlib inline

from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.5)

xdata = (lfw_people.images)
y = lfw_people.target
plt.imshow(np.array(xdata[200,:]).reshape(lfw_people.images.shape[1],lfw_people.images.shape[2]),cmap='gray')

xtrain, xtest, ytrain,ytest = train_test_split(xdata,y,test_size=0.3)
xtrain = xtrain/255 #obligar a flotante normalizado 0 a 1
xtest = xtest/255 
print(xtrain.shape, xtest.shape)
type(xtrain)
print('Numero de clases:',len(np.unique(ytrain)))
l1=0.00001
l2=0.000001
tf.keras.backend.clear_session()
inputA = tf.keras.layers.Input(shape=(xtrain.shape[1],xtrain.shape[2]), name='entradaA')
flattenA = tf.keras.layers.Flatten(input_shape=(xtrain.shape[1],xtrain.shape[2]))(inputA)
h1A = tf.keras.layers.Dense(70,activation='relu',name='h1A',kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1,l2=l2))(flattenA)
outputA = tf.keras.layers.Dense(10,activation="softmax",name='persona')(h1A)
model_fun = tf.keras.Model(inputs=inputA,outputs=outputA)
tf.keras.utils.plot_model(model_fun)
model_fun.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy()
                  ,loss_weights = 0.5
                  ,optimizer=tf.keras.optimizers.Adam(learning_rate=0.005)
                  ,metrics=["accuracy"])
history = model_fun.fit(x =xtrain, y=ytrain, 
                        epochs=40,batch_size=64,
                        validation_split=0.3)
                       
import pandas as pd
hpd = pd.DataFrame(history.history)
hpd.plot()
plt.grid(True)
plt.ylim(0,2)
plt.show()
hpd[['loss','val_loss']].plot()
plt.grid(True)
plt.ylim(0,2)
plt.show()
W1 = abs(model_fun.get_layer('h1A').get_weights()[0]).sum(axis=1).reshape(xtest.shape[1],xtest.shape[2])
Wc = np.c_[W1]
Wc /=np.max(Wc)
plt.imshow(Wc,vmin=0,vmax=1)
plt.colorbar()
plt.show()
#matriz de confusion
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
from sklearn.utils.multiclass import unique_labels
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
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
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
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
    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

ytest_e = model_fun.predict(xtest)
print(ytest_e.shape)
plot_confusion_matrix(ytest, ytest_e.argmax(axis=1),classes=np.unique(lfw_people.target_names))
plt.title('Multiclase')
print(classification_report(ytest, ytest_e.argmax(axis=1)))


ii = 25
pe= model_fun.predict([xtest[ii][np.newaxis,:,:]])
print(pe.argmax(),)
plt.imshow(np.c_[xtest[ii]], cmap='gray',vmin=0,vmax=1)
plt.show()

print(lfw_people.target_names[pe.argmax()])

ii = 60
pe= model_fun.predict([xtest[ii][np.newaxis,:,:]])
print(pe.argmax(),)
plt.imshow(np.c_[xtest[ii]], cmap='gray',vmin=0,vmax=1)
plt.show()

print(lfw_people.target_names[pe.argmax()])
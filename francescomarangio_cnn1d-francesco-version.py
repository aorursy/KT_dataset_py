from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten, SpatialDropout1D, GlobalAveragePooling1D
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import Callback,EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
# General packages
import numpy as np
import os
import h5py
import seaborn as sns
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import pandas as pd
TPU=0

# Data preparation and validation packages
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
# Jupyter interactive plotting
from IPython.display import clear_output
pd.set_option('display.max_rows', None)
class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracies.append(logs.get('val_acc'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.accuracies, label="accuracy")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.plot(self.x, self.val_accuracies, label="val_accuracy")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)),plt.grid()
        plt.show();
        
plot_losses = PlotLosses()
detector="et"
poisson=10
#TRAINING+VALIDATION SET(85 % OF DATA)


for i in np.arange(0,85,85):
    print('BBH'+str(poisson)+'s_serie'+str(i)+'.csv')
    if detector=="ligo":
        inputdata=pd.read_csv("/kaggle/input/"+detector+"bbh"+str(poisson)+"s/new_BBH"+str(poisson)+"s_serie"+str(i)+".csv") 
    else:
        inputdata=pd.read_csv("/kaggle/input/"+detector+"bbh"+str(poisson)+"s/BBH"+str(poisson)+"s_serie"+str(i)+".csv") 
    del inputdata['Unnamed: 0']
    #print(inputdata.shape[0])
#     print(inputdata)
#     s=inputdata.iloc[0][:4096]
#     n=inputdata.iloc[197][:4096]
#     signal=pd.DataFrame(data=s)
#     noise=pd.DataFrame(data=n)
#     signal.to_csv("signal.csv")
#     noise.to_csv("noise.csv")
#     print(s)
#     print(n)
#     s.plot(color='blue')
#     n.plot(color='orange')
#     fs=4096
#     f, t, Sxx = signal.spectrogram(n, fs)
#     plt.pcolormesh(t, f, Sxx)
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [sec]')
#     plt.show()
#     print(inputdata)
    for j in range(i+1,i+85):
        if j==23 and detector=="ligo" and poisson==10:
            continue
        if j==1 and detector=="et" and poisson==4:
            continue
        if detector=="ligo":
            inputdata_append=pd.read_csv("/kaggle/input/"+detector+"bbh"+str(poisson)+"s/new_BBH"+str(poisson)+"s_serie"+str(j)+".csv")
        else:
            inputdata_append=pd.read_csv("/kaggle/input/"+detector+"bbh"+str(poisson)+"s/BBH"+str(poisson)+"s_serie"+str(j)+".csv")
        print('BBH'+str(poisson)+'s_serie'+str(j)+'.csv')
        del inputdata_append['Unnamed: 0']
        inputdata=pd.concat([inputdata,inputdata_append])

    inputdata=shuffle(inputdata)
    columnsET=[str(l) for l in range(4096)]
    inputs=inputdata[columnsET].values
    inputs=preprocessing.minmax_scale(inputs.T).T
    
    targets=inputdata['Label'].values
    
    inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], 1))
    onehot_encoder = OneHotEncoder(sparse=False)
    targets = targets.reshape(targets.shape[0], 1)
    targets = onehot_encoder.fit_transform(targets)
    x_train, x_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.3)
    n_classes = len(np.unique(targets))

#TESTING SET (15 % OF DATA)

for i in np.arange(85,100,15):
    print('BBH'+str(poisson)+'s_serie'+str(i)+'.csv')
    if detector=="ligo":
        testdata=pd.read_csv("/kaggle/input/"+detector+"bbh"+str(poisson)+"s/new_BBH"+str(poisson)+"s_serie"+str(i)+".csv")
    else:
        testdata=pd.read_csv("/kaggle/input/"+detector+"bbh"+str(poisson)+"s/BBH"+str(poisson)+"s_serie"+str(i)+".csv")
  
    del testdata['Unnamed: 0']
    for j in range(i+1,i+15):
        if detector=="ligo":
            testdata_append=pd.read_csv("/kaggle/input/"+detector+"bbh"+str(poisson)+"s/new_BBH"+str(poisson)+"s_serie"+str(j)+".csv")
        else:
            testdata_append=pd.read_csv("/kaggle/input/"+detector+"bbh"+str(poisson)+"s/BBH"+str(poisson)+"s_serie"+str(j)+".csv")
        print('BBH'+str(poisson)+'s_serie'+str(j)+'.csv')
        del testdata_append['Unnamed: 0']
        testdata=pd.concat([testdata,testdata_append])
# #NEW MODEL1 
# model = Sequential()

# model.add(Conv1D(filters=32, kernel_size=32, activation="relu", input_shape=(4096,1)))
# model.add(MaxPool1D(4))
# model.add(Conv1D(filters=24, kernel_size=32, activation="relu"))
# model.add(MaxPool1D(4))
# model.add(Conv1D(filters=16, kernel_size=32 ,activation="relu"))
# model.add(MaxPool1D(4))
# model.add(Flatten())

# model.add(Dense(100, activation="relu"))
# model.add(Dense(50, activation="relu"))
# model.add(Dense(2, activation="softmax"))
# #model.summary()
# opt = optimizers.Adam(lr=0.0001)
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["acc"])
# model_checkpoint = ModelCheckpoint('best_weights_poisson'+str(poisson)+'.h5', monitor='val_acc',verbose=1, save_best_only=True)
# es=EarlyStopping(monitor='val_acc',min_delta=0.00005,mode='max',verbose=1,patience=50,restore_best_weights=True)
#NEW MODEL2
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=46, activation="relu", input_shape=(4096,1)))
model.add(MaxPool1D(2))
model.add(Conv1D(filters=32, kernel_size=46, activation="relu"))
model.add(MaxPool1D(2))
model.add(Conv1D(filters=32, kernel_size=46 ,activation="relu"))
model.add(MaxPool1D(2))
model.add(Conv1D(filters=32, kernel_size=46 ,activation="relu"))
model.add(MaxPool1D(2))
model.add(Conv1D(filters=32, kernel_size=46 ,activation="relu"))
model.add(MaxPool1D(2))
model.add(Conv1D(filters=32, kernel_size=46 ,activation="relu"))
model.add(MaxPool1D(2))



model.add(Flatten())

# model.add(Dense(100, activation="relu"))
# model.add(Dense(50, activation="relu"))
model.add(Dense(2, activation="softmax"))
model.summary()

opt = optimizers.Adam(lr=0.0001)
#lr = ReduceLROnPlateau(monitor='val_acc', min_delta=0.00005,patience=5, min_lr=0.000001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["acc"])
model_checkpoint = ModelCheckpoint('best_weights_poisson'+str(poisson)+detector+'.h5', monitor='val_acc',verbose=1, save_best_only=True)
es=EarlyStopping(monitor='val_acc',min_delta=0.00005,mode='max',verbose=1,patience=50,restore_best_weights=True)
#TRAINING
model.fit(x_train, y_train,   
                        epochs=100,
                        batch_size=64,
                        validation_data=(x_val, y_val),
                        callbacks=[plot_losses,es,model_checkpoint],shuffle=True)
          

model.save_weights('final_weights_poisson'+str(poisson)+detector+'.h5')
#TESTING-CONFUSION MATRIX

model.load_weights('best_weights_poisson'+str(poisson)+detector+'.h5')
testdata=shuffle(testdata)
columnsET=[str(l) for l in range(4096)]
inputs=testdata[columnsET].values
inputs=preprocessing.minmax_scale(inputs.T).T
targets=testdata['Label'].values
inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], 1))
onehot_encoder = OneHotEncoder(sparse=False)
targets = targets.reshape(targets.shape[0], 1)
targets = onehot_encoder.fit_transform(targets)
n_classes = len(np.unique(targets))

y_predicted = model.predict(inputs)
cms = confusion_matrix(targets.argmax(1), y_predicted.argmax(1)) 
test_score = np.trace(cms) / np.sum(cms) 
new_cms = np.zeros((n_classes,n_classes))
for x in range(n_classes):
    for y in range(n_classes):
        new_cms[x,y] = round(cms[x,y] / np.sum(cms[x])*100,1)

plt.rc("font", size=20)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
im = ax.imshow(np.transpose(new_cms), interpolation="nearest", cmap="cool")
for x in range(0, n_classes):
    for y in range(0, n_classes):
        ax.text(x, y, new_cms[x,y], color="black", ha="center", va="center")
plt.title("Total accuracy: " + str(np.around(test_score*100, 1)), fontsize=20)

plt.colorbar(im)

classes_values = []
classes_labels = ["Noise", "BBH"]
for n in range(n_classes):
    classes_values.append(n)

plt.xticks(classes_values, classes_labels, rotation=45, fontsize=15)
plt.yticks(classes_values, classes_labels, fontsize=15)
plt.xlabel("Real data", fontsize=15)
plt.ylabel("Predicted data", fontsize=15), plt.ylim([-0.5,n_classes-0.5])
axis = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])

#TPU APPLICATION
if TPU==1:
    import tensorflow as tf
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    # instantiate a distribution strategy
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    BATCH_SIZE = 16 * tpu_strategy.num_replicas_in_sync
# instantiating the model in the strategy scope creates the model on the TPU
    with tpu_strategy.scope():
        model = tf.keras.Sequential([tf.keras.layers.Conv1D(filters=32, kernel_size=32, activation="relu", input_shape=(4096,1)),
                                     tf.keras.layers.MaxPool1D(4),
                                     tf.keras.layers.Conv1D(filters=24, kernel_size=32, activation="relu"),
                                     tf.keras.layers.MaxPool1D(4),
                                     tf.keras.layers.Conv1D(filters=16, kernel_size=32,activation="relu"),
                                     tf.keras.layers.MaxPool1D(4),
                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(100, activation="relu"),
                                     tf.keras.layers.Dense(50, activation="relu"),
                                     tf.keras.layers.Dense(2, activation="softmax")])
        
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["acc"])
        model_checkpoint = ModelCheckpoint('best_weights_poisson'+str(poisson)+'.h5', monitor='val_acc',verbose=1, save_best_only=True)
        es=EarlyStopping(monitor='val_acc',min_delta=0.00005,mode='max',verbose=1,patience=50,restore_best_weights=True)
    model.fit(x_train.astype(np.float32), y_train.astype(np.float32),   
                        epochs=300,
                        batch_size=24427,
                        validation_data=(x_val.astype(np.float32), y_val.astype(np.float32)),
                        callbacks=[plot_losses,es,model_checkpoint],shuffle=True)
else:
    model = Sequential()

    model.add(Conv1D(filters=64, kernel_size=16, activation="relu", input_shape=(4096,1)))
    model.add(MaxPool1D(4))
    model.add(Conv1D(filters=64,kernel_size=16, activation="relu"))
    model.add(MaxPool1D(4))
    model.add(Conv1D(filters=64,kernel_size=16, activation="relu"))
    model.add(MaxPool1D(4))

    
    model.add(Flatten())
    
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    model.summary()
    opt = optimizers.Adam(lr=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["acc"])
    model_checkpoint = ModelCheckpoint('best_weights_poisson'+str(poisson)+detector+'.h5', monitor='val_acc',verbose=1, save_best_only=True)
    es=EarlyStopping(monitor='val_acc',min_delta=0.00005,mode='max',verbose=1,patience=50,restore_best_weights=True)

    

#TESTING FOR POISSON PARAMETER
progress=np.zeros(100)
directories=['/home/shared/MDC/long_data/bbh_4','/home/shared/MDC/long_data/bbh_10','/home/shared/MDC/long_data/noise']# RAW DATA
directory=directories[1]
targets=np.zeros([2048,1])
onehot_encoder = OneHotEncoder(sparse=False)
targets = targets.reshape(targets.shape[0], 1)
targets = onehot_encoder.fit_transform(targets)

for j in np.arange(0,100,1):
    fileserie=directory+'/serie_'+str(j)+'.dat.gz'
    print(fileserie)
    ETdata=df=pd.read_csv(fileserie,sep=' ', lineterminator='\n',header=None,names=['t0','ET1','ET2','ET3'],usecols=['ET1',])
    datalist=[]
    if t0!=1:
        targets=np.zeros([2047,1])
        
    for k in np.arange(0,2048):
        if (4096*k+t*4096+4096)<2048*4096:
            frame=ETdata[(4096*k+int(t*4096)):(4096*k+int(t*4096)+4096)]
            frame=frame.transpose()
            frame=np.array(frame)

            try:
                datalist=np.vstack((datalist,frame))
            except:
                datalist=frame    

    inputs=preprocessing.minmax_scale(datalist.T).T
    inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], 1))
    y_predicted=model.predict(inputs)
    cms = confusion_matrix(targets.argmax(1), y_predicted.argmax(1))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(np.transpose(cms), interpolation="nearest", cmap="cool")
    rows = cms.shape[0]
    cols = cms.shape[1]
    for x in range(0, rows):
        for y in range(0, cols):
            value = int(cms[x, y])
            ax.text(x, y, value, color="black", ha="center", va="center", fontsize=25)

    test_score= format(1/(cms[0,1]/2047),'.3g')
    progress[j]=test_score
    plt.title("Distribution:" + str(test_score)+'s', fontsize=25)
    plt.colorbar(im)

    classes_values = []
    classes_labels = []
    for n in range(n_classes):
        classes_values.append(n)
        classes_labels.append(str(n))

    plt.xticks(classes_values, classes_labels, rotation=45, fontsize=15)
    plt.yticks(classes_values, classes_labels, fontsize=15)
    plt.xlabel("Real data", fontsize=15)
    plt.ylabel("Predicted data", fontsize=15), plt.ylim([-0.5,n_classes-0.5])
    axis = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
#GRID SEARCH
maxpool=[2]
nb_filter=[16,32,48,64]
filter_length=[46,48,50]

for pool in maxpool:
    for nfilter in nb_filter:
        for sizefilter in filter_length:
            model = Sequential()
            model.add(Conv1D(activation="relu", input_shape=(inputs.shape[1],1), filters=nfilter, kernel_size=sizefilter ))
            model.add(MaxPool1D(pool))
            #model.add(SpatialDropout1D(0.5))

            model.add(Conv1D(activation="relu", filters=nfilter, kernel_size=sizefilter ))
            model.add(MaxPool1D(pool))
            #model.add(SpatialDropout1D(0.4))

            model.add(Conv1D(activation="relu",filters=nfilter, kernel_size=sizefilter ))
            model.add(MaxPool1D(pool))

            model.add(Flatten())

            #model.add(Dense(100, activation="relu"))
            #model.add(Dense(50, activation="relu"))
            model.add(Dense(2, activation="softmax"))

            opt = optimizers.Adam(lr=0.0001)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["acc"])
            #model_checkpoint = ModelCheckpoint('best_weights_poisson'+str(poisson)+'.h5', monitor='val_acc',verbose=1, save_best_only=True)
            #es=EarlyStopping(monitor='val_acc',min_delta=0.00005,mode='max',verbose=1,patience=50,restore_best_weights=True)
            history=model.fit(x_train, y_train,   
            epochs=70,
            batch_size=64,
            validation_data=(x_val, y_val),verbose=0)
            print("Maxpool:"+str(pool)+","+str(pool)+","+str(pool)+" Conv filter:"+str(sizefilter)+","+str(sizefilter)+","+str(sizefilter)+" Num filters:"+str(nfilter)+","+str(nfilter)+","+str(nfilter)+" Max Val accuracy:"+str(max(history.history['val_acc'])))

#Hyperparamters search scikit learn
testdata=shuffle(testdata)
columnsET=[str(l) for l in range(4096)]
inputs=testdata[columnsET].values
inputs=preprocessing.minmax_scale(inputs.T).T

targets=testdata['Label'].values

inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], 1))
onehot_encoder = OneHotEncoder(sparse=False)
targets = targets.reshape(targets.shape[0], 1)
targets = onehot_encoder.fit_transform(targets)
x_train, x_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.3)

def create_model(filters1=32,kernel1=32,pool1=4,filters2=24,kernel2=32,pool2=4,filters3=16,kernel3=32
                 ,pool3=4,dense1=100,dense2=50):
    model = Sequential()
    
    model.add(Conv1D(filters=filters1, kernel_size=kernel1, activation="relu", input_shape=(4096,1)))
    model.add(MaxPool1D(pool1))
    model.add(Conv1D(filters=filters2,kernel_size=kernel2, activation="relu"))
    model.add(MaxPool1D(pool2))
    model.add(Conv1D(filters=filters3,kernel_size=kernel3, activation="relu"))
    model.add(MaxPool1D(pool3))


    model.add(Flatten())
    
    model.add(Dense(dense1, activation="relu"))
    model.add(Dense(dense2, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    opt = optimizers.Adam(lr=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["acc"])
    return model

model = KerasClassifier(build_fn=create_model, verbose=0)

filters1=[8,12]
filters2=[8,12]
filters3=[8,12]
kernel1=[3,6]
kernel2=[3,6]
kernel3=[3,6]
dense1=[1500,250]
dense2=[100,50]
param_grid = dict(filters1=filters1,filters2=filters2,filters3=filters3,kernel1=kernel1,kernel2=kernel2,kernel3=kernel3,dense1=dense1,dense2=dense2)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
a=0

grid_result = grid.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=2)
# summarize results
print(a)
#
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# #MODEL 75.5% OLD
# model = Sequential()

# model.add(Conv1D(filters=16, kernel_size=32, activation="relu", input_shape=(4096,1)))

# model.add(MaxPool1D(8))
# #model.add(SpatialDropout1D(0.2))

# model.add(Conv1D(filters=12, kernel_size=16, activation="relu"))
# model.add(MaxPool1D(8))
# #model.add(SpatialDropout1D(0.2))

# model.add(Conv1D(filters=8, kernel_size=8 ,activation="relu"))
# model.add(MaxPool1D(8))

# #model.add(SpatialDropout1D(0.2))

# #model.add(Dropout(0.1))
# model.add(Flatten())

# model.add(Dense(100, activation="relu"))
# model.add(Dense(50, activation="relu"))
# model.add(Dense(2, activation="softmax"))
# #model.summary()
# opt = optimizers.Adam(lr=0.0001)
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["acc"])
# model_checkpoint = ModelCheckpoint('best_weights_poisson'+str(poisson)+'.hdf5', monitor='val_acc',verbose=1, save_best_only=True)
# es=EarlyStopping(monitor='val_acc',min_delta=0.00005,mode='max',verbose=1,patience=50,restore_best_weights=True)
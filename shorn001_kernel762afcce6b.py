from keras.utils import to_categorical

def dataset_to_tensor(data):

    feature= np.array(data[['userAcceleration.x','userAcceleration.y','userAcceleration.z']])

    label =  np.array(to_categorical(data[['act']]))

    from keras.preprocessing.sequence import TimeseriesGenerator

    data_gen = TimeseriesGenerator(feature, label,

                               length=100,stride=100,sampling_rate=1,batch_size=len(data))

    return data_gen[0]
import numpy as np

import pandas as pd

from keras.preprocessing.sequence import TimeseriesGenerator

def get_ds_infos():

    """

    Read the file includes data subject information.

    

    Data Columns:

    0: code [1-24]

    1: weight [kg]

    2: height [cm]

    3: age [years]

    4: gender [0:Female, 1:Male]

    

    Returns:

        A pandas DataFrame that contains inforamtion about data subjects' attributes 

    """ 



    dss = pd.read_csv("/kaggle/input/motionsense-dataset/data_subjects_info.csv")

    print("[INFO] -- Data subjects' information is imported.")

    

    return dss



def set_data_types(data_types=["userAcceleration"]):

    """

    Select the sensors and the mode to shape the final dataset.

    

    Args:

        data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration] 



    Returns:

        It returns a list of columns to use for creating time-series from files.

    """

    dt_list = []

    for t in data_types:

        if t != "attitude":

            dt_list.append([t+".x",t+".y",t+".z"])

        else:

            dt_list.append([t+".roll", t+".pitch", t+".yaw"])



    return dt_list





def creat_time_series(dt_list, act_labels, trial_codes, mode="mag", labeled=True):

    """

    Args:

        dt_list: A list of columns that shows the type of data we want.

        act_labels: list of activites

        trial_codes: list of trials

        mode: It can be "raw" which means you want raw data

        for every dimention of each data type,

        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].

        or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)

        labeled: True, if we want a labeld dataset. False, if we only want sensor values.



    Returns:

        It returns a time-series of sensor data.

    

    """

    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list*3)

    

    tensor_vals = np.empty([0,100,num_data_cols] ,dtype=object) #TOdo initialize Tuple Correctly

    tensor_labels = np.empty([0,len(act_labels)])

    print (tensor_vals.shape)

    print(tensor_labels.shape)



    

    if labeled:

        dataset = np.zeros((0,num_data_cols+7)) # "7" --> [act, code, weight, height, age, gender, trial] 

    else:

        dataset = np.zeros((0,num_data_cols))

        

    ds_list = get_ds_infos()

    

    print("[INFO] -- Creating Time-Series")

    for sub_id in ds_list["code"]:

        for act_id, act in enumerate(act_labels):

            for trial in trial_codes[act_id]:

                fname = '/kaggle/input/motionsense-dataset/A_DeviceMotion_data/A_DeviceMotion_data/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'

                raw_data = pd.read_csv(fname)

                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)

                vals = np.zeros((len(raw_data), num_data_cols))

                for x_id, axes in enumerate(dt_list):

                    if mode == "mag":

                        vals[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5        

                    else:

                        vals[:,x_id*3:(x_id+1)*3] = raw_data[axes].values

                    vals = vals[:,:num_data_cols]

                if labeled:

                    lbls = np.array([[act_id,

                            sub_id-1,

                            ds_list["weight"][sub_id-1],

                            ds_list["height"][sub_id-1],

                            ds_list["age"][sub_id-1],

                            ds_list["gender"][sub_id-1],

                            trial          

                           ]]*len(raw_data))



                    data_gen = TimeseriesGenerator(vals, np.full(len(vals),act_id),length=100,stride=100,sampling_rate=1,batch_size=len(vals)) 

                    vals = np.concatenate((vals, lbls), axis=1)



                dataset = np.append(dataset,vals, axis=0)

                tensor_vals= np.append(tensor_vals,data_gen[0][0],axis=0) 

                tensor_labels=np.append(tensor_labels,to_categorical(data_gen[0][1],num_classes=len(act_labels)),axis=0)

                

    cols = []

    for axes in dt_list:

        if mode == "raw":

            cols += axes

        else:

            cols += [str(axes[0][:-2])]

            

    if labeled:

        cols += ["act", "id", "weight", "height", "age", "gender", "trial"]

    

    dataset = pd.DataFrame(data=dataset, columns=cols)

    return dataset,tensor_vals,tensor_labels

#________________________________





ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]

TRIAL_CODES = {

    ACT_LABELS[0]:[1,2,11],

    ACT_LABELS[1]:[3,4,12],

    ACT_LABELS[2]:[8,15], #Removed Trail 7 to avoid imbalanced target classes

    ACT_LABELS[3]:[9,16],

    ACT_LABELS[4]:[6,14],

    ACT_LABELS[5]:[5,13]

}



## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"

## attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)

sdt = ["userAcceleration"]

print("[INFO] -- Selected sensor data types: "+str(sdt))    

act_labels = ACT_LABELS [0:4]

print("[INFO] -- Selected activites: "+str(act_labels))    

trial_codes = [TRIAL_CODES[act] for act in act_labels]

dt_list = set_data_types(sdt)

dataset,tensor_vals,tensor_labels= creat_time_series(dt_list, act_labels, trial_codes, mode="raw", labeled=True)

print("[INFO] -- Shape of time-Series dataset:"+str(dataset.shape))    

dataset['act'] = pd.Categorical(dataset['act'].astype(int))

dataset['act'].cat.rename_categories(ACT_LABELS[0:4],inplace=True)

dataset.head()
#tensor_vals = np.empty([0,100,3] ,dtype=object) #TOdo initialize Tuple Correctly

#tensor_labels = np.empty([0,len(act_labels)])



#for g in dataset.groupby(['act','id']).apply(lambda x : TimeseriesGenerator(x[['userAcceleration.x','userAcceleration.y','userAcceleration.z']].values.tolist(), np.full(len(x['act'].cat.codes.tolist()),x['act'].cat.codes.tolist()),length=100,stride=100,sampling_rate=1,batch_size=len(x['act'].cat.codes.tolist()))):

   # data_gen = g

   # tensor_vals= np.append(tensor_vals,data_gen[0][0],axis=0) 

   # tensor_labels=np.append(tensor_labels,to_categorical(data_gen[0][1],num_classes=len(act_labels)),axis=0)

import matplotlib.pyplot as plt

plt.figure(figsize=(30,10))

dataset.groupby('act')['userAcceleration.x'].count().plot(kind='bar')

plt.title('Number of Timeseries Datapoints by Actitity')

plt.xlabel('Activity')

plt.ylabel('Number of Datapoints')

plt.show()

plt.savefig('datapoints_by_label.svg')

#print (dataset.dtypes)
dataset.describe()
dataset[['userAcceleration.x','userAcceleration.y','userAcceleration.z']].describe()
dataset[['userAcceleration.x','userAcceleration.y','userAcceleration.z']].boxplot()

fig = plt.gcf()

fig.set_size_inches(15, 10)


dataset[['weight','height','age']].boxplot()

fig = plt.gcf()

fig.set_size_inches(30, 15)
import numpy as np

from sklearn.base import TransformerMixin

from sklearn.preprocessing import StandardScaler

class NDStandardScaler(TransformerMixin):

    def __init__(self, **kwargs):

        self._scaler = StandardScaler(copy=True, **kwargs)

        self._orig_shape = None



    def fit(self, X, **kwargs):

        X = np.array(X)

        # Save the original shape to reshape the flattened X later

        # back to its original shape

        if len(X.shape) > 1:

            self._orig_shape = X.shape[1:]

        X = self._flatten(X)

        self._scaler.fit(X, **kwargs)

        return self



    def transform(self, X, **kwargs):

        X = np.array(X)

        X = self._flatten(X)

        X = self._scaler.transform(X, **kwargs)

        X = self._reshape(X)

        return X



    def _flatten(self, X):

        # Reshape X to <= 2 dimensions

        if len(X.shape) > 2:

            n_dims = np.prod(self._orig_shape)

            X = X.reshape(-1, n_dims)

        return X



    def _reshape(self, X):

        # Reshape X back to it's original shape

        if len(X.shape) >= 2:

            X = X.reshape(-1, *self._orig_shape)

        return X
from sklearn import preprocessing

#print('Number of Batches: '+ str(len(tensor_data)))



#features,labels = tensor_data

scaler = NDStandardScaler()

features = scaler.fit_transform(tensor_vals)

labels = tensor_labels



n_features = len(features[0][0])

n_timesteps = len(features[0])

n_outputs = len(labels[0])



print('Shape of Sequences within a Batch')

print(features.shape)

print('Shape of Labels within a Batch')

print(labels.shape)



print('Number of features :'+ str(n_features))

print('Number of timestamps: '+ str(n_timesteps))

print('Number of outputs: '+ str(n_outputs))

print('Number of samples:'+ str(len(features)))


plt.bar(ACT_LABELS[0:4],np.bincount(np.argmax(labels,axis=1)))

plt.title('Distribution of Activitys among Samples')

plt.xlabel('Activity')

plt.ylabel('Number of Samples')

plt.show()
from sklearn.model_selection import train_test_split



# Split the data

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, shuffle= True)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.33, shuffle= True)



scores = dict()
def to_categorial(vector):

    res =[]

    for x in vector:

        res.append(np.argmax(x))

    return res

        


from sklearn.svm import LinearSVC

naive_model = LinearSVC()

naive_model.fit(x_train.reshape(len(x_train),300,order='F'),to_categorial(y_train))

from sklearn import metrics

prediction = naive_model.predict(x_valid.reshape(len(x_valid),300,order='F'))

f1 = metrics.f1_score(prediction,to_categorial(y_valid),average='weighted')

accurancy = metrics.accuracy_score(prediction,to_categorial(y_valid))

recall = metrics.recall_score(prediction,to_categorial(y_valid),average='weighted')

precision = metrics.precision_score(prediction,to_categorial(y_valid),average='weighted')

loss = metrics.hamming_loss(prediction,to_categorial(y_valid))

#TODO Calculate Value Loss

scores['navie']= {'loss':loss,'f1_m':f1,'accurancy':accurancy,'precision_m':precision,'recall_m':recall}

print ("F1 Socre of Naive Model: "+ str(f1) )
from keras import backend as K

import matplotlib.pyplot as plt



def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))





def train_model(model,name,training_data = (x_train,y_train),validation_data=(x_valid,y_valid),epochs=20,plot=False):

    from keras.callbacks import ModelCheckpoint

    model.compile(loss='categorical_crossentropy', optimizer='adam', weighted_metrics=[f1_m,'acc',precision_m, recall_m])

 

    checkpointer = ModelCheckpoint(filepath='/kaggle/working/weights.best.'+name+'.hdf5', 

                               verbose=1, save_best_only=True)



    history = model.fit(training_data[0], training_data[1], 

          validation_data=(validation_data[0], validation_data[1]),

          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

    if plot:

            # Plot training & validation accuracy values

        plt.plot(history.history['acc'])

        plt.plot(history.history['val_acc'])

        plt.title('Model accuracy')

        plt.ylabel('Accuracy')

        plt.xlabel('Epoch')

        plt.legend(['Train', 'Test'], loc='upper left')

        plt.show()



        # Plot training & validation loss values

        plt.plot(history.history['loss'])

        plt.plot(history.history['val_loss'])

        plt.title('Model loss')

        plt.ylabel('Loss')

        plt.xlabel('Epoch')

        plt.legend(['Train', 'Test'], loc='upper left')

        plt.show()

    



def evaluate_model(model,name,validation_data=(x_valid,y_valid)):

    model.load_weights('/kaggle/working/weights.best.'+name+'.hdf5')

    scores[name] = dict(zip (model.metrics_names,model.evaluate(validation_data[0],validation_data[1])))

    f1 = scores[name]['f1_m']*100

    print('Model: '+name+' test F1: %.4f%%' % f1)
from keras.layers import Conv1D, MaxPooling1D,GlobalAveragePooling1D

from keras.layers import Dropout, Flatten, Dense

from keras.models import Sequential



model_cnn = Sequential()

model_cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))

model_cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

model_cnn.add(Dropout(0.5))

model_cnn.add(MaxPooling1D(pool_size=2))

model_cnn.add(Flatten())

model_cnn.add(Dense(100, activation='relu'))

model_cnn.add(Dense(n_outputs, activation='softmax'))



model_cnn.summary()

train_model(model_cnn,'cnn')
evaluate_model(model_cnn,'cnn')
from keras.layers import LSTM

from keras.models import Sequential



model_lstm = Sequential()

model_lstm.add(LSTM(100, input_shape=(n_timesteps,n_features)))

model_lstm.add(Dropout(0.5))

model_lstm.add(Dense(100, activation='relu'))

model_lstm.add(Dense(n_outputs, activation='softmax'))

model_lstm.summary()

train_model(model_lstm,'lstm')
evaluate_model(model_lstm,'lstm')
from keras.models import Sequential

from keras.layers import Conv1D, MaxPooling1D,GlobalAveragePooling1D

from keras.layers import Dropout, Flatten, Dense,TimeDistributed,LSTM





n_steps, n_length = 4, 25

model_cnn_lstm = Sequential()

model_cnn_lstm.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(n_steps,n_length,n_features)))

model_cnn_lstm.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))

model_cnn_lstm.add(TimeDistributed(Dropout(0.5)))

model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=2)))

model_cnn_lstm.add(TimeDistributed(Flatten()))

model_cnn_lstm.add(LSTM(100))

model_cnn_lstm.add(Dropout(0.5))

model_cnn_lstm.add(Dense(100, activation='relu'))

model_cnn_lstm.add(Dense(n_outputs, activation='softmax'))

model_cnn_lstm.summary()
x_train_cnn_lstm = x_train.reshape((x_train.shape[0], n_steps, n_length, n_features))

x_valid_cnn_lstm = x_valid.reshape((x_valid.shape[0], n_steps, n_length, n_features))

x_test_cnn_lstm = x_test.reshape((x_test.shape[0], n_steps, n_length, n_features))
train_model(model_cnn_lstm,'cnn_lstm',training_data=(x_train_cnn_lstm,y_train),validation_data=(x_valid_cnn_lstm,y_valid))
evaluate_model(model_cnn_lstm,'cnn_lstm',validation_data=(x_valid_cnn_lstm,y_valid))


print('%-15s'%('Model'),end='')

for key,value in scores['cnn'].items():

    print ('%-15s'%(key),end='')

print('\r\n---------------------------------------------------------------------------------------------------',end='')

for key, value in scores.items():

    print( '\r\n%-15s' %(key),end='')

    for k, val in value.items():

        print('%-15f'%(val),end='')

    
from keras.models import Sequential

from keras.layers import Conv1D, MaxPooling1D,GlobalAveragePooling1D

from keras.layers import Dropout, Flatten, Dense,TimeDistributed,LSTM

from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

n_steps, n_length = 4, 25



def create_model(n_dense=100,n_filters=64,kernel_size=3,pool_size=2,n_steps=4,n_length=25):

    

    model = Sequential()

    model.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu'), input_shape=(n_steps,n_length,n_features)))

    model.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu')))

    model.add(TimeDistributed(Dropout(0.5)))

    model.add(TimeDistributed(MaxPooling1D(pool_size=pool_size)))

    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(n_dense))

    model.add(Dropout(0.5))

    model.add(Dense(n_dense, activation='relu'))

    model.add(Dense(n_outputs, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', weighted_metrics=[f1_m,'acc',precision_m, recall_m])

    return model
# define the grid search parameters

param_grid = {

    #'n_dense': [100],

    'n_filters': [32, 64],

    'kernel_size': [3, 6,8],

    'pool_size': [4,8],

    #'n_dense':[25,100]

    #'optimizer':['RMSprop', 'Adam', 'Adamax', 'sgd'],

}
x_train_cnn_lstm = x_train.reshape((x_train.shape[0], n_steps, n_length, n_features))

x_valid_cnn_lstm = x_valid.reshape((x_valid.shape[0], n_steps, n_length, n_features))

x_test_cnn_lstm = x_test.reshape((x_test.shape[0], n_steps, n_length, n_features))
# fix random seed for reproducibility

seed = 7

np.random.seed(seed)

kears_estimator = KerasClassifier(build_fn=create_model,epochs=20, verbose=1)



grid = GridSearchCV(estimator=kears_estimator,   

                    verbose=1,

                    n_jobs=1,

                    return_train_score=True,

                    param_grid=param_grid,)



grid_result = grid.fit(x_train_cnn_lstm[0:1000],y_train[0:1000],validation_data=(x_valid_cnn_lstm,y_valid)) 
scores_df = pd.DataFrame(grid_result.cv_results_).sort_values(by='rank_test_score')

result_df =scores_df[['mean_test_score','std_test_score','param_kernel_size','param_n_filters','param_pool_size']]

result_df.columns = ['accurancy','loss','kernel_size','n_filter','pool_size']

result_df.head(100)
# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print('%10s %s %10s %10s %10s' % ('kernel_size','n_filters','pool_size'))

    print('%10f %10f %10f %10f %10f '(mean,stddef,param['kernel_size'],param['pool_size']))

    #print("%f (%f) with: %r" % (mean, stdev, param))
final_model = create_model(kernel_size=8,n_filters=64,pool_size=4)

train_model(final_model,'final',training_data=(x_train_cnn_lstm,y_train),validation_data=(x_valid_cnn_lstm,y_valid),epochs=20,plot=True)

evaluate_model(final_model,'final',validation_data=(x_valid_cnn_lstm,y_valid))
evaluate_model(final_model,'final',validation_data=(x_test_cnn_lstm,y_test))
scores['final']
y_predicted = final_model.predict(x_test_cnn_lstm)



x_visualize=[]

y_vis_predicted=[]

y_true = []



classes_tp = [0,1,2,3]#+[0,1,2,3]+[0,1,2,3]

classes_tn =  [0,1,2,3]



for el in enumerate(y_predicted):

    i = el[0]

    if (np.argmax(y_test[i]) != np.argmax(y_predicted[i]) and np.argmax(y_test[i]) in classes_tn):

        y_vis_predicted.append(np.argmax(y_predicted[i]))

        y_true.append(np.argmax(y_test[i]))

        x_visualize.append(x_test[i])

        classes_tn.remove(np.argmax(y_test[i]))

    if (np.argmax(y_test[i]) == np.argmax(y_predicted[i]) and np.argmax(y_test[i]) in classes_tp):

        y_vis_predicted.append(np.argmax(y_predicted[i]))

        y_true.append(np.argmax(y_test[i]))

        x_visualize.append(x_test[i])

        classes_tp.remove(np.argmax(y_test[i]))
import numpy as np

import matplotlib.pylab as pl

import matplotlib.gridspec as gridspec



colors =['red','blue', 'green', 'gray']

labels =ACT_LABELS[0:4] + ['false classified: '+l for l in ACT_LABELS[0:4]]

# Create 2x2 sub plots

gs = gridspec.GridSpec(3, 1,hspace=0.4)



pl.figure(figsize=(20,15))



ax1 = pl.subplot(gs[0, 0],title='userAcceleration.x',ylabel='value',xlabel='timestep') # row 0, col 0

ax2 = pl.subplot(gs[1, 0],title='userAcceleration.y',ylabel='value',xlabel='timestep') # row 0, col 1

ax3 = pl.subplot(gs[2, 0],title='userAcceleration.z',ylabel='value',xlabel='timestep') # row 1, span all columns





for ts in enumerate(x_visualize):

    x,y,z = zip(*ts[1])

    linestyle = '-'

    if y_true[ts[0]] != y_vis_predicted[ts[0]]:

        linestyle = '--'

        label = 4+y_true[ts[0]]

    else:

        label = y_true[ts[0]]

    ax1.plot(x,color=colors[y_true[ts[0]]],linestyle=linestyle,label=labels[label])

    ax2.plot(y,color=colors[y_true[ts[0]]],linestyle=linestyle,label=labels[label])

    ax3.plot(y,color=colors[y_true[ts[0]]],linestyle=linestyle,label=labels[label])

ax1.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0. )
#imports we know we'll need

import skopt

# !pip install scikit-optimize if  necessary

from skopt import gbrt_minimize, gp_minimize

from skopt.utils import use_named_args

from skopt.space import Real, Categorical, Integer  



import keras

from keras.models import Sequential

from keras.layers import Dense

import tensorflow

from tensorflow.python.keras import backend as K
@use_named_args(dimensions=dimensions)

def fitness(n_dense,n_filters,kernel_size,pool_size):

    n_steps=4

    n_length=25

    model = create_model(n_dense=n_dense,n_filters=n_filters,kernel_size=kernel_size,pool_size=pool_size,n_steps=n_steps,n_length=n_length)

    

    x_train_cnn_lstm = x_train.reshape((x_train.shape[0], n_steps, n_length, n_features))

    x_valid_cnn_lstm = x_valid.reshape((x_valid.shape[0], n_steps, n_length, n_features))

    #named blackbox becuase it represents the structure

    blackbox = model.fit(x=x_train_cnn_lstm,

                        y=y_train,

                        epochs=3,

                        batch_size=3,

                        validation_data=(x_valid_cnn_lstm,y_valid),

                        )

    #return the validation accuracy for the last epoch.

    accuracy = blackbox.history['val_acc'][-1]



    # Print the classification accuracy.

    print()

    print("Accuracy: {0:.2%}".format(accuracy))

    print()





    # Delete the Keras model with these hyper-parameters from memory.

    del model

    

    # Clear the Keras session, otherwise it will keep adding new

    # models to the same TensorFlow graph each time we create

    # a model with a different set of hyper-parameters.

    K.clear_session()

    tensorflow.reset_default_graph()

    

    # the optimizer aims for the lowest score, so we return our negative accuracy

    return -accuracy
n_dense = Integer(low=1, high=200,  name='n_dense')

n_filters = Integer(low=1, high=128, name='n_filters')

kernel_size = Integer(low=1, high=16, name='kernel_size')

pool_size = Integer(low=1, high=32, name='pool_size')





dimensions = [n_dense,

              n_filters,

              kernel_size,

              pool_size,

             ]

default_parameters = [100,64,3,4]
kernel_size
gp_result = gp_minimize(func=fitness,

                        dimensions=dimensions,

                            n_calls=12,

                            noise= 0.01,

                            n_jobs=-1,

                            kappa = 5,

                            x0=default_parameters)
import numpy as np



from hyperopt import Trials, STATUS_OK, tpe

from keras.datasets import mnist

from keras.layers.core import Dense, Dropout, Activation

from keras.models import Sequential

from keras.utils import np_utils

#!pip install hyperas

from hyperas import optim

from hyperas.distributions import choice, uniform
def data():

    return x_train,y_train,x_valid,y_valid



def new_model(x_train,y_train,x_valid,y_valid):

    n_steps=4

    n_length=25

    model = create_model(n_dense={{choice([50, 100, 200])}},n_filters={{choice([16,32, 64,128])}},kernel_size={{choice([3,6, 8,16])}},pool_size={{choice([2,4, 8,16])}},n_steps=n_steps,n_length=n_length)

    

    x_train_cnn_lstm = x_train.reshape((x_train.shape[0], n_steps, n_length, n_features))

    x_valid_cnn_lstm = x_valid.reshape((x_valid.shape[0], n_steps, n_length, n_features))

    #named blackbox becuase it represents the structure

    blackbox = model.fit(x=x_train_cnn_lstm,

                        y=y_train,

                        epochs=20,

                        batch_size=20,

                        validation_data=(x_valid_cnn_lstm,y_valid),

                        )

    #get the highest validation accuracy of the training epochs

    validation_acc = np.amax(result.history['val_acc']) 

    print('Best validation acc of epoch:', validation_acc)

    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}
    best_run, best_model = optim.minimize(model=new_model,

                                          data=data,

                                          algo=tpe.suggest,

                                          max_evals=5,

                                          notebook_name='kernel762afcce6b',

                                          trials=Trials())
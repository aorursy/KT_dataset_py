import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Input, Dense, concatenate, Concatenate, PReLU
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
def load_data():
    path = '../input/mobile-sensors-and-directions'
    gens = None
    paths = os.listdir(path)
    for p in paths:
        p = path +'/'+ p
        for file in os.listdir(p):
            gen = pd.read_csv(p + '/' +file)
            if gens is None:
                gens = gen
            else:
                gens = pd.concat([gens, gen])
    return gens

gens = load_data()
gens.head()
gens.groupby(['tag'])['tag'].count()
def create_X_y(gens):
    gens = gens.dropna()
    X = gens.drop(['tag','Unnamed: 0'], axis=1)
    y = gens['tag']
    return X, y

X, y = create_X_y(gens)
def split_train_test(X, y, precentage = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=precentage)
    print('X train shape: ',X_train.shape, ' y train shape: ', y_train.shape)
    print('X test shape: ',X_test.shape, ' y test shape: ', y_test.shape)
    return X_train, X_test, y_train, y_test

def y_to_categorical(y_train, y_test):
    dic = {x:i for i,x in enumerate(y_train.unique())}
    names = [x for x in y_train.unique()]
    len_train = len(y_train)
    y = np.concatenate([y_train.to_numpy(), y_test.to_numpy()])
    y = [dic[x] for x in y]
    y = to_categorical(y)
    y_train = y[0:len_train]
    y_test = [dic[x] for x in y_test]
    print('Changed to catigorical -> ','y train shape: ',len(y_train), '-> y test shape: ', len(y_test))
    return y_train, y_test, dic, names

X_train, X_test, y_train, y_test = split_train_test(X, y)
y_train, y_test, y_dic, y_names = y_to_categorical(y_train, y_test)
BATCH_SIZE = 32
EPOCKS = 25
CLASSES = 9
FEATURES = 8
FEATURES_SENSOR = 3
POINTS = 5
def create_callbacks(name, patience=3):
    early_stopping = EarlyStopping(patience=patience)
    cheak_point = ModelCheckpoint(name)
    return [early_stopping, cheak_point]

def create_metrics():
    return ['accuracy']

def create_simple_model(size=FEATURES):
    inp = Input(shape=(None,size))
    x = Dense(32, activation='relu')(inp)
    x = Dense(64, activation='relu')(x)
    x = Dense(CLASSES, activation='softmax')(x)
    model = Model(inputs=inp, outputs=x, name='simple_model') 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=create_metrics())
    return model

simple_model = create_simple_model()
callbacks = create_callbacks('simple_model.h5')
history = simple_model.fit(X_train, y_train, epochs=EPOCKS, batch_size=BATCH_SIZE, validation_split=0.15, callbacks=callbacks)
def quick_plot_loss(history, field, metric, ax):
    # Plot training & validation loss values
    ax.plot(history.history[field])
    ax.plot(history.history['val_'+field])
    ax.set_title('Model '+ metric)
    ax.set_ylabel(metric)
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Validation'], loc='upper left')
                
    
def quick_plot_history(history):
    fig = plt.figure(figsize=(18, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('loss')
    quick_plot_loss(history, 'loss', 'categorical_crossentropy', ax)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('accrucy')
    quick_plot_loss(history, 'accuracy', 'accuracy', ax)
    
quick_plot_history(history)
def test_results(model,X_test,y_test, LOAD=True):
    preds = model.predict(X_test)
    pred_cat = np.argmax(preds,axis=1) #takes the maximum prediction and compare it to the real prediction
    acc = accuracy_score(y_test,pred_cat)*100
    acc_saved = -1
    if LOAD:
        saved_model = load_model('./'+model.name+'.h5')
        preds = saved_model.predict(X_test)
        pred_cat = np.argmax(preds,axis=1) #takes the maximum prediction and compare it to the real prediction
        acc_saved = accuracy_score(y_test,pred_cat)*100
    if acc >= acc_saved:
        if LOAD:
            model.save('./'+model.name+'.h5')
        print('NEW: model accuracy on test set is: {0:.2f}%'.format(acc))
    else:
        print('model accuracy on test set is: {0:.2f}%'.format(acc_saved))        

test_results(simple_model, X_test, y_test)
forest = RandomForestRegressor(n_estimators = 100, max_depth = 8)
forest.fit(X_train,y_train)
test_results(forest, X_test, y_test, LOAD=False)
def split_input_channels(X):
    acc = X.drop(['gyroscope_x','gyroscope_y','gyroscope_z','angle','diff'], axis=1)
    gyro = X.drop(['accelometer_x','accelometer_y','accelometer_z','angle','diff'], axis=1)
    angle = X.drop(['gyroscope_x','gyroscope_y','gyroscope_z','accelometer_x','accelometer_y','accelometer_z'], axis=1)
    return acc, gyro, angle


def create_siamese_model(acc_size, gyro_size, angle_size):
    acc_inp = Input(shape=acc_size)
    acc = Dense(32, activation='relu')(acc_inp)
    acc = Dense(64, activation='relu')(acc)
    
    gyro_inp = Input(shape=gyro_size)
    gyro = Dense(32, activation='relu')(gyro_inp)
    gyro = Dense(64, activation='relu')(gyro)
    
    angle_inp = Input(shape=angle_size)
    angle = Dense(32, activation='relu')(angle_inp)
    angle = Dense(64, activation='relu')(angle)
    
    x = concatenate([acc, gyro, angle])
    x = Dense(CLASSES, activation='softmax')(x)
    model = Model(inputs=[acc_inp, gyro_inp, angle_inp], outputs=x, name='siamese_model') 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=create_metrics())
    return model

siamese_model = create_siamese_model(FEATURES_SENSOR, FEATURES_SENSOR, 2)
acc, gyro, angle = split_input_channels(X_train)
callbacks = create_callbacks('siamese_model.h5')
history = siamese_model.fit([acc,gyro,angle], y_train, epochs=EPOCKS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=callbacks)
quick_plot_history(history)
acc, gyro, angle = split_input_channels(X_test)
test_results(siamese_model, [acc, gyro, angle], y_test)
def split_train_test_by_sliding_window(X,y, N=2):
    grouped = gens.groupby('tag')
    X_tmp = None
    X_original = None
    y_original = None
    start = N
    end = None
    for g in grouped:
        X, y = create_X_y(g[1])
        end = len(X)
        tmp_dic = {}
        if X_original is None:
            X_original = X.iloc[start:]
            y_original = y.iloc[start:]
        else:
            X_original = pd.concat([X_original,X.iloc[start:]])
            y_original = pd.concat([y_original, y.iloc[start:]])
        for col in X.columns:
            for i in range(1,N,1):
                name = 'prev_'+str(i)+'_'+col

                tmp_dic[name] = X[col].iloc[start-i: end-i].values
        if X_tmp is None:
            X_tmp = pd.DataFrame.from_dict(tmp_dic)
        else:
            X_tmp = pd.concat([X_tmp, pd.DataFrame.from_dict(tmp_dic)])
    X_original = X_original.reset_index()
    X_tmp = X_tmp.reset_index()
    X_tmp = pd.concat([X_tmp, X_original], axis=1)
    X_tmp = X_tmp.drop(['index'], axis=1)
    print('X devide: ',X_tmp.shape, ' y devide: ', y_original.shape)
    X_train, X_test, y_train, y_test = split_train_test(X_tmp, y_original)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_train_test_by_sliding_window(X, y, N=POINTS)
y_train, y_test, y_dic, y_names = y_to_categorical(y_train, y_test)
X_train.head()
def sp_acc_txt(prev=None):
    if prev is None:
        return ['accelometer_x','accelometer_y','accelometer_z']
    name = 'prev_'+str(prev)+'_'
    return [name+'accelometer_x',name+'accelometer_y',name+'accelometer_z']

def split_acc(points,index=None):
    l = []
    if index is None or index != 0:
        l = sp_acc_txt()
    for i in range(1,points):
        if index is None or i != index:
            l += sp_acc_txt(prev=i)
    return l

def sp_gyro_txt(prev=None):
    if prev is None:
        return ['gyroscope_x','gyroscope_y','gyroscope_z']
    name = 'prev_'+str(prev)+'_'
    return [name+'gyroscope_x',name+'gyroscope_y',name+'gyroscope_z']

def split_gyro(points,index=None):
    l = []
    if index is None or index != 0:
        l = sp_gyro_txt()
    for i in range(1,points):
        if index is None or i != index:
            l += sp_gyro_txt(prev=i)
    return l

def sp_angle_txt(prev=None):
    if prev is None:
        return ['angle','diff']
    name = 'prev_'+str(prev)+'_'
    return [name+'angle',name+'diff']

def split_angle(points,index=None):
    l = []
    if index is None or index != 0:
        l = sp_angle_txt()
    for i in range(1,points):
        if index is None or i != index:
            l += sp_angle_txt(prev=i)
    return l 

def split_channels(X, p=5):
    t = ()
    l = None
    acc = None
    gyto = None
    angle = None
    for i in range(p):
        l = split_acc(p, index=i) + split_gyro(p) + split_angle(p)
        acc = X.drop(l, axis=1)
        l = split_acc(p) + split_gyro(p, index=i) + split_angle(p)
        gyro = X.drop(l, axis=1)
        l = split_acc(p) + split_gyro(p) + split_angle(p, index=i)
        angle = X.drop(l, axis=1)
        t += (acc , gyro, angle) 
    return t
def dense_depth_block(inp, N=32, extend_N=4, deapth=5):
    if deapth==0:
        return inp
    x = None 
    prev_x = None
    xs = []
    for i in range(0, deapth):
        if x is None:
            x = Dense(N)(inp)
            x = PReLU()(x)
        else:
            prev_x = x
            x = Dense(N + i*extend_N)(prev_x)
            x = PReLU()(x)
        xs.append(x)
    if len(xs) != 1:
        x = concatenate(xs)
    return x

def create_chanel(size=FEATURES_SENSOR, deapth=5, N=32,extend_N=16):
    inp = Input(shape=size)
    x = dense_depth_block(inp, deapth=deapth, N=N, extend_N=extend_N)
    return inp, x
def create_complex_siamese_model(p=POINTS):
    inputs= []
    cs = []
    for i in range(p):
        inp_acc, c_acc = create_chanel(size=3,deapth=2,N=16)
        inputs.append(inp_acc)
        cs.append(c_acc)
        inp_gyro, c_gyro = create_chanel(size=3,deapth=2,N=16)
        inputs.append(inp_gyro)
        cs.append(c_gyro)
        inp_angle, c_angle = create_chanel(size=2,deapth=2,N=16)
        inputs.append(inp_angle)
        cs.append(c_angle)
    x = concatenate(cs)
    x = Dense(256)(x)
    x = PReLU()(x)
    x = Dense(128)(x)
    x = PReLU()(x)
    x = Dense(CLASSES, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x, name='complex_siamese_'+str(p)+'_model') 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=create_metrics())
    return model, 'complex_siamese_'+str(p)+'_model'
POINTS = 2
X_train, X_test, y_train, y_test = split_train_test_by_sliding_window(X, y, N=POINTS)
y_train, y_test, y_dic, y_names = y_to_categorical(y_train, y_test)
complex_siamese_model, name = create_complex_siamese_model(p=POINTS)
callbacks = create_callbacks(name+'.h5')
history = complex_siamese_model.fit(split_channels(X_train, p=POINTS), y_train, epochs=EPOCKS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=callbacks)
quick_plot_history(history)
test_results(complex_siamese_model, split_channels(X_test, p=POINTS), y_test)
def plot_confusion_matrix(y_test, preds, y_names):
    con = confusion_matrix(y_test, preds)
    con = con / np.sum(con, axis=1)
    plt.figure(figsize=(15,10), dpi=50)
    sns.set(font_scale=1.5)
    sns.heatmap(con, xticklabels=y_names, yticklabels=y_names, linewidths=2, annot=True, fmt = '.1%',cmap="YlGnBu",square=True)
    plt.xlabel('Predictions')
    plt.ylabel('Accpected')
    
preds = complex_siamese_model.predict(split_channels(X_test, p=POINTS))
preds = np.argmax(preds,axis=1)
plot_confusion_matrix(y_test, preds, y_names)
POINTS = 5
X_train, X_test, y_train, y_test = split_train_test_by_sliding_window(X, y, N=POINTS)
y_train, y_test, y_dic, y_names = y_to_categorical(y_train, y_test)
complex_siamese_model, name = create_complex_siamese_model(p=POINTS)
callbacks = create_callbacks(name+'.h5')
history = complex_siamese_model.fit(split_channels(X_train, p=POINTS), y_train, epochs=EPOCKS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=callbacks)
quick_plot_history(history)
test_results(complex_siamese_model, split_channels(X_test, p=POINTS), y_test)
preds = complex_siamese_model.predict(split_channels(X_test, p=POINTS))
preds = np.argmax(preds,axis=1)
plot_confusion_matrix(y_test, preds, y_names)
POINTS = 9
X_train, X_test, y_train, y_test = split_train_test_by_sliding_window(X, y, N=POINTS)
y_train, y_test, y_dic, y_names = y_to_categorical(y_train, y_test)
complex_siamese_model, name = create_complex_siamese_model(p=POINTS)
callbacks = create_callbacks(name+'.h5')
history = complex_siamese_model.fit(split_channels(X_train, p=POINTS), y_train, epochs=EPOCKS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=callbacks)
quick_plot_history(history)
test_results(complex_siamese_model, split_channels(X_test, p=POINTS), y_test)
preds = complex_siamese_model.predict(split_channels(X_test, p=POINTS))
preds = np.argmax(preds,axis=1)
plot_confusion_matrix(y_test, preds, y_names)
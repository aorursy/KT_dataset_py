import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from IPython.display import display
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Dropout
def load_activity_map():
    map = {}
    map[0] = 'transient'
    map[1] = 'lying'
    map[2] = 'sitting'
    map[3] = 'standing'
    map[4] = 'walking'
    map[5] = 'running'
    map[6] = 'cycling'
    map[7] = 'Nordic_walking'
    map[9] = 'watching_TV'
    map[10] = 'computer_work'
    map[11] = 'car driving'
    map[12] = 'ascending_stairs'
    map[13] = 'descending_stairs'
    map[16] = 'vacuum_cleaning'
    map[17] = 'ironing'
    map[18] = 'folding_laundry'
    map[19] = 'house_cleaning'
    map[20] = 'playing_soccer'
    map[24] = 'rope_jumping'
    return map
def generate_three_IMU(name):
    x = name +'_x'
    y = name +'_y'
    z = name +'_z'
    return [x,y,z]

def generate_four_IMU(name):
    x = name +'_x'
    y = name +'_y'
    z = name +'_z'
    w = name +'_w'
    return [x,y,z,w]

def generate_cols_IMU(name):
    # temp
    temp = name+'_temperature'
    output = [temp]
    # acceleration 16
    acceleration16 = name+'_3D_acceleration_16'
    acceleration16 = generate_three_IMU(acceleration16)
    output.extend(acceleration16)
    # acceleration 6
    acceleration6 = name+'_3D_acceleration_6'
    acceleration6 = generate_three_IMU(acceleration6)
    output.extend(acceleration6)
    # gyroscope
    gyroscope = name+'_3D_gyroscope'
    gyroscope = generate_three_IMU(gyroscope)
    output.extend(gyroscope)
    # magnometer
    magnometer = name+'_3D_magnetometer'
    magnometer = generate_three_IMU(magnometer)
    output.extend(magnometer)
    # oreintation
    oreintation = name+'_4D_orientation'
    oreintation = generate_four_IMU(oreintation)
    output.extend(oreintation)
    return output

def load_IMU():
    output = ['time_stamp','activity_id', 'heart_rate']
    hand = 'hand'
    hand = generate_cols_IMU(hand)
    output.extend(hand)
    chest = 'chest'
    chest = generate_cols_IMU(chest)
    output.extend(chest)
    ankle = 'ankle'
    ankle = generate_cols_IMU(ankle)
    output.extend(ankle)
    return output
    
def load_subjects(root='../input/ass2-time-series/PAMAP2_Dataset/Protocol/subject'):
    output = pd.DataFrame()
    cols = load_IMU()
    
    for i in range(101,110):
        path = root + str(i) +'.dat'
        subject = pd.read_table(path, header=None, sep='\s+')
        subject.columns = cols 
        subject['id'] = i
        output = output.append(subject, ignore_index=True)
    output.reset_index(drop=True, inplace=True)
    return output

data = load_subjects()
data
def fix_data(data):
    data = data.drop(data[data['activity_id']==0].index)
    data = data.interpolate()
    # fill all the NaN values in a coulmn with the mean values of the column
    for colName in data.columns:
        data[colName] = data[colName].fillna(data[colName].mean())
    activity_mean = data.groupby(['activity_id']).mean().reset_index()
    return data

data = fix_data(data)
data.describe()
print('Size of the data: ', data.size)
print('Shape of the data: ', data.shape)
print('Number of columns in the data: ', len(data.columns))
result_id = data.groupby(['id']).mean().reset_index()
print('Number of uniqe ids in the data: ', len(result_id))
result_act = data.groupby(['activity_id']).mean().reset_index()
print('Numbe of uniqe activitys in the data: ',len(result_act))
def pd_fast_plot(pd,column_a,column_b,title, figsize=(10,6)):
    plt.rcParams.update({'font.size': 16})
    size = range(len(pd))
    f, ax = plt.subplots(figsize=figsize) 
    plt.bar(size, pd[column_a], color=plt.cm.Paired(size))
    a = ax.set_xticklabels(pd[column_b])
    b = ax.legend(fontsize = 20)
    c = ax.set_xticks(np.arange(len(pd)))
    d = ax.set_title(title)
    plt.show()
    
sampels = data.groupby(['id']).count().reset_index()
sampels_to_subject = pd.DataFrame()
sampels_to_subject['id'] = sampels['id']
sampels_to_subject['sampels'] = sampels['time_stamp']
sampels_to_subject = sampels_to_subject.sort_values(by=['sampels'])
pd_fast_plot(sampels_to_subject,'sampels','id','Number Of Samepls By Users')
map_ac = load_activity_map()
sampels = data.groupby(['activity_id']).count().reset_index()
sampels_to_subject = pd.DataFrame()
sampels_to_subject['activity'] = [map_ac[x] for x in sampels['activity_id']]
sampels_to_subject['sampels'] = sampels['time_stamp']
sampels_to_subject = sampels_to_subject.sort_values(by=['sampels'])
pd_fast_plot(sampels_to_subject,'sampels','activity','Number Of Samepls By Activity',figsize=(40,7))
sampels_heart_rate = pd.DataFrame()
sampels_heart_rate['id'] = result_id['id']
sampels_heart_rate['heart_rate'] = result_id['heart_rate']
sampels_heart_rate = sampels_heart_rate.sort_values(by=['heart_rate'])
pd_fast_plot(sampels_heart_rate,'heart_rate','id','Avg heart Rate by Subject')
map_ac = load_activity_map()
sampels_heart_rate = pd.DataFrame()
sampels_heart_rate['activity'] = [map_ac[x] for x in result_act['activity_id']]
sampels_heart_rate['heart_rate'] = result_act['heart_rate']
sampels_heart_rate = sampels_heart_rate.sort_values(by=['heart_rate'])
pd_fast_plot(sampels_heart_rate,'heart_rate','activity','Avg heart Rate by Activity',figsize=(40,10))
samepls_tempreture = pd.DataFrame()
samepls_tempreture['hand'] = result_id['hand_temperature']
samepls_tempreture['chest'] = result_id['chest_temperature']
samepls_tempreture['ankle'] = result_id['ankle_temperature']

ax = samepls_tempreture.plot(kind='line', figsize=(20,6), title='Avg Tempatures by Subjects')
a = ax.set_xticklabels(result_id['id'])
b = ax.legend(fontsize = 20)
c = ax.set_xticks(np.arange(len(samepls_tempreture)))
map_ac = load_activity_map()
samepls_tempreture = pd.DataFrame()
samepls_tempreture['activity'] = [map_ac[x] for x in result_act['activity_id']]
samepls_tempreture['hand'] = result_act['hand_temperature']
samepls_tempreture['chest'] = result_act['chest_temperature']
samepls_tempreture['ankle'] = result_act['ankle_temperature']

ax = samepls_tempreture.plot(kind='line', figsize=(40,6), title='Avg Tempatures by Activity')
a = ax.set_xticklabels(samepls_tempreture['activity'])
b = ax.legend(fontsize = 20)
c = ax.set_xticks(np.arange(len(samepls_tempreture)))
samepls = pd.DataFrame()
samepls['hand_x'] = result_id['hand_3D_acceleration_16_x']
samepls['hand_y'] = result_id['hand_3D_acceleration_16_y']
samepls['hand_z'] = result_id['hand_3D_acceleration_16_z']
ax = samepls.plot(kind='line', figsize=(20,6), title='Avg Hand Acceleration Value by Subjects')
a = ax.set_xticklabels(result_id['id'])
b = ax.legend(fontsize = 20)
c = ax.set_xticks(np.arange(len(samepls)))
from sklearn.preprocessing import MinMaxScaler

def split_train_test(data):
    # create the test data
    subject107 = data[data['id'] == 107]
    subject108 = data[data['id'] == 108]
    test = subject107.append(subject108)

    # create the train data
    train = data[data['id'] != 107]
    train = data[data['id'] != 108]

    # drop the columns id and time
    test = test.drop(["id"], axis=1)
    train = train.drop(["id"], axis=1)

    # split train and test to X and y
    X_train = train.drop(['activity_id','time_stamp'], axis=1).values
    X_test = test.drop(['activity_id','time_stamp'], axis=1).values
    
    # make data scale to min max beetwin 0 to 1
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X_train)
    min_max_scaler.fit(X_test)
    X_train = min_max_scaler.transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    
    y_train = train['activity_id'].values
    y_test = test['activity_id'].values
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_train_test(data)
print('Train shape X :',X_train.shape,' y ', y_train.shape)
print('Test shape X :',X_test.shape,' y ', y_test.shape)
from sklearn.preprocessing import MinMaxScaler

def base_line_preprocess(data):
    train = data.groupby('activity_id')
    X_base = train.mean().reset_index().drop(['activity_id','id','time_stamp'],axis=1).values
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X_base)
    X_base = min_max_scaler.transform(X_base)
    y_base = np.array(train['activity_id'].unique().explode().values).astype('float64')
    return X_base, y_base

X_base, y_base = base_line_preprocess(data)
print('X base shape: ', X_base.shape)
print('y base shape: ', y_base.shape)

reg = LogisticRegression()
reg.fit(X_base, y_base)
preds = reg.predict(X_train)
print('Logistic regression accuracy on train: ', accuracy_score(y_train,preds)*100)
preds = reg.predict(X_test)
print('Logistic regression accuracy on test: ', accuracy_score(y_test,preds)*100)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
# predict on the train and the test
preds = dt.predict(X_train)
print('Decision Tree Regressor accuracy on train: ', accuracy_score(y_train,preds)*100)
preds = dt.predict(X_test)
print('Decision Tree Regressor accuracy on test: ', accuracy_score(y_test,preds)*100)
preds = 0
def create_lstm_data(X, y, step_back = 5, step_forword = 1):
    out_X = []
    out_y = []
    size = len(X)
    for i, features in enumerate(X):
        if i >= step_back and i < size - step_forword:
            tmp_X = []
            tmp_y = []
            for j in range(i-step_back,i):
                tmp_X.extend([X[j]])
            out_X.append(tmp_X)
            for j in range(i,i+step_forword):
                tmp_y.extend([y[j]])
            out_y.append(tmp_y)
    return np.array(out_X), np.array(out_y)

X_lstm_train, y_lstm_train = create_lstm_data(X_train, y_train)
X_lstm_test, y_lstm_test = create_lstm_data(X_test, y_test)
from sklearn.preprocessing import OneHotEncoder
hot = OneHotEncoder(handle_unknown='ignore', sparse=False)
hot.fit(y_lstm_train)
hot.fit(y_lstm_test)

y_lstm_train = hot.transform(y_lstm_train)
y_lstm_test = hot.transform(y_lstm_test)
print('Train shape X lstm :',X_lstm_train.shape,' y ', y_lstm_train.shape)
print('Test shape X lstm :',X_lstm_test.shape,' y ', y_lstm_test.shape)

lstm_model = Sequential()
lstm_model.add(LSTM(6,input_shape=(X_lstm_train.shape[1],X_lstm_train.shape[2])))
lstm_model.add(Dense(16 ,activation='relu'))
lstm_model.add(Dense(y_lstm_train.shape[1], activation='softmax'))

lstm_model.summary()
lstm_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=3)
history = lstm_model.fit(X_lstm_train, y_lstm_train, validation_split = 0.2 , epochs = 10, callbacks=[early_stopping_monitor])
# taken from https://keras.io/visualization/
def quick_plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

quick_plot_history(history)
y = y_test[5:-1]
preds = lstm_model.predict(X_lstm_test)
preds_cat = np.argmax(preds,axis=1)
# building a map of result to activity
result = np.unique(preds_cat).tolist() 
expected = np.unique(y).tolist() 
combined = list(zip(result,expected))
conf_map = dict(combined)
# transfoms the prediction to an activity
results = [conf_map[x] for x in preds_cat]
print('model accuracy on test :',accuracy_score(y,results)*100)
def quick_plot_con_matrix(y, results,labels):
    # now print confusion metrix
    con = confusion_matrix(y,results)
    a = plt.figure(figsize=(25,15), dpi=50)
    a = sns.heatmap(con, cmap='YlGnBu', annot=True, fmt = 'd', xticklabels=labels, yticklabels=labels)
    a = plt.rcParams.update({'font.size': 20})
    a = plt.title('Confusion Matrix')
    a = plt.xlabel('Predictions')
    a = plt.ylabel('Accpected')

activity_map = load_activity_map()
labels = [activity_map[conf_map[x]] for x in result]
quick_plot_con_matrix(y,results, labels)
def create_lstm_data(X, y, step_back = 5, step_forword = 1):
    out_X = []
    out_y = []
    size = len(X)
    for i, features in enumerate(X):
        if i >= step_back and i < size - step_forword:
            tmp_X = []
            tmp_y = []
            for j in range(i-step_back,i):
                tmp_X.extend([X[j]])
            out_X.append(tmp_X)
            for j in range(i,i+step_forword):
                tmp_y.extend([y[j]])
            out_y.append(tmp_y)
    return np.array(out_X), np.array(out_y)
# Freeze the layers 
for i,layer in enumerate(lstm_model.layers):
    if i < 1:
        print(layer)
        layer.trainable = False
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=3)
history = lstm_model.fit(X_lstm_train, y_lstm_train, validation_split = 0.2 , epochs = 10, callbacks=[early_stopping_monitor])
quick_plot_history(history)
y = y_test[5:-1]
preds = lstm_model.predict(X_lstm_test)
preds_cat = np.argmax(preds,axis=1)
# building a map of result to activity
result = np.unique(preds_cat).tolist() 
expected = np.unique(y).tolist() 
combined = list(zip(result,expected))
conf_map = dict(combined)
# transfoms the prediction to an activity
results = [conf_map[x] for x in preds_cat]
print('model accuracy on test :',accuracy_score(y,results)*100)
new_data = data.copy().reset_index()
# new_data = new_data.drop('index',axis=1)
# new_data = new_data.reset_index()
new_cols = None 
for subject in range(101,110):
    prev_act_1 = new_data[new_data['id'] == subject]
    start = prev_act_1.head(2).index[1]
    end = prev_act_1.tail(1).index[0]
    prev_act_1 = prev_act_1.loc[start:end+1]
    new_cols_1 = pd.DataFrame()
    new_cols_1['prev_aid'] = prev_act_1['activity_id']
    new_cols_1['prev_hr'] = prev_act_1['heart_rate']
    new_cols_1['index'] = prev_act_1['index'] + 1
    if new_cols is None:
        new_cols = new_cols_1
    else:
        new_cols = new_cols.append(new_cols_1)
new_cols = new_data.merge(new_cols, on='index', how='left')
new_cols = new_cols.dropna()
from sklearn.preprocessing import OneHotEncoder

X_train, X_test, y_train, y_test = split_train_test(new_cols)
print('Train shape X :',X_train.shape,' y ', y_train.shape)
print('Test shape X :',X_test.shape,' y ', y_test.shape)

X_lstm_train, y_lstm_train = create_lstm_data(X_train, y_train)
X_lstm_test, y_lstm_test = create_lstm_data(X_test, y_test)
hot = OneHotEncoder(handle_unknown='ignore', sparse=False)
hot.fit(y_lstm_train)
hot.fit(y_lstm_test)

y_lstm_train = hot.transform(y_lstm_train)
y_lstm_test = hot.transform(y_lstm_test)
print('Train shape X lstm :',X_lstm_train.shape,' y ', y_lstm_train.shape)
print('Test shape X lstm :',X_lstm_test.shape,' y ', y_lstm_test.shape)

from keras.layers import Dropout
lstm_model = Sequential()
lstm_model.add(LSTM(16,input_shape=(X_lstm_train.shape[1],X_lstm_train.shape[2])))
lstm_model.add(Dense(64 ,activation='relu'))
lstm_model.add(Dense(64 ,activation='relu'))
lstm_model.add(Dropout(0.1))
lstm_model.add(Dense(64 ,activation='relu'))
lstm_model.add(Dense(64 ,activation='relu'))
lstm_model.add(Dense(y_lstm_train.shape[1], activation='softmax'))

lstm_model.summary()
lstm_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=3)
history = lstm_model.fit(X_lstm_train, y_lstm_train, validation_split = 0.2 , epochs = 10, callbacks=[early_stopping_monitor])
# taken from https://keras.io/visualization/
def quick_plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

quick_plot_history(history)
y = y_test[5:-1]
preds = lstm_model.predict(X_lstm_test)
preds_cat = np.argmax(preds,axis=1)
# building a map of result to activity
result = np.unique(preds_cat).tolist() 
expected = np.unique(y).tolist() 
combined = list(zip(result,expected))
conf_map = dict(combined)
# transfoms the prediction to an activity
results = [conf_map[x] for x in preds_cat]
print('model accuracy on test :',accuracy_score(y,results)*100)
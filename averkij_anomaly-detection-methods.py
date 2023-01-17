!pip install pyreadr



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pyreadr

import seaborn as sns

import scipy.stats as stats

import torch

import lightgbm as lgb

from sklearn import preprocessing

from sklearn.metrics import mean_squared_error

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

import tensorflow as tf



sns.set()
train_normal_path = '/kaggle/input/tennessee-eastman-process-simulation-dataset/TEP_FaultFree_Training.RData'

train_fault_path = '/kaggle/input/tennessee-eastman-process-simulation-dataset/TEP_Faulty_Training.RData'



train_df = pyreadr.read_r(train_normal_path)['fault_free_training']

test_df = pyreadr.read_r(train_fault_path)['faulty_training']
train_df.head()
train_df.faultNumber.value_counts()
test_df.head()
test_df.faultNumber.value_counts()
def qqplot_by_fault(data_df, cols, fault_number):

    plt.figure(figsize=(14,14))

    

    for i in range(len(cols)):

        ax = plt.subplot(4, 4, i+1)

        data = data_df[(data_df.faultNumber==fault_number) & (data_df.simulationRun.isin(range(10)))][cols[i]]

        

        #data = np.log(data)

        

        stats.probplot(x=data, plot=plt)

        ax.set_title(cols[i])

        ax.set_xlabel('')

        ax.set_ylabel('')

        shapiro = stats.shapiro(data)

        ax.text(0.99, 0.01, '{0:.4f}'.format(shapiro[1]),

            verticalalignment='bottom', horizontalalignment='right',

            transform=ax.transAxes,

            color='green', fontsize=15)

        

    plt.show()
qqplot_by_fault(train_df, train_df.columns[3:19], fault_number=0)
qqplot_by_fault(train_df, train_df.columns[19:35], fault_number=0)
qqplot_by_fault(train_df, train_df.columns[35:51], fault_number=0)
qqplot_by_fault(train_df, train_df.columns[51:], fault_number=0)
qqplot_by_fault(test_df, test_df.columns[3:19], fault_number=1)
qqplot_by_fault(test_df, test_df.columns[3:19], fault_number=6)
qqplot_by_fault(test_df, test_df.columns[3:19], fault_number=17)
alpha = 0.05



def get_shapiro(data_df, cols, fault_number):

    shapiro_history = []

    for i in range(len(cols)):

        data = data_df[(data_df.faultNumber==fault_number) & (data_df.simulationRun.isin(range(9)))][cols[i]]

        #data = np.log(data)

        shapiro = stats.shapiro(data)

        shapiro_history.append(shapiro)

    return shapiro_history



def plot_shapiro(shapiro_history):

    W,p = zip(*shapiro_history)

    plt.plot(p)

    plt.axhline(y=alpha, color='r', linestyle='-')

    plt.title('p_value >= {0} for {1} features'.format(alpha, len([i for i in p if i >= alpha])))

    plt.xlabel('feature')

    plt.ylabel('p_value')

    plt.show()
faultless_shapiro = get_shapiro(train_df, train_df.columns[3:55], fault_number=0)

plot_shapiro(faultless_shapiro)



w,p_values = zip(*faultless_shapiro)

faultless_cond = pd.DataFrame(index=train_df.columns[3:55],columns=['isNormal']).fillna(False)

i=0

for p in p_values:

    if p >= 0.05:

        faultless_cond.iloc[i] = True

    i+=1

all_shap = []

for fault in range(1,21):

    shap = get_shapiro(test_df, test_df.columns[3:55], fault_number=fault)

    all_shap.append(shap)
fault_numbers = []

skewed_features = pd.DataFrame(index=train_df.columns[3:55],columns=['frequency']).fillna(0)



for i in range(20):

    w,p_values = zip(*all_shap[i])

    p_count = len([a for a in p_values if a >= 0.05])

    #print("%d -> %d" % (i + 1, p_count))

    if (p_count < 20):

        fault_numbers.append(i+1)

        

    feature_number = 0

    for p in p_values:

        if p < 0.05:

            skewed_features.iloc[feature_number] += 1

        feature_number+=1



#print("------------")

#print(len(fault_numbers))
faulty_shapiro = get_shapiro(test_df, train_df.columns[3:55], fault_number=2)

plot_shapiro(faulty_shapiro)
faulty_shapiro = get_shapiro(test_df, train_df.columns[3:55], fault_number=6)

plot_shapiro(faulty_shapiro)
faulty_shapiro = get_shapiro(test_df, train_df.columns[3:55], fault_number=20)

plot_shapiro(faulty_shapiro)
plt.figure(figsize=(12,6))



skewed_features[faultless_cond.isNormal.values].frequency.sort_values(ascending=False).plot(kind='bar');
n_prob_features=10



feature_cols = skewed_features[faultless_cond.isNormal].frequency.sort_values(ascending=False).head(n_prob_features).index.values

print(feature_cols)
scaler = preprocessing.MinMaxScaler()

data = pd.DataFrame(scaler.fit_transform(X = train_df[(train_df.faultNumber == 0) & (train_df.simulationRun.isin(range(400)))].loc[:,feature_cols]))



means = data.mean()

variances = data.std()



model_prob = pd.DataFrame(index = feature_cols, data = {'mean': means.values, 'variance': variances.values})

model_prob.head(n_prob_features)
def gaussian(x, mu, sig):

    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)



def get_probability(x, model):

    res = gaussian(x.values, model['mean'].values, model['variance'].values)

    return res.prod()



def alert_condition(x):

    return np.max(x)



def metric_prob(probs, window_minutes, eps):    

    #samples are sampled each 3 minutes

    window_points = window_minutes // 3

    rolled = probs.rolling(window=window_points).apply(alert_condition, raw=True)    

    ind = np.where(rolled < eps)

    

    if len(ind[0])>0:

        return ind[0][0]



    return 0
for fault in range(21):

    plt.figure(figsize=(12,4))

    if fault==0:

        scaled = pd.DataFrame(scaler.transform(train_df[(train_df.faultNumber == fault) & (train_df.simulationRun == 401)].loc[:, feature_cols]))

        plt.title('no faults')

    else:

        scaled = pd.DataFrame(scaler.transform(test_df[(test_df.faultNumber == fault) & (test_df.simulationRun == np.random.randint(500))].loc[:, feature_cols]))

        plt.title('fault type {0}'.format(fault))

    probs = scaled.apply(lambda x: get_probability(x, model_prob), axis=1)

    

    plt.plot(np.linspace(0,300,300), probs[:300])    

    plt.axvline(x=20, color='r', linestyle='-')



    #alert_point = metric_prob(probs, 30, 1e+05)

    #window_points = 10

    #plt.axvline(x=alert_point, color='y', linestyle='--')

    #plt.axvline(x=alert_point-window_points, color='y', linestyle='--')



    plt.show()
scaler = preprocessing.MinMaxScaler()



features_df = train_df[train_df.faultNumber==0].iloc[:,3:]

features_df = pd.DataFrame(scaler.fit_transform(X = features_df), columns = features_df.columns)
def train(df, cols_to_predict):

    models = {}

    for col in cols_to_predict:

        print('training model for', col)

        model = lgb.LGBMRegressor(learning_rate=0.1)

        tr_x = features_df.drop([col],axis=1)

        target = features_df[col]

        

        model.fit(X=tr_x, y=target)

        models[col] = model

    

    return models



def predict(models, df, cols_to_predict):

    preds = []

    for col in cols_to_predict:

        test_x = df.drop([col],axis=1)

        test_y = df[col]

        pred = models[col].predict(test_x)

        preds.append(pred)

        

        #err.append(np.square((test_y - pred)**2).values)

        

        #plt.figure(figsize=(15,10))

        #x = np.linspace(0,len(test_y),len(test_y))

        #plt.plot(x, test_y, label='Actual value')

        #plt.plot(x, pred, label='Prediction')

        #plt.legend(loc='best')

        #plt.title(col)

        #plt.plot()

        #plt.show()

        

        #qqplot_data(pred)

    

    return preds
features_to_predict = train_df.columns[3:]

models = train(train_df[(train_df.faultNumber==0) & (train_df.simulationRun.isin(range(400)))], features_to_predict)
def get_mse(sample, preds):

    return np.square((sample.loc[:,features_to_predict] - np.transpose(preds))**2).mean(axis=1)



plt.figure(figsize=(14,6))



normal_sample = pd.DataFrame(scaler.transform(train_df[(train_df.simulationRun==np.random.randint(500)) & (train_df.faultNumber==0)].iloc[:,3:]), columns = features_df.columns)

normal_preds = predict(models, normal_sample, features_to_predict)



faulty_sample = pd.DataFrame(scaler.transform(test_df[(test_df.simulationRun==np.random.randint(500)) & (test_df.faultNumber==1)].iloc[:,3:]), columns = features_df.columns)

faulty_preds = predict(models, faulty_sample, features_to_predict)



plt.axvline(x=20,color='r',linestyle='--')



plt.title('MSE for a normal and fault samples')

plt.plot(get_mse(normal_sample, normal_preds));

plt.plot(get_mse(faulty_sample, faulty_preds));
plt.figure(figsize=(14,6))

plt.yscale('log')



plt.axvline(x=20,color='r',linestyle='--')

plt.axhline(y=1e-2,color='g',linestyle='--')



plt.title('MSE for a normal and fault samples')

plt.plot(get_mse(normal_sample, normal_preds));

plt.plot(get_mse(faulty_sample, faulty_preds));
plt.figure(figsize=(16,10))

plt.yscale('log')



for i in [0,1,2,4,6,7,8]:

    if i == 0:

        tree_sample = train_df[(train_df.simulationRun==500) & (train_df.faultNumber==i)].iloc[:,3:]

    else:

        tree_sample = test_df[(test_df.simulationRun==np.random.randint(500)) & (test_df.faultNumber==i)].iloc[:,3:]

    tree_sample = pd.DataFrame(scaler.transform(tree_sample), columns = features_df.columns)

    tree_preds = predict(models, tree_sample, features_to_predict)

    plt.axvline(x=20,color='r',linestyle='--')

    if i==0:

        label='normal conditions'

    else:

        label='fault type %d' % i

    plt.plot(get_mse(tree_sample,tree_preds),label=label)

    plt.title('MSE for normal conditions and faults')

    plt.legend()
plt.figure(figsize=(16,10))

plt.yscale('log')



#11,12,13

for i in [0,5,14,17,18]:

    if i == 0:

        tree_sample = train_df[(train_df.simulationRun==500) & (train_df.faultNumber==i)].iloc[:,3:]

    else:

        tree_sample = test_df[(test_df.simulationRun==np.random.randint(500)) & (test_df.faultNumber==i)].iloc[:,3:]

    tree_sample = pd.DataFrame(scaler.transform(tree_sample), columns = features_df.columns)

    tree_preds = predict(models, tree_sample, features_to_predict)

    plt.axvline(x=20,color='r',linestyle='--')

    if i==0:

        label='normal conditions'

    else:

        label='fault type %d' % i

    plt.plot(get_mse(tree_sample,tree_preds),label=label)

    plt.title('MSE for normal conditions and faults')

    plt.legend()
plt.figure(figsize=(16,10))

#plt.yscale('log')



for i in [0,11,19,20]:

    if i == 0:

        tree_sample = train_df[(train_df.simulationRun==500) & (train_df.faultNumber==i)].iloc[:,3:]

    else:

        tree_sample = test_df[(test_df.simulationRun==np.random.randint(500)) & (test_df.faultNumber==i)].iloc[:,3:]

    tree_sample = pd.DataFrame(scaler.transform(tree_sample), columns = features_df.columns)

    tree_preds = predict(models, tree_sample, features_to_predict)

    plt.axvline(x=20,color='r',linestyle='--')

    if i==0:

        label='normal conditions'

    else:

        label='fault type %d' % i

    plt.plot(get_mse(tree_sample,tree_preds),label=label)

    plt.title('MSE for normal conditions and faults')

    plt.legend()
def series_to_lstm(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]

    df = pd.DataFrame(data)

    cols, names = list(), list()

    #columns = data.columns

    for i in range(n_in, 0, -1):

        cols.append(df.shift(i))

        names += [('%d(t-%d)' % (j, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)

    for i in range(0, n_out):

        cols.append(df.shift(-i))

        if i == 0:

            names += [('%d(t)' % (j)) for j in range(n_vars)]

        else:

            names += [('%d(t+%d)' % (j, i)) for j in range(n_vars)]

    # put it all together

    agg = pd.concat(cols, axis=1)

    agg.columns = names

    # drop rows with NaN values

    if dropnan:

        agg.dropna(inplace=True)

    return agg
scaler = preprocessing.MinMaxScaler()



dat = train_df[(train_df.faultNumber==0) & (train_df.simulationRun.isin(range(200)))]

dat = dat.iloc[:,3:]

dat = scaler.fit_transform(dat)

print(dat.shape)

#dat.head()



test = train_df[(train_df.faultNumber==0) & (train_df.simulationRun.isin(range(400,500)))]

test = test.iloc[:,3:]

test = scaler.transform(test)

print(test.shape)
time_steps = 10



ref = series_to_lstm(dat,time_steps,1)

print(ref.shape)



ref_test = series_to_lstm(test,time_steps,1)

print(ref_test.shape)
train_x = ref.values[:,:-52]

train_y = ref.values[:,-52:]

train_x = train_x.reshape(train_x.shape[0],time_steps,train_x.shape[1]//time_steps)



test_x = ref_test.values[:,:-52]

test_y = ref_test.values[:,-52:]

test_x = test_x.reshape(test_x.shape[0],time_steps,test_x.shape[1]//time_steps)
print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)
model_lstm = Sequential()

model_lstm.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))

model_lstm.add(Dense(52))

model_lstm.compile(loss='mse', optimizer='adam')



#with tf.device('/gpu:0'):

history = model_lstm.fit(train_x, train_y, epochs=25, batch_size=50, validation_data=(test_x, test_y), verbose=2, shuffle=False)



plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,10))

plt.yscale('log')



for i in [0,1,2,4,6,7]:    

    if i == 0:

        test_sample = train_df[(train_df.simulationRun==500) & (train_df.faultNumber==i)].iloc[:,3:]

    else:

        test_sample = test_df[(test_df.simulationRun==np.random.randint(500)) & (test_df.faultNumber==i)].iloc[:,3:]

        

    test_sample = scaler.transform(test_sample)

    test_sample = series_to_lstm(test_sample,time_steps,1)

    

    test_x = test_sample.iloc[:,:-52]

    test_y = test_sample.iloc[:,-52:]

    test_x = test_x.values.reshape(test_x.shape[0],time_steps,test_x.shape[1]//time_steps)

    

    pred = model_lstm.predict(test_x)    

    

    plt.axvline(x=20,color='r',linestyle='--')

    if i==0:

        label='normal conditions'

    else:

        label='fault type %d' % i

    plt.plot(np.square((test_y.iloc[:,:]-pred[:,:])**2).mean(axis=1), label=label)

    plt.legend()

    

plt.figure(figsize=(16,10))

plt.yscale('log')



for i in [0,5,14,17,18]:    

    if i == 0:

        test_sample = train_df[(train_df.simulationRun==500) & (train_df.faultNumber==i)].iloc[:,3:]

    else:

        test_sample = test_df[(test_df.simulationRun==np.random.randint(500)) & (test_df.faultNumber==i)].iloc[:,3:]

        

    test_sample = scaler.transform(test_sample)

    test_sample = series_to_lstm(test_sample,time_steps,1)

    

    test_x = test_sample.iloc[:,:-52]

    test_y = test_sample.iloc[:,-52:]

    test_x = test_x.values.reshape(test_x.shape[0],time_steps,test_x.shape[1]//time_steps)

    

    pred = model_lstm.predict(test_x)    

    

    plt.axvline(x=20,color='r',linestyle='--')

    if i==0:

        label='normal conditions'

    else:

        label='fault type %d' % i

    plt.plot(np.square((test_y.iloc[:,:]-pred[:,:])**2).mean(axis=1), label=label)

    plt.legend()
plt.figure(figsize=(16,10))



for i in [0,11,19,20]:

    if i == 0:

        test_sample = train_df[(train_df.simulationRun==500) & (train_df.faultNumber==i)].iloc[:,3:]

    else:

        test_sample = test_df[(test_df.simulationRun==np.random.randint(500)) & (test_df.faultNumber==i)].iloc[:,3:]

        

    test_sample = scaler.transform(test_sample)

    test_sample = series_to_lstm(test_sample,time_steps,1)

    

    test_x = test_sample.iloc[:,:-52]

    test_y = test_sample.iloc[:,-52:]

    test_x = test_x.values.reshape(test_x.shape[0],time_steps,test_x.shape[1]//time_steps)

    

    pred = model_lstm.predict(test_x)    

    

    plt.axvline(x=20,color='r',linestyle='--')

    if i==0:

        label='normal conditions'

    else:

        label='fault type %d' % i

    plt.plot(np.square((test_y.iloc[:,:]-pred[:,:])**2).mean(axis=1), label=label)
plt.figure(figsize=(16,10))

plt.yscale('log')



for i in [0,12,13]:    

    if i == 0:

        test_sample = train_df[(train_df.simulationRun==500) & (train_df.faultNumber==i)].iloc[:,3:]

    else:

        test_sample = test_df[(test_df.simulationRun==np.random.randint(500)) & (test_df.faultNumber==i)].iloc[:,3:]

        

    test_sample = scaler.transform(test_sample)

    test_sample = series_to_lstm(test_sample,time_steps,1)

    

    test_x = test_sample.iloc[:,:-52]

    test_y = test_sample.iloc[:,-52:]

    test_x = test_x.values.reshape(test_x.shape[0],time_steps,test_x.shape[1]//time_steps)

    

    pred = model_lstm.predict(test_x)    

    

    plt.axvline(x=20,color='r',linestyle='--')

    if i==0:

        label='normal conditions'

    else:

        label='fault type %d' % i

    plt.plot(np.square((test_y.iloc[:,:]-pred[:,:])**2).mean(axis=1), label=label)

    plt.legend()
plt.figure(figsize=(16,10))



for i in [0,19]:    

    if i == 0:

        test_sample = train_df[(train_df.simulationRun==500) & (train_df.faultNumber==i)].iloc[:,3:]

    else:

        test_sample = test_df[(test_df.simulationRun==np.random.randint(500)) & (test_df.faultNumber==i)].iloc[:,3:]

        

    test_sample = scaler.transform(test_sample)

    test_sample = series_to_lstm(test_sample,time_steps,1)

    

    test_x = test_sample.iloc[:,:-52]

    test_y = test_sample.iloc[:,-52:]

    test_x = test_x.values.reshape(test_x.shape[0],time_steps,test_x.shape[1]//time_steps)

    

    pred = model_lstm.predict(test_x)    

    

    plt.axvline(x=20,color='r',linestyle='--')

    if i==0:

        label='normal conditions'

    else:

        label='fault type %d' % i

    plt.plot(np.square((test_y.iloc[:,:]-pred[:,:])**2).mean(axis=1), label=label)

    plt.legend()
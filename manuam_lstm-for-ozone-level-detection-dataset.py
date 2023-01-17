import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from numpy.random import seed
def supervised_lag(data, max_lag):
    df = data.copy() # Avoids overwriting of the orginal df
    c = pd.DataFrame() # Empty dataframe
    for name in df.columns:
        for i in range(max_lag,0,-1): # Count backwards
            c[name+str(-i)]=df[name].shift(i) # Lag
        c[name] = df[name]
    
    c = c.dropna().reset_index(drop=True)
    # Reshape : number of observations, variables, lags +1 to include t
    
    return c
def ts_train_test(data, predictors, target, time):
    """
    predictors: list with the predictors' names
    targer: target variable name (string)
    time: length of test (int)
    """
    train_X = data[predictors][:-time]
    train_y = data[target][:-time]
    test_X = data[predictors][-time:]
    test_y = data[target][-time:]
    
    return train_X, train_y, test_X, test_y
def ts_lag(data, max_lag):
    df = data.copy() # Avoids overwriting of the orginal df
    c = pd.DataFrame() # Empty dataframe
    for name in df.columns:
        for i in range(max_lag,0,-1): # Count backwards
            c[name+str(-i)]=df[name].shift(i) # Lag
        c[name] = df[name]
    
    c = c.dropna().reset_index(drop=True)
    # Reshape : number of observations, variables, lags +1 to include t
    c = c.values.reshape(len(c),len(data.columns),max_lag+1) 
    
    # Above code reshape in horizontal, we need each column = variable -> Transpose
    l = [] 
    for i in range(0,len(c)):
        l.append(c[i].transpose())
    
    return np.array(l)
def ts_lag_test(train_original, test, max_lag):
    train = train_original.copy() # To avoid inplace errors.
    test = test.copy()
    c = pd.DataFrame() # Empty dataset to append the new lag variables in order.
    for name in test.columns: # Name of the variables.
        for i in range(max_lag,0,-1): # We want a sorted sequence t-2,t-1,t So we first compute de max lag and then go decreasing. 0 is not included.
            c[name+str(-i)]=test[name].shift(i)
            c[name+str(-i)][0:i] = train[name][len(train)-i:len(train)] # Replace the NA value with the last observation of the complete train.
        c[name] = test[name]
        
        
    c = c.dropna().reset_index(drop=True)
    # Reshape : number of observations, variables, lags +1 to include t
    c = c.values.reshape(len(c),len(train_original.columns),max_lag+1) 
    
    # Above code reshape in horizontal, we need each column = variable -> Transpose
    l = [] 
    for i in range(0,len(c)):
        l.append(c[i].transpose())
    
    return np.array(l)
def make_LSTM(train_X,train_y,test_X,test_y,units,epochs,batch_size,seed):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    model = Sequential()
    model.add(LSTM(units,
                      input_shape=(train_X.shape[1],train_X.shape[2]),kernel_initializer="he_uniform"
                  ))
    model.add(Dense(1,activation="sigmoid")) # Must be sigmoid for 0 - 1. 
    model.compile(loss="BinaryCrossentropy", optimizer="rmsprop", metrics=["AUC"])
    history=model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,
                         validation_data=(test_X,test_y), verbose=0, shuffle = False)
 
    return model, history
def evaluate_nn(train_X_R, test_X_R, test_y, seed=[303,305,3,1], units=16,epochs=50,batch_size=64):
    loss_list,auc_list=list(),list()
    for i in seed:
        model, _=make_LSTM(train_X_R, train_y, test_X_R, test_y, units, epochs, batch_size, seed=i)
        loss, auc = model.evaluate(test_X_R,test_y,verbose=1)
        loss_list.append(loss)
        auc_list.append(auc)
    return np.mean(loss_list),np.mean(auc_list)

def plot_loss(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.title("LOSS")
    plt.show()
    
def plot_auc(history):
    plt.plot(history.history['auc'], label='train')
    plt.plot(history.history['val_auc'], label='test')
    plt.legend()
    plt.title("AUC")
    plt.show()
    
def plot_roc(y_real, y_pred):
    false_positive_rate, recall, thresholds = roc_curve(y_real, y_pred)
    roc_auc = auc(false_positive_rate, recall)
    print(f'- AUC: {roc_auc}')
    print(f'- AVG Recall: {np.mean(recall)}')
    plt.plot(false_positive_rate, recall, 'b')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('ROC AUC = %0.2f' % roc_auc)
def tune_nn(units, epochs, batch_size):
    """
    # LIST format []
    """
    dic={
        "units":[],
        "epochs":[],
        "batch_size":[],
        "AUC":[]
    }
    for u in range(len(units)):
        for i in range(len(epochs)):
            for j in range(len(batch_size)):
                print(units[u],epochs[i],batch_size[j])
                _,avg_auc=evaluate_nn(train_X_R, test_X_R, test_y, seed=[303,305], units=units[u],epochs=epochs[i],batch_size=batch_size[j])
                dic["units"].append(units[u])
                dic["epochs"].append(epochs[i])
                dic["batch_size"].append(batch_size[j])
                dic["AUC"].append(avg_auc)
    return pd.DataFrame(dic)
def filter_startswith(columns_vector, pattern):
    mask = []
    for i in columns_vector.values: mask.append(i.startswith(pattern))
    return columns_vector.values[mask]

def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])
ozone = pd.read_csv(
    "https://raw.githubusercontent.com/r4msi/LSTM_Ozone/master/ozone.csv",
    parse_dates=["Date"],
    index_col="Date",
    dtype=float,
    na_values="?"
)
# ozone_report = sv.analyze(ozone, target_feat="target")
# ozone_report.show_html('ozone.html')
print(f" -Observations: {ozone.shape[0]}\n -Variables: {ozone.shape[1]}")
ozone.head()
ozone.describe()
print(f"{np.min(ozone.index)}\n{np.max(ozone.index)}")
np.array(ozone.isnull().sum())
fig=sns.countplot(ozone.target)
fig=plt.title("Ozone Days")
pd.crosstab(ozone.target, "count")
fig=plt.figure(figsize=(15,6))
fig=plt.subplot()
fig = ozone.target[:365].plot()
fig = ozone.target[365:730].plot()
fig = ozone.target[730:1095].plot()
fig = ozone.target[1095:1460].plot()
fig = ozone.target[1460:1825].plot()
fig = ozone.target[1825:2190].plot()
fig=ozone.target[-365:].plot()
fig=plt.title("Ozone")
series = ["WSR_AV","T_AV","T85","RH50","U70","V50","HT50","KI","TT","SLP","Precp","target"] # Just a few...
for i in series:
    fig=plt.figure(figsize=(10,5))
    fig=ozone[i].plot(c=np.random.rand(3,))
    fig=plt.title(i)
    fig=None
fig=plt.figure(figsize=(14,5))
fig=plt.subplot(1,2,1)
fig=ozone.T_AV.interpolate().plot(c="blue",title="T_AV")
fig=plt.subplot(1,2,2)
fig=ozone.WSR_AV.interpolate().plot(title="WSR_AV")
WSR_columns = filter_startswith(ozone.columns, "WSR")
len(WSR_columns)
ozone.iloc[:,26:]=ozone.drop(WSR_columns,axis=1).interpolate(method="linear",limit_direction='forward')
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
ozone[ozone.columns]=imputer.fit_transform(ozone)
ozone.isnull().sum()
fig=plt.figure(figsize=(14,5))
fig=plt.subplot(1,2,1)
fig=ozone.T_AV.interpolate().plot(c="blue",title="T_AV")
fig=plt.subplot(1,2,2)
fig=ozone.WSR_AV.interpolate().plot(title="WSR_AV")
T_columns = filter_startswith(ozone.columns, "T")
T_columns = np.append(T_columns,"target")

corr = ozone[T_columns].corr()
fig = plt.figure(figsize=(20,10))
fig = sns.heatmap(round(corr,2), cmap="coolwarm",fmt='.2f',annot=True)
fig = plt.title("Correlation Temperature")
ozone["T14_16"]=(ozone.T14+ozone.T15+ozone.T16)/3
ozone[["target","T14_16", "T14","T15", "T16", "T_PK"]].corr("kendall")
WSR_columns = filter_startswith(ozone.columns, "WSR")
WSR_columns = np.append(WSR_columns,"target")

corr = ozone[WSR_columns].corr()
fig = plt.figure(figsize=(20,10))
fig = sns.heatmap(round(corr,3), cmap="coolwarm",fmt='.2f')
fig = plt.title("Correlation WSR")
ozone["WSR10_13"]=(ozone.WSR10+ozone.WSR11+ozone.WSR12)/3
features = ["WSR_PK","T_PK","T85","T70","RH70","RH50","U70","U50","V50","V70","HT70","HT50","KI","TT","SLP","SLP_","Precp","target"]
corr = ozone[features].corr()
fig = plt.figure(figsize=(11,7))
fig = sns.heatmap(round(corr,2), cmap="coolwarm",fmt='.2f', annot=True)
fig = plt.title("Correlation Importance Variables")
scaler=MinMaxScaler()
x=scaler.fit_transform(ozone.drop("target",axis=1))
tsne_fitting = TSNE(n_components=2, perplexity=25, n_iter=250, learning_rate=100, metric="braycurtis")
X_embedded = tsne_fitting.fit_transform(x)
plot_data = pd.DataFrame([])
plot_data['tsne-one'] = X_embedded[:,0]
plot_data['tsne-two'] = X_embedded[:,1]
plot_data['y'] = ozone.target.values
plt.figure(figsize=(16,10))
plt.title('t-SNE vs Target')
fig=sns.scatterplot(
    x="tsne-one", y="tsne-two",
    hue="y",
    data=plot_data,
)
predictors = ["WSR_PK","T_PK","T14_16","WSR10_13","RH70","RH50","U70","U50","V50","V70","HT70","HT50","KI","TT","SLP","SLP_","Precp"]
X = supervised_lag(ozone[predictors], 3)
y = ozone.target.values[3:]
X.head(2)
impRF = RandomForestClassifier(n_estimators=500, criterion="gini", random_state=123)
impRF.fit(X,y)

imp = {}
for i in range(len(X.columns)):
    imp[X.columns[i]] = [impRF.feature_importances_[i]]
pd.DataFrame.from_dict(imp, orient="index", columns=["Importance"]).sort_values("Importance", ascending=False).head(25).style.background_gradient()
ozone["month"]=pd.DataFrame(ozone.index).Date.apply(lambda x: x.strftime("%b")).values
dummy_month=pd.get_dummies(ozone.month)
ozone=pd.concat([ozone.drop("month",axis=1),dummy_month],axis=1)
final_predictors = ["WSR10_13","T14_16", "U70", "RH50","V70", "TT", "KI"]
final_predictors = np.append(final_predictors, dummy_month.columns)
final_predictors
train_X, train_y, test_X, test_y = ts_train_test(ozone, final_predictors, "target", 640)
train_y.tail(1)
test_X.head(1)
pd.crosstab(train_y,"count",normalize=True)
pd.crosstab(test_y,"count",normalize=True)
scaler = MinMaxScaler().fit(train_X)
train_X[final_predictors] = scaler.transform(train_X)
test_X[final_predictors] = scaler.transform(test_X)
train_X.columns
train_X_R = ts_lag(train_X, 2)
print(train_X_R.shape)
train_y=train_y[2:]
test_X_R = ts_lag_test(train_X,test_X,2)
test_X_R.shape
model, history, = make_LSTM(train_X_R, train_y, test_X_R, test_y, units=16, epochs=50, batch_size=64,seed=303)
evaluate_nn(train_X_R, test_X_R, test_y)
model.evaluate(test_X_R,test_y)
plot_loss(history)
plot_auc(history)
plot_roc(test_y, model.predict(test_X_R))
units = [8,16,32] # Try [8,16,32,64]
epochs=[256] # Try [50,64,128,256]
batch_size=[128] # Try [32,64,128]
tune_nn(units=units, epochs=epochs, batch_size=batch_size)
model, history, = make_LSTM(train_X_R, train_y, test_X_R, test_y, units=32, epochs=256, batch_size=128,seed=303)
model.evaluate(test_X_R,test_y)
plot_loss(history)
plot_auc(history)
plot_roc(test_y, model.predict(test_X_R))
y_prob=model.predict(test_X_R)
Find_Optimal_Cutoff(test_y,y_prob)
y_pred=np.where(y_prob>0.1977,1,0) # Optimum cut
visual=pd.DataFrame()
visual["target"]=ozone.target[-640:]
visual["pred"]=y_pred
fig=visual.plot(alpha=.8)
fig=plt.figure()
fig=visual.target.plot()
confusion_matrix(test_y,y_pred,labels=(1,0))
print(f"Acc: {(24+531)/(24+531+81+4)}, Recall:{24/(24+4)}, Especificidad: {531/(531+81)}")
def make_Stacked_LSTM(train_X,train_y,test_X,test_y,units,epochs,batch_size,seed):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    model = Sequential()
    model.add(LSTM(units, return_sequences=True,
                      input_shape=(train_X.shape[1],train_X.shape[2]),kernel_initializer="he_uniform"
                  ))
    model.add(LSTM(units,kernel_initializer="he_uniform"))
    model.add(Dense(1,activation="sigmoid"))
    model.compile(loss="BinaryCrossentropy", optimizer="rmsprop", metrics=["AUC"])
    history=model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,
                         validation_data=(test_X,test_y), verbose=0, shuffle = False)
 
    return model, history
model, history, = make_Stacked_LSTM(train_X_R, train_y, test_X_R, test_y, units=32, epochs=115, batch_size=128,seed=303)
model.evaluate(test_X_R,test_y)
plot_loss(history)
plot_auc(history)
plot_roc(test_y, model.predict(test_X_R))
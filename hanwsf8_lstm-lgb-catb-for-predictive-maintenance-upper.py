import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix,accuracy_score



from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM, Activation

from keras.callbacks import EarlyStopping



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline  
dataset_train=pd.read_csv('../input/PM_train.txt',sep=' ',header=None).drop([26,27],axis=1)

col_names = ['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']

dataset_train.columns=col_names

print('Shape of Train dataset: ',dataset_train.shape)

dataset_train.head()

dataset_train[dataset_train.id==1].cycle.max()
for i in dataset_train.columns:

    print(i,": ",len(dataset_train[i].unique()))
dataset_test=pd.read_csv('../input/PM_test.txt',sep=' ',header=None).drop([26,27],axis=1)

dataset_test.columns=col_names

#dataset_test.head()

print('Shape of Test dataset: ',dataset_train.shape)

dataset_train.head()
for i in dataset_test.columns:

    print(i,": ",len(dataset_test[i].unique()))

#这里测试的都是正常的运行，还没有到失效的状态
pm_truth=pd.read_csv('../input/PM_truth.txt',sep=' ',header=None).drop([1],axis=1)

pm_truth.columns=['more']

pm_truth['id']=pm_truth.index+1

pm_truth
# generate column max for test data这里是取得每个id最大的运行次数

rul = pd.DataFrame(dataset_test.groupby('id')['cycle'].max()).reset_index()

rul.columns = ['id', 'max']

rul.head()
rul[17:18]
# run to failure到失效运行的次数

pm_truth['rtf']=pm_truth['more']+rul['max']

pm_truth.head()
pm_truth[17:18] #161-133=27,是说明这个18号正好有3个落入到失效前30个记录中
pd.set_option('display.max_columns', None)

pm_truth.drop('more', axis=1, inplace=True)

dataset_test=dataset_test.merge(pm_truth,on=['id'],how='left')#这个仅将每条记录后面将rtf的数值加上

dataset_test.head()
dataset_test['ttf']=dataset_test['rtf'] - dataset_test['cycle']

dataset_test.drop('rtf', axis=1, inplace=True)

dataset_test.head()
pd.set_option('display.max_rows', None)

dataset_test[30:145]#这里仅仅将现有的未到失效的记录加上ttf数据
dataset_train['ttf'] = dataset_train.groupby(['id'])['cycle'].transform(max)-dataset_train['cycle']

dataset_train.head()
df_train=dataset_train.copy() #这里没有回收

df_test=dataset_test.copy()

period=30  #这个的作用,就是离失效还有30个循环的时候标记为1.

df_train['label_bc'] = df_train['ttf'].apply(lambda x: 1 if x <= period else 0)

df_test['label_bc'] = df_test['ttf'].apply(lambda x: 1 if x <= period else 0)

df_train.head()
df_train[170:200]
# len(df_test[df_test.label_bc==1])

#test中有1？332
features_col_name=['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',

                   's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

target_col_name='label_bc'
df_train.describe()
sc=MinMaxScaler()

df_train[features_col_name]=sc.fit_transform(df_train[features_col_name])

df_test[features_col_name]=sc.transform(df_test[features_col_name])
df_train.describe()#全部变成0-1，并且将常量变为0
def gen_sequence(id_df, seq_length, seq_cols):

    df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)

    id_df=df_zeros.append(id_df,ignore_index=True)

    data_array = id_df[seq_cols].values

    num_elements = data_array.shape[0]

    lstm_array=[]

    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):

        lstm_array.append(data_array[start:stop, :])

    return np.array(lstm_array)



# function to generate labels

def gen_label(id_df, seq_length, seq_cols,label):

    df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)

    id_df=df_zeros.append(id_df,ignore_index=True)

    data_array = id_df[seq_cols].values

    num_elements = data_array.shape[0]

    y_label=[]

    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):

        y_label.append(id_df[label][stop])

    return np.array(y_label)
# timestamp or window size

seq_length=50

seq_cols=features_col_name
# generate X_train

X_train=np.concatenate(list(list(gen_sequence(df_train[df_train['id']==id], seq_length, seq_cols)) for id in df_train['id'].unique()))

print(X_train.shape)

# generate y_train

y_train=np.concatenate(list(list(gen_label(df_train[df_train['id']==id], 50, seq_cols,'label_bc')) for id in df_train['id'].unique()))

print(y_train.shape)
y_train[0:5]
X_train[1,48,:]
# df_test['id'].unique()
# generate X_test

X_test=np.concatenate(list(list(gen_sequence(df_test[df_test['id']==id], seq_length, seq_cols)) for id in df_test['id'].unique()))

print(X_test.shape)

# generate y_test

y_test=np.concatenate(list(list(gen_label(df_test[df_test['id']==id], 50, seq_cols,'label_bc')) for id in df_test['id'].unique()))

print(y_test.shape)
nb_features =X_train.shape[2]

timestamp=seq_length #50



model = Sequential()



model.add(LSTM(

         input_shape=(timestamp, nb_features),

         units=100,  #前面只有24？

         return_sequences=True))

model.add(Dropout(0.2))



model.add(LSTM(

          units=50,

          return_sequences=False))

model.add(Dropout(0.2))



model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
# fit the network

model.fit(X_train, y_train, epochs=10, batch_size=200, validation_split=0.05, verbose=1,

          callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])
# training metrics

scores = model.evaluate(X_train, y_train, verbose=1, batch_size=200)

print('Accurracy: {}'.format(scores[1]))
y_pred=model.predict_classes(X_test)

print('Accuracy of model on test data: ',accuracy_score(y_test,y_pred))

print('Confusion Matrix: \n',confusion_matrix(y_test,y_pred))
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

print('scores {:.5f} accuracy_score on the test set.'.format(accuracy_score(y_test, y_pred)))

print('scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, y_pred)))

print('scores {:.5f} precision_score on the test set.'.format(precision_score(y_test, y_pred)))

print('scores {:.5f} recall_score on the test set.'.format(recall_score(y_test, y_pred)))

print('scores {:.5f} f1_score on the test set.'.format(f1_score(y_test, y_pred,'binary')))

confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
def prob_failure(machine_id):

    machine_df=df_test[df_test.id==machine_id]

    machine_test=gen_sequence(machine_df,seq_length,seq_cols)

    print('Shape of %0.0f is: %0.0f' % (machine_id, len(machine_test)))

    m_pred=model.predict(machine_test)

#     print('m_pred is: ', m_pred)

    failure_prob=list(m_pred[-1]*100)[0]

    return failure_prob
machine_id=81

print('Probability that machine will fail within 30 days: ',prob_failure(machine_id))
df_train.head()
df_test.head()
train = df_train.drop(['id','cycle','ttf'],axis=1).copy()

train.head()
corrmat = train.corr()

corrmat
# type(train)
from sklearn.model_selection import KFold, cross_val_score, train_test_split

X_train, X_test, y_train, y_test=train_test_split(train.iloc[0:-2000:,0:-1],train.iloc[0:-2000:,-1], test_size=0.2, random_state=3)

# gc.collect()  

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)
import lightgbm as lgb

clf5 = lgb.LGBMClassifier(learning_rate=0.05,n_estimators=10000,num_leaves=100,objective='binary', metrics='auc',random_state=50,n_jobs=-1)

clf5.fit(X_train, y_train, eval_set=(train.iloc[-2000:-1,0:-1], train.iloc[-2000:-1,-1]),

#         eval_metric='auc',#缺省用logloss

         #n_jobs=-1,

        early_stopping_rounds=50)

clf5.score(X_test, y_test)

preds2 = clf5.predict(X_test)

print('scores {:.5f} accuracy_score on the test set.'.format(accuracy_score(y_test, preds2)))

print('scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, preds2)))

print('scores {:.5f} precision_score on the test set.'.format(precision_score(y_test, preds2)))

print('scores {:.5f} recall_score on the test set.'.format(recall_score(y_test, preds2)))

print('scores {:.5f} f1_score on the test set.'.format(f1_score(y_test, preds2,'binary')))
confusion_matrix(y_test, preds2, labels=None, sample_weight=None)
from catboost import CatBoostClassifier, Pool, cv



clf6 = CatBoostClassifier(

    custom_loss=['Accuracy'],#这里是提供个列表

    random_seed=42,

    logging_level='Silent')

clf6.fit(

    X_train, y_train,

    #cat_features=list(categorical_features_indices),

    eval_set=(train.iloc[-2000:-1,0:-1], train.iloc[-2000:-1,-1]),

    logging_level='Verbose',  # you can uncomment this for text output

    plot=True,

     early_stopping_rounds=50

)

preds2 = clf6.predict(X_test)

print('scores {:.5f} accuracy_score on the test set.'.format(accuracy_score(y_test, preds2)))

print('scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, preds2)))

print('scores {:.5f} precision_score on the test set.'.format(precision_score(y_test, preds2)))

print('scores {:.5f} recall_score on the test set.'.format(recall_score(y_test, preds2)))

print('scores {:.5f} f1_score on the test set.'.format(f1_score(y_test, preds2,'binary')))

confusion_matrix(y_test, preds2, labels=None, sample_weight=None)
##测试text数据集

preds3 = clf5.predict(df_test.drop(['id','cycle','ttf','label_bc'],axis=1))



print('scores {:.5f} accuracy_score on the test set.'.format(accuracy_score(df_test.iloc[:,-1], preds3)))

print('scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(df_test.iloc[:,-1], preds3)))

print('scores {:.5f} precision_score on the test set.'.format(precision_score(df_test.iloc[:,-1], preds3)))

print('scores {:.5f} recall_score on the test set.'.format(recall_score(df_test.iloc[:,-1], preds3)))

print('scores {:.5f} f1_score on the test set.'.format(f1_score(df_test.iloc[:,-1], preds3,'binary')))

confusion_matrix(df_test.iloc[:,-1], preds3, labels=None, sample_weight=None)
##测试text数据集

preds3 = clf6.predict(df_test.drop(['id','cycle','ttf','label_bc'],axis=1))



print('scores {:.5f} accuracy_score on the test set.'.format(accuracy_score(df_test.iloc[:,-1], preds3)))

print('scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(df_test.iloc[:,-1], preds3)))

print('scores {:.5f} precision_score on the test set.'.format(precision_score(df_test.iloc[:,-1], preds3)))

print('scores {:.5f} recall_score on the test set.'.format(recall_score(df_test.iloc[:,-1], preds3)))

print('scores {:.5f} f1_score on the test set.'.format(f1_score(df_test.iloc[:,-1], preds3,'binary')))

confusion_matrix(df_test.iloc[:,-1], preds3, labels=None, sample_weight=None)
id_num =81

predict_df = df_test[df_test.id==id_num].iloc[-1,:]#[df_test.cycle==df_test.cycle.max()]

# predict_df

pred_probe = clf6.predict_proba(pd.DataFrame(predict_df).T.drop(['id','cycle','ttf','label_bc'],axis=1))

pred_probe[0][1]
import numpy as np
import pandas as pd
import time
import gc
# Input data files are available in the "../input/" directory
dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}

path = '../input/'

# Funcoes para manejar os dados de click_time
def diff_timestamp(df, drop_na = False):
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['diff'] = df.groupby('ip').click_time.diff().dt.total_seconds()
    # Se drop_na = True, a funcao joga fora informacao
    if drop_na == True:
        nulls = df['diff'].isnull()
        df.drop(nulls[nulls == True].index, inplace = True)
    return df

def handleClickHour_raul(df):
    df['click_hour']= pd.to_datetime(df['click_time']).dt.hour.astype('uint8')
    #df['click_minute'] = pd.to_datetime(df['click_time']).dt.minute.astype('uint8')
    #df['click_second'] = pd.to_datetime(df['click_time']).dt.second.astype('uint8')
    df = df.drop(['click_time'], axis=1)   
    return df
# Importando dados de teste
train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
start_time = time.time()
df_train_30m = pd.read_csv(path + 'train.csv', dtype=dtypes, skiprows=range(1,163333333), 
                           nrows=35000000, usecols=train_columns)
print('Loaded df_train_30m with {:f} seconds'.format(time.time() - start_time))
# Importando dados de treinamento
start_time = time.time()
df_test = pd.read_csv(path + 'test.csv', dtype=dtypes)
print('Loaded df_test in {:.2f} seconds'.format(round(time.time() - start_time, 3)))

train_record_index = df_train_30m.shape[0]

# Ajeitando o click_hour 
df_train_30m = diff_timestamp(df_train_30m)
df_train_30m = handleClickHour_raul(df_train_30m)
df_test = diff_timestamp(df_test)
df_test = handleClickHour_raul(df_test)
gc.collect();
print('ClickTime data correctly handled.')

# DataFrame de submissão
df_submit = pd.DataFrame()
df_submit['click_id'] = df_test['click_id']

# Extraindo o target de treinamento
Learning_Y = df_train_30m['is_attributed']
print('Training target correctly extracted.')

# Dropando informação redundante
df_test = df_test.drop(['click_id'], axis=1)
df_train_30m = df_train_30m.drop(['is_attributed'], axis=1)
gc.collect();

# Juntando os dois datasets para podermos criar features nos dois ao mesmo tempo
df_merge = pd.concat([df_train_30m, df_test])

# Liberando espaço na memória
del df_train_30m, df_test
gc.collect();
print('Data was correctly concatenated')
# Criando feature que conta quantos cliques aquele IP deu
start_time = time.time()
df_ip_count = df_merge['ip'].value_counts().reset_index(name = 'ip_count')
df_ip_count.columns = ['ip', 'ip_count']
print('Loaded df_ip_count with {:.2f} seconds'.format(time.time() - start_time))
print('Starting to merge with main dataset...')
df_merge = df_merge.merge(df_ip_count, on='ip', how='left', sort=False)
df_merge['ip_count'] = df_merge['ip_count'].astype('uint16')
print('Merging operation completed.')
del df_ip_count
df_merge = df_merge.drop(['ip'], axis=1)
gc.collect();
df_merge.head()
# Usando a métrica do ROC Score
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

def clf_eval(y_true, y_pred):
    print('Classification Report')
   # print('F-1 Score: {}'.format(f1_score(y_true, y_pred)))
    print('ROC Score: {}'.format(roc_auc_score(y_true, y_pred)))
    return roc_auc_score(y_true, y_pred)

# Recuperando os dados de treino e teste
df_train = df_merge[:train_record_index]
df_test = df_merge[train_record_index:]
del df_merge
gc.collect();
# Checando que fração dos dois datasets tem clicks com NaN na coluna diff (que seriam primeiros cliques!)
print('Fração de NaN em df_train: {:.4f} %'.format(100*len(df_train[df_train['diff'].isnull()])/len(df_train)))
print('Fração de NaN em df_test: {:.4f} %'.format(100*len(df_test[df_test['diff'].isnull()])/len(df_test)))
df_train['diff'].fillna(0, inplace = True)
df_test['diff'].fillna(0, inplace = True)
gc.collect();

# Otimizando a memória
df_train['diff'] = df_train['diff'].astype('uint8')
df_test['diff'] = df_test['diff'].astype('uint8')
Learning_Y = Learning_Y.astype('uint8')
df_test.info()
# Criando conjunto de cross-validation
from sklearn import model_selection
X_train, X_cv, Y_train, Y_cv = model_selection.train_test_split(df_train, Learning_Y, train_size = 0.8)
gc.collect();
print('Data splitting into training and cross validation is done.')
from sklearn.ensemble import RandomForestClassifier
print('Starting to fit Random Forest Model... The machine is learning...')
start_time = time.time()
rf = RandomForestClassifier(n_estimators=13, max_depth=13, random_state=13, verbose=2, n_jobs = 4)

cols = ['app', 'os', 'channel', 'device', 'click_hour', 'diff', 'ip_count']
rf.fit(X_train[cols], Y_train)
print('The machine has learned.')
print('RandomForest has fitted X_train with {:.2f} seconds'.format(time.time() - start_time))
print('Starting cross-validation prediction phase...')
start_time = time.time()
predictions = rf.predict_proba(X_cv[cols])[:,1]
print('Prediction done. Elapsed time: {:.2f} seconds'.format(time.time() - start_time))

# Avaliando
clf_eval(Y_cv, predictions)
# Implementando UnderSampler
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(return_indices=True)
X_resampled, Y_resampled, idx_resampled = rus.fit_sample(X = X_train, y = Y_train)
# Transformando em Dataframes
X_resampled = pd.DataFrame(X_resampled, columns=df_train.columns)
Y_resampled = pd.DataFrame(Y_resampled, columns=['is_attributed'])
# Implementando o Random Forest com resampling
rf_resampled = RandomForestClassifier(n_estimators=13, 
                                      max_depth=13, 
                                      random_state=13, 
                                      verbose=2, 
                                      n_jobs = 5)
rf_resampled.fit(X_resampled, Y_resampled)
print('The machine has learned.')
print('RandomForest has fitted X_train with {:.2f} seconds'.format(time.time() - start_time))
print('Starting cross-validation prediction phase...')
predictions = rf.predict_proba(X_cv)[:,1]
print('Prediction done. Elapsed time: {:.2f} seconds'.format(time.time() - start_time))

# Avaliando
clf_eval(Y_cv, predictions)
# Predicao
print('Starting prediction phase...')
start_time = time.time()
predictions = rf.predict_proba(df_test[cols])
print('Prediction done. Elapsed time: {:.2f} seconds'.format(time.time() - start_time))

# Creating the submission dataset
df_submit['is_attributed'] = predictions[:,1]
print('Submission dataset created.')

# Preparing submssion
df_submit.to_csv('timed_rf_raul.csv', index=False)
print('Submission dataset saved correctly.')
import numpy as np

import pandas as pd

import seaborn as sns

import random

import datetime

from catboost import CatBoostClassifier

from sklearn.model_selection import KFold

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from time import time

from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def qwk(act,pred,n=4,hist_range=(0,3)):

    

    O = confusion_matrix(act,pred)

    O = np.divide(O,np.sum(O))

    

    W = np.zeros((n,n))

    for i in range(n):

        for j in range(n):

            W[i][j] = ((i-j)**2)/((n-1)**2)

            

    act_hist = np.histogram(act,bins=n,range=hist_range)[0]

    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]

    

    E = np.outer(act_hist,prd_hist)

    E = np.divide(E,np.sum(E))

    

    num = np.sum(np.multiply(W,O))

    den = np.sum(np.multiply(W,E))

        

    return 1-np.divide(num,den)
train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
# encode title

# здесь объединили все колонки title - и в тесте и в трейне

list_of_user_activities = list(set(train['title'].value_counts().index).union(set(test['title'].value_counts().index)))

# вручную перекодировали игры из колонки title - по порядку от 0 до N

activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))



# заменили все значения в зависимости от кодировки

train['title'] = train['title'].map(activities_map)

test['title'] = test['title'].map(activities_map)

train_labels['title'] = train_labels['title'].map(activities_map)



# кодируем для каждой игры win_code - метку, которая обозначает выигрыш в этой игре

# словарь для выигрышей

win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))

win_code[activities_map['Bird Measurer (Assessment)']] = 4110
# переводим время в формат timestamp

train['timestamp'] = pd.to_datetime(train['timestamp'])

test['timestamp'] = pd.to_datetime(test['timestamp'])
feat_dict = {'video_start': ['27253bdc','b1d5101d','d51b1749','c189aaf2','c1cac9a2','3b2048ee','27253bdc',

               '46b50ba8','4a09ace1','99ea62f3','9e6b7fb5'],

'skip_end_tuttorial' : ['d2659ab4','d88ca108','dcb55a27','3bb91ced','9ed8f6da','7d5c30a2','06372577'],

'game_start': ['7d093bf9','f93fc684','b5efe37','b7dc8128','99abe2bb','1cf54632','3dfd4aa4',

               '736f9581','2c4e6db0','c7f7f0e1','532a2afb','dcaede90','abc5811c','77261ab5',

               '7d093bf9','7ad3efc6','5a848010','3bfd1a65','8ac7cce4','e64e2cfd','90d848e0',

               '8d84fa81','f56e0afc','51311d7a','7cf1bc53','9b01374f','b7530680','65abac75',

               '63f13dd7','155f62a4','0086365d','87d743c1','26fd2d99','9b23e8ee','7961e599',

               'f806dc10','5b49460a','29bdd9ba','ec138c1c','fd20ea40','c2baf0bd','1beb320a',

               'ecc36b7f','1575e76c','15ba1109','f93fc684','d9c005dd','48349b14','cc5087a3',

               '756e5507','4901243f','a592d54e','f32856e4','6d90d394','5be391b5','c6971acf',

               'd2278a3b','9c5ef70c','7040c096','29f54413'],

'game_exit' : ['a8cc6fec','e9c52111','f5b8c21a','86ba578b','1b54d27f','17113b36','b738d3d3',

              '25fa8af4','6088b756','4d6737eb','3393b68b','2b058fe3','93b353f2','b012cd7f',

               'c54cf6c5','3323d7e9','9565bea6','895865f3','070a5291','392e14df','17ca3959',

               '36fa3ebe','a5be6304','3ccd3f02','3bf1cf26','f6947f54','222660ff','4074bac2',

               'b2e5b0f1','16dffff1','003cd2ee'],

'game_instructions': ['3babcb9b','7f0836bf','ab3136ba','bbfe0445','33505eae','15f99afc','795e4a37',

                     'a1e4395d','6043a2b4','1375ccb7','363d3849','923afab1','67439901',

                     '71fe8f75','a29c5338','ea321fb1','0d1da71f','f7e47413','3dcdda7f',

                     'df4940d3','2dcad279','bdf49a58','a52b92d5','6cf7d25c','1bb5fbdb',

                     '84b0e0c8','7ab78247','7ec0c298','b2dba42b','69fdac0a','49ed92e9',

                      'bd701df8','832735e1','2a512369','5154fc30','9e4c8c7b','beb0a7b9',

                      '0413e89d','0a08139c','b80e5e84','f71c4741','56cd3b43','b88f38da',

                      '828e68f9','f28c589a','15eb4a7d','a1bbe385','8d7e386c'],

'game_obr_time_true': ['2b9272f4','47026d5f','3afde5dd','e720d930','3ddc79c3','709b1251',

                      '4d911100','45d01abe','6f4adc4b','cf7638f3','d3268efa','ecaab346',

                      'e5c9df6f','77ead60d','a8a78786','9b4001e4','3afb49e6','b5053438',

                      '250513af','55115cbd','c7fe2a55','c74f40cd','e4f1efe6','73757a5e',

                       'cb6010f8','e3ff61fb','7525289a','daac11b0','a8876db3','9d29771f',

                      '1f19558b','58a0de5c'],

'game_obr_time_false': ['df4fe8b6','d88e8f25','c277e121','160654fd','ea296733','5859dfb6',

                       'e04fb33d','28a4eb9a','7423acbc','e57dd7af','04df9b66','2230fab4',

                       'c51d8688','1af8be29','89aace00','763fc34e','5290eab1','90ea0bac',

                        '8b757ab8','e5734469','9de5e594','d45ed6a1','ac92046e','ad2fc29c',

                        '5de79a6a','88d4a5be','907a054b','e37a2b78','31973d56','44cb4907',

                        '0330ab6a'],

'game_events_special': ['bc8f2793','8fee50e2','d02b7a8e','30614231','3d8c61b0',

                       'a8efe47b','65a38bf7','461eace6','5f0eb72c','56bcd38d','08fd73f3',

                       'd122731b','a5e9da97','0db6d71d','91561152','6c930e6e','792530f8',

                       '14de4c5d','5348fd84','8af75982','363c86c9','c0415e5c','71e712d8',

                       'c58186bf','5c2f29ca','84538528','3bb91dda','d38c2fd7','857f21c0',

                       '9d4e7b25','9ce586dd','1cc7cfca','5e812b27','e694a35b','d3f1e122',

                        '37ee8496','a76029ee','a1192f43','85de926c','9ee1c98c','f50fc6c1',

                        '8f094001','598f4598','3d0b9317','4a4c3d21','499edb7c','4ef8cdd3',

                        '2dc29e21','fbaf3456','74e5f8a7','6bf9e3e1','5e109ec3','262136f4',

                        '90efca10','02a42007','804ee27f','5c3d2b2f','4c2ec19f','fcfdffb6',

                        '86c924c4','46cd75b4','de26c3a6','ad148f58','5d042115','db02c830',

                        '51102b85','37db1c2f','c7128948','1c178d24','562cec5f','cfbd47c8',

                        '3edf6747','1996c610','022b4259','bb3e370b','e7561dd2','a6d66e51',

                        'd06f75b5','d2e9262e','2fb91ec1','28ed704e','83c6c409','38074c54'

                       '2a444e03'],

'game_events_true' :['56817e2b','28520915','b74258a0','37c53127','ca11f653'],

'game_events_false' : ['cdd22e43','3d63345e','b120f2ac','0d18d96c','d185d3ea','a2df0760'],

'game_otvleksya' : ['bcceccc6','1325467d','a7640a16','bd612267','76babcde','6c517a88',

                    '4bb2f698','7dfe6d8a','9e34ea74','c952eb01','3ee399c3','7da34a02',

                    '565a3990','a44b10dc','acf5c23f','587b5989','a16a373e','a0faea5d',

                    'cf82af56','884228c8','5e3ea25a'],

'game_find_fail' : ['e79f3763'],

'user_see':['119b5b02','93edfe2e','ecc6157f','7fd1ac25','5dc079d8','53c6e11a','05ad839b',

           '26a5a3dd','9554a50b','37937459','37937459','28f975ea','6aeafed4','6f8106d9',

            '15a43e5b','77c76bc5','f3cd5473','16667cc5','7372e1a5','f54238ee','6f445b57',

           '4e5fc6f5','47f43a44','85d1b0de','29a42aea','bfc77bd6','13f56524','30df3273',

           '5f5b2617','47efca07','01ca3a3c','6077cc36','731c0cbe','e080a381','92687c59',

            'e4d32835','ab4ec3a4','eb2c19cd','6f4bd64e','08ff79ad','2ec694de','3a4be871',

            'cb1178ad','dcb1663e','0ce40006','1340b8d7','67aa2ada','e79f3763','611485c5',

            '19967db1','e7e44842','8d748b58','d3640339']}
def get_data(user_sample, test_set=False):

    last_activity = 0

    user_activities_count = {'Clip': 0, 'Activity': 0, 'Assessment': 0, 'Game': 0}

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    all_assessments = []

    accumulated_accuracy_group = 0

    accumulated_accuracy=0

    accumulated_correct_attempts = 0 

    accumulated_uncorrect_attempts = 0 

    accumulated_actions = 0

    counter = 0

    durations = []



    for i, session in user_sample.groupby('game_session', sort=False):

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

#         sess_xx = session['xx'].iloc[0]

#         sess_yy = session['yy'].iloc[0]

        event_game_start, event_game_exit, event_game_instructions = 0,0,0

        event_game_obr_time_true,event_game_obr_time_false = 0,0

        event_game_events_special, event_game_otvleksya = 0,0

        event_user_see, event_game_events_special = 0, 0

        event_game_events_true, event_game_events_false = 0,0

    

        if test_set == True:

            second_condition = True

        else:

            if len(session)>1:

                second_condition = True

            else:

                second_condition= False

        # генерирую свои признаки

        for i in session['event_id']:

            if i in feat_dict['game_start']:

                event_game_start +=1 

            elif i in feat_dict['game_exit']:

                event_game_exit+=1

            if i in feat_dict['game_instructions']:

                event_game_instructions+=1

            elif i in feat_dict['game_obr_time_true']:

                event_game_obr_time_true+=1

            elif i in feat_dict['game_obr_time_false']:

                event_game_obr_time_false +=1

            elif i in feat_dict['game_events_special']:

                event_game_events_special +=1

            elif i in feat_dict['game_otvleksya']:

                event_game_otvleksya += 1

#             elif i in feat_dict['user_see']:

#                 event_user_see += 1

            elif i in feat_dict['game_events_special']:

                event_game_events_special += 1

            elif i in feat_dict['game_events_true']:

                event_game_events_true += 1

            elif i in feat_dict['game_events_false']:

                event_game_events_false += 1

            

            

        if (session_type == 'Assessment') & (second_condition):

            all_attempts = session.query(f'event_code == {win_code[session_title]}')

            # считаем суммы побед и поражений

            true_attempts = all_attempts['event_data'].str.contains('true').sum()

            false_attempts = all_attempts['event_data'].str.contains('false').sum()

            # перевожу победы и поражения из чисел в проценты...кстати не факт что зайдет такой формат

            # ведь по итогу мы смотрим на число удачных попыток

#             true_attempts = true_attempts/(true_attempts+false_attempts)

#             false_attempts = false_attempts/(true_attempts+false_attempts)

            

            features = user_activities_count.copy()

    #         features['installation_id'] = session['installation_id'].iloc[0]

#             features['game_session'] = i

            features['session_title'] = session['title'].iloc[0] 

            features['accumulated_correct_attempts'] = accumulated_correct_attempts

            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts

            accumulated_correct_attempts += true_attempts 

            accumulated_uncorrect_attempts += false_attempts

            #  странный синтаксис, но тут кажется он считает длительность игры

            # вычитая из максимума минимум

            if durations == []:

                features['duration_mean'] = 0

                features['duration_std'] = 0

#                 features['duration_max'] = 0

#                 features['duration_min'] = 0

            else:

                features['duration_mean'] = np.mean(durations)

                features['duration_std'] = np.std(durations)

#                 features['duration_max'] = np.max(durations)

#                 features['duration_min'] = np.min(durations)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            # тут мы считаем то что я предлагал выше, точность по всем победам/поражениям и в зависимости 

            # от этого выбираем группу точности 

            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0

            accumulated_accuracy += accuracy

            if accuracy == 0:

                features['accuracy_group'] = 0

            elif accuracy == 1:

                features['accuracy_group'] = 3

            elif accuracy == 0.5:

                features['accuracy_group'] = 2

            else:

                features['accuracy_group'] = 1



            features.update(accuracy_groups)

            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0

            features['accumulated_actions'] = accumulated_actions

            accumulated_accuracy_group += features['accuracy_group']

            accuracy_groups[features['accuracy_group']] += 1

            if test_set == True:

                all_assessments.append(features)

            else:

                if true_attempts+false_attempts > 0:

                    all_assessments.append(features)

                

            counter += 1



    #         break

        

        # число всех строк в сессии взял как число событий и все сложил - логично

        accumulated_actions += len(session)

        if last_activity != session_type:

            user_activities_count[session_type] += 1

            last_activitiy = session_type

            

        try: 

#             fulls = event_game_start+event_game_exit+event_game_instructions+event_game_obr_time_true+ \

#                     event_game_obr_time_true+event_game_obr_time_false+event_game_events_special + \

#                     event_game_otvleksya + event_user_see+event_game_events_special+\

#                         event_game_events_true+event_game_events_false

            features['game_start'] = event_game_start

            features['game_exit'] = event_game_exit

            features['game_instructions'] = event_game_instructions

            features['game_obr_time_true'] = event_game_obr_time_true

            features['game_obr_time_false'] = event_game_obr_time_false

            features['game_events_special'] = event_game_events_special

            features['game_otvleksya'] = event_game_otvleksya

#             features['user_see'] = event_user_see/fulls

            features['game_events_true'] = event_game_events_true

            features['game_events_false'] = event_game_events_false

            

        except:

            NameError

    if test_set:

        return all_assessments[-1] 

    return all_assessments
compiled_data = []

ids_list = []

# когда такое пишем, сначала идет installation_id, потом вся инфа которая к нему относится

# и получается что каждый вызов user_samle генерирует фичи для уникального installation_id

# total - число всех уников из датасета

for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False)), total=17000):

    ids_list.append(ins_id)

    compiled_data += get_data(user_sample)

#     print('I = ', i)

#     print('INSTALLATION_ID = ', ins_id)

#     print('COMPILED_DATA', compiled_data)
new_train = pd.DataFrame(compiled_data)

# new_train['game_start'] = new_train['game_start'].apply(lambda x: np.log(x))

# new_train['game_exit'] = new_train['game_exit'].apply(lambda x: np.log(x))

# new_train['user_see'] = new_train['user_see'].apply(lambda x: np.log(x+0.001))

# new_train['game_events_true'] = new_train['game_events_true'].apply(lambda x: np.log(x+0.001))

# new_train['game_events_false'] = new_train['game_events_false'].apply(lambda x: np.log(x+0.001))

del compiled_data

new_train.shape
new_train.fillna(0, inplace=True)

new_train.head(5)
'''

# plt.scatter(new_train['accuracy_group'], new_train['game_obr_time_true']);

ss='game_otvleksya'

# new_train[new_train['accuracy_group']==0][ss].hist(bins=100);

new_train[new_train['accuracy_group']==1][ss].hist(bins=100);

new_train[new_train['accuracy_group']==2][ss].hist(bins=100);

# new_train[new_train['accuracy_group']==3][ss].hist(alpha=0.3,bins=100);

'''
'''

from sklearn.manifold import TSNE

tsn = TSNE()

res_tsne = tsn.fit_transform(new_train)



plt.figure(figsize=(8,8))

sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=new_train['accuracy_group'], palette='copper');



for col in list(new_train.columns):

    plt.figure(figsize=(8,8))

    # plt.scatter(res_tsne[:,0],res_tsne[:,1]);

    sns.scatterplot(x=res_tsne[:,0],y=res_tsne[:,1],s=100, hue=new_train[col], palette='copper');

    plt.show()

'''
all_features = [x for x in new_train.columns if x not in ['accuracy_group']]

cat_features = ['session_title']

X, y = new_train[all_features], new_train['accuracy_group']

del train
def make_classifier(lrs,it):

    clf = CatBoostClassifier(

                               loss_function='MultiClass',

    #                            eval_metric="AUC",

                               task_type="CPU",

                               learning_rate=lrs,

                               iterations=it,

                               od_type="Iter",

                               depth=11,

                               early_stopping_rounds=500,

                               l2_leaf_reg=2,

    #                            border_count=96,

#                              bootstrap_type = 'Bayesian',

#                                 random_strength = 0.6,

#                                 bagging_temperature = 0.4,

#                                 use_best_model=True,

                               random_seed=457

                              )

        

    return clf

oof = np.zeros(len(X))
'''

# preds = np.zeros(len(X_test))

# for lr in [0.01,0.03,0.05,0.1]:

#     for itr in [1000,1300,1600,2000,2500]:

#         for dep in [6,10,13]:



# for itr in [1000,1200,1500,1700]:

#     for dept in [8,10,12,16]:

dept, itr = 12, 500

#     lr, itr = 0.02, 2700

print('DEPTH = ', dept)

print('ITERS = ', itr)

for splt in [2,3,4,5,6,7,8]: 

    print('SPLT = ', splt)

    oof = np.zeros(len(X))

    NFOLDS = 5

    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=27644437)



    training_start_time = time()

    for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):

        start_time = time()

        print(f'Training on fold {fold+1}')

    #     clf = make_classifier(lrs=lr, it=itr)

    #     clf.fit(X.loc[trn_idx, all_features], y.loc[trn_idx], eval_set=(X.loc[test_idx, all_features], y.loc[test_idx]),

    #                           use_best_model=True, verbose=1000, cat_features=cat_features)

        clf = XGBClassifier(random_state=457)

        clf.fit(X.loc[trn_idx, all_features], y.loc[trn_idx])



    #     preds += clf.predict(X_test).reshape(len(X_test))/NFOLDS

        oof[test_idx] = clf.predict(X.loc[test_idx, all_features]).reshape(len(test_idx))



        print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))



    print('-' * 30)

    print('OOF QWK:', qwk(y, oof))

    print('-' * 30)

'''
xgbcl = XGBClassifier(n_estimators=700, max_depth=10, verbosity=2)

xgbcl.fit(X,y)
lr, itr = 0.03, 2800

cats = make_classifier(lrs=lr, it=itr)

cats.fit(X, y, verbose=500, cat_features=cat_features)
# lr, itr = 0.03, 2800

# clf = make_classifier(lrs=lr, it=itr)

# clf.fit(X, y, verbose=500, cat_features=cat_features)



# del X, y

clf = RandomForestClassifier(n_estimators=600, max_depth=16, random_state=457, verbose=2,

                            class_weight='balanced')

clf.fit(X,y)

# del X, y
n = 11341042 #number of records in file

s = 1000000 #desired sample size

random.seed(457)

filename = '../input/data-science-bowl-2019/train.csv'

skip = sorted(random.sample(range(n),n-s))

train_lstm = pd.read_csv(filename, skiprows=skip)

train_lstm.columns = ['event_id','game_session','timestamp','event_data',

            'installation_id','event_count','event_code','game_time','title','type','world']
test_lstm = pd.read_csv('../input/data-science-bowl-2019/test.csv')

labels_lstm = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
from more_itertools import sliced

from keras.models import Sequential

from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from keras.callbacks.callbacks import EarlyStopping



full = train_lstm.merge(labels_lstm, how='inner', on=['installation_id','game_session'])

train_ls = full[['installation_id','game_session','event_id']]

# convert to str

train_ls['event_id'] = train_ls['event_id'].apply(lambda x: str(x))



del train_lstm



def events_all(aa):

    xx = ''

    for i in aa: 

        xx += i + ' '

    xx = xx.rstrip()

    return xx



result = train_ls.groupby(['installation_id','game_session']).sum().reset_index()

result['event_id'] = result['event_id'].apply(lambda x: list(sliced(x, 8)))

result['new_event'] = result['event_id'].apply(events_all)

result = result.merge(labels_lstm, how='inner', on=['installation_id','game_session'])[['new_event','accuracy_group']]



# The maximum number of words to be used. (most frequent)

MAX_NB_WORDS = 100

# Max number of words in each complaint.

MAX_SEQUENCE_LENGTH = 100

# This is fixed.

EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(result['new_event'].values)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))



X = tokenizer.texts_to_sequences(result['new_event'].values)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(result['accuracy_group']).values

print('Shape of label tensor:', Y.shape)



X_train, X_tes, Y_train, Y_tes = train_test_split(X,Y, test_size = 0.05, random_state = 457)

print(X_train.shape,Y_train.shape)

# print(X_test.shape,Y_test.shape)



model = Sequential()

model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))

model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# choose epochs and batch_size

epochs = 6

batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001)])
# accr = model.evaluate(X_test,Y_test)

# print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))



last_test = test_lstm[['installation_id','game_session',

                  'timestamp']].groupby(['installation_id']).tail(1)[['installation_id','game_session']]

test_ = test_lstm.merge(last_test,how='inner', on=['installation_id','game_session'])



test_ls = test_[['installation_id','game_session','event_id']]

# test_ls = test[['installation_id','game_session','event_id']]

test_ls['event_id'] = test_ls['event_id'].apply(lambda x: str(x))

res_test = test_ls.groupby(['installation_id','game_session']).sum().reset_index()

res_test['event_id'] = res_test['event_id'].apply(lambda x: list(sliced(x, 8)))

res_test['new_event'] = res_test['event_id'].apply(events_all)



X_ts = tokenizer.texts_to_sequences(res_test['new_event'].values)

X_ts = pad_sequences(X_ts, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', X_ts.shape)
test.shape
# process test set

new_test = []

for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=1000):

    a = get_data(user_sample, test_set=True)

    new_test.append(a)

    

X_test = pd.DataFrame(new_test)

del test
X_test.columns
X_test = X_test[[                          'Clip',                       'Activity',

                           'Assessment',                           'Game',

                        'session_title',   'accumulated_correct_attempts',

       'accumulated_uncorrect_attempts',                  'duration_mean',

                         'duration_std',           'accumulated_accuracy',                              0,

                                      1,                                2,

                                      3,     'accumulated_accuracy_group',

                  'accumulated_actions',                     'game_start',

                            'game_exit',              'game_instructions',

                   'game_obr_time_true',            'game_obr_time_false',

                  'game_events_special',                 'game_otvleksya',

                     'game_events_true',              'game_events_false']]

pred_1 = clf.predict(X_test)

pred_2 = cats.predict(X_test)

pred_3 = xgbcl.predict(X_test)

pred_4 = model.predict(X_ts)

# del X_test
full = pd.concat([pd.DataFrame(pred_1),pd.DataFrame(pred_2),pd.DataFrame(pred_3),

                  pd.DataFrame(pred_4).idxmax(1)], axis=1)

# preds = full.mode(axis=1)[0]

preds = full.mean(axis=1)

preds = preds.apply(lambda x: int(x))
preds
# # make predictions on test set once

# preds = clf.predict(X_test)

# del X_test

# make predictions on test set once

# X_test = X_test[[                          'Clip',                       'Activity',

#                            'Assessment',                           'Game',

#                         'session_title',   'accumulated_correct_attempts',

#        'accumulated_uncorrect_attempts',                  'duration_mean',

#                          'duration_std',           'accumulated_accuracy',                              0,

#                                       1,                                2,

#                                       3,     'accumulated_accuracy_group',

#                   'accumulated_actions',                     'game_start',

#                             'game_exit',              'game_instructions',

#                    'game_obr_time_true',            'game_obr_time_false',

#                   'game_events_special',                 'game_otvleksya',

#                      'game_events_true',              'game_events_false']]

# preds = clf.predict(X_test)

# del X_test
submission['accuracy_group'] = np.round(preds).astype('int')

submission.to_csv('submission.csv', index=None)

submission.head()
submission['accuracy_group'].plot(kind='hist')
train_labels['accuracy_group'].plot(kind='hist')
pd.Series(oof).plot(kind='hist')
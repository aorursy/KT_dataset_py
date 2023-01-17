import math

import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

import operator

from collections import Counter

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import json

from tqdm import tqdm
train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')



print('train', train.shape)

print('test', test.shape)

train.head()
keep_id = train[train.type == "Assessment"][['installation_id']].drop_duplicates()

train = pd.merge(train, keep_id, on="installation_id", how="inner")

print(train.shape)
def feature_extraction(input_df):

    """

    Extract keys and values from dict and store them in a DataFrame.

    

    input_df : Dataframe like train or test (intput)

    df : DataFrame (output)

    """

    df = pd.DataFrame()

    

    for i in tqdm(range(len(input_df))):

        temp = pd.DataFrame(json.loads(input_df['event_data'][i]).items()).transpose()

        temp.columns = temp.iloc[0]

        temp = temp.drop([0])



        df = pd.concat([df, temp], ignore_index=True)

    

    return df



extract = feature_extraction(train.iloc[:100,:])

extract
train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

submissions = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

print('train_labels', train_labels.shape)

print('submissions', submissions.shape)

train_labels.head()
plt.rcParams.update({'font.size': 16})



se = train_labels.groupby(['title', 'accuracy_group'])['accuracy_group'].count().unstack('title')

se.plot.bar(stacked=True, rot=0, figsize=(12,10))

plt.title("Counts of accuracy group")

plt.show()
train[~train.installation_id.isin(train_labels.installation_id.unique())].installation_id.nunique()
train = train[train.installation_id.isin(train_labels.installation_id.unique())]

train.shape
print('Nombre de num_correct = 1:', train_labels[train_labels['num_correct']==1].shape[0])

print('Nombre de num_correct = 0:', train_labels[train_labels['num_correct']==0].shape[0])

print('game_session unique:', len(train_labels['game_session'].drop_duplicates()))

print('installation_id unique:', len(train_labels['installation_id'].drop_duplicates()))
def get_accuracy(data):

    """

    input: data

    output: data_labels for each of the 5 games

    """

    

    df = pd.DataFrame()

    

    games = ['Bird Measurer', 'Cart Balancer', 'Cauldron Filler', 'Chest Sorter', 'Mushroom Sorter']

    

    # Filtre parmis 1 des 5 jeux (game)

    for game in games:

        tmp = data[data['title'].str.contains(game)]

        

        # Filtre dernier event = assessment 4110/4100 (code)

        if game == 'Bird Measurer':

            tmp = tmp[tmp['event_code'] == 4110]

        else:

            tmp = tmp[tmp['event_code'] == 4100]

    

        # num_correct and num_incorrect

        correct = ["NA" for i in range(np.shape(tmp)[0])]

        incorrect = ["NA" for i in range(np.shape(tmp)[0])]

        for i in range(np.shape(tmp)[0]):

            if ('correct":false' in tmp.loc[tmp.index[i], 'event_data']):

                correct[i] = 0

                incorrect[i] = 1

            elif ('correct":true' in tmp.loc[tmp.index[i], 'event_data']):

                correct[i] = 1

                incorrect[i] = 0

            else:

                correct[i] = 'NA'

                incorrect[i] = 'NA'

        tmp['num_correct'] = correct

        tmp['num_incorrect'] = incorrect

        tmp = pd.DataFrame(tmp.groupby(('installation_id','game_session','title')).sum())

            

        # accuracy

        accuracy = tmp['num_correct'] / (tmp['num_correct'] + tmp['num_incorrect'])

        tmp['accuracy'] = accuracy



        # accuracy_group

        tmp["accuracy_group"] = tmp["accuracy"].apply(lambda x: 0 if x==0 else (1 if x<0.5 else (2 if x<0.9 else 3)))

        df = pd.concat([df, tmp])

        

    df = df.reset_index()[['game_session','installation_id','title','num_correct','num_incorrect','accuracy','accuracy_group']]

    return(df)
my_train_labels = get_accuracy(train)

my_train_labels = my_train_labels.sort_values(['installation_id', 'game_session']).reset_index(drop=True)



print(np.shape(my_train_labels), np.shape(train_labels))



my_train_labels == train_labels
test_labels = get_accuracy(test)

test_labels = test_labels.sort_values(['installation_id', 'game_session']).reset_index(drop=True)



print(test_labels.shape)

test_labels.head()
test[test['event_data'].str.contains('false') & test['event_code'].isin([4100, 4110])]
from collections import Counter



#Credit to Erik Bruin



def encode_title(train, test, train_labels):

    # encode title

    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))

    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))

    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    # make a list with all the unique 'titles' from the train and test set

    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))

    # make a list with all the unique 'event_code' from the train and test set

    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))

    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))

    # make a list with all the unique worlds from the train and test set

    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))

    # create a dictionary numerating the titles

    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))

    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))

    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))

    # replace the text titles with the number titles from the dict

    train['title'] = train['title'].map(activities_map)

    test['title'] = test['title'].map(activities_map)

    train['world'] = train['world'].map(activities_world)

    test['world'] = test['world'].map(activities_world)

    train_labels['title'] = train_labels['title'].map(activities_map)

    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))

    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest

    win_code[activities_map['Bird Measurer (Assessment)']] = 4110

    # convert text into datetime

    train['timestamp'] = pd.to_datetime(train['timestamp'])

    test['timestamp'] = pd.to_datetime(test['timestamp'])

    

    

    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code



def get_data(user_sample, test_set=False):

    '''

    The user_sample is a DataFrame from train or test where the only one 

    installation_id is filtered

    And the test_set parameter is related with the labels processing, that is only requered

    if test_set=False

    '''

    # Constants and parameters declaration

    last_activity = 0

    

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    

    # new features: time spent in each activity

    last_session_time_sec = 0

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    all_assessments = []

    accumulated_accuracy_group = 0

    accumulated_accuracy = 0

    accumulated_correct_attempts = 0 

    accumulated_uncorrect_attempts = 0

    accumulated_actions = 0

    counter = 0

    time_first_activity = float(user_sample['timestamp'].values[0])

    durations = []

    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}

    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}

    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}

    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 

    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}

    

    # itarates through each session of one instalation_id

    for i, session in user_sample.groupby('game_session', sort=False):

        # i = game_session_id

        # session is a DataFrame that contain only one game_session

        

        # get some sessions information

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        session_title_text = activities_labels[session_title]

                    

            

        # for each assessment, and only this kind off session, the features below are processed

        # and a register are generated

        if (session_type == 'Assessment') & (test_set or len(session)>1):

            # search for event_code 4100, that represents the assessments trial

            all_attempts = session.query(f'event_code == {win_code[session_title]}')

            # then, check the numbers of wins and the number of losses

            true_attempts = all_attempts['event_data'].str.contains('true').sum()

            false_attempts = all_attempts['event_data'].str.contains('false').sum()

            # copy a dict to use as feature template, it's initialized with some itens: 

            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

            features = user_activities_count.copy()

            features.update(last_accuracy_title.copy())

            features.update(event_code_count.copy())

            features.update(event_id_count.copy())

            features.update(title_count.copy())

            features.update(title_event_code_count.copy())

            features.update(last_accuracy_title.copy())

            

            # get installation_id for aggregated features

            features['installation_id'] = session['installation_id'].iloc[-1]

            # add title as feature, remembering that title represents the name of the game

            features['session_title'] = session['title'].iloc[0]

            # the 4 lines below add the feature of the history of the trials of this player

            # this is based on the all time attempts so far, at the moment of this assessment

            features['accumulated_correct_attempts'] = accumulated_correct_attempts

            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts

            accumulated_correct_attempts += true_attempts 

            accumulated_uncorrect_attempts += false_attempts

            # the time spent in the app so far

            if durations == []:

                features['duration_mean'] = 0

            else:

                features['duration_mean'] = np.mean(durations)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            # the accurace is the all time wins divided by the all time attempts

            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0

            accumulated_accuracy += accuracy

            last_accuracy_title['acc_' + session_title_text] = accuracy

            # a feature of the current accuracy categorized

            # it is a counter of how many times this player was in each accuracy group

            if accuracy == 0:

                features['accuracy_group'] = 0

            elif accuracy == 1:

                features['accuracy_group'] = 3

            elif accuracy == 0.5:

                features['accuracy_group'] = 2

            else:

                features['accuracy_group'] = 1

            features.update(accuracy_groups)

            accuracy_groups[features['accuracy_group']] += 1

            # mean of the all accuracy groups of this player

            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0

            accumulated_accuracy_group += features['accuracy_group']

            # how many actions the player has done so far, it is initialized as 0 and updated some lines below

            features['accumulated_actions'] = accumulated_actions

            

            # there are some conditions to allow this features to be inserted in the datasets

            # if it's a test set, all sessions belong to the final dataset

            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')

            # that means, must exist an event_code 4100 or 4110

            if test_set:

                all_assessments.append(features)

            elif true_attempts+false_attempts > 0:

                all_assessments.append(features)

                

            counter += 1

        

        # this piece counts how many actions was made in each event_code so far

        def update_counters(counter: dict, col: str):

                num_of_session_count = Counter(session[col])

                for k in num_of_session_count.keys():

                    x = k

                    if col == 'title':

                        x = activities_labels[k]

                    counter[x] += num_of_session_count[k]

                return counter

            

        event_code_count = update_counters(event_code_count, "event_code")

        event_id_count = update_counters(event_id_count, "event_id")

        title_count = update_counters(title_count, 'title')

        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')



        # counts how many actions the player has done so far, used in the feature of the same name

        accumulated_actions += len(session)

        if last_activity != session_type:

            user_activities_count[session_type] += 1

            last_activitiy = session_type 

                        

    # if it't the test_set, only the last assessment must be predicted, the previous are scraped

    if test_set:

        return all_assessments[-1]

    # in the train_set, all assessments goes to the dataset

    return all_assessments



def get_train_and_test(train, test):

    compiled_train = []

    compiled_test = []

    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 3614):

        compiled_train += get_data(user_sample)

    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):

        test_data = get_data(user_sample, test_set = True)

        compiled_test.append(test_data)

    reduce_train = pd.DataFrame(compiled_train)

    reduce_test = pd.DataFrame(compiled_test)

    categoricals = ['session_title']

    return reduce_train, reduce_test, categoricals



# get usefull dict with maping encode

train2, test2, train_labels2, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)

# tranform function to get the train and test set

reduce_train, reduce_test, categoricals = get_train_and_test(train2, test2)



print(reduce_train.shape)

print(reduce_test.shape)
def preprocess(reduce_train, reduce_test):

    for df in [reduce_train, reduce_test]:

        df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')

        df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')

        #df['installation_duration_std'] = df.groupby(['installation_id'])['duration_mean'].transform('std')

        df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')

        

        df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 

                                        4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 

                                        2040, 4090, 4220, 4095]].sum(axis = 1)

        

        df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')

        #df['installation_event_code_count_std'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('std')

        

    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns

    features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in assess_titles]

   

    return reduce_train, reduce_test, features

# call feature engineering function

reduce_train, reduce_test, features = preprocess(reduce_train, reduce_test)



print(reduce_train.shape)

print(reduce_test.shape)

reduce_train.head()
#on importe les données

reduced_truncated_data = pd.read_csv("../input/reduced-truncated-data/reduced_truncated_data.csv")

accuracy_group_to_predict = pd.read_csv('../input/accuracy-group-to-predict/accuracy_group_to_predict.csv')





#on ne garde que les colonnes communes, afin que le modèle puisse s'appliquer sur l'un puis sur l'autre

reduced_truncated_data = reduced_truncated_data[reduced_truncated_data.columns.intersection(reduce_test.columns)]

reduced_test = reduce_test[reduced_truncated_data.columns.intersection(reduce_test.columns)]

print(reduced_test.shape)
#On obtient Y contenant l'accuracy_group pour chaque ligne de reduced_truncated_data

Y = pd.merge(reduced_truncated_data, accuracy_group_to_predict, on='installation_id', how='outer')

Y = Y[["installation_id","accuracy_group_y"]]

Y.columns = ['installation_id', 'accuracy_group']
print(pd.isnull(reduced_truncated_data).any(axis=1).any(axis=0)) #aucun NA

print(pd.isnull(Y).any(axis=1).any(axis=0)) #au moins un NA



rows_to_keep = list(map(operator.not_,pd.isnull(Y[Y.columns[1]])))

Y = Y[rows_to_keep]

reduced_truncated_data = reduced_truncated_data[rows_to_keep]



print(np.shape(reduced_truncated_data))
print(np.shape(Y))

print(np.shape(reduced_truncated_data))

print(np.shape(reduced_test))

print(np.shape(accuracy_group_to_predict))

Y.head()
#il faut enlever les colonnes non numériques 

msk = reduced_truncated_data.dtypes == np.object

print(reduced_truncated_data.loc[:,msk].columns)



X = reduced_truncated_data.copy()

del X['installation_id']



Y2 = Y.copy()

del Y2['installation_id']



print(np.shape(X))

print(np.shape(reduced_truncated_data))

model = LinearRegression().fit(X, Y2)


y_pred = model.predict(X)

y_pred = np.around(y_pred)

print(y_pred)



#les valeurs sont-elles toutes 0,1,2 ou3 ? Si oui, on les transforme en 0 ou 3

msk = y_pred>3

print(y_pred[msk])

y_pred[msk] = 3



msk = y_pred<0

print(y_pred[msk])

y_pred[msk] = 0
#Par rapport aux vraies valeurs, quelle est notre proportion de bonnes valeurs

compar = (Y2 == y_pred)

compar[compar.columns[0]].value_counts()

#Rq : ici, on a un accuracy_group par ligne (par action) alors qu'on veut seulement l'accuracy_group pour la dernière action
#on récupère les installations id qu'on avait dû enlever pour lancer l'apprentissage

install_ids = pd.DataFrame(reduced_truncated_data['installation_id']).reset_index(drop=True)



#on obtient l'accuracy_group par ligne, avec à chaque fois l'installation_id

y_pred2 = pd.concat([install_ids,pd.DataFrame(y_pred)], axis=1)

y_pred2.columns = ['installation_id', 'accuracy_group']
#On récupère maintenant l'accuracy correspondant à la dernière ligne de chaque installation_id, c'est-à-dire à l'assessment à prédire

def accuracy_by_installation_id(y_pred2):

    unique_id = np.unique(y_pred2['installation_id'])

    

    new_data = pd.DataFrame(columns = ["installation_id","accuracy_group"])

    

    for id in unique_id:

        #last line, so the assessment we want to predict

        last_truncated_id = y_pred2[y_pred2['installation_id'] == id].tail(1)



        #Update new_data

        new_data = pd.concat([new_data, last_truncated_id])

    return(new_data.reset_index(drop=True))





y_pred_final = accuracy_by_installation_id(y_pred2)

Y_final = accuracy_by_installation_id(Y)
#Enfin, on compare l'accruacy_group pour chaque assessment qu'on avait à prédire

compar = (Y_final == y_pred_final)

compar[compar.columns[0]].value_counts()
#On a donc que des réussites, ce qui est rassurant car ça veut dire que l'entrainement s'est bien passé

#Toutefois, il y a éventuellement un risque de surapprentissage

#Passons maintenant aux données à prédire (reduced_test)



#il faut enlever les colonnes non numériques 

msk = reduced_test.dtypes == np.object

print(reduced_test.loc[:,msk].columns)



X_test = reduced_test.copy()

del X_test['installation_id']
y_pred_test = model.predict(X_test)

y_pred_test = np.around(y_pred_test)



#les valeurs sont-elles toutes 0,1,2 ou3 ? Si non, on les transforme en 0 ou 3

msk = y_pred_test>3

print(y_pred_test[msk])

y_pred_test[msk] = 3



msk = y_pred_test<0

print(y_pred_test[msk])

y_pred_test[msk] = 0
#on récupère les installations id qu'on avait dû enlever pour lancer l'apprentissage

install_ids = pd.DataFrame(reduced_test['installation_id']).reset_index(drop=True)



#on obtient l'accuracy_group par ligne, avec à chaque fois l'installation_id

y_pred_test2 = pd.concat([install_ids,pd.DataFrame(y_pred_test)], axis=1)

y_pred_test2.columns = ['installation_id', 'accuracy_group']
y_pred_test_final = accuracy_by_installation_id(y_pred_test2)

y_pred_test_final["accuracy_group"] = y_pred_test_final["accuracy_group"].astype(int)

print(y_pred_test_final.head())
submissions = y_pred_test_final

submissions.to_csv('submission2.csv', index=False)
import pandas

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline



from sklearn.model_selection import train_test_split

from keras.optimizers import SGD, Adam, RMSprop

from keras.layers import Dropout

from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import PReLU

from keras.callbacks import ReduceLROnPlateau

from sklearn.base import BaseEstimator, TransformerMixin

from keras.callbacks import Callback

from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import StandardScaler
y = reduce_train['accuracy_group']



X =reduce_train[features]



dummy_y = np_utils.to_categorical(y)



input_dim= X.shape[1]

print('input_dim is:', input_dim)

features = X.columns

X.head(5)
def qwk(a1, a2):

    """

    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168



    :param a1:

    :param a2:

    :param max_rat:

    :return:

    """

    max_rat = 3

    a1 = np.asarray(a1, dtype=int)

    a2 = np.asarray(a2, dtype=int)



    hist1 = np.zeros((max_rat + 1, ))

    hist2 = np.zeros((max_rat + 1, ))



    o = 0

    for k in range(a1.shape[0]):

        i, j = a1[k], a2[k]

        hist1[i] += 1

        hist2[j] += 1

        o +=  (i - j) * (i - j)



    e = 0

    for i in range(max_rat + 1):

        for j in range(max_rat + 1):

            e += hist1[i] * hist2[j] * (i - j) * (i - j)



    e = e / a1.shape[0]



    return 1 - o / e





def eval_qwk_lgb(y_true, y_pred):

    """

    Fast cappa eval function for lgb.

    """



    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)

    return 'cappa', qwk(y_true, y_pred), True
class roc_callback(Callback):

    def __init__(self,training_data,validation_data):

        self.x = training_data[0]

        self.y = training_data[1]

        self.x_val = validation_data[0]

        self.y_val = validation_data[1]



    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict(self.x)

        roc = eval_qwk_lgb(self.y, y_pred)

        y_pred_val = self.model.predict(self.x_val)

        roc_val = eval_qwk_lgb(self.y_val, y_pred_val)

        print('\rqwk: %s - qwk_val: %s' % (str(roc),str(roc_val)),end=100*' '+'\n')

        return
sc = StandardScaler()

X = sc.fit_transform(X)



model = Sequential()

model.add(Dense(80, input_dim=input_dim, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(200,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))



# model.add(Dense(300, input_dim=input_dim, activation='relu'))

# model.add(BatchNormalization())

# model.add(Dropout(0.5))

# model.add(Dense(200, activation='relu'))

# model.add(BatchNormalization())

# model.add(Dropout(0.5))

# model.add(Dense(150, activation='relu'))

# model.add(BatchNormalization())

# model.add(Dropout(0.5))

# model.add(Dense(100, activation='relu'))

# model.add(BatchNormalization())

# model.add(Dropout(0.5))

# model.add(Dense(4, activation='softmax'))

# Compile model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])



train_x, valid_x , train_y, valid_y = train_test_split(X, dummy_y, test_size=0.2, random_state=2020)



from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.008)

model.fit(train_x, train_y, batch_size = 32, epochs = 100,validation_data=(valid_x, valid_y),

               callbacks=[reduce_lr,roc_callback(training_data=(train_x, train_y),validation_data=(valid_x, valid_y)),early_stopping],verbose=1)
preds = model.predict(sc.transform(reduce_test[features]))



sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')

sample_submission['accuracy_group'] = pd.DataFrame(preds).idxmax(axis=1).astype(int)

sample_submission.to_csv('submission.csv', index=False)
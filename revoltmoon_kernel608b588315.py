import numpy as np

import pandas as pd

import os

import copy

import xgboost as xgb

from typing import List

from collections import Counter

from tqdm import tqdm

pd.options.display.precision = 15

pd.set_option('max_rows', 500)



def read_input():

    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

    return train, test, train_labels, specs, sample_submission
def encode_title(train, test, train_labels):

    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))

    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))

    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    # делаем list с уникальными 'titles' из тренировочных и тестовых данных

    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))

    # делаем list с уникальными 'event_code' из тренировочных и тестовых данных

    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))

    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))

    # делаем list с уникальными world 

    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))

    # словарь с нумерацией заголовков

    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))

    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))

    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))

    # заменяем текстовые заголовки номерами

    train['title'] = train['title'].map(activities_map)

    test['title'] = test['title'].map(activities_map)

    train['world'] = train['world'].map(activities_world)

    test['world'] = test['world'].map(activities_world)

    train_labels['title'] = train_labels['title'].map(activities_map)

    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))

    win_code[activities_map['Bird Measurer (Assessment)']] = 4110

    # конвертируем text в datetime

    train['timestamp'] = pd.to_datetime(train['timestamp'])

    test['timestamp'] = pd.to_datetime(test['timestamp'])

    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code
def get_data(user_sample, test_set=False):

    last_activity = 0

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    # время затраченное на каждое занятие

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

    

    for i, session in user_sample.groupby('game_session', sort=False):

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        session_title_text = activities_labels[session_title]

        if (session_type == 'Assessment') & (test_set or len(session)>1):

            # ищем event_code 4100, это представляет собой оценку испытаний

            all_attempts = session.query(f'event_code == {win_code[session_title]}')

            # затем проверяем количество выигрышей и проигрышей

            true_attempts = all_attempts['event_data'].str.contains('true').sum()

            false_attempts = all_attempts['event_data'].str.contains('false').sum()

            # копируем словарь для использования в качетсве шаблона

            features = user_activities_count.copy()

            features.update(last_accuracy_title.copy())

            features.update(event_code_count.copy())

            features.update(event_id_count.copy())

            features.update(title_count.copy())

            features.update(title_event_code_count.copy())

            features.update(last_accuracy_title.copy())

            

            # получаем installation_id для агрегированных функций

            features['installation_id'] = session['installation_id'].iloc[-1]

            # добавляем title(название игры) как функцию 

            features['session_title'] = session['title'].iloc[0]

            # история испытаний игрока(на момент оценки)

            features['accumulated_correct_attempts'] = accumulated_correct_attempts

            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts

            accumulated_correct_attempts += true_attempts 

            accumulated_uncorrect_attempts += false_attempts

            # затраченное время

            if durations == []:

                features['duration_mean'] = 0

            else:

                features['duration_mean'] = np.mean(durations)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            # точность - выигрыш за все время/на все попытки

            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0

            accumulated_accuracy += accuracy

            last_accuracy_title['acc_' + session_title_text] = accuracy

            # точность по категориям

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

            # среднее значение групп точности игрока

            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0

            accumulated_accuracy_group += features['accuracy_group']

            # сколько действий совершил игрок обновляется ниже

            features['accumulated_actions'] = accumulated_actions

            # Есть некоторые условия, позволяющие вставлять эти функции в наборы данных если это тестовый набор, все сеансы принадлежат к окончательному набору данных если это тренировочный, его нужно пропустить через этот пункт: session.query (f'event_code == {win_code [session_title]} ') это значит, должен существовать код события 4100 или 4110

            if test_set:

                all_assessments.append(features)

            elif true_attempts+false_attempts > 0:

                all_assessments.append(features)

                

            counter += 1

        

        # сколько действий было сделано в каждом event_code

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



        # сколько действий совершил игрок

        accumulated_actions += len(session)

        if last_activity != session_type:

            user_activities_count[session_type] += 1

            last_activitiy = session_type 

                        

    # если не test_set, только последняя должна быть предсказана

    if test_set:

        return all_assessments[-1]

    # в train_set, все оценки летят в датасет

    return all_assessments



def get_training_and_testing(train, test):

    compiled_train = []

    compiled_test = []

    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):

        compiled_train += get_data(user_sample)

    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):

        test_data = get_data(user_sample, test_set = True)

        compiled_test.append(test_data)

    reduce_train = pd.DataFrame(compiled_train)

    reduce_test = pd.DataFrame(compiled_test)

    categoricals = ['session_title']

    return reduce_train, reduce_test, categoricals
train, test, train_labels, specs, sample_submission = read_input()

train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)

reduce_train, reduce_test, categoricals = get_training_and_testing(train, test)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4641, reg_lambda=0.8572,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)
y = reduce_train['accuracy_group']

reduce_train.drop([ 'installation_id','accuracy_group'], inplace = True, axis = 1)

model_xgb.fit(X=reduce_train, y=y)

reduce_test.drop([ 'installation_id','accuracy_group'], inplace = True, axis = 1)

preds = model_xgb.predict(reduce_test)

df_ajust = pd.DataFrame(preds, columns = ['preds'])



q1 = df_ajust['preds'].quantile(0.0045)

q2 = df_ajust['preds'].quantile(0.99)

df_ajust['preds'] = df_ajust['preds'].apply(lambda x: x if x > q1 else x*0.77)

df_ajust['preds'] = df_ajust['preds'].apply(lambda x: x if x < q2 else x*1.1)



pr1 = df_ajust['preds'].values

pr1[pr1 <= 1.12232214] = 0

pr1[np.where(np.logical_and(pr1 > 1.12232214, pr1 <= 1.73925866))] = 1

pr1[np.where(np.logical_and(pr1 > 1.73925866, pr1 <= 2.22506454))] = 2

pr1[pr1 > 2.22506454] = 3

sample_submission['accuracy_group'] = pr1.astype(int)

sample_submission.to_csv('submission.csv', index=False)
import numpy as np

import pandas as pd

import matplotlib.pylab as plt

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Activation, Embedding

from keras.layers import LSTM, SpatialDropout1D
train = pd.read_csv('../input/data-science-bowl-2019/train.csv')

labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

sample = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
labels['accuracy_group'].hist()
labels_0 = labels.loc[labels['accuracy_group'] == 0]

labels_1 = labels.loc[labels['accuracy_group'] == 1]

labels_2 = labels.loc[labels['accuracy_group'] == 2]

labels_3 = labels.loc[labels['accuracy_group'] == 3]
labels_0 = labels_0.iloc[:2205]

labels_1 = labels_1.iloc[:2205]

labels_2 = labels_2.iloc[:2205]

labels_3 = labels_3.iloc[:2205]
labels = pd.concat([labels_0, labels_1, labels_2, labels_3])
labels = labels.sort_index()
labels['accuracy_group'].hist()
# очищаем train от данных об игроках, которых нет в labels

train = train.loc[train['installation_id'].isin(labels['installation_id'].unique())]
new_columns = ['world_clip', 

               'world_activity', 

               'world_game', 

               'world_assessment', 

               'other_clip', 

               'other_activity', 

               'other_game', 

               'other_assessment']

for i in new_columns:

    labels[i] = 0
%%time

prev_installation_id = 0

for i, row in labels[['game_session', 'installation_id']].iterrows():

    next_installation_id = row['installation_id']

    game_session = row['game_session']

    # находим в train все записи по конкретному installation_id из labels, при условии,

    # что он отличается от предыдущего

    if next_installation_id != prev_installation_id:

        data_player = train.loc[train['installation_id'] == next_installation_id]

    # находим индекс первой строки с проверяемой game_session

    index = data_player[data_player['game_session'] == game_session].index[0]

    # локация, в которой игрок проходит испытание

    world = data_player['world'][index]

    # выбираем весь игровой опыт до проверяемой game_session

    # -1 потому что loc включает последнюю строку тоже

    game_experience = data_player.loc[:(index - 1)]



    for game_session in game_experience['game_session'].unique():

        # находим индекс последней строки каждой игровой сессии, чтобы взять оттуда игровое время

        time_index = game_experience[game_experience['game_session'] == game_session].index[-1]

        game_time = game_experience['game_time'][time_index] // 1000

        if game_time != 0:

            # определяем мир, в котором играл ребенок

            game_world = game_experience['world'][time_index]

            # определяем тип активности

            activity_type = game_experience['type'][time_index]

            # добавляем игровое время в соответствующий столбец

            if game_world == world:

                if activity_type == 'Clip':

                    labels['world_clip'][i] += game_time

                elif activity_type == 'Activity':

                    labels['world_activity'][i] += game_time

                elif activity_type == 'Game':

                    labels['world_game'][i] += game_time

                elif activity_type == 'Assessment':

                    labels['world_assessment'][i] += game_time

            else:

                if activity_type == 'Clip':

                    labels['other_clip'][i] += game_time

                elif activity_type == 'Activity':

                    labels['other_activity'][i] += game_time

                elif activity_type == 'Game':

                    labels['other_game'][i] += game_time

                elif activity_type == 'Assessment':

                    labels['other_assessment'][i] += game_time

    prev_installation_id = next_installation_id    
# определяем количество уникальных значений, которые будут передаваться в НС

max_features = 0

for column in new_columns:

    max_in_column = labels[column].max()

    if max_in_column > max_features:

        max_features = max_in_column

max_features += 1
X = labels[new_columns].values
Y = pd.get_dummies(labels['accuracy_group']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
model = Sequential()

model.add(Embedding(max_features, 100, input_length=X.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
%%time

epochs = 2

batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1)
accr = model.evaluate(X_test,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
plt.title('Loss')

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show();
for i in new_columns:

    sample[i] = 0
%%time

for i, row in sample.iterrows():

    installation_id = row['installation_id']

    data_player = test.loc[test['installation_id'] == installation_id]

    # находим индекс строки с проверяемой game_session

    index = data_player['installation_id'].index[-1]

    # локация, в которой игрок проходит последнее испытание

    world = data_player['world'][index]

    

    for game_session in data_player['game_session'].unique():

        # находим индекс последней строки каждой игровой сессии, чтобы взять оттуда игровое время

        time_index = data_player[data_player['game_session'] == game_session].index[-1]

        game_time = data_player['game_time'][time_index] // 1000

        if game_time != 0:

            # определяем мир, в котором играл ребенок

            game_world = data_player['world'][time_index]

            # определяем тип активности

            activity_type = data_player['type'][time_index]

            # добавляем игровое время в соответствующий столбец

            if game_world == world:

                if activity_type == 'Clip':

                    sample['world_clip'][i] += game_time

                elif activity_type == 'Activity':

                    sample['world_activity'][i] += game_time

                elif activity_type == 'Game':

                    sample['world_game'][i] += game_time

                elif activity_type == 'Assessment':

                    sample['world_assessment'][i] += game_time

            else:

                if activity_type == 'Clip':

                    sample['other_clip'][i] += game_time

                elif activity_type == 'Activity':

                    sample['other_activity'][i] += game_time

                elif activity_type == 'Game':

                    sample['other_game'][i] += game_time

                elif activity_type == 'Assessment':

                    sample['other_assessment'][i] += game_time
X_test2 = sample[new_columns].values
test_pred = model.predict(X_test2)
submission = pd.concat([sample['installation_id'], pd.DataFrame(test_pred).idxmax(1)], axis=1)

submission.columns = ['installation_id','accuracy_group']
submission.to_csv('submission.csv', index=None)
submission['accuracy_group'].hist()
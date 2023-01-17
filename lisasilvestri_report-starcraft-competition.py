import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

import matplotlib.pyplot as plt
trainingset = "../input/starcraft-2-player-prediction-challenge-2019/TRAIN.CSV"

testset = "../input/starcraft-2-player-prediction-challenge-2019/TEST.CSV"



train_data = pd.read_fwf(trainingset, header=None)

train_data = train_data[0].str.split(',', expand=True)



test_data = pd.read_fwf(testset, header=None)

test_data = test_data[0].str.split(',', expand=True)



#loading of the training data



#creation of the matrix to store the players activity

train_columns_mode = ['id_list','id_player','race','s','f00','f01','f02','f10','f11','f12','f20','f21','f22','f30','f31','f32','f40','f41','f42','f50','f51','f52','f60','f61','f62','f70','f71','f72','f80','f81','f82','f90','f91','f92']

train_columns = ['id_list','id_player', 'race','s','f1','f2','f3','f4','f5','f6','f7','f8','f9','f0']

train_dataframe = pd.DataFrame([], columns=train_columns)

train_dataframe_mode = pd.DataFrame([], columns=train_columns_mode)



#creation of the matrix to store the players activity

test_columns_mode = ['id_list','race','s','f00','f01','f02','f10','f11','f12','f20','f21','f22','f30','f31','f32','f40','f41','f42','f50','f51','f52','f60','f61','f62','f70','f71','f72','f80','f81','f82','f90','f91','f92']

test_columns = ['id_list',

               'race','s','f1','f2','f3','f4','f5','f6','f7','f8','f9','f0']

test_dataframe = pd.DataFrame([], columns=test_columns)

test_dataframe_mode = pd.DataFrame([], columns=test_columns_mode)
def loadMatrix(data, dataframe, train):

    id_list = 0

    for index in data.iterrows():

        features = [0 for _ in range(10)]

        time = 1

        s_nb = 0

        for value in index[1]:

            if value:

                if ("hotkey" in value):

                    hotkey_number = value[-2]

                    try:

                        hotkey_number = int(hotkey_number)

                        features[hotkey_number] += 1

                    except:

                        print("Error: no int in hotkey")

                elif (value == "s"):

                    s_nb += 1

                elif (value[0] == "t"):

                    try:

                        time = int(value[1:])

                    except:

                        print("Error: Time is wrong") 

                        print(value[1:])

        #Insert into dataframe

        if train and time>=30:

            id_player = index[1][0]

            race = index[1][1]

            df = pd.DataFrame([[id_list, id_player, race, s_nb/time, features[1]/time, features[2]/time, features[3]/time, features[4]/time, features[5]/time, features[6]/time, features[7]/time, features[8]/time, features[9]/time, features[0]/time]], columns=['id_list',

               'id_player','race','s','f1','f2','f3','f4','f5','f6','f7','f8','f9','f0'])

            dataframe = dataframe.append(df)

            id_list += 1

        elif not train:

            race = index[1][0]

            df = pd.DataFrame([[id_list, race, s_nb/time, features[1]/time, features[2]/time, features[3]/time, features[4]/time, features[5]/time, features[6]/time, features[7]/time, features[8]/time, features[9]/time, features[0]/time]], columns=['id_list',

               'race','s','f1','f2','f3','f4','f5','f6','f7','f8','f9','f0'])

            dataframe = dataframe.append(df)

            id_list += 1

    return dataframe



def loadMatrix_mode(data, dataframe, train):

    id_list = 0

    for index in data.iterrows():

        features = [0 for _ in range(30)]

        time = 1

        s_nb = 0

        for value in index[1]:

            if value:

                if ("hotkey" in value):

                    hotkey_number = value[-2]

                    hotkey_mode = value[-1]

                    try:

                        hotkey_number = int(hotkey_number)

                        hotkey_mode = int(hotkey_mode)

                        features[hotkey_number*3 + hotkey_mode] += 1

                    except:

                        print("Error: no int in hotkey")

                elif (value == "s"):

                    s_nb += 1

                elif (value[0] == "t"):

                    try:

                        time = int(value[1:])

                    except:

                        print("Error: Time is wrong") 

                        print(value[1:])

        #Insert into dataframe

        if train and time>=30:

            id_player = index[1][0]

            race = index[1][1]

            df = pd.DataFrame([[id_list, id_player, race, s_nb/time, features[0]/time, features[1]/time, features[2]/time, features[3]/time, features[4]/time, features[5]/time, features[6]/time, features[7]/time, features[8]/time, features[9]/time, features[10]/time, features[11]/time, features[12]/time, features[13]/time, features[14]/time, features[15]/time, features[16]/time, features[17]/time, features[18]/time, features[19]/time, features[20]/time, features[21]/time, features[22]/time, features[23]/time, features[24]/time, features[25]/time, features[26]/time, features[27]/time, features[28]/time, features[29]/time]], columns=train_columns_mode)

            dataframe = dataframe.append(df)

            id_list += 1

        elif not train:

            race = index[1][0]

            df = pd.DataFrame([[id_list, race, s_nb/time,features[0]/time, features[1]/time, features[2]/time, features[3]/time, features[4]/time, features[5]/time, features[6]/time, features[7]/time, features[8]/time, features[9]/time, features[10]/time, features[11]/time, features[12]/time, features[13]/time, features[14]/time, features[15]/time, features[16]/time, features[17]/time, features[18]/time, features[19]/time, features[20]/time, features[21]/time, features[22]/time, features[23]/time, features[24]/time, features[25]/time, features[26]/time, features[27]/time, features[28]/time, features[29]/time]], columns=test_columns_mode)

            dataframe = dataframe.append(df)

            id_list += 1

    return dataframe
train_dataframe = loadMatrix(train_data, train_dataframe, True)

test_dataframe = loadMatrix(test_data, test_dataframe, False)



train_dataframe_mode = loadMatrix_mode(train_data, train_dataframe_mode, True)

test_dataframe_mode = loadMatrix_mode(test_data, test_dataframe_mode, False)



#Using one hot encoding for races

test_dataframe = pd.get_dummies(test_dataframe, columns=['race'])

train_dataframe = pd.get_dummies(train_dataframe, columns=['race'])



test_dataframe_mode = pd.get_dummies(test_dataframe_mode, columns=['race'])

train_dataframe_mode = pd.get_dummies(train_dataframe_mode, columns=['race'])
#separating between labels and data

attributes = train_dataframe.iloc[:,2:].values

labels = train_dataframe.iloc[:,1].values

tests = test_dataframe.iloc[:,1:].values



#train and evaluate the classifier by separating training data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(attributes, labels, test_size=0.2, random_state=30)



classifier = RandomForestClassifier(n_estimators=400, random_state=20)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)



print("score with hotkeys : ", classifier.score(X_test,y_test))



#modes

attributes_mode = train_dataframe_mode.iloc[:,2:].values

labels_mode = train_dataframe_mode.iloc[:,1].values

tests_mode = test_dataframe_mode.iloc[:,1:].values



X_train_mode, X_test_mode, y_train_mode, y_test_mode = train_test_split(attributes_mode, labels_mode, test_size=0.2, random_state=30)

classifier.fit(X_train_mode, y_train_mode)

y_pred_mode = classifier.predict(X_test_mode)



print("score with hotkeys and mode : ", classifier.score(X_test_mode,y_test_mode))
#coying dataframe to avoid data corruption and converting player id to number

test_dataframe = train_dataframe = pd.DataFrame([], columns=train_columns)

test_dataframe = loadMatrix(train_data, test_dataframe, True)



test_dataframe = pd.get_dummies(test_dataframe, columns=['race'])

test_dataframe['id_player'] = test_dataframe['id_player'].astype('category').cat.codes



var_correlation = test_dataframe.corr()

sns.heatmap(var_correlation, xticklabels=var_correlation.columns, yticklabels=var_correlation.columns, annot=False)
test_datraframe_mode = train_dataframe_mode.copy()

test_datraframe_mode['id_player'] = test_datraframe_mode['id_player'].astype('category').cat.codes



var_correlation_mode = test_datraframe_mode.corr()

sns.heatmap(var_correlation_mode, xticklabels=var_correlation_mode.columns, yticklabels=var_correlation_mode.columns, annot=False)
optimal_time_cut = 60

optimal_minimal_time = 30



def loadMatrix_time(data, dataframe, train):

    id_list = 0

    for index in data.iterrows():

        features = [0 for _ in range(10)]

        time = 1

        s_nb = 0

        for value in index[1]:

            if value:

                if time <=optimal_time_cut:

                    if ("hotkey" in value):

                        hotkey_number = value[-2]

                        try:

                            hotkey_number = int(hotkey_number)

                            features[hotkey_number] += 1

                        except:

                            print("Error: no int in hotkey")

                    elif (value == "s"):

                        s_nb += 1

                if (value[0] == "t"):

                    try:

                        time = int(value[1:])

                    except:

                        print("Error: Time is wrong") 

                        print(value[1:])

        #Insert into dataframe

        if train and time>=optimal_minimal_time:

            id_player = index[1][0]

            race = index[1][1]

            df = pd.DataFrame([[id_list, id_player, race, s_nb/time, features[1]/time, features[2]/time, features[3]/time, features[4]/time, features[5]/time, features[6]/time, features[7]/time, features[8]/time, features[9]/time, features[0]/time]], columns=['id_list',

               'id_player','race','s','f1','f2','f3','f4','f5','f6','f7','f8','f9','f0'])

            dataframe = dataframe.append(df)

            id_list += 1

        elif not train:

            race = index[1][0]

            df = pd.DataFrame([[id_list, race, s_nb/time, features[1]/time, features[2]/time, features[3]/time, features[4]/time, features[5]/time, features[6]/time, features[7]/time, features[8]/time, features[9]/time, features[0]/time]], columns=['id_list',

               'race','s','f1','f2','f3','f4','f5','f6','f7','f8','f9','f0'])

            dataframe = dataframe.append(df)

            id_list += 1

    return dataframe
dataframe =  pd.DataFrame([], columns=train_columns)

dataframe = loadMatrix_time(train_data, dataframe, True)

dataframe = pd.get_dummies(dataframe, columns=['race'])



attributes = dataframe.iloc[:,2:16].values

labels = dataframe.iloc[:,1].values

tests = test_dataframe.iloc[:,1:15].values



X_train, X_test, y_train, y_test = train_test_split(attributes, labels, test_size=0.2, random_state=30)



classifier = RandomForestClassifier(n_estimators=400, random_state=20)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)



print("score with optimal time : ", classifier.score(X_test,y_test))



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder





def read_train_file(train_path):

    train_df = pd.read_csv(train_path)

    rows, columns = train_df.shape

    # MAP WILDERNESS TO CLASSIFICATION TYPE

    wilderness_area_list = sorted(list(set(train_df['Wilderness_Area'])))

    wilderness_area_dict = {k: v for v, k in enumerate(wilderness_area_list)}

    train_df['Wilderness_Area'] = train_df['Wilderness_Area'].map(wilderness_area_dict)

    wilderness_list = train_df['Wilderness_Area']

    # MAP COVER_TYPE TO CLASSIFICATION TYPE

    cover_type_list = sorted(list(set(train_df['Cover_Type'])))

    cover_type_dict = {k: v for v, k in enumerate(cover_type_list)}

    train_df['Cover_Type'] = train_df['Cover_Type'].map(cover_type_dict)

    cover_list = train_df['Cover_Type']

    # CLASSIFY SOIL_TYPE

    soil_type_list = list(train_df['Soil_Type'])

    climate_type_list = []

    geo_type_list = []

    for soil_type in soil_type_list:

        climate_type = int(soil_type / 1000)

        climate_type_list.append(climate_type)

        geo_type = int((soil_type % 1000) / 100)

        geo_type_list.append(geo_type)

    yTr = np.array(train_df['Cover_Type']).reshape(rows, 1)

    # NORMALIZE TRAINNING DATA SET

    train_df = train_df.iloc[:, 1 : -3]

    train_df_mean = train_df.mean()

    train_df_std = train_df.std()

    normalized_df = normalize(train_df, train_df_mean, train_df_std)

    # COMBINE TRAINING DATA SET INTO NUMPY ARRAY

    class_df = pd.DataFrame({

        'Wilderness_Area': wilderness_list,

        'Climater_Type': climate_type_list,

        'Geo_Type': geo_type_list,

        'Cover_Type': cover_list

    })

    train_df = pd.concat([normalized_df, class_df], axis = 1)

    xTr = np.array(train_df)

    np.save('./xTr.npy', xTr)

    np.save('./yTr.npy', yTr)

    print(xTr.shape)

    print(yTr.shape)

    return train_df_mean, train_df_std





def read_test_file(test_path, train_df_mean, train_df_std):

    test_df = pd.read_csv(test_path)

    rows, columns = test_df.shape

    # MAP WILDERNESS TO CLASSIFICATION TYPE

    wilderness_area_list = sorted(list(set(test_df['Wilderness_Area'])))

    wilderness_area_dict = {k: v for v, k in enumerate(wilderness_area_list)}

    test_df['Wilderness_Area'] = test_df['Wilderness_Area'].map(wilderness_area_dict)

    wilderness_list = test_df['Wilderness_Area']

    # CLASSIFY SOIL_TYPE

    soil_type_list = list(test_df['Soil_Type'])

    climate_type_list = []

    geo_type_list = []

    for soil_type in soil_type_list:

        climate_type = int(soil_type / 1000)

        climate_type_list.append(climate_type)

        geo_type = int((soil_type % 1000) / 100)

        geo_type_list.append(geo_type)

    # NORMALIZE TEST DATA SET

    test_df = test_df.iloc[:, 1 : -2]

    normalized_df = normalize(test_df, train_df_mean, train_df_std)

    # COMBINE TRAINING DATA SET INTO NUMPY ARRAY

    class_df = pd.DataFrame({

        'Wilderness_Area': wilderness_list,

        'Climater_Type': climate_type_list,

        'Geo_Type': geo_type_list

    })

    test_df = pd.concat([normalized_df, class_df], axis = 1)

#     print(test_df)

    xTe = np.array(test_df)

    np.save('./xTe.npy', xTe)

    print(xTe.shape)





def num_encode(column):

    # binary encode

    # enc = OneHotEncoder(sparse=False)

    enc = OneHotEncoder()

    column = column.reshape(-1, 1)

    enc.fit(column)

    encode_col = enc.transform(column).toarray()

    return encode_col





def normalize(df, df_mean, df_std):

    normalized_df = (df - df_mean) / df_std

    return normalized_df





if __name__ == "__main__":

    train_path = '../input/wustl-517a-sp20-milestone2/train.csv'

    train_df_mean, train_df_std = read_train_file(train_path)

    test_path = '../input/wustl-517a-sp20-milestone2/test.csv'

    read_test_file(test_path, train_df_mean, train_df_std)


from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error



xTr = np.load('xTr.npy')

yTr = np.load('yTr.npy')

xTe = np.load('xTe.npy')



forest_model = RandomForestClassifier(random_state=1)

forest_model.fit(xTr, yTr.ravel())



yTe_pred = forest_model.predict(xTe)

#yTe_pred_label = np.argmax(yTe_pred, axis = -1)



yTr_pred = forest_model.predict(xTr)

print(mean_absolute_error(yTr, yTr_pred))



test_df = pd.read_csv('../input/wustl-517a-sp20-milestone2/test.csv')

train_df = pd.read_csv('../input/wustl-517a-sp20-milestone2/train.csv')

cover_type_list = sorted(list(set(train_df['Cover_Type'])))

cover_type_dict = {k: v for k, v in enumerate(cover_type_list)}

yid = test_df['ID'].values



yTe_pred_label = yTe_pred

yTe_pred_text = np.array([cover_type_dict[k] for k in yTe_pred_label])

submit_df = pd.DataFrame({'ID': yid, 'Cover_Type': yTe_pred_text})

submit_df.to_csv('submission.csv', index = False, sep = ',')
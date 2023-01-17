from sklearn.feature_extraction.text import CountVectorizer

import os

import pandas as pd

import json

import numpy as np

from datetime import datetime

import xgboost as xgb
def read_data(category, train=True):

    # derive file paths

    path_dir_data = os.path.join('..', 'input')

    path_dir_map = os.path.join('..', 'input')



    file_data_suffix = '_data_info_train_competition.csv' if train else '_data_info_val_competition.csv'

    file_data = category + file_data_suffix

    file_map = category + '_profile_train.json'

    

    path_data = os.path.join(path_dir_data, file_data)

    path_map = os.path.join(path_dir_map, file_map)

    

    # read data file

    df_data = pd.read_csv(path_data)

    

    # read mapping file

    with open(path_map) as file_json: 

        dict_map = json.load(file_json)

    

    return df_data, dict_map



def tidy_train_data(df_train, dict_map):

    # flatten dictionary into dataframe

    df_map = pd.DataFrame()

    for attribute in list(dict_map.keys()):

        df_map_attribute = pd.DataFrame(list(dict_map[attribute].items()), columns = ['class', 'class_id'])

        df_map_attribute['attribute'] = attribute

        df_map = pd.concat([df_map, df_map_attribute])



    # prepare training dataframe

    df_train_label = (pd.melt(df_train, 

        id_vars=['itemid', 'title', 'image_path'], 

        value_vars=list(dict_map.keys()),

        var_name='attribute',

        value_name='class_id')

        .dropna() # do not train example if attribute is unlabelled

        .assign(class_id = lambda x: x['class_id'].apply(int)) # match df_map int type

        .merge(df_map, on=['attribute', 'class_id'], how='left'))

    

    return df_train_label



def predict_class(model, feature_test, k=2):

    # predict a probability vector

    # then return classes for highest k of each probability vector

    y_prob = model.predict_proba(feature_test)

    y_pred = [model.classes_[np.flip(np.argsort(y))[0:k]] for y in y_prob]

    return y_pred



def pick_top_2(y_pred):

    y_pred_1 = [y[0] for y in y_pred]

    y_pred_2 = [y[1] for y in y_pred]

    return y_pred_1, y_pred_2



def generate_model():

    # for this demo only implement one baseline xgb model

    # can be fine-tuned for specific attributes

    model_class = xgb.XGBClassifier

    model_params = {

        "max_depth": 15,

        "min_child_weight": 1,

        "learning_rate": 0.2,

        "n_estimators": 150,

        "reg_alpha": 0,

        "reg_lambda": 1,

        "objective": "multi:softprob",

        "n_jobs": -1

    }

    return model_class(**model_params)



def generate_text_features(doc_train, doc_test):

    # convert documents to simple raw count doc-term matrix

    cv = CountVectorizer(analyzer='word', token_pattern=r'\S{1,}')

    cv.fit(doc_train)

    feature_train = cv.transform(doc_train)

    feature_test = cv.transform(doc_test)

    return feature_train, feature_test



def run_submission():



    df_submission = pd.DataFrame()

    

    for category in ['fashion', 'beauty', 'mobile']:

        # read and prepare category training data,

        df_train, dict_map = read_data(category, train=True)

        df_test, dict_map = read_data(category, train=False)

        df_train_label = tidy_train_data(df_train, dict_map)

        attributes = list(dict_map.keys())

        

        for attribute in attributes:

            

            # extract train and test data

            id_train = df_train_label.loc[df_train_label['attribute'] == attribute, 'itemid'].tolist()

            doc_train = df_train_label.loc[df_train_label['attribute'] == attribute, 'title'].tolist()

            image_train = df_train_label.loc[df_train_label['attribute'] == attribute, 'image_path'].tolist()

            y_train = df_train_label.loc[df_train_label['attribute'] == attribute, 'class_id'].tolist()



            id_test = df_test['itemid'].tolist()

            doc_test = df_test['title'].tolist()

            image_test = df_test['image_path'].tolist()

            print("Running {c} - {a}".format(c=category, a=attribute))



            # generate text features

            feature_train, feature_test = generate_text_features(doc_train, doc_test)



            # generate results for attribute and append

            model = generate_model()

            model = model.fit(feature_train, y_train, verbose=True)

            y_pred = predict_class(model, feature_test, k=2)

            y_pred_1, y_pred_2 = pick_top_2(y_pred)



            # generate submission format

            item_attr = df_test.apply(lambda x:'{}_{}'.format(x['itemid'], attribute), axis=1)

            tagging = ['{} {}'.format(y1, y2) for y1, y2 in zip(y_pred_1, y_pred_2)]

            df_submission_attribute = pd.DataFrame({'id': item_attr, 'tagging': tagging})

            df_submission = df_submission.append(df_submission_attribute)



            print('{t}: Completed {c} - {a}'.format(

                t=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

                c=category, 

                a=attribute)

            )

            

    print('Completed generating submissions!')



    return df_submission
# warning: takes many hours to run!

df_submission = run_submission()

print(df_submission.shape)

print(df_submission.head())

df_submission.to_csv("xgb_baseline.csv", index=False)
import os,json

from time import time

import pandas as pd

import numpy as np

%matplotlib inline



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

data_path = '../input'

submission_path = '.'

def load_df(file):

    print('loading file {} >>>'.format(file))

    df = pd.read_csv(os.path.join(data_path,file))

    print('file dimension:', df.shape)

#     display(df.head())

    return df



def load_json(file):

    with open(os.path.join(data_path,file)) as json_file:  

        attr_map = json.load(json_file)

    return attr_map
# update name to map to submission format

def expand_attr(df, attrs):

    r = []

    for col in attrs:

        sub_df = df[df[col].notna()]

        tmp = sub_df['itemid'].astype(str).apply(lambda s: s+'_'+col)

        try:

            sub_df[col] = sub_df[col].astype(int).astype(str)

        except:

            sub_df[col] = sub_df[col].astype(str)

        tmp = pd.concat([tmp,sub_df[col]], axis=1)

        tmp.columns = ['id','tagging']

        r.append(tmp)

#         display(tmp.head(2))

    return pd.concat(r)





def create_submit(submit_df, pred_df):

    return pd.concat([submit_df,pred_df])
%%time

submit_df = pd.DataFrame([],columns=['id','tagging'])



for cat in ['beauty','mobile','fashion']:

    print('#'*30,'Category:',cat,'#'*30)

    train = load_df(cat+'_data_info_train_competition.csv')

    test = load_df(cat+'_data_info_val_competition.csv')    

    attr_info = load_json(cat+'_profile_train.json')

    

    for col in attr_info.keys():

        print('\t processing attribute:',col)

        pipeline = Pipeline([

            ('vect', CountVectorizer(min_df=1,ngram_range=(1,3))),

            ('clf', LogisticRegression()),

        ])

        

        parameters = {

            'vect__ngram_range': [(1,3),(1,5)],

        }



        # fit first model

        train_wo_na = train[~train.isna()[col]]

        pipeline.fit(train_wo_na['title'], train_wo_na[col])

        

        # grid search for best estimator

        t0 = time()

        grid_search = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1, verbose=2,scoring='accuracy')

        grid_search.fit(train_wo_na['title'], train_wo_na[col])

        

        print("done in %0.3fs" % (time() - t0))

        print(grid_search.cv_results_['mean_test_score'])



        print("Best score: %0.5f" % grid_search.best_score_)

        print("Best parameters set:")

        best_parameters = grid_search.best_estimator_.get_params()

        for param_name in sorted(parameters.keys()):

            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        

        # predict on test dataset, select top 2, instead of just one.

        estimator = grid_search.best_estimator_

        probs = estimator.predict_proba(test['title'])

        best_2 = pd.DataFrame(np.argsort(probs, axis=1)[:,-2:],columns=['top2','top1'])

        test[col] = best_2.apply(lambda row: ' '.join(estimator.classes_[[row['top1'],row['top2']]].astype(int).astype(str)) ,axis=1)

        

        # save models

#         joblib.dump(pipeline, os.path.join(checkpoints_path,'model_{}_{}.ckpt'.format(cat,col)))

        

    display(test.head(2))

    test.fillna(0,inplace=True)

    pred_df = expand_attr(test, attr_info); print(pred_df.shape)

    submit_df = create_submit(submit_df, pred_df); print(submit_df.shape)

    
submit_df.to_csv(os.path.join(submission_path,'baseline_top2.csv'),index=False)
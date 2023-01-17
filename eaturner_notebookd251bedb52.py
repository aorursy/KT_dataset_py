# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
key = pd.read_csv('../input/key_1.csv')

train = pd.read_csv('../input/train_1.csv')
def get_language(page):

    import re

    

    res = re.search('[a-z][a-z].wikipedia.org',page)

    if res:

        return res[0][0:2]

    return 'na'



train['lang'] = train.Page.map(get_language)
#splits training and testing sets into lists of sets per language

def lang_find( lang, df ):

    indices = np.where( df['lang'] == lang )[0]

    

    return df.loc[indices, :].drop('lang', axis = 1).copy()



def lang_df( df ):

    lang_list = df['lang'].unique()

        

    return {lang : lang_find(lang, df) for lang in lang_list}
lang_train = lang_df( train )
#strips dates in testing data

def strip_date( page ):

    date_str = page.split('_')[-1:][0]

    

    return date_str



key['date'] = key.Page.map(strip_date)

key['lang'] = key.Page.map(get_language)
lang_test = lang_df( key )
#feature engineering per language

def date_year( date ):

    return int(date.split('-')[0])



def date_month( date ):

    return int(date.split('-')[1])



def date_day( date ):

    return int(date.split('-')[2])



def lang_date_visit( df_lang ):

    visits = df_lang.drop('Page', axis = 1).median()

    

    df_visits = pd.DataFrame( columns = ['date', 'visit'] )

    

    df_visits['date'] = visits.index

    df_visits['visit'] = visits.values

    

    df_visits['year'] = df_visits.date.map(date_year)

    df_visits['month'] = df_visits.date.map(date_month)

    df_visits['day'] = df_visits.date.map(date_day)

     

    df_visits.drop('date', axis = 1, inplace = True)

    

    return df_visits



#evaluation metric

def smape(y_true, y_pred):

    denominator = (y_true + np.abs(y_pred)) / 200.0

    diff = np.abs(y_true - y_pred) / denominator

    diff[denominator == 0] = 0.0

    return np.mean(diff)



#returns the model for the particular language

def lang_fit( df_lang ):

    

    from sklearn.metrics import make_scorer

    from sklearn.model_selection import TimeSeriesSplit

    from sklearn.linear_model import RidgeCV

    

    loss = make_scorer(smape, greater_is_better=False)

    

    df_visits = lang_date_visit( df_lang )

    

    model = RidgeCV(scoring = loss, cv = TimeSeriesSplit(10) )

    

    y = df_visits.pop('visit')

    

    print ('Fitting Model')

    

    return model.fit(df_visits, y)



#fits all the models per language

def fit( df_dict ):

    

    return { key : lang_fit( df_dict.get(key) ) for key in df_dict.keys() }



#makes prediction per language

def lang_predict( model, df_lang ):

    

    sub = pd.DataFrame( columns = ['Id', 'Prediction'])

    

    sub['Id'] = df_lang.pop('Id')

    

    df_lang['year'] = df_lang.date.map(date_year)

    df_lang['month'] = df_lang.date.map(date_month)

    df_lang['day'] = df_lang.date.map(date_day)

    

    df_lang.drop(['Page', 'date'], axis = 1, inplace = True)

    

    sub['Prediction'] = model.predict( df_lang )



    print ('Prediction Finished')

    

    return sub



#prediction for all languages

def predict( train_dict, test_dict ):

    

    model_dict = fit( train_dict )

    

    return { key : lang_predict( model_dict.get(key),

                                test_dict.get(key) ) for key in test_dict.keys() }
for key in lang_train.keys():

    df_lang = lang_train.get(key)

    df_new = df_lang
df_vis = df_lang.dropna().T.drop('Page', axis = 0).m(axis = 1).reset_index()
df_vis.columns = ['date', 'visit']
df_vis.plot()
pred_dict = predict(lang_train, lang_test)
subs = pd.DataFrame( columns = ['Id', 'Prediction'] )



for key in pred_dict.keys():

    subs = subs.append( pred_dict.get(key) )
subs.to_csv('../output/sub.csv', index = False)
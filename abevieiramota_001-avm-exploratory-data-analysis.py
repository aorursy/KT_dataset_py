!head -2 ../input/gendered-pronoun-resolution/sample_submission_stage_1.csv
!head -2 ../input/gendered-pronoun-resolution/test_stage_1.tsv
import pandas as pd



pd.set_option('max_colwidth', 4000)
def read_df(path):

    

    df = pd.read_csv(path, index_col='ID', sep='\t', encoding='utf-8')



    # add some columns



    # the ending offset of the pronoun and the candidates referring entities

    df['Pronoun-offset-end'] = df['Pronoun-offset'] + df['Pronoun'].str.len()

    df['A-offset-end'] = df['A-offset'] + df['A'].str.len()

    df['B-offset-end'] = df['B-offset'] + df['B'].str.len()



    # text length



    df['Text-length'] = df['Text'].str.len()

    

    return df
df = read_df('../input/gendered-pronoun-resolution/test_stage_1.tsv')



df.sample()
len(df)
from spacy import displacy



def display_entry(entry):



    data = entry.to_dict()

    

    colors = {

        'Pronoun': '#aa9cfc',

        'A': '#fc9ce7' if not 'A-coref' in data or not data['A-coref'] else '#FFE14D',

        'B': '#fc9ce7' if not 'B-coref' in data or not data['B-coref'] else '#FFE14D'

    }



    options = {

        'colors': colors

    }

    

    render_data = {

        'text': data['Text'],

        'ents': sorted([

            {

                'start': data['Pronoun-offset'],

                'end': data['Pronoun-offset-end'],

                'label': 'Pronoun'

            },

            {

                'start': data['A-offset'],

                'end': data['A-offset-end'],

                'label': 'A'

            },

            {

                'start': data['B-offset'],

                'end': data['B-offset-end'],

                'label': 'B'

            }

        ], key=lambda x: x['start'])

    }

    

    displacy.render(render_data, style='ent', manual=True, jupyter=True, options=options)
sample = df.sample(random_state=100)



display_entry(sample.iloc[0])
result = pd.DataFrame({'ID': df.index, 'A': 1, 'B': 1, 'NEITHER': 1})



result.to_csv('dummy_all_equal.csv', index=False)
result['A'] = 1

result['B'] = 0

result['NEITHER'] = 0



result.to_csv('dummy_A.csv', index=False)
result['A'] = 0

result['B'] = 1

result['NEITHER'] = 0



result.to_csv('dummy_B.csv', index=False)
result['A'] = 0

result['B'] = 0

result['NEITHER'] = 1



result.to_csv('dummy_NEITHER.csv', index=False)
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html

from sklearn.metrics import log_loss



y_true = ["spam", "ham", "ham", "spam"]

# The labels in y_pred are assumed to be ordered alphabetically, as done by preprocessing.LabelBinarizer

# ["ham", "spam"]

y_pred = [

    [.1, .9], 

    [.9, .1], 

    [.8, .2], 

    [.35, .65]

]



log_loss(y_true, y_pred)
# all prediction is correct

y_pred = [

    [0., 1.],

    [1., 0.],

    [1., 0.],

    [0., 1.]

]



log_loss(y_true, y_pred)
# all prediction is wrong

y_pred = [

    [1., 0.],

    [0., 1.],

    [0., 1.],

    [1., 0.]

]



log_loss(y_true, y_pred)
samples = df.sample(n=10, random_state=100)



for _, s in samples.iterrows():

    

    display_entry(s)
import spacy



nlp = spacy.load('en_core_web_lg')



displacy.render(nlp(samples.iloc[-2]['Text']), style='dep', jupyter=True, options={'distance': 150})
!ls ../input/googlegapcoreference/
# following: https://www.kaggle.com/keyit92/end2end-coref-resolution-by-attention-rnn/data



train_df = read_df("../input/googlegapcoreference/gap-test.tsv")

test_df = read_df("../input/googlegapcoreference/gap-development.tsv")

dev_df = read_df("../input/googlegapcoreference/gap-validation.tsv")
print(f"Train: {train_df.shape}\nTest: {test_df.shape}\nDevelopment: {dev_df.shape}")
sample = train_df.sample(random_state=555)



display_entry(sample.iloc[0])
# just testing if there is any entry with more than one answer



train_df[train_df[['A-coref', 'B-coref']].sum(axis=1) > 1]
# adding a column with the answer

def get_answer(row):

    

    if row['A-coref']:

        return 'A'

    

    if row['B-coref']:

        return 'B'

    

    return 'NEITHER'

    

train_df['answer'] = train_df.apply(get_answer, axis=1)
train_df['answer'].value_counts()
train_df['Text-length'].hist()
train_df.groupby(pd.qcut(train_df['Text-length'], q=[0, .25, .5, .75, 1.]))['answer'].value_counts().unstack()
train_df_A = train_df[train_df['answer'] == 'A']

train_df_B = train_df[train_df['answer'] == 'B']

train_df_NEITHER = train_df[train_df['answer'] == 'NEITHER']



X_A_A = train_df_A.rename(columns={

    'A': 'RE',

    'A-offset': 'RE-offset',

    'A-offset-end': 'RE-offset-end'

})[['Text', 'Pronoun', 'RE', 'RE-offset', 'RE-offset-end', 'URL', 'Text-length']]



X_A_A['y'] = 1

X_A_A['referred-expression'] = 'A'



X_A_B = train_df_A.rename(columns={

    'B': 'RE',

    'B-offset': 'RE-offset',

    'B-offset-end': 'RE-offset-end'

})[['Text', 'Pronoun', 'RE', 'RE-offset', 'RE-offset-end', 'URL', 'Text-length', 'answer']]



X_A_B['y'] = 0

X_A_B['referred-expression'] = 'A'



X_B_B = train_df_B.rename(columns={

    'B': 'RE',

    'B-offset': 'RE-offset',

    'B-offset-end': 'RE-offset-end'

})[['Text', 'Pronoun', 'RE', 'RE-offset', 'RE-offset-end', 'URL', 'Text-length', 'answer']]



X_B_B['y'] = 1

X_B_B['referred-expression'] = 'B'



X_B_A = train_df_B.rename(columns={

    'A': 'RE',

    'A-offset': 'RE-offset',

    'A-offset-end': 'RE-offset-end'

})[['Text', 'Pronoun', 'RE', 'RE-offset', 'RE-offset-end', 'URL', 'Text-length', 'answer']]



X_B_A['y'] = 0

X_B_A['referred-expression'] = 'B'



X_NEITHER_A = train_df_NEITHER.rename(columns={

    'A': 'RE',

    'A-offset': 'RE-offset',

    'A-offset-end': 'RE-offset-end'

})[['Text', 'Pronoun', 'RE', 'RE-offset', 'RE-offset-end', 'URL', 'Text-length', 'answer']]



X_NEITHER_A['y'] = 0

X_NEITHER_A['referred-expression'] = 'A'



X_NEITHER_B = train_df_NEITHER.rename(columns={

    'B': 'RE',

    'B-offset': 'RE-offset',

    'B-offset-end': 'RE-offset-end'

})[['Text', 'Pronoun', 'RE', 'RE-offset', 'RE-offset-end', 'URL', 'Text-length', 'answer']]



X_NEITHER_B['y'] = 0

X_NEITHER_B['referred-expression'] = 'B'





X_df = pd.concat((X_A_A, X_A_B, X_B_A, X_B_B, X_NEITHER_A, X_NEITHER_B))
X_df.shape
X_df.sample(random_state=1)
import re



PARENTHESIS_RE = re.compile(r'(.*?)\((.*?)\)')

CAMELCASE_RE = re.compile(r'([a-z])([A-Z])')



def preprocess_so(so):



    parenthesis_preprocessed = PARENTHESIS_RE.sub('\g<2> \g<1>', so)

    underline_removed = parenthesis_preprocessed.replace('_', ' ')

    camelcase_preprocessed = CAMELCASE_RE.sub('\g<1> \g<2>', underline_removed)



    return camelcase_preprocessed.strip().replace('"', '')
from textacy import similarity



def add_features(df, re_col, inplace=False):

    

    if inplace:

        df_ = df

    else:

        df_ = df.copy()

    

    df_['URL_last_part'] = df_['URL'].str.rsplit('/', n=1, expand=True)[1].apply(preprocess_so)

    

    df_['URL_distance_jaro_winkler'] = df_.apply(lambda row: similarity.jaro_winkler(row['URL_last_part'], row[re_col]), axis=1)

    df_['URL_distance_levenshtein'] = df_.apply(lambda row: similarity.levenshtein(row['URL_last_part'], row[re_col]), axis=1)

    df_['URL_distance_token_sort_ratio'] = df_.apply(lambda row: similarity.token_sort_ratio(row['URL_last_part'], row[re_col]), axis=1)

    

    return df_

    



add_features(X_df, 'RE', inplace=True)



X_df.sample(5, random_state=800)[['URL_last_part', 'URL']]
X_df.hist(column='URL_distance_jaro_winkler', by='y', figsize=(20, 5), bins=20, sharey=True)
X_df.hist(column='URL_distance_levenshtein', by='y', figsize=(20, 5), bins=20, sharey=True)
X_df.hist(column='URL_distance_token_sort_ratio', by='y', figsize=(20, 5), bins=20, sharey=True)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



X = X_df[['URL_distance_token_sort_ratio', 'URL_distance_levenshtein', 'URL_distance_jaro_winkler', 'Text-length', 'referred-expression']]

y = X_df['y']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)



X_train_features = X_train.drop(columns='referred-expression')

X_train_referred_expression = X_train['referred-expression']



X_test_features = X_test.drop(columns='referred-expression')

X_test_referred_expression = X_test['referred-expression']



lr = LinearRegression(normalize=True).fit(X_train_features, y_train)



y_pred_ = lr.predict(X_test_features)
import numpy as np



def transform_to_submit(y_pred_, referred_expression):

    

    y_pred_comp = 1 - y_pred_

    all_zero = np.zeros_like(y_pred_).reshape((-1, 1))



    y_pred = np.hstack((

                np.where(referred_expression == 'A', y_pred_, y_pred_comp).reshape((-1, 1)),

                np.where(referred_expression == 'B', y_pred_comp, y_pred_).reshape((-1, 1)),

                all_zero

    ))

    

    return y_pred



y_true = np.hstack((

            np.where(((X_test_referred_expression == 'A') & (y_test)), 1, 0).reshape((-1, 1)),

            np.where(((X_test_referred_expression == 'B') & (y_test)), 1, 0).reshape((-1, 1)),

            np.zeros_like(y_test).reshape((-1, 1))

))





# TODO: refact

# one of the ideas is to run the model over all the referred expressions and then calculate the final answer

#y_pred_A = lr.predict(df_A).reshape(-1, 1)

#y_pred_B = lr.predict(df_B).reshape(-1, 1)

#all_zero = np.zeros_like(y_pred_A)



#X_test_A = add_features(X_test, 'A', inplace=False)[['URL_distance_token_sort_ratio', 'URL_distance_levenshtein', 'URL_distance_jaro_winkler', 'Text-length']]

#X_test_B = add_features(X_test, 'B', inplace=False)[['URL_distance_token_sort_ratio', 'URL_distance_levenshtein', 'URL_distance_jaro_winkler', 'Text-length']]



#y_pred = np.hstack((y_pred_A,

#                    y_pred_B,

#                    all_zero

#                   ))

#y_true[(np.abs(y_true[:, 0] - y_true[:, 1]) < 0.1) & (y_true[:, 0] < .3), 2] = .5

    

log_loss(y_true, transform_to_submit(y_pred_, X_test_referred_expression))
y_true
X_features = X.drop(columns='referred-expression')

X_referred_expression = X['referred-expression']



lr.fit(X_features, y)
df_A = add_features(df, 'A', inplace=False)[['URL_distance_token_sort_ratio', 'URL_distance_levenshtein', 'URL_distance_jaro_winkler', 'Text-length']]

df_B = add_features(df, 'B', inplace=False)[['URL_distance_token_sort_ratio', 'URL_distance_levenshtein', 'URL_distance_jaro_winkler', 'Text-length']]



y_pred_A = lr.predict(df_A).reshape(-1, 1)

y_pred_B = lr.predict(df_B).reshape(-1, 1)

all_zero = np.zeros_like(y_pred_A)



y_pred = np.hstack((y_pred_A,

                    y_pred_B,

                    all_zero

                   ))



result = pd.DataFrame(y_pred, index=df.index, columns=['A', 'B', 'NEITHER'])



result.loc[((result['A'] - result['B']).abs() < 0.1) & (result['A'] < .3), 'NEITHER'] = .3



result.to_csv('lr_over_URL_similarity.csv')
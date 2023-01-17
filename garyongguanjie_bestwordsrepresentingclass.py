import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

import copy
df_train = pd.read_csv('../input/student-shopee-code-league-sentiment-analysis/train.csv')

df_test = pd.read_csv('../input/student-shopee-code-league-sentiment-analysis/test.csv')
def compare(df,rating1,rating2):

    df = copy.deepcopy(df)

    df = df[(df.rating==rating1)|(df.rating==rating2)]

    df = df.drop_duplicates(subset=['review'])

    # replacing values so we can easily get unbiased weights

    df.loc[df['rating']==rating1,'rating'] = 0

    df.loc[df['rating']==rating2,'rating'] = 1

    df = df.sample(frac=1)

    text_clf = Pipeline([

                      ('vect',CountVectorizer(ngram_range=(1,3),min_df=5,binary=True)),

                      ('clf',LogisticRegression(max_iter=1000))]



    )

    text_clf.fit(df.review.values,df.rating.values)

    coef = text_clf.named_steps['clf'].coef_ #this is w

    vect = text_clf.named_steps['vect'].get_feature_names() # this is x

    x = np.argsort(coef[0])[-10:][::-1]#argsort descending order (taking top 10 results with largest weights)

    for i in x:

        print(vect[i])
compare(df_train,4,5)
compare(df_train,1,5)
compare(df_train,3,4)
compare(df_train,1,3)
compare(df_train,2,3)
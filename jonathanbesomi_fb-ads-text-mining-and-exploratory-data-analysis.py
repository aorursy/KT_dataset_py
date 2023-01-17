import pandas as pd

!pip install texthero -q

import texthero as hero
ads_df = pd.read_csv("/kaggle/input/political-advertisements-from-facebook/fbpac-ads-en-US.csv")

ads_df.head()
ads_df.shape
SAMPLE_SIZE = 10000

ads_df_ = ads_df.sample(SAMPLE_SIZE)
print(ads_df_.columns.values)
ads_df_.describe()
def get_text_columns(df):

    text_columns = []

    for col in df.select_dtypes('object'):

        if (df[col].str.split().str.len() > 5).any():

            text_columns.append(df[col].name)

    return text_columns



get_text_columns(ads_df_)
TOP_WORDS = 10



hero.top_words(ads_df_.title)[:TOP_WORDS]
hero.top_words(ads_df_.title.str.lower())[:TOP_WORDS]
(

    ads_df_.title.str.lower()

    .dropna()

    .pipe(hero.remove_stopwords)

    .pipe(hero.top_words)[:10]

)
(

    ads_df_.message.str.lower()

    .dropna()

    .pipe(hero.remove_stopwords)

    .pipe(hero.top_words)[:10]

)
def remove_html_tags(s: pd.Series) -> pd.Series():

    """Remove all html entities from a pandas series"""

    

    # TODO. Consider this more sophisticated solution: ('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

    

    return s.str.replace('<.*?>', '')



s = pd.Series("<p>Hello world!</p>")

remove_html_tags(s)
ads_df_['message'] = remove_html_tags(ads_df_['message'])



(

    ads_df_.message.str.lower()

    .dropna()

    .pipe(hero.remove_stopwords)

    .pipe(hero.top_words)[:10]

)
hero.wordcloud(ads_df_.title)
hero.wordcloud(ads_df_.message)
ads_df['advertiser'].value_counts()[:10]
trump_df = ads_df[ads_df['advertiser'] == 'Donald J. Trump'].copy()

trump_df.title.unique()
trump_df['message'] = remove_html_tags(trump_df['message'])



(

    trump_df.message.str.lower()

    .pipe(hero.remove_stopwords)

    .pipe(hero.top_words)[:10]

)
trump_df['noun_chunks'] = hero.nlp.noun_chunks(trump_df.message)

trump_df['noun_chunks'].head(2)
help(hero.nlp.noun_chunks)
trump_df['noun_chunks'].apply(lambda row: [r[0] for r in row]).explode().value_counts()[:20]
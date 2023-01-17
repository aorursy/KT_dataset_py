import numpy as np

import pandas as pd 

import scattertext as st
df = pd.read_csv('/kaggle/input/380000-lyrics-from-metrolyrics/lyrics.csv')
df = df[df.lyrics.str.len() > 0]
df_brit = df[(df.artist == 'beyonce') 

          | (df.artist == 'beyonce-knowles') 

          | df.artist.str.startswith('britney-spears')]
df_brit['parse'] = df_brit.lyrics.apply(st.whitespace_nlp_with_sentences)
df_brit['Artist'] = df_brit.artist.apply(lambda x: 'Beyonce' if x.startswith('beyo') else 'Britney')
corpus = st.CorpusFromParsedDocuments(

    df_brit, category_col='Artist', parsed_col='parse'

).build().get_unigram_corpus()
print(term_scorer.get_score_df().sort_values(by='cohens_d', ascending=True).head())
print('Total number of songs per artist')

print(pd.Series(corpus.get_category_names_by_row()).value_counts())

del out_df

out_df = corpus.get_term_freq_df('')

pd.merge(pd.merge(out_df.rename(columns={'Beyonce':'Beyonce Catalog', 'Britney':'Britney Catalog'}), 

                  (st.OncePerDocFrequencyRanker(corpus)

                  .get_ranks('')

                  .rename(columns={'Beyonce': "Beyonce # Songs", 'Beyonce': "Beyonce # Songs"})), 

                  left_index=True, right_index=True),

         (out_df/pd.Series(corpus.get_category_names_by_row()).value_counts())

         .rename(columns={'Beyonce':'Beyonce Per Song',

                          'Britney':'Britney Per Song'}),

         left_index=True, 

         right_index=True)
html = st.produce_scattertext_explorer(

    corpus,

    category='Britney', not_category_name='Beyonce',

    minimum_term_frequency=0, pmi_threshold_coefficient=0,

    metadata = corpus.get_df()['song'],

    transform=st.Scalers.dense_rank

)

open('./demo_compact.html', 'w').write(html)

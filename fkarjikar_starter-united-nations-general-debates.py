%matplotlib inline
import string
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import psycopg2
import textatistic
import seaborn as sbn
from altair import Chart, X, Y, Color, Scale
import altair as alt
from vega_datasets import data
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords 
matplotlib.style.use('ggplot')
undf = pd.read_csv('Data/un-general-debates.csv')
len(undf)
undf.sort_values('year', ascending=False).head()
pd.set_option('display.max_colwidth', -1)
print(undf[(undf.year == 1970) & (undf.country == 'USA')].text)
pd.set_option('display.max_colwidth', 50)
by_year = undf.groupby('year', as_index=False)['text'].count()
by_year.head()
alt.Chart(by_year).mark_bar().encode(x='year:N',y='text')
by_country = undf.groupby('country',as_index=False)['text'].count()
by_country.head()
alt.Chart(by_country,title='speech distribution').mark_bar().encode(x=alt.X('text',bin=True),y='count()')

by_country.loc[by_country.text.idxmax()]
by_country.loc[by_country.text.idxmin()]
#c_codes = pd.read_csv('Data/country_codes.csv')
#c_codes.head()
c_codes = pd.read_csv('Data/country_codes.csv', encoding='iso-8859-1')
c_codes.head()
undf.columns = ['session', 'year', 'code_3', 'text']
undf.head()

undfe = undf.merge(c_codes[['code_3', 'country', 'continent', 'sub_region']])
undfe.head()
undfe[undf.code_3 == 'EU ']
undfe = undf.merge(c_codes[['code_3', 'country', 'continent', 'sub_region']], how='outer')
undfe.head()
undfe[undfe.country.isna()].code_3.unique()
undfe[undfe.text.isna()].code_3.unique()
undfe[undfe.text.isna()].country.unique()
undfe.loc[undfe.code_3 == 'EU', 'country'] = 'European Union'

by_country = undfe.groupby('country',as_index=False)['text'].count()
by_country.loc[by_country.text.idxmin()]


c_codes[c_codes.code_2 == 'EU']
len(undfe)
len(undf.code_3.unique())
len(undfe.code_3.unique())
set(undf.code_3.unique()) - set(undfe.code_3.unique())
speeches_1970 = undf[undf.year == 1970].copy()

speeches_1970['text'] = speeches_1970.text.apply(lambda x: x.lower())
speeches_1970['text'] = speeches_1970.text.apply(lambda x: x.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))))

speeches_1970['word_list'] = speeches_1970.text.apply(nltk.word_tokenize)
from collections import Counter
c = Counter(speeches_1970.word_list.sum())
c.most_common(10)
c.most_common()[-10:]
sw = set(stopwords.words('english'))
len(sw)
speeches_1970['word_list'] = speeches_1970.word_list.apply(lambda x: [y for y in x if y not in sw])
c = Counter(speeches_1970.word_list.sum())
c.most_common(25)
c.most_common()[-25:]



topics = [' nuclear', ' weapons', ' nuclear weapons', ' chemical weapons', 
          ' biological weapons', ' mass destruction', ' peace', ' war',
          ' nuclear war', ' civil war', ' terror', ' genocide', ' holocaust',
          ' water', ' famine', ' disease', ' hiv', ' aids', ' malaria', ' cancer',
          ' poverty', ' human rights', ' abortion', ' refugee', ' immigration',
          ' equality', ' democracy', ' freedom', ' sovereignty', ' dictator',
          ' totalitarian', ' vote', ' energy', ' oil',  ' coal',  ' income',
          ' economy', ' growth', ' inflation', ' interest rate', ' security',
          ' cyber', ' trade', ' inequality', ' pollution', ' global warming',
          ' hunger', ' education', ' health', ' sanitation', ' infrastructure',
          ' virus', ' regulation', ' food', ' nutrition', ' transportation',
          ' violence', ' agriculture', ' diplomatic', ' drugs', ' obesity',
          ' islam', ' housing', ' sustainable', 'nuclear energy']
undf.head()
year_summ = undf.groupby('year', as_index=False)['text'].sum()
year_summ.head()
year_summ['gw'] = year_summ.text.str.count('global warming')
year_summ['cc'] = year_summ.text.str.count('climate change')
year_summ
alt.Chart(year_summ[['year', 'gw', 'cc']]).mark_line().encode(x='year',y='gw')
alt.Chart(year_summ[['year', 'gw', 'cc']].melt(id_vars='year', value_vars=['cc','gw'])
         ).mark_line().encode(x='year:O',y='value', color='variable')
year_summ['pollution'] = year_summ.text.str.count('pollution')
year_summ['terror'] = year_summ.text.str.count('terror')
alt.Chart(year_summ[['year','terror']]).mark_line().encode(x='year:O', y='terror')
import numpy as np
nrows, ncols = 100000, 100
rng = np.random.RandomState(43)
df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows,ncols)) for i in range(4))
%timeit df1 + df2 + df3 + df4
%timeit pd.eval('df1 + df2 + df3 + df4')
undf['text_len'] = undf.text.map(lambda x : len(x.split()))
undf.head()
undf.groupby('code_3', as_index=False)['text_len'].mean().head()
alt.Chart(undf.groupby('code_3', as_index=False)['text_len'].mean()).mark_bar().encode(
alt.X('text_len', bin=True), y='count()')
undf.groupby('code_3', as_index=False)['text_len'].mean().sort_values('text_len').head()
undf.groupby('code_3', as_index=False)['text_len'].mean().sort_values('text_len').tail()
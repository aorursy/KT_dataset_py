# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

# Introductory data

df=pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

df.head()

import pandasql as ps

country = "Spain" # we can change the country 



SQL=f"""

    SELECT year , 

            sum("suicides/100k pop")/12 as suicides_{country} 

    FROM df 

    WHERE country = '{country}' group by year order by year"""  

df_es =  ps.sqldf( SQL, locals())

df_es.set_index('year', inplace=True)







df_es[f'suicides_{country}'].plot( figsize=(15,4), title=f"Suicides {country}, rate /100k pop")
df_es.head()

kk = pd.DataFrame()

groups = df.age.unique()



for sex in ['male','female'] :

    for age_group in groups :

        kk[ age_group.replace('5-14','05-14' )+'_'+sex] = df.loc[ (df['country']==country) & (df['age'] == age_group) & ( df['sex'] == sex) ]['suicides/100k pop'].values



kk.index = df.loc[ (df['country']==country) & (df['age'] == age_group) & ( df['sex'] == sex) ]['year'].values



df_es_gender_age = kk



df_es_gender_age.head()
df_female= pd.DataFrame()

df_male= pd.DataFrame()



for c in df_es_gender_age.columns.values :

    if "female" in c : 

        df_female[c] = df_es_gender_age[c]

    else : 

        df_male[c] = df_es_gender_age[c]

        



df_male.plot(   figsize=(15,4), title=f"Male suicide rates in {country}, per age")



df_female.plot( figsize=(15,4), title=f"Female suicide rates in {country}, per age")
import matplotlib.pyplot as plt

compare=[]

plt.style.use('ggplot')



for age in groups :

    compare_male_female = pd.DataFrame()

    compare_male_female[age+'_male']   = df_male  [age.replace('5-14','05-14' )+'_male']

    compare_male_female[age+'_female'] = df_female[age.replace('5-14','05-14' )+'_female']

    compare_male_female.plot(   figsize=(15,4), title=f"Male/Female suicide rates in {country}, per age, {age}")
import matplotlib.pyplot as plt



    

for age in groups :

    compare_male_female = pd.DataFrame()

    compare=[]

    compare_male_female[age+'_male']   = df_male  [age.replace('5-14','05-14' )+'_male']

    compare_male_female[age+'_female'] = df_female[age.replace('5-14','05-14' )+'_female']

    compare.insert(0,compare_male_female[age+'_male'  ].mean())

    compare.insert(1,compare_male_female[age+'_female'].mean())

    male_percentage = str(compare[0]/(compare[0]+compare[1])*100)[0:5]

    plt.title(f"Suicide rated in {country}, mean values, {age}, {male_percentage}% male")

    plt.bar(["male","female"],compare , color='green',)

    plt.show()
headlines = pd.read_csv('../input/million-headlines/abcnews-date-text.csv')

headlines.head() 







SQL=f"SELECT publish_date  , headline_text FROM headlines WHERE headline_text LIKE '%{country}%'"  

hds =  ps.sqldf( SQL, locals())



country_lwr=country.lower()

hds['year'] = hds['publish_date'].astype(str).str[:4]

hds['headline_text_new'] = hds['headline_text'].astype(str).str.replace(f"in {country}","").replace(f"{country_lwr}","").replace(";","").replace(",","")

hds['headline_text'] = hds['headline_text_new']



hds.drop("headline_text_new", axis=1, inplace=True)

hds.drop("publish_date", axis=1, inplace=True)



hds.head()



from stop_words import get_stop_words

stop  =  get_stop_words('english') + [country.lower(),country.lower()+'s',",",";"]

# That would be the line to remove stop word, but we use the vectorizer to remote it, so we do not need next line

# hds['headline_text'] = hds['headline_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))



print(stop)
from sklearn.feature_extraction.text import TfidfVectorizer



import numpy as np



MAX_FEATURES=10

def get_topics_year( year ):

    """Get the most frecuent topic for a given year.

    Args:

        yeat: Year of the news to be considered.

    Returns:

        Array with the news



    """



    corpus = hds.loc[ hds['year'] == year ]

    corpus.drop("year", axis=1, inplace=True)



    tf = TfidfVectorizer(analyzer='word',

                         ngram_range=(2,2),

                         max_features=MAX_FEATURES,

                         min_df = 0, 

                         stop_words = stop, 

                         sublinear_tf=True)

    X = tf.fit_transform(corpus['headline_text'].values)

    feature_names = tf.get_feature_names()

    tf.get_feature_names()

    return tf.get_feature_names()



topics = pd.DataFrame()

rows = []

for i in range(2003,2016):

    topic_this_year = {'year' : str(i) , 'topics' : get_topics_year(str(i)   ) }

    rows.append(topic_this_year)

    

topics=pd.DataFrame.from_dict(rows, orient='columns')

topics.set_index('year', inplace=True)



topics
for i in range(2003,2016):

    this_year_suicide = df_es.loc[i,f'suicides_{country}'] -  df_es.loc[i-1,f'suicides_{country}']

    topics.loc[str(i),'this_year_suicide'] = this_year_suicide
print("year","this_year_suicide variation", "topics")

my_rows=[]

for i, row in topics.iterrows():

    print( i,row['this_year_suicide'],row['topics']) 

    

    my_rows.append ({ "year" : i ,

                     "this_year_suicide" : row['this_year_suicide'] ,

                     "topics" : row['topics'] })



summary_country =pd.DataFrame.from_dict(my_rows, orient='columns')

summary_country['headlines']= summary_country['topics'].apply(', '.join)



summary_country[['year', 'this_year_suicide','headlines']]
from IPython.core.display import HTML 



s = f"""<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plot.ly/~jaimevalero78/48.embed"></iframe>"""

display(HTML(s))

def accumulate_frecuency_terms(df,d):

    for i, row in df.iterrows():

        for term in row['topics'] :

            if term not in d : 

                d[term] =           row['this_year_suicide']

            else :             

                d[term] = d[term] + row['this_year_suicide']

            

from collections import defaultdict

d = defaultdict(float)



#d={}

accumulate_frecuency_terms(topics,d)

#print(d)

text=""





from IPython.display import Image

from IPython.core.display import HTML 





array=[]

results_contry = pd.DataFrame()

for w in sorted(d, key=d.get, reverse=True):

    array.append( { "term" : w , "suicide_impact" :  d[w]})

    



results_country=pd.DataFrame.from_dict(array, orient='columns')



s = f"""<h3>Top headlines in years with increases in suicide rates, for {country}: </h3>"""

s = s + f"""Aka: Bad news for {country}  :-("""

display(HTML(s))

results_country.head(20)





s = f"""<h3>Top headlines in years with decreases in suicide rates, for {country}: </h3>"""

s = s + f"""Aka: Good news for  {country} :-) """

display(HTML(s))

results_country.sort_values(by=['suicide_impact'],ascending = True ).head(15)
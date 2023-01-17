import numpy as np 

import pandas as pd 

import re 

import string

from wordcloud import WordCloud

from textblob import TextBlob

from sklearn.feature_extraction import text

from collections import Counter

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import os

warnings.filterwarnings("ignore")



print(os.listdir("../input"))

df = pd.read_csv(r'../input/employee_reviews.csv')
df.head( n = 3)
df.shape
df.dtypes
for var in df.columns.values:

    if df[var].isnull().sum() > 0:

        print('{}  {}'.format(var , df[var].isnull().sum()))
convert_dtype_list = ['work-balance-stars' , 'culture-values-stars' , 'carrer-opportunities-stars',

                     'comp-benefit-stars','senior-mangemnet-stars', 'helpful-count']



for var in convert_dtype_list: 

    df[var] = df[var].replace('none' , 0)

    df[var] =  df[var].astype(np.float32)
df.dtypes
df.describe()
plt.style.use('fivethirtyeight')

plt.figure(1 , figsize = (15 , 7))

sns.countplot(y = 'company' , data = df , palette = 'rocket' , 

              order = df['company'].value_counts().index)

plt.show()
common_job_titles = []

c = Counter(df['job-title']).most_common()[:11]

for n in range(11):

    common_job_titles.append(c[n][0])

    

plt.figure(1 , figsize = (15 , 8))

sns.countplot(y = 'job-title' , data = df[df['job-title'].isin(common_job_titles)] ,

              palette = 'rocket' , 

              order = df[df['job-title'].isin(common_job_titles)]['job-title'].value_counts().index)

plt.show()
common_location = []

c = Counter(df['location']).most_common()[:11]

for n in range(11):

    common_location.append(c[n][0])

    

plt.figure(1 , figsize = (15 , 8))

sns.countplot(y = 'location' , data = df[df['location'].isin(common_location)] ,

              palette = 'rocket' , 

              order = df[df['location'].isin(common_location)]['location'].value_counts().index)

plt.title('')

plt.show()
def year(x):

    if x == 'None':

        year = 0 

    else:

        year = int(x.split(',')[1])

    return year 

getYear =  lambda x : year(x)

df['year'] = df['dates'].apply(getYear)

c = Counter(df['year'].sort_values())



plt.figure(1 , figsize = (15 , 7))

plt.scatter(x = np.arange(len(c.values())) , y =  c.values() , s = 200 )

plt.plot(np.arange(len(c.values())) , c.values() , alpha = 0.7)

plt.xticks(np.arange(len(c.values())) , c.keys())

plt.ylabel('counts of reviews')

plt.show()
plt.figure(1 , figsize = (15 , 6))

plt.subplot(1 , 2  , 1)

sns.distplot(df['overall-ratings'])



plt.subplot(1 , 2 , 2)

sns.violinplot(x = 'overall-ratings' , data = df)



plt.show()
plt.figure(1 , figsize = (15 , 9))

n = 0 

for company in df['company'].unique():

    n += 1

    plt.subplot(3 , 2 , n )

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    sns.violinplot(x = 'overall-ratings' , data = df.where(df['company'] == company))

    plt.xlabel('')

    plt.ylabel(company)

plt.show()
features = ['work-balance-stars' , 'culture-values-stars' , 'carrer-opportunities-stars',

                     'comp-benefit-stars','senior-mangemnet-stars']

index_companies = ['google' , 'amazon' , 'facebook' , 'netflix' , 'apple' , 'microsoft']



mean_rating_df = pd.DataFrame(index = index_companies)



for feature in features:

    f = []

    for comp in index_companies:

        f.append(np.mean(df[feature][df['company'] == comp]))

    mean_rating_df[feature] = f

del f



plt.figure(1 , figsize = (15 , 6))

colors = ['#3fa35d' , '#776e6e' , '#3490c1' , 

         '#e80909' , '#ad2694'  ,  '#ede62f']

for n , c  in zip(range(6) , colors):

    plt.scatter(x = np.arange(5) , y = mean_rating_df.iloc[n , :].values , s = 200 , c = c  , label = mean_rating_df.index[n])

    plt.plot(np.arange(5) , mean_rating_df.iloc[n , :].values , '-' , color = c , 

             alpha = 0.2)

    plt.xticks(np.arange(5) , features)



plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    

plt.show()


print('Job title : {}\nLocation : {}\ndate posted : {}\nCompany : {}'.format(

    df['job-title'][df['helpful-count'] ==  max(df['helpful-count'])].values[0],

    df['location'][df['helpful-count'] ==  max(df['helpful-count'])].values[0],

    df['dates'][df['helpful-count'] ==  max(df['helpful-count'])].values[0],

    df['company'][df['helpful-count'] ==  max(df['helpful-count'])].values[0]))



print('\nReview :\n{}\n\nPros:\n{}\n\nCons:\n{}\n\nAdvice to management:\n{}\n\nHelful Count :{}'.format(

    df['summary'][df['helpful-count'] ==  max(df['helpful-count'])].values[0],

    df['pros'][df['helpful-count'] ==  max(df['helpful-count'])].values[0],

    df['cons'][df['helpful-count'] ==  max(df['helpful-count'])].values[0],

    df['advice-to-mgmt'][df['helpful-count'] ==  max(df['helpful-count'])].values[0],

    df['helpful-count'][df['helpful-count'] ==  max(df['helpful-count'])].values[0]))



print('\n\nRatings:')

print('Overall : {}'.format(df['overall-ratings'][df['helpful-count'] ==  max(df['helpful-count'])].values[0]))

print('Work balance : {}'.format(df['work-balance-stars'][df['helpful-count'] ==  max(df['helpful-count'])].values[0]))

print('Culture and Values : {}'.format(df['culture-values-stars'][df['helpful-count'] ==  max(df['helpful-count'])].values[0]))

print('Career Opportunities : {}'.format(df['carrer-opportunities-stars'][df['helpful-count'] ==  max(df['helpful-count'])].values[0]))

print('Comp and Benefits : {}'.format(df['comp-benefit-stars'][df['helpful-count'] ==  max(df['helpful-count'])].values[0]))

print('Senior Management : {}'.format(df['senior-mangemnet-stars'][df['helpful-count'] ==  max(df['helpful-count'])].values[0]))
helpful_sum = []

for comp in index_companies:

    helpful_sum.append(np.sum(df['helpful-count'][df['company'] == comp]))

    

plt.figure(1 , figsize = (15 , 6))

sns.barplot(x = np.arange(6) , y = helpful_sum )

plt.xticks(np.arange(6) , index_companies)

plt.show()
def clean_txt(text):

    text = str(text)

    for n in range(10):

        text = text.replace(str(n) , '')

    text = text.lower()

    text = text.replace("(" , "")

    text = text.replace(")" , "")

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\w*\d\w*', '', text)

    text = re.sub('[‘’“”…]', '', text)

    text = re.sub(r'[^\x00-\x7f]', '', text)

    text = text.replace( " \ " , "" )

    text = text.replace("/" , "")

    

    return text



cleaning = lambda x : clean_txt(x)



df['summary'] =  df['summary'].apply(cleaning)

df['pros'] = df['pros'].apply(cleaning)

df['cons'] = df['cons'].apply(cleaning)

df['advice-to-mgmt'] = df['advice-to-mgmt'].apply(cleaning)

combined_smry_dict = {}

for comp in df['company'].unique():

    combined_summary = "" 

    for summary in df['summary'][df['company'] == comp]:

        combined_summary = combined_summary +" "+summary

    

    combined_smry_dict[comp] = combined_summary



df_summary = pd.DataFrame(data=combined_smry_dict , index = [0])

df_summary = pd.DataFrame.copy(df_summary.T)

df_summary.columns = ["summary_corpus"]

df_summary['company'] = df_summary.index



combined_pros_dict = {}

for comp in df['company'].unique():

    combined = "" 

    for pros in df['pros'][df['company'] == comp]:

        combined = combined +" "+pros

    

    combined_pros_dict[comp] = combined



df_pros = pd.DataFrame(data=combined_pros_dict , index = [0])

df_pros = pd.DataFrame.copy(df_pros.T)

df_pros.columns = ["pros_corpus"]

df_pros['company'] = df_pros.index



combined_cons_dict = {}

for comp in df['company'].unique():

    combined = "" 

    for cons in df['cons'][df['company'] == comp]:

        combined = combined +" "+cons

    

    combined_cons_dict[comp] = combined



df_cons = pd.DataFrame(data=combined_cons_dict , index = [0])

df_cons = pd.DataFrame.copy(df_cons.T)

df_cons.columns = ["cons_corpus"]

df_cons['company'] = df_cons.index



combined_advice_dict = {}

for comp in df['company'].unique():

    combined = "" 

    for adv in df['advice-to-mgmt'][df['company'] == comp]:

        combined = combined +" "+adv

    

    combined_advice_dict[comp] = combined



df_adv = pd.DataFrame(data=combined_advice_dict , index = [0])

df_adv = pd.DataFrame.copy(df_adv.T)

df_adv.columns = ["adive_corpus"]

df_adv['company'] = df_adv.index



pol = lambda x: TextBlob(x).sentiment.polarity

sub = lambda x: TextBlob(x).sentiment.subjectivity



df['summary_polarity'] = df['summary'].apply(pol)

df['summary_subjectivity'] = df['summary'].apply(sub)



df['pros_polarity'] = df['pros'].apply(pol)

df['pros_subjectivity'] = df['pros'].apply(sub)



df['cons_polarity']  = df['cons'].apply(pol)

df['cons_subjectivity'] = df['cons'].apply(sub)



df['advice_polarity'] = df['advice-to-mgmt'].apply(pol)

df['advice_subjectivity'] = df['advice-to-mgmt'].apply(sub)
plt.figure(1 , figsize = (15 , 4))

plt.hist(df['summary_polarity'] , bins = 50)

plt.title('Polarity in Summary')



plt.figure(2 , figsize = (15 , 7))

n = 0 

for comp , c in zip(index_companies , colors):

    n += 1

    plt.subplot(2 , 3 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    plt.hist(df['summary_polarity'][df['company'] == comp] , bins = 50 , color = c)

    plt.title(comp)

plt.show()
plt.figure(1 , figsize = (15 , 4))

plt.hist(df['summary_subjectivity'] , bins = 50)

plt.title('Subjectivity')



plt.figure(2 , figsize = (15 , 7))

n = 0 

for comp , c in zip(index_companies , colors):

    n += 1

    plt.subplot(2 , 3 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    plt.hist(df['summary_subjectivity'][df['company'] == comp] , bins = 50 , color = c )

    plt.title(comp)

plt.show()
wc = WordCloud(stopwords = text.ENGLISH_STOP_WORDS, 

               background_color = "white" , 

               colormap = "Dark2" ,

               max_font_size = 150 , 

               random_state = 42)

corpus = ''

for corp in df_summary['summary_corpus'].values:

    corpus = corpus+' '+corp

wc.generate(corpus)

plt.figure(1 , figsize = (15 , 8))

plt.imshow(wc , interpolation="bilinear")

plt.axis("off")

plt.show()
wc = WordCloud(stopwords = text.ENGLISH_STOP_WORDS.union(['apple' , 'netflix' , 'google']), 

               background_color = "white" , 

               colormap = "Dark2" ,

               max_font_size = 150 , 

               random_state = 42)



plt.figure(1 , figsize = (15 , 9))

for corpus , i in zip(df_summary['summary_corpus'].values,range(6)):

    wc.generate(corpus)

    plt.subplot(3 , 2 , i + 1)

    plt.imshow(wc , interpolation="bilinear")

    plt.axis("off")

    plt.title(df_summary.index[i])

plt.show()
plt.figure(1 , figsize = (15 , 4))

plt.hist(df['pros_subjectivity'] , bins = 50)

plt.title('Subjectivity in Pros')



plt.figure(2 , figsize = (15 , 7))

n = 0 

for comp , c in zip(index_companies , colors):

    n += 1

    plt.subplot(2 , 3 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    plt.hist(df['pros_subjectivity'][df['company'] == comp] , bins = 50 , color = c)

    plt.title(comp)

plt.show()
wc = WordCloud(stopwords = text.ENGLISH_STOP_WORDS, 

               background_color = "white" , 

               colormap = "Dark2" ,

               max_font_size = 150 , 

               random_state = 42)

corpus = ''

for corp in df_pros['pros_corpus'].values:

    corpus = corpus+' '+corp

wc.generate(corpus)

plt.figure(1 , figsize = (15 , 8))

plt.imshow(wc , interpolation="bilinear")

plt.axis("off")

plt.show()
wc = WordCloud(stopwords = text.ENGLISH_STOP_WORDS.union(['apple' , 'google' , 'facebook' , 'amazon' , 'netflix' , 'microsoft']), 

               background_color = "white" , 

               colormap = "Dark2" ,

               max_font_size = 150 , 

               random_state = 42)



plt.figure(1 , figsize = (15 , 7))

for corpus , i in zip(df_pros['pros_corpus'].values,range(6)):

    wc.generate(corpus)

    plt.subplot(3 , 2 , i + 1)

    plt.imshow(wc , interpolation="bilinear")

    plt.axis("off")

    plt.title(df_summary.index[i])

plt.show()
plt.figure(1 , figsize = (15 , 4))

plt.hist(df['cons_subjectivity'] , bins = 50)

plt.title('Subjectivity in Cons')



plt.figure(2 , figsize = (15 , 7))

n = 0 

for comp , c in zip(index_companies , colors):

    n += 1

    plt.subplot(2 , 3 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    plt.hist(df['cons_subjectivity'][df['company'] == comp] , bins = 50 , color = c)

    plt.title(comp)

plt.show()
wc = WordCloud(stopwords = text.ENGLISH_STOP_WORDS, 

               background_color = "white" , 

               colormap = "Dark2" ,

               max_font_size = 150 , 

               random_state = 42)

corpus = ''

for corp in df_cons['cons_corpus'].values:

    corpus = corpus+' '+corp

wc.generate(corpus)

plt.figure(1 , figsize = (15 , 8))

plt.imshow(wc , interpolation="bilinear")

plt.axis("off")

plt.show()
wc = WordCloud(stopwords = text.ENGLISH_STOP_WORDS.union(['apple' , 'google' , 'facebook' , 'amazon' , 'netflix' , 'microsoft']), 

               background_color = "white" , 

               colormap = "Dark2" ,

               max_font_size = 150 , 

               random_state = 42)



plt.figure(1 , figsize = (15 , 7))

for corpus , i in zip(df_cons['cons_corpus'].values,range(6)):

    wc.generate(corpus)

    plt.subplot(3 , 2 , i + 1)

    plt.imshow(wc , interpolation="bilinear")

    plt.axis("off")

    plt.title(df_summary.index[i])

plt.show()
plt.figure(1 , figsize = (15 , 4))

plt.hist(df['advice_polarity'] )

plt.title('Polarity in Advice to management')



plt.figure(2 , figsize = (15 , 7))

n = 0 

for comp , c in zip(index_companies , colors):

    n += 1

    plt.subplot(2 , 3 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    plt.hist(df['advice_polarity'][df['company'] == comp]  ,color = c)

    plt.title(comp)

plt.show()
plt.figure(1 , figsize = (15 , 4))

plt.hist(df['advice_subjectivity'] )

plt.title('Subjectivity in Advice to the management')



plt.figure(2 , figsize = (15 , 7))

n = 0 

for comp , c in zip(index_companies , colors):

    n += 1

    plt.subplot(2 , 3 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    plt.hist(df['advice_subjectivity'][df['company'] == comp] , color = c )

    plt.title(comp)

plt.show()
wc = WordCloud(stopwords = text.ENGLISH_STOP_WORDS, 

               background_color = "white" , 

               colormap = "Dark2" ,

               max_font_size = 150 , 

               random_state = 42)

corpus = ''

for corp in df_adv['adive_corpus'].values:

    corpus = corpus+' '+corp

wc.generate(corpus)

plt.figure(1 , figsize = (15 , 8))

plt.imshow(wc , interpolation="bilinear")

plt.axis("off")

plt.show()
wc = WordCloud(stopwords = text.ENGLISH_STOP_WORDS.union(['apple' , 'google' , 'facebook' , 'amazon' , 'netflix' , 'microsoft']), 

               background_color = "white" , 

               colormap = "Dark2" ,

               max_font_size = 150 , 

               random_state = 42)



plt.figure(1 , figsize = (15 , 7))

for corpus , i in zip(df_adv['adive_corpus'].values,range(6)):

    wc.generate(corpus)

    plt.subplot(3 , 2 , i + 1)

    plt.imshow(wc , interpolation="bilinear")

    plt.axis("off")

    plt.title(df_summary.index[i])

plt.show()
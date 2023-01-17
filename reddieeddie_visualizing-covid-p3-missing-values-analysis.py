from IPython.display import display

import matplotlib.pyplot as plt

import pandas as pd

# !pip install missingno

import missingno as mn

path = '../input/novel-corona-virus-2019-dataset/' 

covid_data = pd.DataFrame(pd.read_csv(path+'covid_19_data.csv'))

indiv_list = pd.DataFrame(pd.read_csv(path+'COVID19_line_list_data.csv'))

open_list = pd.DataFrame(pd.read_csv(path+'COVID19_open_line_list.csv'))

confirmed_US = pd.DataFrame(pd.read_csv(path+'time_series_covid_19_confirmed_US.csv'))

confirmed = pd.DataFrame(pd.read_csv(path+'time_series_covid_19_confirmed.csv'))

deaths_US = pd.DataFrame(pd.read_csv(path+'time_series_covid_19_deaths_US.csv'))

deaths = pd.DataFrame(pd.read_csv(path+'time_series_covid_19_deaths.csv'))

recovered = pd.DataFrame(pd.read_csv(path+'time_series_covid_19_recovered.csv'))

# Check tail for most recent date

display(covid_data.tail())

# Define dates for time series

dates = confirmed.columns[4:]

# dates[-1]
import matplotlib.pyplot as plt 

import missingno as mn

mn.bar(covid_data)
dataframes= [confirmed, deaths ,recovered ,  covid_data,indiv_list, open_list]

# # Lazy way

names = ['confirmed', 'deaths' ,'recovered',  'covid_data', 'indiv_list', 'open_list']

fig, axes = plt.subplots(figsize=(10,10))

for num in range(len(dataframes)):#df in dataframes:

    plt.subplot(3,2,num+1).title.set_text(names[num])

    #plt.title(dataframes[num])

    mn.bar(dataframes[num],color='DarkBlue')

fig.tight_layout()

 
# Get rid of blank columns (all missing entries) in open_list and indiv_list

# print("Proportion of missing values", open_list.isnull().mean())

# Extract columns where not all values are missing

cols = open_list.columns[open_list.isnull().mean() != 1]

open_list = open_list[cols] # Get rid of totally missing values in the column

# Repeat for indiv_list

cols = indiv_list.columns[indiv_list.isnull().mean() != 1]

indiv_list = indiv_list[cols]



# Bar shows the proportion/number of non-missing values

plt.subplot(121)

plt.gca().set_title('open_list', fontsize=30) 

mn.bar(open_list,color=(0.25, 0.5, 0.25)) # ignore completely missing columns

plt.subplot(122)

plt.gca().set_title('indiv_list', fontsize=30) 

mn.bar(indiv_list,color=(0.5, 0.25, 0.25))
# Visualize the missing values in the dataframe

mn.matrix(indiv_list,color=(0.5, 0.25, 0.25)) # adds more red
# indiv_list contains less variables (columns)

# heatmap to pick up interesting correlations between missing values

mn.heatmap(indiv_list)
#dendrogram reports on the closeness of missing values in different variables

mn.dendrogram(open_list)

mn.dendrogram(indiv_list) 
display(open_list[['date_confirmation','geo_resolution', 'longitude', 'latitude', 'country_new','admin_id']].head(10))

# indiv_list['death'].value_counts().head(10)
display(indiv_list[['reporting date','link','source','recovered','death','visiting Wuhan', 'id','location']].head(2))

display("Main sources of news mainly include offical government sources with some popular media outlets:")

indiv_list['source'].value_counts().head(10)
import seaborn as sb



data=indiv_list[['visiting Wuhan','death','recovered']]

# [el not in ['0','1'] for el in indiv_list['recovered'].values]

 

 

select = [el not in ['0','1'] for el in data['recovered'].values]

data = data.copy()

data.loc[select,'recovered'] = '1'

select = [el not in ['0','1'] for el in data['death'].values]

data.loc[select,'death'] = '1'

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

sb.countplot(x='visiting Wuhan', data=data,ax=axes[0], hue='death').set_title('Patients who Visited Wuhan: Deaths')

#  

sb.countplot(x='visiting Wuhan', data=data,ax=axes[1], hue='recovered').set_title('Patients who Visited Wuhan: Recovered')

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import nltk

def word_counts(df):

    StopWords = ['confirmed','COVID-19','patient','new',"'new",'onset','female','male','went']#,"'new"," in ","on","to","of","and","went",",","from"]

    txt = str(df.tolist()).split(' ')

    text1 = [str(item).replace(" in ","") for item in txt if str(item) not in StopWords]

    text = [str(item).replace(",","").replace("'","") for item in text1 if str(item) not in STOPWORDS]

    counts = pd.Series(text).value_counts()

    wordcloud = WordCloud(background_color="white").generate_from_frequencies(counts)

    plt.imshow(wordcloud,  interpolation="bilinear")

    plt.axis("off")

    return counts

# Visualize the wordcloud

fig = plt.figure(figsize=(20,10))

df1 = indiv_list['summary'][indiv_list['from Wuhan'] < 1]

plt.subplot(1,2,1).title.set_text("Text Summary for patients, not from Wuhan")

text_notfromWuhan = word_counts(df1)

#

df2 = indiv_list['summary'][indiv_list['from Wuhan'] > 0]

plt.subplot(1,2,2).title.set_text("Text Summary for patients from Wuhan")

text_fromWuhan = word_counts(df2)

# display("Cases not from Wuhan:", len(df1), text_notfromWuhan.head(20),"Cases from Wuhan:", len(df2), text_fromWuhan.head(20))
import seaborn as sb

data=indiv_list[['gender','age','death','recovered']]

# [el not in ['0','1'] for el in indiv_list['recovered'].values]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

sb.violinplot(x='gender', y='age', data=data,ax=axes[0]).set_title('Violinplot: Age vs. Gender')

plt.title('Boxplot: Age vs. Gender')

sb.boxplot(x='gender', y='age', data=data,ax=axes[1])



# pd.options.mode.chained_assignment = None

data=indiv_list[['gender','age','death','recovered']]

select = [el not in ['0','1'] for el in data['recovered'].values]

data['recovered'][select] = '1'

select = [el not in ['0','1'] for el in data['death'].values]

data['death'][select] = '1'

# data.loc[select, 'death'] = 1

# plot

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

sb.stripplot(x='gender', y='age', data=data,ax=axes[0], hue='recovered').set_title('Deaths: Age vs. Gender')

plt.title('Recovered Patients: Age vs. Gender')

sb.stripplot(x='gender', y='age', data=data,ax=axes[1], hue='death')



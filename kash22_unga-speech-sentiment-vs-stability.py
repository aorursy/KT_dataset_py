import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
#import nltk.stem
#from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
#from nltk.corpus import stopwords

import string
import re
#from string import punctuation
#import re
#Load UN General debates
df_debates = pd.read_csv('../input/un-general-debates/un-general-debates.csv')
df_debates
df_debates.describe(include='all')
#Load UNSD countries dataset
df_countries = pd.read_csv('../input/unsd-country-codes-and-development-status/UNSD_M49_CountryCodes.csv')
df_countries
#Merge debates dataset with countries on country codes
df_debates = pd.merge(df_debates, 
                   df_countries[['RegionName','CountryName','ISOAlpha3Code', 'DevelopmentStatus']],
                   how='inner', left_on='country', right_on='ISOAlpha3Code')

#remove duplicate column as country code already exist in debates df
df_debates.drop('ISOAlpha3Code',axis=1, inplace=True) 

df_debates
df_debates.isnull().sum()
df_debates.drop(df_debates[(df_debates.year < 2000) | 
                           (~df_debates.RegionName.isin(['Asia']))].index, 
                inplace=True)
df_debates
df_debates.nunique()
from wordcloud import WordCloud

#word cloud plot
def plot_word_cloud(allwords, maxwords):     
    
    wordcloud = WordCloud(background_color='white', width=1600, height=900,                          
                          max_words=maxwords).generate(allwords)
    plt.figure(figsize = (20,10), dpi=1200)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    
    return
#word cloud plot
def plot_word_cloud_fast(allwords, maxwords):     
    
    wordcloud = WordCloud(background_color='white',
                          max_words=maxwords).generate(allwords)
    plt.figure(figsize = (20,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    
    return
rawtext = df_debates.text.sum()
plot_word_cloud_fast(rawtext, 2000)
#download nltk stopwords for unhelpful words
nltk.download("stopwords")

#download nltk word tokenizer trained on English
nltk.download("punkt")
formal_words=['united nation', 'united nations', 'general assembly', 'republic of', 'secretary general', 'the world', 
              'international community', 'security council', 'member state', 'country', 'must', 'many' ]

#create a function for pre processing of the text
def preprocess_debate(text):    
    #change to lower case and remove special characters
    text = re.sub('\W+', ' ', text.lower())
    
    #remove formal words used in address
    for x in formal_words:
        text = text.replace(x,'')
        
    #tokenize text to get list of individual words
    tokens = nltk.word_tokenize(text)
    
    #remove uninformative words e.g. 'of', 'that', 'is' etc & punctiations e.g. '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    uninformative_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
    tokens = [token for token in tokens if len(token) > 3 and
             token not in uninformative_words]
    
    #remove inflectional endings and get the root word (lemma):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    
    return lemmas
#apply pre processing on texts and create new features 'tokens' and 'freq_dist'
df_debates['tokens'] = df_debates['text'].apply(preprocess_debate)
df_debates['lemma_text'] = [' '.join(x) for x in df_debates['tokens']]

df_debates['freq_dist'] = df_debates['tokens'].apply(nltk.probability.FreqDist)

df_debates[['text','tokens', 'freq_dist']]
#viualize most common words (after pre-processing) in all debates through world cloud

#combine all words from all debates
#all_words_list = [' '.join(x) for x in df_debates['tokens']]
#all_words = ' '.join(map(str,all_words_list))
all_words = df_debates.lemma_text.sum()
plot_word_cloud_fast(all_words, 4000) 
#plot_word_cloud(all_words, 4000) 
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analyze_polarity(text):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(text)
    
    #return the overall sentiment rating of the text
    return sentiment_dict["compound"]

def get_sentiment(polarity):    
    
    # decide sentiment as positive, negative and neutral 
    if polarity >= 0.05: 
        sent = "Positive"  
    elif polarity <= - 0.05: 
        sent = "Negative"
    else: 
        sent = "Neutral"

    return sent
#get overall polarity scores of the raw text
df_debates['polarity'] = np.array([ analyze_polarity(text) for text in df_debates['text'] ])

#get sentiment value for the given polarity
df_debates['sentiment'] = np.array([ get_sentiment(pol) for pol in df_debates['polarity'] ])

df_debates.head()
df_debates[df_debates.country=='PAK']['sentiment'].value_counts()
temp = df_debates.groupby(['country','sentiment'], as_index=False).session.count()
temp.rename(columns={'session':'count'}, inplace=True)
for c in temp.country.unique():
    try:
        p = temp[(temp.sentiment =='Positive') & (temp.country == c)]['count'].values[0]
        n = temp[(temp.sentiment =='Negative') & (temp.country == c)]['count'].values[0]
        if(n>p):
            print(c)
    except:
        print(c,'error')
#get ndarry from dataframe for plotting
years = df_debates['year'].values
polarity = df_debates['polarity'].values
sentiment = df_debates['sentiment'].values
#Quick Plot - Polarity
#fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
#ax.bar(years,polarity)
plt.plot(years,polarity, 'bo')
plt.show()
#Quick Plot - Sentiment
#fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
#ax.bar(years,sentiment)
plt.plot(years,sentiment, 'bo')
plt.show()
stability_df = pd.read_excel('../input/governance-indicator-for-political-stability/Political_Stability_Absence_Violence_Terrorism_W_G_Indicator.xlsx','Data')
stability_df
#Missing years before 2000, dropping those columns
stability_df.drop([1996,1998], axis=1, inplace=True)


#reshape data from wide format to long format
stability_df = stability_df .melt(id_vars=['CountryName', 'CountryCode'],
                                  var_name='Indi_Year',
                                  value_name='Indi_Value')
stability_df
#drop non-asian countries
#since df_debates is already filtered on Asian region, hence using countries from it
stability_df.drop(stability_df[~stability_df.CountryCode.isin(df_debates.country)].index, inplace=True)
stability_df
plt.plot(stability_df.Indi_Year.unique(), stability_df[stability_df.CountryCode=='PAK'].Indi_Value.values)
plt.show()
plt.plot(df_debates.year.unique(), df_debates[df_debates.country=='PAK'].polarity.values, 'bo')
plt.show()
#Visualizing relation through a scatter plot

fig, ax = plt.subplots()
ax.scatter(x=stability_df[stability_df.CountryCode=='PAK'].Indi_Value.values, 
           y=df_debates[df_debates.country=='PAK'].polarity.values, alpha=0.7, color='b')



#set grid and fig size
ax.grid(True, ls=':')
fig.set_figheight(6)
fig.set_figwidth(10)
fig.tight_layout()

plt.show()
fig, ax = plt.subplots()

#plot stability
ax.plot(stability_df.Indi_Year.unique(), stability_df[stability_df.CountryCode=='BGD'].Indi_Value.values, color='orange', marker='o')
ax.set_xlabel('Year')
ax.set_ylabel('stability')
ax.set_ylim([0, 25])
ax.grid(True, ls=':')

#get twin object for two different y-axis on the same plot
ax2 = ax.twinx()

#plot polarity
ax2.plot(stability_df.Indi_Year.unique(),df_debates[df_debates.country=='BGD'].polarity.values, 'bo')
ax2.set_ylabel('polarity')
ax2.set_ylim([-1.5, 1.5])
ax2.grid(True, ls=':')

#set legend and fig size
fig.set_figheight(6)
fig.set_figwidth(12)
fig.tight_layout()
#fig.legend([mort_df['IndicatorName'].iloc[0], epc_df['IndicatorName'].iloc[0]], 
#           loc="upper right",
#           bbox_to_anchor=(1,1), 
#           bbox_transform=ax.transAxes)

#plt.title('Female Mortality Vs Health Expenditure in South Asia')
plt.show()

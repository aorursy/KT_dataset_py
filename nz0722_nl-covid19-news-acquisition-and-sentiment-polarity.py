import re

import time

import random



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from bs4 import BeautifulSoup



pd.options.display.max_colwidth = 1000



KEYWORD = "covid-19 OR corona OR pneumonia"

LOAD_AMOUNT = 100

MIN_SLEEPING_TIME = 5
# selenium setup reference: https://www.kaggle.com/dierickx3/kaggle-web-scraping-via-headless-firefox-selenium

def load_tom_selenium_settings():

    # LOOK AT INPUT FOLDER, WE SHOULD SEE "firefox-63.0.3" FOLDER ALREADY THERE

    !ls -l "../input"



    # WE WILL MAKE NEW SUBFOLDER IN WORKING FOLDER (WHICH ISN'T READ-ONLY)

    !mkdir "../working/firefox"

    !ls -l "../working"



    # COPY OVER FIREFOX FOLDER INTO NEW SUBFOLDER JUST CREATED

    !cp -a "../input/firefox-63.0.3.tar.bz2/firefox/." "../working/firefox"

    !ls -l "../working/firefox"



    # ADD READ/WRITE/EXECUTE CAPABILITES

    !chmod -R 777 "../working/firefox"

    !ls -l "../working/firefox"



    # INSTALL PYTHON MODULE FOR AUTOMATIC HANDLING OF DOWNLOADING AND INSTALLING THE GeckoDriver WEB DRIVER WE NEED

    !pip install webdriverdownloader



    # INSTALL LATEST VERSION OF THE WEB DRIVER

    from webdriverdownloader import GeckoDriverDownloader

    gdd = GeckoDriverDownloader()

    gdd.download_and_install("v0.23.0")



    # INSTALL SELENIUM MODULE FOR AUTOMATING THINGS

    !pip install selenium



    # LAUNCHING FIREFOX, EVEN INVISIBLY, HAS SOME DEPENDENCIES ON SOME SCREEN-BASED LIBARIES

    !apt-get install -y libgtk-3-0 libdbus-glib-1-2 xvfb
load_tom_selenium_settings()

# selenium setup reference: https://www.kaggle.com/dierickx3/kaggle-web-scraping-via-headless-firefox-selenium



from selenium import webdriver as selenium_webdriver

from selenium.webdriver.firefox.options import Options as selenium_options

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities as selenium_DesiredCapabilities
browser_options = selenium_options()

browser_options.add_argument("--headless")



capabilities_argument = selenium_DesiredCapabilities().FIREFOX

capabilities_argument["marionette"] = True



browser = selenium_webdriver.Firefox(

    options=browser_options,

    firefox_binary="../working/firefox/firefox",

    capabilities=capabilities_argument

)
url = "https://nos.nl/zoeken/?q=" + str(KEYWORD)



def load_all_pages(url, page_nr):

    

    browser.get(url)

    

    for i in range(page_nr):

        time.sleep(random.randrange(MIN_SLEEPING_TIME, MIN_SLEEPING_TIME + 5, 1))

        browser.execute_script("window.scrollTo(0,document.body.scrollHeight)")

        browser.find_element_by_xpath("//div[@id='column-main']/ul/div/span").click()

        

    page_source = browser.page_source

    

    return page_source
html = load_all_pages(url, LOAD_AMOUNT)
soup = BeautifulSoup(html, "html.parser")



title = soup.find_all("h3", attrs={"class": "search-results__title"})

date_time = soup.find_all("time", attrs={"class": "search-results__time"})

category = soup.find_all("span", attrs={"class": "search-results__category"})

link = soup.find_all("a", attrs={"class": "search-results__link"})



result_table = pd.DataFrame(columns=['title', 'date_time', 'category', 'link'])



for i in range(len(title)):

    single_result = [title[i].string, 

                     date_time[i].get("datetime"), 

                     category[i].text, 

                     link[i].get("href")]

    result_table.loc[i] = single_result

    

result_table.shape
result_table.link = 'https://nos.nl' + result_table.link.astype(str)
result_table.head()
!pip install mtranslate

from mtranslate import translate
result_table.insert(1, "title_translated", np.nan) 



for i in range(result_table.shape[0]):

    print ("translating row " + str(i))

    result_table.loc[i, "title_translated"] = translate(result_table.loc[i, "title"])

    time.sleep(0.5)
result_table.head()
result_table['category'] = result_table['category'].replace('\n','', regex=True)

result_table['category'] = result_table['category'].str.strip()

result_table['category'] = result_table['category'].replace('in ','', regex=True)

result_table['category'] = result_table['category'].replace(', +',',', regex=True)



category_list = []

for i in result_table['category'].unique():

    i = i.split(",")

    category_list = category_list + i

    

category_list_trans = [translate(i) for i in category_list]

category_dictionary = dict(zip(category_list, category_list_trans))
def translating_category(text):

    for key, value in category_dictionary.items():

        text = text.replace(key, value)

    return text



result_table['category_translated'] = result_table['category'].apply(translating_category)

result_table['utc_time'] = pd.to_datetime(result_table['date_time'], utc=True).values.astype('datetime64')

result_table = result_table.drop_duplicates()

result_table = result_table[['title', 'title_translated', 'utc_time', 'category', 'category_translated', 'link', 'date_time']]



result_table.head()
result_table.to_csv('covid-19_news.csv', index=False)
!pip install vadersentiment



from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
df = result_table.copy()



neg, neu, pos, compound = ([] for i in range(4))



analyzer = SentimentIntensityAnalyzer()



for sentence in df['title_translated']:

    vader_scores = analyzer.polarity_scores(sentence)

    neg.append(vader_scores['neg'])

    neu.append(vader_scores['neu'])

    pos.append(vader_scores['pos'])

    compound.append(vader_scores['compound'])

    

df['vader_neg_score'] = neg

df['vader_neu_score'] = neu

df['vader_pos_score'] = pos

df['vader_compound_score'] = compound





df['date'] = df['utc_time'].dt.date

df['week'] = df['utc_time'].dt.week

df['weekly_average'] = df['vader_compound_score'].groupby(df['week']).transform('mean')



df = df.sort_values(by=['utc_time'])
fig, ax = plt.subplots(nrows=2, ncols=1, 

                       figsize=(14, 7), 

                       gridspec_kw={'height_ratios': [10, 2]})



ax[0].scatter(df['utc_time'],

              df['vader_compound_score'],

              color='purple',

              s=1)

ax[0].set_ylabel('Sentiment Scores')

ax[0].title.set_text('Covid-19 Related News Sentiment Analysis\nSource: NOS.nl')



ax[1].plot(df['date'], 

           df['weekly_average'], 

           '-r')

ax[1].set_ylim([-0.2, 0.2])

ax[1].title.set_text('Weekly Average')               



plt.axhline(y=0, linewidth=0.5, color='r')



for i in range(2):

    ax[i].set_xlim([pd.Timestamp('2020-01-01T00'), df['utc_time'].max()])



fig.show()
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



# text = " ".join(title for title in df[df.vader_compound_score < 0].title_translated)

# stopwords = set(STOPWORDS)



# wordcloud = WordCloud(stopwords=stopwords,

#                       width = 1000, 

#                       height = 500, 

#                       background_color="black").generate(text)



# fig = plt.figure(figsize = (30, 20))

# plt.imshow(wordcloud, interpolation='bilinear')

# plt.axis("off")

# plt.show()
df.to_csv('covid-19_news_and_sentiment_polarity.csv', index=False)
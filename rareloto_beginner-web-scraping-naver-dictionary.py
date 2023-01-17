import pandas as pd



convos_samp = pd.read_csv('../input/naver-dictionary-conversation-of-the-day/conversations.csv').fillna('')

convos_samp
convo_titles_samp = pd.read_csv('../input/naver-dictionary-conversation-of-the-day/conversation_titles.csv').fillna('')

convo_titles_samp
import urllib.request

import re



# Let's try one webpage for now

url = 'https://learn.dict.naver.com/conversation#/korean-en'



page = urllib.request.urlopen(url)

page = str(page.read().decode())



# problem: not retrieving complete page html like Firefox html download does

#          BeautifulSoup doesn't seem to retrieve complete html either

# possible solution: should i try using headless firefox?

# solved! - In-kernel Web scraping via a headless Firefox browser with Selenium at the bottom page
# regex to find conversation date

date = re.findall(r'var regionDate = "([0-9]+)"', page)

# regex to find conversation title

convo_title = re.findall(r'id="ogTitle" content="(.+)">', page)

# regex to extract sentence pairs

eng_sents = re.findall(r'<div class="txt_trans ng-binding" ng-show="transDisplay" ng-bind="item.trsl_sentence">(.+)<.div>', page)

# how to strip html from text - jxb-bind-compiled-html binding ?

kor_sents = re.findall(r'<span class="u_word_dic" data-hook="tip" data-type="arken" data-lang="ko">(.+)</span>.</span></span>', page)



# extracting other data (e.g. conversation title, grammar, grammar description, related words)
date
kor_sents
eng_sents
# Checking kernel OS info

!cat /etc/os-release
# Downloading Firefox for Linux

!wget 'https://download-installer.cdn.mozilla.net/pub/firefox/releases/79.0/linux-x86_64/en-US/firefox-79.0.tar.bz2'



# Extracting Firefox binary

!tar -xjf 'firefox-79.0.tar.bz2'
# Checking working directory

!ls /kaggle/working
# Adding read/write/execute capabilities to 'firefox' directory

!chmod -R 777 '../working/firefox'
# Installing Firefox dependencies

!apt-get install -y libgtk-3-0 libdbus-glib-1-2 xvfb
# Installing Python module for automatic handling of GeckoDriver download and installation

!pip install webdriverdownloader
# Installing GeckoDriver

from webdriverdownloader import GeckoDriverDownloader



gdd = GeckoDriverDownloader()

gdd.download_and_install('v0.23.0')
# Installing Selenium

!pip install selenium
# Loading Python modules to use

import pandas as pd

import seaborn as sns

from IPython.display import Image

import time



from selenium import webdriver as selenium_webdriver

from selenium.webdriver.firefox.options import Options as selenium_options

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities as selenium_DesiredCapabilities



from selenium.webdriver.common.by  import By as selenium_By

from selenium.webdriver.support.ui import Select as selenium_Select

from selenium.webdriver.support.ui import WebDriverWait as selenium_WebDriverWait

from selenium.webdriver.support    import expected_conditions as selenium_ec
# Setting up a virtual screen for Firefox

!export DISPLAY=:99
# Firing up a headless browser session with a screen size of 1920x1080

browser_options = selenium_options()

browser_options.add_argument("--headless")

browser_options.add_argument("--window-size=1920,1080")



capabilities_argument = selenium_DesiredCapabilities().FIREFOX

capabilities_argument["marionette"] = True



browser = selenium_webdriver.Firefox(

    options=browser_options,

    firefox_binary="../working/firefox/firefox",

    capabilities=capabilities_argument

)
# Navigating to website

browser.get("https://learn.dict.naver.com/conversation#/korean-en")

print(browser.current_url)



# Giving the page up to 10 seconds to load

wait = selenium_WebDriverWait(browser, 10)

wait.until(selenium_ec.visibility_of_element_located((selenium_By.XPATH, '//div[@class="reading_lst_wrap"]')))



# Taking a screenshot of the webpage

browser.save_screenshot("screenshot.png")

Image("screenshot.png", width=800, height=500)
# Waiting for another 10 seconds to make sure the page is complete

time.sleep(10)



# Retrieving page source

page = browser.page_source

page[0:1000]
# regex to extract conversation date

date = re.findall(r'var regionDate = "([0-9]+)"', page)

# regex to extract conversation title in korean

kor_title = re.findall(r'id="ogTitle" content="오늘의 회화 - (.+)">', page)

# regex to extract conversation title in english

eng_title = re.findall(r'<span class="txt_trans ng-binding" ng-bind="title_translation">(.+)</span>', page)

# regex to extract sentence pairs

eng_sents = re.findall(r'<div.+item.trsl_sentence">(.+)</div>', page)

kor_sents = re.findall(r'<span class="u_word_dic" data-hook="tip" data-type="arken" data-lang="ko">(.+)</span></span>', page)
date
kor_title
eng_title
eng_sents
kor_sents[0:3]
# Stripping HTML tags from text

def strip_tags(sent):

    sent = re.sub(r'<.+?>', '', sent)

    return sent
kor_sents = list(map(strip_tags, kor_sents))

kor_sents = kor_sents[0:len(eng_sents)]

kor_sents
# Extracting grammar of the day

grammar = re.findall(r'<span jxb-bind-compiled-html.+item[.]entry_name.+"ng-scope">(.+)</span></span>\s+</div>', page)

grammar = list(map(strip_tags, grammar))

grammar
# Extracting grammar description

grammar_desc = re.findall(r'<span class="txt_trans ng-binding" ng-bind="item.mean">(.+)</span>\s+</div>', page)

grammar_desc = list(map(strip_tags, grammar_desc))

grammar_desc
# Extracting grammar of the day sentence examples

grammar_sents_eng = re.findall(r'<span class="txt_trans ng-binding" ng-bind-html="desc[.]trans.+toHtml">(.+)</span>', page)

grammar_sents_eng
grammar_sents_kor = re.findall(r'<span class="txt_origin ng-isolate-scope" jxb-bind-compiled-html="toAutolinkText\(desc[.]origin\)"><span class="ng-scope"><span class="u_word_dic" data-hook="tip" data-type="arken" data-lang="ko">(.+)</span></span>', page)

grammar_sents_kor = list(map(strip_tags, grammar_sents_kor))

grammar_sents_kor
# convo_titles dataframe columns

title_cols = {

    'date': date,

    'kor_title': kor_title,

    'eng_title': eng_title,

    'grammar': grammar,

    'grammar_desc': grammar_desc

}



# Creating convo_titles DataFrame

convo_titles = pd.DataFrame(title_cols)

convo_titles
# Adding new columns: grammar sentence examples    

for i in range(len(grammar_sents_eng)):

    col = f'grammar_kor_sent_{i+1}'

    convo_titles[col] = grammar_sents_kor[i]

    col = f'grammar_eng_sent_{i+1}'

    convo_titles[col] = grammar_sents_eng[i]

    

convo_titles
# convos dataframe columns

convos_cols = {

    'date': [date for date in date for _ in range(len(eng_sents))],

    'conversation_id': [id+1 for id, _ in enumerate(eng_sents)],

    'kor_sent': kor_sents,

    'eng_sent': eng_sents,

    'qna_id': ''  # from sender or receiver, message or feedback

}



# Creating convos DataFrame

convos = pd.DataFrame(convos_cols)

convos
# Creating 2 empty DataFrames to hold conversations and conversation titles

title_cols = [

    'date',  # 'Conversation of the Day' date

    'kor_title',  # 'Conversation of the Day' title in Korean

    'eng_title',  # english translation of the title

    'grammar',  # grammar of the day

    'grammar_desc'  # grammar description

]

convo_titles = pd.DataFrame(columns = title_cols)



convos_cols = [

    'date',  # 'Conversation of the Day' date

    'conversation_id',  # ordered numbering to indicate conversation flow

    'kor_sent',  # korean sentence

    'eng_sent',  # english translation

    'qna_id'  # from sender or receiver, message or feedback

]

convos = pd.DataFrame(columns = convos_cols)
# function to strip html tags from text

def strip_tags(sent):

    sent = re.sub(r'<.+?>', '', sent)

    return sent
%%time

start_time = time.time()



start_date = '12/04/2017'

end_date = '8/19/2020'



for d in pd.date_range(start=start_date, end=end_date):

    

    # Skip date if Sunday (Weekly Review Quiz)

    if d.day_name() == 'Sunday':

        continue

    

    date = d.strftime('%Y%m%d')

    

    # Navigating to website

    url = f"https://learn.dict.naver.com/conversation#/korean-en/{date}"

    browser.get(url)

    # print(browser.current_url)

    

    # Giving the page up to 10 seconds to load

    wait = selenium_WebDriverWait(browser, 10)

    wait.until(selenium_ec.visibility_of_element_located((selenium_By.XPATH, '//div[@class="reading_lst_wrap"]')))

    

    # Waiting for another 10 seconds before retrieving page source

    time.sleep(10)

    

    # Retrieving page source

    page = browser.page_source

    

    # Extracting data from page

    # regex to extract conversation title in korean

    kor_title = re.findall(r'id="ogTitle" content="오늘의 회화 - (.+)">', page)

    # regex to extract conversation title in english

    eng_title = re.findall(r'<span class="txt_trans ng-binding" ng-bind="title_translation">(.+)</span>', page)

    # regex to extract sentence pairs

    eng_sents = re.findall(r'<div.+item.trsl_sentence">(.+)</div>', page)

    kor_sents = re.findall(r'<span class="u_word_dic" data-hook="tip" data-type="arken" data-lang="ko">(.+)</span></span>', page)

    

    # Stripping HTML tags from kor_sents

    kor_sents = list(map(strip_tags, kor_sents))

    kor_sents = kor_sents[0:len(eng_sents)]

    

    # Extracting grammar of the day

    grammar = re.findall(r'<span jxb-bind-compiled-html.+item[.]entry_name.+"ng-scope">(.+)</span></span>\s+</div>', page)

    grammar = list(map(strip_tags, grammar))

    

    # Extracting grammar description

    grammar_desc = re.findall(r'<span class="txt_trans ng-binding" ng-bind="item.mean">(.+)</span>\s+</div>', page)

    grammar_desc = list(map(strip_tags, grammar_desc))

    

    # Extracting grammar of the day sentence examples

    grammar_sents_eng = re.findall(r'<span class="txt_trans ng-binding" ng-bind-html="desc[.]trans.+toHtml">(.+)</span>', page)

    grammar_sents_kor = re.findall(r'<span class="txt_origin ng-isolate-scope" jxb-bind-compiled-html="toAutolinkText\(desc[.]origin\)"><span class="ng-scope"><span class="u_word_dic" data-hook="tip" data-type="arken" data-lang="ko">(.+)</span></span>', page)

    grammar_sents_kor = list(map(strip_tags, grammar_sents_kor))

    

    # Creating new DataFrame to append to convo_titles

    title_data = {

        'date': date,

        'kor_title': kor_title,

        'eng_title': eng_title,

        'grammar': ['. '.join(grammar)],

        'grammar_desc': ['. '.join(grammar_desc) if len(grammar_desc) > 0 else '']

    }

    title = pd.DataFrame(title_data)

    

#     # Additional columns of title DataFrame

#     for i in range(len(grammar_sents_eng)):

#         col = f'grammar_kor_sent_{i+1}'

#         title[col] = grammar_sents_kor[i]

#         col = f'grammar_eng_sent_{i+1}'

#         title[col] = grammar_sents_eng[i]

    

    # Creating new DataFrame to append to convos

    convo_data = {

        'date': [date for date in [date] for _ in range(len(eng_sents))],

        'conversation_id': [id+1 for id, _ in enumerate(eng_sents)],

        'kor_sent': kor_sents,

        'eng_sent': eng_sents,

        'qna_id': ''

    }

    convo = pd.DataFrame(convo_data)

    

    # Appending extracted data to convo_titles and convos DataFrames

    convo_titles = convo_titles.append(title, ignore_index = True)

    convos = convos.append(convo, ignore_index = True)

    

# Printing shapes

print('convos shape:', convos.shape)

print('convo_titles shape:', convo_titles.shape)

print('Time taken to extract data:', '{:.2f}'.format((time.time() - start_time) / 60))
convos
convo_titles
# Exporting to CSV files

convos.to_csv('conversations.csv', index = False)

convo_titles.to_csv('conversation_titles.csv', index = False)
# Deleting unwanted files in working directory

!rm -rf firefox

!rm firefox-79.0.tar.bz2

!rm geckodriver.log

!ls ../working
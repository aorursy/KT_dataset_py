# Libraries

import pandas as pd

import numpy as np

from requests import get

from bs4 import BeautifulSoup

from tqdm import tqdm

import os

import re
# The URL I want to scrap data on

url = 'https://www.phrases.org.uk/meanings/phrases-and-sayings-list.html'



# Prepare GET request

response = get(url)



# Retrieve the webpage and store it as an bs4.BeautifulSoup object

html_soup = BeautifulSoup(response.text, 'html.parser')
quotes = html_soup.find_all('p', class_ = 'phrase-list')

size = len(quotes)

quotes[:5]
cleaned_quotes = [quotes[i].text for i in range(size)]

cleaned_quotes[:5]
href_quotes = [quotes[i].a['href'] for i in range(size)]
# The base link

BASE_LINK = 'https://www.phrases.org.uk/meanings/'



def get_explanations(url):

    

    # This chunk of code is the same we used in the begining of this notebook

    url = url

    response = get(BASE_LINK + url)

    html_soup = BeautifulSoup(response.text, 'html.parser')

    

    quote_explanation = html_soup.find_all('p', class_ = 'meanings-body')

    if len(quote_explanation) >= 1:

        quote_explanation = str(quote_explanation[0].text)

    else:

        quote_explanation = "NO INFORMATION"

        

    return quote_explanation
%%time

# This might take a while, you can grab a coffee or just reduce dimensionality.

# Here I chose only the five first quotes so it runs faster

number_of_quotes = 5

assert number_of_quotes < len(quotes)



explanations = [get_explanations(i) for i in tqdm(href_quotes[:number_of_quotes])]
# Constructing the final dataframe

quotes_dataframe = pd.DataFrame()

quotes_dataframe['text'] = quotes[:number_of_quotes]

quotes_dataframe['text'] = quotes_dataframe['text'].apply(lambda x:x.text)

quotes_dataframe['explanation'] = explanations

quotes_dataframe['origin'] = 'English'



quotes_dataframe.head()
# Save all the data in .csv file

quotes_dataframe.to_csv('English_phrases_and_sayings.csv')
# The URL I want to scrap data on

url = 'https://www.chinahighlights.com/travelguide/learning-chinese/chinese-sayings.htm'



# Prepare GET request

response = get(url)



# Retrieve the webpage and store it as an bs4.BeautifulSoup object

html_soup = BeautifulSoup(response.text, 'html.parser')
proverbs_container = html_soup.find_all('div', class_ = 'col-md-19 col-sm-19 col-xs-24 pull-right')
def starts_with_digit(text):

    

    """

    Return boolean, does 'text' starts with a number

    text: string

    """

    

    output = False

    # This pattern finds if a string is starting is a number

    pattern = re.compile(r'\d*')

    # If this pattern match with something, return True

    if pattern.search(text).group() != '':

        output = True

        

    return output
# All the p tags in proverbs_container

to_browse = proverbs_container[0].find_all('p')
# List of sentence starting with a number

mask = [starts_with_digit(quote.text) for quote in to_browse]



# Filter to get all the proverbs

list_of_proverbs =[to_browse[i].text for i in range(len(mask)) if mask[i] == True]
pattern_chinese = re.compile(r'((?<=\d\.)(.*?)(?=\())')

pattern_chinese.search("6. 三个和尚没水喝。 (Sān gè héshàng méi shuǐ hē. 'three monks have no water to drink') — Too many cooks spoil the broth.").group()
pattern_pin_yin = re.compile(r'((?<=\()(.*?)(?=\)))')

pattern_pin_yin.search("6. 三个和尚没水喝。 (Sān gè héshàng méi shuǐ hē. 'three monks have no water to drink') — Too many cooks spoil the broth.").group()
pattern_translation = re.compile(r'(?<=\—)(.*?)$')

pattern_translation.search("6. 三个和尚没水喝。 (Sān gè héshàng méi shuǐ hē. 'three monks have no water to drink') — Too many cooks spoil the broth.").group()
chinese_proverbs = pd.DataFrame()

chinese_proverbs['all_text'] = list_of_proverbs

chinese_proverbs['in_chinese'] = chinese_proverbs['all_text'].apply(lambda x:pattern_chinese.search(x).group())

chinese_proverbs['pin_yin'] = chinese_proverbs['all_text'].apply(lambda x:pattern_pin_yin.search(x).group())

chinese_proverbs['text'] = chinese_proverbs['all_text'].apply(lambda x:pattern_translation.search(x).group())

chinese_proverbs['category'] = "-1"

chinese_proverbs['origin'] = "Chinese"



chinese_proverbs = chinese_proverbs.drop(['all_text'], axis=1)
proverbs_container[0].find_all('h2')
# I define a list with the name of categories

categories = ['Wisdom', 'Friendship', 'Love', 'Family', 'Encouragement', 'Education', 'Literature', 'Dragons']



# I define a list of proverbs per category (same order)

number_of_quotes_per_category = [26, 10, 10, 10, 21, 10 ,30 ,10]



# Put both list in a dict

dict_categories = dict(zip(categories, number_of_quotes_per_category))
# I ensure that we have same number of proverbs

len(list_of_proverbs) == sum(number_of_quotes_per_category)
# I define a cumsum list to get range index of proverbs that are contained in a category

cumsum = np.cumsum(number_of_quotes_per_category)
# I complete the 'category' column

chinese_proverbs.loc[:cumsum[0], 'category'] = categories[0]

for index in range(6):

    chinese_proverbs.loc[cumsum[index]:cumsum[index+1], 'category'] = categories[index+1]

chinese_proverbs.loc[cumsum[6]:, 'category'] = categories[7]
chinese_proverbs.head()
# Save all the data in .csv file

chinese_proverbs.to_csv('Chinese_proverbs.csv')
# The URL I want to scrap data on

url = "https://frenchtogether.com/french-idioms/"



# Prepare GET request

response = get(url)



# Retrieve the webpage and store it as an bs4.BeautifulSoup object

html_soup = BeautifulSoup(response.text, 'html.parser')
quotes = html_soup.find_all('h3')



# Get the list of all french quotes

french_quotes = [quote.text for quote in quotes]



# I got these elements that I need to clean

print(french_quotes[-2:])



# Excluding the two last elements which are not quotes

french_quotes = french_quotes[:-2]
# For each quote, get all p tags that have all the necessary information

all_texts = html_soup.find_all('p')
def has_strong_tag(quote, chunk):

    assert chunk in ['Literally', 'Meaning', 'English counterpart']

    if chunk in quote.contents[0] :

        return True

    else:

        return False
# Get indexes of elements that contain specified keywords 

literally_text_indexes = np.where([has_strong_tag(all_texts[i], 'Literally') for i in range(len(all_texts))])[0]

meaning_text_indexes = np.where([has_strong_tag(all_texts[i], 'Meaning') for i in range(len(all_texts))])[0]

eng_cnt_text_indexes = np.where([has_strong_tag(all_texts[i], 'English counterpart') for i in range(len(all_texts))])[0]
all_texts_arr = np.array(all_texts)



# Filter to keep all the Literally texts

literally_filtered = list(all_texts_arr[literally_text_indexes])

literally_real = [i.contents[1] for i in literally_filtered if 'strong' in str(i.contents[0])]



# Filter to keep all the Meaning texts

meanings_filtered = list(all_texts_arr[meaning_text_indexes])

meanings_real = [i.contents[1] for i in meanings_filtered if 'strong' in str(i.contents[0])]



# Filter to keep all the English Counterpart texts

eng_cnt_filtered = list(all_texts_arr[eng_cnt_text_indexes])

eng_cnt_real = [i.contents[1] for i in eng_cnt_filtered if 'strong' in str(i.contents[0])]
# Cleaning

to_find = [not 'strong' in str(i.contents[0]) for i in literally_filtered]

print(np.where(to_find)[0][0])

print(literally_filtered[67])

to_find = [not ':' in str(i.contents[1]) for i in meanings_filtered if 'strong' in str(i.contents[0])]

print(np.where(to_find)[0][0])

print(meanings_filtered[1])
all_texts_arr[literally_text_indexes[67]]

all_texts_arr[meaning_text_indexes[1]]
literally_text_indexes = [i for i in literally_text_indexes if i != literally_text_indexes[67]]

meaning_text_indexes = [i for i in meaning_text_indexes if i != meaning_text_indexes[1]]
# Construct DataFrame

column_data = ['in_french', 'literally', 'meaning', 'text']

french_expressions = pd.DataFrame(index = range(0, 90) ,columns = column_data)



french_expressions['in_french'] = french_quotes

french_expressions['literally'] = literally_real    

french_expressions['meaning'] = meanings_real

french_expressions['lit_index'] = literally_text_indexes

french_expressions['mea_index'] = meaning_text_indexes

french_expressions['origin'] = "French"



for index, pivot in enumerate(eng_cnt_text_indexes):

    for i in range(len(french_expressions)-1):

        if ((french_expressions.loc[i,'mea_index']<pivot) and (french_expressions.loc[i+1,'mea_index']>pivot)):

            french_expressions.loc[i, 'text'] = eng_cnt_real[index]

french_expressions.loc[89, 'text'] = eng_cnt_real[-1]
# Cleaning dataframe

french_expressions.literally = french_expressions.literally.apply(lambda x:str(x).replace(':','').replace(';', ''))

french_expressions.meaning = french_expressions.meaning.apply(lambda x:str(x).replace(':','').replace(';', ''))

french_expressions.text = french_expressions.text.apply(lambda x:str(x).replace(':','').replace(';', ''))

french_expressions = french_expressions.drop(['lit_index', 'mea_index'], axis = 1)
french_expressions.head()
# Save all the data in .csv file

french_expressions.to_csv('French_expressions.csv', index=False)
Eng_quotes = pd.read_csv("../input/phrases-and-sayings/English_phrases_and_sayings.csv")

Chi_quotes = pd.read_csv("../input/phrases-and-sayings/Chinese_proverbs.csv")

Fre_quotes = pd.read_csv("../input/phrases-and-sayings/French_expressions.csv")



new_df = pd.concat([Eng_quotes, Chi_quotes, Fre_quotes], join="inner").reset_index(drop=True)

new_df.to_csv('Concatenated_quotes.csv')
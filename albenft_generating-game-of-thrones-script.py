import requests # Used for doing the http request

from bs4 import BeautifulSoup # Used for assisting the HTML read
season_urls = ['https://genius.com/albums/Game-of-thrones/Season-' + str(season_number) + '-scripts' for season_number in range(1,9)]
for season_url in season_urls:

    print(season_url)
r = requests.get(season_urls[0])

html_doc = r.text

soup = BeautifulSoup(html_doc)



# only view snippet because the result is too large

str(soup)[:1000]
url_containers = soup.find_all('a', class_='u-display_block')

# take a look at one of the items

url_containers[0]
urls = [url_container['href'] for url_container in url_containers]



# Take a look at the URLs inside

for url in urls:

    print(url)
urls = []

for season_url in season_urls:

    

    r = requests.get(season_url)

    html_doc = r.text

    soup = BeautifulSoup(html_doc)

    

    url_containers = soup.find_all('a', class_='u-display_block')

    

    for url_container in url_containers:

        urls.append(url_container['href'])
# show number of episodes

len(urls)
for url in urls:

    print(url)
urls = [url for url in urls if 'season' not in url]

len(urls)
url = urls[0]

url
r = requests.get(url)

html_doc = r.text

soup = BeautifulSoup(html_doc)



# take a look inside

# again only snippet because the result too large

str(soup)[:2000]
episode = soup.find_all('div', class_='track_listing-track track_listing-track--current')

# take a look inside

episode
episode = episode[0].text

# take a look inside

episode
# creating a list by splitting the string using '\n'

episode = episode.split('\n')



# remove unused and empty strings

episode = ''.join(e + ' ' for e in episode)

episode = episode.split(' ')

episode = list(filter(None, episode))



# assign episode number and episode title to different variables

episode_number = ''.join('Episode ' + episode[0].split('.')[0])

episode_title = ''.join(e + ' ' for e in episode[1:])[:-1]



# show the results

print(episode_number)

print(episode_title)
# get all elements inside 'a' tag, remove enters, convert to list splitted by empty space

season = soup.find_all('a', class_='song_album-info-title')[0].text.replace('\n','').split(' ')



# remove empty strings and concat all the remaining

season = list(filter(None, season))

season = ''.join(s + ' ' for s in season[:-1])[:-1]



print(season)
# get all elements inside 'a' tag

release_date = soup.find_all('span', class_='metadata_unit-info metadata_unit-info--text_only')

release_date = release_date[0].text



print(release_date)
from datetime import datetime



release_date = datetime.strptime(release_date, '%B %d, %Y')

release_date = datetime.strftime(release_date, '%Y-%m-%d')



print(release_date)
lyrics = soup.find_all("div", class_="lyrics")[0]



# again only snippet because the result too large

str(lyrics)[:2000]
lyrics = BeautifulSoup(str(lyrics))

[s.extract() for s in lyrics('br')]

[s.extract() for s in lyrics('i')]

[s.extract() for s in lyrics('hr')]

[s.extract() for s in lyrics('h1')]

[s.extract() for s in lyrics('h2')]

[s.extract() for s in lyrics('h3')]



# take a look inside

# again only snippet because the result too large

print(str(lyrics)[:3000])
# get the 'p' tags inner HTML

paragraphs = lyrics.find_all('p')



# create variable to store the conversations

conversations = []



# iterating all 'p' tags found

for p in paragraphs:

    # get the inner text of p, create list by splitting text using '\n', and extend them to list outside the loop

    conversations.extend(p.text.split('\n'))

    

# remove empty strings

conversations = list(filter(None, conversations))



# by following the [person]:[sentences] pattern, convert the string inside list to tuple format

conversations = [tuple(s.split(':')) for s in conversations]



for conversation in conversations[:10]:

    print(conversation)
for index, conversation in enumerate(conversations[255:265]):

    if len(conversation) >= 2:

        print(str(index) + ' | 2 values | ' + ''.join(str(c) + ':' for c in conversation)[:-1])

    else:

        print(str(index) + ' | 1 value | ' + ''.join(str(c) + ':' for c in conversation)[:-1])
import re

# regex to find conversations in [ some text ] format

regex = '(.+)\[.+\](.+)|(.+)\[.+\]|\[.+\]'

pattern = re.compile(regex)



for index, conversation in enumerate(conversations):

    if len(conversation) <= 1:

        match = pattern.findall(conversation[0])

        if len(match) > 0:

            conversations[index] = tuple((''.join(e + ' ' for e in list(filter(None, match[0]))).replace('    ',' ').replace('   ',' ').replace('  ', ' ')).split('\n'))



conversations = list(filter(None, conversations))

conversations = [c for c in conversations if len(c[0]) > 0]



# show

conversations[15:25]
# regex that match for '[ some text' and 'some text ]' format

regex = '^\[.+|.+\]$'

pattern = re.compile(regex)



for index, conversation in enumerate(conversations):

    if len(conversation) <= 1:

        match = pattern.search(conversation[0])

        if match:

            conversations[index] = None



conversations = list(filter(None, conversations))

conversations[15:25]
for index, conversation in enumerate(conversations):

    if len(conversation) < 2:

        print(str(index) + ' | ' + conversation[0])
# regex to match with '[person in uppercase] [rest of the text]' and '[person in uppercase] (some text) [rest of the text]' format

regex = '^([A-Z]{2,})(.+)'

pattern = re.compile(regex)



for index, conversation in enumerate(conversations):

    if len(conversation) <= 1:

        match = pattern.findall(conversation[0])

        if len(match) > 0:

            conversations[index] = (match[0][0], match[0][-1])

    

# take a look

conversations[125:135]
for conversation in conversations:

    if len(conversation) < 2:

        print(conversation)
conversations = [conversation for conversation in conversations if len(conversation) > 1]



# take a look

conversations[:10]
import pandas as pd
person = pd.Series([c[0] for c in conversations])

sentence = pd.Series([c[1] for c in conversations])
print(person.head())

print(sentence.head())
script = pd.DataFrame({

    'Season': season,

    'Episode': episode_number,

    'Episode Title': episode_title,

    'Sentence': sentence,

    'Name': person,

    'Release Date': release_date

})

script = script[['Release Date','Season','Episode','Episode Title','Name','Sentence']]

print(script.info())

script.head()
def get_episode(soup):

    episode = soup.find_all('div', class_='track_listing-track track_listing-track--current')[0].text.split('\n')

    episode = ''.join(e + ' ' for e in episode)

    episode = episode.split(' ')

    episode = list(filter(None, episode))



    episode_number = ''.join('Episode ' + episode[0].split('.')[0])

    episode_title = ''.join(e + ' ' for e in episode[1:])[:-1]

    

    return episode_number, episode_title
def get_season(soup):

    season = soup.find_all('a', class_='song_album-info-title')[0].text.replace('\n','').split(' ')

    season = list(filter(None, season))

    season = ''.join(s + ' ' for s in season[:-1])[:-1]

    

    return season
from datetime import datetime



def get_release_date(soup):

    release_date = soup.find_all('span', class_='metadata_unit-info metadata_unit-info--text_only')[0].text

    release_date = datetime.strptime(release_date, '%B %d, %Y')

    release_date = datetime.strftime(release_date, '%Y-%m-%d')

    

    return release_date
import re



def get_conversations(soup):

    lyrics = soup.find_all("div", class_="lyrics")[0]



    lyrics = BeautifulSoup(str(lyrics))

    [s.extract() for s in lyrics('br')]

    [s.extract() for s in lyrics('i')]

    [s.extract() for s in lyrics('hr')]

    [s.extract() for s in lyrics('h1')]

    [s.extract() for s in lyrics('h2')]

    [s.extract() for s in lyrics('h3')]



    paragraphs = lyrics.find_all('p')



    conversations = []



    for p in paragraphs:

        conversations.extend(p.text.split('\n'))



    conversations = list(filter(None, conversations))

    conversations = [tuple(s.split(':')) for s in conversations]

    

    regex = '(.+)\[.+\](.+)|(.+)\[.+\]|\[.+\]'

    pattern = re.compile(regex)

    

    for index, conversation in enumerate(conversations):

        if len(conversation) <= 1:

            match = pattern.findall(conversation[0])

            if len(match) > 0:

                conversations[index] = tuple((''.join(e + ' ' for e in list(filter(None, match[0]))).replace('    ',' ').replace('   ',' ').replace('  ', ' ')).split('\n'))

                

    conversations = list(filter(None, conversations))

    conversations = [c for c in conversations if len(c[0]) > 0]

    

    regex = '^\[.+|.+\]$'

    pattern = re.compile(regex)

    

    for index, conversation in enumerate(conversations):

        if len(conversation) <= 1:

            match = pattern.search(conversation[0])

            if match:

                conversations[index] = None

                

    conversations = list(filter(None, conversations))

    

    regex = '^([A-Z]{2,})(.+)'

    pattern = re.compile(regex)

    

    for index, conversation in enumerate(conversations):

        if len(conversation) <= 1:

            match = pattern.findall(conversation[0])

            if len(match) > 0:

                conversations[index] = (match[0][0], match[0][-1])

                

    conversations = [conversation for conversation in conversations if len(conversation) > 1]

    

    return conversations
def create_dataframe(**kwargs):

    

    person = pd.Series([c[0] for c in conversations])

    sentence = pd.Series([c[1] for c in conversations])

    

    script = pd.DataFrame({

        'Season': season,

        'Episode': episode_number,

        'Episode Title': episode_title,

        'Sentence': sentence,

        'Name': person,

        'Release Date': release_date

    })

    

    script = script[['Release Date','Season','Episode','Episode Title','Name','Sentence']]

    

    return script
# initiate an empty list to store dataframes from each episode

scripts = []

for url in urls:

    r = requests.get(url)

    html_doc = r.text

    soup = BeautifulSoup(html_doc)

    

    episode_number, episode_title = get_episode(soup)

    season = get_season(soup)

    release_date = get_release_date(soup)

    conversations = get_conversations(soup)

    

    df_scripts = create_dataframe(episode_number = episode_number, 

                                  episode_title = episode_title,

                                  season = season,

                                  release_date = release_date,

                                  conversations = conversations)

    

    scripts.append(df_scripts)

    print('Script from: ' + url + ' added')
script_dataframe = pd.concat(scripts)

script_dataframe.info()
script_dataframe = script_dataframe.dropna()

script_dataframe.info()
import re

def remove_bracketed(text):

    regex = '\([^)]*\)'

    text = re.sub(regex, '', text).replace('  ',' ')

    return text



script_dataframe['Name'] = script_dataframe['Name'].apply(remove_bracketed)

script_dataframe['Sentence'] = script_dataframe['Sentence'].apply(remove_bracketed)



script_dataframe.head(3)
script_dataframe['Name'] = script_dataframe['Name'].apply(lambda x: str(x).lower())

script_dataframe.head(3)
import re

def remove_non_alphabetic(text):

    regex = '[^A-Za-z\s]'

    text = re.sub(regex, '', text).replace('  ',' ')

    text = text if text[-1] != ' ' else text[:-1]

    return text
script_dataframe['Name'] = script_dataframe['Name'].apply(remove_non_alphabetic)

script_dataframe.head(3)
script_dataframe['First Token'] = script_dataframe['Name'].apply(lambda x: str(x).split(' ')[0])

script_dataframe.head(3)
script_dataframe = script_dataframe[(script_dataframe['First Token'] != 'cut') &

                                    (script_dataframe['First Token'] != 'int') &

                                    (script_dataframe['First Token'] != 'ext') &

                                    (script_dataframe['First Token'] != 'episode')]

script_dataframe.head(3)
script_dataframe['Name Length'] = script_dataframe['Name'].apply(lambda x: len(str(x)))

print(script_dataframe['Name Length'].describe(percentiles=[.8,.9,.95,.99,.999,.9999,.99999,.999999]))

print(script_dataframe.info())

print(script_dataframe['Name Length'].value_counts().sort_values().head())
script_dataframe = script_dataframe[script_dataframe['Name Length'] <= 28]

script_dataframe.info()
appearance_counts = script_dataframe.groupby(['Name'])['Sentence'].count().reset_index()

appearance_counts.Sentence.describe(percentiles=[.8,.9,.95,.99,.999])
most_sentence_characters = appearance_counts[appearance_counts['Sentence'] > 80].sort_values(by=['Sentence'], ascending=[0])
most_sentence_characters = most_sentence_characters[(most_sentence_characters['Name'] != 'man') &

                                                    (most_sentence_characters['Name'] != 'soldier')]
char_names = most_sentence_characters['Name'].unique()

print('total: ' + str(len(char_names)))

char_names
char_names = ['tyrion lannister', 'jon snow', 'jaime lannister', 'sansa stark', 'arya stark', 'davos',

              'theon greyjoy', 'bronn', 'varys', 'brienne', 'bran stark', 'tywin lannister', 'jorah mormont', 'stannis baratheon',

              'margaery tyrell', 'ramsay bolton', 'melisandre', 'robb stark', 'jon snow', 'shae', 'gendry baratheon',

              'tormund', 'gilly', 'tyrion lannister', 'missandei', 'catelyn stark', 'ygritte', 'olenna tyrell', 'daario',

              'podrick', 'yara greyjoy', 'osha', 'oberyn martell', 'jaqen hghar','grey worm', 'qyburn', 'talisa', 'meera', 'catelyn stark',

              'thoros','robert baratheon', 'arya stark', 'shireen', 'sparrow', 'beric', 'euron greyjoy','sansa stark', 'grenn', 'jorah mormont']



alias_mapper = ['sandor clegane','petyr baelish','petyr baelish','sam tarly','eddard stark','cersei lannister','joffrey lannister',

                'tommen lannister','daenerys targaryen','daenerys targaryen']



alias = ['hound','littlefinger','baelish','samwell tarly','ned stark','cersei baratheon','joffrey baratheon',

         'tommen baratheon','daenerys stormborn','dany']



char_names = sorted(list(pd.Series(char_names).unique()))

char_alias = [None for i in range(0, len(char_names))]

char_names.extend(alias_mapper)

char_alias.extend(alias)

name_dictionary = pd.DataFrame({

    "Base Name": char_names,

    "Alias": char_alias

})



name_dictionary = name_dictionary[['Base Name','Alias']]

name_dictionary = name_dictionary.sort_values(by=['Base Name'])

name_dictionary.head()
def clean_words(x):

    new_name = x.replace('the ','')

    new_name = new_name.replace('high ', '')

    return new_name
script_dataframe['Name'] = script_dataframe['Name'].apply(clean_words)
script_for_mapper = script_dataframe.copy()
script_for_mapper['Cartesian Key'] = 0

name_dictionary['Cartesian Key'] = 0

script_for_mapper = script_for_mapper.merge(name_dictionary, on=['Cartesian Key'], how='outer')

script_for_mapper.info()
!pip install jellyfish
from jellyfish import jaro_winkler



def get_similarity(row):

    current_name = row['Name']

    base_name = row['Base Name']

    alias = row['Alias']

    

    score_base_name = 0

    score_alias = 0

    

    if current_name == base_name:

        score_base_name = 1

    else:

        listed_current_name = current_name.split(' ')

        listed_base_name = base_name.split(' ')

        

        if len(listed_current_name) > 1 and len(listed_base_name) > 1:

            family_name_similarity = jaro_winkler(listed_current_name[1], listed_base_name[1])

            if family_name_similarity > .9:

                score_base_name = jaro_winkler(listed_current_name[0], listed_base_name[0])

            else:

                score_base_name = jaro_winkler(current_name, base_name)

        elif len(listed_base_name) > 1:

            score_base_name = jaro_winkler(current_name, listed_base_name[0])

        else:

            score_base_name = jaro_winkler(current_name, base_name)

        

        if alias != None:

            listed_alias = alias.split(' ')

            if len(listed_current_name) > 1 and len(listed_alias) > 1:

                family_name_similarity = jaro_winkler(listed_current_name[1], listed_alias[1])

                if family_name_similarity > .9:

                    score_base_name = jaro_winkler(listed_current_name[0], listed_alias[0])

                else:

                    score_base_name = jaro_winkler(current_name, alias)

            elif len(listed_alias) > 1:

                score_base_name = jaro_winkler(current_name, listed_alias[0])

            else:

                score_base_name = jaro_winkler(current_name, alias)

    

    return score_base_name if score_base_name > score_alias else score_alias
script_for_mapper['Name Similarity'] = script_for_mapper.apply(get_similarity, axis=1)
def get_homogenized_name(x):

    similarity = x['Name Similarity']

    name = x['Name']

    base_name = x['Base Name']

    

    if similarity > .89:

        return base_name

    else:

        return None
script_for_mapper['Homogenized Name'] = script_for_mapper.apply(get_homogenized_name, axis=1)

script_for_mapper.head()
script_for_mapper = script_for_mapper[['Name','Homogenized Name']].dropna().drop_duplicates()

script_for_mapper.head()
script_for_mapper = script_for_mapper.drop(1031097)
script_dataframe = script_dataframe.merge(script_for_mapper, on=['Name'], how='left')

script_dataframe.head()
script_dataframe['Homogenized Name'] = script_dataframe['Homogenized Name'].fillna('')

script_dataframe['Name'] = script_dataframe[['Name','Homogenized Name']].apply(lambda x: x[1] if x[1] != '' else x[0], axis=1)

script_dataframe.head()
script_dataframe = script_dataframe[['Release Date','Season','Episode','Episode Title','Name','Sentence']].drop_duplicates()

script_dataframe.head()
import re

def clean_sentence(text):

    

    text_list = text.split(' ')

    

    if len(text_list) > 1:

        text = ''.join(' ' + word for word in text_list if word != '')[1:].replace('    ',' ').replace('   ',' ').replace('  ',' ')

        if len(text) > 1:

            text = text[:-1] if text[-1] == ' ' else text

            if text[0] == '"' and text[-1] == '"':

                text = text[1:-1]

            if text[0] == '\'' and text[-1] == '\'':

                text = text[1:-1]



        regex = '^[^A-Za-z0-9]*'

        text = re.sub(regex, '', text).replace('  ',' ')

        if len(text) > 0:

            text = text if text[-1] != ' ' else text[:-1]

    

    return text
script_dataframe['Clean Sentence'] = script_dataframe['Sentence'].apply(clean_sentence)

script_dataframe.head(3)
script_dataframe['Length Sentence'] = script_dataframe['Clean Sentence'].apply(len)

script_dataframe = script_dataframe[script_dataframe['Length Sentence'] > 1]

script_dataframe.head(3)
script_dataframe['Sentence'] = script_dataframe['Clean Sentence']

script_dataframe = script_dataframe[['Release Date','Season','Episode','Episode Title','Name','Sentence']].drop_duplicates()

script_dataframe.head()
script_dataframe['Sentence'] = script_dataframe['Sentence'].apply(lambda x: str(x).replace('’', '\''))

script_dataframe['Sentence'] = script_dataframe['Sentence'].apply(lambda x: str(x).replace('d\'', 'do '))

script_dataframe.head()
script_dataframe.info()
script_dataframe.to_csv('Game_of_Thrones_Script.csv', encoding='utf-8', index=False)
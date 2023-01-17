# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import pandas as pd

import re

import emoji

import numpy as np



import nltk



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nltk.download('punkt')
df=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
df_test=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
df.head()
# https://stackoverflow.com/a/34682849

from bs4 import BeautifulSoup

import unicodedata

from nltk.tokenize import word_tokenize



def untokenize(words):

    """

    Untokenizing a text undoes the tokenizing operation, restoring

    punctuation and spaces to the places that people expect them to be.

    Ideally, `untokenize(tokenize(text))` should be identical to `text`,

    except for line breaks.

    """

    text = ' '.join(words)

    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .', '...')

    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")

    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)

    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)

    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(

        "can not", "cannot")

    step6 = step5.replace(" ` ", " '")

    return step6.strip()





# https://stackoverflow.com/a/47091490

def decontracted(phrase):

    """Convert contractions like "can't" into "can not"

    """

    # specific

    phrase = re.sub(r"won\'t", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase





# https://github.com/rishabhverma17/sms_slang_translator/blob/master/slang.txt

slang_abbrev_dict = {

    'AFAIK': 'As Far As I Know',

    'AFK': 'Away From Keyboard',

    'ASAP': 'As Soon As Possible',

    'ATK': 'At The Keyboard',

    'ATM': 'At The Moment',

    'A3': 'Anytime, Anywhere, Anyplace',

    'BAK': 'Back At Keyboard',

    'BBL': 'Be Back Later',

    'BBS': 'Be Back Soon',

    'BFN': 'Bye For Now',

    'B4N': 'Bye For Now',

    'BRB': 'Be Right Back',

    'BRT': 'Be Right There',

    'BTW': 'By The Way',

    'B4': 'Before',

    'B4N': 'Bye For Now',

    'CU': 'See You',

    'CUL8R': 'See You Later',

    'CYA': 'See You',

    'FAQ': 'Frequently Asked Questions',

    'FC': 'Fingers Crossed',

    'FWIW': 'For What It\'s Worth',

    'FYI': 'For Your Information',

    'GAL': 'Get A Life',

    'GG': 'Good Game',

    'GN': 'Good Night',

    'GMTA': 'Great Minds Think Alike',

    'GR8': 'Great!',

    'G9': 'Genius',

    'IC': 'I See',

    'ICQ': 'I Seek you',

    'ILU': 'I Love You',

    'IMHO': 'In My Humble Opinion',

    'IMO': 'In My Opinion',

    'IOW': 'In Other Words',

    'IRL': 'In Real Life',

    'KISS': 'Keep It Simple, Stupid',

    'LDR': 'Long Distance Relationship',

    'LMAO': 'Laugh My Ass Off',

    'LOL': 'Laughing Out Loud',

    'LTNS': 'Long Time No See',

    'L8R': 'Later',

    'MTE': 'My Thoughts Exactly',

    'M8': 'Mate',

    'NRN': 'No Reply Necessary',

    'OIC': 'Oh I See',

    'OMG': 'Oh My God',

    'PITA': 'Pain In The Ass',

    'PRT': 'Party',

    'PRW': 'Parents Are Watching',

    'QPSA?': 'Que Pasa?',

    'ROFL': 'Rolling On The Floor Laughing',

    'ROFLOL': 'Rolling On The Floor Laughing Out Loud',

    'ROTFLMAO': 'Rolling On The Floor Laughing My Ass Off',

    'SK8': 'Skate',

    'STATS': 'Your sex and age',

    'ASL': 'Age, Sex, Location',

    'THX': 'Thank You',

    'TTFN': 'Ta-Ta For Now!',

    'TTYL': 'Talk To You Later',

    'U': 'You',

    'U2': 'You Too',

    'U4E': 'Yours For Ever',

    'WB': 'Welcome Back',

    'WTF': 'What The Fuck',

    'WTG': 'Way To Go!',

    'WUF': 'Where Are You From?',

    'W8': 'Wait',

    '7K': 'Sick:-D Laugher'

}





def unslang(text):

    """Converts text like "OMG" into "Oh my God"

    """

    if text.upper() in slang_abbrev_dict.keys():

        return slang_abbrev_dict[text.upper()]

    else:

        return text







# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

def replace_emoji_to_text(text):

    return emoji.demojize(text)









def replace_urls(text):

    text=re.sub(r"http\S+", 'URL', text)

    text=re.sub(r"www\S+", 'URL', text)

    text=re.sub(r"pic.twitter.com\S+", 'URL', text)

    text=re.sub(r'https.*[^ ]', 'URL',text)

    

    return text



def clean(reg_exp, text):

    text = re.sub(reg_exp, " ", text)



    # replace multiple spaces with one.

    text = re.sub('\s{2,}', ' ', text)



    return text



def remove_accented_chars(text):

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    return text



def strip_html_tags(text):

    soup = BeautifulSoup(text, "html.parser")

    stripped_text = soup.get_text()

    return stripped_text



def clean_all(t):

    

    # first do bulk cleanup on tokens that don't depend on word tokenization



    # remove xml tags



    # remove URLs

    t = replace_urls(t)



    

    

    # Removing HTML tags

    t=strip_html_tags(t)

    

    #remove_accented_chars

    t=remove_accented_chars(t)



    # https://stackoverflow.com/a/35041925

    # replace multiple punctuation with single. Ex: !?!?!? would become ?

    t = clean(r'[\?\.\!]+(?=[\?\.\!])', t)



    t = replace_emoji_to_text(t)



    # expand common contractions like "I'm" "he'll"

    t = decontracted(t)



    # now remove/expand bad patterns per word

    words = word_tokenize(t)



    clean_words = []



    for w in words:

        # normalize punctuation

        w = re.sub(r'&', 'and', w)



        # expand slang like OMG = Oh my God

        w = unslang(w)





        clean_words.append(w)



    # join the words back into a full string

    t = untokenize(clean_words)





    # finally, remove any non ascii and special characters that made it through

    t = clean(r"[^A-Za-z0-9\.\'!\?,\$]", t)



    return t
sentences=list(df.text)

labels=list(df.target)

location=list(df.location)

keyword=list(df.keyword)
sentences_test=list(df_test.text)

id_test=list(df_test.id)

location_test=list(df_test.location)

keyword_test=list(df_test.keyword)
sentences = [clean_all(str(sentence)) for sentence in sentences]

sentences_test = [clean_all(str(sentence)) for sentence in sentences_test]
d = {'text':sentences,'target': labels, 'location': location,'keyword': keyword}

df_train = pd.DataFrame(data=d)
df_train.to_csv('train_clean.csv', index=False)
d = {'text':sentences_test,'id': id_test, 'location': location_test,'keyword': keyword_test}

df_test = pd.DataFrame(data=d)
df_test.to_csv('test_clean.csv', index=False)
df.info()
set(df['keyword'])
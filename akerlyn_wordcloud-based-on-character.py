import pandas as pd

import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
conver_df = pd.read_csv('../input/movie_conversations.tsv',encoding='ISO-8859-2',warn_bad_lines =False,sep='\t',header=None)

lines_df = pd.read_csv('../input/movie_lines.tsv',sep='\t',error_bad_lines=False,warn_bad_lines =False,header=None)

characters_df = pd.read_csv('../input/movie_characters_metadata.tsv',sep='\t',warn_bad_lines =False,error_bad_lines=False,header=None)
characters_df.columns=['chId','chName','mId','mName','gender','score']

characters_df.head()
conver_df.columns = ['chId','chId2','mId','lines']

conver_df.head()
### Conver_df contains lines that characters talk
lines_df.columns = ['lineId','chId','mId','chName','dialogue']

lines_df.head()
df = pd.merge(lines_df, characters_df, how='inner', on=['chId','mId','chName'],

         left_index=False, right_index=False, sort=True,

         suffixes=('_x', '_y'), copy=True, indicator=False)

df.head()
#Select only dialogue that is not null

df = df[df['dialogue'].notnull()]
wordnet_lemmatizer = WordNetLemmatizer()

def clean_dialogue( dialogue ):

    # Function to convert a raw review to a string of words

    # The input is a single string (a raw movie review), and 

    # the output is a single string (a preprocessed movie review)

    #

    # 1. Remove HTML

    #

    # 2. Remove non-letters        

    letters_only = re.sub("[^a-zA-Z]", " ", dialogue) 

    #

    # 3. Convert to lower case, split into individual words

    words = letters_only.lower().split()                             

    #

    # 4. In Python, searching a set is much faster than searching

    #   a list, so convert the stop words to a set

    stops = set(stopwords.words("english"))   

    

    # 5. Use lemmatization and remove stop words

    meaningful_words = [wordnet_lemmatizer.lemmatize(w) for w in words if not w in stops]   

    #

    # 6. Join the words back into one string separated by space, 

    # and return the result.

    return( " ".join( meaningful_words ))   

df['cleaned_dialogue'] = df['dialogue'].apply(clean_dialogue)
df[['cleaned_dialogue','dialogue']].head()
#Get only lines spoken by out focus characters

def getDialogue(name='BIANCA',mName='10 things i hate about you'):

    dialogs = df[(df['chName']==name)&(df['mName']==mName)]['cleaned_dialogue'].values

    if dialogs.shape[0]==0:

        print("Not Found")

        return "Not Found"

    return dialogs

getDialogue('BIANCA','10 things i hate about you')
def getWordCloud(chName,mName):

    dialogues= list(getDialogue(chName,mName))

    words = [word  for dialog in dialogues for word in dialog.split(" ")]

    wordcloud = WordCloud(max_font_size=40,background_color="white").generate(" ".join(words))

    plt.figure()

    plt.title("%s's word cloud from \"%s\""%(chName,mName))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    

    plt.show()

getWordCloud('BIANCA','10 things i hate about you')
def randomWordCloud():

    sample = df.sample(1)

    getWordCloud(sample['chName'].values[0],sample['mName'].values[0])
randomWordCloud()
randomWordCloud()
randomWordCloud()
randomWordCloud()
randomWordCloud()
randomWordCloud()
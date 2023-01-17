import pandas as pd
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import re
#  loading raw data
datasets_dir="/kaggle/input/"
vnrows=None
fake = pd.read_csv(datasets_dir + "Fake.csv",nrows=vnrows, encoding='utf-8')
true = pd.read_csv(datasets_dir + "True.csv",nrows=vnrows, encoding='utf-8')
print (true.shape)
print (fake.shape)
print (true.columns)
print (fake.columns)
# classify the two groups
fake['label']=1
true['label']=0
# merge datasets
df=pd.concat([fake,true]).reset_index(drop=True)
# combineing title and text
df['combined']=df['title']+' '+df['text']

# new feature word count raw content
df['raw_doc_length'] = None
df['raw_doc_length'] = df['combined'].apply(lambda words: len(words.split(" ")))
# implemented preprocessing functions
def basic_clean_text_r1(text):
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Apply a second round of cleaning
def basic_clean_text_r2(text):
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    text = text.lower()
    return text

def remove_stop_words(text):
    STOPWORDS = set(stopwords.words('english'))
      #3. Remove stopwords
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

def stem_words(text):
    stemmer = PorterStemmer()
    final_text = []
    for i in text.split():
            word = stemmer.stem(i.strip())
            """
            if word != i:
                print ("before:" + i)
                print ("after:" + word)
            """                
            final_text.append(word)
    return " ".join(final_text)
# remove text in square brackets, remove punctuation and remove words containing numbers
df['text_round_1'] = None
df['text_round_1'] = df.combined.apply(basic_clean_text_r1)
# lower case / remove additional punctuation and non-sensical text was missed the previous step
df['text_round_2'] = None
df['text_round_2'] = df.text_round_1.apply(basic_clean_text_r2)
# remove stopwords 
df['text_round_3'] = None
df['text_round_3'] = df.text_round_2.apply(remove_stop_words)
# stemming final step  takes several minutes
df['target_text'] = None
df['target_text'] = df.text_round_3.apply(stem_words)
# word count final text
df['doc_length'] = df['target_text'].apply(lambda words: len(words.split(" ")))
# word count before and after preparation
print (df['raw_doc_length'].sum())
print (df['doc_length'].sum())
df.columns
# drop temporary and not required columns
df = df.drop(columns=['subject', 'title', 'text','combined','text_round_1','text_round_2', 'text_round_3']).reset_index(drop=True)
# analyse example: articles  with less then 10 words
df[(df['raw_doc_length'] < 10)].head(5)
# remove news with less than 99 words
df = df[(df['raw_doc_length'] > 99)].reset_index(drop=True)
# evaluating the corpus top frequent words /takes a few minutes
df.target_text.str.split(expand=True).stack().value_counts().head(5)
# final structure
print (df.columns)
print (df.shape)
# peek at the data
df[['target_text','label']]
print (df.groupby(['label']).size())
from wordcloud import WordCloud,  ImageColorGenerator
import matplotlib.pyplot as plt
# assembling the wordcloud content
text = df.target_text.to_string()
#text = X_test_words
# create and generate a word cloud image
wordcloud = WordCloud().generate(text)
# display image
plt.imshow(wordcloud, interpolation='bilinear')
vtitle='Wordcloud perspective after preparation'
plt.title(vtitle)
plt.axis("off")
plt.show()
# are words well balanced between true and false ? Example takes few minutes
print ('"trump" word categorisation:')
print (df['label'].groupby([df['target_text'].str.contains("trump")]).count())

# interesting, negative campaings ?
print ('"obama" word categorisation:')
print (df['label'].groupby([df['target_text'].str.contains("obama")]).count())
# slice data in two datafile one for trainining and test, the smaller for additional testing

# shuffle
df = df.sample(frac=1).reset_index(drop=True)

# assembling main prepared dataset 35.000 articles
df_main = df[:35000]
print ("final main dataset:")
print (df_main.groupby(['label']).size())

# assembling additional   prepared dataset 35.000 articles
df_small=df[35001:]

print ("final additional dataset:")
print (df_small.groupby(['label']).size())
# eventaully save preprocessed data in two files
datasets_dir ="/kaggle/working/"
df_main.to_csv(datasets_dir  + "fakenews_preprocessed_35k.csv")
df_small.to_csv(datasets_dir  + "fakenews_preprocessed_4k.csv")
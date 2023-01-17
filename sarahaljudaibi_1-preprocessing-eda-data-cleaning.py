from IPython.display import Image
Image("../input/imageseda/live_chat_anim_2.gif")

import numpy as np 
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from plotly.offline import init_notebook_mode, iplot 
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from PIL import Image
from collections import Counter
import nltk
import emoji

stopwords = set(STOPWORDS)

validation_data=pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
training_data=pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

testing_data=pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/test.csv')
print('traning data, validation data, test data ')
training_data.shape, validation_data.shape, testing_data.shape


training_data.head()
validation_data.head()
testing_data.head()
training_data.isna().sum()
validation_data.isna().sum()
testing_data.isna().sum()
# find how many unique values in the training dataframe
training_data.nunique()
#validation data has 3 languages non of them english
validation_data.nunique()
#vtest data has 6 languages non of them english
testing_data.nunique()
print(validation_data.lang.unique())
#replace column values with the language name instead of the language code
validation_data.lang.replace('es','Spanish',inplace=True)
validation_data.lang.replace('it','Italian',inplace=True)
validation_data.lang.replace('tr','Turkish',inplace=True)

print(validation_data.lang.unique())
print(testing_data.lang.unique())
#replace column values with the language name instead of the language code
testing_data.lang.replace('es','Spanish',inplace=True)
testing_data.lang.replace('it','Italian',inplace=True)
testing_data.lang.replace('tr','Turkish',inplace=True)
testing_data.lang.replace('ru','Russian',inplace=True)
testing_data.lang.replace('fr','French',inplace=True)
testing_data.lang.replace('pt','Portuguese',inplace=True)

print(testing_data.lang.unique())
training_data.info()
# select toxic comments from data
print("toxic comments:")
print(training_data[training_data.toxic==1].iloc[10,1],'\n')
print(training_data[training_data.toxic==1].iloc[500,1],'\n')
print(training_data[training_data.toxic==1].iloc[1573,1],'\n')
print(training_data[training_data.toxic==1].iloc[4310,1],'\n')


# select non-toxic comments from data
print("non-toxic comments:")
print(training_data[training_data.toxic==0].iloc[10,1],'\n')
print(training_data[training_data.toxic==0].iloc[90,1],'\n')
print(training_data[training_data.toxic==0].iloc[210,1],'\n')
print(training_data[training_data.toxic==0].iloc[4311,1],'\n')

training_data.describe()
# find the top words distribution in the training data

#take the column comment_text and split each word in a column
toxic_words = training_data['comment_text'].str.split(expand=True).unstack().value_counts()

#plot bar chart with 100 value in x and y
data = [go.Bar(x = toxic_words.index.values[:100],y = toxic_words.values[:100],
marker= dict(colorscale='Viridis',color = toxic_words.values[:100]),text='Word counts')]

layout = go.Layout(title='Top 100 Word frequencies in the training dataset without stopword')

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='Word frequencies bar chart')

#use stopword to remove unessesry common words in english
stopwords = nltk.corpus.stopwords.words('english')

# RegEx for stopwords
RE_stopwords = r'\b(?:{})\b'.format('|'.join(stopwords))

# replace characters with ' ' and drop all stopwords
words = (training_data['comment_text'].str.lower().replace([r"[\.\'\,\-\"\?\()\==]", 
RE_stopwords], [' ', ''], regex=True).str.cat(sep=' ').split())


# add the new words frequances in dataframe
rslt = pd.DataFrame(Counter(words).most_common(100),
                    columns=['Word', 'Frequency']).set_index('Word')


#plot bar chart with the most frequancies words
data = [go.Bar(x = rslt.index.values,y = rslt.Frequency.values,
marker= dict(colorscale='Viridis',color = rslt.Frequency.values[:100]),text='Word counts')]

layout = go.Layout(title='Top 100 Word frequencies in the training dataset with stopword')

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='Word frequencies bar chart')


#after we saw the most common words in all the train data 
#let's plot only the most common toxic words 

toxity_data=training_data[training_data.toxic==1]

#use stopword to remove unessesry common words in english
stopwords = nltk.corpus.stopwords.words('english')

# RegEx for stopwords
RE_stopwords = r'\b(?:{})\b'.format('|'.join(stopwords))

# replace characters with ' ' and drop all stopwords
words = (toxity_data['comment_text'].str.lower().replace([r"[\.\'\,\-\"\?\()\==\!]", 
RE_stopwords], [' ', ''], regex=True).str.cat(sep=' ').split())


# add the new words frequances in dataframe
rslt_T = pd.DataFrame(Counter(words).most_common(100),
                    columns=['Word', 'Frequency']).set_index('Word')


#plot bar chart with the most frequancies words
data = [go.Bar(x = rslt_T.index.values[:25],y = rslt_T.Frequency.values[:25],
marker= dict(colorscale='Viridis',color = rslt_T.Frequency.values[:20]),text='Word counts')]

layout = go.Layout(title='Top 100 Toxic Word frequencies in the training dataset with stopword')

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='Word frequencies bar chart')



#plot pie chart to display the toxic words percentage between different toxics words types

#select colors for the pie plot
colors=['gold','darkslateblue','mediumturquoise','lightcoral','lightskyblue','lightseagreen']

#plotting pie  with 5 columns 
fig = go.Figure(data=[go.Pie(labels=training_data.columns[[3,4,5,6,7]],
values=training_data.iloc[:,[3,4,5,6,7]].sum().values, marker=dict(colors=colors))])

# choose to display the percentage outside the circle with color black
fig.update_traces(textposition='outside', textfont=dict(color="black"))
fig.update_layout(title_text="toxic comments types")
fig.show()
#select columns for the first row plot to show the Occurrences of toxic words in these columns
x_data_1=training_data.iloc[:,[3,4,5,6,7]].sum()

#plot
plt.figure(1,figsize=(24,17))
plt.subplot(211)
ax= sns.barplot(x_data_1.index, x_data_1.values, alpha=0.8,palette='mako')
plt.title("toxic comment in each column",fontsize=30)
plt.ylabel('number of Occurrences', fontsize=23,labelpad=20)
plt.xlabel('Type of the toxic comment', fontsize=23,labelpad=20)

plt.show()

#plot the number of Occurrences for the languages in validation data
plt.figure(1,figsize=(10,6))
sns.countplot(validation_data.lang,alpha=0.8,palette='mako')
plt.title("Appearance of languages in validation data ",fontsize=20)
plt.ylabel('number of Occurrences', fontsize=15,labelpad=20)
plt.xlabel('Languages', fontsize=15,labelpad=20)

plt.show()
#plot the number of Occurrences for the languages in validation data
plt.figure(1,figsize=(10,6))
sns.countplot(testing_data.lang,alpha=0.8,palette='mako')
plt.title("Appearance of languages in validation data ",fontsize=20)
plt.ylabel('number of Occurrences', fontsize=15,labelpad=20)
plt.xlabel('Languages', fontsize=15,labelpad=20)
plt.show()
plt.figure(1,figsize=(10,6))
sns.countplot(training_data.toxic,alpha=0.8,palette='mako')
plt.title("Appearance of languages in validation data ",fontsize=20)
plt.ylabel('number of Occurrences', fontsize=15,labelpad=20)
plt.xlabel('Languages', fontsize=15,labelpad=20)

plt.show()
#using the wordCloud to display the text on comment text column 
#first we will use stopword to remove unnessecery common word in english 
comment_words = '' 
stopwords = set(STOPWORDS) 
for word in training_data['comment_text']: 
      
    word = str(word) 
  
    # split the value 
    tokens = word.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
     #add token to a list 
    comment_words += " ".join(tokens)+" "
  
#define a wordcloud with the most common word in comment text column
wordcloud = WordCloud(background_color='black', collocations=False,
 width=1400, height=1200,stopwords=stopwords).generate(comment_words)
fig = px.imshow(wordcloud)
fig.update_layout(title_text='Common words in comment text column')
#define a figure for the four wordcloud plot
fig= plt.figure(figsize=(30,30))

ax = fig.add_subplot(221)
#choose the image as the mask for the plot
threat_mask = np.array(Image.open("../input/imageseda/bomb.jpg"))
# choose the threat comments only from comment_text column
subset=training_data[training_data.threat==1]
text=subset.comment_text.values

#plot the data with wordcloud 
wordcloud = WordCloud(mask=threat_mask,background_color='black',
 stopwords=stopwords).generate("".join(text))

plt.axis('off')
plt.imshow( wordcloud.recolor(colormap= 'Paired', random_state=24))
plt.title('Common threat words',fontsize=23)

############################
ax2 = fig.add_subplot(222)
#choose the image as the mask for the plot
insult_mask = np.array(Image.open("../input/imageseda/bullying.jpg"))
# choose the insult comments only from comment_text column
subset=training_data[training_data.insult==1]
text=subset.comment_text.values

#plot the data with wordcloud 
wordcloud = WordCloud(mask=insult_mask,background_color='black',
 stopwords=stopwords).generate("".join(text))
plt.axis('off')
plt.imshow( wordcloud.recolor(colormap= 'Paired', random_state=24))
plt.title('Common insult words',fontsize=23)

#########################
ax3 = fig.add_subplot(223)
#choose the image as the mask for the plot
toxic_mask = np.array(Image.open("../input/imageseda/spider3.jpg"))

# choose the toxic comment only from comment_text column
subset=training_data[training_data.toxic==1]
text=subset.comment_text.values

#plot the data with wordcloud 
wordcloud = WordCloud(mask=toxic_mask,background_color='black',
 stopwords=stopwords).generate("".join(text))
plt.axis('off')
plt.imshow( wordcloud.recolor(colormap= 'Paired', random_state=24))
plt.title('Common toxic words',fontsize=23)

########################
ax4 = fig.add_subplot(224)
#choose the image as the mask for the plot
attack_mask = np.array(Image.open("../input/imageseda/stop.jpg"))

# choose the identity comment only from comment_text column
subset=training_data[training_data.identity_hate==1]
text=subset.comment_text.values

#plot the data with wordcloud 
wordcloud = WordCloud(mask=attack_mask,background_color='black',
 stopwords=stopwords).generate("".join(text))

plt.axis('off')
plt.imshow( wordcloud.recolor(colormap= 'Paired', random_state=24))
plt.title('Common identity attack words',fontsize=23);
def remove_emoji(text):
   # decode the text from UTF-8 source format
    allchars = [str for str in text.decode('utf-8')]
    
    #define a list of emoji from the library"emoji"
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    
    #for each word in the text, if the word are not an emoji in the list, split and add the word in (clean_text)
    clean_text = ' '.join([str for str in text.decode('utf-8').split() if not any(i in str for i in emoji_list)])
    return clean_text
#the comment before removing emoji
training_data["comment_text"][3250]
text_no_emoji=[]
for text in training_data["comment_text"]:
    text_no_emoji.append(remove_emoji(text.encode('utf8')))

training_data["comment_text"]=text_no_emoji
#the comment after removing emoji
training_data["comment_text"][3250]

#comment before removing emoji
validation_data["comment_text"][197]
text_no_emoji_valid=[]
for text in validation_data["comment_text"]:
    text_no_emoji_valid.append(remove_emoji(text.encode('utf8')))
validation_data["comment_text"]=text_no_emoji_valid
#comment after removing emoji
validation_data["comment_text"][197]
#comment before removing emoji
testing_data["content"][48657]
text_no_emoji_test=[]
for text in testing_data["content"]:
    text_no_emoji_test.append(remove_emoji(text.encode('utf8')))
    
testing_data["content"]=text_no_emoji_test

#comment after removing emoji
testing_data["content"][48657]
def remove_punctuations(text):
    # define a list with repeated symbol in the dataset
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',
          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',
          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√',
          '\t','\n','❖','«','✉','❽','♪♫','☆','ψ']

    #in for loop replace each symbol with space
    for punctuation in puncts:
        text = text.replace(punctuation, ' ')
    return text
#the comment before removing punctuations
training_data["comment_text"] [2508]
training_data["comment_text"] = training_data['comment_text'].apply(remove_punctuations)
#the comment after removing punctuations
training_data["comment_text"] [2508]
#the comment before removing punctuations
validation_data["comment_text"] [361]
validation_data["comment_text"] = validation_data["comment_text"].apply(remove_punctuations)
#the comment after removing punctuations
validation_data["comment_text"][361]
#the comment before removing punctuations
testing_data["content"][76]
testing_data["content"] = testing_data["content"].apply(remove_punctuations)
#the comment after removing punctuations
testing_data["content"][76]
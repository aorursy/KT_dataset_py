import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt
train_df = pd.read_csv("../input/train.csv")
#test_df = pd.read_csv("ProjectData/test.csv")
train_df.head(5)
train_text = train_df['question_text']
type(train_text)
train_label = train_df['target']
train_label.value_counts()
from wordcloud import WordCloud
insincere_text = train_df[train_df.target ==1 ]['question_text']
insincere_text = " ".join(text for text in insincere_text)
insincere_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(insincere_text)
# Display the generated image:
plt.figure(figsize=(15,10))
plt.imshow(insincere_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
insincere_questions_corpus = train_df[train_df.target ==1 ]['question_text'].values.tolist()
insincere_questions_corpus[:11]
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer

def clean_text(insincere_words):
    #print(insincere_words)
    insincere_words_tokenized = word_tokenize(insincere_words)
    #insincere_words_tokenized = [i for i in insincere_words_tokenized if nltk.pos_tag ]
    insincere_words_cleaned = [i.lower() for i in insincere_words_tokenized]
    insincere_words_cleaned = [WordNetLemmatizer().lemmatize(i) for i in insincere_words_cleaned]
    insincere_words_cleaned = [i for i in insincere_words_cleaned if i not in string.punctuation]
    insincere_words_cleaned = [i for i in insincere_words_cleaned if i not in stopwords.words('english')]
    return(insincere_words_cleaned)
insincere_questions_cleaned =[clean_text(doc) for doc in insincere_questions_corpus]
insincere_questions_cleaned[:2]
insincere_questions_text = " ".join(str(i) for i in insincere_questions_cleaned)
insincere_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(insincere_questions_text)
# Display the generated image:
plt.figure(figsize=(15,10))
plt.imshow(insincere_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
dictionary = Dictionary(insincere_questions_cleaned)
len(dictionary)
insincere_bow = [dictionary.doc2bow(doc) for doc in insincere_questions_cleaned]
from gensim.models.ldamodel import LdaModel
lda_model = LdaModel(corpus = insincere_bow, num_topics=5, id2word=dictionary, passes = 10,random_state = 1)
lda_model.show_topics(num_topics= 5)
import pyLDAvis.gensim
lda_visualization = pyLDAvis.gensim.prepare(lda_model,insincere_bow,dictionary,sort_topics = False)
pyLDAvis.display(lda_visualization)
filename = '../input/en.yusufali.csv'

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_colwidth', 1000)
csv = pd.read_csv(filename, header=0,  delimiter=",", quotechar='"')
csv.head(7)
txt = ' '.join(csv.Text.tolist())
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('rocks')
lemmatizer.lemmatize('said',pos='v')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
sent = 'I gave her my flowers but she wanted my shower'

pos_tag(word_tokenize(sent), tagset='universal')
from nltk.tokenize import word_tokenize
def tokenize(text):

    tokens = word_tokenize(text)

    tokens_pos = pos_tag(tokens,tagset='universal')

    normal_forms = []

    for i in tokens_pos:

        if i[1] in ('NOUN','VERB','ADJ'):

            if i[1] == 'NOUN': normal_forms.append(lemmatizer.lemmatize(i[0]));

            if i[1] == 'VERB': normal_forms.append(lemmatizer.lemmatize(i[0],pos='v'));        

            if i[1] == 'ADJ': normal_forms.append(lemmatizer.lemmatize(i[0],pos='a'));                

    return normal_forms
tokenize(sent)
# stop words list from here: http://xpo6.com/list-of-english-stop-words/

stop_words = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"];
stop_words.extend(['ye','thou','shall','thee','thy','ha','i','o'])
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer(stop_words=stop_words,tokenizer=tokenize)

matrix_count = count_vec.fit_transform([txt]).toarray()
words = [i[0] for i in sorted(count_vec.vocabulary_.items(), key=lambda x: x[1])]
df_words = pd.DataFrame(data=matrix_count,columns=words)
df_words_t = df_words.transpose()
df_words_t.sort_values([0],ascending=False).head(10)
ngrams_vec = CountVectorizer(stop_words=stop_words, ngram_range=(2,2))

matrix_ngrams = ngrams_vec.fit_transform([txt]).toarray()
ngrams = [i[0] for i in sorted(ngrams_vec.vocabulary_.items(), key=lambda x: x[1])]

df_ngrams = pd.DataFrame(data=matrix_ngrams,columns=ngrams)

df_ngrams_t = df_ngrams.transpose()

df_ngrams_t.sort_values([0],ascending=False).head(10)
import nltk

quran_text = nltk.Text(word_tokenize(txt.lower()))
from nltk.book import *

print(text3)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer(tokenizer=tokenize, stop_words=stop_words)

def cosine_sim(txt1, txt2):

    tfidf = tfidf_vec.fit_transform([txt1, txt2])

    return ((tfidf * tfidf.T).A)[0,1]
i = 1

while i<=9:

    book = 'text' + str(i)

    exec('print(' + book + ')')

    exec("book_txt = ' '.join(" + book + ")")

    print(cosine_sim(txt,book_txt))

    i = i+1

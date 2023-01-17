from nltk.corpus import brown
brown.words() # Returns a list of strings
len(brown.words()) # No. of words in the corpus
brown.sents() # Returns a list of list of strings 
brown.sents(fileids='ca01') # You can access a specific file with `fileids` argument.
len(brown.fileids()) # 500 sources, each file is a source.
print(brown.fileids()[:100]) # First 100 sources.
print(brown.raw('cb01').strip()[:1000]) # First 1000 characters.
from nltk.corpus import webtext
webtext.fileids()
# Each line is one advertisement.
for i, line in enumerate(webtext.raw('overheard.txt').split('\n')):
    if i > 10: # Lets take a look at the first 10 ads.
        break
    print(str(i) + ':\t' + line)
single_no8 = webtext.raw('overheard.txt').split('\n')[8]
print(single_no8)
from nltk import sent_tokenize, word_tokenize
sent_tokenize(single_no8)
for sent in sent_tokenize(single_no8):
    print(word_tokenize(sent))
sent_tokenize(single_no8)
for sent in sent_tokenize(single_no8):
    # It's a little in efficient to loop through each word,
    # after but sometimes it helps to get better tokens.
    print([word.lower() for word in word_tokenize(sent)])
    # Alternatively:
    #print(list(map(str.lower, word_tokenize(sent))))
print(word_tokenize(single_no8))  # Treats the whole line as one document.
single_no9 = webtext.raw('singles.txt').split('\n')[9]
sent_tokenize(single_no9)
from nltk.corpus import stopwords

stopwords_en = stopwords.words('english')
print(stopwords_en)
# Treat the multiple sentences as one document (no need to sent_tokenize)
# Tokenize and lowercase
single_no8_tokenized_lowered = list(map(str.lower, word_tokenize(single_no8)))
print(single_no8_tokenized_lowered)
stopwords_en = set(stopwords.words('english')) # Set checking is faster in Python than list.

# List comprehension.
print([word for word in single_no8_tokenized_lowered if word not in stopwords_en])
from string import punctuation
# It's a string so we have to them into a set type
print('From string.punctuation:', type(punctuation), punctuation)
stopwords_en_withpunct = stopwords_en.union(set(punctuation))
print(stopwords_en_withpunct)
print([word for word in single_no8_tokenized_lowered if word not in stopwords_en_withpunct])
# Stopwords from stopwords-json
stopwords_json = {"en":["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"]}
stopwords_json_en = set(stopwords_json['en'])
stopwords_nltk_en = set(stopwords.words('english'))
stopwords_punct = set(punctuation)
# Combine the stopwords. Its a lot longer so I'm not printing it out...
stoplist_combined = set.union(stopwords_json_en, stopwords_nltk_en, stopwords_punct)

# Remove the stopwords from `single_no8`.
print('With combined stopwords:')
print([word for word in single_no8_tokenized_lowered if word not in stoplist_combined])
from nltk.stem import PorterStemmer
porter = PorterStemmer()

for word in ['walking', 'walks', 'walked']:
    print(porter.stem(word))
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

for word in ['walking', 'walks', 'walked']:
    print(wnl.lemmatize(word))
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' # if mapping isn't found, fall back to Noun.
    
# `pos_tag` takes the tokenized sentence as input, i.e. list of string,
# and returns a tuple of (word, tg), i.e. list of tuples of strings
# so we need to get the tag from the 2nd element.

walking_tagged = pos_tag(word_tokenize('He is walking to school'))
print(walking_tagged)
[wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) for word, tag in walking_tagged]
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' 
    
def lemmatize_sent(text): 
    # Text input is string, returns lowercased strings.
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(word_tokenize(text))]

lemmatize_sent('He is walking to school')
print('Original Single no. 8:')
print(single_no8, '\n')
print('Lemmatized and removed stopwords:')
print([word for word in lemmatize_sent(single_no8) 
       if word not in stoplist_combined
       and not word.isdigit() ])
def preprocess_text(text):
    # Input: str, i.e. document/sentence
    # Output: list(str) , i.e. list of lemmas
    return [word for word in lemmatize_sent(text) 
            if word not in stoplist_combined
            and not word.isdigit()]
from collections import Counter

sent1 = "The quick brown fox jumps over the lazy brown dog."
sent2 = "Mr brown jumps over the lazy fox."

# Lemmatize and remove stopwords
processed_sent1 = preprocess_text(sent1)
processed_sent2 = preprocess_text(sent2)
print('Processed sentence:')
print(processed_sent1)
print()
print('Word counts:')
print(Counter(processed_sent1))
print('Processed sentence:')
print(processed_sent2)
print()
print('Word counts:')
print(Counter(processed_sent2))
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer

sent1 = "The quick brown fox jumps over the lazy brown dog."
sent2 = "Mr brown jumps over the lazy fox."

with StringIO('\n'.join([sent1, sent2])) as fin:
    # Create the vectorizer
    count_vect = CountVectorizer()
    count_vect.fit_transform(fin)
# We can check the vocabulary in our vectorizer
# It's a dictionary where the words are the keys and 
# The values are the IDs given to each word. 
count_vect.vocabulary_
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer

sent1 = "The quick brown fox jumps over the lazy brown dog."
sent2 = "Mr brown jumps over the lazy fox."

with StringIO('\n'.join([sent1, sent2])) as fin:
    # Override the analyzer totally with our preprocess text
    count_vect = CountVectorizer(stop_words=stoplist_combined,
                                 tokenizer=word_tokenize)
    count_vect.fit_transform(fin)
count_vect.vocabulary_
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer

sent1 = "The quick brown fox jumps over the lazy brown dog."
sent2 = "Mr brown jumps over the lazy fox."

with StringIO('\n'.join([sent1, sent2])) as fin:
    # Override the analyzer totally with our preprocess text
    count_vect = CountVectorizer(analyzer=preprocess_text)
    count_vect.fit_transform(fin)
count_vect.vocabulary_ 
count_vect.transform([sent1, sent2])
from operator import itemgetter

# Print the words sorted by their index
words_sorted_by_index, _ = zip(*sorted(count_vect.vocabulary_.items(), key=itemgetter(1)))

print(preprocess_text(sent1))
print(preprocess_text(sent2))
print()
print('Vocab:', words_sorted_by_index)
print()
print('Matrix/Vectors:\n', count_vect.transform([sent1, sent2]).toarray())
import json

with open('../input/random-acts-of-pizza/train.json') as fin:
    trainjson = json.load(fin)
trainjson[0]
print('UID:\t', trainjson[0]['request_id'], '\n')
print('Title:\t', trainjson[0]['request_title'], '\n')
print('Text:\t', trainjson[0]['request_text_edit_aware'], '\n')
print('Tag:\t', trainjson[0]['requester_received_pizza'], end='\n')
import pandas as pd
df = pd.io.json.json_normalize(trainjson) # Pandas magic... 
df_train = df[['request_id', 'request_title', 
               'request_text_edit_aware', 
               'requester_received_pizza']]
df_train.head()
import json

with open('../input/random-acts-of-pizza/test.json') as fin:
    testjson = json.load(fin)
print('UID:\t', testjson[0]['request_id'], '\n')
print('Title:\t', testjson[0]['request_title'], '\n')
print('Text:\t', testjson[0]['request_text_edit_aware'], '\n')
print('Tag:\t', testjson[0]['requester_received_pizza'], end='\n')
import pandas as pd
df = pd.io.json.json_normalize(testjson) # Pandas magic... 
df_test = df[['request_id', 'request_title', 
               'request_text_edit_aware']]
df_test.head()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 

# It doesn't really matter what the function name is called
# but the `train_test_split` is splitting up the data into 
# 2 parts according to the `test_size` argument you've set.

# When we're splitting up the training data, we're spltting up 
# into train, valid split. The function name is just a name =)
train, valid = train_test_split(df_train, test_size=0.2)
# Initialize the vectorizer and 
# override the analyzer totally with the preprocess_text().
# Note: the vectorizer is just an 'empty' object now.
count_vect = CountVectorizer(analyzer=preprocess_text)

# When we use `CounterVectorizer.fit_transform`,
# we essentially create the dictionary and 
# vectorize our input text at the same time.
train_set = count_vect.fit_transform(train['request_text_edit_aware'])
train_tags = train['requester_received_pizza']

# When vectorizing the validation data, we use `CountVectorizer.transform()`.
valid_set = count_vect.transform(valid['request_text_edit_aware'])
valid_tags = valid['requester_received_pizza']
# When vectorizing the test data, we use `CountVectorizer.transform()`.
test_set = count_vect.transform(df_test['request_text_edit_aware'])
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB() 

# To train the classifier, simple do 
clf.fit(train_set, train_tags) 
from sklearn.metrics import accuracy_score

# To predict our tags (i.e. whether requesters get their pizza), 
# we feed the vectorized `test_set` to .predict()
predictions_valid = clf.predict(valid_set)

print('Pizza reception accuracy = {}'.format(
        accuracy_score(predictions_valid, valid_tags) * 100)
     )
count_vect = CountVectorizer(analyzer=preprocess_text)

full_train_set = count_vect.fit_transform(df_train['request_text_edit_aware'])
full_tags = df_train['requester_received_pizza']

# Note: We have to re-vectorize the test set since
#       now our vectorizer is different using the full 
#       training set.
test_set = count_vect.transform(df_test['request_text_edit_aware'])

# To train the classifier
clf = MultinomialNB() 
clf.fit(full_train_set, full_tags) 
# To predict our tags (i.e. whether requesters get their pizza), 
# we feed the vectorized `test_set` to .predict()
predictions = clf.predict(test_set)
success_rate = sum(df_train['requester_received_pizza']) / len(df_train) * 100
print(str('Of {} requests, only {} gets their pizzas,'
          ' {}% success rate...'.format(len(df_train), 
                                        sum(df_train['requester_received_pizza']), 
                                       success_rate)
         )
     )
success_rate = sum(predictions) / len(predictions) * 100
print(str('Of {} requests, only {} gets their pizzas,'
          ' {}% success rate...'.format(len(predictions), 
                                        sum(predictions), 
                                       success_rate)
         )
     )
df_sample_submission = pd.read_csv('../input/patching-pizzas/sampleSubmission.csv')
df_sample_submission.head()
# We've kept the `request_id` previous in the `df_test` dataframe.
# We can simply merge that column with our predictions.
df_output = pd.DataFrame({'request_id': list(df_test['request_id']), 
                          'requester_received_pizza': list(predictions)}
                        )
# Convert the predictions from boolean to integer.
df_output['requester_received_pizza'] = df_output['requester_received_pizza'].astype(int)
df_output.head()
# Create the csv file.
df_output.to_csv('basic-nlp-submission.csv')

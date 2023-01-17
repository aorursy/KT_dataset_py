import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

import seaborn as sns

warnings.filterwarnings('ignore')



dt=pd.read_csv("../input/Reviews.csv")

''' We only used the csv file to read and understand the daata '''

print(dt.head(2))

print("The shape of the data :",dt.shape)

print("The column names are :",dt.columns.values)
# data cleaning

import sqlite3 

con = sqlite3.connect('../input/database.sqlite') 
user_list=pd.read_sql_query(""" SELECT * FROM Reviews WHERE  Score != 3 LIMIT 5000""", con)

# we are using sql as it will be easy to limit the 5000 users using sql query

user_list.shape
# i checked for entire review data in the begining i got dense error while applying toDense function to the vectorized data 

#  so i am limiting the review data to 5K which is working fine



# we can determine the review is positive or not if score is 3

# print(user_list.columns.values)

# print(set(user_list))





sort_data=user_list.sort_values('ProductId', axis=0,kind='mergesort', inplace=False, ascending=True)

# The use of sort_values is mentioned here https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html



"""

I have observed that when i took the whole reviews

    3287690 no of rows have came when subset is {'UserId', 'ProfileName', 'Time'}

    3632240 no of rows have came when subset is {'UserId', 'ProfileName', 'Time', 'Summary'}

    so there may be scenario in which 2 comments getting update by same user at same time so taking 4 attributes will make it unique

    2 comments by same user can get updated at same time may be due to multiple devices or network issue

"""

# case 1 which i tried earlier and observed the above issue

# Eliminating the duplicate data points based on: 'UserId', 'ProfileName', 'Time'

# sort_data1=user_list.drop_duplicates(subset={'UserId', 'ProfileName', 'Time'}, keep='first', inplace=False)

# data1 = sort_data1[sort_data1['HelpfulnessDenominator'] >= sort_data1['HelpfulnessNumerator']]



# case 2 which we are using now

# Eliminating the duplicate data points based on: 'UserId', 'ProfileName', 'Time', 'Summary'

sort_data.drop_duplicates(subset=['UserId', 'ProfileName', 'Time', 'Summary'], keep='first', inplace=True)

## There are some users 'SELECT * FROM Reviews WHERE ProductId=7310172001   UserId=AJD41FBJD9010 Time=1233360000'

##  which has same summary so taking summary also unique





# kepping inplace=True as it will save memory instead of holding duplicate values seperately in other variable



data = sort_data[sort_data['HelpfulnessDenominator'] >= sort_data['HelpfulnessNumerator']]

# as HelpfulnessDenominator should cannot be less than HelpfulnessNumerator



print("The size which remained after deduplication is : ")



print(data.shape)

#print(data1.size)  here data1 is used to understand the data when we used subset parameter as 'UserId', 'ProfileName', 'Time'

# sort_data.merge(sort_data1,indicator = True, how='left').loc[lambda x : x['_merge']!='both'] 

print(data[:5])

#just checking the random text reviews to understand the data format

ar=[2500,300,2342,0,1000]

print("Checking the random texts to understand and applying the above mentioned cleaning techniques")

for i in ar:

    print(i)

    print(data["Text"].values[i])

    print("="*50)

    
import re



def removeHtml(text):

    cleanTxt=re.sub(re.compile('<.*>'),' ',text)

    return cleanTxt



# contractions words are taken from https://stackoverflow.com/a/47091490/4084039

contractions = {"ain't": "am not / are not / is not / has not / have not","aren't": "are not / am not","can't": "cannot","can't've": "cannot have","'cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have","didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not","haven't": "have not","he'd": "he had / he would","he'd've": "he would have","he'll": "he shall / he will","he'll've": "he shall have / he will have","he's": "he has / he is","how'd": "how did","how'd'y": "how do you","how'll": "how will","how's": "how has / how is / how does","I'd": "I had / I would","I'd've": "I would have","I'll": "I shall / I will","I'll've": "I shall have / I will have","I'm": "I am","I've": "I have","isn't": "is not","it'd": "it had / it would","it'd've": "it would have","it'll": "it shall / it will","it'll've": "it shall have / it will have","it's": "it has / it is","let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not","mightn't've": "might not have","must've": "must have","mustn't": "must not","mustn't've": "must not have","needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not","shan't've": "shall not have","she'd": "she had / she would","she'd've": "she would have","she'll": "she shall / she will","she'll've": "she shall have / she will have","she's": "she has / she is","should've": "should have","shouldn't": "should not","shouldn't've": "should not have","so've": "so have","so's": "so as / so is","that'd": "that would / that had","that'd've": "that would have","that's": "that has / that is","there'd": "there had / there would","there'd've": "there would have","there's": "there has / there is","they'd": "they had / they would","they'd've": "they would have","they'll": "they shall / they will","they'll've": "they shall have / they will have","they're": "they are","they've": "they have","to've": "to have","wasn't": "was not","we'd": "we had / we would","we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have","weren't": "were not","what'll": "what shall / what will","what'll've": "what shall have / what will have","what're": "what are","what's": "what has / what is","what've": "what have","when's": "when has / when is","when've": "when have","where'd": "where did","where's": "where has / where is","where've": "where have","who'll": "who shall / who will","who'll've": "who shall have / who will have","who's": "who has / who is","who've": "who have","why's": "why has / why is","why've": "why have","will've": "will have","won't": "will not","won't've": "will not have","would've": "would have","wouldn't": "would not","wouldn't've": "would not have","y'all": "you all","y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you had / you would","you'd've": "you would have","you'll": "you shall / you will","you'll've": "you shall have / you will have","you're": "you are","you've": "you have"}



def decontracted(text):

    temp_txt=""

    for ele in text.split(" "):

        if ele in contractions:

            temp_txt = temp_txt+ " "+ contractions[ele].split("/")[0] # we are taking the only first value before the / so to avoid duplicate words

        else:

            temp_txt = temp_txt+ " " +ele

    return  temp_txt



stopwords=["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]

# removed the words "no","nor","not",

# the words like "don't","aren't" are there in the list as the text is decontracted before



from tqdm import tqdm

cleaned_reviews = []

# tqdm is for printing the status bar

i=0

for txt in tqdm(data['Text'].values):

    txt = removeHtml( re.sub(r"http\S+", " ", txt)) # remove all the <html > tags and http links (step 1)

    

    txt = re.sub(r'[?|.|!|*|@|#|"|,|)|(|\|/]', r'', txt) # removing punctuations (step 2)

    txt = re.sub('[^A-Za-z]+', ' ', txt)  # checking the alphanumeric characters (step 3)

    txt = re.sub("\S*\d\S*", " ", txt).strip() # removing numeric characters 

    txt = decontracted(txt)  # to remove the contacted words (step 6)

   

    # https://gist.github.com/sebleier/554280

    txt = ' '.join(e.lower() for e in txt.split() if e.lower() and len(e)>2 not in stopwords) 

    txt = ' '.join(e for e in txt.split() if e!=(len(e) *e[0])  not in stopwords) 

    

    # to check characters like 'a' 'aaaa' 'bbbbb' 'hhhhhhhhhh' 'mmmmmmm' which doesn't make sense

    

    # (step 4) and  (step 6) checking if length is less than 2 and converting to lower case

    

    cleaned_reviews.append(txt.strip())

    

print(data['Text'].values[:2])    

print("#"*50+"to compare the changes\n\n")

print(cleaned_reviews[:2])    

  
# Making the seperation of positive reviews in seperate columns

def sep(score):

    if score<3:

        return 'Positive'

    else :

        return 'Negative'

Type_review = data['Score']

Type_review = Type_review.map(sep)

data.loc[:,'Type'] = Type_review

print(data["Type"][:6])
from sklearn.feature_extraction.text import CountVectorizer



vectorizer  = CountVectorizer() #in scikit-learn

'''

vectorizer.fit(cleaned_reviews)

# fit - Learn a vocabulary dictionary of all tokens in the raw documents.

Bag_of_words = vectorizer.transform(cleaned_reviews) 

# Here cleaned_reviews is our data corpus

'''

 # this line replaces the above 2 lines 

Bag_of_words = vectorizer.fit_transform(cleaned_reviews)



print("Shape of the Bag of words formed :",Bag_of_words.get_shape())

print(vectorizer.get_feature_names()[:10])

# Bag of words are formed in the  above code

# as we see the first 10 attributes some doesn't make sense But there occurence is very low and are negligible

print("Bag of words data type :",type(Bag_of_words))

print("Cleaned reviews data type :",type(cleaned_reviews))

Bag_of_words_dense = Bag_of_words.todense()

# converting sparse matrix to dense as we need dense matrix for standardising



from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE



Bag_of_words_Standardised= StandardScaler().fit_transform(Bag_of_words_dense)

'''

 perplexity 2 ,iterations 250  ---  the iteration is very close

 perplexity 50 ,iterations 250

 perplexity 50 ,iterations 5000

 perplexity 150 ,iterations 5000

 perplexity 400 ,iterations 5000



'''

ar=[(2,250),(50,250),(50,5000),(150,5000),(400,5000)]

i1=1



for i in tqdm(ar):

    i1=i1+1

    plt.subplot(330+i1)

    Bag_of_words_model = TSNE(n_components=2,perplexity=i[0],random_state = 0,n_iter=i[1])

    # n_iter value should be atleast 250

    Bag_of_words_data = Bag_of_words_model.fit_transform(Bag_of_words_Standardised)

    

    Bag_of_words_vstackdata = np.vstack((Bag_of_words_data.T,data['Type'])).T

    

    Bag_of_words_plotData  = pd.DataFrame(Bag_of_words_vstackdata,columns=('1st Higher Dimension','2nd Higher Dimension','Category'))

    sns.FacetGrid(Bag_of_words_plotData,hue='Category',size=8).map(plt.scatter,'1st Higher Dimension','2nd Higher Dimension').add_legend()

    

    plt.title('Bag of words n_iter='+str(i[1])+' Perplexity = '+str(i[0]))

    plt.show()



'''

This block contains generation of Bi-grams and N-grams

'''



vectorizer  = CountVectorizer(ngram_range=(1,2),max_features=5000, min_df=10) #in scikit-learn



# ngram_range --- 2 parameters min and maximum



# min_df ---When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature.



# build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.



Bi_grams = vectorizer.fit_transform(cleaned_reviews)



print("Shape of the Bag of words formed :",Bi_grams.get_shape())

print(vectorizer.get_feature_names()[:10])

'''

 perplexity 50 ,iterations 5000

 perplexity 150 ,iterations 5000

 perplexity 400 ,iterations 5000



'''

ar=[(50,5000),(150,5000),(400,5000)]

i1=1



for i in tqdm(ar):

    i1=i1+1

    plt.subplot(330+i1)

    Bi_gram_model = TSNE(n_components=2,perplexity=i[0],random_state = 0,n_iter=i[1])

    # n_iter value should be atleast 250

    Bi_gram_data = Bi_gram_model.fit_transform(Bag_of_words_Standardised)

    

    Bi_gram_vstackdata = np.vstack((Bi_gram_data.T,data['Type'])).T

    

    Bi_gram_plotData  = pd.DataFrame(Bi_gram_vstackdata,columns=('1st Higher Dimension','2nd Higher Dimension','Category'))

    sns.FacetGrid(Bi_gram_plotData,hue='Category',size=8).map(plt.scatter,'1st Higher Dimension','2nd Higher Dimension').add_legend()

    

    plt.title('Bi grams n_iter='+str(i[1])+' Perplexity = '+str(i[0]))

    plt.show()

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

'''

This block contains generation of Tf-Idf data

'''

tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=10)

# # ngram_range --- 2 parameters min and maximum

tf_idf_data = tf_idf_vectorizer.fit_transform(cleaned_reviews)



print(tf_idf_vectorizer.get_feature_names()[:10])

print('='*50)

print("data type of  data : ",type(tf_idf_data))

print('='*50)

print("size of data :",tf_idf_data.get_shape())



'''

 perplexity 50 ,iterations 5000

 perplexity 150 ,iterations 5000

 perplexity 400 ,iterations 5000



'''

from sklearn.preprocessing import StandardScaler





i1=1

tf_idf_densedata =tf_idf_data.todense()

TfIdf_standardized_data = StandardScaler().fit_transform(tf_idf_densedata)

#print(TfIdf_standardized_data[:10])

ar=[(30,5000),(50,5000),(150,5000),(700,5000)]

#ar=[]

for i in tqdm(ar):

    i1=i1+1

    plt.subplot(330+i1)

    tf_idf_model = TSNE(n_components=2,perplexity=i[0],random_state = 0,n_iter=i[1])

    # n_iter value should be atleast 250

    

    tf_idf_fit_data = tf_idf_model.fit_transform(TfIdf_standardized_data)

    

    tf_idf_vstackdata = np.vstack((tf_idf_fit_data.T,data['Type'])).T

    

    tf_idf_plotData  = pd.DataFrame(tf_idf_vstackdata,columns=('1st Higher Dimension','2nd Higher Dimension','Category'))

    sns.FacetGrid(tf_idf_plotData,hue='Category',size=6).map(plt.scatter,'1st Higher Dimension','2nd Higher Dimension').add_legend()

    

    plt.title('tf-idf n_iter='+str(i[1])+' Perplexity = '+str(i[0]))

    plt.grid()

    plt.show()

print(TfIdf_standardized_data.shape)
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer



from gensim.models import Word2Vec

from gensim.models import KeyedVectors



i=0

list_of_words=[]

for sentance in cleaned_reviews:

    list_of_words.append(sentance.split())



w2v_model=Word2Vec(list_of_words,min_count=5,size=50, workers=4)

# training the w2v model

print("Data type of word 2 vec model : ",type(w2v_model))


''' here w2v_model.wv.vocab contains all the vocabulary words in the reviews '''

print("number of words in vocabulary :",len(w2v_model.wv.vocab),"\n")

print(w2v_model.wv.most_similar('taste'),"\n")

print('='*50,"\n")

print(w2v_model.wv.most_similar('yummy'),"\n")



w2v_words = list(w2v_model.wv.vocab)
''' Construction of Avg Word 2 vec from built word 2 vec '''



sentence_vectors=[]

for sent in list_of_words:

    sent_vec_var = np.zeros(50)

    count_word =0

    for word in sent :

        if word in w2v_words:

            vector = w2v_model.wv[word]

            sent_vec_var += vector

            count_word += 1

    if count_word !=0:

        sent_vec_var /= count_word

    sentence_vectors.append(sent_vec_var)    

print("The size of sentence : ",len(sentence_vectors))        



from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE



avgw2v_standardized = StandardScaler().fit_transform(sentence_vectors)



i1=1



ar=[(30,5000),(50,5000),(700,5000)]

#ar=[]

for i in tqdm(ar):

    i1=i1+1

    plt.subplot(330+i1)

    avgw2v_model = TSNE(n_components=2,perplexity=i[0],n_iter=i[1])

    

    avgw2v_fit_data = avgw2v_model.fit_transform(avgw2v_standardized)

    

    avgw2v_vstackdata = np.vstack((avgw2v_fit_data.T,data['Type'])).T

    

    avgw2v_plotData  = pd.DataFrame(avgw2v_vstackdata,columns=('1st Higher Dimension','2nd Higher Dimension','Category'))

    sns.FacetGrid(avgw2v_plotData,hue='Category',size=8).map(plt.scatter,'1st Higher Dimension','2nd Higher Dimension').add_legend()

    

    plt.title('tf-idf n_iter='+str(i[1])+' Perplexity = '+str(i[0]))

    plt.grid()

    plt.show()



tf_idf_vectorizer_model = TfidfVectorizer()

tf_idf_vectorizer_model.fit(cleaned_reviews)





dictionary = dict(zip(tf_idf_vectorizer_model.get_feature_names(), list(tf_idf_vectorizer_model.idf_)))

# we are converting a dictionary with word as a key, and the idf as a value

tfidf_features = tf_idf_vectorizer_model.get_feature_names()

# tfidf words/col-names

tfidf_sentence_vectors = [];



# the tfidf-w2v for each sentence/review is stored in this list

row=0;

# tf_idf_data is constructed already in TFIDF

for sent in list_of_words:

    sent_vec = np.zeros(50)

    weight_sum =0

    for word in sent:

        if word in tfidf_features and word in w2v_words:

            vector = w2v_model.wv[word]

            # to reduce the computation we are 

            # dictionary[word] = idf value of word in whole courpus

            # sent.count(word) = tf valeus of word in this review

            tf_idf = dictionary[word]*(sent.count(word)/len(sent))

            sent_vec += (vector * tf_idf)

            weight_sum += tf_idf

    if weight_sum != 0:

        sent_vec /= weight_sum

    tfidf_sentence_vectors.append(sent_vec)

    row += 1    

print(len(tfidf_sentence_vectors))            

print(tfidf_sentence_vectors[0])            


Tfidf_w2v_standardized = StandardScaler().fit_transform(tfidf_sentence_vectors)



i1=1



ar=[(30,5000),(50,5000),(700,5000)]

#ar=[]

for i in tqdm(ar):

    i1=i1+1

    plt.subplot(330+i1)

    Tfidf_w2v_model = TSNE(n_components=2,perplexity=i[0],n_iter=i[1])

    

    Tfidf_w2v_fit_data = Tfidf_w2v_model.fit_transform(avgw2v_standardized)

    

    Tfidf_w2v_vstackdata = np.vstack((Tfidf_w2v_fit_data.T,data['Type'])).T

    

    Tfidf_w2v_plotData  = pd.DataFrame(Tfidf_w2v_vstackdata,columns=('1st Higher Dimension','2nd Higher Dimension','Category'))

    sns.FacetGrid(Tfidf_w2v_plotData,hue='Category',size=8).map(plt.scatter,'1st Higher Dimension','2nd Higher Dimension').add_legend()

    

    plt.title('tf-idf n_iter='+str(i[1])+' Perplexity = '+str(i[0]))

    plt.grid()

    plt.show()

    
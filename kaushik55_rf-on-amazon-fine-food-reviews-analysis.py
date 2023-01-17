import pandas as pd

import numpy as np

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import warnings

import seaborn as sns

warnings.filterwarnings('ignore')



dt=pd.read_csv("../input/Reviews.csv")

''' We only used the csv file to read and understand the daata '''
print("The shape of the data :",dt.shape)

print("The column names are :",dt.columns.values)
# data cleaning

import sqlite3 

con = sqlite3.connect('../input/database.sqlite') 
user_list=pd.read_sql_query(""" SELECT * FROM Reviews WHERE  Score != 3 """, con)

# we are using sql as it will be easy to limit the 5000 users using sql query

user_list.shape
sort_data=user_list.sort_values('ProductId', axis=0,kind='mergesort', inplace=False, ascending=True)

# The use of sort_values is mentioned here https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html



"""

I have observed that when i took the whole reviews

    3287690 no of rows have came when subset is {'UserId', 'ProfileName', 'Time'}

    3632240 no of rows have came when subset is {'UserId', 'ProfileName', 'Time', 'Summary'}



so there may be scenario in which 2 comments getting update by same user at same time so taking 4 attributes will make it unique

    as 2 comments by same user can get updated at same time may be due to multiple devices or network issue

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

print(data[:2])

#just checking the random text reviews to understand the data format

ar=[2500,300,2342,0,1000]

print("Checking the random texts to understand and applying the above mentioned cleaning techniques")

for i in ar:

    print(i)

    print(data["Text"].values[i])

    print("="*50)

    
import re

import progressbar

progress = progressbar.ProgressBar()

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

for txt in progress(data['Text'].values):

    txt = removeHtml( re.sub(r"http\S+", " ", txt)) # remove all the <html > tags and http links (step 1)

    

    txt = re.sub(r'[?|.|!|*|@|#|"|,|)|(|\|/]', r' ', txt) # removing punctuations (step 2)

    txt = re.sub('[^A-Za-z]+', ' ', txt)  # checking the alphanumeric characters (step 3)

    txt = re.sub("\S*\d\S*", " ", txt).strip() # removing numeric characters 

    txt = decontracted(txt)  # to remove the contacted words (step 6)

   

    # https://gist.github.com/sebleier/554280

    txt = ' '.join(e.lower() for e in txt.split() if e.lower() and len(e)>2 not in stopwords) 

    txt = ' '.join(e for e in txt.split() if e!=(len(e) *e[0])  not in stopwords) 

    

    # the above line is to check characters like 'a' 'aaaa' 'bbbbb' 'hhhhhhhhhh' 'mmmmmmm' which doesn't make sense    

    # (step 4) and  (step 6) checking if length is less than 2 and converting to lower case

    

    cleaned_reviews.append(txt.strip())

    

print(data['Text'].values[:2])    

data['cleaned_reviews']=cleaned_reviews

print('\n')

print("#"*50+" to compare the changes\n\n")



print(cleaned_reviews[:2])    

  
# Making the seperation of positive reviews in seperate columns

def sep(score):

    if score<3:

        return 1

    else :

        return 0

Type_review = data['Score']



# 0 negative 1 positive

data.loc[:,'Type'] = Type_review.map(sep)



print(data["Type"].value_counts(),"\n"*2)

print("score counts:")

print(data["Score"].value_counts(),"\n"*2)

print("Shape of data after deduplication and cleaning = ",data.shape)

print("To see the column names : ",data.columns)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

def settingTrainingTestingData(val):

    global X,Y,X_train,X_test,y_train,y_test,data,X_cv,y_cv

    

    # here input param is amount of data we are working on it will be 50k for brute force and 20k for kdtree

    # and 50k  means 50k negative review data and 50k positive review data to make it balanced

    

    positive_sample_data=data[data['Type']==1].sample(n=val)

    negative_sample_data=data[data['Type']==0].sample(n=val)

    # 0 negative 1 positive

    knn_Data= pd.concat([negative_sample_data,positive_sample_data])

    knn_Data=knn_Data.sort_values(by='Time')



    # as time based sorting yeileds better result in this dataset

    

    X=knn_Data['cleaned_reviews']

    print("Shape of Cleaned reviews in data : ",X.shape)

    Y=knn_Data['Type']

    print("Shape of Type in data : ",Y.shape)



    #splitting the data into train and test

    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42)



    # there is a inbuilt function in scikit to calculate cross value score so we are not splitting those values cross_val_score

    #print("X_train.shape:",X_train.shape,"y_train.shape:",y_train.shape,"y_test.shape:",y_test.shape,"X_test.shape:",X_test.shape)



    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, shuffle=Flase)# this is for time series split

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33) # this is random splitting

    X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.33) # this is random splitting





    print("X_train.shape :",X_train.shape,"\ty_train.shape", y_train.shape)

    print('X_cv.shape :',X_cv.shape, '\ty_cv.shape :',y_cv.shape)

    print('X_test.shape :',X_test.shape,'\ty_test.shape :', y_test.shape)
from scipy.sparse import csr_matrix

def convertingToDenseMatrix():

    # this function converts the X_train ,X-test data to dense which is used when the algo is kdtree

    global X,Y,X_train,X_test,y_train,y_test,data,X_cv,y_cv

    

    X_train =X_train.todense()

    X_test =X_test.todense()

    X_cv =X_cv.todense()

    #y_train =y_train.todense()

    #y_test =y_test.todense()
'''

As the given data is imbalanced data set we can get high accuracy for the invalid model as well

so making it balanced data set by using under sampling

we have only 57016 positive reviews so we can make at max 104k points with undersampling



So feeding 100k datapoints for brute force and 40k datapoints for kd-tree algorithm

'''

# declaring constants which we use for each vectorization

bruteForceSize=5000

kdtreeSize=2000

df = pd.DataFrame()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')



import progressbar

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix



def findKFromData(algoType): 

    ''' This function finds the best k and plots the confusion matrix with best K'''

    train_data_AUC_ar=[]

    cv_data_AUC_ar=[]

    global X,Y,X_train,X_test,y_train,y_test,data,X_cv,y_cv

    

    K = list(range(3,50,2))  # range of k

    bestVal = 1

    bestK=-1

    progress = progressbar.ProgressBar()

    

    for i in progress(K):

        neigh = KNeighborsClassifier(n_neighbors=i,algorithm=algoType)

        # algorithm=kd_tree or brute

        

        neigh.fit(X_train, y_train)   #fitting the data with train data



        y_pred_proba = neigh.predict_proba(X_train)[:,1]   # predicting the score with train data

        y_cv_proba = neigh.predict_proba(X_cv)[:,1]  # predicting the score with cv data



        #fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)



        train_data_AUC =roc_auc_score(y_train,y_pred_proba) # computing the roc with train data

        cv_data_AUC =roc_auc_score(y_cv,y_cv_proba) # computing the roc with cv data



        train_data_AUC_ar.append(train_data_AUC) 

        cv_data_AUC_ar.append(cv_data_AUC)



        if bestVal > train_data_AUC - cv_data_AUC :

            bestVal = train_data_AUC - cv_data_AUC

            bestK = i   #getting the best k my comparing the minimum difference of train and cv data

    

    plt.grid()

    plt.plot(train_data_AUC_ar,K, label='train-AUC',color='blue')

    plt.plot(cv_data_AUC_ar,K, label='train-AUC',color='red')

    plt.legend()

    plt.xlabel('AUC')

    plt.ylabel('K')

    plt.title('AUC vs K plot for train and cv data')

    plt.show()

    

    print("The values of AUC for training data :")

    print([ float('{:.3f}'.format(train_data_AUC)[:-1]) for train_data_AUC in  train_data_AUC_ar ])



    print("The values of AUC for cv data :")

    print([ float('{:.3f}'.format(train_data_AUC)[:-1]) for train_data_AUC in  cv_data_AUC_ar ])



    print("\nBest k :",bestK)

    print("Minimum difference between train data and cv data is :",bestVal)

    

    # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

    plottingTrainAndTestData(bestK,algoType)

    # returning the result for each model and displayed in conclusion

    return [{"best K":bestK,"best value ":bestVal,"Algorithm ":algoType}]
from scipy.sparse import hstack

from scipy.sparse import vstack

def plottingTrainAndTestData(bestK,algoType):    

    ''' plots the confusion matrix with best K '''

    global X,Y,X_train,X_test,y_train,y_test,data,X_cv,y_cv

    

    X_train=vstack((X_train, X_cv)) # as xtrain is sparse matrix 

    y_train=pd.concat([y_train,y_cv]) # as ytrain is series

    

    neigh = KNeighborsClassifier(n_neighbors=bestK,algorithm=algoType)  # knn classifier    

    neigh.fit(X_train, y_train)   #fitting the data with train data

    

    '''

    y_train_proba=[]

    for I in range(0,  X_train.shape[0], 1000):

          y_train_proba.extend(neigh.predict(X_train[i : i+1000,:])[:,1])

            

    y_test_proba=[]

    for I in range(0, X_test.shape[0], 1000):

          y_test_proba.extend(neigh.predict(X_test[i : i+1000,:])[:,1])

    '''

            

    y_train_proba = neigh.predict_proba(X_train)[:,1]   # predicting the score with train data

    y_test_proba = neigh.predict_proba(X_test)[:,1]      # predicting the score with test data



    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

    plt.plot([0,1],[0,1],'k--')

    

    plt.plot(fpr,tpr, label='test-Knn',color='red')

    

    fpr, tpr, thresholds = roc_curve(y_train, y_train_proba)

    plt.plot(fpr,tpr, label='train-Knn',color='blue')

    

    plt.legend()

    plt.xlabel('fpr')

    plt.ylabel('tpr')

    plt.title('Knn(n_neighbors='+str(bestK)+') ROC curve')

    plt.show()

    

    # drawing confusion matrix for test and train data    

    drawConfusionMatrix(neigh,X_train,y_train,'train')

    drawConfusionMatrix(neigh,X_test,y_test)
def drawConfusionMatrix(neigh,X_conf_test,y_conf_test,type='test'):

    # predict the response

    pred = neigh.predict(X_conf_test)



    cm = confusion_matrix(y_conf_test, pred)



    class_label = ["negative", "positive"]

    df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)

    sns.heatmap(df_cm, annot = True, fmt = "d")

    

    plt.title("Confusion Matrix for test data")

    if type is 'train':

        plt.title("Confusion Matrix for train data")

    

    plt.xlabel("Predicted Label")

    plt.ylabel("True Label")

    plt.show()
settingTrainingTestingData(bruteForceSize)

# as time based sorting yeileds better result in this dataset
Bag_of_words_vect=CountVectorizer()

Bag_of_words_vect.fit(X_train)



X_train=Bag_of_words_vect.transform(X_train)



X_cv = Bag_of_words_vect.transform(X_cv)



X_test = Bag_of_words_vect.transform(X_test)

#X_test_bow = Bag_of_words_vect.transform(X_test)



temp=findKFromData('brute')

temp[0]["Model"]="Bag of words"

df=df.append(temp)

# pushing the result in table
# training our data with bag of words

settingTrainingTestingData(kdtreeSize)



Bag_of_words_vect=CountVectorizer(max_features=150)



Bag_of_words_vect.fit(X_train)

X_train=Bag_of_words_vect.transform(X_train)



X_cv = Bag_of_words_vect.transform(X_cv)



X_test = Bag_of_words_vect.transform(X_test)



#print(type(X_test))

#print(type(X_cv))

convertingToDenseMatrix() 



temp=findKFromData('kd_tree')

temp[0]["Model"]="Bag of words"

df=df.append(temp)
from sklearn.feature_extraction.text import TfidfVectorizer

settingTrainingTestingData(bruteForceSize) # used for brute force

tf_idf_vect=TfidfVectorizer()



tf_idf_vect.fit(X_train)



X_train=tf_idf_vect.transform(X_train)



X_cv = tf_idf_vect.transform(X_cv)



X_test = tf_idf_vect.transform(X_test)

#X_test_bow = Bag_of_words_vect.transform(X_test)



temp=findKFromData('brute')

temp[0]["Model"]="TFIDF"

df=df.append(temp)
settingTrainingTestingData(kdtreeSize) # used for kd tree

tf_idf_vect=TfidfVectorizer(max_features=150)

# here we are restricting the matrix to 500 features as the dense matrix is consuming the ram 



tf_idf_vect=TfidfVectorizer()



tf_idf_vect.fit(X_train)



X_train=tf_idf_vect.transform(X_train)



X_cv = tf_idf_vect.transform(X_cv)



X_test = tf_idf_vect.transform(X_test)

convertingToDenseMatrix() # 



temp=findKFromData('kd_tree')

temp[0]["Model"]="TFIDF"

df=df.append(temp)
from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

import gensim



def Transformfun(list_of_t):

    sent_vectors_cv = [] # the avg-w2v for each sentence/review is stored in this list

    for sent in list_of_t: # for each review/sentence

        sent_vec = np.zeros(50) # as word vectors are of zero length

        cnt_words =0; # num of words with a valid vector in the sentence/review

        for word in sent: # for each word in a review/sentence

            try:

                vec = w2v_model.wv[word]

                sent_vec += vec

                cnt_words += 1

            except:

                pass

        sent_vec /= cnt_words

        sent_vectors_cv.append(sent_vec)    

    #print("inside functions",sent_vectors_cv[:3])    

    return sent_vectors_cv
def creatingAVGW2V(X_gen):

    list_of_sent=[]    

    for sent in X_gen:  # here X_gen can be X_train ,X_cv or X_test

        filtered_sentence=[]

        for w in sent.split():

            for cleaned_words in w.split():

                if(cleaned_words.isalpha()):    

                    filtered_sentence.append(cleaned_words.lower())

                else:

                    continue 

        list_of_sent.append(filtered_sentence)

    return list_of_sent   
settingTrainingTestingData(bruteForceSize)



list_of_sent=creatingAVGW2V(X_train)



w2v_model=gensim.models.Word2Vec(list_of_sent,min_count=5,size=50, workers=4)

w2v = w2v_model[w2v_model.wv.vocab]



## fitting the w2v model with train data



list_of_sent_test = creatingAVGW2V(X_test) 

list_of_sent_cv = creatingAVGW2V(X_cv) 



'''**********************************************************************************'''

# Transformfun is transforming the data with above model

sent_vectors = Transformfun(list_of_sent)

sent_vectors_test = Transformfun(list_of_sent_test)

sent_vectors_cv=Transformfun(list_of_sent_cv)



'''**********************************************************************************'''

X_train = np.nan_to_num(sent_vectors)

X_test = np.nan_to_num(sent_vectors_test)

X_cv = np.nan_to_num(sent_vectors_cv)



temp=findKFromData('brute')

temp[0]["Model"]="Avg W2V"

df=df.append(temp)
settingTrainingTestingData(bruteForceSize)



list_of_sent=creatingAVGW2V(X_train)



w2v_model=gensim.models.Word2Vec(list_of_sent,min_count=5,size=50, workers=4)

w2v = w2v_model[w2v_model.wv.vocab]



list_of_sent_test = creatingAVGW2V(X_test) 

list_of_sent_cv = creatingAVGW2V(X_cv) 



'''**********************************************************************************'''

sent_vectors = Transformfun(list_of_sent)

sent_vectors_test = Transformfun(list_of_sent_test)

sent_vectors_cv=Transformfun(list_of_sent_cv)

'''**********************************************************************************'''

X_train = np.nan_to_num(sent_vectors)

X_test = np.nan_to_num(sent_vectors_test)

X_cv = np.nan_to_num(sent_vectors_cv)



temp=findKFromData('kd_tree')

temp[0]["Model"]="Avg W2V"

df=df.append(temp)
def getTfidfW2v(list_of_words,tfidf_features):

    tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list

    row=0;

    for sent in list_of_sent: # for each review/sentence

        sent_vec = np.zeros(50) # as word vectors are of zero length

        weight_sum =0; # num of words with a valid vector in the sentence/review

        for word in sent: # for each word in a review/sentence

            try:

                vec = w2v_model.wv[word]

                # obtain the tf_idfidf of a word in a sentence/review

                tfidf = final_tf_idf[row, tfidf_feat.index(word)]

                sent_vec += (vec * tf_idf)

                weight_sum += tf_idf

            except:

                pass

        sent_vec /= weight_sum

        tfidf_sent_vectors.append(sent_vec)

        row += 1          

    return tfidf_sent_vectors   
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

'''

This block contains generation of Tf-Idf data

'''



''' Construction of Avg Word 2 vec from built word 2 vec '''

settingTrainingTestingData(bruteForceSize)



tf_idf_vectorizer_model = TfidfVectorizer(max_features=150)

tf_idf_vectorizer_model.fit(X_train)



list_of_sent=creatingAVGW2V(X_train)



w2v_model=gensim.models.Word2Vec(list_of_sent,min_count=5,size=50, workers=4)

w2v = w2v_model[w2v_model.wv.vocab]



list_of_sent_test = creatingAVGW2V(X_test) 

list_of_sent_cv = creatingAVGW2V(X_cv) 



# TF-IDF weighted Word2Vec

tfidf_feat = tf_idf_vectorizer_model.get_feature_names() # tfidf words/col-names



# Transformfun is transforming the data with above model

X_test =getTfidfW2v(list_of_sent_test,tfidf_feat)

X_cv =getTfidfW2v(list_of_sent_cv,tfidf_feat)

X_train=getTfidfW2v(list_of_sent,tfidf_feat)



X_train = np.nan_to_num(sent_vectors)

X_test = np.nan_to_num(sent_vectors_test)

X_cv = np.nan_to_num(sent_vectors_cv)



temp=findKFromData('brute')

temp[0]["Model"]="TFIDF W2V"

df=df.append(temp)
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

'''

This block contains generation of Tf-Idf data

'''



''' Construction of Avg Word 2 vec from built word 2 vec '''

settingTrainingTestingData(bruteForceSize)



tf_idf_vectorizer_model = TfidfVectorizer(max_features=150)

tf_idf_vectorizer_model.fit(X_train)



list_of_sent=creatingAVGW2V(X_train)



w2v_model=gensim.models.Word2Vec(list_of_sent,min_count=5,size=50, workers=4)

w2v = w2v_model[w2v_model.wv.vocab]



list_of_sent_test = creatingAVGW2V(X_test) 

list_of_sent_cv = creatingAVGW2V(X_cv) 



# TF-IDF weighted Word2Vec

tfidf_feat = tf_idf_vectorizer_model.get_feature_names() # tfidf words/col-names



X_test =getTfidfW2v(list_of_sent_test,tfidf_feat)

X_cv =getTfidfW2v(list_of_sent_cv,tfidf_feat)

X_train=getTfidfW2v(list_of_sent,tfidf_feat)



X_train = np.nan_to_num(sent_vectors)

X_test = np.nan_to_num(sent_vectors_test)

X_cv = np.nan_to_num(sent_vectors_cv)



temp=findKFromData('kd_tree')

temp[0]["Model"]="TFIDF W2V"

df=df.append(temp)
# model

df

#df.sort_values(by='Model', ascending=False)
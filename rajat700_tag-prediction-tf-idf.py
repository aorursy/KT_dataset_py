import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn import preprocessing
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import ast
import seaborn as sns
%matplotlib inline
os.chdir('../input/transfer-learning-on-stack-exchange-tags')
os.listdir()
biology_data=pd.read_csv('biology.csv.zip')
cooking_data=pd.read_csv('cooking.csv.zip')
crypto_data=pd.read_csv('crypto.csv.zip')
diy_data=pd.read_csv('diy.csv.zip')
robotics_data=pd.read_csv('robotics.csv.zip')
travel_data=pd.read_csv('travel.csv.zip')
test_data=pd.read_csv('test.csv.zip')
test_data.head()
combined_data=pd.DataFrame()
combined_data=combined_data.append([biology_data,cooking_data,crypto_data,diy_data,robotics_data,travel_data])
combined_data.shape
combined_data.columns
combined_data.head()
combined_data.shape
def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)  
def remove_space(text):
    "Remove spaces from the text"
    s=text
    s=s.strip()
    return s
combined_data['title']=combined_data['title'].apply(lambda x: remove_html_tags(x))
combined_data['content']=combined_data['content'].apply(lambda x: remove_html_tags(x))
combined_data['title']=combined_data['title'].apply(lambda x: remove_space(x))
combined_data['content']=combined_data['content'].apply(lambda x: remove_space(x))

combined_data=combined_data.drop_duplicates(subset=['title'],)      #### Removing rows with duplicate titles

combined_data.shape
combined_data.head()
combined_data.reset_index(drop=True,inplace=True)
def lemmatization(tokens):
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")
    stemmed=[stemmer.stem(x) for x in tokens]
    return stemmed
    
def tokenize(text):
    from nltk.tokenize import sent_tokenize, word_tokenize 
    return word_tokenize(text)    
def remove_punctuation(tokens):
    words = [word for word in tokens if word.isalpha()]
    return words
def remove_stopwords(tokens):
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    extra_words = ['a', "a's", 'able', 'about', 'above', 'according', 'accordingly',
              'across', 'actually', 'after', 'afterwards', 'again', 'against',
              "ain't", 'all', 'allow', 'allows', 'almost', 'alone', 'along',
              'already', 'also', 'although', 'always', 'am', 'among', 'amongst',
              'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone',
              'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear',
              'appreciate', 'appropriate', 'are', "aren't", 'around', 'as',
              'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away',
              'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes',
              'becoming', 'been', 'before', 'beforehand', 'behind', 'being',
              'believe', 'below', 'beside', 'besides', 'best', 'better',
              'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', "c'mon",
              "c's", 'came', 'can', "can't", 'cannot', 'cant', 'cause',
              'causes', 'certain', 'certainly', 'changes', 'clearly', 'co',
              'com', 'come', 'comes', 'concerning', 'consequently', 'consider',
              'considering', 'contain', 'containing', 'contains',
              'corresponding', 'could', "couldn't", 'course', 'currently', 'd',
              'definitely', 'described', 'despite', 'did', "didn't",
              'different', 'do', 'does', "doesn't", 'doing', "don't", 'done',
              'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight',
              'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially',
              'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone',
              'everything', 'everywhere', 'ex', 'exactly', 'example', 'except',
              'f', 'far', 'few', 'fifth', 'first', 'five', 'followed',
              'following', 'follows', 'for', 'former', 'formerly', 'forth',
              'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets',
              'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got',
              'gotten', 'greetings', 'h', 'had', "hadn't", 'happens', 'hardly',
              'has', "hasn't", 'have', "haven't", 'having', 'he', "he's",
              'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter',
              'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him',
              'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit',
              'however', 'i', "i'd", "i'll", "i'm", "i've", 'ie', 'if',
              'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed',
              'indicate', 'indicated', 'indicates', 'inner', 'insofar',
              'instead', 'into', 'inward', 'is', "isn't", 'it', "it'd", "it'll",
              "it's", 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps',
              'kept', 'know', 'knows', 'known', 'l', 'last', 'lately', 'later',
              'latter', 'latterly', 'least', 'less', 'lest', 'let', "let's",
              'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks',
              'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean',
              'meanwhile', 'merely', 'might', 'more', 'moreover', 'most',
              'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely',
              'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither',
              'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody',
              'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing',
              'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often',
              'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only',
              'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our',
              'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own',
              'p', 'particular', 'particularly', 'per', 'perhaps', 'placed',
              'please', 'plus', 'possible', 'presumably', 'probably',
              'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're',
              'really', 'reasonably', 'regarding', 'regardless', 'regards',
              'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw',
              'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing',
              'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves',
              'sensible', 'sent', 'serious', 'seriously', 'seven', 'several',
              'shall', 'she', 'should', "shouldn't", 'since', 'six', 'so',
              'some', 'somebody', 'somehow', 'someone', 'something', 'sometime',
              'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry',
              'specified', 'specify', 'specifying', 'still', 'sub', 'such',
              'sup', 'sure', 't', "t's", 'take', 'taken', 'tell', 'tends', 'th',
              'than', 'thank', 'thanks', 'thanx', 'that', "that's", 'thats',
              'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence',
              'there', "there's", 'thereafter', 'thereby', 'therefore',
              'therein', 'theres', 'thereupon', 'these', 'they', "they'd",
              "they'll", "they're", "they've", 'think', 'third', 'this',
              'thorough', 'thoroughly', 'those', 'though', 'three', 'through',
              'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took',
              'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying',
              'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless',
              'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used',
              'useful', 'uses', 'using', 'usually', 'uucp', 'v', 'value',
              'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants',
              'was', "wasn't", 'way', 'we', "we'd", "we'll", "we're", "we've",
              'welcome', 'well', 'went', 'were', "weren't", 'what', "what's",
              'whatever', 'when', 'whence', 'whenever', 'where', "where's",
              'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon',
              'wherever', 'whether', 'which', 'while', 'whither', 'who',
              "who's", 'whoever', 'whole', 'whom', 'whose', 'why', 'will',
              'willing', 'wish', 'with', 'within', 'without', "won't", 'wonder',
              'would', 'would', "wouldn't", 'x', 'y', 'yes', 'yet', 'you',
              "you'd", "you'll", "you're", "you've", 'your', 'yours',
              'yourself', 'yourselves', 'z', 'zero', '','is','based','aa','aaa','aac','aad','aav','ab','aa','aa aa',
 'aa ab',
 'aa batteri',
 'aa lt',
 'aaa',
 'aabb',
 'aabb aabb',
 'aabbcc',
 'aac',
 'aad',
 'aasa',
 'aav',
 'ab']
    
    new_stop=stop_words + extra_words
    new_stop=list(set(new_stop))
    filtered_words=[word for word in tokens if word not in new_stop]
    return filtered_words
def lower_word(tokens):
    words = [word.lower() for word in tokens]
    return words

combined_data['title_words']=combined_data['title'].apply(lambda x: tokenize(x))
combined_data['content_words']=combined_data['content'].apply(lambda x: tokenize(x))
combined_data['title_words']=combined_data['title_words'].apply(lambda x: remove_punctuation(x))
combined_data['content_words']=combined_data['content_words'].apply(lambda x: remove_punctuation(x))

combined_data['title_words']=combined_data['title_words'].apply(lambda x: lower_word(x))
combined_data['content_words']=combined_data['content_words'].apply(lambda x: lower_word(x))
combined_data['title_words']=combined_data['title_words'].apply(lambda x: remove_stopwords(x))
combined_data['content_words']=combined_data['content_words'].apply(lambda x: remove_stopwords(x))
combined_data.reset_index(drop=True,inplace=True)
combined_data.head()
combined_data['text']=combined_data['title_words']+ combined_data['content_words']
combined_data.head()
#combined_data.loc[0,'title_words']
combined_data['text']=combined_data['text'].apply(lambda x: lemmatization(x))
combined_data.text.head()
combined_data['text']=combined_data['text'].apply(lambda x: ' '.join(x))
####################### Save Combined Data ############################################################
#combined_data.to_csv('combined_data_preprocessed.csv',index=False)
biology_data=biology_data.drop_duplicates(subset=['title'],) #### Removing rows with duplicate titles
travel_data=travel_data.drop_duplicates(subset=['title'],)      #### Removing rows with duplicate titles
cooking_data=cooking_data.drop_duplicates(subset=['title'],)      #### Removing rows with duplicate titles
robotics_data=robotics_data.drop_duplicates(subset=['title'],)      #### Removing rows with duplicate titles
diy_data=diy_data.drop_duplicates(subset=['title'],)      #### Removing rows with duplicate titles
crypto_data=crypto_data.drop_duplicates(subset=['title'],)      #### Removing rows with duplicate titles

datapoints=[]
datapoints.extend((biology_data.shape[0],travel_data.shape[0],cooking_data.shape[0],robotics_data.shape[0],diy_data.shape[0],
                  crypto_data.shape[0]))
topics=['biology','travel','cooking','robotics','diy','crypto']
topic_count=pd.DataFrame({'topics':topics,'datapoints':datapoints})
topic_count['percentage']=(topic_count['datapoints']/topic_count['datapoints'].sum())*100
topic_count.head()
sns.barplot(x=topic_count['topics'],y=topic_count['datapoints'])
sns.barplot(x=topic_count['topics'],y=topic_count['percentage'])
combined_data.head()
tags_count = combined_data["tags"].apply(lambda x: len(x.split(" "))) # counting the number of tags for each datapoint

combined_data['Tags_Count'] = tags_count

combined_data.head()
print("Maximum number of tags per question = "+str(max(combined_data['Tags_Count'])))
print("Minimum number of tags per question = "+str(min(combined_data['Tags_Count'])))
print("Avg number of tags per question = "+str(sum(combined_data['Tags_Count'])/len(combined_data['Tags_Count'])))
questions_per_tag=combined_data['Tags_Count'].value_counts()
questions_per_tag=pd.DataFrame(questions_per_tag)
questions_per_tag.reset_index(level=0,inplace=True)
questions_per_tag=questions_per_tag.rename(columns={'index':'tag_count','Tags_Count':'question_count'})
questions_per_tag['percentage']=(questions_per_tag['question_count']/questions_per_tag['question_count'].sum())*100
questions_per_tag.head()
sns.barplot(x=questions_per_tag['tag_count'],y=questions_per_tag['question_count'])
sns.barplot(x=questions_per_tag['tag_count'],y=questions_per_tag['percentage'])
combined_data['text_original']=combined_data['title_words'] + combined_data['content_words']
combined_data['tags_words']=combined_data['tags'].apply(lambda x: x.split(' '))
combined_data.drop(['text_original'],inplace=True,axis=1)
combined_data.head()
for i in range(0,len(combined_data)):
    #print(i)
    tag_words=combined_data.loc[i,'tags_words']
    title_words=combined_data.loc[i,'title_words']
    common_title_words=set(tag_words)&set(title_words)
    combined_data.loc[i,'title_overlap']=len(common_title_words)
for i in range(0,len(combined_data)):
    #print(i)
    tag_words=combined_data.loc[i,'tags_words']
    content_words=combined_data.loc[i,'content_words']
    common_content_words=set(tag_words)&set(content_words)
    combined_data.loc[i,'content_overlap']=len(common_content_words)
combined_data.head()
combined_data['title_overlap_percent']=(combined_data['title_overlap']/combined_data['Tags_Count'])*100
combined_data['content_overlap_percent']=(combined_data['content_overlap']/combined_data['Tags_Count'])*100
combined_data.head()
print("Average Title Overlap = {}".format(combined_data.title_overlap_percent.mean()))
print("Average Content Overlap = {}".format(combined_data.content_overlap_percent.mean()))

vectorizer = CountVectorizer(tokenizer = lambda x: x.split(" "))
tagcount = vectorizer.fit_transform(combined_data['tags'])
print("Total number of datapoints = {}".format(tagcount.shape[0]))
print("Total number of unique tags = {}".format(tagcount.shape[1]))
print(vectorizer.get_feature_names()[0:10])
#top 10 highest occurring tags
col_sum = tagcount.sum(axis = 0).A1 
feat_count = dict(zip(vectorizer.get_feature_names(), col_sum))
feat_count_sorted = dict(sorted(feat_count.items(), key = lambda x: x[1], reverse = True))
count_data = {"Tags":list(feat_count_sorted.keys()), "Count": list(feat_count_sorted.values())}
count_df = pd.DataFrame(data = count_data)
count_df[:10]
count_df['Percentage']=(count_df['Count']/count_df['Count'].sum())*100
count_df=count_df[:10]
count_df
plt.figure(figsize = (12, 4))
sns.barplot(x=count_df['Tags'],y=count_df['Count'])
plt.figure(figsize = (12, 4))
sns.barplot(x=count_df['Tags'],y=count_df['Percentage'])
vectorizer = CountVectorizer(tokenizer = lambda x: x.split(" "), binary = True)
labels = vectorizer.fit_transform(combined_data['tags'])
labels.shape
col_sum = labels.sum(axis = 0).A1   
col_sum
sorted_tags = np.argsort(-col_sum)  
sorted_tags
def top_n_tags(n):
    multilabel_yn = labels[:,sorted_tags[:n]] 
    return multilabel_yn

def questionsCovered(n):
    multilabel_yn = top_n_tags(n)
    NonZeroQuestions = multilabel_yn.sum(axis = 1)  
    return np.count_nonzero(NonZeroQuestions), NonZeroQuestions
questionsExplained = []
numberOfTags = []
for i in range(500,4268,500):
    questionsExplained.append(round((questionsCovered(i)[0]/labels.shape[0])*100,2))
    numberOfTags.append(i)
    
plt.figure(figsize = (16, 8))
plt.plot(numberOfTags, questionsExplained)
plt.title("Number of Tags VS Percentage of Questions Explained(%)", fontsize=20)
plt.xlabel("Number of Tags", fontsize=15)
plt.ylabel("Percentage of Questions Explained(%)", fontsize=15)
plt.scatter(x = numberOfTags, y = questionsExplained, c = "blue", s = 70)
for x, y in zip(numberOfTags, questionsExplained):
    plt.annotate(s = '({},{}%)'.format(x, y), xy = (x, y), fontweight='bold', fontsize = 12, xytext=(x+70, y-0.3), rotation = -20)
sumOfRows = questionsCovered(500)[1]
RowIndicesZero = np.where(sumOfRows == 0)[0]  #this contains indices of all the questions for which the tags are removed
data_new = combined_data.drop(labels = RowIndicesZero, axis = 0)
data_new.reset_index(drop = True, inplace = True)
print("Size of new data = ",data_new.shape[0])
#removing tags from data
data_tags = top_n_tags(500)
df = pd.DataFrame(data_tags.toarray())
TagsDF_new = df.drop(labels = RowIndicesZero, axis = 0)
TagsDF_new.reset_index(drop = True, inplace = True)
print("Size of new data = ",TagsDF_new.shape[0])
allTags = sparse.csr_matrix(TagsDF_new.values)
x_train, x_test, y_train, y_test = train_test_split(data_new, allTags, test_size=0.20, random_state=42)

print("Train data shape ",x_train.shape)
print("Train label shape", y_train.shape)
print("Test data shape ",x_test.shape)
print("Test label shape", y_test.shape)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range = (1,3), tokenizer = lambda x: x.split(" "))
TrainData = vectorizer.fit_transform(x_train['text'])
TestData = vectorizer.transform(x_test['text'])

#sparse.save_npz("FinalTrain.npz", TrainData)       ####### Saving Training and Test data in sparse format for later use  
#sparse.save_npz("FinalTest.npz", TestData)
#sparse.save_npz("FinalTrainLabels.npz", y_train)
#sparse.save_npz("FinalTestLabels.npz", y_test)
#FinalTrain = sparse.load_npz("FinalTrain.npz")      ####### Loading Training and Test Data for training
#FinalTest = sparse.load_npz("FinalTest.npz")
#FinalTrainLabels = sparse.load_npz("FinalTrainLabels.npz")
#FinalTestLabels = sparse.load_npz("FinalTestLabels.npz")

FinalTrain=TrainData
FinalTest=TestData
FinalTrainLabels=y_train
FinalTestLabels=y_test

print("Dimension of train data = ",TrainData.shape)
print("Dimension of test data = ",TestData.shape)
print("Dimension of train labels ",y_train.shape)
print("Dimension of Test labels ", y_test.shape)
classifier= OneVsRestClassifier(LogisticRegression(C=0.9,penalty='l1',solver='saga'), n_jobs=-1)
classifier.fit(FinalTrain, FinalTrainLabels)
predictions = classifier.predict(FinalTest)
prediction_train=classifier.predict(FinalTrain)
print("Train Accuracy :",accuracy_score(FinalTrainLabels,prediction_train))
print("Train Macro f1 score :",f1_score(FinalTrainLabels, prediction_train, average = 'macro'))
print("Train Micro f1 scoore :",f1_score(FinalTrainLabels, prediction_train, average = 'micro'))
print("Train Classification Report :\n",classification_report(FinalTrainLabels, prediction_train))

print("Validation Accuracy :",accuracy_score(FinalTestLabels,predictions))
print("Validation Macro f1 score :",f1_score(FinalTestLabels, predictions, average = 'macro'))
print("Validation Micro f1 scoore :",f1_score(FinalTestLabels, predictions, average = 'micro'))
print("Validation Classification Report :\n",classification_report(FinalTestLabels, predictions))

################## Save Model for later use #################################################
##filename = 'best_model_l1_saga_f1_0.47.sav'
#joblib.dump(classifier, filename)
classifier_1= OneVsRestClassifier(MultinomialNB(alpha=0.35), n_jobs=-1)
classifier_1.fit(FinalTrain, FinalTrainLabels)
predictions_1 = classifier_1.predict(FinalTest)
prediction_train_1=classifier_1.predict(FinalTrain)
print("Train Accuracy :",accuracy_score(FinalTrainLabels,prediction_train_1))
print("Train Macro f1 score :",f1_score(FinalTrainLabels, prediction_train_1, average = 'macro'))
print("Train Micro f1 scoore :",f1_score(FinalTrainLabels, prediction_train_1, average = 'micro'))
print("Train Classification Report :\n",classification_report(FinalTrainLabels, prediction_train_1))
print("Validation Accuracy :",accuracy_score(FinalTestLabels,predictions_1))
print("Validation Macro f1 score :",f1_score(FinalTestLabels, predictions_1, average = 'macro'))
print("Validation Micro f1 scoore :",f1_score(FinalTestLabels, predictions_1, average = 'micro'))
print("Validation Classification Report :\n",classification_report(FinalTestLabels, predictions_1))
test_data.head()
test_data['title']=test_data['title'].apply(lambda x: remove_html_tags(x))
test_data['content']=test_data['content'].apply(lambda x: remove_html_tags(x))
test_data['title']=test_data['title'].apply(lambda x: remove_space(x))
test_data['content']=test_data['content'].apply(lambda x: remove_space(x))

test_data['title_words']=test_data['title'].apply(lambda x: tokenize(x))

test_data['content_words']=test_data['content'].apply(lambda x: tokenize(x))
test_data['title_words']=test_data['title_words'].apply(lambda x: remove_punctuation(x))
test_data['content_words']=test_data['content_words'].apply(lambda x: remove_punctuation(x))
test_data['title_words']=test_data['title_words'].apply(lambda x: lower_word(x))
test_data['content_words']=test_data['content_words'].apply(lambda x: lower_word(x))
test_data['title_words']=test_data['title_words'].apply(lambda x: remove_stopwords(x))
test_data['content_words']=test_data['content_words'].apply(lambda x: remove_stopwords(x))

test_data.reset_index(drop=True,inplace=True)
test_data['text']=test_data['title_words']+ test_data['content_words']

test_data['text']=test_data['text'].apply(lambda x: lemmatization(x))

test_data.head()
test_data['text']=test_data['text'].apply(lambda x: ' '.join(x))
#test_data.to_csv('test_data_preprocessed.csv',index=False)
test_data.head()
test_data.to_csv('test_data_preprocessed.csv',index=False)
#test_data['title_words']=test_data['title_words'].apply(lambda x: ast.literal_eval(x))
#test_data['content_words']=test_data['content_words'].apply(lambda x: ast.literal_eval(x))
#test_data['text']=test_data['title_words']*3 + test_data['content_words']
test_data.text.head()
test_data.columns
len(vectorizer.get_feature_names())
#vectorizer = TfidfVectorizer(max_features=50000, ngram_range = (1,3), tokenizer = lambda x: x.split(" "))
#TrainData = vectorizer.fit_transform(x_train['text'])
test_data_features = vectorizer.transform(test_data['text'])   ##### We will use the vectoriser which we fit on training data
test_data_features.shape
predictions_test=classifier.predict(test_data_features)
predictions_test.shape
predictions_test_df=pd.DataFrame(predictions_test.toarray())
predictions_test_df.head()
vectorizer_label = CountVectorizer(tokenizer = lambda x: x.split(" "), binary = True)
new_labels = vectorizer_label.fit_transform(combined_data['tags'])
len(vectorizer_label.get_feature_names())
top_label_indices=sorted_tags[0:500]
top_500=[vectorizer_label.get_feature_names()[i] for i in top_label_indices]

predictions_probability=classifier.predict_proba(test_data_features)
predictions_probability=pd.DataFrame(predictions_probability)
predictions_probability.columns=top_500
predictions_probability.head()
for c in predictions_probability.columns.values.tolist():
    predictions_probability[c]=np.where(predictions_probability[c] >= 0.03,1,0)
cols_test = predictions_probability.columns

bt = predictions_probability.apply(lambda x: x > 0)

bt.head()
result=bt.apply(lambda x: list(cols_test[x.values]), axis=1)
result=pd.DataFrame(result)
result.columns=['tag']
result['tag']=result['tag'].apply(lambda x: ' '.join(x))
test_data.reset_index(drop=True,inplace=True)
result.reset_index(drop=True,inplace=True)
final_result=pd.concat([test_data,result],axis=1)
final_result.tag.unique()
final_result.loc[final_result['tag']=="electrical"]
final_result.head()
submission=final_result[['id','tag']]
submission.columns=['id','tags']
submission.head()
submission.to_csv('twelth_submission.csv',index=False)

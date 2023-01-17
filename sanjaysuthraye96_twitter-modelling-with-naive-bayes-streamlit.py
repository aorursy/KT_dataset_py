import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

import string

import missingno as msno

import nltk



import warnings   #Warnings

warnings.filterwarnings("ignore")



from wordcloud import WordCloud           

from itertools import chain

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import GridSearchCV

from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

%matplotlib inline 



print("Libraries Imported")
data=pd.read_csv("../input/twitter-60k/twitter.csv",encoding='latin-1' ) 

data.head(5)
print(data.info())

print("-"*80)

print("The Dataset contains class level count as\n", data["class_label"].value_counts())
data['class_label'] = data['class_label'].replace([4],1)
#Checking for Null values 

print(msno.matrix(data))

print(data.isnull().sum())
at=data[data['tweet'].str.contains("@")]

print(len(data.tweet))

print(len(data.tweet.unique()))
data.drop_duplicates(subset=['tweet', 'userid'],keep = False, inplace = True)

for index,text in enumerate(data['tweet'][18:20]):

  print('tweet %d:\n'%(index+1),text)
def rem(i):

    j=" ".join(filter(lambda x:x[0]!='@', i.split())) #removing words starting with '@'

    j1=re.sub(r"http\S+", "", j)                       # removing urls

    j2=" ".join(filter(lambda x:x[0]!='&', j1.split()))  #the "&" was found to be used in many misspelled words and usernames

    return j2

       
data["tweet"]=data["tweet"].apply(rem) #applying the function
for index,text in enumerate(data['tweet'][18:20]):

  print('tweet %d:\n'%(index+1),text)
def smiley(a):

    x1=a.replace(":‑)","happy")

    x2=x1.replace(";)","happy")

    x3=x2.replace(":-}","happy")

    x4=x3.replace(":)","happy")

    x5=x4.replace(":}","happy")

    x6=x5.replace("=]","happy")

    x7=x6.replace("=)","happy")

    x8=x7.replace(":D","happy")

    x9=x8.replace("xD","happy")

    x10=x9.replace("XD","happy")

    x11=x10.replace(":‑(","sad")    #using 'replace' to convert emoticons

    x12=x11.replace(":‑[","sad")

    x13=x12.replace(":(","sad")

    x14=x13.replace("=(","sad")

    x15=x14.replace("=/","sad")

    x16=x15.replace(":[","sad")

    x17=x16.replace(":{","sad")

  

    x18=x17.replace(":P","playful")

    x19=x18.replace("XP","playful")

    x20=x19.replace("xp","playful")

  

    

    x21=x20.replace("<3","love")

    x22=x21.replace(":o","shock")

    x23=x22.replace(":-/","sad")

    x24=x23.replace(":/","sad")

    x25=x24.replace(":|","sad")

    return x25
data['tweet']=data['tweet'].apply(smiley)
data['tweet']=data['tweet'].apply(lambda x: x.lower())
def decontracted(phrase):

    # specific

    phrase = re.sub(r"won\'t", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)   #using regular expressions to expand the contractions

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
data['tweet']=data['tweet'].apply(decontracted)
data['tweet']=data['tweet'].apply(lambda x: re.sub('\w*\d\w*','', x))
data['tweet']=data['tweet'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
def odd(a):

    words = ['wi','ame','quot','ti','im']

    querywords = a.split()



    resultwords  = [word for word in querywords if word.lower() not in words]

    result = ' '.join(resultwords)

    return result
data["tweet"]=data["tweet"].apply(odd)
data["tweet"]=data["tweet"].apply(lambda x: re.sub(' +', ' ', x))
stop=["i", "me", "my", "myself", "we", "our","will", "go","got","ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",  "nor", "only", "own", "same", "so", "than", "too", "very", "s", "t", "just", "don", "should", "now"]
print(stop)
data["tweet"]=data['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
data["tweet"]=data["tweet"].apply(lambda x: re.sub(' +', ' ', x))
data["word_count"] = data['tweet'].apply(lambda x: len(str(x).split()))
sns.distplot(data.word_count, kde=False, rug=True)
data.drop("word_count", axis=1, inplace=True)
pos = data[data.class_label==1]

cloud= (' '.join(pos['tweet']))

wcloud = WordCloud(width = 1000, height = 500).generate(cloud)

plt.figure(figsize=(15,5))

plt.imshow(wcloud)

plt.axis('off')
pos = data[data.class_label==0]

cloud= (' '.join(pos['tweet']))

wcloud = WordCloud(width = 1000, height = 500).generate(cloud)

plt.figure(figsize=(15,5))

plt.imshow(wcloud)

plt.axis('off')
data["word_count"] = data['tweet'].apply(lambda x: len(str(x).split()))
X=data.drop(['class_label',"id","date","flag","userid"], axis = 1)  # seperating the class label

y=data["class_label"].values
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
#Vectorizing

vectorizer = TfidfVectorizer() #Using TFIDF to vectorize the text
vectorizer = TfidfVectorizer(min_df=10,ngram_range=(1,3)) 

vectorizer.fit(X_train['tweet'].values)           # Training the TFIDF model

x_tr=vectorizer.transform(X_train['tweet'].values)

x_te=vectorizer.transform(X_test['tweet'].values)
x_tr.shape   #We get 4191 features after vectorizing 
model = MultinomialNB()  

parameters = {'alpha':[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5,

10, 50, 100]}

clf = GridSearchCV(model, parameters, cv=10,scoring='roc_auc',return_train_score=True)

clf.fit(x_tr, y_train)
results = pd.DataFrame.from_dict(clf.cv_results_)   # converting the results in to a dataframe

results = results.sort_values(['param_alpha'])  

results.head()
train_auc= results['mean_train_score'].values  #extracting the auc scores 

cv_auc = results['mean_test_score'].values
a1=[]

for i in parameters.values():

    a1.append(i)

alphas = list(chain.from_iterable(a1))
plt.plot(alphas, train_auc, label='Train AUC')

plt.plot(alphas, cv_auc, label='CV AUC')

plt.scatter(alphas, train_auc, label='Train AUC points')

plt.scatter(alphas, cv_auc, label='CV AUC points')



plt.legend()

plt.xlabel("Alpha: hyperparameter")

plt.ylabel("AUC")

plt.title("Hyper parameter Vs AUC plot")  

plt.grid()

plt.show()
bestparam=clf.best_params_['alpha']   #extracting the best hyperparameter

print("The best Alpha=",bestparam)
mul_model = MultinomialNB(alpha=bestparam) #Building a Naive Bayes model with the best alpha

mul_model.fit(x_tr,y_train)               #Training the model
y_train_pred = mul_model.predict_proba(x_tr)[:,1]  #Prediction using the model(log probability of each class)

y_test_pred = mul_model.predict_proba(x_te)[:,1]

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)

test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)   

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.title("AUC PLOTS")             #Plotting train and test AUC 

plt.grid()

plt.show()
trauc=round(auc(train_fpr, train_tpr),3)

teauc=round(auc(test_fpr, test_tpr),3)

print('Train AUC=',trauc)

print('Test AUC=',teauc)

def find_best_threshold(threshould, fpr, tpr):

    t = threshould[np.argmax(tpr*(1-fpr))]      #finding the best threashold 

    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))

    return t



def predict_with_best_t(proba, threshould):

    predictions = []

    for i in proba:

        if i>=threshould:

            predictions.append(1)

        else:                                 #building a confusion matrix with the best threashold 

            predictions.append(0)

    return predictions
best_t = find_best_threshold(tr_thresholds, train_fpr, train_tpr)

TRCM=confusion_matrix(y_train, predict_with_best_t(y_train_pred, best_t))

TECM=confusion_matrix(y_test, predict_with_best_t(y_test_pred, best_t))
def CM(x,y):

    labels = ['TN','FP','FN','TP']

    group_counts = ["{0:0.0f}".format(value) for value in x.flatten()]

                    

    labels = [f"{v1}\n{v2}" for v1, v2 in

    zip(labels,group_counts)]

    labels = np.asarray(labels).reshape(2,2)       #Building a design for the confusion matrix

    sns.heatmap(x, annot=labels, fmt='', cmap='BuPu')

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title(y)

    plt.plot()
CM(TRCM,'Train Confusion Matrix')
CM(TECM,'Test Confusion Matrix')
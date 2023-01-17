



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import string

import re

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.manifold import TSNE



from nltk.tokenize import word_tokenize

from nltk.tokenize import sent_tokenize

from nltk.stem import WordNetLemmatizer 



from imblearn.over_sampling import SMOTE
def get_sentence_word_count(text_list):

    sent_count = 0

    word_count = 0

    vocab = {}

    for text in text_list:

        sentences=sent_tokenize(str(text).lower())

        sent_count = sent_count + len(sentences)

        for sentence in sentences:

            words=word_tokenize(sentence)

            for word in words:

                if(word in vocab.keys()):

                    vocab[word] = vocab[word] +1

                else:

                    vocab[word] =1 

    word_count = len(vocab.keys())

    return sent_count,word_count

    
clinical_text_df = pd.read_csv("/kaggle/input/medicaltranscriptions/mtsamples.csv")



print(clinical_text_df.columns)

clinical_text_df.head(5)


clinical_text_df = clinical_text_df[clinical_text_df['transcription'].notna()]

sent_count,word_count= get_sentence_word_count(clinical_text_df['transcription'].tolist())

print("Number of sentences in transcriptions column: "+ str(sent_count))

print("Number of unique words in transcriptions column: "+str(word_count))







data_categories  = clinical_text_df.groupby(clinical_text_df['medical_specialty'])

i = 1

print('===========Original Categories =======================')

for catName,dataCategory in data_categories:

    print('Cat:'+str(i)+' '+catName + ' : '+ str(len(dataCategory)) )

    i = i+1

print('==================================')







filtered_data_categories = data_categories.filter(lambda x:x.shape[0] > 50)

final_data_categories = filtered_data_categories.groupby(filtered_data_categories['medical_specialty'])

i=1

print('============Reduced Categories ======================')

for catName,dataCategory in final_data_categories:

    print('Cat:'+str(i)+' '+catName + ' : '+ str(len(dataCategory)) )

    i = i+1



print('============ Reduced Categories ======================')

plt.figure(figsize=(10,10))

sns.countplot(y='medical_specialty', data = filtered_data_categories )

plt.show()
data = filtered_data_categories[['transcription', 'medical_specialty']]

data = data.drop(data[data['transcription'].isna()].index)

data.shape

print('Sample Transcription 1:'+data.iloc[5]['transcription']+'\n')

print('Sample Transcription 2:'+data.iloc[125]['transcription']+'\n')

print('Sample Transcription 3:'+data.iloc[1000]['transcription'])

def clean_text(text ): 

    text = text.translate(str.maketrans('', '', string.punctuation))

    text1 = ''.join([w for w in text if not w.isdigit()]) 

    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

    #BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

    

    text2 = text1.lower()

    text2 = REPLACE_BY_SPACE_RE.sub('', text2) # replace REPLACE_BY_SPACE_RE symbols by space in text

    #text2 = BAD_SYMBOLS_RE.sub('', text2)

    return text2



def lemmatize_text(text):

    wordlist=[]

    lemmatizer = WordNetLemmatizer() 

    sentences=sent_tokenize(text)

    

    intial_sentences= sentences[0:1]

    final_sentences = sentences[len(sentences)-2: len(sentences)-1]

    

    for sentence in intial_sentences:

        words=word_tokenize(sentence)

        for word in words:

            wordlist.append(lemmatizer.lemmatize(word))

    for sentence in final_sentences:

        words=word_tokenize(sentence)

        for word in words:

            wordlist.append(lemmatizer.lemmatize(word))       

    return ' '.join(wordlist) 



data['transcription'] = data['transcription'].apply(lemmatize_text)

data['transcription'] = data['transcription'].apply(clean_text)
print('Sample Transcription 1:'+data.iloc[5]['transcription']+'\n')

print('Sample Transcription 2:'+data.iloc[125]['transcription']+'\n')

print('Sample Transcription 3:'+data.iloc[1000]['transcription'])
vectorizer = TfidfVectorizer(analyzer='word', stop_words='english',ngram_range=(1,3), max_df=0.75, use_idf=True, smooth_idf=True, max_features=1000)

tfIdfMat  = vectorizer.fit_transform(data['transcription'].tolist() )

feature_names = sorted(vectorizer.get_feature_names())

print(feature_names)
import gc

gc.collect()

tfIdfMatrix = tfIdfMat.todense()

labels = data['medical_specialty'].tolist()

tsne_results = TSNE(n_components=2,init='random',random_state=0, perplexity=40).fit_transform(tfIdfMatrix)

plt.figure(figsize=(16,10))

palette = sns.hls_palette(21, l=.6, s=.9)

sns.scatterplot(

    x=tsne_results[:,0], y=tsne_results[:,1],

    hue=labels,

    palette= palette,

    legend="full",

    alpha=0.3

)

plt.show()



gc.collect()

pca = PCA(n_components=0.95)

tfIdfMat_reduced = pca.fit_transform(tfIdfMat.toarray())

labels = data['medical_specialty'].tolist()

category_list = data.medical_specialty.unique()

X_train, X_test, y_train, y_test = train_test_split(tfIdfMat_reduced, labels, stratify=labels,random_state=1)   

print('Train_Set_Size:'+str(X_train.shape))

print('Test_Set_Size:'+str(X_test.shape))
clf = LogisticRegression(penalty= 'elasticnet', solver= 'saga', l1_ratio=0.5, random_state=1).fit(X_train, y_train)

y_test_pred= clf.predict(X_test)
labels = category_list

cm = confusion_matrix(y_test, y_test_pred, labels)


fig = plt.figure(figsize=(20,20))

ax= fig.add_subplot(1,1,1)

sns.heatmap(cm, annot=True, cmap="Greens",ax = ax,fmt='g'); #annot=True to annotate cells



# labels, title and ticks

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 

ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);

plt.setp(ax.get_yticklabels(), rotation=30, horizontalalignment='right')

plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')     

plt.show()
print(classification_report(y_test,y_test_pred,labels=category_list))
filtered_data_categories['medical_specialty'] =filtered_data_categories['medical_specialty'].apply(lambda x:str.strip(x))

mask = filtered_data_categories['medical_specialty'] == 'Surgery'

filtered_data_categories = filtered_data_categories[~mask]

final_data_categories = filtered_data_categories.groupby(filtered_data_categories['medical_specialty'])

mask = filtered_data_categories['medical_specialty'] == 'SOAP / Chart / Progress Notes'

filtered_data_categories = filtered_data_categories[~mask]

mask = filtered_data_categories['medical_specialty'] == 'Office Notes'

filtered_data_categories = filtered_data_categories[~mask]

mask = filtered_data_categories['medical_specialty'] == 'Consult - History and Phy.'

filtered_data_categories = filtered_data_categories[~mask]

mask = filtered_data_categories['medical_specialty'] == 'Emergency Room Reports'

filtered_data_categories = filtered_data_categories[~mask]

mask = filtered_data_categories['medical_specialty'] == 'Discharge Summary'

filtered_data_categories = filtered_data_categories[~mask]



'''

mask = filtered_data_categories['medical_specialty'] == 'Pediatrics - Neonatal'

filtered_data_categories = filtered_data_categories[~mask]

'''

mask = filtered_data_categories['medical_specialty'] == 'Pain Management'

filtered_data_categories = filtered_data_categories[~mask]

mask = filtered_data_categories['medical_specialty'] == 'General Medicine'

filtered_data_categories = filtered_data_categories[~mask]





mask = filtered_data_categories['medical_specialty'] == 'Neurosurgery'

filtered_data_categories.loc[mask, 'medical_specialty'] = 'Neurology'

mask = filtered_data_categories['medical_specialty'] == 'Nephrology'

filtered_data_categories.loc[mask, 'medical_specialty'] = 'Urology'





i=1

print('============Reduced Categories======================')

for catName,dataCategory in final_data_categories:

    print('Cat:'+str(i)+' '+catName + ' : '+ str(len(dataCategory)) )

    i = i+1



print('============Reduced Categories======================')





data = filtered_data_categories[['transcription', 'medical_specialty']]

data = data.drop(data[data['transcription'].isna()].index)

data.shape
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_ner_bionlp13cg_md-0.2.5.tar.gz

import spacy

import en_ner_bionlp13cg_md

nlp = en_ner_bionlp13cg_md.load()
def process_Text( text):

    wordlist=[]

    doc = nlp(text)

    for ent in doc.ents:

        wordlist.append(ent.text)

    return ' '.join(wordlist)     
data['transcription'] = data['transcription'].apply(process_Text)

data['transcription'] = data['transcription'].apply(lemmatize_text)

data['transcription'] = data['transcription'].apply(clean_text)



print('Sample Transcription 1:'+data.iloc[5]['transcription']+'\n')

print('Sample Transcription 2:'+data.iloc[125]['transcription']+'\n')

print('Sample Transcription 3:'+data.iloc[1000]['transcription'])


vectorizer = TfidfVectorizer(analyzer='word', stop_words='english',ngram_range=(1,3), max_df=0.75,min_df=5, use_idf=True, smooth_idf=True,sublinear_tf=True, max_features=1000)

tfIdfMat  = vectorizer.fit_transform(data['transcription'].tolist() )

feature_names = sorted(vectorizer.get_feature_names())

print(feature_names)
import gc

gc.collect()

tfIdfMatrix = tfIdfMat.todense()

labels = data['medical_specialty'].tolist()

tsne_results = TSNE(n_components=2,init='random',random_state=0, perplexity=40).fit_transform(tfIdfMatrix)

plt.figure(figsize=(20,10))

palette = sns.hls_palette(12, l=.3, s=.9)

sns.scatterplot(

    x=tsne_results[:,0], y=tsne_results[:,1],

    hue=labels,

    palette= palette,

    legend="full",

    alpha=0.3

)

plt.show()



pca = PCA(n_components=0.95)

tfIdfMat_reduced = pca.fit_transform(tfIdfMat.toarray())

labels = data['medical_specialty'].tolist()

category_list = data.medical_specialty.unique()
X_train, X_test, y_train, y_test = train_test_split(tfIdfMat_reduced, labels, stratify=labels,random_state=1)   

print('Train_Set_Size:'+str(X_train.shape))

print('Test_Set_Size:'+str(X_test.shape))
#clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

clf = LogisticRegression(penalty= 'elasticnet', solver= 'saga', l1_ratio=0.5, random_state=1).fit(X_train, y_train)

y_test_pred= clf.predict(X_test)

labels = category_list

cm = confusion_matrix(y_test, y_test_pred, labels)



fig = plt.figure(figsize=(20,20))

ax= fig.add_subplot(1,1,1)

sns.heatmap(cm, annot=True, cmap="Greens",ax = ax,fmt='g'); #annot=True to annotate cells



# labels, title and ticks

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 

ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);

plt.setp(ax.get_yticklabels(), rotation=30, horizontalalignment='right')

plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')     

plt.show()

print(classification_report(y_test,y_test_pred,labels=category_list))

smote_over_sample = SMOTE(sampling_strategy='minority')

labels = data['medical_specialty'].tolist()

X, y = smote_over_sample.fit_resample(tfIdfMat_reduced, labels)



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1)   

print('Train_Set_Size:'+str(X_train.shape))

print('Test_Set_Size:'+str(X_test.shape))
clf = LogisticRegression(penalty= 'elasticnet', solver= 'saga', l1_ratio=0.5, random_state=1).fit(X_train, y_train)

y_test_pred= clf.predict(X_test)
labels = category_list

cm = confusion_matrix(y_test, y_test_pred, labels)



fig = plt.figure(figsize=(20,20))

ax= fig.add_subplot(1,1,1)

sns.heatmap(cm, annot=True, cmap="Greens",ax = ax,fmt='g'); #annot=True to annotate cells



# labels, title and ticks

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 

ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);

plt.setp(ax.get_yticklabels(), rotation=30, horizontalalignment='right')

plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')     

plt.show()

print(classification_report(y_test,y_test_pred,labels=category_list))
mask = filtered_data_categories['medical_specialty'] == 'Radiology'

radiologyData = filtered_data_categories[mask]

print(radiologyData['transcription'].tolist()[1])
mask = clinical_text_df['medical_specialty'] ==  ' Pediatrics - Neonatal'

pediaData = clinical_text_df[mask]

print(pediaData ['transcription'].tolist()[1])


mask = clinical_text_df['medical_specialty'] ==  ' Hematology - Oncology'

oncoData = clinical_text_df[mask]

print(oncoData ['transcription'].tolist()[1])
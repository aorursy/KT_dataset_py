import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.rcParams['patch.force_edgecolor']=True
Data = pd.read_csv("../input/spam.csv",engine='python')
Data.head()
Data=Data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
Data.rename(columns={'v1':'Category',
                    'v2':'Message'},inplace=True)
Data.head()
print('No. of Samples: {}'.format(Data.index.max()))
print('No. of nulls:\n{}'.format(Data.isnull().sum()))
Data['Msg_Length']=Data['Message'].apply(lambda X:len(X))
Data.head()
Data['Category'].value_counts()
sns.set_context(context='notebook',font_scale=2)
Data.hist(column='Msg_Length',by='Category',bins=100,figsize=(16,6))
print('Average length of spam messages: ',Data[Data['Category']=='spam']['Msg_Length'].mean(),'characters')
print('Average length of ham messages: ',Data[Data['Category']=='ham']['Msg_Length'].mean(),'characters')
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
corpus=[]
for i in range(0,5572):
    msg=re.sub('[^a-zA-Z]',' ',Data['Message'][i])
    msg=msg.lower()
    msg=msg.split()
    msg=[word for word in msg if not word in set(stopwords.words('english'))]
    msg=' '.join(msg)
    corpus.append(msg)
corpus[0:5]
words=[]
for i in range(0,5572):
    msg=re.sub('[^a-zA-Z]',' ',Data['Message'][i])
    msg=msg.lower()
    msg=msg.split()
    msg=[word for word in msg if not word in set(stopwords.words('english'))]
    for word in msg:
        words.append(word)
df=pd.DataFrame(words,columns=['Words'])
df=df['Words'].value_counts().to_frame().reset_index()
df.head()
print('Total words in whole dataset: ',df.index.max())
df=df[df['Words']>5]
print('Total words with frequency greater than 5 in whole dataset: ',df.index.max())
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=Data.iloc[:,0].values
X
X=np.concatenate((X,np.array(Data['Msg_Length']).reshape(5572,1)),axis=1)
X
X.shape
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,classification_report
Pipelines=[]
Pipelines.append(('Random Forest Classifier', Pipeline(steps=[('RF',RandomForestClassifier())])))
Pipelines.append(('SVC',Pipeline([('SVC',LinearSVC())])))
Pipelines.append(('MultinomialNB',Pipeline([('mNB',MultinomialNB())])))
Pipelines.append(('KNeighborsClassifier',Pipeline([('KNN',KNeighborsClassifier())])))
Pipelines.append(('GradientBoostingClassifier',Pipeline([('GBC',GradientBoostingClassifier())])))
Pipelines.append(('LogisticRegression',Pipeline([('LR',LogisticRegression())])))
for name,model in Pipelines:
    model.fit(X_train,y_train)
    print('CM of '+name+':'+'\n',confusion_matrix(y_test,model.predict(X_test)),'\n')
    print('CR of '+name+':'+'\n',classification_report(y_test,model.predict(X_test)),'\n')
Classifier=MultinomialNB()
Classifier.fit(X_train,y_train)
print('Confusion Matrix: \n', confusion_matrix(y_test,Classifier.predict(X_test)))
print('Classification Report: \n', classification_report(y_test,Classifier.predict(X_test)))
ham_words = ''
spam_words = ''
spam = Data[Data['Category']=='spam']
ham = Data[Data['Category']=='ham']
for msg in spam['Message']:
    text=re.sub('[^a-zA-Z]',' ',msg)
    text = text.lower()
    text=text.split()
    text=[word for word in text if not word in set(stopwords.words('english'))]
    for words in text:
        spam_words = spam_words+words+' '
        
for msg in ham['Message']:
    text=re.sub('[^a-zA-Z]',' ',msg)
    text = text.lower()
    text=text.split()
    text=[word for word in text if not word in set(stopwords.words('english'))]
    for words in text:
        ham_words = ham_words+words+' '
from wordcloud import WordCloud
# Generate a word cloud image
spam_wordcloud = WordCloud(width=1200, height=720,random_state=101).generate(spam_words)
ham_wordcloud = WordCloud(width=1200, height=720,random_state=101).generate(ham_words)
#Spam Word cloud
plt.figure( figsize=(16,9), facecolor='w')
plt.imshow(spam_wordcloud)
plt.title('Spam word cloud')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
# Ham word cloud
plt.figure( figsize=(16,9), facecolor='w')
plt.imshow(ham_wordcloud)
plt.title('Ham word cloud')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
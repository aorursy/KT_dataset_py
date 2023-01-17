import numpy as np

import pandas as pd

import os

import email

import email.policy

from bs4 import BeautifulSoup

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_score, recall_score, accuracy_score

from sklearn.ensemble import GradientBoostingClassifier



#Let's explore the directory segmentation

os.listdir('/kaggle/input/ham-and-spam-dataset/')
ham_filenames = [name for name in sorted(os.listdir('/kaggle/input/ham-and-spam-dataset/ham/')) if len(name) > 20]

spam_filenames = [name for name in sorted(os.listdir('/kaggle/input/ham-and-spam-dataset/spam')) if len(name) > 20]
#How's the dataset structure? How many ham/spam emails does it contain?



print('Total ham emails: ',len(ham_filenames))

print('Total spam emails: ',len(spam_filenames))

print('Spam percentage: ',100*(len(spam_filenames)/(len(ham_filenames)+len(spam_filenames))))
#Let's load an email to see how it looks like:



#Using email.parser: https://docs.python.org/3/library/email.parser.html

#"The email package provides a standard parser that understands most email document structures, including MIME documents"



with open(os.path.join('/kaggle/input/ham-and-spam-dataset/ham/', ham_filenames[0]), "rb") as file:

    ham_email =  email.parser.BytesParser(policy=email.policy.default).parse(file)



print('Header field names: ',ham_email.keys())

print('\n -------------------------------------- \n')

print('Message field values: ',ham_email.values())

print('\n -------------------------------------- \n')

print('Message content:',ham_email.get_content()[:500])
#Let's extract some email fields



email_subject = ham_email.get_all('Subject')

email_from = ham_email.get_all('From')

email_to = ham_email.get_all('To')



print('Email from: ',email_from)

print('Email to: ',email_to)

print('Email subject: ',email_subject)
def upload_ham(filename):

    """This function process a ham email file located at a specified directory and returns it as an email object"""

    directory = '/kaggle/input/ham-and-spam-dataset/ham/'

    with open(os.path.join(directory, filename), "rb") as file:

        return email.parser.BytesParser(policy=email.policy.default).parse(file)



def upload_spam(filename):

    """This function process a spam email file located at a specified directory and returns it as an email object"""

    directory = '/kaggle/input/ham-and-spam-dataset/spam/'

    with open(os.path.join(directory, filename), "rb") as file:

        return email.parser.BytesParser(policy=email.policy.default).parse(file)

    

ham_emails = [upload_ham(filename=name) for name in ham_filenames]

spam_emails = [upload_spam(filename=name) for name in spam_filenames]
#Checking if everything was uploaded properly:



print(ham_emails[0].get_all('Subject'))

print(ham_emails[0].get_content())

print('\n\n -----------------------------------------------------------\n\n')

print(spam_emails[1].get_all('Subject'))

print(spam_emails[1].get_content())
#Let's research about what email content types are:



ham_email_types = []

spam_email_types = []



for i in range(len(ham_filenames)):

    ham_email_types.append(ham_emails[i].get_content_type())



for i in range(len(spam_filenames)):

    spam_email_types.append(spam_emails[i].get_content_type())



print('Ham content types: ',set(ham_email_types))

print('Spam content types: ',set(spam_email_types))
#We need to identify what the multipart emails are structured of



def email_content_type(email):

    """This function returns the content type of an email and if it has a multipart shape then returns the multiparts type"""

    if isinstance(email, str):

        return email

    payload = email.get_payload()

    if isinstance(payload, list):

        return "multipart({})".format(", ".join([email_content_type(sub_email) for sub_email in payload]))

    else:

        return email.get_content_type()
ham_email_types = []

spam_email_types = []



for i in range(len(ham_filenames)):

    ham_email_types.append(email_content_type(ham_emails[i]))



for i in range(len(spam_filenames)):

    spam_email_types.append(email_content_type(spam_emails[i]))



print('Ham content types: ',set(ham_email_types))

print('Spam content types: ',set(spam_email_types))
#Now that we've identified what are the email content types, we need to transform all html emails to plain format.



from bs4 import BeautifulSoup

html = spam_emails[1].get_content()

soup = BeautifulSoup(html)

print(soup.get_text().replace('\n\n',''))
#Let's build a function with the previous process to convert all html emails into plain text



def html_to_plain(email):

    soup = BeautifulSoup(email.get_content())

    return soup.get_text().replace('\n\n','').replace('\n',' ') 
#Now all emails which have/contain HTML tags will be converted to plain text



def email_to_plain(email):

    content_type = email_content_type(email)

    for part in email.walk(): 

        #The .walk() documentation at https://docs.python.org/3/library/email.message.html

        #"The walk() method is an all-purpose generator which can be used to iterate over all 

        #the parts and subparts of a message object tree, in depth-first traversal order."

        partContentType = part.get_content_type()

        if partContentType not in ['text/plain','text/html']:

            continue

        try:

            partContent = part.get_content()

        except: 

            partContent = str(part.get_payload())

        if partContentType == 'text/plain':

            return partContent

        else:

            return html_to_plain(part)
#Let's test this out.

email_test1 = email_to_plain(spam_emails[1])

email_test2 = email_to_plain(spam_emails[227])

print(email_test1)

print('\n\n')

print(email_test2[:1000])
#Spam email #226 contains an unknown encoding so we will remove it from the list

del spam_emails[226]
ham_dataset = []

spam_dataset = []



#Ham processing

for i in range(len(ham_emails)):

    ham_dataset.append(email_to_plain(ham_emails[i]))

ham_dataset = pd.DataFrame(ham_dataset,columns=['Email content'])

ham_dataset['Label'] = 0



#Spam processing

for i in range(len(spam_emails)):

    spam_dataset.append(email_to_plain(spam_emails[i]))

spam_dataset = pd.DataFrame(spam_dataset,columns=['Email content'])

spam_dataset['Label'] = 1



dataset = pd.concat([ham_dataset,spam_dataset])

dataset.head()
#We will shuffle the data and also reset indexes

dataset = dataset.dropna()

dataset = dataset.sample(frac=1).reset_index(drop=True)

dataset.head()
#Removing special chars because they just add noise and make the models poor when predicting.

for i in range(len(dataset)):

    dataset.at[i,'Email content'] = dataset.loc[i]['Email content'].replace('!','').replace('?','').replace(',','').replace('[','').replace(']','').replace('(','').replace(')','').replace('...','')

    dataset.at[i,'Email content'] = dataset.loc[i]['Email content'].replace('>','').replace('<','').replace('\n',' ').replace('-','').replace('+','').replace('#','')

dataset.head()
def input_preprocessing(text):

    text = text.replace('!','').replace('?','').replace(',','').replace('[','').replace(']','').replace('(','').replace(')','').replace('...','')

    text = text.replace('>','').replace('<','').replace('\n',' ').replace('-','').replace('+','').replace('#','')

    return text
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(dataset['Email content'],dataset['Label'],shuffle=True,random_state=0)
#checking if everything went OK.

print (len(X_train),len(X_test),len(y_train),len(y_test))
pipe = make_pipeline(CountVectorizer(min_df=5,analyzer='char_wb'), LogisticRegression(max_iter=10000))

param_grid = {"logisticregression__C": [0.1, 1, 10, 100],

"countvectorizer__ngram_range": [(1, 2), (1, 3),(2,5)]}

grid = GridSearchCV(pipe, param_grid, cv=5)

grid.fit(X_train, y_train)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))

print("Best parameters:\n{}".format(grid.best_params_))
vect = CountVectorizer(min_df=5, analyzer='char_wb', ngram_range=[2,5])



X_train_vectorized = vect.fit_transform(X_train)

X_test_vectorized = vect.transform(X_test)



clf = LogisticRegression(C=0.1,max_iter=500).fit(X_train_vectorized, y_train)

y_predicted = clf.predict(X_test_vectorized)
print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_predicted)))

print("Accuracy: {:.2f}%".format(100 * accuracy_score(y_test, y_predicted)))

print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_predicted)))

print("AUC score: {:.2f}%".format(100 * roc_auc_score(y_test, y_predicted)))
pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression(max_iter=10000))

# running the grid search takes a long time because of the

# relatively large grid and the inclusion of trigrams

param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],

"tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3),(2,5)]}

grid = GridSearchCV(pipe, param_grid, cv=5)

grid.fit(X_train, y_train)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))

print("Best parameters:\n{}".format(grid.best_params_))
vect = TfidfVectorizer(min_df=5,ngram_range=(1,2)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

X_test_vectorized = vect.transform(X_test)

clf = LogisticRegression(C=100,max_iter=10000).fit(X_train_vectorized, y_train)

y_predicted = clf.predict(X_test_vectorized)
print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_predicted)))

print("Accuracy: {:.2f}%".format(100 * accuracy_score(y_test, y_predicted)))

print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_predicted)))

print("AUC score: {:.2f}%".format(100 * roc_auc_score(y_test, y_predicted)))
vect = TfidfVectorizer(min_df=5,ngram_range=(2,5)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

X_test_vectorized = vect.transform(X_test)

param_grid = {'C':[1,10,100,1000,10000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}

grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=0)

grid.fit(X_train_vectorized, y_train)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))

print("Best parameters:\n{}".format(grid.best_params_))
vect = TfidfVectorizer(min_df=5,ngram_range=(2,5)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

X_test_vectorized = vect.transform(X_test)

clf = SVC(C=100,gamma= 0.1,kernel='rbf').fit(X_train_vectorized, y_train)

y_predicted = clf.predict(X_test_vectorized)
print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_predicted)))

print("Accuracy: {:.2f}%".format(100 * accuracy_score(y_test, y_predicted)))

print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_predicted)))

print("AUC score: {:.2f}%".format(100 * roc_auc_score(y_test, y_predicted)))
vect = TfidfVectorizer(min_df=5,ngram_range=(2,5)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

X_test_vectorized = vect.transform(X_test)

param_grid = {'max_depth':[2,5,10,20,30,50,100,200],'n_estimators':[10,20,50]}

grid = GridSearchCV(RandomForestClassifier(random_state=0),param_grid,refit = True, verbose=0)

grid.fit(X_train_vectorized, y_train)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))

print("Best parameters:\n{}".format(grid.best_params_))
vect = TfidfVectorizer(min_df=5,ngram_range=(2,5)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

X_test_vectorized = vect.transform(X_test)

clf = RandomForestClassifier(n_estimators=50,max_depth=100,random_state=0).fit(X_train_vectorized, y_train)

y_predicted = clf.predict(X_test_vectorized)
print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_predicted)))

print("Accuracy: {:.2f}%".format(100 * accuracy_score(y_test, y_predicted)))

print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_predicted)))

print("AUC score: {:.2f}%".format(100 * roc_auc_score(y_test, y_predicted)))
vect = TfidfVectorizer(min_df=5,ngram_range=(2,5)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

X_test_vectorized = vect.transform(X_test)

param_grid = {'max_depth':[30,50,100]}

grid = GridSearchCV(GradientBoostingClassifier(random_state=0,n_estimators=50),param_grid,refit = True, verbose=0)

grid.fit(X_train_vectorized, y_train)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))

print("Best parameters:\n{}".format(grid.best_params_))
vect = TfidfVectorizer(min_df=5,ngram_range=(2,5)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

X_test_vectorized = vect.transform(X_test)

clf = GradientBoostingClassifier(max_depth=30,random_state=0).fit(X_train_vectorized, y_train)

y_predicted = clf.predict(X_test_vectorized)
print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_predicted)))

print("Accuracy: {:.2f}%".format(100 * accuracy_score(y_test, y_predicted)))

print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_predicted)))

print("AUC score: {:.2f}%".format(100 * roc_auc_score(y_test, y_predicted)))
vect = TfidfVectorizer(min_df=5,ngram_range=(2,5)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

X_test_vectorized = vect.transform(X_test)

clf = SVC(C=100,gamma= 0.1,kernel='rbf').fit(X_train_vectorized, y_train)

y_predicted = clf.predict(X_test_vectorized)
print("Vocabulary size: {}".format(len(vect.vocabulary_)))

print("Features with highest idf:\n{}".format(vect.get_feature_names()[-50:]))
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

matrix = confusion_matrix(y_test, y_predicted)
import seaborn as sns

conf_matrix = pd.DataFrame(matrix, index = ['Ham','Spam'],columns = ['Ham','Spam'])

#Normalizing

conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize = (15,15))

sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15})
email_test = ['Good morning mates. This is just to let you all know we have scheduled a meeting for EOD.']

email_test[0] = input_preprocessing(email_test[0])

email_test = vect.transform(email_test)

prediction = clf.predict(email_test)



if prediction == 0:

    print('The email has not been flagged as SPAM.')

else:

    print('The email has been flagged as SPAM.')
email_test = ['Dear Sergio, Flash Sale at Walmart! 25% OFF all weekend!']

email_test[0] = input_preprocessing(email_test[0])

email_test = vect.transform(email_test)

prediction = clf.predict(email_test)



if prediction == 0:

    print('The email has not been flagged as SPAM.')

else:

    print('The email has been flagged as SPAM.')
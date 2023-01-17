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
pd.DataFrame(list(zip(X_train,y_train)))
!pip install transformers==2.10.0
!pip install simpletransformers
from simpletransformers.classification import ClassificationModel

import pandas as pd





# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present, the Dataframe should contain at least two columns, with the first column is the text with type str, and the second column in the label with type int.

# train_data = [['Example sentence belonging to class 1', 1], ['Example sentence belonging to class 0', 0], ['Example eval senntence belonging to class 2', 2]]

train_df = pd.DataFrame(list(zip(X_train,y_train)))



# eval_data = [['Example eval sentence belonging to class 1', 1], ['Example eval sentence belonging to class 0', 0], ['Example eval senntence belonging to class 2', 2]]

eval_df = pd.DataFrame(list(zip(X_test,y_test)))



# Create a ClassificationModel

model = ClassificationModel('bert', 'bert-base-cased', num_labels=3, args={'reprocess_input_data': True, 'overwrite_output_dir': True},use_cuda=False)

# You can set class weights by using the optional weight argument



# Train the model

model.train_model(train_df)



# Evaluate the model

result, model_outputs, wrong_predictions = model.eval_model(eval_df)



predictions, raw_outputs = model.predict(X_test.to_list())
result
predictions, raw_outputs = model.predict(X_test.to_list())
y_predicted= predictions
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

matrix = confusion_matrix(y_test, y_predicted)
import seaborn as sns

conf_matrix = pd.DataFrame(matrix, index = ['Ham','Spam'],columns = ['Ham','Spam'])

#Normalizing

conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize = (15,15))

sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15})
email_test = ['Dear Sergio, Flash Sale at Walmart! 25% OFF all weekend!']

prediction = model.predict(email_test)



if prediction == 0:

    print('The email has not been flagged as SPAM.')

else:

    print('The email has been flagged as SPAM.')
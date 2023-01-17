# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
rawdata = pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')
rawdata.head()
rawdata =rawdata.drop(['job_id','title','location','department'],axis =1)
#rawdata.head()
rawdata  =rawdata.sort_values('fraudulent',ascending = False)
no_of_ones = int(np.sum(rawdata['fraudulent']))
no_of_ones
balanceddata = rawdata.head((2*no_of_ones))## data is balanced
balanceddata.shape
data_preprocessed = balanceddata.drop(['function','salary_range'],axis =1)
data_preprocessed.head()
cat_columns = ['employment_type', 'required_experience', 'required_education']

for col in cat_columns:
    data_preprocessed[col].fillna("Unknown", inplace=True)

text_columns = ['company_profile', 'description', 'requirements', 'benefits']

data_preprocessed = data_preprocessed.dropna(subset=text_columns, how='all')

for col in text_columns:
    data_preprocessed[col].fillna(' ', inplace=True)
data_preprocessed.reset_index(drop= True,inplace = True)

shuffled_indicies = np.arange(data_preprocessed.shape[0])
np.random.shuffle(shuffled_indicies)

shuffled_data = data_preprocessed.iloc[shuffled_indicies]
data_w_dummies = pd.get_dummies(shuffled_data, columns=['has_company_logo',
                                                        'employment_type',
                               'has_questions',
                               'employment_type',
                               'required_experience',
                               'required_education',
                               ])
#to get dummies
textdata = ['company_profile','description','requirements','benefits','industry']
#textdata
data_w_dummies['industry']= data_w_dummies['industry'].astype(str)
import re
import string

def clean_text(text):
    text = text.lower()                                              # make the text lowercase
    text = re.sub('\[.*?\]', '', text)                               # remove text in brackets
    text = re.sub('http?://\S+|www\.\S+', '', text)                  # remove links
    text = re.sub('https?://\S+|www\.\S+', '', text)                 # remove links
    text = re.sub('<.*?>+', '', text)                                # remove HTML stuff
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # get rid of punctuation
    text = re.sub('\n', '', text)                                    # remove line breaks
    #text = re.sub('\w*\d\w*', '', text)                             # remove anything with numbers, if you want
    #text = re.sub(r'[^\x00-\x7F]+',' ', text)                       # remove unicode
    return text

for c in textdata:
         data_w_dummies[c] = data_w_dummies[c].apply(lambda x: clean_text(x))



#data_w_dummies.head()
list(data_w_dummies.columns)
inputs = data_w_dummies[['company_profile',
 'description',
 'requirements',
 'benefits',
 'telecommuting',
 'industry',
  'has_company_logo_0',
 'has_company_logo_1',
 'employment_type_Contract',
 'employment_type_Full-time',
 'employment_type_Other',
 'employment_type_Part-time',
 'employment_type_Temporary',
 'employment_type_Unknown',
 'has_questions_0',
 'has_questions_1',
 'employment_type_Contract',
 'employment_type_Full-time',
 'employment_type_Other',
 'employment_type_Part-time',
 'employment_type_Temporary',
 'employment_type_Unknown',
 'required_experience_Associate',
 'required_experience_Director',
 'required_experience_Entry level',
 'required_experience_Executive',
 'required_experience_Internship',
 'required_experience_Mid-Senior level',
 'required_experience_Not Applicable',
 'required_experience_Unknown',
 'required_education_Associate Degree',
 "required_education_Bachelor's Degree",
 'required_education_Certification',
 'required_education_Doctorate',
 'required_education_High School or equivalent',
 "required_education_Master's Degree",
 'required_education_Professional',
 'required_education_Some College Coursework Completed',
 'required_education_Some High School Coursework',
 'required_education_Unknown',
 'required_education_Unspecified',
 'required_education_Vocational',
 'required_education_Vocational - Degree',
 'required_education_Vocational - HS Diploma']]

target = data_w_dummies['fraudulent']
target.head()
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_words = stopwords.words('english')

def remove_stopwords(text):
    words = [w for w in text if w not in stop_words]
    return words

def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text

for c in textdata:
        inputs[c] = inputs[c].apply(lambda x: tokenizer.tokenize(x))
        inputs[c] = inputs[c].apply(lambda x : remove_stopwords(x))
        inputs[c] = inputs[c].apply(lambda x : combine_text(x))

inputs.head()
textinputs = inputs[textdata]
nontextinputs = inputs.drop(textdata,axis = 1)
#nontextinputs.head()
#inputs.drop(textdata)
textcombined = textinputs['company_profile'] + " " + textinputs['description']+ " " + textinputs['requirements'] + " " + textinputs['benefits'] + " " +textinputs['industry']
#textcombined.head()

from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()

vectorinputs = []
#for c in textdata:
vectorinputs = count_vectorizer.fit_transform(textcombined)
#vectorinputs = vectorinputs.todense()
#vectorinputs = vectorinputs.transpose()
#vectorinputs = np.array(list(x for x in vectorinputs))
vectorinputs
df = pd.DataFrame(vectorinputs.todense())
newinputs = pd.concat([df,nontextinputs],axis=1, sort=False)
# Count the total number of samples
samples_count = newinputs.shape[0]

# Count the samples in each subset, assuming we want 80-10-10 distribution of training, validation, and test.
# Naturally, the numbers are integers.
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)

# The 'test' dataset contains all remaining data.
test_samples_count = samples_count - train_samples_count - validation_samples_count

# Create variables that record the inputs and targets for training
# In our shuffled dataset, they are the first "train_samples_count" observations
train_inputs = newinputs[:train_samples_count]
train_targets = target[:train_samples_count]

# Create variables that record the inputs and targets for validation.
# They are the next "validation_samples_count" observations, folllowing the "train_samples_count" we already assigned
validation_inputs = newinputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = target[train_samples_count:train_samples_count+validation_samples_count]

# Create variables that record the inputs and targets for test.
# They are everything that is remaining.
test_inputs = newinputs[train_samples_count+validation_samples_count:]
test_targets = target[train_samples_count+validation_samples_count:]

# We balanced our dataset to be 50-50 (for targets 0 and 1), but the training, validation, and test were 
# taken from a shuffled dataset. Check if they are balanced, too. Note that each time you rerun this code, 
# you will get different values, as each time they are shuffled randomly.
# Normally you preprocess ONCE, so you need not rerun this code once it is done.
# If you rerun this whole sheet, the npzs will be overwritten with your newly preprocessed data.

# Print the number of targets that are 1s, the total number of samples, and the proportion for training, validation, and test.
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)
import tensorflow as tf
from scipy.sparse import dok_matrix
# we extract the inputs using the keyword under which we saved them
# to ensure that they are all floats, let's also take care of that
train_inputs = train_inputs.astype(np.float)
# targets must be int because of sparse_categorical_crossentropy (we want to be able to smoothly one-hot encode them)
train_targets = train_targets.astype(np.int)

# we load the validation data in the temporary variable
#npz = np.load('Audiobooks_data_validation.npz')
# we can load the inputs and the targets in the same line
validation_inputs, validation_targets = validation_inputs.astype(np.float), validation_targets.astype(np.int)

# we load the test data in the temporary variable
#npz = np.load('Audiobooks_data_test.npz')
# we create 2 variables that will contain the test inputs and the test targets
test_inputs, test_targets = test_inputs.astype(np.float), test_targets.astype(np.int)
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(200, input_dim=(newinputs.shape[1]), activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())



model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Activation('sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)
model.fit(train_inputs, y=train_targets, batch_size=64, 
          epochs=10,callbacks=[early_stopping], verbose=2, 
          validation_data=(validation_inputs, validation_targets))
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)

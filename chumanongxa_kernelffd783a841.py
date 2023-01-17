import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.corpus import stopwords

# sklearn Library

from sklearn.feature_extraction.text import TfidfVectorizer 

from sklearn.metrics import f1_score # better metric 

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.svm import SVC

# Visualization Libraries

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

import re

import string

# ignoring warnings for goof appearance of the notebook. 

import warnings 

warnings.filterwarnings('ignore')
#Use pandas to read the csv files

df_train = pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/test.csv')
#Explore a few rows in the train data

df_train.head()
#Total number of posts for each personality type

personality_types = df_train['type'].value_counts()
#Check the ditribution of MBTI personality types

plt.figure(figsize=(12,4))

sns.barplot(personality_types.index, personality_types.values, alpha=0.8)

plt.ylabel('Number of posts', fontsize=12)

plt.xlabel('Personality Types', fontsize=12)

plt.title('Total posts for each of the personality types')

plt.show()
fig, ax = plt.subplots(len(df_train['type'].unique()), sharex=True, figsize=(10,10*len(df_train['type'].unique())))

count = 0

for i in df_train['type'].unique():

    df_4 = df_train[df_train['type'] == i]

    wordcloud = WordCloud(max_font_size=40, max_words=100, background_color="gray").generate(df_4['posts'].to_string())

    ax[count].imshow(wordcloud, interpolation='bilinear')

    ax[count].set_title(i)

    ax[count].axis("off")

    count+=1
#concatenate the train and the test dataset

all_data=pd.concat([df_train[['posts']],df_test[['posts']]])
#checking the data for missing values

all_data.isnull().any()
#check the size of the dataset

all_data.shape
def clean_data(df, column):

    '''

    This function applies methodologies that clean the data.



    parameters:

    df (obj) : dataframe of uncleaned posts

    column (obj) : string value, column name of dataframe df



    return:

    df (obj) : dataframe of cleaned data post column



    '''

    # remove url links

    df[column] = df[column].apply(lambda x: re.sub(r'https?:\/\/.*?[\s+]', '', x.replace("|", " ") + " "))

    # strip punctuation

    df[column] = df[column].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    # convert text to lowercase

    df[column] = df[column].str.lower()

    # remove numbers from the dataframe

    df[column] = df[column].apply(lambda x: re.sub('[0-9]+', '', x))

    # returning the clean dataframe

    return df
# Making use of the clean_data function

all_data=clean_data(all_data, 'posts')
#Explore the data after cleaning

all_data.head()
# saving the clean data to a csv

all_data.to_csv('all_data.csv', index=False)
# reading the csv of the clean dataframe

all_data = pd.read_csv('all_data.csv')
# Create the transform 

vectorizer = TfidfVectorizer(ngram_range=(1,1), analyzer='word' ,stop_words='english')

all_data = vectorizer.fit_transform(all_data['posts'])
# Demerging the train and test dataframe from the all_data dataframe. 

len_slice = df_train.shape[0] 

# Slice train data frame from all_data dataframe

train = all_data[:len_slice]

# Slice test data frame from all_data dataframe

test = all_data[len_slice:]

# Target variable

y = df_train['type'] 
# Applying the encoding to the target variable y

mind = y.apply(lambda x: 0 if x[0] == 'I' else 1)

energy = y.apply(lambda x: 0 if x[1] == 'S' else 1)

nature = y.apply(lambda x: 0 if x[2] == 'F' else 1)

tactics = y.apply(lambda x: 0 if x[3] == 'P' else 1)
#Balanced weighted Logistic Regression

classifier = LogisticRegression(random_state=0, class_weight='balanced') 
# Defining the k-fold

kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

score = {'acc': 'accuracy', 'f1': 'f1_micro'}
# Applying cross validation where train is the feature and mind is the target variable

val_results = cross_validate(classifier,train, mind, scoring=score,

                        cv=kfolds, n_jobs=-1, verbose=1)
print('Logistic Regression performance for mind:')

for values in val_results:

        print(values + ' : ', val_results[values].mean())
# Applying cross validation where train is the feature and energy is the target variable

val_results = cross_validate(classifier,train, energy, scoring=score,cv=kfolds, n_jobs=-1, verbose=1)
print('Logistic Regression performance for energy:')

for values in val_results:

    print(values + ' : ', val_results[values].mean())
# applying cross validation where train is the feature and nature is the target variable

val_results = cross_validate(classifier,train, nature, scoring=score,cv=kfolds, n_jobs=-1, verbose=1)
print('Logistic Regression performance for nature:')

for values in val_results:

    print(values + ' : ', val_results[values].mean())
# applying cross validation where train is the feature and nature is the tactics variable

val_results = cross_validate(classifier,train, tactics, scoring=score,cv=kfolds, n_jobs=-1, verbose=1)
print('Logistic Regression performance for tactics:')

for values in val_results:

    print(values + ' : ', val_results[values].mean())
# Balanced weighted SVC

svm_classifier = SVC(kernel='linear', random_state=0, class_weight='balanced')
# Applying cross validation where train is the feature and mind is the target variable

val_results2 = cross_validate(svm_classifier,train, mind, scoring=score,

                        cv=kfolds, n_jobs=-1, verbose=1)
print('SVM performance for mind:')

for values in val_results2:

        print(values + ' : ', val_results2[values].mean())
# Applying cross validation where train is the feature and energy is the target variable

val_results2 = cross_validate(svm_classifier,train, energy, scoring=score,cv=kfolds, n_jobs=-1, verbose=1)
print('SVM performance for energy:')

for values in val_results2:

        print(values + ' : ', val_results2[values].mean())
# Applying cross validation where train is the feature and tactics is the target variable

val_results2 = cross_validate(svm_classifier,train, tactics, scoring=score,cv=kfolds, n_jobs=-1, verbose=1)
print('SVM performance for tactics:')

for values in val_results2:

        print(values + ' : ', val_results2[values].mean())
# Applying cross validation where train is the feature and nature is the target variable

val_results2 = cross_validate(svm_classifier,train, nature, scoring=score,cv=kfolds, n_jobs=-1, verbose=1)
print('SVM performance for nature:')

for values in val_results2:

        print(values + ' : ', val_results2[values].mean())
def Logistic_Regression(train_df, test_df, cate_df):

    '''

    This function fits the train dataframe and the test dataframe and makes probability predictions on the categorical dataframe.



    parameters:

    train_df (obj) : dataframe of train data

    test_df (obj) :  dataframe of test data

    cate_df (obj) : dataframe of encoded vLUES



    return:

    predictions (obj) : dataframe of predicted values



    '''

    # Instantiating the object

    classifier = LogisticRegression(random_state=0, class_weight='balanced')

    # Fitting the model

    classifier.fit(train_df, cate_df)

    # Predictions on the test dataframe

    prediction = pd.DataFrame(classifier.predict(test_df))

    # Returning the result

    return prediction

# Applying the Logistic_Regression function on the mind category

M_prediction = Logistic_Regression(train, test, mind)
# Applying the Logistic_Regression function on the energy category

E_prediction = Logistic_Regression(train, test, energy)
# Applying the Logistic_Regression function on the nature category

N_prediction = Logistic_Regression(train, test, nature)
# Applying the Logistic_Regression function on the tactics category

T_prediction = Logistic_Regression(train, test, tactics)
# Concatenating our predictions into one dataframe

sub = pd.concat([M_prediction,E_prediction,N_prediction,T_prediction], axis=1)
# Renaming the columns of the sub dataframe

sub.reset_index(inplace=True)

sub['index'] = sub['index'] +1

sub.columns = ['id', 'mind', 'energy', 'nature', 'tactics']
# Submitting the results to a scv

sub.to_csv('LogisticRegression.csv', index=False)
def Support_Vector_Classifier(train_df, test_df, cate_df):

    '''

    This function fits the train dataframe and the test dataframe and makes probability predictions on the categorical dataframe.



    parameters:

    train_df (obj) : dataframe of train data

    test_df (obj) :  dataframe of test data

    cate_df (obj) : dataframe of encoded vLUES



    return:

    predictions (obj) : dataframe of predicted values



    '''

    # instantiating the object

    svm_classifier = SVC(kernel='linear', random_state=0, class_weight='balanced', probability=True)

    # fitting the model

    svm_classifier.fit(train_df, cate_df)

    # predictions on the test dataframe

    prediction = pd.DataFrame(svm_classifier.predict_proba(test_df))[1]

    # returning the result

    return prediction

# Applying the Logistic_Regression function on the mind category

M_prediction2 = Support_Vector_Classifier(train, test, mind)
# Applying the Logistic_Regression function on the energy category

E_prediction2 = Support_Vector_Classifier(train, test, energy)
# Applying the Logistic_Regression function on the nature category

N_prediction2 = Support_Vector_Classifier(train, test, nature)
# Applying the Logistic_Regression function on the tactics category

T_prediction2 = Support_Vector_Classifier(train, test, tactics)
# Concatenating our predictions into one dataframe

sub2 = pd.concat([M_prediction2,E_prediction2,N_prediction2,T_prediction2], axis=1)
# Renaming the columns of the sub2 dataframe

sub2.reset_index(inplace=True)

sub2['index'] = sub2['index'] +1 

sub2.columns = ['id', 'mind', 'energy', 'nature', 'tactics']
# Submitting the results to a scv

sub2.to_csv('SVM.csv', index=False)
#LOAD ALL PACKAGES

!pip install contractions



import pandas as pd

from sklearn.model_selection import train_test_split

import contractions

import re

import en_core_web_sm

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from imblearn.over_sampling import SMOTE

from keras.layers import Input,Embedding, LSTM, Dense, Concatenate

from keras.models import Model

from keras.utils import plot_model
#LOAD DATA

j = pd.read_csv('../input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')



#DROP NAN ROWS

j = j.dropna()



list(j.columns)
#CHECK CLASSES DISTRIBUTION

j['fraudulent'].value_counts()



#DUMMY CODING CATEGORICAL VARIABLES

j = pd.get_dummies(j, columns=['has_company_logo',

                               'has_questions',

                               'employment_type',

                               'required_experience',

                               'required_education',

                               ])



list(j.columns)
#PREPARE X AND Y

X = j[['title',

 'location',

 'department',

 'company_profile',

 'description',

 'requirements',

 'benefits',

 'industry',

 'function',

 'has_company_logo_0',

 'has_company_logo_1',

 'has_questions_0',

 'has_questions_1',

 'employment_type_Contract',

 'employment_type_Full-time',

 'employment_type_Other',

 'employment_type_Part-time',

 'employment_type_Temporary',

 'required_experience_Associate',

 'required_experience_Director',

 'required_experience_Entry level',

 'required_experience_Executive',

 'required_experience_Internship',

 'required_experience_Mid-Senior level',

 'required_experience_Not Applicable',

 'required_education_Associate Degree',

 "required_education_Bachelor's Degree",

 'required_education_Certification',

 'required_education_High School or equivalent',

 "required_education_Master's Degree",

 'required_education_Professional',

 'required_education_Some College Coursework Completed',

 'required_education_Unspecified',

 'required_education_Vocational',

 'required_education_Vocational - HS Diploma']]



y = j['fraudulent'].to_list()



#CONCAT ALL TEXT COLUMNS

X['text'] = X[['title',

 'location',

 'department',

 'company_profile',

 'description',

 'requirements',

 'benefits',

 'industry',

 'function',]].agg('-'.join, axis=1) 



X = X[['text',

       'has_company_logo_0',

 'has_company_logo_1',

 'has_questions_0',

 'has_questions_1',

 'employment_type_Contract',

 'employment_type_Full-time',

 'employment_type_Other',

 'employment_type_Part-time',

 'employment_type_Temporary',

 'required_experience_Associate',

 'required_experience_Director',

 'required_experience_Entry level',

 'required_experience_Executive',

 'required_experience_Internship',

 'required_experience_Mid-Senior level',

 'required_experience_Not Applicable',

 'required_education_Associate Degree',

 "required_education_Bachelor's Degree",

 'required_education_Certification',

 'required_education_High School or equivalent',

 "required_education_Master's Degree",

 'required_education_Professional',

 'required_education_Some College Coursework Completed',

 'required_education_Unspecified',

 'required_education_Vocational',

 'required_education_Vocational - HS Diploma']]



#PREPARE INPUT

X['text'] = X['text'].apply(lambda x: contractions.fix(x))

X['text'] = X['text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))



def lowerCase(input_str):

    input_str = input_str.lower()

    return input_str



X['text'] = X['text'].apply(lambda x: lowerCase(x))



def lemma(input_str):

    sp = en_core_web_sm.load()

    s = sp(input_str)

    

    input_list = []

    for word in s:

        w = word.lemma_

        input_list.append(w)

        

    output = ' '.join(input_list)

    return output



X['text'] = X['text'].apply(lambda x: lemma(x))

X['text'] = X['text'].str.replace('\xa0', '')
#VECTORIZE

def wordCount(df_column):

    #df_column in df['column_name']

    list_text = df_column.to_list()

    one_string = ' '.join(list_text)

    string_list = list(one_string.split(' '))

    list_unique = list(set(string_list))

    wordcount = len(list_unique)

    

    return wordcount



wordCount(X['text'])



tokenizer = Tokenizer(num_words = 25000, split = ' ')

tokenizer.fit_on_texts(X['text'].values)



X_nlp = tokenizer.texts_to_sequences(X['text'].values)

X_nlp = pad_sequences(X_nlp)



X_meta = X[['has_company_logo_0',

 'has_company_logo_1',

 'has_questions_0',

 'has_questions_1',

 'employment_type_Contract',

 'employment_type_Full-time',

 'employment_type_Other',

 'employment_type_Part-time',

 'employment_type_Temporary',

 'required_experience_Associate',

 'required_experience_Director',

 'required_experience_Entry level',

 'required_experience_Executive',

 'required_experience_Internship',

 'required_experience_Mid-Senior level',

 'required_experience_Not Applicable',

 'required_education_Associate Degree',

 "required_education_Bachelor's Degree",

 'required_education_Certification',

 'required_education_High School or equivalent',

 "required_education_Master's Degree",

 'required_education_Professional',

 'required_education_Some College Coursework Completed',

 'required_education_Unspecified',

 'required_education_Vocational',

 'required_education_Vocational - HS Diploma']]



X_nlp_train, X_nlp_test, y_train, y_test = train_test_split(

        X_nlp, y, test_size=0.2, random_state=42)



X_meta_train, X_meta_test, y_train, y_test = train_test_split(

        X_meta, y, test_size=0.2, random_state=42)



sm = SMOTE(random_state=42)



X_nlp_train, y_nlp_train = sm.fit_sample(X_nlp_train, y_train)

X_meta_train, y_train = sm.fit_sample(X_meta_train, y_train)



X_meta_train = X_meta_train[['has_company_logo_0',

 'has_company_logo_1',

 'has_questions_0',

 'has_questions_1',

 'employment_type_Contract',

 'employment_type_Full-time',

 'employment_type_Other',

 'employment_type_Part-time',

 'employment_type_Temporary',

 'required_experience_Associate',

 'required_experience_Director',

 'required_experience_Entry level',

 'required_experience_Executive',

 'required_experience_Internship',

 'required_experience_Mid-Senior level',

 'required_experience_Not Applicable',

 'required_education_Associate Degree',

 "required_education_Bachelor's Degree",

 'required_education_Certification',

 'required_education_High School or equivalent',

 "required_education_Master's Degree",

 'required_education_Professional',

 'required_education_Some College Coursework Completed',

 'required_education_Unspecified',

 'required_education_Vocational',

 'required_education_Vocational - HS Diploma']].values

                             

X_meta_test = X_meta_test[['has_company_logo_0',

 'has_company_logo_1',

 'has_questions_0',

 'has_questions_1',

 'employment_type_Contract',

 'employment_type_Full-time',

 'employment_type_Other',

 'employment_type_Part-time',

 'employment_type_Temporary',

 'required_experience_Associate',

 'required_experience_Director',

 'required_experience_Entry level',

 'required_experience_Executive',

 'required_experience_Internship',

 'required_experience_Mid-Senior level',

 'required_experience_Not Applicable',

 'required_education_Associate Degree',

 "required_education_Bachelor's Degree",

 'required_education_Certification',

 'required_education_High School or equivalent',

 "required_education_Master's Degree",

 'required_education_Professional',

 'required_education_Some College Coursework Completed',

 'required_education_Unspecified',

 'required_education_Vocational',

 'required_education_Vocational - HS Diploma']].values
input_1 = Input(shape=(X_nlp.shape[1],))

input_2 = Input(shape=(26,))



embedding_layer = Embedding(25000, 128)(input_1)

LSTM_Layer_1 = LSTM(128)(embedding_layer)



dense_layer_1 = Dense(32, activation='relu')(input_2)

#dropout_layer_1 = Dropout(0.2)(dense_layer_1)

dense_layer_2 = Dense(32, activation='relu')(dense_layer_1)

#dropout_layer_2 = Dropout(0.2)(dense_layer_2)



concat_layer = Concatenate()([LSTM_Layer_1, dense_layer_2])

dense_layer_3 = Dense(16, activation='relu')(concat_layer)



output = Dense(1, activation='sigmoid')(dense_layer_3)



model = Model(inputs=[input_1, input_2], outputs=output)



model.compile(loss='binary_crossentropy',

              optimizer='adam',metrics=['accuracy'])
history = model.fit(x=[X_nlp_train, X_meta_train], y=y_train, 

                    epochs = 10, batch_size = 128, verbose = 1,

                    validation_split=0.2)



score = model.evaluate(x=[X_nlp_test, X_meta_test], y=y_test, verbose=1)



print("Test Score:", score[0])
print("Test Accuracy:", score[1])
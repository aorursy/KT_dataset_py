# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
SOC_MINOR_GROUPS = {

    '11-1': 'Top Executives',

    '11-2': 'Advertising, Marketing, Promotions, Public Relations, and Sales Managers',

    '11-3': 'Operations Specialties Managers',

    '11-9': 'Other Management Occupations',

    '13-1': 'Business Operations Specialists',

    '13-2': 'Financial Specialists',

    '15-1': 'Computer Occupations',

    '15-2': 'Mathematical Science Occupations',

    '17-1': 'Architects, Surveyors, and Cartographers',

    '17-2': 'Engineers',

    '17-3': 'Drafters, Engineering Technicians, and Mapping Technicians',

    '19-1': 'Life Scientists',

    '19-2': 'Physical Scientists',

    '19-3': 'Social Scientists and Related Workers',

    '19-4': 'Life, Physical, and Social Science Technicians',

    '21-1': 'Counselors, Social Workers, and Other Community and Social Service Specialists',

    '21-2': 'Religious Workers',

    '23-1': 'Lawyers, Judges, and Related Workers',

    '23-2': 'Legal Support Workers',

    '25-1': 'Postsecondary Teachers',

    '25-2': 'Preschool, Primary, Secondary, and Special Education School Teachers',

    '25-3': 'Other Teachers and Instructors',

    '25-4': 'Librarians, Curators, and Archivists',

    '25-9': 'Other Education, Training, and Library Occupations',

    '27-1': 'Art and Design Workers',

    '27-2': 'Entertainers and Performers, Sports and Related Workers',

    '27-3': 'Media and Communication Workers',

    '27-4': 'Media and Communication Equipment Workers',

    '29-1': 'Health Diagnosing and Treating Practitioners',

    '29-2': 'Health Technologists and Technicians',

    '29-9': 'Other Healthcare Practitioners and Technical Occupations',

    '31-1': 'Nursing, Psychiatric, and Home Health Aides',

    '31-2': 'Occupational Therapy and Physical Therapist Assistants and Aides',

    '31-9': 'Other Healthcare Support Occupations',

    '33-1': 'Supervisors of Protective Service Workers',

    '33-2': 'Fire Fighting and Prevention Workers',

    '33-3': 'Law Enforcement Workers',

    '33-9': 'Other Protective Service Workers',

    '35-1': 'Supervisors of Food Preparation and Serving Workers',

    '35-2': 'Cooks and Food Preparation Workers',

    '35-3': 'Food and Beverage Serving Workers',

    '35-9': 'Other Food Preparation and Serving Related Workers',

    '37-1': 'Supervisors of Building and Grounds Cleaning and Maintenance Workers',

    '37-2': 'Building Cleaning and Pest Control Workers',

    '37-3': 'Grounds Maintenance Workers',

    '39-1': 'Supervisors of Personal Care and Service Workers',

    '39-2': 'Animal Care and Service Workers',

    '39-3': 'Entertainment Attendants and Related Workers',

    '39-4': 'Funeral Service Workers',

    '39-5': 'Personal Appearance Workers',

    '39-6': 'Baggage Porters, Bellhops, and Concierges',

    '39-7': 'Tour and Travel Guides',

    '39-9': 'Other Personal Care and Service Workers',

    '41-1': 'Supervisors of Sales Workers',

    '41-2': 'Retail Sales Workers',

    '41-3': 'Sales Representatives, Services',

    '41-4': 'Sales Representatives, Wholesale and Manufacturing',

    '41-9': 'Other Sales and Related Workers',

    '43-1': 'Supervisors of Office and Administrative Support Workers',

    '43-2': 'Communications Equipment Operators',

    '43-3': 'Financial Clerks',

    '43-4': 'Information and Record Clerks',

    '43-5': 'Material Recording, Scheduling, Dispatching, and Distributing Workers',

    '43-6': 'Secretaries and Administrative Assistants',

    '43-9': 'Other Office and Administrative Support Workers',

    '45-1': 'Supervisors of Farming, Fishing, and Forestry Workers',

    '45-2': 'Agricultural Workers',

    '45-3': 'Fishing and Hunting Workers',

    '45-4': 'Forest, Conservation, and Logging Workers',

    '47-1': 'Supervisors of Construction and Extraction Workers',

    '47-2': 'Construction Trades Workers',

    '47-3': 'Helpers, Construction Trades',

    '47-4': 'Other Construction and Related Workers',

    '47-5': 'Extraction Workers',

    '49-1': 'Supervisors of Installation, Maintenance, and Repair Workers',

    '49-2': 'Electrical and Electronic Equipment Mechanics, Installers, and Repairers',

    '49-3': 'Vehicle and Mobile Equipment Mechanics, Installers, and Repairers',

    '49-9': 'Other Installation, Maintenance, and Repair Occupations',

    '51-1': 'Supervisors of Production Workers',

    '51-2': 'Assemblers and Fabricators',

    '51-3': 'Food Processing Workers',

    '51-4': 'Metal Workers and Plastic Workers',

    '51-5': 'Printing Workers',

    '51-6': 'Textile, Apparel, and Furnishings Workers',

    '51-7': 'Woodworkers',

    '51-8': 'Plant and System Operators',

    '51-9': 'Other Production Occupations',

    '53-1': 'Supervisors of Transportation and Material Moving Workers',

    '53-2': 'Air Transportation Workers',

    '53-3': 'Motor Vehicle Operators',

    '53-4': 'Rail Transportation Workers',

    '53-5': 'Water Transportation Workers',

    '53-6': 'Other Transportation Workers',

    '53-7': 'Material Moving Workers',

    '55-1': 'Military Officer Special and Tactical Operations Leaders',

    '55-2': 'First-Line Enlisted Military Supervisors',

    '55-3': 'Military Enlisted Tactical Operations and Air/Weapons Specialists and Crew Members'

}
from io import StringIO

import requests

import pandas as pd



from random import seed

import numpy as np

seed(42)

np.random.seed(42)



file_url = 'https://www.onetcenter.org/dl_files/database/db_20_1_text/Sample%20of%20Reported%20Titles.txt'

csv = StringIO(requests.get(file_url).text)



# Load it in a pandas DataFrame and drop a useless column

df = pd.read_csv(csv, sep='\t').drop('Shown in My Next Move', axis=1)



# Get the occupation name from the code and remove the original code column

df['SOC minor group'] = df['O*NET-SOC Code'].apply(lambda x: SOC_MINOR_GROUPS[x[:4]])

df.drop('O*NET-SOC Code', axis=1, inplace=True)



# Lower all job titles for simplicity

df['Reported Job Title'] = df['Reported Job Title'].str.lower()



# Display a few examples

df.iloc[[1,2,3,100,101,102,301,302,303]]
# Count number unique each column

df.nunique()
test_set = df.groupby('SOC minor group', as_index=False)['Reported Job Title'].first()

train_set = df[~df['Reported Job Title'].isin(test_set['Reported Job Title'])]



x_train, y_train = train_set['Reported Job Title'], train_set['SOC minor group']

x_test, y_test = test_set['Reported Job Title'], test_set['SOC minor group']
from sklearn.preprocessing import LabelEncoder



classes_encoder = LabelEncoder()

y_train = classes_encoder.fit_transform(y_train)

y_test = classes_encoder.transform(y_test)
!pip install zeugma
# word embedding basic tranfer learning 

from zeugma import EmbeddingTransformer

embedding = EmbeddingTransformer('glove-twitter-100', aggregation='sum')
from sklearn.pipeline import make_pipeline

from sklearn.neighbors import KNeighborsClassifier



clf = KNeighborsClassifier(n_neighbors=1)



baseline = make_pipeline(embedding, clf)
baseline.fit(x_train, y_train)

print('Train accuracy (baseline): {} %'.format(100*baseline.score(x_train, y_train)))



print('Test accuracy (baseline): {} %'.format(100*baseline.score(x_test, y_test)))
import itertools

from random import sample



jobs_left = []

jobs_right = []

target = []



soc_codes = train_set['SOC minor group'].unique()

for code in soc_codes:

    similar_jobs = train_set[train_set['SOC minor group'] == code]['Reported Job Title']

    

    # pick 1000 random pairs from SOC group job titles combinations

    group_pairs = list(itertools.combinations(similar_jobs, 2))

    positive_pairs = sample(group_pairs, 1000) if len(group_pairs) > 1000 else group_pairs

    jobs_left.extend([p[0] for p in positive_pairs])

    jobs_right.extend([p[1] for p in positive_pairs])

    target.extend([1.]* len(positive_pairs))

    

    # negative sample

    other_jobs = train_set[train_set['SOC minor group'] != code]['Reported Job Title']

    for i in range(len(positive_pairs)):

        jobs_left.append(np.random.choice(similar_jobs))

        jobs_right.append(np.random.choice(similar_jobs))

        target.append(0.)



dataset = pd.DataFrame({

    'job_left': jobs_left,

    'job_right': jobs_right,

    'target': target

}).sample(frac=1).drop_duplicates(subset =['job_left', 'job_right'], keep = False, inplace = False)

 # Shuffle dataset

print(dataset.shape)

dataset.tail()
import re

from sklearn.pipeline import make_pipeline, FeatureUnion

from sklearn.preprocessing import FunctionTransformer

from zeugma import TextsToSequences, Padder, ItemSelector



max_words_job_title = 10  # To avoid very long job titles we limit them to 10 words

vocab_size = 10000  # Number of most-frequent words kept in the vocabulary



def preprocess_job_titles(job_titles):

    """ Return a list of clean job titles """

    def preprocess_job_title(raw_job_title):

        """ Clean a single job title"""

        job_title = re.sub(r'\(.*\)', '', raw_job_title)  # Remove everything between parenthesis

        return job_title.lower().strip()

    return [preprocess_job_title(jt) for jt in job_titles]

    

pipeline = make_pipeline(

    FunctionTransformer(preprocess_job_titles, validate=False),  # Preprocess the text

    TextsToSequences(num_words=vocab_size),  # Turn word sequences into indexes sequences

    Padder(max_length=max_words_job_title),  # Pad shorter job titles with a dummy index

)



# Note that the preprocessing pipeline must be fit on both the right and left examples

# simultaneously

pipeline.fit(list(dataset['job_left']) + list(dataset['job_right']));
x_left = pipeline.transform(dataset['job_left'])

x_right = pipeline.transform(dataset['job_right'])

x_pairs = [x_left, x_right]   # this will be the input of the siamese network



y_pairs = dataset['target'].values
# We re-use the same embedding as with the baseline model

embedding_layer = embedding.model.get_keras_embedding()
from keras.layers import LSTM, Bidirectional

from keras import Model, Sequential

from keras.layers import Input, Dense, Dropout, Lambda, Subtract

from keras import backend as K



def exponent_neg_manhattan_distance(arms_difference):

    """ Compute the exponent of the opposite of the L1 norm of a vector, to get the left/right inputs

    similarity from the inputs differences. This function is used to turned the unbounded

    L1 distance to a similarity measure between 0 and 1"""

    return K.exp(-K.sum(K.abs(arms_difference), axis=1, keepdims=True))



def siamese_lstm(max_length, embedding_layer):

    """ Define, compile and return a siamese LSTM model """

    input_shape = (max_length,)

    left_input = Input(input_shape, name='left_input')

    right_input = Input(input_shape, name='right_input')



    # Define a single sequential model for both arms.

    # In this example I've chosen a simple bidirectional LSTM with no dropout

    seq = Sequential(name='sequential_network')

    seq.add(embedding_layer)

    seq.add(Bidirectional(LSTM(32, dropout=0., recurrent_dropout=0.)))

    

    left_output = seq(left_input)

    right_output = seq(right_input)



    # Here we subtract the neuron values of the last layer from the left arm 

    # with the corresponding values from the right arm

    subtracted = Subtract(name='pair_representations_difference')([left_output, right_output])

    malstm_distance = Lambda(exponent_neg_manhattan_distance, 

                             name='masltsm_distance')(subtracted)



    siamese_net = Model(inputs=[left_input, right_input], outputs=malstm_distance)

    siamese_net.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    return siamese_net



siamese_lstm = siamese_lstm(max_words_job_title, embedding_layer)



# Print a summary of the model mainly to know the number of trainable parameters

siamese_lstm.summary()
siamese_lstm.fit(x_pairs, y_pairs, validation_split=0.1, epochs=1)
x_references = pipeline.transform(x_train)  # Preprocess the training set examples



def get_prediction(job_title):

    """ Get the predicted job title category, and the most similar job title

    in the train set. Note that this way of computing a prediction is highly 

    not optimal, but it'll be sufficient for us now. """

    x = pipeline.transform([job_title])

    

    # Compute similarities of the job title with all job titles in the train set

    similarities = siamese_lstm.predict([[x[0]]*len(x_references), x_references])

    most_similar_index = np.argmax(similarities)

    

    # The predicted category is the one of the most similar example from the train set

    prediction = train_set['SOC minor group'].iloc[most_similar_index]

    most_similar_example = train_set['Reported Job Title'].iloc[most_similar_index]

    return prediction, most_similar_example
sample_idx = 1

pred, most_sim = get_prediction(x_test[sample_idx])



print(f'Sampled test job title: {x_test[sample_idx]}')

print(f'True occupation: {test_set["SOC minor group"].iloc[sample_idx]}')

print(f'Occupation prediction: {pred}')

print(f'Most similar example in train set: {most_sim}') 
from sklearn.metrics import accuracy_score



y_pred = [get_prediction(job_title)[0] for job_title in test_set['Reported Job Title']]

accuracy = accuracy_score(classes_encoder.transform(y_pred), y_test)



print(f'Test accuracy (siamese model): {100*accuracy:.2f} %')
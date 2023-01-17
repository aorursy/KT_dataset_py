import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.pipeline import make_pipeline

from sklearn import preprocessing



filename_test = '../input/naivebayesleariningsamples/play_golf_test.csv'

filename_train = '../input/naivebayesleariningsamples/play_golf_train.csv'



print('Setup complete!')
train_df = pd.read_csv(filename_train, delimiter=';')



# build two different sub dataframe of the test_df: 

# the first contains the features and the second contains the expected results

weather_condition = train_df.loc[:, ['Outlook', 'Temperature Numeric', 'Temperature Nominal', 'Humidity Numeric', 'Humidity Nominal', 'Windy']]



expected_res = np.array((train_df['Play'])).tolist()



weather_cond_list = []

outlooks = []

temperatures = []

humidities = []

wind_list = []



for index, row in weather_condition.iterrows():

    outlooks.append(row[0])

    temperatures.append(row[1])

    humidities.append(row[3])

    wind_list.append(row[5])



# Encode all the values 

le = preprocessing.LabelEncoder() # initialize the label encoder

result_le = preprocessing.LabelEncoder() # initialize the result label encoder

encd_outlooks = le.fit_transform(outlooks)

encd_wind_vals = le.fit_transform(wind_list)

labels = result_le.fit_transform(expected_res)



features = np.asarray([sublist for sublist in zip(encd_outlooks, encd_wind_vals, temperatures, humidities)]) 



train_df
model = GaussianNB() # build the model

# now its time to train the model!

model.fit(features, labels)

print('Model Trained!')
# Here's our test dataframe

# the encoding and features extraction might be redudant but I keep them for sake of simplicity

test_df = pd.read_csv(filename_test, delimiter=';')



weather_condition = test_df.loc[:, ['Outlook', 'Temperature Numeric', 'Temperature Nominal', 'Humidity Numeric', 'Humidity Nominal', 'Windy']]



expected_res = np.array((test_df['Play'])).tolist()



weather_cond_list = []

outlooks = []

temperatures = []

humidities = []

wind_list = []



for index, row in weather_condition.iterrows():

    outlooks.append(row[0])

    temperatures.append(row[1])

    humidities.append(row[3])

    wind_list.append(row[5])



# Encode all the values 

encd_outlooks = le.fit_transform(outlooks)

encd_wind_vals = le.fit_transform(wind_list)

true_res = result_le.fit_transform(expected_res)



test_features = np.asarray([sublist for sublist in zip(encd_outlooks, encd_wind_vals, temperatures, humidities)]) 



test_df
from sklearn.metrics import accuracy_score



pred_list = []



for features, expected in zip(test_features, expected_res):

    pred_list.append(model.predict([features]))



print(f'Accuracy: {(accuracy_score(true_res, pred_list))*100}%')
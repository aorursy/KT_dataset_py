%matplotlib inline

import os

import unicodedata

import warnings

warnings.simplefilter(action='ignore')



import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

import sklearn as sl
medium_data = pd.read_csv("../input/medium-articles-dataset/medium_data.csv")
medium_data.head(2)
medium_data.info()
for idx in range(10):

    print(medium_data.title[idx], medium_data.title[idx].split(" "))
def normalize_text(text : str) -> str:

    """ Normalize the unicode string

        :param text: text data

        :retrns clean_text: clean text

    """

    

    if text != np.nan:

        clean_text = unicodedata.normalize("NFKD",

                                           text)

    else:

        clean_text = text

    

    return clean_text
medium_data['clean_title'] = medium_data.title.apply(lambda x: normalize_text(x) if x!= np.nan else x)
medium_data.title[0], medium_data.clean_title[0]
medium_data['clean_subtitle'] = medium_data.subtitle.apply(lambda x: normalize_text(x) if x!= np.nan and type(x) == str else x)
def create_wc(text : str) -> int:

    """ Count words in a text

        :param text: String to check the len

        :retirns wc: Word count

    """

    

    wc = 0

    

    norm_text = text.lower()

    

    wc = len(norm_text.split(" "))

    

    return wc
medium_data.title[0].lower()
medium_data['title_wc'] = medium_data.title.apply(lambda x: create_wc(x) if x!= np.nan else 0)
medium_data['subtitle_wc'] = medium_data.subtitle.apply(lambda x: create_wc(x) if x!= np.nan and type(x) == str else 0)
medium_data.head()
cout_pub_ax = medium_data.publication.value_counts().plot(kind='bar',

                                                        figsize=(10,6),

                                                        rot=35,

                                                        align='center',

                                                        title="Count of Article by Publication")

cout_pub_ax.set_xlabel("Publication")

cout_pub_ax.set_ylabel("Count")
pub_clap_ax = medium_data.groupby(['publication'])['claps'].agg(sum).plot(kind='bar',

                                                                           figsize=(10,6),

                                                                           rot=35,

                                                                          align='center',

                                                                           title="Claps by Publications")

pub_clap_ax.set_xlabel("Publication")

pub_clap_ax.set_ylabel("Count")
medium_data.title_wc.plot(kind='hist',

                         figsize=(10,6),

                         title="Histogram of Title Word Count")
medium_data.subtitle_wc.plot(kind='hist',

                         figsize=(10,6),

                         title="Histogram of Sub Title Word Count")
medium_data.reading_time.plot(kind='hist',

                         figsize=(10,6),

                         title="Histogram of Reading Time")
model_data = medium_data[['publication','title_wc','subtitle_wc','reading_time','claps']]

model_data.head()
publications_cat = pd.get_dummies(model_data.publication)
model_data.reading_time.clip(lower=1,upper=15,inplace=True)
model_data.reading_time.plot(kind='hist',

                         figsize=(10,6),

                         title="Histogram of Reading Time after Clip")
#model_data.drop('publication',

#                inplace=True,

#               axis=1)

#model_data.head(2)
model_data_treated = pd.concat([publications_cat,model_data],

                              axis=1,

                               sort=False)

model_data_treated.head(2)
from sklearn.model_selection import train_test_split
train,test = train_test_split(model_data_treated,

                              test_size=0.3,

                             stratify=model_data_treated['publication'])
train.drop('publication',

                inplace=True,

               axis=1)

train.head(2)
test.drop('publication',

                inplace=True,

               axis=1)

test.head(2)
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
train_x = train[train.columns[:-1]]

train_y = train[['claps']]
_ = rf_model.fit(train_x,

            train_y)
test_x = test[test.columns[:-1]]

test_y = test[['claps']]
predictions = rf_model.predict(test_x)
test["prediction"] = predictions
sl.metrics.mean_squared_error(test.claps, test.prediction)
y = test.claps.values

fig, ax = plt.subplots(figsize=(10,6))

ax.scatter(y, predictions)

ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

from tpot import TPOTRegressor
automl_reg = TPOTRegressor(generations=10,

                          population_size=100,

                          verbosity=2,

                          random_state=2020,

                          early_stop=3)
automl_reg.fit(train_x,

            train_y)
automl_predict = automl_reg.predict(test_x)
sl.metrics.mean_squared_error(test.claps, automl_predict)
y = test.claps.values

fig, ax = plt.subplots(figsize=(10,6))

ax.scatter(y, automl_predict)

ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')
test["automl_predict"] = automl_predict
test.head(10)
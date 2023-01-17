# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from fastai.text import * 

from sklearn.model_selection import train_test_split
from langdetect import detect
!pip install googletrans
from googletrans import Translator
path = '/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/'
pmc_custom_license_df = pd.read_csv(path + "clean_pmc.csv")

noncomm_use_df = pd.read_csv(path + "clean_noncomm_use.csv")

comm_use_df = pd.read_csv(path + "clean_comm_use.csv")

biorxiv = pd.read_csv(path + "biorxiv_clean.csv")
data = pd.concat([pmc_custom_license_df, noncomm_use_df, comm_use_df, biorxiv], axis =0)
covid_papers = data[data.text.str.contains('COVID-19|SARS-CoV-2|2019-nCov|SARS Coronavirus 2|2019 Novel Coronavirus')][['paper_id', 'text']]
data = covid_papers
data['language'] = data['text'].map(detect)
data.shape
data['language'].value_counts()
indices_to_translate = data[data['language']=='es']['text'].index
translator = Translator()
#3,900 character limit for google translate
for i in indices_to_translate:



    if (len(data.loc[i, 'text']) > 3900):

        data.loc[i, 'text'] = data.loc[i, 'text'][:3900]

    paper = translator.translate(data.loc[i, 'text'])

    data.loc[i, 'text'] = paper.text

data.loc[indices_to_translate, 'text']
data.to_csv('data_covid19_papers.csv')
train, validation = train_test_split(data, test_size=0.1, random_state=42, shuffle= True)
data_lm = TextLMDataBunch.from_df('.',  train, validation,

                                  label_cols='text')
data_lm.save('data_lm_export.pkl')
data_lm = load_data('.', 'data_lm_export.pkl')
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4, 2e-2) 
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(10, slice(1e-4, 1e-3))
learn.save_encoder('ft_enc')
np.random.seed(42)
learn.predict("risks factors", n_words=100, temperature=0.1)
learn.predict("smoking", n_words=100, no_unk=True, temperature=0.1)
learn.predict("pre-existing pulmonary disease", n_words=50, no_unk=True, temperature=0.05)
learn.predict("Co-infections", n_words=80, no_unk=True, temperature=0.1)
learn.predict("comorbidities", n_words=50, no_unk=True, temperature=0.1)
learn.predict("Neonates", n_words=50, no_unk=True, temperature=0.1) 
learn.predict("pregnant women", n_words=50, no_unk=True, temperature=0.1)
learn.predict("Socio-economic factors", n_words=100, no_unk=True, temperature=0.1)  
learn.predict("behavioral factors", n_words=80, no_unk=True, temperature=0.1)
learn.predict("Transmission dynamics of the virus", n_words=50, no_unk=True, temperature=0.1) 
learn.predict("basic reproductive number", n_words=80, no_unk=True, temperature=0.1)
learn.predict("incubation period", n_words=50, no_unk=True, temperature=0.1)
learn.predict("serial interval", n_words=50, no_unk=True, temperature=0.1) 
learn.predict("environmental factors", n_words=50, no_unk=True, temperature=0.1) 
learn.predict("modes of transmission", n_words=50, no_unk=True, temperature=0.1) 
learn.predict("Severity of disease", n_words=50, no_unk=True, temperature=0.1)
learn.predict("risk of fatality among symptomatic hospitalized patients", n_words=50, no_unk=True, temperature=0.1)
learn.predict("high-risk patient", n_words=50, no_unk=True, temperature=0.1)
learn.predict("Susceptibility of populations", n_words=50, no_unk=True, temperature=0.1)
learn.predict("Public health mitigation measures", n_words=50, no_unk=True, temperature=0.1) 
learn.predict("What do we know about COVID-19 risk factors?", n_words=100, no_unk=True, temperature=0.1)
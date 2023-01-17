# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('/kaggle/input/hta-tagging/train.csv')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
texts = []
for i in range(len(df['Filename'])):
    fn = df['Filename'][i]
    folder = fn.split("-")[0]
    f = open('/kaggle/input/hta-tagging/train-data/train-data/'+folder+'/'+fn, "r")
    texts.append(f.read())

df['text'] = texts
df
intervention = df.copy().drop(['Blinding of Outcome assessment','Classes'], axis=1)
outcome = df.copy().drop(['Blinding of intervention','Classes'], axis=1)
intervention
outcome
!pip install pycaret

from pycaret.nlp import *
inter_exp = setup (data = intervention, target='text', session_id=23) 
# interest to try on other topic modeling algorithms??: https://github.com/pycaret/pycaret/blob/master/nlp.py#L549
lda = create_model('lda')
lda
lda_results = assign_model(lda)
lda_results
plot_model(lda)
plot_model(lda, plot = 'bigram')
plot_model(lda, plot = 'topic_distribution')
plot_model(lda, plot = 'tsne')
# plot_model(lda, plot = 'umap')

evaluate_model(lda)
from pycaret.classification import *
lda_data = lda_results.drop(['text','Dominant_Topic','Perc_Dominant_Topic'], axis=1)
lda_data
inter_class = setup(data = lda_data, ignore_features=['Filename'], target='Blinding of intervention', train_size=0.8, session_id = 24)
%time compare_models()
# View the abbrevation of model names here: https://github.com/pycaret/pycaret/blob/master/classification.py#L3022
%time catboost = create_model("catboost")
%time catboost_tune = tune_model("catboost")
# ensemble_catboost = ensemble_model(catboost, method="Bagging") #or Boosting
# rf = create_model('rf')
# xgboost = create_model('xgboost')
# blend_models = blend_models(estimateor_list = [catboost, rf, xgboost])
plot_model(catboost, plot='confusion_matrix')
evaluate_model(catboost)
validation_set = predict_model(catboost)
ready = finalize_model(catboost)
# Don't forget that  N: 0, P: 1, Q: 2
result = predict_model(ready, data=lda_data)#.drop("Blinding of intervention", axis=1))
result
result['Label'] = result['Label'].replace([0,1,2],['N','P','Q'])
result
#Assume that Blinding of interventaion and Blinding of outcome assessement is identical!!!
result['Prediction'] = result['Label']+result['Label']
result['Id'] = result['Filename']
out = result[['Id', 'Prediction']]
out
out.to_csv("out.csv", index=False)

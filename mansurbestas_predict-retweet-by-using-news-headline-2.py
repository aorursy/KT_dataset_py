# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install pycaret
import pandas as pd
import numpy as np
import pycaret
from pycaret.regression import *
retweet = pd.read_csv("../input/turkishnewstwitter/retweet_kaggle.csv")
giris = pd.read_csv("../input/turkishnewstwitter/giris_kaggle.csv")
X = giris.iloc[:,1:]
y = retweet.iloc[:,-1]
y = pd.DataFrame(y)
dataset = pd.concat([X,y], axis=1)
#dataset=dataset[dataset["retweet_toplam"]<30]
#dataset=dataset[dataset["wiki_deger"]<45000]
dataset
data = dataset.sample(frac=0.9, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)
data=data[data["retweet_toplam"]<30]
data=data[data["wiki_deger"]<45000]

data_unseen=data_unseen[data_unseen["retweet_toplam"]<30]
data_unseen=data_unseen[data_unseen["wiki_deger"]<45000]
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))
exp_reg101 = setup(data = data, target = 'retweet_toplam', session_id=123)
compare_models()
ada = create_model('ada')
print(ada)
lightgbm = create_model('lightgbm')
dt = create_model('dt')
tuned_ada = tune_model('ada')
print(tuned_ada)
tuned_lightgbm = tune_model('lightgbm')
tuned_dt = tune_model('dt')
plot_model(tuned_lightgbm)
plot_model(tuned_lightgbm, plot = 'error')
plot_model(tuned_lightgbm, plot='feature')
evaluate_model(tuned_lightgbm)
predict_model(tuned_lightgbm);
final_lightgbm = finalize_model(tuned_lightgbm)
print(final_lightgbm)
predict_model(final_lightgbm);
unseen_predictions = predict_model(final_lightgbm, data=data_unseen)
unseen_predictions.head()
save_model(final_lightgbm,'Final Lightgbm Model 08Feb2020')
saved_final_lightgbm = load_model('Final Lightgbm Model 08Feb2020')
new_prediction = predict_model(saved_final_lightgbm, data=data_unseen)
new_prediction.head()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/fetal-health-classification/fetal_health.csv')

data.head()
data.fetal_health.value_counts()
data.isnull().sum()
health_map = {

    1: 'Normal',

    2: 'Suspect',

    3: 'Pathological'

}



for i in data.index:

    data.loc[i,'fetal_health'] = health_map[data.loc[i, 'fetal_health']]

data.head(20)
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split as TTS



X = data.drop(columns = 'fetal_health')

y = data.fetal_health



Xtrain, Xtest, ytrain, ytest = TTS(X,y, test_size = .3, random_state = 2601744, stratify = y, shuffle = True)

#Used the recommended test size of %30
from catboost import CatBoostClassifier



cat = CatBoostClassifier(learning_rate = 0.03, l2_leaf_reg = 1,

                        iterations = 500, depth = 9,

                        border_count = 20, eval_metric = 'AUC')



cat = cat.fit(Xtrain, ytrain,

             eval_set = (Xtest, ytest),

             early_stopping_rounds = 70, verbose = 20)
from sklearn.metrics import plot_confusion_matrix as PCM



PCM(cat, Xtest, ytest, labels = ['Pathological', 'Suspect', 'Normal'], normalize = 'pred',

   cmap = 'Greens', include_values = True, xticks_rotation = 30)

plt.title('Confusion Matrix by Prediciton', fontdict = {'fontsize': 18}, pad = 15)
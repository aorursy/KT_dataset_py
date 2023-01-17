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
from pycaret.nlp import *
train = pd.read_csv('/kaggle/input/sample_data_123.csv')
train.head()
text_list = list(train['body'])

type(text_list)
nlp_list = setup(data = text_list, session_id = 17)
lda2 = create_model('lda', num_topics = 6, multi_core = True)
lda_results = assign_model(lda2)

lda_results.head()
evaluate_model(lda2)
plot_model(lda2, plot = 'topic_distribution')
plot_model(lda2, plot = 'topic_model')
plot_model(lda2, plot = 'umap')
plot_model(lda2, plot = 'tsne')
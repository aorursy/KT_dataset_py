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
from pycaret.nlp import *
data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
data.head()
data = data[['text', 'target']]
data.head()
setup(data=data, target='text')
exp = create_model(model='lda', num_topics = 3, multi_core=True)
lda_data = assign_model(exp)
lda_data.head()
lda_data.drop(['text', 'Dominant_Topic', 'Perc_Dominant_Topic'], axis=1, inplace=True)
lda_data.head()
from pycaret.classification import * 
exp2 = setup(data=lda_data, target='target')
compare_models()
xgboost = create_model('xgboost')
interpret_model(xgboost)
finalize_model(xgboost)
save_model('xgboost', 'xgb_basic')
from pycaret.nlp import *

test_raw = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
test_raw = test_raw[['id','text']]
setup(data=test_raw, target='text')
lda_test_data = assign_model(exp)
lda_test_data.drop(['text', 'Dominant_Topic', 'Perc_Dominant_Topic'], axis=1, inplace=True)
predictions = predict_model(xgboost, data=lda_test_data)
predictions.head()
output = predictions[['id', 'Label']]
output = output.rename(columns={'Label': 'target'})
output.head()
output.to_csv('output.csv', index=False)
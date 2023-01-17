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
!pip install pycaret-nightly==0.20
from pycaret.datasets import get_data

data = get_data('juice')
from pycaret.classification import *

clf1 = setup(data, target = 'Purchase', silent=True)
t1 = compare_models()
rf = create_model('rf')
tuned_rf = tune_model(rf)
dt = create_model('dt')
ensembled_dt = ensemble_model(dt)
xgboost = create_model('xgboost', fold=5)
dt = create_model(dt, ensemble=True, method = 'Boosting')
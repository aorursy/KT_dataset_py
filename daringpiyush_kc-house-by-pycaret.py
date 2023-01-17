# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
## Load Kc house price data
data = pd.read_csv('/kaggle/input/kc-housesales-data/kc_house_data.csv')
data.head()
# Install pycaret library
!pip install pycaret
# Initiallize the data and fuctioning : Once initialize press 'Y'
from pycaret.regression import *
clf1 = setup(data = data, target = 'price')
# Compare performance of the model through various regression model on various comparision metrics
compare_models()
# Check Linear reg model's performance on 10 kfold CV 
lr = create_model('lr')
# Plot the model
plot_model(lr)
# ensembling Linear regression model (boosting)
lr_boosted = ensemble_model(lr, method = 'Boosting')
# evaluate a model 
evaluate_model(lr_boosted)
# generate predictions on holdout
lr_predictions_holdout = predict_model(lr_boosted)
# finalize model
lr_final = finalize_model(lr_boosted)
# deploy model
deploy_model(model = lr_final, model_name = 'deploy_lr', platform ='flask', authentication = {'bucket' : 'pycaret-test'})

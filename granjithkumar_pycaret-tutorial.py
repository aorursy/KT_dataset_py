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
train  = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')
!pip install pycaret
from pandas_profiling import ProfileReport
from pycaret.classification import *
train_report = ProfileReport(train)
train_report
train_setup = setup(train,target ='price_range')
setup_data = setup(data = train,target = 'price_range',train_size =0.8,categorical_features =['blue','dual_sim','four_g','three_g','touch_screen','wifi'],normalize = True,normalize_method = 'zscore',remove_multicollinearity = True,multicollinearity_threshold = 0.8,pca =True, pca_method ='linear',pca_components = 0.90,ignore_low_variance = True)
setup_data = setup(data = train,target = 'price_range',train_size =0.8,categorical_features =['blue','dual_sim','four_g','three_g','touch_screen','wifi'],normalize = True,normalize_method = 'zscore',remove_multicollinearity = True,multicollinearity_threshold = 0.8,pca =True, pca_method ='linear',pca_components = 0.90,ignore_low_variance = True,numeric_features =['talk_time','fc','n_cores','sc_h','sc_w'])
compare_models()
final_model = create_model('qda')
plot_model(final_model,plot='auc')
plot_model(final_model,plot='pr')
plot_model(final_model,plot='class_report')
plot_model(final_model,plot='learning')
plot_model(final_model,plot='parameter')
evaluate_model(final_model)
final_predictions = predict_model(final_model)
final_predictions
finalize_model(final_model)
save_model(final_model,'final_model')
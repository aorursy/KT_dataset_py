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
#from pycaret.datasets import get_data
df=pd.read_csv(r'/kaggle/input/heart-disease-uci/heart.csv')
df.head()
from pycaret.classification import *
clf=setup(df,target='target')
compare_models()
l=create_model('lr')
dt=create_model('dt')
##tunned_lr=tune_model('lr')
## The latest version of pycaret has made some changes in the tune model function 
## So let's explore more before performing this step
boosted_dt=ensemble_model(dt,method='Boosting')
lr=create_model('lr')
lda=create_model('lda')
gbc=create_model('gbc')
blender=blend_models(estimator_list=[lr,lda,gbc],method='soft')
blender.estimators_
plot_model(blender)
plot_model(blender,plot='confusion_matrix')
plot_model(blender,plot='threshold')
plot_model(blender,plot='boundary')
interpret_model(dt)
interpret_model(dt,plot='correlation')
##deploy_model(dt,model_name='dt-for-aws',platform='aws',authentication={'bucket':'pycaret-test'})
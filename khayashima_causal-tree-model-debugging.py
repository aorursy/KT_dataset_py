# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import Image

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install causalml
train_data = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
train_target = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
from causalml.dataset import make_uplift_classification
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.tree import uplift_tree_string, uplift_tree_plot
from causalml.inference.meta import BaseXClassifier
from xgboost import XGBClassifier,XGBRegressor
train_data['cp_type'].unique()
#uplift_model = UpliftTreeClassifier(max_depth = 4, min_samples_leaf = 200, min_samples_treatment = 50, n_reg = 100, evaluationFunction='KL', control_name='ctl_vehicle')
train_X = train_data.drop(columns = ['sig_id','cp_type','cp_dose']).values
treatment = train_data['cp_type'].values
y=train_target['acat_inhibitor'].values
xl = BaseXClassifier(effect_learner=XGBClassifier(random_state = 42,scale_pos_weight = len(y[y==0]) / len(y[y==1])))

uplift_model.fit(train_data.drop(columns = ['sig_id','cp_type','cp_dose']).values,\
                                 treatment=train_data['cp_type'].values,y=train_target['acat_inhibitor'].values)
train_data.drop(columns = ['sig_id','cp_type','cp_dose']).shape
graph = uplift_tree_plot(uplift_model.fitted_uplift_tree,train_data.drop(columns = ['sig_id','cp_type','cp_dose']).columns)
Image(graph.create_png())


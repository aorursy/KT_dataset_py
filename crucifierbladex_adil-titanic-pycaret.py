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
df=pd.read_csv('../input/titanic/Titanic.csv')

df.head()
df.shape
df.describe()
df.info()
!pip install pycaret
data = df.sample(frac=0.95, random_state=786)

data_unseen = df.drop(data.index)

data.reset_index(inplace=True, drop=True)

data_unseen.reset_index(inplace=True, drop=True)

print('Data for Modeling: ' + str(data.shape))

print('Unseen Data For Predictions: ' + str(data_unseen.shape))
from pycaret.classification import *

clf=setup(data=df,target='survived')
compare_models()
model=create_model('lr')

tuned_model=tune_model(model)
plot_model(tuned_model,plot='auc')
plot_model(tuned_model,plot='feature')
plot_model(tuned_model,plot='confusion_matrix')
evaluate_model(tuned_model)
predict_model(tuned_modelmodel)
end_stage_model=finalize_model(tuned_model)

predict_model(end_stage_model)
y_pred=predict_model(end_stage_model,data=data_unseen)

y_pred.head()
from pycaret.utils import check_metric

check_metric(y_pred.survived,y_pred.Label,'R2')
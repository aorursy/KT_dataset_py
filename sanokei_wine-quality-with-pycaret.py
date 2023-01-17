import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
!pip install pycaret
from pycaret.classification import *
df.head()
clf1 = setup(data = df, 
             target = 'quality',
             numeric_imputation = 'mean',
             silent = True)
compare_models()
et  = create_model('et')      
plot_model(estimator = et, plot = 'learning')
plot_model(estimator = et, plot = 'auc')
plot_model(estimator = et, plot = 'confusion_matrix')
plot_model(estimator = et, plot = 'feature')
interpret_model(et)
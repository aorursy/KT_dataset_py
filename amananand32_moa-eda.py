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
import plotly.express as px
import pandas as pd
from IPython.display import display
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error,accuracy_score
from xgboost import XGBRegressor
df_train = pd.read_csv('../input/lish-moa/train_features.csv')
df_test = pd.read_csv('../input/lish-moa/test_features.csv')
df_train_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
df_train_unscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
df_train[:6]
df_train['cp_type'].value_counts()
df_train_scored[:5]
df_train_unscored[:6]
from pandas_profiling import ProfileReport
pd.set_option('display.max_columns', None)
df_train.head(5)
ProfileReport(df_train.iloc[:,4:776],minimal =True)
ProfileReport(df_train.iloc[:,776:],minimal =True)

df_train.cp_type.value_counts(normalize=True)
df_train.cp_time.value_counts(normalize=True)
df_train.cp_dose.value_counts(normalize=True)
#### Target values distribution¶

plt.hist(df_train_scored.mean())
plt.title('Distribution of mean target in each target column');
df_train_scored.describe()
df_train_scored.mean().min(),df_train_scored.mean().max(),df_train_scored.mean().mean()
#We have a high imbalance as  the max target rate is 0.03, the min is very low
df_train[:2]
plt.plot(df_train.loc[df_train['sig_id'] == 'id_000644bb2',[col for col in df_train if 'g-' in col]].values.reshape(-1,1));
plt.title('g- value of id_000644bb2');
plt.plot(sorted(df_train.loc[df_train['sig_id'] == 'id_000644bb2', [col for col in df_train if 'g-' in col]].values.reshape(-1, 1)))
plt.title('sorted g- value of id_000644bb2');
#Sample
sample = pd.read_csv("../input/lish-moa/sample_submission.csv")

#Test
test_features = pd.read_csv("../input/lish-moa/test_features.csv",index_col='sig_id')

#Train
train_features = pd.read_csv("../input/lish-moa/train_features.csv",index_col='sig_id')
train_nonscore = pd.read_csv("../input/lish-moa/train_targets_nonscored.csv",index_col='sig_id')
train_score = pd.read_csv("../input/lish-moa/train_targets_scored.csv",index_col='sig_id')
g_features = [feature for feature in train_features.columns if feature.startswith('g-')]
c_features = [feature for feature in train_features.columns if feature.startswith('c-')]
other_features = [feature for feature in train_features.columns if feature not in g_features and feature not in c_features]
                                                            

print(f'Number of g- Features: {len(g_features)}')
print(f'Number of c- Features: {len(c_features)}')
print(f'Number of other Features: {len(other_features)} ({other_features})')
cols = train_score.columns
submission = pd.DataFrame({'sig_id': test_features.index})
total_loss = 0

SEED = 42
y = train_score[column]
print(y)
for c, column in enumerate(cols,1):
    
    y = train_score[column]
    
    # Split
    X_train, X_valid, y_train, y_valid = train_test_split(train_features, y, train_size=0.9, test_size=0.1, random_state=SEED)
   
    X_test = test_features.copy()

    # One-hot encoding
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    X_valid = pd.get_dummies(X_valid)
    
    X_train, X_test = X_train.align(X_test, join='left', axis=1)
    X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
    
    model = XGBRegressor(
                         tree_method = 'gpu_hist',
                         min_child_weight = 31.580,
                         learning_rate = 0.055,
                         colsample_bytree = 0.655,
                         gamma = 3.705,
                         max_delta_step = 2.080,
                         max_depth = 25,
                         n_estimators = 170,
                         #subsample =  0.864, 
                         subsample =  0.910,
                         booster='dart',
                         validate_parameters = True,
                         grow_policy = 'depthwise',
                         predictor = 'gpu_predictor'
                              
                        )
                        
    # Train Model
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    
    
    # Loss
    mae = mean_absolute_error(y_valid,pred)
    mdae = median_absolute_error(y_valid,pred)
    mse = mean_squared_error(y_valid,pred)
    
    total_loss += mae
    
    # Prediction
    predictions = model.predict(X_test)
    submission[column] = predictions
    
    print("Regressing through col-"+str(c)+", Mean Abs Error: "+str(mae)+", Median Abs Error: "+str(mdae)+", Mean Sqrd Error: "+str(mse))
    
    
submission.to_csv('submission.csv', index=False)
print("Loss: ", total_loss/206)

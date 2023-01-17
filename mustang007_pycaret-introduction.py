import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import  auc, roc_curve, classification_report 

from lightgbm import LGBMClassifier, plot_importance
train = pd.read_csv('/kaggle/input/janatahack-crosssell-prediction/train.csv')
test = pd.read_csv('/kaggle/input/janatahack-crosssell-prediction/test.csv')

data = pd.concat([train,test], axis=0)
data.head()
gender_bias= {
'Male' : 0,
'Female' : 1
}


vehicle = { '< 1 Year' :0,
'1-2 Year' : 1,
'> 2 Years' : 2}


vehicle_damage = { 'No' : 0,
'Yes' : 1}
data['Gender'] = data['Gender'].map(gender_bias)
data['Vehicle_Age'] = data['Vehicle_Age'].map(vehicle)
data['Vehicle_Damage'] = data['Vehicle_Damage'].map(vehicle_damage)
group_vars = ['Region_Code', 'Policy_Sales_Channel']

agg_vars = ['Annual_Premium', 'Vintage', 'Age']


for g in group_vars:
    for a in agg_vars:
        data[f'{g}_{a}_count'] = data.groupby(data[g])[a].transform('count')
        data[f'{g}_{a}_mean'] = data.groupby(data[g])[a].transform('mean')
        data[f'{g}_{a}_std'] = data.groupby(data[g])[a].transform('std')
        data[f'{g}_{a}_min'] = data.groupby(data[g])[a].transform('min')
        data[f'{g}_{a}_max'] = data.groupby(data[g])[a].transform('max')
! pip install pycaret
import pycaret

X = data.iloc[:len(train)]
Y = data.iloc[len(train):]
X['Response'].tail()
X.fillna(method='ffill', inplace=True)
X.isnull().sum()
from pycaret.classification import *
X
df = X.drop(columns='id')
df.Response = df.Response.astype('int')
df.Response 
session_1 = setup(data=df, target='Response', log_experiment=True)
best_model = compare_models()
models()
best_model = create_model('rf')
best_model
tuned_gbc = tune_model(best_model)
tuned_gbc
plot_model(tuned_gbc)
plot_model(tuned_gbc, plot= 'boundary')
from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy='minority', random_state=55, k_neighbors=5)

session_2 = setup(data=df, target='Response', log_experiment=False, normalize=True, normalize_method='zscore', transformation=True, transformation_method='quantile',
                 fix_imbalance=True, fix_imbalance_method = sm)
best_model = create_model('catboost')
tuned_Cat = tune_model(best_model)
lightgbm_model = create_model('lightgbm')
lightgbm_tuned = tune_model(lightgbm_model)
blend = blend_models(estimator_list = [tuned_Cat, lightgbm_tuned], method='soft')
plot_model(blend)
blend
plot_model(blend, plot= 'confusion_matrix')
plot_model(blend, plot= 'error')
plot_model(blend, plot= 'boundary')
Final = Y.drop(columns=['id','Response'])
# generate predictions on unseen data
predictions = predict_model(blend, data = Final)
predictions
result=pd.DataFrame(Y["id"],columns=["id","Response"])
result["Response"]=predictions['Score']
result.to_csv("LGBM_prediction.csv",index=0)
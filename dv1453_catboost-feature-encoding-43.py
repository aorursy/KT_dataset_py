import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
from sklearn.model_selection import train_test_split, KFold
import xgboost as xb
import lightgbm as lbm
from catboost import Pool, CatBoostClassifier, CatBoost
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from collections import defaultdict
imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
train = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/train_data.csv')
test = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/test_data.csv')
train.isnull().sum()
# type_a = train[train['Hospital_code']==4]
# type_a.nunique()
train.nunique()
train_x = train.drop('Stay', axis=1)
train_y = train['Stay']
test_x = test
train_x.shape, train_y.shape, test_x.shape
le = LabelEncoder()
train_y = le.fit_transform(train_y)
df = train_x.append(test_x)
df["Bed Grade"] = imputer.fit_transform(df[["Bed Grade"]]).ravel()
df["City_Code_Patient"] = imputer.fit_transform(df[["City_Code_Patient"]]).ravel()
df['grouped'] = df['Hospital_code'].astype(str) + df['Hospital_type_code'] + df['City_Code_Hospital'].astype(str)\
                     + df['Hospital_region_code'] + df['Ward_Facility_Code']
df.drop(['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital', 'Hospital_region_code', 
        'Ward_Facility_Code', 'case_id', 'patientid'], axis=1, inplace=True)
categorical_features_names = ['Available Extra Rooms in Hospital', 'Department', 'Ward_Type',
       'Bed Grade', 'City_Code_Patient', 'Type of Admission',
       'Severity of Illness', 'Visitors with Patient', 'Age', 'grouped']
# df[categorical_features_names] = df[categorical_features_names].astype(str)
le2 = LabelEncoder()
for col in categorical_features_names:
    df[col] = le2.fit_transform(df[col])
transformer = RobustScaler(quantile_range=(25, 75))
df[['Available Extra Rooms in Hospital', 'Admission_Deposit']] =  \
    transformer.fit_transform(df[['Available Extra Rooms in Hospital', 'Admission_Deposit']])
df.tail()
train_df = df.iloc[:318438, :]
test_df = df.iloc[318438:, :]
train_df['Stay'] = train_y
cat_features = [1,2,3,4,5,6,7,8,10]
# from sklearn.utils import class_weight
# class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.unique(train_df['Stay']),
#                                                  train_df['Stay'])
# class_weights
model = CatBoostClassifier(loss_function="MultiClass",
                           eval_metric="Accuracy",
                           task_type="GPU",
                           learning_rate=0.01,
                           iterations=20000,
                           l2_leaf_reg=50,
                           random_seed=432013,
                           od_type="Iter",
                           depth=8,
                           early_stopping_rounds=15000,
                           border_count=100, 
                           one_hot_max_size=50 
#                            class_weights = class_weights
                           #has_time= True 
                          )
n_split = 10
kf = KFold(n_splits=n_split, random_state=432013, shuffle=True)
# train_df.head()
for idx, (train_index, valid_index) in enumerate(kf.split(train_df)):
    y_train, y_valid = train_df.Stay.iloc[train_index], train_df.Stay.iloc[valid_index]
    X_train, X_valid = train_df.drop('Stay', 1).iloc[train_index,:], train_df.drop('Stay', 1).iloc[valid_index,:]
    _train = Pool(X_train, label=y_train, cat_features=cat_features)
    _valid = Pool(X_valid, label=y_valid, cat_features=cat_features)
    print( "\nFold ", idx)
    fit_model = model.fit(_train,
                          eval_set=_valid,
                          use_best_model=True,
                          verbose=2000,
                         )
model.get_best_score()
test_dataset = Pool(test_df, cat_features=cat_features)
y_pred = model.predict(test_dataset)
classes = le.inverse_transform(y_pred)
np.unique(classes)
output = pd.DataFrame(test['case_id'].values,columns=['case_id'])
output['Stay'] = classes
output.head()
output.to_csv('Catboost_cv.csv',index=False)
y_probabilites = model.predict_proba(test_dataset)
class_prob = pd.DataFrame(y_probabilites)
class_prob
class_prob.to_csv('Class_prob_catboost1.csv', index=False)

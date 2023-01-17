import pandas as pd
import numpy as np

import random , os 
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import SGDRegressor, BayesianRidge
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
import warnings 
warnings.simplefilter('ignore')
!pip install catboost
from catboost import CatBoostClassifier
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
sub = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
train.isnull().sum()
train['age_approx'].fillna(train['age_approx'].mean(),inplace = True)
train['sex'].fillna('unknown_sex',inplace = True)

train['anatom_site_general_challenge'].fillna('unknown_anatom',inplace = True)
test['anatom_site_general_challenge'].fillna('unknown_anatom',inplace = True)
one_hot_anatom = pd.get_dummies(train.anatom_site_general_challenge ,prefix = 'anatom')
train = train.join(one_hot_anatom)

one_hot_anatom = pd.get_dummies(test.anatom_site_general_challenge ,prefix = 'anatom')
test = test.join(one_hot_anatom)
'''
one_hot_sex = pd.get_dummies(train.sex, prefix='sex')
#one_hot_diagnosis = pd.get_dummies(train.diagnosis , prefix = 'disgnosis')

train = train.join(one_hot_sex)
#train = train.join(one_hot_diagnosis)

train['id'] = train['patient_id'].map(lambda x : int(x[3:]))'''
'''
one_hot_sex = pd.get_dummies(test.sex, prefix='sex')
test = test.join(one_hot_sex)
test['id'] = test['patient_id'].map(lambda x : int(x[3:]))'''
'''train.drop(['sex','diagnosis','anatom_site_general_challenge','benign_malignant','image_name','patient_id'],axis = 1,inplace = True)
test.drop(['sex','anatom_site_general_challenge','image_name','patient_id'],axis = 1,inplace = True)
train.drop(['sex_unknown_sex','anatom_unknown_anatom'],axis=1, inplace = True)'''
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()

train['sex_encoding'] = le.fit_transform(train['sex'].astype(str))
test['sex_encoding'] = le.transform(test['sex'].astype(str))

train['anatom_site_general_challenge_encoding'] = le.fit_transform(train['anatom_site_general_challenge'].astype(str))
test['anatom_site_general_challenge_encoding'] = le.transform(test['anatom_site_general_challenge'].astype(str))
train['n_images'] = train['patient_id'].map(train.groupby(['patient_id']).image_name.count())
test['n_images'] = test['patient_id'].map(test.groupby(['patient_id']).image_name.count())
from sklearn.preprocessing import KBinsDiscretizer

categorize = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'uniform')
train['n_images_encoding'] = categorize.fit_transform(train['n_images'].values.reshape(-1,1)).astype(int).squeeze()
test['n_images_encoding'] = categorize.transform(test['n_images'].values.reshape(-1,1)).astype(int).squeeze()
from sklearn.preprocessing import LabelEncoder 

enc = LabelEncoder()

train['age_enc'] = enc.fit_transform(train['age_approx'].astype('str'))
test['age_enc'] = enc.fit_transform(test['age_approx'].astype('str'))
train_images = train['image_name'].values
train_sizes = np.zeros(train.shape[0])

for i, img_path in enumerate(tqdm(train_images)) :
    train_sizes[i] = os.path.getsize(os.path.join('../input/siim-isic-melanoma-classification/jpeg/train/',f'{img_path}.jpg'))
    
train['image_size'] = train_sizes

test_images = test['image_name'].values
test_sizes = np.zeros(test.shape[0])

for i, img_path in enumerate(tqdm(test_images)) :
    test_sizes[i] = os.path.getsize(os.path.join('../input/siim-isic-melanoma-classification/jpeg/test/',f'{img_path}.jpg'))

test['image_size'] = test_sizes
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

train['image_size_scaled'] = scaler.fit_transform(train['image_size'].values.reshape(-1,1))
test['image_size_scaled'] = scaler.transform(test['image_size'].values.reshape(-1,1))
from sklearn.preprocessing import KBinsDiscretizer 

categorize = KBinsDiscretizer(n_bins = 10,encode = 'ordinal' , strategy = 'uniform')

train['image_size_encoding'] = categorize.fit_transform(train.image_size_scaled.values.reshape(-1,1)).astype(int).squeeze()
test['image_size_encoding'] = categorize.fit_transform(test.image_size_scaled.values.reshape(-1,1)).astype(int).squeeze()
train_color = pd.read_csv('../input/mean-color-isic2020/train_color.csv')
test_color = pd.read_csv('../input/mean-color-isic2020/test_color.csv')
train['mean_color'] = train_color.values
test['mean_color'] = test_color.values
from sklearn.preprocessing import KBinsDiscretizer 

categorize  = KBinsDiscretizer(n_bins = 10 , encode = 'ordinal' , strategy = 'uniform')

train['mean_color_encoding'] = categorize.fit_transform(train['mean_color'].values.reshape(-1,1)).astype(int).squeeze()
test['mean_color_encoding'] = categorize.transform(test['mean_color'].values.reshape(-1,1)).astype(int).squeeze()
train['age_id_min'] = train['patient_id'].map(train.groupby(['patient_id']).age_approx.min())
train['age_id_max'] = train['patient_id'].map(train.groupby(['patient_id']).age_approx.max())

test['age_id_min'] = test['patient_id'].map(test.groupby(['patient_id']).age_approx.min())
test['age_id_max'] = test['patient_id'].map(test.groupby(['patient_id']).age_approx.max())
train['age_approx_mean'] = train['age_approx'].map(train.groupby(['age_approx'])['target'].mean())
test['age_approx_mean'] = test['age_approx'].map(train.groupby(['age_approx'])['target'].mean())
train['sex_encoding_mean'] = train['sex_encoding'].map(train.groupby(['sex_encoding'])['target'].mean())
test['sex_encoding_mean'] = test['sex_encoding'].map(train.groupby(['sex_encoding'])['target'].mean())
train['anatom_site_general_challenge_encoding_mean'] = train['anatom_site_general_challenge_encoding'].map(train.groupby(['anatom_site_general_challenge_encoding'])['target'].mean())
test['anatom_site_general_challenge_encoding_mean'] = test['anatom_site_general_challenge_encoding'].map(train.groupby(['anatom_site_general_challenge_encoding'])['target'].mean())
train['n_images_encoding_mean'] = train['n_images_encoding'].map(train.groupby(['n_images_encoding'])['target'].mean())
test['n_images_encoding_mean'] = test['n_images_encoding'].map(train.groupby(['n_images_encoding'])['target'].mean())
train['image_size_encoding_mean'] = train['image_size_encoding'].map(train.groupby(['image_size_encoding'])['target'].mean())
test['image_size_encoding_mean'] = test['image_size_encoding'].map(train.groupby(['image_size_encoding'])['target'].mean())
corr = train.corr(method = 'pearson')
corr = corr.abs()
corr.style.background_gradient(cmap='inferno')
corr = test.corr(method = 'pearson')
corr = corr.abs()
corr.style.background_gradient(cmap='inferno')
test.columns
test.columns
features = [
    
    #'image_name',
    #'patient_id',
    #'sex', 
    'age_approx',
    #'anatom_site_general_challenge', 
    'sex_encoding',
    'anatom_site_general_challenge_encoding',
    'n_images',
    #'n_images_encoding',
    # 'age_enc',
   # 'image_size',
    'image_size_scaled',
   # 'image_size_encoding',
    'mean_color',
    #'mean_color_encoding',
    'age_id_min',
    'age_id_max',
   # 'age_approx_mean',
    #'sex_encoding_mean',
    #'anatom_site_general_challenge_encoding_mean',
   # 'n_images_encoding_mean',
   # 'image_size_encoding_mean',
       # 'anatom_head/neck',
       # 'anatom_lower extremity',
       #'anatom_oral/genital',
       # 'anatom_palms/soles', 
      #  'anatom_torso',
     #  'anatom_unknown_anatom', 
    #'anatom_upper extremity'
    
]
X = train[features]
y = train['target']
xgb = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.04], #so called `eta` value
              'max_depth': [9],
              'min_child_weight': [5],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
           'n_estimators': [500]}

xgb_grid = GridSearchCV(xgb,
                        parameters,
                        cv = 4,
                        n_jobs = 5,
                        verbose=True
                       )

xgb_grid.fit(X,
         y)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
xgb = XGBRegressor(
                nthread=4, #when use hyperthread, xgboost may become slower
              objective='binary:logistic',
              learning_rate= 0.04, #so called `eta` value
              max_depth = 11,
              min_child_weight = 5,
              silent = 1,
              subsample= 0.7,
              colsample_bytree = 0.7,
           n_estimators =  500

)

folds = StratifiedKFold(n_splits = 5 , shuffle = True, random_state = 42 )
cv_results = cross_val_score(xgb ,X , y, cv = folds, scoring = 'roc_auc', verbose = 3 ) 
print(cv_results.mean())
xgb.fit(X,y)
predictions = xgb.predict(test[features])
sub['target'] = predictions
sub.to_csv('xgb_anatom_ohe.csv',index = False)
sub.head()
model = CatBoostClassifier()
parameters = {'depth'         : [6],
                  'learning_rate' : [0.03],
                  'iterations'    : [600]
                 }
grid1 = GridSearchCV(estimator=model, param_grid = parameters, cv = 3, n_jobs=-1)
grid1.fit(X, y)    
    # Results from Grid Search
print("\n========================================================")
print(" Results from Grid Search " )
print("========================================================")       
print("\n The best estimator across ALL searched params:\n",
          grid1.best_estimator_)    
print("\n The best score across ALL searched params:\n",
          grid1.best_score_)
print("\n The best parameters across ALL searched params:\n",
          grid1.best_params_)  
print("\n ========================================================")
cat = CatBoostClassifier(
    depth = 6,
    iterations = 600,
    learning_rate = 0.03,
    verbose = 0
)

cv_results = cross_val_score(cat ,X , y, cv = folds, scoring = 'roc_auc', verbose = 3 ) 
print(cv_results.mean())
cat.fit(X,y)
predictions = cat.predict(test[features])
sub['target'] = predictions
sub.to_csv('cat_sub.csv',index = False)
sub.head()
model  = GaussianNB()

cv_results = cross_val_score(model,X , y, cv = folds, scoring = 'roc_auc', verbose = 3 ) 
print(cv_results.mean())
model.fit(X,y)
predictions = model.predict(test[features])
sub['target'] = predictions
sub.to_csv('gaussian_sub.csv',index = False)
sub.head()
model  = MultinomialNB()

cv_results = cross_val_score(model,X , y, cv = folds, scoring = 'roc_auc', verbose = 3 ) 
print(cv_results.mean())
model = AdaBoostClassifier()
parameters = {'n_estimators' : [25],
                  'learning_rate' : [0.015],
                 }
grid1 = GridSearchCV(estimator=model, param_grid = parameters, cv = 3, n_jobs=-1)
grid1.fit(X, y)    
    # Results from Grid Search
print("\n========================================================")
print(" Results from Grid Search " )
print("========================================================")       
print("\n The best estimator across ALL searched params:\n",
          grid1.best_estimator_)    
print("\n The best score across ALL searched params:\n",
          grid1.best_score_)
print("\n The best parameters across ALL searched params:\n",
          grid1.best_params_)  
print("\n ========================================================")
model  = AdaBoostClassifier(
)

cv_results = cross_val_score(model,X , y, cv = folds, scoring = 'roc_auc', verbose = 3 ) 
print(cv_results.mean())
model.fit(X,y)
predictions = model.predict(test[features])
sub['target'] = predictions
sub.to_csv('ada_sub.csv',index = False)
sub.head()
model = LGBMClassifier()
parameters = {'n_estimators' : [50,100,150],
                  'learning_rate' : [0.015,0.01,0.005],
                 }
grid1 = GridSearchCV(estimator=model, param_grid = parameters, cv = 3, n_jobs=-1)
grid1.fit(X, y)    
    # Results from Grid Search
print("\n========================================================")
print(" Results from Grid Search " )
print("========================================================")       
print("\n The best estimator across ALL searched params:\n",
          grid1.best_estimator_)    
print("\n The best score across ALL searched params:\n",
          grid1.best_score_)
print("\n The best parameters across ALL searched params:\n",
          grid1.best_params_)  
print("\n ========================================================")
from lightgbm import LGBMClassifier
model  = LGBMClassifier()

cv_results = cross_val_score(model,X , y, cv = folds, scoring = 'roc_auc', verbose = 3 ) 
print(cv_results.mean())
model.fit(X,y)
predictions = model.predict(test[features])
sub['target'] = predictions
sub.to_csv('lgbm_sub.csv',index = False)
sub.head()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout ,BatchNormalization
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow as tf
!pip install tensorflow-addons=='0.9.1'
import tensorflow_addons as tfa
from sklearn.utils import class_weight 
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train.target),
                                                 train.target)

class_weights = { 0 :  0.50897302 , 1 : 28.36130137 }

classweights =[item for k in class_weights for item in (k, class_weights[k])]
print(classweights)
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=(8),
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay=0.0001, amsgrad=False)
model.compile(optimizer= adam,
              loss ='binary_crossentropy', # tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO)
              metrics=['accuracy',tf.keras.metrics.AUC()])
hist = model.fit(X, y,
                    batch_size=32,
                    epochs=100,
                    verbose=1,
                    class_weight = class_weights
                )
model.fit(X,y)
predictions = model.predict(test[features])
sub['target'] = predictions
sub.to_csv('nn_sub.csv',index = False)
sub.head()
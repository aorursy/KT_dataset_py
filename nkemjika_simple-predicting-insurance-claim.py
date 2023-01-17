# Import module feature_selector- it will be used as part of Exploratory Data Analysis
!cp -r ../input/feature-selector/feature-selector/feature-selector-master/feature_selector/* ./
import feature_selector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_selector import FeatureSelector
train_df = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv', low_memory=False, na_values= '-1')
pd.options.display.max_rows=None
pd.options.display.max_columns = None
train_df.head()
train_df.target.value_counts()
train_df['target'].value_counts().plot(kind='bar', figsize=(5,5));

# Sample figsize in inches
fig, ax = plt.subplots(figsize=(20,10))         
# Imbalanced DataFrame Correlation
corr = train_df.corr()
sns.heatmap(corr, cmap='RdYlBu', annot_kws={'size':30}, ax=ax)
ax.set_title("Feature Correlation Matrix", fontsize=14)
plt.show()
# Make a copy of data
train_df_copy = train_df.copy()
def remove_calc(data_df):
  for label, content in data_df.items():
    if '_calc' in label:
      data_df.drop([label], axis=1, inplace=True)

  return data_df

train_df_copy = remove_calc(train_df_copy)

train_df_copy.columns.values
train_df_copy.columns.value_counts()
fig, ax = plt.subplots(figsize=(20,10))         
(train_df.isna().sum()*100/len(train_df)).round(2).plot(kind='bar', color='salmon');
# from google.colab import drive
# drive.mount('/content/drive')

train_df_copy.drop(['ps_car_03_cat', 'ps_car_05_cat'], axis=1, inplace=True)
train_df_copy.info()
train_df_copy.columns.values

categorical_column =[]
categorical_missing_data=[]
not_categorical = []  
# train_target = []
# train_id = []

def preprocess_data(data_df):
  data_df_copy = data_df.copy()

  if 'target' in data_df.columns:
    train_target = data_df.target
    data_df.drop(['target'], axis=1, inplace=True)
  if 'id' in data_df.columns:
    train_id = data_df.id
    data_df.drop(['id'], axis=1, inplace=True) 

  

  for label, content in data_df.items():    
    if '_cat'  in label:
      categorical_column.append(label)
      data_df[label].fillna(value=content.mode()[0], inplace=True)
      data_df[label] = data_df[label].astype('category')

    elif '_bin' in label:
      data_df[label].fillna(value=content.mode()[0], inplace=True)

    else:
      data_df[label].fillna(value=content.median(), inplace=True)
      not_categorical.append(label)    

    
  print(categorical_column)
  if 'target' in data_df_copy.columns:
    data_df.insert(loc=0, column='target', value=train_target)    
    # if (train_target.empty == True) :
      
  if ('id' in data_df_copy.columns):
    data_df.insert(loc=0, column='id', value= train_id)
    # if (train_id.empty == True):

  ### Remove outliers
  # #Dropping the outlier rows with standard deviation
  # factor = 4
  # for label, content in data_df.items():
  #   upper_lim = data_df[label].mean () + data_df[label].std () * factor
  #   lower_lim = data_df[label].mean () - data_df[label].std () * factor

  #   data = data_df[(data_df[label] < upper_lim) & (data_df[label] > lower_lim)]     

  return data_df       
        
preprocessed_train_data = preprocess_data(train_df_copy)


preprocessed_train_data.isna().sum()
preprocessed_train_data.info()
# shuffled_df = preprocessed_train_data
# # shuffled_df.drop(['id'], axis=1, inplace=True)
# shuffled_df[categorical_column].head(10)
len(preprocessed_train_data)
# # Extract Features and target

# X = shuffled_df.drop(['target', 'id'], axis=1)
# y=  shuffled_df['target']

#train_df_copy['ps_ind_02_cat'].value_counts()
len(categorical_column), len(categorical_missing_data), len(not_categorical)
# # from sklearn.preprocessing import OneHotEncoder
# # from sklearn.compose import ColumnTransformer
# # categorical_features = categorical_column
# # one_hot = OneHotEncoder(sparse=False)
# # transformer = ColumnTransformer([('one_hot', one_hot, categorical_features)], remainder='passthrough')

# # transformed_x = transformer.fit_transform(X)
# shuffled_df_encoded = pd.get_dummies(shuffled_df[categorical_column])

# shuffled_df_encoded.head()
# shuffled_df_encoded.isna().sum()
# shuffled_cat_dropped = shuffled_df.drop(categorical_column, axis=1)
# shuffled_df_encoded.drop(['ps_ind_02_cat_3.0'], axis=1, inplace=True)
# shuffled_cat_dropped.head()
# shuffled_upd = pd.concat([shuffled_cat_dropped, shuffled_df_encoded], axis=1)
# shuffled_upd.head()
preprocessed_train_data.head()
def Encode_Scale(data_df,categorical_features):
  """
  Function takes a dataframe, and a list of categorical features, encodes the categorical features
  and scales same.

  """
  data_df_copy = data_df.copy()

  if 'target' in data_df.columns:
    train_target = data_df.target
    data_df.drop(['target'], axis=1, inplace=True)
  if 'id' in data_df.columns:
    train_id = data_df.id
    data_df.drop(['id'], axis=1, inplace=True) 



  #One-Hot Encoding of categorical data
  data_df_encoded = pd.get_dummies(data_df[categorical_column])
  data_df_encoded.head()
  data_df_encoded.isna().sum()

  ### After the one-hot encoding, we drop the original unencoded categorical columns,
  ### then one of the new encoded feature columns to reduce multicollinearity.


  data_cat_dropped = data_df.drop(categorical_column, axis=1)
  data_df_encoded.drop(['ps_ind_02_cat_3.0'], axis=1, inplace=True)
  data_cat_dropped.head()

  ### Concatenate the encoded categorical features with the other features less the unencoded categorical features

  data_upd = pd.concat([data_cat_dropped, data_df_encoded], axis=1)

  if 'target' in data_df_copy.columns:
    data_upd.insert(loc=0, column='target', value=train_target)    
    # if (train_target.empty == True) :
      
  if ('id' in data_df_copy.columns):
    data_upd.insert(loc=0, column='id', value= train_id)
    # if (train_id.empty == True):


  data_upd.head()

  # preferred_data = data_upd[preferred_features]

  # from sklearn.preprocessing import StandardScaler
  # X = StandardScaler().fit_transform(preferred_data)

  # X = pd.DataFrame(X)


  return data_upd

preprocessed_train_data.head()
shuffled_upd = Encode_Scale(preprocessed_train_data, categorical_column)
shuffled_upd.head()
# Extract Features and target

X = shuffled_upd.drop(['target', 'id'], axis=1)
y=  shuffled_upd['target']
from feature_selector import FeatureSelector
fs = FeatureSelector(X, y)
fs.identify_all(selection_params = {'missing_threshold': 0.6, 'correlation_threshold': 0.98, 
                                    'task': 'classification', 'eval_metric': 'auc', 
                                     'cumulative_importance': 0.99})
# justcheckit = fs.one_hot_features
shuffled_df_removed_all_once = fs.remove(methods = 'all', keep_one_hot = True)
shuffled_df_removed_all_once.shape
fs.plot_feature_importances(plot_n = 15, threshold=0.99)
preferred_features = np.array(fs.feature_importances[fs.feature_importances['cumulative_importance']<0.990402]['feature'])
len(preferred_features)
preferred_data = fs.data[preferred_features]
preferred_data.head()
# Using get_dummies to encode categorical features
# cat_df = pd.get_dummies(shuffled_df, columns=[categorical_column])
# cat_df.head()
### Lets scale the features to get them with same range of magnitude

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(preferred_data)
X = pd.DataFrame(X)
X.head()

### Models used
# Models from Scilit-Learn

from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from xgboost import XGBClassifier

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, auc


# np.random.seed(42)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from imblearn.ensemble import BalancedRandomForestClassifier


# build model with embedded undersampling technique 
# param = {'num_leaves': 31, 'objective': 'binary'}
# param['metric'] = 'auc'
mpipeline = make_pipeline_imb(BalancedBaggingClassifier(base_estimator=lgb.LGBMClassifier(n_jobs=-1),
                                                   sampling_strategy='auto',
                                                   replacement=False,
                                                   random_state=0))
model = mpipeline.fit(X_train, y_train)
model.score(X_val, y_val)
bbc_pred = model.predict_proba(X_val)

# build model with SMOTE imblearn
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train2, X_val2, y_train2, y_val2 = train_test_split(X_res, y_res, test_size = 0.2)
smote_model = LogisticRegression(n_jobs=-1)      #XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)
smote_model.fit(X_train2, y_train2)
smote_score = smote_model.score(X_val2, y_val2)

smote_score
smote_pred = smote_model.predict_proba(X_val2)
smote_model.score(X_train2, y_train2)
smote_predict = smote_model.predict(X_val2)
model.score(X_train, y_train)
model.score(X_val, y_val)

# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# scores = cross_val_score(smote_model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
# score_accuracy = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
# score_accuracy.mean()
# from google.colab import drive
# drive.mount('/content/drive')
# Calculating the normalized gini coefficient.
def ginic(actual, pred):
    actual = np.asarray(actual) #In case, someone passes Series or list
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n
 
def gini_normalizedc(a, p):
    if p.ndim == 2:#Required for sklearn wrapper
        p = p[:,1] #If proba array contains proba for both 0 and 1 classes, just pick class 1
    return ginic(a, p) / ginic(a, a)

smote_pred[:,1]
gini_normalizedc(y_val, bbc_pred[:,1])
gini_normalizedc(y_val2, smote_pred[:, 1])
# from google.colab import drive
# drive.mount('/content/drive')
test_df = pd.read_csv('../input/porto-seguro-safe-driver-prediction/test.csv', low_memory=False, na_values='-1')
test_df.head()
test_df.shape
test_df.isna().sum()
test_data_no_id = test_df.drop(['id'], axis=1)
### Check missing values in the test data
fig, ax = plt.subplots(figsize=(20,10))         
(test_df.isna().sum()*100/len(test_df)).round(2).plot(kind='bar', color='salmon');
### Remove fetures having more than 50% of it's data missing
test_df.drop(['ps_car_03_cat', 'ps_car_05_cat'], axis=1, inplace=True)
# test_df.drop(['id'], axis=1, inplace=True)
categorical_column=[]
categorical_missing_data = []
not_categorical = []
preprocessed_test_df = preprocess_data(test_df)
len(categorical_column), len(categorical_missing_data), len(not_categorical)
preprocessed_test_df.isna().sum()
preprocessed_test_df.head()
fig, ax = plt.subplots(figsize=(20,10))         
# Imbalanced DataFrame Correlation
corr = preprocessed_test_df.corr()
sns.heatmap(corr, cmap='RdYlBu', annot_kws={'size':30}, ax=ax)
ax.set_title("Feature Correlation Matrix", fontsize=14)
plt.show()
preprocessed_test_df   = remove_calc(preprocessed_test_df)
preprocessed_test_df.columns.values
preprocessed_test_df.head()
preprocessed_test_df_upd = Encode_Scale(preprocessed_test_df, categorical_column)
# sum_feature_df= pd.DataFrame(transformed_testData_x[:10])
# sum_feature_df
preprocessed_test_df_upd.drop(['id'], axis=1, inplace=True)
preprocessed_test_data = preprocessed_test_df_upd[preferred_features]
preprocessed_test_data.shape
preprocessed_test_data.head()
preprocessed_test_data.head()
X_test = StandardScaler().fit_transform(preprocessed_test_data)
X_test = pd.DataFrame(X_test)
X_test.head()
# from google.colab import drive
# drive.mount('/content/drive')
X_test.shape
test_pred = smote_model.predict_proba(X_test)
test_pred[:,1][:20]
test_pred2 = model.predict_proba(X_test)
test_pred2[:,1][:20]
# preprocessed_test_df.head()
PIC_Submission = pd.DataFrame(test_pred2[:,1], columns=['target'], index=np.arange(0,len(preprocessed_test_df)))
PIC_Submission.head()
len(PIC_Submission)
#PIC_Submission.to_csv('submit_pred_2.csv')
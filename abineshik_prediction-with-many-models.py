import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, FunctionTransformer
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
df = pd.read_csv('Data - Parkinsons')
df.head()
df.shape
df.info()
df.describe().T
df.status = df.status.astype('category')
df.status.value_counts()
df.describe()
corr = df.drop(['name', 'status'], axis = 1).corr()
fig, ax = plt.subplots(figsize = [23,10])

sns.heatmap(corr, annot = True, vmin = -1, vmax = 1, center = 0, cmap="coolwarm")
corr_pos = corr.abs()
mask = (corr_pos < 0.8 ) 
fig, ax = plt.subplots(figsize = [23,10])
sns.heatmap(corr, annot = True, vmin = -1, vmax = 1, center = 0, mask = mask, cmap="coolwarm")
f = ['MDVP:Jitter(%)',	'MDVP:Jitter(Abs)',	'MDVP:RAP',	'MDVP:PPQ',	'Jitter:DDP', 'NHR', 
     'MDVP:Shimmer',	'MDVP:Shimmer(dB)',	'Shimmer:APQ3',	'Shimmer:APQ5',	'MDVP:APQ',	'Shimmer:DDA', 'HNR', 'spread1', 'PPE']
fig, ax = plt.subplots(nrows = 4, ncols = 4, figsize = [23, 13])
for f, ax in zip(f, ax.flatten()):
  sns.distplot(df[f], ax =ax)
  mean = df[f].mean()
  median = df[f].median()
  ax.axvline(mean, color='r', linestyle='--')
  ax.axvline(median, color='b', linestyle='-')
  ax.legend({'Mean':mean, 'Median':median})

features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'NHR', 'HNR', 'RPDE', 'DFA',  'spread2', 'D2', 'PPE']
fig, ax = plt.subplots(nrows = 2, ncols = 5, figsize = [23, 10])
for feature, ax in zip(features, ax.flatten()):
  sns.distplot(df[feature], ax =ax)
  mean = df[feature].mean()
  median = df[feature].median()
  ax.axvline(mean, color='r', linestyle='--')
  ax.axvline(median, color='b', linestyle='-')
  ax.legend({'Mean':mean, 'Median':median})

fig, ax = plt.subplots(nrows = 2, ncols = 5, figsize = [23, 10])
for feature, ax in zip(features, ax.flatten()):
  sns.boxplot(df[feature], ax =ax, orient = 'v')
  ax.set_title(feature)

fig, ax = plt.subplots(nrows = 2, ncols = 5, figsize = [23, 10])
for feature, ax in zip(features, ax.flatten()):
  sns.boxplot( x= 'status', y  = feature, data = df, ax =ax)
p_NHR = np.percentile(df.NHR, [25,75])
p_Jitter = np.percentile(df['MDVP:Jitter(%)'], [25,75])
iqr_NHR = p_NHR[1] = p_NHR[0]
iqr_Jitter = p_Jitter[1] = p_Jitter[0]
corrected_NHR = df.NHR.clip(p_NHR[0]-1.5*iqr_NHR, p_NHR[1]+1.5*iqr_NHR)
corrected_Jitter = df['MDVP:Jitter(%)'].clip(p_Jitter[0]-1.5*iqr_Jitter, p_Jitter[1]+1.5*iqr_Jitter)
corr = np.corrcoef(corrected_Jitter,corrected_NHR)
corr
p_HNR = np.percentile(df.HNR, [25,75])
p_Shimmer = np.percentile(df['MDVP:Shimmer'], [25,75])
iqr_HNR = p_HNR[1] = p_HNR[0]
iqr_Shimmer = p_Shimmer[1] = p_Shimmer[0]
corrected_HNR = df.NHR.clip(p_HNR[0]-1.5*iqr_HNR, p_HNR[1]+1.5*iqr_HNR)
corrected_Shimmer = df['MDVP:Shimmer'].clip(p_Shimmer[0]-1.5*iqr_Shimmer, p_Shimmer[1]+1.5*iqr_Shimmer)
corr = np.corrcoef(corrected_Shimmer,corrected_HNR)
corr
#funection to treat outliers in the features
def outlier_correction(df):
  outlier_features = ['MDVP:Fo(Hz)', 'RPDE', 'DFA',  'spread2', 'D2', 'PPE']
  for col in outlier_features:
    p = np.percentile(df[col], [25, 75])
    iqr = p[1]-p[0]
    df.loc[:, col].clip(lower = p[0]-1.5*iqr, upper = p[1]+1.5*iqr, inplace = True)
  return df
#Removing all highly correlated features and selecting only these features
features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'NHR', 'HNR', 'RPDE', 'DFA',  'spread2', 'D2', 'PPE']
#f = [ i for i in df.columns if i not in  ['name', 'status', 'spread1']]
column = ColumnTransformer( 
                              transformers = [('encoding', MinMaxScaler(), features)]
                      )
pipeline_model = Pipeline(
    [('outliers', FunctionTransformer(outlier_correction)),
     ('scaling', column),
     ('model', LogisticRegression())
     ]
)
seed = 4
#Seperating input and target features
x = df.drop('status', axis =1)
y = df['status']
#splitting into train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, stratify = y, random_state = seed)
#Creating many possible set of paramters for hyperparameter tuning for logistic regression
params_logistic = [ {
    'scaling__encoding' : [MinMaxScaler(), StandardScaler(), RobustScaler()],
    'model':[LogisticRegression()],
    'model__penalty' : ['l1', 'l2'],
    'model__solver' : ['liblinear'],
    'model__C' : np.logspace(0, 4, 10),
    'model__random_state' : [seed]
},
{
    'scaling__encoding' : [MinMaxScaler(), StandardScaler(), RobustScaler()],
    'model':[LogisticRegression()],
    'model__penalty' : ['l2'],
    'model__solver' : ['newton-cg', 'sag' , 'saga'  'ibfgs'],
    'model__C' : np.logspace(0, 4, 10),
    'model__random_state' : [seed]
},
{
    'scaling__encoding' : [MinMaxScaler(), StandardScaler(), RobustScaler()],
    'model':[LogisticRegression()],
    'model__penalty' : ['elasticnet'],
    'model__solver' : ['saga'],
    'model__C' : np.logspace(0, 4, 10),
    'model__random_state' : [seed]
}
]
#Creating many possible set of paramters for hyperparameter tuning for KNN
params_knn = {
    'scaling__encoding' : [MinMaxScaler(), StandardScaler(), RobustScaler()],
    'model': [KNeighborsClassifier()],
    'model__n_neighbors' : [i for i in range(5, 15, 2)],
    'model__leaf_size' : [i for i in range(3, 50, 5)],
    # As we saw some outliers helped in classification and also we havent removed that outliers.
    # Making the weight uniform make sure that model gives uniform weightage to outliers too.
    'model__weights' : ['uniform'] 
}
#Creating many possible set of paramters for hyperparameter tuning for SVM
params_svm = {
    'scaling__encoding' : [MinMaxScaler(), StandardScaler(), RobustScaler()],
    'model' : [SVC(probability = True)],
    'model__C': [i for i in range(1, 30, 1)],
    'model__kernel' : ['linear']
            }
#Creating a dictionary of the parameters of all models
params = {
          'Logistic Regression': params_logistic,
          'KNN': params_knn,
          'SVM': params_svm
          }
#Creating a array to store the best estimate
model =[]
#Creating DataFrame to store the performance metrics of each model.
performance = pd.DataFrame(columns = ['Model', 'Train Accuracy', 'Test Accuracy', 'Mean Cross Validation Accuracy',
                                   '+/- Deviation in Cross validation accuracy'])
#Iterating throught each model
for key in params:
  #Using StratifiedKFold to make sure that the both class of target is evenly distributed
  stf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
  gridSearch = GridSearchCV(pipeline_model, params[key], n_jobs= -1, cv = stf)
  gridSearch.fit(x_train, y_train)
  #Storing the tuple of best estimate of each model which can be fed as input to ensemble classifier.
  model.append((key, gridSearch.best_estimator_))
  cv = cross_val_score(gridSearch.best_estimator_, x, y, cv = stf)
  performance = performance.append({'Model': key, 
                            'Train Accuracy':  np.round(gridSearch.score(x_train, y_train)*100, decimals = 2),
                            'Test Accuracy': np.round(gridSearch.score(x_test, y_test)*100, decimals = 2),
                            'Mean Cross Validation Accuracy': np.round(cv.mean()*100, decimals = 2),
                            '+/- Deviation in Cross validation accuracy': np.round(cv.std()*2*100, decimals = 2)}, ignore_index = True) 
performance
#giving all the best estimators in list of tuples
#It chooses Logistic Regression as a metaclassifier
sk = StackingClassifier( estimators = model)
sk.fit(x_train, y_train)
cv_sk = cross_val_score(sk, x, y, cv = stf)
performance = performance.append({'Model': 'StackingClassifier', 
                            'Train Accuracy':  np.round(sk.score(x_train, y_train)*100, decimals = 2),
                            'Test Accuracy': np.round(sk.score(x_test, y_test)*100, decimals = 2),
                            'Mean Cross Validation Accuracy': np.round(cv_sk.mean()*100, decimals = 2),
                            '+/- Deviation in Cross validation accuracy': np.round(cv_sk.std()*2*100, decimals = 2)}, ignore_index = True) 
performance
import sklearn
sklearn.__version__
params_ramdomforest = {
    'model': [RandomForestClassifier(n_estimators = 300,  ccp_alpha = 0.02,
                                     max_features = 0.4,  random_state = seed)],
    'model__criterion': ['gini', 'entropy']
}
ramdomforest = GridSearchCV(pipeline_model, params_ramdomforest, n_jobs= -1)
ramdomforest.fit(x_train, y_train)
cv_ramdomforest = cross_val_score(ramdomforest.best_estimator_, x, y, cv = stf)
performance = performance.append({'Model': 'Random Forest', 
                          'Train Accuracy':  np.round(ramdomforest.score(x_train, y_train)*100, decimals = 2),
                          'Test Accuracy': np.round(ramdomforest.score(x_test, y_test)*100, decimals = 2),
                          'Mean Cross Validation Accuracy': np.round(cv_ramdomforest.mean()*100, decimals = 2),
                          '+/- Deviation in Cross validation accuracy': np.round(cv_ramdomforest.std()*2*100, decimals = 2)}, ignore_index = True) 
params_xgboost = {
    'model': [XGBClassifier(random_state= seed)],
    'model__booster': ['gbtree', 'gblinear'],
    'model__gamma': [ 1.5, 2, 3, 4],
    'model__subsample': [0.6, 0.8, 1.0],
    'model__colsample_bytree': [0.6, 0.8, 1.0],
    'model__max_depth': [3, 4, 5]
}
xgboost = GridSearchCV(pipeline_model, params_xgboost, n_jobs= -1)
xgboost.fit(x_train, y_train)
cr_xgboost = cross_val_score(xgboost.best_estimator_, x, y, cv = stf)
performance = performance.append({'Model': 'XGBoost', 
                          'Train Accuracy':  np.round(xgboost.score(x_train, y_train)*100, decimals = 2),
                          'Test Accuracy': np.round(xgboost.score(x_test, y_test)*100, decimals = 2),
                          'Mean Cross Validation Accuracy': np.round(cr_xgboost.mean()*100, decimals = 2),
                          '+/- Deviation in Cross validation accuracy': np.round(cr_xgboost.std()*2*100, decimals = 2)}, ignore_index = True) 
performance
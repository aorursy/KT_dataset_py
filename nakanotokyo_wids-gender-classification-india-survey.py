import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
# Import train dataset using pandas
df = pd.read_csv("../input/train.csv")
# Rows and columns
print('Total rows and columns = ', df.shape)
# Take a look at the dataframe
df.head()
df.isnull().sum().head(5)
# Check for nulls and drop variables that 80% of the answers are missing

nulls = df.isnull().sum().sort_values(ascending = False).to_frame()
nulls = nulls.rename(columns = {0:'q_nulls'})
threshold = nulls['q_nulls'] > 0.8 * len(df)
print('Total columns to drop = ', len(nulls[threshold]))
columns_to_drop = nulls[threshold].index

#columns_to_drop
df = df.drop(columns_to_drop, axis = 1)
print('Total columns in dataset after dropping columns with nulls =', len(df.columns))
# Select only numerical variables as XGBOOST can't process categorical variables

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df = df.select_dtypes(include=numerics)
print('Total columns after dropping categoricals = ',len(df.columns))
# group by target 
target_grouped = df.groupby(['is_female']).count()

# group by target (%)

target_grouped_perc = target_grouped.apply(lambda r: r/r.sum(), axis=0)
target_grouped_perc
# select variables that separate target

dif = target_grouped_perc.diff().abs()
difference = dif.iloc[[1]]

difference_transposed = difference.transpose()

class_sep = difference_transposed[1] > 0.35
sep_variables = difference_transposed[class_sep].sort_values(1, ascending = False)
sep_variables.head(5)
#  Total casses in each variables that separate the most between male and female. 
target_grouped[sep_variables.head(5).index]
#Exploration of DL2 question
df2 = df[['is_female', 'DL2']]

fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.countplot( hue = 'is_female',y='DL2', data = df2 )

# NAs in DL2: Replacing missing values with 0 and compare
dfback = df.copy()
dfback[['DL2']] = dfback[['DL2']].fillna(value=0)

fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.countplot( hue = 'is_female',y='DL2', data = dfback[['is_female', 'DL2']])

# Exploration of MT7 question

df5 = df[['is_female', 'MT7']]

#fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
sns.countplot( hue = 'is_female',y='MT7', data = df5)
# Define function to plot confusion matrix 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Separate target column (is_female)

X = df.drop(['is_female'], axis = 1)
Y = df[['is_female']]

# Split dataset in training and testing to train model (using train) and evaluate results (using test)
seed = 27
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Identify model to be used and training and test datasets

model = XGBClassifier(seed = 27)
model.fit(X_train, y_train)

# Print parameters used
print(model)

# Apply Xgboost model to predict if a respondant is female in the test dataset

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

#Print accuracy

base_accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (base_accuracy * 100.0))
    
# Plot Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
class_names = np.array(["male","female"])

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
# Select top 10 features 

headers = ["variable", "score"]
values = sorted(zip(X_train.columns, model.feature_importances_) , key=lambda x: x[1] * -1)
tabla = tabulate(values[1:10], headers, tablefmt="plain")
print(tabla)
# Dataset size-down: Variables that are not reported as important in model feature importance are dropped

# convert feature important list to dataframe
labels = ['variable','score']
table = pd.DataFrame.from_records(values, columns = labels)

# select variables which importance score is = 0
not_important = table['score'] == 0
table_not_important = table [not_important]

# copy datadrame
df_copy = df.copy()

df_small = df_copy.drop(table_not_important['variable'].tolist(), axis = 1)

#df_chico = df[["is_female","DG6","DG3","DL1","DG1","DL15","DG4","MT6","MT1A","MT18_4","AA14","DG8a","DG5_4","DL0","DL2","G2P1_11","MT18_5","IFI16_1","AA7","MT18_3","FL4","MT11","DL7","IFI6_4","LN1A","DG9c","MT7A","MT9","MT18_2","FF3","IFI16_2","IFI20_9","G2P5_8","MT14A_2","IFI16_4","IFI17_2","FL10","FB26_1","AA15","DG9a","DG10c","DL5","DL26_5","G2P3_9","MT3_3","MT4_1","MT6C","MM12","IFI3_3","IFI17_1","FB15","FB19","FB19B_3","FB20","FB28_2","GN1","train_id","DG3A","DG5_2","DG5_5","DG5_6","DL8","DL25_8","G2P3_11","MT4_5","MT14_2","MT14C_2","MT14C_3","MT14C_4","MT17_1","MT17_3","MT17_7","MT17_8","FF16_1","IFI14_2","IFI16_7","IFI21","FL3","FL9B","FL13","FL15","FL18","FB4_1","LN2_1","GN5","DG12C_2","DL4_5","DL11","DL14","DL24","DL25_7","DL28","G2P1_8","G2P1_9","MT3_1","MT5","MT6A","MT6B","MT13_7","MT13_11","MT14A_11","MT17_2","MT18_1","MT18A_3","FF2","FF2A","FF5","FF6_7","FF14_1","FF16_2","MMP2_1","IFI4_3","IFI2_5","IFI2_7","IFI8_1","IFI12_2","IFI14_3","IFI17_4","FL2","FL8_1","FL8_2","FL8_7","FL9C","FL11","FL12","FL14","FB2","FB13","FB18","FB19A_3","FB21","FB28_1","LN2_4","GN2","GN3"]]
df_small.shape
# Smaller dataset
X = df_small.drop(['is_female'], axis = 1)
Y = df_small[['is_female']]

# split data into train and test sets
seed = 27
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = XGBClassifier( seed = 27)
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

#evaluation
ds_accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (ds_accuracy * 100.0))
cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
class_names = np.array(["male","female"])

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
# Feature importance table

headers = ["name", "score"]
values = sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1] * -1)
tabla = tabulate(values[1:10], headers, tablefmt="plain")
print(tabla)
# DL2_no_nulls
dfback = dfback.rename(columns={'DL2': 'DL2_no_nulls'})
DL2_no_nulls = dfback[['DL2_no_nulls']]

# DL2_no_nulls_one_hot_encoding
x = dfback[['DL2_no_nulls']]

encoded_DL2_no_nulls = None
for i in range(0, x.shape[1]):
    label_encoder = LabelEncoder()
    feature = label_encoder.fit_transform(x.iloc[:,i])
    feature = feature.reshape(x.shape[0], 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    feature = onehot_encoder.fit_transform(feature)
    if encoded_DL2_no_nulls is None:
        encoded_DL2_no_nulls = feature
    else:
        encoded_x = np.concatenate((encoded_DL2_no_nulls, feature), axis=1)
encoded_DL2_no_nulls = pd.DataFrame(encoded_DL2_no_nulls)
encoded_DL2_no_nulls.columns = ["encoded_DL2_no_nulls_0","encoded_DL2_no_nulls_1","encoded_DL2_no_nulls_2","encoded_DL2_no_nulls_3",
                               "encoded_DL2_no_nulls_4","encoded_DL2_no_nulls_5","encoded_DL2_no_nulls_6","encoded_DL2_no_nulls_7","encoded_DL2_no_nulls_8",
                               "encoded_DL2_no_nulls_9","encoded_DL2_no_nulls_10","encoded_DL2_no_nulls_11","encoded_DL2_no_nulls_12",
                               "encoded_DL2_no_nulls_13","encoded_DL2_no_nulls_14","encoded_DL2_no_nulls_15",
                               "encoded_DL2_no_nulls_16","encoded_DL2_no_nulls_17","encoded_DL2_no_nulls_18",
                               "encoded_DL2_no_nulls_19","encoded_DL2_no_nulls_20","encoded_DL2_no_nulls_21",
                               "encoded_DL2_no_nulls_22","encoded_DL2_no_nulls_23","encoded_DL2_no_nulls_24",
                               "encoded_DL2_no_nulls_25","encoded_DL2_no_nulls_26","encoded_DL2_no_nulls_27",
                               "encoded_DL2_no_nulls_28",
                               "encoded_DL2_no_nulls_29", "encoded_DL2_no_nulls_30","encoded_DL2_no_nulls_96","encoded_DL2_no_nulls_32"]
x = df[['DG6']]

encoded_DG6 = None
for i in range(0, x.shape[1]):
    label_encoder = LabelEncoder()
    feature = label_encoder.fit_transform(x.iloc[:,i])
    feature = feature.reshape(x.shape[0], 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    feature = onehot_encoder.fit_transform(feature)
    if encoded_DG6 is None:
        encoded_DG6 = feature
    else:
        encoded_DG6 = np.concatenate((encoded_DG6, feature), axis=1)
encoded_DG6 = pd.DataFrame(encoded_DG6)
encoded_DG6.columns = ["DG6_1","DG6_2","DG6_3","DG6_4","DG6_5","DG6_6","DG6_7","DG6_8","DG6_9"]
x = df[['DG3']]

encoded_DG3 = None
for i in range(0, x.shape[1]):
    label_encoder = LabelEncoder()
    feature = label_encoder.fit_transform(x.iloc[:,i])
    feature = feature.reshape(x.shape[0], 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    feature = onehot_encoder.fit_transform(feature)
    if encoded_DG3 is None:
        encoded_DG3 = feature
    else:
        encoded_x = np.concatenate((encoded_DG3, feature), axis=1)
encoded_DG3 = pd.DataFrame(encoded_DG3)
encoded_DG3.columns = ["DG3_1","DG3_2","DG3_3","DG3_4","DG3_5","DG3_6","DG3_7","DG3_8","DG3_99"]
#education_level

conditions = [
    (df['DG4']  < 5 ) & (df['DL15'] == 1), # condition_1
    (df['DG4'] == 5) & (df['DL15'] == 2), # condition_2
    (df['DG4'] > 5) & (df['DG4'] < 11) & (df['DL15'] == 3), # condition_3
    (df['DL15'] == 4)]
choices = [1, 1, 1, 2]
education_level = np.select(conditions, choices, default=3)
education_level = pd.DataFrame(education_level)
#education_level_one_hot_encoding
x = education_level

encoded_educationl = None
for i in range(0, x.shape[1]):
    label_encoder = LabelEncoder()
    feature = label_encoder.fit_transform(x.iloc[:,i])
    feature = feature.reshape(x.shape[0], 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    feature = onehot_encoder.fit_transform(feature)
    if encoded_educationl is None:
        encoded_educationl = feature
    else:
        encoded_x = np.concatenate((eencoded_educationl, feature), axis=1)
encoded_educationl = pd.DataFrame(encoded_educationl)
encoded_educationl.columns = ["educationl_1","educationl_2","educationl_3"]
#laescala
conditions = [
    (df['DL0'] == 2 ) & (df['DL15'] == 1) & (df['DL1'] == 7)]
choices = [100]
laescala = np.select(conditions, choices, default=0)
laescala = pd.DataFrame(laescala)
laescala.columns = ['laescala']
#lasuma
conditions = [
    (df['DL0'] == 2),
    (df['DL0'] == 1)]

choices = [10,1]
lasuma_DL0 = np.select(conditions, choices, default=0)
lasuma_DL0 = pd.DataFrame(lasuma_DL0)
lasuma_DL0.columns = ['lasuma_DL0']

conditions = [
    (df['DL15'] == 1),
    (df['DL15'] == 2),
    (df['DL15'] == 3),
    (df['DL15'] == 4)]
choices = [10,7.5,5,2.5]
lasuma_DL15 = np.select(conditions, choices, default=0)
lasuma_DL15 = pd.DataFrame(lasuma_DL15)
lasuma_DL15.columns = ['lasuma_DL15']

conditions = [
    (df['DL1'] == 7),
    (df['DL1'] == 1),
    (df['DL1'] == 8),
    (df['DL1'] == 4),
    (df['DL1'] == 99),
    (df['DL1'] == 3),
    (df['DL1'] == 5),
    (df['DL1'] == 2),
    (df['DL1'] == 6),
    (df['DL1'] == 10),
    (df['DL1'] == 9),
    (df['DL1'] == 96),]
choices = [10,9.2,8.4,7.6,6.8,6,5.2,4.4,3.6,2.8,2,1]
lasuma_DL1 = np.select(conditions, choices, default=0)
lasuma_DL1 = pd.DataFrame(lasuma_DL1)
lasuma_DL1.columns = ['lasuma_DL1']


lasuma = lasuma_DL0.merge(lasuma_DL15, how = 'left', left_index = True, right_index = True)
lasuma = lasuma.merge(lasuma_DL1, how = 'left', left_index = True, right_index = True)
lasuma = lasuma.apply(lambda row: row.lasuma_DL0 + row.lasuma_DL1 + row.lasuma_DL15, axis=1)
lasuma = pd.DataFrame(lasuma)
lasuma.columns = ['lasuma']
# merge new variables with dataset


df = df.merge(DL2_no_nulls, how = "left", left_index = True, right_index = True)
df = df.merge(encoded_DL2_no_nulls, how = "left", left_index = True, right_index = True)
df = df.merge(encoded_DG6, how = "left", left_index = True, right_index = True)
df = df.merge(encoded_DG3, how = "left", left_index = True, right_index = True)
df = df.merge(education_level, how = "left", left_index = True, right_index = True)
df = df.merge(encoded_educationl, how = "left", left_index = True, right_index = True)
df = df.merge(laescala, how = "left", left_index = True, right_index = True)
df = df.merge(lasuma, how = "left", left_index = True, right_index = True)

# separate target
X = df.drop(['is_female'], axis = 1)
Y = df[['is_female']]

# split data into train and test sets
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#model
model = XGBClassifier(seed = 27)
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
new_accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (new_accuracy * 100.0))
cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
class_names = np.array(["male","female"])

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
# feature importance
headers = ["name", "score"]
values = sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1] * -1)
tabla = tabulate(values[1:10], headers, tablefmt="plain")
print(tabla)
#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
train = X_train.merge(Y, how = 'left', left_index = True, right_index = True)
target = 'is_female'
IDcol = train.index
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['is_female'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print ("\n Resultados del modelo")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['is_female'].values, dtrain_predictions))    
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['is_female'], dtrain_predprob))
                    
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')
#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target]]

xgb1 = XGBClassifier(
 seed=27)
modelfit(xgb1, train, predictors)
print(xgb1)
#Tunning de max_depth and min_child_weight

#param_test1 = {
# 'max_depth':(2,3,10),
# 'min_child_weight':(1,2,5)
#}
#gsearch1 = GridSearchCV(estimator = XGBClassifier( base_score=0.5, booster='gbtree', colsample_bylevel=1,
#       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
#       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
#       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=27, silent=True,
#       subsample=1), 
#param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch1.fit(train[predictors],train[target])
#gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#Tunning gamma and update max_depth and min_child_weight

#param_test3 = {
# 'gamma':[i/10.0 for i in range(0,5)]
#}
#gsearch3 = GridSearchCV(estimator = XGBClassifier(  base_score=0.5, booster='gbtree', colsample_bylevel=1,
#       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#       max_depth=10, min_child_weight=5, missing=None, n_estimators=100,
#       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
#       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=27, silent=True,
#       subsample=1), 
 #param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

#gsearch3.fit(train[predictors],train[target])
#gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
#Tunning subsample y colsample_bytree

#param_test4 = {
# 'subsample':[i/10.0 for i in range(6,10)],
# 'colsample_bytree':[i/10.0 for i in range(6,10)]
#}

#gsearch4 = GridSearchCV(estimator = XGBClassifier(  base_score=0.5, booster='gbtree', colsample_bylevel=1,
#       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#       max_depth=10, min_child_weight=5, missing=None, n_estimators=100,
#       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
#       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=27, silent=True,
#       subsample=1),
# param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch4.fit(train[predictors],train[target])
#gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
# split data into train and test sets
seed = 27
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#model
model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.7, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=10, min_child_weight=5, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=27, silent=True,
       subsample=0.9)
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
final_accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (final_accuracy * 100.0))
cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
class_names = np.array(["male","female"])

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

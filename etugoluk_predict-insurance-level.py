import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

#tools for preparing data and evaluation models
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, r2_score, mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score,roc_curve, auc, balanced_accuracy_score,precision_recall_curve,roc_auc_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

#for models
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from imblearn.ensemble import EasyEnsembleClassifier
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
df = pd.read_csv('/kaggle/input/prudential-life-insurance-assessment/train.csv.zip')
df.head(10)
df.info()
df.describe()
fig_size = (20, 20)
fig, ax = plt.subplots(figsize=fig_size)
sns.heatmap(df.corr(), cmap="Blues")
threshold = 0.1
corr_matrix = df.corr()
columns = []
for i in range(len(corr_matrix.columns)):
    if (abs(corr_matrix.iloc[i, 126]) >= threshold):
        columns.append(df.columns[i + 1])
        print(df.columns[i + 1], corr_matrix.iloc[i, 126])
norm_corr = df[columns]
fig_size = (9, 7)
fig, ax = plt.subplots(figsize=fig_size)
sns.heatmap(norm_corr.corr())
print (df.corr()["Response"].sort_values(ascending=False))
fig, ax = plt.subplots(figsize=(9, 7))
df['Response'].hist(grid=False, bins=8)
df['Response'].value_counts()
fig, ax = plt.subplots(figsize=(20,8))  
sns.heatmap(df.isnull(), cbar=False)
missing_val_count_by_column = (df.isnull().sum()/len(df))
print(missing_val_count_by_column[missing_val_count_by_column>0.3].sort_values(ascending=False))
# #normalize Employment_Info_2
# min_max_scaler = preprocessing.MinMaxScaler()
# Employment_Info_2_scaled = min_max_scaler.fit_transform(df["Employment_Info_2"].values.reshape(-1, 1))
# df["Employment_Info_2"] = Employment_Info_2_scaled

# #normalize Employment_Info_3
# df["Employment_Info_3"] = df["Employment_Info_3"] * 0.5 - 0.5
# df["Employment_Info_3"] = df["Employment_Info_3"].astype(int)

# #normalize Employment_Info_5
# df["Employment_Info_5"] = df["Employment_Info_5"] - 2
#Convert Product_Info_2 into dummy variables
Product_Info_2_dummies = pd.get_dummies(df.Product_Info_2)
df = df.drop('Product_Info_2',axis = 1)
df = df.join(Product_Info_2_dummies)
#new features
med_keyword_columns = df.columns[df.columns.str.startswith('Medical_Keyword_')]
df['Med_Keywords_Count'] = df[med_keyword_columns].sum(axis=1)

#sum of all Medical_Keyword columns
# med_keyword_columns = copy_df.columns[copy_df.columns.str.startswith('Medical_Keyword_')]
# copy_df['Med_Keywords_Count'] = copy_df[med_keyword_columns].sum(axis=1)
# copy_df['BMI_Age'] = copy_df['BMI'] * copy_df['Ins_Age']
#Handle missing columns
copy_df = df.copy()
imputer = SimpleImputer()
copy_df = pd.DataFrame(imputer.fit_transform(copy_df))
copy_df.columns = df.columns
copy_df.head()
#split DataFrame into input data and target
X = copy_df.drop(['Id', 'Response'], axis=1)
y = copy_df.Response.values.astype('int')
#split into train/valid sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=17)
cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
#define model
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=42)
#get cross-validation score
rf_cv_score = cross_val_score(rf_model, X_train, y_train, cv=cv)
print (rf_cv_score)
#train model
rf_model.fit(X_train, y_train)

#predictiion for validation set
y_pred = rf_model.predict(X_valid)

#confusion matrix
print(pd.crosstab(y_valid, y_pred, rownames=['Actual Species'], colnames=['Predicted Species']), '\n')

#classification report
print (classification_report(y_valid, y_pred))
xgb_model = xgb.XGBClassifier(max_depth=10, objective='multi:softmax', num_class=8)
xgb_cv_score = cross_val_score(xgb_model, X_train, y_train, cv=cv)
print (xgb_cv_score)
#train model
xgb_model.fit(X_train, y_train)

#predict
y_pred = xgb_model.predict(X_test)
#confusion matrix
print(pd.crosstab(y_valid.astype('int'), y_pred.astype('int'), rownames=['Actual'], colnames=['Predicted']))
#classification report
print (classification_report(y_valid, y_pred.astype('int')))
#define model
nn = Sequential()
nn.add(Dense(128, activation='relu'))
nn.add(Dense(64, activation='relu'))
nn.add(Dense(8, activation='softmax'))
nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
y_train_one_hot = to_categorical(y_train-1, num_classes=8, dtype='float32')
nn_cv_score = []

for i, (train_index, val_index) in enumerate(cv.split(X_train.values, y_train_one_hot.argmax(1))):
    print ('Fold %i' %i)
    x_train_cv, x_val_cv = X_train.values[train_index], X_train.values[val_index]
    y_train_cv, y_val_cv = y_train_one_hot[train_index], y_train_one_hot[val_index]
    hist = nn.fit(x_train_cv, y_train_cv, epochs=20, verbose=0, validation_data=(x_val_cv, y_val_cv))
    nn_cv_score.append(hist.history['accuracy'][-1])

print (nn_cv_score)
#train model
nn.fit(X_train, y_train_one_hot, epochs=50)
y_pred = nn.predict(X_valid)
y_pred = np.argmax(y_pred, axis=1) + 1 #decoding
#confusion matrix
print(pd.crosstab(y_valid.astype('int'), y_pred.astype('int'), rownames=['Actual Species'], colnames=['Predicted Species']))
#classification report
print (classification_report(y_valid, y_pred.astype('int')))
new_df = copy_df.copy()
for i in range(1, 8):
    new_df['x > %d' %i] = copy_df.Response.apply(lambda x: 1 if x > i else 0)
new_df.head()
prob_cols = pd.DataFrame()
testprob_cols = pd.DataFrame()
X_i = new_df.drop(['Id', 'Response', 'x > 1', 'x > 2', 'x > 3', 'x > 4', 'x > 5', 'x > 6', 'x > 7'], axis=1)
testX_i = testcopy_df.drop(['Id'], axis=1)
for i in range(1, 8):
    y_i = new_df['x > %d' %i]
    xgbmodel = xgb.XGBClassifier()
    xgbmodel.fit(X_i, y_i)
    yi_pred = xgbmodel.predict_proba(X_i)[:,1]
    yi_test = xgbmodel.predict_proba(testX_i)[:,1]
    prob_cols['x > %d' %i] = yi_pred
    testprob_cols['x > %d' %i] = yi_test

prob_cols['prob_sum'] = prob_cols.sum(axis=1)
testprob_cols['prob_sum'] = testprob_cols.sum(axis=1)
prob_cols.head(5)
testprob_cols.head()
testprob_cols.shape
prob_cols
new_df[prob_cols.columns] = prob_cols
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
newX = new_df.drop(['Id', 'Response'], axis=1)
newY = new_df.Response.values.astype('int')
newX_train, newX_valid, newY_train, newY_valid = train_test_split(newX, newY, test_size=0.2, random_state=17)
valid_cv_results = pd.DataFrame()
valid_cv_results['Real'] = newY_valid

i = 0
for train_index, test_index in cv.split(newX_train, newY_train):
    valid_cv_results['cv_'+str(i)] = 0
    print (i)
#     print(i, "TRAIN:", train_index, "TEST:", test_index)
    trainX, testX = newX.iloc[train_index], newX.iloc[test_index]
    trainY, testY = newY[train_index], newY[test_index]
    
    xgb_model = xgb.XGBClassifier(max_depth=10, objective='multi:softmax', num_class=8)
    xgb_model.fit(trainX, trainY)
#     y_pred = xgb_model.predict(newX_valid)
    valid_cv_results['cv_'+str(i)] = xgb_model.predict(newX_valid)
    i += 1

valid_cv_results['median'] = valid_cv_results[valid_cv_results.columns[1:]].median(axis=1).astype('int')
valid_cv_results
#confusion matrix
print(pd.crosstab(valid_cv_results['Real'], valid_cv_results['median'], rownames=['Actual'], colnames=['Predicted']))
print (classification_report(valid_cv_results['Real'], valid_cv_results['median']))
confusion_matrix(valid_cv_results['Real'], valid_cv_results['median'])
#Generate cost of life insurance
def generate_cost(x):
    if x['Ins_Age'] < 0.25:
        cost = 25
    elif x['Ins_Age'] < 0.35:
        cost = 30
    elif x['Ins_Age'] < 0.45:
        cost = 40
    elif x['Ins_Age'] < 0.35:
        cost = 30
    elif x['Ins_Age'] < 0.55:
        cost = 40
    elif x['Ins_Age'] < 0.7:
        cost = 80
    elif x['Ins_Age'] < 0.9:
        cost = 200
    else:
        return 600
    cost += cost * abs(x['BMI'] - 0.5)
    cost += np.random.normal(0, cost / 10)
    return cost

copy_df['Cost'] = copy_df.apply(generate_cost, axis=1)
copy_df['Cost'].hist(grid=False, bins=100, figsize=(20, 6))
copy_df['Cost'].hist(by=copy_df['Response'], figsize=(20, 10));
X2 = copy_df.drop(['Id', 'Cost', 'Response'], axis=1)
y2 = copy_df.Cost.values.astype('int')

#split into train/valid sets
X2_train, X2_valid, y2_train, y2_valid = train_test_split(X2, y2, test_size=0.2, random_state=17)
#define model
lr_cost = LinearRegression()
#cross-validation score
lr_cv_score = cross_val_score(lr_cost, X2_train, y2_train, cv=cv)
print (lr_cv_score)
#fit model
lr_cost.fit(X2_train,y2_train)
#predictiion for validation set
y2_pred = lr_cost.predict(X2_valid)
#scores
print ('R2 score:',lr_cost.score(X2_valid,y2_valid))
print('Mean Absolute Error:', mean_absolute_error(y2_valid, y2_pred))  
print('Mean Squared Error:', mean_squared_error(y2_valid, y2_pred))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y2_valid, y2_pred)))
#define model
rf_cost = RandomForestRegressor(n_estimators = 100,
                              criterion = 'mse',
                              random_state = 1,
                              n_jobs = -1)
#cross-validation score
rf_cost_cv_score = cross_val_score(rf_cost, X2_train, y2_train, cv=cv)
print (rf_cost_cv_score)
#fit model
rf_cost.fit(X2_train, y2_train)
#predictiion for validation set
y2_pred = rf_cost.predict(X2_valid)
#scores
print ('R2 score:',rf_cost.score(X2_valid,y2_valid))
print('Mean Absolute Error:', mean_absolute_error(y2_valid, y2_pred))  
print('Mean Squared Error:', mean_squared_error(y2_valid, y2_pred))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y2_valid, y2_pred)))
#define model
xgb_cost = xgb.XGBRegressor()

#cross-validation score
# xgb_cost_cv_score = cross_val_score(xgb_cost, X2_train, y2_train, cv=cv)
# print (xgb_cost_cv_score)

#fit model
xgb_cost.fit(X2_train, y2_train)

#predictiion for validation set
y2_pred = xgb_cost.predict(X2_valid)

#scores
print ('R2 score:',xgb_cost.score(X2_valid,y2_valid))
print('Mean Absolute Error:', mean_absolute_error(y2_valid, y2_pred))  
print('Mean Squared Error:', mean_squared_error(y2_valid, y2_pred))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y2_valid, y2_pred)))
losses = abs((y2_valid - y2_pred).sum())
print ("Losses because of error model: ", losses)

avoid_losses = (y2_valid - y2_pred + round(losses / len(y2_valid), 2)).sum()
print ("Losses because of error model, when add price for error: ", avoid_losses)
appr_df = copy_df.copy()

def approve(x):
    if (x['Response'] > 2):
        val=1
    else:
        val=0
    return val

# copy_df = df.copy()
appr_df['Approve'] = appr_df.apply(approve, axis=1)
appr_df['Approve'].value_counts(normalize=True)
sns.countplot(x=appr_df.Approve).set_title('Distribution of rows by response categories')
#split DataFrame into input data and target
X3 = appr_df.drop(['Id', 'Response', 'Approve'], axis=1)
y3 = appr_df.Approve
#split into train/valid sets
X3_train, X3_valid, y3_train, y3_valid = train_test_split(X3, y3, test_size=0.2, random_state=42)
#define model
rf_appr = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=42)
#get cross-validation score
rf_appr_cv_score = cross_val_score(rf_appr, X3_train, y3_train, cv=cv)
print (rf_appr_cv_score)
#train model
rf_appr.fit(X3_train, y3_train)
#predictiion for validation set
y3_pred = rf_appr.predict(X3_valid)
#confusion matrix
print(pd.crosstab(y3_valid, y3_pred, rownames=['Actual class'], colnames=['Predicted class']))
y3_pred = rf_appr.predict_proba(X3_valid)[:,1]
#classification report
print (classification_report(y3_valid, y3_pred.astype('int')))
#define model
nn_appr = Sequential()
nn_appr.add(Dense(128, activation='relu'))
nn_appr.add(Dense(64, activation='relu'))
nn_appr.add(Dense(2, activation='softmax'))
nn_appr.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
y_train_one_hot = to_categorical(y3_train, num_classes=2, dtype='float32')
nn_appr_cv_score = []

for i, (train_index, val_index) in enumerate(cv.split(X3_train.values, y_train_one_hot.argmax(1))):
    print ('Fold %i' %i)
    x_train_cv, x_val_cv = X3_train.values[train_index], X3_train.values[val_index]
    y_train_cv, y_val_cv = y_train_one_hot[train_index], y_train_one_hot[val_index]
    hist = nn_appr.fit(x_train_cv, y_train_cv, epochs=20, verbose=0, validation_data=(x_val_cv, y_val_cv))
    nn_appr_cv_score.append(hist.history['accuracy'][-1])

print (nn_appr_cv_score)
#train model
nn_appr.fit(X3_train, y_train_one_hot, epochs=50)
y3_pred = nn_appr.predict(X3_valid)
y3_pred = np.argmax(y3_pred, axis=1) #decoding
#confusion matrix
print(pd.crosstab(y3_valid, y3_pred, rownames=['Actual class'], colnames=['Predicted class']))
#classification report
print (classification_report(y3_valid, y3_pred.astype('int')))
xgb_appr = xgb.XGBClassifier()
xgb_cv_score = cross_val_score(xgb_appr, X3_train, y3_train, cv=cv)
print (xgb_cv_score)
#train model
xgb_appr.fit(X3_train, y3_train)
y3_pred = xgb_appr.predict(X3_valid)
#confusion matrix
print(pd.crosstab(y3_valid, y3_pred, rownames=['Actual class'], colnames=['Predicted class']))
#classification report
print (classification_report(y3_valid, y3_pred.astype('int')))
cv = StratifiedKFold(n_splits=20, random_state=42, shuffle=True)
y3_train = y3_train.reset_index(drop=True)
rf_valid_cv_appr_results = pd.DataFrame()
rf_valid_cv_appr_results['Real'] = y3_valid

i = 0
for train_index, test_index in cv.split(X3_train, y3_train):
    rf_valid_cv_appr_results['cv_'+str(i)] = 0
    print(i, "TRAIN:", train_index, "TEST:", test_index)
    trainX, testX = X3_train.iloc[train_index], X3_train.iloc[test_index]
    trainY, testY = y3_train[train_index], y3_train[test_index]
    eec = EasyEnsembleClassifier(random_state=42)
    eec.fit(trainX, trainY)
    y3_pred = eec.predict_proba(X3_valid)[:,1]
    rf_valid_cv_appr_results['cv_'+str(i)] = y3_pred
    i += 1
rf_valid_cv_appr_results['avg'] = rf_valid_cv_appr_results[rf_valid_cv_appr_results.columns[1:]].median(axis=1).round()
print(pd.crosstab(rf_valid_cv_appr_results['Real'], rf_valid_cv_appr_results['avg'], rownames=['Actual class'], colnames=['Predicted class']))
print (classification_report(rf_valid_cv_appr_results['Real'], rf_valid_cv_appr_results['avg']))
precisions, recalls, thresholds = precision_recall_curve(y3_valid, y3_pred)

plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.legend(loc="lower right", fontsize=16)
plt.xlabel("Threshold", fontsize=16)
plt.grid(True)
plt.show()

plt.plot(recalls, precisions, "b-", linewidth=2)
plt.xlabel("Recall", fontsize=16)
plt.ylabel("Precision", fontsize=16)
plt.grid(True)
plt.show()

fpr, tpr, thresholds = roc_curve(y3_valid, y3_pred)
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
plt.ylabel('True Positive Rate (Recall)', fontsize=16)
plt.grid(True)
plt.show()

roc_auc_score = roc_auc_score(y3_valid, y3_pred.round())
print (roc_auc_score)
res = [1 if x > 0.49655 else 0 for x in y3_pred]
len(res)
print (classification_report(rf_valid_cv_appr_results['Real'], res))
confusion_matrix(rf_valid_cv_appr_results['Real'], res)
print(pd.crosstab(rf_valid_cv_appr_results['Real'], res, rownames=['Actual class'], colnames=['Predicted class']))

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer, FunctionTransformer
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score, classification_report
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
pd.set_option('display.max_columns', None)
train = pd.read_csv("/kaggle/input/av-janatahack-crosssell-prediction/train.csv")
test = pd.read_csv("/kaggle/input/av-janatahack-crosssell-prediction/test.csv")
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)
sample = pd.read_csv("/kaggle/input/av-janatahack-crosssell-prediction/sample.csv")
train.info()
train.head()
train.isna().sum()
test.isna().sum()
for col in train.columns:
    print(f"{col} : {train[col].nunique()}")
    print(train[col].unique())
#separating continuous and categorical variables
cat_var = ["Gender","Driving_License","Previously_Insured","Vehicle_Age","Vehicle_Damage"]
con_var = list(set(train.columns).difference(cat_var+["Response"]))
train.Response.value_counts(normalize=True)
sns.countplot(train.Response)
plt.title("Class count")
plt.show()
sns.pairplot(train, hue='Response', diag_kind='hist')
plt.show()
def map_val(data):
    data["Gender"] = data["Gender"].replace({"Male":1, "Female":0})
    data["Vehicle_Age"] = data["Vehicle_Age"].replace({'> 2 Years':2, '1-2 Year':1, '< 1 Year':0 })
    data["Vehicle_Damage"] = data["Vehicle_Damage"].replace({"Yes":1, "No":0})
    return data

train = map_val(train)
test = map_val(test)
fig, ax = plt.subplots(2,3 , figsize=(16,6))
ax = ax.flatten()
i = 0
for col in cat_var:
    sns.pointplot(col, 'Response', data=train, ax = ax[i])
    i+=1
plt.tight_layout()
plt.show()
sns.catplot('Gender', 'Response',hue='Vehicle_Age', row = 'Previously_Insured',col='Vehicle_Damage',data=train, kind='point', height=3, aspect=2)
plt.show()
fig, ax = plt.subplots(2,3 , figsize=(16,6))
ax = ax.flatten()
i = 0
for col in con_var:
    sns.boxplot( 'Response', col, data=train, ax = ax[i])
    i+=1
plt.tight_layout()
plt.show()
sns.catplot('Gender', 'Vintage',hue='Response', row = 'Previously_Insured',col='Vehicle_Damage',data=train, kind='box', height=3, aspect=2)
plt.show()
sns.catplot('Gender', 'Age',hue='Response', row = 'Previously_Insured',col='Vehicle_Damage',data=train, kind='box', height=3, aspect=2)
plt.show()
sns.catplot('Gender', 'Annual_Premium',hue='Response', row = 'Previously_Insured',col='Vehicle_Damage',data=train, kind='box', height=3, aspect=2)
plt.show()
plt.figure(figsize=(30,5))
sns.heatmap(pd.crosstab([train['Previously_Insured'], train['Vehicle_Damage']], train['Region_Code'],
                        values=train['Response'], aggfunc='mean', normalize='columns'), annot=True, cmap='inferno')
plt.show()
corr = train.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)]=True
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='YlGnBu', mask=mask)
plt.title("Correlation Heatmap")
plt.show()
train.skew()
train['log_premium'] = np.log(train.Annual_Premium)
train['log_age'] = np.log(train.Age)
test['log_premium'] = np.log(test.Annual_Premium)
test['log_age'] = np.log(test.Age)
train.groupby(['Previously_Insured','Gender'])['log_premium'].plot(kind='kde')
plt.show()
train.groupby(['Previously_Insured','Gender'])['log_age'].plot(kind='kde')
plt.show()
def feature_engineering(data, col):
    mean_age_insured = data.groupby(['Previously_Insured','Vehicle_Damage'])[col].mean().reset_index()
    mean_age_insured.columns = ['Previously_Insured','Vehicle_Damage','mean_'+col+'_insured']
    mean_age_gender = data.groupby(['Previously_Insured','Gender'])[col].mean().reset_index()
    mean_age_gender.columns = ['Previously_Insured','Gender','mean_'+col+'_gender']
    mean_age_vehicle = data.groupby(['Previously_Insured','Vehicle_Age'])[col].mean().reset_index()
    mean_age_vehicle.columns = ['Previously_Insured','Vehicle_Age','mean_'+col+'_vehicle']
    data = data.merge(mean_age_insured, on=['Previously_Insured','Vehicle_Damage'], how='left')
    data = data.merge(mean_age_gender, on=['Previously_Insured','Gender'], how='left')
    data = data.merge(mean_age_vehicle, on=['Previously_Insured','Vehicle_Age'], how='left')
    data[col+'_mean_insured'] = data['log_age']/data['mean_'+col+'_insured']
    data[col+'_mean_gender'] = data['log_age']/data['mean_'+col+'_gender']
    data[col+'_mean_vehicle'] = data['log_age']/data['mean_'+col+'_vehicle']
    data.drop(['mean_'+col+'_insured','mean_'+col+'_gender','mean_'+col+'_vehicle'], axis=1, inplace=True)
    return data

train = feature_engineering(train, 'log_age')
test = feature_engineering(test, 'log_age')

train = feature_engineering(train, 'log_premium')
test = feature_engineering(test, 'log_premium')

train = feature_engineering(train, 'Vintage')
test = feature_engineering(test, 'Vintage')
X = train.drop(["Response"], axis=1)
Y = train["Response"]
dummy = ["Vehicle_Age"]
passthru = con_var = list(set(X.columns).difference(dummy))

onehot = OneHotEncoder(handle_unknown='ignore')
label = OrdinalEncoder()
scaler = StandardScaler()

feat_rf = RandomForestClassifier(n_jobs=4, random_state=1, class_weight='balanced_subsample')
feat_xgb = XGBClassifier(n_jobs=4, random_state=1, objective='binary:logistic')
selector_rf = SelectFromModel(feat_xgb, threshold=0.001)

transformers_onehot = [('pass','passthrough',passthru),
                       ('onehot', onehot, dummy) ]
ct_onehot = ColumnTransformer( transformers=transformers_onehot )

transformers_label = [('pass','passthrough',passthru),
                      ('onehot', label, dummy) ]
ct_label = ColumnTransformer( transformers=transformers_label )

pipe = Pipeline([('ct', ct_onehot),
                 ('scaler', scaler)])
poly = PolynomialFeatures(degree= 2, interaction_only=True)
pca = PCA(n_components=0.99)
kbest = SelectKBest(k=6)

pipe_pca = Pipeline([('ct', ct_onehot),
                      ('poly', poly),
                      ('scaler', scaler),
                      ('pca',pca)])

pipe_kbest = Pipeline([('ct', ct_onehot),
                       ('poly', poly),
                       ('scaler', scaler),
                       ('kbest',kbest)])

pipe_union = FeatureUnion([('pca',pipe_pca),
                           ('kbest',pipe_kbest)])
# merging the PCA components and KBest features from the data
pipe_union.fit(X, Y)
X_union = pipe_union.transform(X)
test_union = pipe_union.transform(test)
#np.cumsum(pipe_union.transformer_list[0][1].named_steps['pca'].explained_variance_ratio_)
ct_onehot.fit(X)
categories = ct_onehot.named_transformers_['onehot'].categories_
onehot_cols = [col+"_"+str(cat) for col,cats in zip(dummy, categories) for cat in cats]
all_columns = passthru + onehot_cols

X_transform = pd.DataFrame(pipe.fit_transform(X), columns = all_columns)
test_transform = pd.DataFrame(pipe.transform(test), columns = all_columns)

selector_rf.fit(X_transform, Y)
rf_cols = [col for col, flag in zip(X_transform.columns, selector_rf.get_support()) if flag]
print(rf_cols)
X_select = pd.DataFrame(selector_rf.transform(X_transform), columns = rf_cols)
test_select = pd.DataFrame(selector_rf.transform(test_transform), columns = rf_cols)
def submission(preds, model):
    sample["Response"] = preds
    sample.to_csv("model_"+model+".csv", index=False)
model_lr = LogisticRegression(n_jobs=4, random_state=1, class_weight='balanced')
model_rfc = RandomForestClassifier(n_jobs=4, random_state=1, class_weight='balanced_subsample')
# scale pos weight for class imbalance
model_xgb = XGBClassifier(n_jobs=4, random_state=1, scale_pos_weight=7, objective='binary:logistic')
model_lgbm = LGBMClassifier(n_jobs=4, random_state=1, is_unbalance=True, objective='binary')
model_cat = CatBoostClassifier(random_state=1, verbose=0, scale_pos_weight=7, custom_metric=['AUC'])

models = []
models.append(("LR",model_lr))
models.append(("RF",model_rfc))
models.append(("XGB",model_xgb))
models.append(("LGBM",model_lgbm))
models.append(("CAT",model_cat))

cv = StratifiedShuffleSplit(n_splits=5, random_state=1, train_size=0.8)
results = []
names = []
for name, model in models:
    print("Training..."+name)
    scores = cross_val_score(model, X_select, Y, scoring='roc_auc', n_jobs=-1, cv = cv, verbose=0)
    results.append(scores)
    names.append(name)
    print("Model %s mean score : %.4f variance error: %.4f"%(name, np.mean(scores), np.std(scores)))
plt.boxplot(results)
plt.xticks(np.arange(1,len(names)+1), names)
plt.title("Model comparison")
plt.show()
results_union = []
names = []
for name, model in models:
    print("Training..."+name)
    scores = cross_val_score(model, X_union, Y, scoring='roc_auc', n_jobs=-1, cv = cv, verbose=0)
    results_union.append(scores)
    names.append(name)
    print("Model %s mean score : %.4f variance error: %.4f"%(name, np.mean(scores), np.std(scores)))
plt.boxplot(results_union)
plt.xticks(np.arange(1,len(names)+1), names)
plt.title("Model comparison")
plt.show()
def eval_model(model, x, Y):
    model.fit(x, Y)

    trainpred  = model.predict(x)
    proba = model.predict_proba(x)[:,1]

    print("Accuracy score : %.4f"%accuracy_score(Y, trainpred))
    print("ROC AUC score : %.4f"%roc_auc_score(Y, proba))
    print("Classification report")
    print(classification_report(Y, trainpred))
    
def metrics_score(model, X, Y):
    pred = model.predict_proba(X)[:,1]
    print("ROC AUC score : %.4f"%roc_auc_score(Y, pred))
model_xgb = XGBClassifier(n_jobs=4, random_state=1, scale_pos_weight=7, objective='binary:logistic')
model_lgbm = LGBMClassifier(n_jobs=4, random_state=1, is_unbalance=True, objective='binary')
model_cat = CatBoostClassifier(random_state=1, verbose=0, scale_pos_weight=7, custom_metric=['AUC'])
model_xgb.fit(X_select, Y)
model_lgbm.fit(X_select, Y)
model_cat.fit(X_select, Y)

pred_xgb = model_xgb.predict_proba(test_select)[:,1]
pred_lgbm = model_lgbm.predict_proba(test_select)[:,1]
pred_cat = model_cat.predict_proba(test_select)[:,1]

submission(pred_xgb, 'xgb')
submission(pred_lgbm, 'lgbm')
submission(pred_cat, 'cat')

prediction = np.mean((pred_xgb, pred_lgbm, pred_cat), axis=0)
submission(prediction, 'all')

metrics_score(model_xgb, X_select, Y)
metrics_score(model_lgbm, X_select, Y)
metrics_score(model_cat, X_select, Y)

cv = StratifiedShuffleSplit(n_splits=10, random_state=1, train_size=0.7)
predictions_lgbm = []

for train_index, test_index in cv.split(X_select, Y):
    xtrain, xtest = X_select.iloc[train_index], X_select.iloc[test_index]
    ytrain, ytest = Y[train_index], Y[test_index]
    
    model_lgbm.fit(xtrain, ytrain)
    trainpred = model_lgbm.predict_proba(xtrain)[:,1]
    testpred = model_lgbm.predict_proba(xtest)[:,1]
    print("Train ROC AUC : %.4f Test ROC AUC : %.4f"%(roc_auc_score(ytrain, trainpred),roc_auc_score(ytest, testpred)))
    prediction = model_lgbm.predict_proba(test_select)[:,1]
    predictions_lgbm.append(prediction)
submission(np.mean(predictions_lgbm, axis=0), 'lgbm_stack')
# run this again
#cv = StratifiedShuffleSplit(n_splits=5, random_state=1, train_size=0.9)
cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
predictions_xgb = []

for train_index, test_index in cv.split(X_select, Y):
    xtrain, xtest = X_select.iloc[train_index], X_select.iloc[test_index]
    ytrain, ytest = Y[train_index], Y[test_index]
    
    model_xgb.fit(xtrain, ytrain)
    trainpred = model_xgb.predict_proba(xtrain)[:,1]
    testpred = model_xgb.predict_proba(xtest)[:,1]
    print("Train ROC AUC : %.4f Test ROC AUC : %.4f"%(roc_auc_score(ytrain, trainpred),roc_auc_score(ytest, testpred)))
    prediction = model_xgb.predict_proba(test_select)[:,1]
    predictions_xgb.append(prediction)
submission(np.mean(predictions_xgb, axis=0), 'xgb_stack')
cv = StratifiedShuffleSplit(n_splits=10, random_state=1, train_size=0.7)
predictions_cat = []

for train_index, test_index in cv.split(X_select, Y):
    xtrain, xtest = X_select.iloc[train_index], X_select.iloc[test_index]
    ytrain, ytest = Y[train_index], Y[test_index]
    
    model_cat.fit(xtrain, ytrain)
    trainpred = model_cat.predict_proba(xtrain)[:,1]
    testpred = model_cat.predict_proba(xtest)[:,1]
    print("Train ROC AUC : %.4f Test ROC AUC : %.4f"%(roc_auc_score(ytrain, trainpred),roc_auc_score(ytest, testpred)))
    prediction = model_cat.predict_proba(test_select)[:,1]
    predictions_cat.append(prediction)
submission(np.mean(predictions_cat, axis=0), 'cat_stack')

def plot_nn(history, metric):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,5))
    ax1.plot(history.history['loss'], color='r', label='Train loss')
    ax1.plot(history.history['val_loss'], color='g', label='Validation loss')
    ax1.legend()

    ax2.plot(history.history[metric], color='r', label='Train '+metric)
    ax2.plot(history.history['val_'+metric], color='g', label='Validation '+metric)
    ax2.legend()
    
    plt.show()
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
inputs = X_select.shape[1]

es = EarlyStopping(monitor='val_loss', min_delta=0.01, patience = 50, mode='auto', baseline=0.85, restore_best_weights=True)

optimizer = Adam(learning_rate=0.01)
model = Sequential()
model.add( Dense( 64, input_dim = inputs, activation='relu', kernel_initializer='random_normal'))
model.add( Dense( 128, input_dim = inputs, activation='relu', kernel_initializer='random_normal'))
model.add( Dense( 256, input_dim = inputs, activation='relu', kernel_initializer='random_normal'))
model.add( Dropout(0.01))
model.add( Dense(1, activation='sigmoid'))

model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics = ['AUC'])
history = model.fit(X_select, Y, batch_size=128, epochs = 20, validation_split=0.3, verbose=0)
plot_nn(history, 'auc')
pred_nn = model.predict_proba(test_select)
submission(pred_nn, 'nn')
pred_stack = np.mean((pred_xgb, pred_lgbm, pred_cat, pred_nn[:,0]), axis=0)
submission(pred_stack, 'stack')

from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier,EasyEnsembleClassifier
from imblearn.metrics import classification_report_imbalanced
model_bbag = BalancedBaggingClassifier(n_jobs=4, random_state=1, base_estimator=model_xgb)
model_brf = BalancedRandomForestClassifier(n_jobs=4, random_state=1, class_weight='balanced')
model_easy = EasyEnsembleClassifier(n_jobs=4, random_state=1, base_estimator=model_xgb)

imb_models = []
imb_models.append(('Bag', model_bbag))
imb_models.append(('BagRF', model_brf))
imb_models.append(('Easy', model_easy))

cv = StratifiedShuffleSplit(n_splits=5, random_state=1, train_size=0.8)
results_imb = []
names_imb = []
for name, model in imb_models:
    print("Training..."+name)
    scores = cross_val_score(model, X_select, Y, scoring='roc_auc', n_jobs=-1, cv = cv, verbose=0)
    results_imb.append(scores)
    names_imb.append(name)
    print("Model %s mean score : %.4f variance error: %.4f"%(name, np.mean(scores), np.std(scores)))
plt.boxplot(results_imb)
plt.xticks(np.arange(1,len(names_imb)+1), names_imb)
plt.title("Model comparison Imblearn")
plt.show()
model_bbag.fit(X_select, Y)
model_brf.fit(X_select, Y)
model_easy.fit(X_select, Y)

metrics_score(model_bbag, X_select, Y)
metrics_score(model_brf, X_select, Y)
metrics_score(model_easy, X_select, Y)

pred_bbag = model_bbag.predict_proba(test_select)[:,1]
pred_brf = model_brf.predict_proba(test_select)[:,1]
pred_easy = model_easy.predict_proba(test_select)[:,1]
submission(pred_bbag, 'imb_bbag')
submission(pred_brf, 'imb_brf')
submission(pred_easy, 'imb_easy')

pred_imb_stack = np.mean((pred_bbag, pred_brf, pred_easy), axis=0)
submission(pred_imb_stack, 'imb_stack')

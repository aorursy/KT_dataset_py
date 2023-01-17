import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

df_labels = pd.read_csv('../input/train_labels.csv')
df_values = pd.read_csv('../input/train_values.csv')
test = pd.read_csv('../input/test_values.csv')

print('Train Label columns:', list(df_labels.columns.values))
print('Train Values columns:', list(df_values.columns.values))
print('Test columns:', list(test.columns.values))


ax = sns.countplot(df_labels['heart_disease_present'],label="Count")

print('Training values stats', df_values.info())
print('Test values stats', test.info())

df_values.head()

print('Slope of peak:', np.unique(df_values['slope_of_peak_exercise_st_segment']))
print('Thal:', np.unique(df_values['thal']))
print('Chest pain type:', np.unique(df_values['chest_pain_type']))
print('Major vessels:', np.unique(df_values['num_major_vessels']))
print('Blood sugar gt:', np.unique(df_values['fasting_blood_sugar_gt_120_mg_per_dl']))
print('Resting ekg results:', np.unique(df_values['resting_ekg_results']))
print('Exercise induced angina:', np.unique(df_values['exercise_induced_angina']))

df = pd.merge(df_values, df_labels, on='patient_id')
df.head(10)
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
ax.set_title('Correlations')
plt.show()
rawTrain = pd.read_csv('../input/train_values.csv', index_col=0)
rawTest = pd.read_csv('../input/test_values.csv', index_col=0)

# change categorical data into one-hot:
trainSlope_oneHot = pd.get_dummies(rawTrain['slope_of_peak_exercise_st_segment'], prefix='slope')
trainThal_oneHot = pd.get_dummies(rawTrain['thal'])
trainChestPain_oneHot = pd.get_dummies(rawTrain['chest_pain_type'], prefix='chestPain')
trainResting_oneHot = pd.get_dummies(rawTrain['resting_ekg_results'], prefix='restingEkg')
testSlope_oneHot = pd.get_dummies(rawTest['slope_of_peak_exercise_st_segment'], prefix='slope')
testThal_oneHot = pd.get_dummies(rawTest['thal'])
testChestPain_oneHot = pd.get_dummies(rawTest['chest_pain_type'], prefix='chestPain')
testResting_oneHot = pd.get_dummies(rawTest['resting_ekg_results'], prefix='restingEkg')

rawTrain.drop(['slope_of_peak_exercise_st_segment','thal','chest_pain_type','resting_ekg_results'], axis=1, inplace=True)
rawTrain = rawTrain.join([trainSlope_oneHot, trainThal_oneHot, trainChestPain_oneHot, trainResting_oneHot])
rawTest.drop(['slope_of_peak_exercise_st_segment','thal','chest_pain_type','resting_ekg_results'], axis=1, inplace=True)
rawTest = rawTest.join([testSlope_oneHot, testThal_oneHot, testChestPain_oneHot, testResting_oneHot])

numCols = ['resting_blood_pressure', 'serum_cholesterol_mg_per_dl', 'oldpeak_eq_st_depression', 'max_heart_rate_achieved']

rawTrain.to_csv('../train_values_normalized.csv')
rawTest.to_csv('../test_values_normalized.csv')

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(rawTrain.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
ax.set_title('Correlations')
plt.show()
rawTrain.head()
# for preprocessing the data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
# the model
from sklearn.linear_model import LogisticRegression
# for combining the preprocess with model training
from sklearn.pipeline import Pipeline
# for optimizing parameters of the pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss

y = pd.read_csv('../input/train_labels.csv', index_col=0)
X_test = rawTest.copy()

print(rawTrain.columns)
notallFeatures = rawTrain.copy()
del notallFeatures['age']
del notallFeatures['resting_blood_pressure']

X_test2 = X_test.copy()
del X_test2['age']
del X_test2['resting_blood_pressure']
#del X_test2['max_heart_rate_achieved']
pipe = Pipeline(steps=[('scale', MaxAbsScaler()), 
                       ('logistic', LogisticRegression())])

param_grid = {'logistic__C': [ 3], 
              'logistic__penalty': ['l1', 'l2']}
gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  cv=3)

gs.fit(notallFeatures,np.ravel(y))


in_sample_preds = gs.predict_proba(notallFeatures)

plt.plot(np.sort(in_sample_preds[:,1]))
plt.ylabel('output (P(x=1))')
plt.xlabel('train set samples sorted by output')
plt.title('Distribution of output of train set')
plt.grid(b=1)


print(log_loss(y, in_sample_preds))
res = gs.predict_proba(X_test2)[:,1]
d = {'heart_disease_present': res}
submission_df = pd.DataFrame(data=d, index=X_test2.index)
submission_df.to_csv("logreg_res_v2")

submission_df.head()


weights, params = [], []
for c in np.arange(-5, 5, dtype=np.float):
    lr = LogisticRegression(C=10**c, random_state=0, solver = 'liblinear')
    lr.fit(notallFeatures,np.ravel(y))
    weights.append(lr.coef_[0])
    params.append(10**c)
weights = np.array(weights)

for i in np.arange(0, 19, 1):
    plt.plot(params, weights[:, i], linestyle = '--')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain, np.ravel(y))
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain)
    dtrain_predprob = alg.predict_proba(dtrain)[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, dtrain, np.ravel(y), cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(np.ravel(y), dtrain_predprob))
    
    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        

#Choose all predictors except target & IDcols
notallFeatures2 = rawTrain.copy()
X_test3 =X_test.copy()
predictors = [x for x in notallFeatures2.columns]
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, notallFeatures2, predictors)

import seaborn as sns

for i in notallFeatures2.columns:
    ax = sns.countplot(x=i, data=notallFeatures2)

    plt.show()



# ------
del notallFeatures2['age']
del notallFeatures2['resting_blood_pressure']
del notallFeatures2['fasting_blood_sugar_gt_120_mg_per_dl']
del X_test3['age']
del X_test3['resting_blood_pressure']
del X_test3['fasting_blood_sugar_gt_120_mg_per_dl']
scaler1 = StandardScaler()
rescaled_notallFeatures2 = scaler1.fit_transform(notallFeatures2)
rescaled_X_test3 = scaler1.fit_transform(X_test3)
# ------
param_test1 = {'n_estimators':range(20,81,10), 'learning_rate':np.arange(0.5,1.3,0.05)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(random_state=0, max_depth = 1), 
param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(notallFeatures2, np.ravel(y))
gsearch1.best_params_, gsearch1.best_score_


from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import log_loss
from sklearn.linear_model import SGDClassifier

notallFeatures2 = rawTrain.copy()
notallFeatures3 = rawTrain.copy()
X_test3 =X_test.copy()

del notallFeatures2['age']
del notallFeatures2['resting_blood_pressure']
del notallFeatures2['fasting_blood_sugar_gt_120_mg_per_dl']
#del notallFeatures2['restingEkg_1'] #new
#del notallFeatures2['fixed_defect'] #new
#del notallFeatures2['slope_3'] #new

del X_test3['age']
del X_test3['resting_blood_pressure']
del X_test3['fasting_blood_sugar_gt_120_mg_per_dl']
#del X_test3['restingEkg_1'] #new
#del X_test3['fixed_defect'] #new
#del X_test3['slope_3'] #new
scaler1 = StandardScaler()
rescaled_notallFeatures2 = scaler1.fit_transform(notallFeatures2)
rescaled_X_test3 = scaler1.fit_transform(X_test3)
print(rescaled_notallFeatures2.shape)
modelGBC = GradientBoostingClassifier(n_estimators=30, learning_rate=0.9, max_depth=1, random_state=0, max_features  = 2) # min_samples_leaf = 0.3

# Train the model using the training sets and check score
modelGBC.fit(rescaled_notallFeatures2,np.ravel(y))


#Predict Output
predictedGBC= modelGBC.predict(rescaled_X_test3)
predictedGBC2 = modelGBC.predict_proba(rescaled_X_test3)[:,1]

d2 = {'heart_disease_present': predictedGBC2}
submission_df2 = pd.DataFrame(data=d2, index=X_test3.index)
submission_df2.to_csv("GBC")
print(submission_df2.head())
submission_df2.tail()
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from random import randint
import random
from sklearn.linear_model import SGDClassifier


Train_Df = rawTrain.copy()
Test_Df = X_test.copy()
#del Train_Df['age']
#del Train_Df['resting_blood_pressure']
#del Train_Df['fasting_blood_sugar_gt_120_mg_per_dl']
del Train_Df['restingEkg_1'] #new
del Train_Df['fixed_defect'] #new
del Train_Df['slope_3'] #new
#del Test_Df['age']
#del Test_Df['resting_blood_pressure']
#del Test_Df['fasting_blood_sugar_gt_120_mg_per_dl']
del Test_Df['restingEkg_1'] #new
del Test_Df['fixed_defect'] #new
del Test_Df['slope_3'] #new
Train_Df = scaler1.fit_transform(Train_Df)
Test_Df = scaler1.fit_transform(Test_Df)
randsample = random.sample(range(1, 101), 50)


'''for j in np.arange(0.1,1.5, 0.05):
    loss = []
    for i in randsample:
        XX_train, XX_test, yy_train, yy_test = train_test_split(Train_Df, y, test_size=0.33, random_state=i)
        mgbc = GradientBoostingClassifier(n_estimators=25, learning_rate=j, max_depth=1, random_state=0)
        # Train the model using the training sets and check score
        mgbc.fit(XX_train,np.ravel(yy_train))
        pmgbc = mgbc.predict_proba(XX_test)
        loss_r = log_loss(yy_test, pmgbc)
        loss.append(loss_r)
    print('Loss shape: ', len(loss), ' mean loss: ', np.mean(loss), ' learning rate: ', j)'''

'''for j in np.arange(0.1, 1.5, 0.1):
    loss = []
    for i in randsample:
        XX_train, XX_test, yy_train, yy_test = train_test_split(Train_Df, y, test_size=0.33, random_state=i)
        mgbc = GradientBoostingClassifier(n_estimators=30, learning_rate=j, max_depth=1, random_state=0, min_samples_leaf = 0.3, max_features  = 2)
        # Train the model using the training sets and check score
        mgbc.fit(XX_train,np.ravel(yy_train))
        pmgbc = mgbc.predict_proba(XX_test)
        loss_r = log_loss(yy_test, pmgbc)
        loss.append(loss_r)
    print('Loss shape: ', len(loss), ' mean loss: ', np.mean(loss), ' number of estimators: ', j)'''

for j in np.arange(50, 100, 10):
    loss = []
    for i in randsample:
        XX_train, XX_test, yy_train, yy_test = train_test_split(Train_Df, y, test_size=0.33, random_state=i)
        msgd = SGDClassifier(loss='log', penalty='elasticnet', alpha=0.1, l1_ratio=0.1, fit_intercept = False, max_iter=300)
        # Train the model using the training sets and check score
        msgd.fit(XX_train,np.ravel(yy_train))
        pmsgd = msgd.predict_proba(XX_test)
        loss_r = log_loss(yy_test, pmsgd)
        loss.append(loss_r)
    print('Loss shape: ', len(loss), ' mean loss: ', np.mean(loss), ' number of estimators: ', j)

#print('Mean loss: ', np.mean(loss))
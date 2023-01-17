import numpy as np 
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
%matplotlib inline 
import matplotlib.pyplot as plt  
import time
KAGGLE_DIR = '../input/'

df = pd.read_csv(KAGGLE_DIR + "train/train.csv")
df['is_Set'] = 'train'
test_df = pd.read_csv(KAGGLE_DIR + "test/test.csv")
test_df['is_Set'] = 'test'
both = pd.concat([df, test_df], ignore_index=True)
n_train = both['is_Set'].value_counts()[0]
n_train
summary_df = pd.concat([pd.DataFrame(both.columns), pd.DataFrame(both.values.reshape(-1,1)),
                        pd.DataFrame(both.isnull().sum().values), pd.DataFrame([both[name].nunique() for name in both.columns])],
                       axis=1)
summary_df.columns = ['Variable Name', 'Quick Look', 'Nulls', 'Unique Values']
summary_df.head(25)
both['Age_years'] = both['Age']/12
both['Age_years'] = both['Age_years'].apply(np.round)
pd.cut(both['Age_years'], bins=[0,1,2,3,4,6,10,20,300]).value_counts(dropna=False)
dummies_age = pd.get_dummies(pd.cut(both['Age'], 
                                    bins=[0,.5,1,2,3,4,6,10,20,300]), dummy_na=True)
dummies_age.columns = ['is_age_0',
                       'is_age_1', 
                       'is_age_2',
                       'is_age_3',
                       'is_age_4',
                       'is_age_5_6',
                       'is_age_7_10',
                       'is_age_11_20',
                       'is_age_over_20',
                       'is_age_nan']
dummy_cols = ['Type','Breed1', 'Breed2', 'Gender', 'Color1',
       'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Vaccinated',
       'Dewormed', 'Sterilized', 'Health','State',
       'VideoAmt', 'PhotoAmt']
get_them_dummies = pd.get_dummies(both, columns = dummy_cols)
df_with_dummies = pd.concat([get_them_dummies,dummies_age],axis=1)
df_with_dummies.head()
#get the column names for variable selection
df_with_dummies.columns.values
var_selected = [
       'Type_1', 'Type_2', 'Breed1_0',
       'Breed1_1', 'Breed1_2', 'Breed1_3', 'Breed1_5', 'Breed1_6',
       'Breed1_7', 'Breed1_10', 'Breed1_11', 'Breed1_14', 'Breed1_15',
       'Breed1_16', 'Breed1_17', 'Breed1_18', 'Breed1_19', 'Breed1_20',
       'Breed1_21', 'Breed1_23', 'Breed1_24', 'Breed1_25', 'Breed1_26',
       'Breed1_31', 'Breed1_32', 'Breed1_39', 'Breed1_42', 'Breed1_44',
       'Breed1_49', 'Breed1_50', 'Breed1_51', 'Breed1_56', 'Breed1_58',
       'Breed1_60', 'Breed1_61', 'Breed1_64', 'Breed1_65', 'Breed1_69',
       'Breed1_70', 'Breed1_71', 'Breed1_72', 'Breed1_75', 'Breed1_76',
       'Breed1_78', 'Breed1_81', 'Breed1_82', 'Breed1_83', 'Breed1_85',
       'Breed1_88', 'Breed1_93', 'Breed1_94', 'Breed1_97', 'Breed1_98',
       'Breed1_99', 'Breed1_100', 'Breed1_102', 'Breed1_103', 'Breed1_104',
       'Breed1_105', 'Breed1_108', 'Breed1_109', 'Breed1_111',
       'Breed1_112', 'Breed1_114', 'Breed1_116', 'Breed1_117',
       'Breed1_119', 'Breed1_122', 'Breed1_123', 'Breed1_125',
       'Breed1_126', 'Breed1_128', 'Breed1_129', 'Breed1_130',
       'Breed1_132', 'Breed1_139', 'Breed1_141', 'Breed1_142',
       'Breed1_143', 'Breed1_145', 'Breed1_146', 'Breed1_147',
       'Breed1_148', 'Breed1_150', 'Breed1_152', 'Breed1_154',
       'Breed1_155', 'Breed1_165', 'Breed1_167', 'Breed1_169',
       'Breed1_173', 'Breed1_176', 'Breed1_178', 'Breed1_179',
       'Breed1_182', 'Breed1_185', 'Breed1_187', 'Breed1_188',
       'Breed1_189', 'Breed1_190', 'Breed1_192', 'Breed1_195',
       'Breed1_197', 'Breed1_199', 'Breed1_200', 'Breed1_201',
       'Breed1_202', 'Breed1_203', 'Breed1_204', 'Breed1_205',
       'Breed1_206', 'Breed1_207', 'Breed1_212', 'Breed1_213',
       'Breed1_214', 'Breed1_215', 'Breed1_217', 'Breed1_218',
       'Breed1_222', 'Breed1_224', 'Breed1_227', 'Breed1_228',
       'Breed1_231', 'Breed1_232', 'Breed1_233', 'Breed1_234',
       'Breed1_237', 'Breed1_239', 'Breed1_240', 'Breed1_241',
       'Breed1_242', 'Breed1_243', 'Breed1_244', 'Breed1_245',
       'Breed1_246', 'Breed1_247', 'Breed1_248', 'Breed1_249',
       'Breed1_250', 'Breed1_251', 'Breed1_252', 'Breed1_253',
       'Breed1_254', 'Breed1_256', 'Breed1_257', 'Breed1_258',
       'Breed1_260', 'Breed1_262', 'Breed1_263', 'Breed1_264',
       'Breed1_265', 'Breed1_266', 'Breed1_267', 'Breed1_268',
       'Breed1_269', 'Breed1_270', 'Breed1_271', 'Breed1_272',
       'Breed1_273', 'Breed1_274', 'Breed1_276', 'Breed1_277',
       'Breed1_278', 'Breed1_279', 'Breed1_280', 'Breed1_281',
       'Breed1_282', 'Breed1_283', 'Breed1_284', 'Breed1_285',
       'Breed1_286', 'Breed1_287', 'Breed1_288', 'Breed1_289',
       'Breed1_290', 'Breed1_292', 'Breed1_293', 'Breed1_294',
       'Breed1_295', 'Breed1_296', 'Breed1_297', 'Breed1_298',
       'Breed1_299', 'Breed1_300', 'Breed1_301', 'Breed1_302',
       'Breed1_303', 'Breed1_304', 'Breed1_305', 'Breed1_306',
       'Breed1_307', 'Breed2_0', 'Breed2_1', 'Breed2_2', 'Breed2_4',
       'Breed2_5', 'Breed2_10', 'Breed2_14', 'Breed2_16', 'Breed2_17',
       'Breed2_18', 'Breed2_19', 'Breed2_20', 'Breed2_21', 'Breed2_24',
       'Breed2_25', 'Breed2_26', 'Breed2_36', 'Breed2_39', 'Breed2_40',
       'Breed2_44', 'Breed2_49', 'Breed2_50', 'Breed2_51', 'Breed2_58',
       'Breed2_60', 'Breed2_62', 'Breed2_65', 'Breed2_69', 'Breed2_70',
       'Breed2_72', 'Breed2_75', 'Breed2_76', 'Breed2_78', 'Breed2_83',
       'Breed2_91', 'Breed2_96', 'Breed2_98', 'Breed2_100', 'Breed2_102',
       'Breed2_103', 'Breed2_104', 'Breed2_109', 'Breed2_111',
       'Breed2_115', 'Breed2_117', 'Breed2_119', 'Breed2_122',
       'Breed2_128', 'Breed2_129', 'Breed2_130', 'Breed2_141',
       'Breed2_146', 'Breed2_147', 'Breed2_150', 'Breed2_152',
       'Breed2_155', 'Breed2_159', 'Breed2_165', 'Breed2_167',
       'Breed2_168', 'Breed2_169', 'Breed2_173', 'Breed2_176',
       'Breed2_178', 'Breed2_179', 'Breed2_182', 'Breed2_187',
       'Breed2_188', 'Breed2_189', 'Breed2_190', 'Breed2_192',
       'Breed2_195', 'Breed2_200', 'Breed2_201', 'Breed2_202',
       'Breed2_203', 'Breed2_204', 'Breed2_205', 'Breed2_206',
       'Breed2_207', 'Breed2_210', 'Breed2_212', 'Breed2_213',
       'Breed2_218', 'Breed2_227', 'Breed2_228', 'Breed2_233',
       'Breed2_237', 'Breed2_239', 'Breed2_240', 'Breed2_241',
       'Breed2_242', 'Breed2_243', 'Breed2_245', 'Breed2_246',
       'Breed2_247', 'Breed2_248', 'Breed2_249', 'Breed2_250',
       'Breed2_251', 'Breed2_252', 'Breed2_253', 'Breed2_254',
       'Breed2_256', 'Breed2_257', 'Breed2_258', 'Breed2_260',
       'Breed2_261', 'Breed2_262', 'Breed2_263', 'Breed2_264',
       'Breed2_265', 'Breed2_266', 'Breed2_267', 'Breed2_268',
       'Breed2_270', 'Breed2_271', 'Breed2_272', 'Breed2_274',
       'Breed2_276', 'Breed2_277', 'Breed2_278', 'Breed2_279',
       'Breed2_280', 'Breed2_281', 'Breed2_282', 'Breed2_283',
       'Breed2_284', 'Breed2_285', 'Breed2_288', 'Breed2_289',
       'Breed2_290', 'Breed2_291', 'Breed2_292', 'Breed2_293',
       'Breed2_294', 'Breed2_295', 'Breed2_296', 'Breed2_297',
       'Breed2_299', 'Breed2_300', 'Breed2_301', 'Breed2_302',
       'Breed2_303', 'Breed2_304', 'Breed2_305', 'Breed2_306',
       'Breed2_307', 'Gender_1', 'Gender_2', 'Gender_3', 'Color1_1',
       'Color1_2', 'Color1_3', 'Color1_4', 'Color1_5', 'Color1_6',
       'Color1_7', 'Color2_0', 'Color2_2', 'Color2_3', 'Color2_4',
       'Color2_5', 'Color2_6', 'Color2_7', 'Color3_0', 'Color3_3',
       'Color3_4', 'Color3_5', 'Color3_6', 'Color3_7', 'MaturitySize_1',
       'MaturitySize_2', 'MaturitySize_3', 'MaturitySize_4', 'FurLength_1',
       'FurLength_2', 'FurLength_3', 'Vaccinated_1', 'Vaccinated_2',
       'Vaccinated_3', 'Dewormed_1', 'Dewormed_2', 'Dewormed_3',
       'Sterilized_1', 'Sterilized_2', 'Sterilized_3', 'Health_1',
       'Health_2', 'Health_3', 'State_41324', 'State_41325', 'State_41326',
       'State_41327', 'State_41330', 'State_41332', 'State_41335',
       'State_41336', 'State_41342', 'State_41345', 'State_41361',
       'State_41367', 'State_41401', 'State_41415', 'VideoAmt_0',
       'VideoAmt_1', 'VideoAmt_2', 'VideoAmt_3', 'VideoAmt_4',
       'VideoAmt_5', 'VideoAmt_6', 'VideoAmt_7', 'VideoAmt_8',
       'VideoAmt_9', 'PhotoAmt_0.0', 'PhotoAmt_1.0', 'PhotoAmt_2.0',
       'PhotoAmt_3.0', 'PhotoAmt_4.0', 'PhotoAmt_5.0', 'PhotoAmt_6.0',
       'PhotoAmt_7.0', 'PhotoAmt_8.0', 'PhotoAmt_9.0', 'PhotoAmt_10.0',
       'PhotoAmt_11.0', 'PhotoAmt_12.0', 'PhotoAmt_13.0', 'PhotoAmt_14.0',
       'PhotoAmt_15.0', 'PhotoAmt_16.0', 'PhotoAmt_17.0', 'PhotoAmt_18.0',
       'PhotoAmt_19.0', 'PhotoAmt_20.0', 'PhotoAmt_21.0', 'PhotoAmt_22.0',
       'PhotoAmt_23.0', 'PhotoAmt_24.0', 'PhotoAmt_25.0', 'PhotoAmt_26.0',
       'PhotoAmt_27.0', 'PhotoAmt_28.0', 'PhotoAmt_29.0', 'PhotoAmt_30.0',
       'is_age_0', 'is_age_1',
       'is_age_11_20', 'is_age_2', 'is_age_3', 'is_age_4', 'is_age_5_6',
       'is_age_7_10', 'is_age_nan', 'is_age_over_20']
X_orig_train = df_with_dummies.loc[:(n_train-1), var_selected]
X_test = df_with_dummies.loc[n_train:, var_selected].reset_index(drop=True)
y_orig_train = df_with_dummies.loc[:(n_train-1), 'AdoptionSpeed']
y_test = df_with_dummies.loc[n_train:, 'AdoptionSpeed'].reset_index(drop=True)
X_train, X_valid, y_train, y_valid = train_test_split(X_orig_train, y_orig_train, 
                                                      test_size=0.25, random_state=202)
def metric(y1,y2):
    return cohen_kappa_score(y1,y2, weights='quadratic')
algorithm_starts = time.time()
rf = RandomForestClassifier(n_estimators=80,
                             max_depth=None, 
                             min_samples_split=12, 
                             min_samples_leaf=5, 
                             min_weight_fraction_leaf=0.0, 
                             max_features='auto', 
                             min_impurity_decrease=0.0, 
                             min_impurity_split=None, 
                             bootstrap=True, 
                             oob_score=True, 
                             random_state=202, 
                             verbose=1, 
                             warm_start=True, 
                             class_weight=None)
rf.fit(X_train, y_train)
rf_valid_all = pd.DataFrame(rf.predict(X_valid))
time.time() - algorithm_starts
#Check Score
metric(rf_valid_all, y_valid)
algorithm_starts = time.time()
lr_train = LogisticRegression(C=1, random_state=202, solver='saga',
                        multi_class='multinomial')
lr_train.fit(X_train, y_train)
lr_valid_all = pd.DataFrame(lr_train.predict(X_valid))
time.time() - algorithm_starts
#Check Score
metric(lr_valid_all, y_valid)
dt = DecisionTreeClassifier(min_samples_split=500, max_depth=30, random_state=202)
dt_train = dt.fit(X_train, y_train)
dt_prob_all = pd.DataFrame(dt_train.predict_proba(X_valid))
dt_valid_all = pd.DataFrame(dt_train.predict(X_valid))
#Check Score
metric(dt_valid_all, y_valid)
etc = ExtraTreesClassifier(n_estimators=500, 
                            #max_features=2, 
                            min_samples_leaf=5, 
                            random_state=202, 
                            max_features='auto', 
                            n_jobs=-1)
etc_train = etc.fit(X_train, y_train)
etc_valid_all = pd.DataFrame(etc.predict(X_valid))
#Check Score
metric(etc_valid_all, y_valid)
#Create the matrix needed for XGBoost
X_train_xgb = xgb.DMatrix(X_train, label = y_train)
X_valid_xgb = xgb.DMatrix(X_valid)
X_only_train_xgb = xgb.DMatrix(X_train)
#Parameters and CV
num_round_for_cv = 60
param = {'max_depth':9, 'eta':0.03, 'seed':202, 'objective':'multi:softmax', 'nthread':3,
        'num_class':5}
#Cross Validation: you want the lowest error
algorithm_starts = time.time()
xgb_output = xgb.cv(param,
       X_train_xgb,
       num_round_for_cv,
       nfold = 5,
       show_stdv = False,
       verbose_eval = True,
       as_pandas = False)
time.time() - algorithm_starts
#Retrieve the round to use for your XGBoost model
rounds = pd.DataFrame(xgb_output)
round_to_use = rounds['test-merror-mean'].idxmin() + 1
round_to_use
algorithm_starts = time.time()
num_round = round_to_use
xgb_train = xgb.train(param, X_train_xgb, num_round)
#xgb_valid_prob = pd.Series(xgb_train.predict(X_only_train_xgb))
time.time() - algorithm_starts
#predctions
xgb_valid_all = pd.Series(xgb_train.predict(X_valid_xgb))
#Check Score
metric(xgb_valid_all, y_valid)
print('Extra Trees:        ', metric(etc_valid_all, y_valid))
print('Decision Trees:     ', metric(dt_valid_all, y_valid))
print('Random Forest:      ', metric(rf_valid_all, y_valid))
print('Logistic Regression:', metric(lr_valid_all, y_valid))
print('XGBoost:            ', metric(xgb_valid_all, y_valid))
algorithm_starts = time.time()
rf_test = RandomForestClassifier(n_estimators=80, 
                             max_depth=None, 
                             min_samples_split=12, 
                             min_samples_leaf=5, 
                             min_weight_fraction_leaf=0.0, 
                             max_features='auto', 
                             min_impurity_decrease=0.0, 
                             min_impurity_split=None, 
                             bootstrap=True, 
                             oob_score=True, 
                             random_state=202, 
                             verbose=1, 
                             warm_start=True, 
                             class_weight=None)
rf_test.fit(X_orig_train, y_orig_train)
time.time() - algorithm_starts
# Get and store predictions
rf_predictions = rf_test.predict(X_test)
algorithm_starts = time.time()
lr_test = LogisticRegression(C=1, random_state=202, solver='saga',
                        multi_class='multinomial')
lr_test_sub = lr_test.fit(X_orig_train, y_orig_train)
time.time() - algorithm_starts
# Get and store predictions
lr_predictions = lr_test_sub.predict(X_test)
dt_sub = DecisionTreeClassifier(min_samples_split=500, max_depth=30, random_state=202)
dt_test_sub = dt_sub.fit(X_orig_train, y_orig_train)
# Get and store predictions
dt_predictions = dt_test_sub.predict(X_test)
etc_test = ExtraTreesClassifier(n_estimators=500, 
                            #max_features=2, 
                            min_samples_leaf=5, 
                            random_state=201, 
                            max_features='auto', 
                            n_jobs=-1)
etc_test = etc_test.fit(X_orig_train, y_orig_train)
# Get and store predictions
etc_predictions = etc_test.predict(X_test)
X_orig_train_xgb = xgb.DMatrix(X_orig_train, label = y_orig_train)
X_test_xgb = xgb.DMatrix(X_test)
xgb_orig_train = xgb.train(param, X_orig_train_xgb, num_round)
xgb_predictions = xgb_orig_train.predict(X_test_xgb)
#Submitting the Logistic Regression Model as it had the highest metric
model_submission  = pd.DataFrame(lr_predictions).apply(np.round)
submission = pd.DataFrame(data={"PetID" : test_df["PetID"], 
                                   "AdoptionSpeed" : model_submission[0]})
submission.AdoptionSpeed = submission.AdoptionSpeed.astype(int)
submission.to_csv("submission.csv", index=False)
#woc = (rf_predictions + dt_predictions + lr_predictions + etc_predictions)/4
#woc = pd.DataFrame(woc).apply(np.round)
#submission = pd.DataFrame(data={"PetID" : test_df["PetID"],"AdoptionSpeed" : woc[0]})
#submission.AdoptionSpeed = submission.AdoptionSpeed.astype(int)
#submission.to_csv("submission.csv", index=False)
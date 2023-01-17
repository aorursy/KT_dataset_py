import pandas as pd

import numpy as np

import typing as tp

import pydicom

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer



from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.linear_model import HuberRegressor



from statsmodels.formula.api import quantreg
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

df = pd.concat([train_df, test_df], ignore_index=True)

df['Patient_Week'] = df['Patient'].astype(str) + '_ '+ df['Weeks'].astype(str)

df.head()
print('Shape of Training data: ', train_df.shape)

print('Shape of Test data: ', test_df.shape)
def add_height(data)->'dataframe':

    data['Height'] = 0

    data['Height'] = data.apply(lambda x: (x.FVC / (21.78 - (0.101 * x.Age))) if x.Sex == 1 else (x.FVC / (27.63 - (0.112 * x.Age))), axis=1)

    

def add_norm(data)->'dataframe':

    return (data - data.mean()) / data.std()
df['Sex'] = df['Sex'].map({'Female': 0, 'Male': 1})

df['SmokingStatus'] = df['SmokingStatus'].map({'Currently smokes': 0, 'Never smoked': 1, 'Ex-smoker': 2})

df = df.drop('Patient_Week', axis=1)

df = df.set_index('Patient')

add_height(df)

df[df.columns[~df.columns.isin(['FVC', 'Sex', 'Weeks', 'SmokingStatus'])]] = add_norm(df[df.columns[~df.columns.isin(['FVC', 'Sex', 'Weeks', 'SmokingStatus'])]])

df.head()
sub_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

sub_df.drop(['FVC'], axis=1, inplace=True)

sub_df['Patient'] = sub_df['Patient_Week'].apply(lambda x: x.split('_')[0])

sub_df['pred_Weeks'] = sub_df['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)
test_FVC = pd.merge(sub_df, df, how='left', on=['Patient'])

test_FVC = test_FVC[['Patient_Week', 'pred_Weeks', 'Percent', 'Age', 'Sex', 'SmokingStatus', 'Height']]
test_FVC.rename(columns={'pred_Weeks':'Weeks'}, inplace=True)

test_FVC = test_FVC.groupby('Patient_Week').mean()

test_FVC.sort_values(['Weeks'], inplace=True)

test_FVC
print(test_FVC.shape)
X = df.iloc[:, df.columns != "FVC"]

y = df["FVC"]



X_train = X[:-5]

y_train = y[:-5]

X_val = X[-5:]

y_val = y[-5:]



print(X.shape)

print(y.shape)

print(X_train.shape)

print(y_train.shape)

print(X_val.shape)

print(y_val.shape)
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

train_df.boxplot('FVC', by='SmokingStatus', ax = axs[0, 0])

train_df.boxplot('Percent', by='SmokingStatus', ax = axs[0, 1])

train_df.boxplot('FVC', by='Sex', ax = axs[1, 0])

train_df.boxplot('Percent', by='Sex', ax = axs[1, 1])



plt.show()
img = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/100.dcm"

ds = pydicom.dcmread(img)

plt.figure(figsize = (7,7))

plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
img_1 = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/100.dcm"

img_2 = "../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362/10.dcm"



fig, ax = plt.subplots(1, 2, figsize=(10, 10))

ds = pydicom.dcmread(img_1)

ax[0].set_title('Patient 1: Ex-Smoker')

ax[0].imshow(ds.pixel_array, cmap=plt.cm.bone)



ds = pydicom.dcmread(img_2)

ax[1].set_title('Patient 2: Never smoked')

ax[1].imshow(ds.pixel_array, cmap=plt.cm.bone)



plt.show
def baseline_loss_metric(trueFVC, predFVC, predSTD=100):

    clipSTD = np.clip(predSTD, 70 , 9e9)  

    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0 , 1000)  

#     error = -1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD)

    error = np.mean(-1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD))

    return error
class model_selection(): 

    

    def __init__(self): 

        self.y_pred_FVC = pd.DataFrame()

#         self.my_scorer = make_scorer(baseline_loss_metric, greater_is_better=False)

        self.best_param = None

        self.scoring_train = 0

        self.scoring_val = 0

    

    

    def xgboost(self, X, y, X_val, y_val,test): 

        parameters = {'learning_rate': [0.1], 'n_estimators':[10000], 'max_depth':[3], 'reg_alpha':[0.005]}

        clf = GridSearchCV(XGBRegressor(min_child_weight=0, gamma=0, 

                                        colsample_bytree=0.7, objective='reg:squarederror', nthread=-1,

                                        scale_pos_weight=1, subsample=.7, seed=27), 

                           param_grid=parameters, 

                           scoring=make_scorer(baseline_loss_metric, greater_is_better=True))

        clf.fit(X, y)

        self.best_param = clf.best_params_

        self.scoring_train = clf.score(X, y)

        self.scoring_val = clf.score(X_val, y_val)

        y_pred_xgb_FVC = clf.predict(test)

        

        self.y_pred_FVC = pd.concat([pd.Series(test.index), pd.Series(y_pred_xgb_FVC)], axis=1)

#         self.y_pred_FVC = y_pred_FVC.groupby('Patient_Week').mean()

        return self.y_pred_FVC, self.best_param, self.scoring_train, self.scoring_val

        

    

    def lightgbm(self, X, y, X_val, y_val,test): 

        parameters = {'learning_rate': [0.05], 'n_estimators':[3000], 'num_leaves':[45]}

        clf = GridSearchCV(LGBMRegressor(boosting_type='gbdt', 

                                         objective='regression',

                                         bagging_fraction=0.8,

                                         bagging_freq=1, 

                                         verbose=-1,), 

                           param_grid=parameters, 

                           scoring=make_scorer(baseline_loss_metric, greater_is_better=True))

        clf.fit(X, y)

        self.best_param = clf.best_params_

        self.scoring_train = clf.score(X, y)

        self.scoring_val = clf.score(X_val, y_val)

        y_pred_lgb_FVC = clf.predict(test)

        

        self.y_pred_FVC = pd.concat([pd.Series(test.index), pd.Series(y_pred_lgb_FVC)], axis=1)

#         self.y_pred_FVC = self.y_pred_FVC.groupby('Patient_Week').mean()

        return self.y_pred_FVC, self.best_param, self.scoring_train, self.scoring_val

    

    

    def HuberRegressor(self, X, y, test): 

        hbr = HuberRegressor(max_iter=200)

        hbr.fit(X, y)

        y_pred_hbr_FVC = hbr.predict(test)

        

        self.y_pred_FVC = pd.concat([pd.Series(test.index), pd.Series(y_pred_hbr_FVC)], axis=1)

#         self.y_pred_FVC = self.y_pred_FVC.groupby('Patient_Week').mean()

        return self.y_pred_FVC
model = model_selection()

output = model.xgboost(X_train, y_train, X_val, y_val,test_FVC)
print(output[1]) # best params on previous model {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 10000, 'reg_alpha': 0.005}

print(output[2]) # error on training -4.954089770686458

print(output[3]) # error on validation -4.954233496482487
y_pred_xgb_FVC = output[0]

y_pred_xgb_FVC.rename(columns={0:'FVC'}, inplace=True)

y_pred_xgb_FVC
model = model_selection()

output = model.lightgbm(X_train, y_train, X_val, y_val,test_FVC)
print(output[1]) # best params on previous model {'learning_rate': 0.05, 'n_estimators': 3000, 'num_leaves': 45}

print(output[2]) # error on training -4.97098194152317

print(output[3]) # error on validation -4.958517234575529
y_pred_lgb_FVC = output[0]

y_pred_lgb_FVC.rename(columns={0:'FVC'}, inplace=True)

y_pred_lgb_FVC
formula = 'FVC ~ Weeks+Percent+Age+Sex+SmokingStatus'

def QuantileRegression(train, test, printsumL=False,quanL=0.15, quanH=0.85): 

        modelL = quantreg(formula, train).fit(q=quanL)

        modelH = quantreg(formula, train).fit(q=quanH)

        test['FVC_L'] = modelL.predict(test).values

        test['FVC_H'] = modelH.predict(test).values

        if printsumL: 

            print(modelL.summary())
test_conf = pd.merge(sub_df, df, how='left', on=['Patient'])

test_conf = test_conf.rename(columns={'Patient_Week_x':'Patient_Week'})

test_conf = test_conf.drop(['Confidence', 'Weeks'], axis=1)

test_conf = test_conf.groupby('Patient_Week').mean()

test_conf.sort_values(['Patient_Week'], inplace=True)



test_conf = pd.merge(test_conf, y_pred_xgb_FVC, how='left', on=['Patient_Week']).rename(columns={'FVC_x':'FVC', 'FVC_y':'pred_FVC', 'pred_Weeks':'Weeks'})

test_conf.set_index('Patient_Week', inplace=True)

test_conf.head()
QuantileRegression(df, test_conf,True)
# test_conf = test_conf[['Weeks', 'FVC', 'Age', 'Sex', 'SmokingStatus', 'Height']]

test_conf['Confidence'] = 0.55*np.abs(test_conf['FVC_H'] - test_conf['pred_FVC'])+0.45*np.abs(test_conf['pred_FVC'] - test_conf['FVC_L'])

test_conf.head()
# y_pred_conf = pd.DataFrame(competition_metric(test_conf['FVC'], test_conf['pred_FVC'], 100))

y_pred_conf = test_conf[['Confidence']]

y_pred_conf.rename(columns={0:'Confidence'}, inplace=True)

y_pred_conf
def competition_metric(trueFVC, predFVC, predSTD):

    clipSTD = np.clip(predSTD, 70 , 9e9)  

    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0 , 1000)  

#     error = -1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD)

    error = np.mean(-1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD))

    return error
competition_metric(test_conf['FVC'], test_conf['pred_FVC'], test_conf['Confidence'])
submission = pd.merge(sub_df,y_pred_xgb_FVC, how='left', on=['Patient_Week'])

submission = pd.merge(submission,y_pred_conf, how='left', on=['Patient_Week']) # Predicted Confidence

submission = submission[['Patient_Week', 'FVC', 'Confidence_y']].rename(columns={'Confidence_y':'Confidence'})



# #baseline Confidence

# submission = pd.merge(submission, test_FVC, how='left', on=['Patient_Week']) # Raw Confidence (Percent)

# submission = submission[['Patient_Week', 'FVC', 'Confidence']].rename(columns={'Percent':'Confidence'}) # for baseline



submission.set_index('Patient_Week', inplace=True)

submission
submission.to_csv('./submission.csv')
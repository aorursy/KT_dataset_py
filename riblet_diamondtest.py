import numpy as np 

import pandas as pd 

from sklearn.model_selection import train_test_split 

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

%matplotlib inline 

import matplotlib.pyplot as plt  

import seaborn as sns
diamond_df = pd.read_csv('../input/diamond1/diamond1.csv') 
diamond_df.head()
diamond_df.shape
diamond_df.dtypes
diamond_df.describe()
diamond_df.iloc[5995:6005,]
log_price = np.log(diamond_df['Price'])

diamond_df['Log Price'] = log_price

diamond_df.head()
log_carat_weight = np.log((diamond_df['Carat Weight']))

diamond_df[('Log Carat Weight')] = log_carat_weight

diamond_df.head()
#diamond_df[('Top_Notch_Diamond')] = (diamond_df['Cut'] == 'Signature-Ideal') & (diamond_df['Color'] == 'D')
diamond_df['Cut'].value_counts(dropna=False)
dummies_Cut = pd.get_dummies(diamond_df['Cut'])

dummies_Cut.columns = ['Cut_' + str(col) for col in dummies_Cut.columns]

dummies_Cut.head(10)
diamond_df['Color'].value_counts(dropna=False)
dummies_Color = pd.get_dummies(diamond_df['Color'])

dummies_Color.columns = ['Color_' + str(col) for col in dummies_Color.columns]

dummies_Color.head(10)
dummies_Clarity = pd.get_dummies(diamond_df['Clarity'])

dummies_Clarity.columns = ['Clarity_' + str(col) for col in dummies_Clarity.columns]

dummies_Clarity.head(10)
dummies_Polish = pd.get_dummies(diamond_df['Polish'])

dummies_Polish.columns = ['Polish_' + str(col) for col in dummies_Polish.columns]

dummies_Polish.head(10)
dummies_Symmetry = pd.get_dummies(diamond_df['Symmetry'])

dummies_Symmetry.columns = ['Symmetry_' + str(col) for col in dummies_Symmetry.columns]

dummies_Symmetry.head(10)
df_with_dummies = pd.concat([diamond_df[['Price', 'Carat Weight', 'Log Price', 'Log Carat Weight']], dummies_Cut, dummies_Color, dummies_Clarity, dummies_Polish, dummies_Symmetry], axis=1)

df_with_dummies.head(10)
df_with_dummies[('int_EX_Ideal')] = df_with_dummies['Symmetry_EX']*df_with_dummies['Cut_Ideal']
df_with_dummies[('int_VG_VeryGood')] = df_with_dummies['Symmetry_VG']*df_with_dummies['Cut_Very Good']
df_with_dummies[('int_EX_VeryGood')] = df_with_dummies['Symmetry_EX']*df_with_dummies['Cut_Very Good']
df_with_dummies[('int_VG_Ideal')] = df_with_dummies['Symmetry_EX']*df_with_dummies['Cut_Ideal']
df_with_dummies[('int_G_Ideal')] = df_with_dummies['Color_G']*df_with_dummies['Cut_Ideal']
df_with_dummies[('int_G_VeryGood')] = df_with_dummies['Color_G']*df_with_dummies['Cut_Very Good']
df_with_dummies[('int_G_SigIdeal')] = df_with_dummies['Color_G']*df_with_dummies['Cut_Signature-Ideal']
df_with_dummies[('int_H_Ideal')] = df_with_dummies['Color_H']*df_with_dummies['Cut_Ideal']
df_with_dummies[('int_D_VeryGood')] = df_with_dummies['Color_D']*df_with_dummies['Cut_Very Good']
df_with_dummies[('int_E_VeryGood')] = df_with_dummies['Color_E']*df_with_dummies['Cut_Very Good']
df_with_dummies[('int_D_SI1')] = df_with_dummies['Color_D']*df_with_dummies['Clarity_SI1']
df_with_dummies[('int_E_SI1')] = df_with_dummies['Color_E']*df_with_dummies['Clarity_SI1']
df_with_dummies[('int_F_SI1')] = df_with_dummies['Color_F']*df_with_dummies['Clarity_SI1']
df_with_dummies[('int_G_SI1')] = df_with_dummies['Color_G']*df_with_dummies['Clarity_SI1']
df_with_dummies[('int_G_VS1')] = df_with_dummies['Color_G']*df_with_dummies['Clarity_VS1']
df_with_dummies[('int_G_VS2')] = df_with_dummies['Color_G']*df_with_dummies['Clarity_VS2']
df_with_dummies[('int_G_VVS2')] = df_with_dummies['Color_G']*df_with_dummies['Clarity_VVS2']
df_with_dummies[('int_H_SI1')] = df_with_dummies['Color_H']*df_with_dummies['Clarity_SI1']
df_with_dummies[('int_I_SI1')] = df_with_dummies['Color_I']*df_with_dummies['Clarity_SI1']
df_with_dummies[('int_Ideal_SI1')] = df_with_dummies['Cut_Ideal']*df_with_dummies['Clarity_SI1']
df_with_dummies[('int_veryGood_SI1')] = df_with_dummies['Cut_Very Good']*df_with_dummies['Clarity_SI1']
df_with_dummies[('int_Ideal_VS1')] = df_with_dummies['Cut_Ideal']*df_with_dummies['Clarity_VS1']
df_with_dummies[('int_veryGood_VS1')] = df_with_dummies['Cut_Very Good']*df_with_dummies['Clarity_VS1']
df_with_dummies[('int_Ideal_VS2')] = df_with_dummies['Cut_Ideal']*df_with_dummies['Clarity_VS2']
df_with_dummies[('int_Ideal_VVS2')] = df_with_dummies['Cut_Ideal']*df_with_dummies['Clarity_VVS2']
df_with_dummies[('int_veryGood_VS2')] = df_with_dummies['Cut_Very Good']*df_with_dummies['Clarity_VS2']
df_with_dummies[('int_Ideal_Carat')] = df_with_dummies['Log Carat Weight']*df_with_dummies['Cut_Ideal']
df_with_dummies[('int_EX_Carat')] = df_with_dummies['Log Carat Weight']*df_with_dummies['Symmetry_EX']
#def create_interaction(diamond_df,dummies_Cut,dummies_Cut):

 #   dummy_CutColor = dummies_Cut + "*" + dummies_Cut

  #  diamond_df[dummy_CutColor] = pd.Series(diamond_df[dummy_Cut] * diamond_df[dummy_Color], name=name)

    

#diamond_df[('dummy_CutColor')] = pd.concat([df, pd.Series(variables[col1] * variables[col2], name=name)], axis=1)

#diamond_df['dummy_CutColor'] = dmatrices('log_price ~ C(dummy_Cut) * C(dummy_Color)', diamond_df(), return_type="diamond_df()")

#regression = LinearRegression(normalize=True)

#x = np.arange(df_with_dummies).reshape(10,30)

#crossvalidation = KFold(n=df_with_dummies.shape[0], n_folds=10, shuffle=True, random_state=1)

#ind_var_selected_inter = interaction.fit_transform(df_with_dummies)





#interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)



df_with_dummies.columns
#Model 1 - No interactions

ind_var_selected1 = ['Carat Weight', 'Cut_Fair', 'Cut_Good', 'Cut_Ideal',

                    'Cut_Signature-Ideal', 'Cut_Very Good', 'Color_D', 'Color_E', 'Color_F',

                    'Color_G', 'Color_H', 'Color_I', 'Clarity_FL', 'Clarity_IF', 'Clarity_SI1',

                    'Clarity_VS1', 'Clarity_VS2', 'Clarity_VVS1', 'Clarity_VVS2', 'Polish_EX',

                    'Polish_G', 'Polish_ID', 'Polish_VG', 'Symmetry_EX', 'Symmetry_G', 'Symmetry_ID',

                    'Symmetry_VG', 'Log Carat Weight']
#Model 2 - Interactions

ind_var_selected = ['Carat Weight','Cut_Fair', 'Cut_Good', 'Cut_Ideal',

                    'Cut_Signature-Ideal', 'Cut_Very Good', 'Color_D', 'Color_E', 'Color_F',

                    'Color_G', 'Color_H', 'Color_I', 'Clarity_FL', 'Clarity_IF', 'Clarity_SI1',

                    'Clarity_VS1', 'Clarity_VS2', 'Clarity_VVS1', 'Clarity_VVS2', 

                    'Polish_EX','Polish_G', 'Polish_ID', 'Polish_VG', 

                    'Symmetry_EX', 'Symmetry_G', 'Symmetry_ID', 'Symmetry_VG', 

                    'Log Carat Weight',

                    'int_EX_Ideal','int_VG_VeryGood','int_EX_VeryGood','int_VG_Ideal',

                    'int_G_Ideal','int_G_VeryGood','int_G_SigIdeal','int_H_Ideal',#'int_D_VeryGood', 'int_E_VeryGood',

                    'int_D_SI1','int_E_SI1','int_F_SI1','int_G_SI1','int_G_VS2',

                    #'int_G_VS1','int_G_VVS2','int_H_SI1','int_I_SI1',

                    'int_Ideal_SI1','int_veryGood_SI1','int_Ideal_VS1','int_veryGood_VS1','int_Ideal_VS2'

                    #,'int_Ideal_VVS2','int_veryGood_VS2','int_Ideal_Carat','int_EX_Carat'

                    ]
X_orig_train1 = df_with_dummies.loc[:5999, ind_var_selected1]

y_orig_train1 = df_with_dummies.loc[:5999, 'Log Price']
X_test1 = df_with_dummies.loc[6000:, ind_var_selected1]

y_test1 = df_with_dummies.loc[6000:, 'Log Price']
X_orig_train = df_with_dummies.loc[:5999, ind_var_selected]

y_orig_train = df_with_dummies.loc[:5999, 'Log Price']
X_test = df_with_dummies.loc[6000:, ind_var_selected]

y_test = df_with_dummies.loc[6000:, 'Log Price']
X_train1, X_valid1, y_train1, y_valid1 = train_test_split(X_orig_train1, y_orig_train1, test_size=0.25, random_state=201)
X_train1.head()
X_train, X_valid, y_train, y_valid = train_test_split(X_orig_train, y_orig_train, test_size=0.25, random_state=201)
X_train.head()
X_valid1.head()
X_valid.head()
X_train1.shape

X_train.shape
X_valid1.shape
X_valid.shape
def RMSLE(predictions, realizations):

    predictions_use = predictions.clip(0)

    rmsle = np.sqrt(np.mean(np.array(np.log(predictions_use + 1) - np.log(realizations + 1))**2))

    return rmsle





#def MAPE(predictions, realizations): 

 #   predictions, realizations = np.array(predictions), np.array(realizations)

  #  MAPE = np.mean(np.abs((predictions - realizations) / predictions)) * 100

   # return MAPE



#def rmse(predictions, realizations):

  #  rmse = np.sqrt(((predictions - realizations).astype('double') ** 2).mean())

 #   return rmse

    # rmse = np.sqrt(((predictions - realizations) ** 2).mean())



    #from sklearn.metrics import mean_squared_error

    #from math import sqrt

    #predictions_use = predictions.clip(0)

    #rmse = sqrt(mean_squared_error(predictions_use, realizations))

    #return rmse
rf = RandomForestRegressor(n_estimators=500, max_features=46, min_samples_leaf=5, random_state=201)

rf_model = rf.fit(X_train, y_train)

rf_pred = rf_model.predict(X_valid)
pd.DataFrame(rf_model.feature_importances_, index=ind_var_selected)
RMSLE(rf_pred, y_valid)
xgb_train1 = xgb.DMatrix(X_train1, label = y_train1)

xgb_valid1 = xgb.DMatrix(X_valid1)
num_round_for_cv = 500

param = {'max_depth':6, 'eta': 0.1, 'seed': 201, 'objective':'reg:linear'}
xgb.cv(param,

       xgb_train1,

       num_round_for_cv,

       nfold = 5,

       show_stdv = False,

       verbose_eval = True,

       as_pandas = False)
num_round = 172

xgb_model1 = xgb.train(param, xgb_train1, num_round)

xgb_pred1 = xgb_model1.predict(xgb_valid1)
xgb_model1.get_fscore()
RMSLE(xgb_pred1, y_valid1)
xgb_train = xgb.DMatrix(X_train, label = y_train)

xgb_valid = xgb.DMatrix(X_valid)
num_round_for_cv = 500

param = {'max_depth':6, 'eta': 0.1, 'seed': 201, 'objective':'reg:linear'}
xgb.cv(param,

       xgb_train,

       num_round_for_cv,

       nfold = 5,

       show_stdv = False,

       verbose_eval = True,

       as_pandas = False)
num_round = 172 #187

xgb_model = xgb.train(param, xgb_train, num_round)

xgb_pred = xgb_model.predict(xgb_valid)
xgb_model.get_fscore()
xgb.plot_importance(xgb_model)
RMSLE(xgb_pred, y_valid)
rt = DecisionTreeRegressor(min_samples_split=25, random_state=201)

rt_model = rt.fit(X_train, y_train)

rt_pred = rt_model.predict(X_valid)
pd.DataFrame(rt_model.feature_importances_, index=ind_var_selected)
RMSLE(rt_pred, y_valid)
xgb_orig_train1 = xgb.DMatrix(X_orig_train1, label = y_orig_train1)

xgb_test1 = xgb.DMatrix(X_test1)

xgb_model_retrain1 = xgb.train(param, xgb_orig_train1, num_round)

xgb_pred_test1 = xgb_model_retrain1.predict(xgb_test1)

xgb_pred_test_clipped1 = pd.Series(xgb_pred_test1.clip(0))
xgb_orig_train = xgb.DMatrix(X_orig_train, label = y_orig_train)

xgb_test = xgb.DMatrix(X_test)

xgb_model_retrain = xgb.train(param, xgb_orig_train, num_round)

xgb_pred_test = xgb_model_retrain.predict(xgb_test)

xgb_pred_test_clipped = pd.Series(xgb_pred_test.clip(0))
diamond_submission = pd.read_csv('diamondSubmission.csv')
xgb_submission = diamond_submission

xgb_submission['Price'] = (1/2*(xgb_pred_test_clipped + xgb_pred_test_clipped1))

xgb_submission.to_csv('xgbSubmissionv_int5.csv', index=False)
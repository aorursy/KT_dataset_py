import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb
rf_df = pd.read_pickle('/kaggle/input/alex-f-french-motor-claims-analysis/Alex_Farquharson_rf_dataframe.gzip')

glm_df = pd.read_pickle('/kaggle/input/models-of-french-motor-claims/df_validation_GLM_preds.gzip')

xgb_df = pd.read_pickle('/kaggle/input/chuns-french-motor-claims-project/xgb_filtered_pred_valid_set_new.gzip')







print(rf_df.describe)

print(xgb_df.describe)

print(glm_df.describe)
from joblib import dump,load

import pickle

rf_model = load('/kaggle/input/alex-f-french-motor-claims-analysis/rf_model.gzip')





glm_model = pickle.load(open('/kaggle/input/models-of-french-motor-claims/GLMResults_obj.pkl',"rb"))

xgb_model= pickle.load(open('/kaggle/input/chuns-french-motor-claims-project/xgbmodel.pkl', "rb"))

xgb_regressor_model = pickle.load(open('/kaggle/input/chuns-french-motor-claims-project/regressorxgbmodel.pkl',"rb"))
df_two = glm_df.merge(xgb_df,left_on='IDpol',right_on='IDpol')



df_two.describe()
print("Random Forest Predicted is "+ str(rf_df.iloc[:,0].sum()))

print("XGB Predicted is "+ str(xgb_df.iloc[:,4].sum()))

print("GLM Predicted is "+ str(glm_df.iloc[:,-1].sum()))

print("Actual Claim Number is "+ str(glm_df.iloc[:,-2].sum()))

# y = actual freq , p = pred req , w = exposure



def poisson_deviance(y, p):



    d = -2 * np.where(y == 0, -(y - p), (y * np.log(y / p)) - (y - p))



    deviance = sum(d)



    return(deviance)









def gamma_deviance(y, p, w):



    d = -2 * w * (-np.log(y / p) + ((y - p)/p))



    deviance = sum(d)



    return(deviance)













def proportion_deviance_explained(y, p, w, family):



    assert family in ['poisson', 'gamma'], 'family must be poisson or gamma'



    if family == 'poisson':



        deviance = poisson_deviance(y, p)



        null_deviance = poisson_deviance(y, w * np.sum(y) / np.sum(w))



    elif family == 'gamma':



        deviance = gamma_deviance(y, p, w)



        null_deviance = gamma_deviance(y, np.sum(y * w) / np.sum(w), w)



    propn_deviance_explained = (null_deviance - deviance) / null_deviance



    return(deviance, null_deviance, propn_deviance_explained)
rf_y = rf_df.iloc[:,1] #Valid

rf_p = rf_df.iloc[:,0] #Pred

rf_w = rf_df.iloc[:,2] #Exposure

proportion_deviance_explained(rf_y,rf_p,rf_w,'poisson')
xgb_y = xgb_df.iloc[:,1]

xgb_p = xgb_df.iloc[:,4]

xgb_w = xgb_df.iloc[:,2]

proportion_deviance_explained(xgb_y,xgb_p,xgb_w,'poisson')
glm_y = glm_df.iloc[:,-2]

glm_p = glm_df.iloc[:,-1]

glm_w = glm_df.iloc[:,2]

proportion_deviance_explained(glm_y,glm_p,glm_w,'poisson')
from sklearn.metrics import mean_squared_error



print("RF MSE equals " + str(mean_squared_error(rf_y, rf_p)))

print("xgb MSE equals " + str(mean_squared_error(xgb_y, xgb_p)))

print("glm MSE equals " + str(mean_squared_error(glm_y, glm_p)))

print("-----")



print("RF RMSE equals " + str(mean_squared_error(rf_y, rf_p, squared=False)))

print("xgb RMSE equals " + str(mean_squared_error(xgb_y, xgb_p, squared=False)))

print("glm RMSE equals " + str(mean_squared_error(glm_y, glm_p, squared=False)))



from pdpbox import pdp, get_dataset, info_plots
def frange(start,end=None,inc=None):

    "A range function, that accept float increments"

    

    if end == None:

        end = start + 0.0

        start = 0.0

        

    if inc == None:

        inc = 1.0

        

    L = []

    while 1:

        next = start + len(L) * inc

        if inc > 0 and next >= end:

            break

        elif inc < 0 and next <= end:

            break

        L.append(next)

    return L


#Model

xgb_model 



#Dataset

x_train = pd.read_pickle('/kaggle/input/chuns-french-motor-claims-project/xtrain.gzip')



#Features

features=['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density', 'LogDensity', 'Area_A', 'Area_B', 'Area_C', 'Area_D', 'Area_E', 'Area_F', 'VehBrand_B1', 'VehBrand_B10', 'VehBrand_B11', 'VehBrand_B12', 'VehBrand_B13', 'VehBrand_B14', 'VehBrand_B2', 'VehBrand_B3', 'VehBrand_B4', 'VehBrand_B5', 'VehBrand_B6', 'VehGas_Diesel', 'VehGas_Regular', 'Region_R11', 'Region_R21', 'Region_R22', 'Region_R23', 'Region_R24', 'Region_R25', 'Region_R26', 'Region_R31', 'Region_R41', 'Region_R42', 'Region_R43', 'Region_R52', 'Region_R53', 'Region_R54', 'Region_R72', 'Region_R73', 'Region_R74', 'Region_R82', 'Region_R83', 'Region_R91', 'Region_R93', 'Region_R94']



#Target

target = ['pred_freq']

def plotpdp(feature_name):

    

    pdp_feature_location = x_train.columns.get_loc(feature_name)

    median_values = x_train.median()

    median_values = pd.DataFrame(median_values).transpose()

    

    start = x_train[feature_name].min()

    stop = x_train[feature_name].max()

    step = (x_train[feature_name].max() - x_train[feature_name].min())/100

    

    feature_value = []

    prediction = []

    

    for i in frange(start,stop,step):

        median_values.iloc[0,pdp_feature_location]=i

        DM_pred=xgb.DMatrix(data=median_values.values,feature_names=list(x_train))

        pred = xgb_model .predict(DM_pred)

        feature_value.append(i)

        prediction.append(pred[0])

        

    pdp = pd.DataFrame({'Feature_Value': feature_value, 'Prediction': prediction})

    

    ax1 = pdp.plot.line(x='Feature_Value',y='Prediction')

    ax2 = ax1.twinx()

    ax3 = x_train[feature_name].plot.hist(bins=40,figsize=(15,6),title='PDP: '+ feature_name,alpha = 0.9)
    plotpdp('DrivAge')

    plotpdp('VehPower')

    plotpdp('VehAge')

    plotpdp('BonusMalus')

    plotpdp('Density')
#Model

xgb_model 

xgb_regressor_model



#Dataset

xgb_pdp_df = pd.read_pickle('/kaggle/input/chuns-french-motor-claims-project/xgb_pred_reg_valid_set_new.gzip')





#Features

xgb_features=['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density', 'LogDensity', 'Area_A', 'Area_B', 'Area_C', 'Area_D', 'Area_E', 'Area_F', 'VehBrand_B1', 'VehBrand_B10', 'VehBrand_B11', 'VehBrand_B12', 'VehBrand_B13', 'VehBrand_B14', 'VehBrand_B2', 'VehBrand_B3', 'VehBrand_B4', 'VehBrand_B5', 'VehBrand_B6', 'VehGas_Diesel', 'VehGas_Regular', 'Region_R11', 'Region_R21', 'Region_R22', 'Region_R23', 'Region_R24', 'Region_R25', 'Region_R26', 'Region_R31', 'Region_R41', 'Region_R42', 'Region_R43', 'Region_R52', 'Region_R53', 'Region_R54', 'Region_R72', 'Region_R73', 'Region_R74', 'Region_R82', 'Region_R83', 'Region_R91', 'Region_R93', 'Region_R94']



#Target

xgb_target = ['pred_freq']

def xgb_pdp(feature):

    pdp_features = pdp.pdp_isolate(model=xgb_regressor_model, dataset=xgb_pdp_df, model_features=xgb_features, feature=feature)

    fig, axes = pdp.pdp_plot(pdp_features, feature_name = feature, plot_lines=True, frac_to_plot=500)
xgb_pdp('DrivAge')
xgb_pdp('VehPower')
xgb_pdp('VehAge')
xgb_pdp('BonusMalus')
xgb_pdp('Density')
# Model

glm_model



# Dataset

glm_df



# Features

features_glm=['BonusMalus_over_50','VehPower','VehAge','DrivAge','BonusMalus','Density','Frequency','DrivAge_capped','DrivAge_pow2','BonusMalus_mod3','VehAge_capped','Density_log']



# Target

target_glm = ['pred_freq']



def glm_pdp(feature):

    pdp_features = pdp.pdp_isolate(model=glm_model, dataset=glm_df, model_features=features_glm, feature=feature)

    fig, axes = pdp.pdp_plot(pdp_features, feature_name = feature, plot_lines=True, frac_to_plot=500)
# Model

rf_model



# Dataset 

rf_X_train = pd.read_pickle('/kaggle/input/alex-f-french-motor-claims-analysis/Alex_Farquharson_X_train_dataframe.gzip')

rf_y_train = pd.read_pickle('/kaggle/input/alex-f-french-motor-claims-analysis/Alex_Farquharson_y_train_dataframe.gzip')

rf_pdp_dataset = pd.concat((rf_X_train,rf_y_train),axis=1)



# Features



rf_features=['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density', 'B', 'C', 'D','E', 'F', 'B10', 'B11', 'B12', 'B13', 'B14', 'B2', 'B3', 'B4', 'B5','B6', 'Regular',

               'R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R31', 'R41','R42', 'R43', 'R52', 'R53', 'R54', 'R72', 'R73', 'R74', 'R82', 'R83','R91', 'R93', 'R94']



# Target

target_rf = ['Rando Forest Predictions']
def rf_pdp(feature):

    pdp_features = pdp.pdp_isolate(model=rf_model, dataset=rf_pdp_dataset, model_features=rf_features, feature=feature)

    fig, axes = pdp.pdp_plot(pdp_features, feature_name = feature, plot_lines=True, frac_to_plot=500)
rf_pdp('DrivAge')
rf_pdp('VehPower')
rf_pdp('VehAge')
rf_pdp('BonusMalus')
rf_pdp('Density')
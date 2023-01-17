
df_raw.head()
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from pandas._libs.lib import is_integer
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import gzip
#importing dataframes
rf_df = pd.read_pickle('/kaggle/input/alex-farquharson-rf-dataframe/Alex_Farquharson_rf_dataframe.gzip')
glm_df = pd.read_pickle('/kaggle/input/french-motor-claims-glm-output/df_validation_GLM_preds.gzip')
gbm_df = pd.read_pickle('/kaggle/input/gbm-french-claims/xgb_filtered_pred_valid_set_new.gzip')
gbm_df_with_columns = pd.read_pickle('/kaggle/input/gbm-french-claims/xgb_pred_valid_set_new.gzip')
df_raw = pd.read_csv('/kaggle/input/french-motor-claims-datasets-fremtpl2freq/freMTPL2freq.csv')
df_raw['Freq'] = df_raw['ClaimNb']/df_raw['Exposure']
#importing models
from joblib import dump,load
import pickle
gbm_model = pickle.load(open('/kaggle/input/gbm-french-claims/xgbmodel.pkl', "rb"))
rf_model = load('/kaggle/input/alex-farquharson-rf-dataframe/rf_model.gzip')
glm_model = load('/kaggle/input/french-motor-claims-glm-output/GLMResults_obj.pkl')

models = {'rf_model':rf_model,'glm_model':glm_model,'gbm_model':gbm_model}

#check common columns match
rf_df['ClaimNb'].value_counts() == gbm_df_with_columns['ClaimNb'].value_counts()
gbm_df['ClaimNb'].value_counts() == glm_df['ClaimNb'].value_counts()
#get datasets ready to merge
rf_df['IDpol'] = glm_df['IDpol']
gbm_df_with_columns['gbm predictions'] = gbm_df_with_columns['pred_ClaimNb']
#merge datasets
master_df = pd.merge(rf_df,glm_df,how='outer',on='IDpol')
master_df = pd.merge(master_df,gbm_df_with_columns[['gbm predictions','IDpol']],how='outer',on='IDpol')
assert all(master_df.notnull()) ==True
print('no nulls')
assert len(master_df)==101702
print('length unchanged')
assert master_df['ClaimNb_x'].sum() - master_df['ClaimNb_y'].sum() ==0
print('columns appear to be concatenated correctly')
master_df.drop(['ClaimNb_y','Exposure_y','wgt','pred_freq','act_Nb',],axis=1,inplace=True)
master_df.rename(columns = {'pred_Nb':'glm predictions'},inplace=True)
predictions_df = master_df[['Random Forest Predictions','glm predictions','gbm predictions', 'ClaimNb_x','Exposure_x']]
predictions_df.sum()
print('rf',(predictions_df['Random Forest Predictions']*predictions_df['Exposure_x']).sum())
print('glm',(predictions_df['glm predictions']*predictions_df['Exposure_x']).sum())
print('gbm',(predictions_df['gbm predictions']*predictions_df['Exposure_x']).sum())
print('actual',(predictions_df['ClaimNb_x']*predictions_df['Exposure_x']).sum())
predictions_df.head()
#lift_chart function
#to plot lift chart, predicted colm actual colm and weights colm take just the predictions_df colm names only, q is no. of quartiles
def lift_chart(predicted_colm,actual_colm, weights_colm,q,y_min=0,y_max=0.18):
    predictions_df.sort_values(by=predicted_colm,inplace=True)
    def weighted_qcut(predicted_colm, weights_colm, q):
        #Return weighted quantile cuts from a given series
        if is_integer(q):
            quantiles = np.linspace(0, 1, q + 1)
        else:
            quantiles = q
        order = predictions_df[weights_colm].iloc[predictions_df[predicted_colm].argsort()].cumsum() #makes series of cumulative exposure (sorted by values)
        bins = pd.cut(order / order.iloc[-1], quantiles, labels=False) #cuts into q quantiles along order (cumulative exposure) column
        return bins.sort_index() #makes column in line with index of original dataframe
    #3 cut by weight
    predictions_df['weighted_cut'] = weighted_qcut(predicted_colm,weights_colm,10)
    #4 function to make means and plot them
    def means(predicted_colm,actual_colm):
        predicted_mean = []
        actual_mean = []
        for x in np.arange(10):
            pred = predictions_df[predictions_df['weighted_cut']==x][predicted_colm].mean()
            predicted_mean.append(pred)
            actual = predictions_df[predictions_df['weighted_cut']==x][actual_colm].mean()
            actual_mean.append(actual)
        means = pd.DataFrame(data = list(zip(predicted_mean,actual_mean)), columns = ['predicted','actual'])
        sns.scatterplot(data=means,x='actual',y='actual')
        sns.scatterplot(data=means, x='actual',y='predicted')
        a = means.iloc[9]['actual'] / means.iloc[0]['actual']
        b = means.iloc[9]['predicted'] / means.iloc[0]['predicted']
        print(predicted_colm[:-12],'actual differentiation',a)
        print(predicted_colm[:-12], 'model differentiation',b)
        print(predicted_colm[:-12], 'factor',b/a )
    means(predicted_colm,actual_colm)
    plt.title(predicted_colm)
    plt.ylim(y_min,y_max)
plt.subplots(1,3,figsize=(20,4))
plt.subplot(1,3,1)
lift_chart('Random Forest Predictions','ClaimNb_x','Exposure_x',10)
plt.subplot(1,3,2)
lift_chart('glm predictions','ClaimNb_x','Exposure_x',10)
plt.subplot(1,3,3)
lift_chart('gbm predictions','ClaimNb_x','Exposure_x',10)
#double_lift_chart function
#to plot double lift chart, predicted colm1, predicted colm2 actual colm and weights colm take just the predictions_df colm names only, q is no. of quartiles
def double_lift_chart(predicted_colm1,predicted_colm2, actual_colm, weights_colm,q,y_min=0,y_max=0.18):
    predictions_df['ratio'] = predictions_df[predicted_colm1]/predictions_df[predicted_colm2]
    predictions_df.sort_values('ratio',inplace = True)
    def weighted_qcut(predicted_colm1, predicted_colm2, weights_colm, q):
        #Return weighted quantile cuts from a given series
        if is_integer(q):
            quantiles = np.linspace(0, 1, q + 1)
        else:
            quantiles = q
        order = predictions_df[weights_colm].iloc[predictions_df['ratio'].argsort()].cumsum() #makes series of cumulative exposure (sorted by values)
        bins = pd.cut(order / order.iloc[-1], quantiles, labels=False) #cuts into q quantiles along order (cumulative exposure) column
        return bins.sort_index() #makes column in line with index of original dataframe
    #3 cut by weight
    predictions_df['weighted_cut'] = weighted_qcut(predicted_colm1, predicted_colm2,weights_colm,10)
    #4 function to make means and plot them
    def means(predicted_colm1,predicted_colm2,actual_colm):
        predicted_1_mean = []
        predicted_2_mean = []
        actual_mean = []
        for x in np.arange(10):
            pred = predictions_df[predictions_df['weighted_cut']==x][predicted_colm1].mean()
            predicted_1_mean.append(pred)
            pred = predictions_df[predictions_df['weighted_cut']==x][predicted_colm2].mean()
            predicted_2_mean.append(pred)
            actual = predictions_df[predictions_df['weighted_cut']==x][actual_colm].mean()
            actual_mean.append(actual)
        means = pd.DataFrame(data = list(zip(predicted_1_mean,predicted_2_mean,actual_mean)), columns = ['predicted 1','predicted 2','actual'])
        sns.scatterplot(data=means,x='actual',y='actual',color='blue')
        sns.scatterplot(data=means, x='actual',y='predicted 1',color = 'green')
        sns.scatterplot(data=means, x='actual',y='predicted 2', color = 'red')
        a = means.iloc[9]['actual'] / means.iloc[0]['actual']
        b = means.iloc[9]['predicted 1'] / means.iloc[0]['predicted 1']
        c = means.iloc[9]['predicted 2'] / means.iloc[0]['predicted 2']
        print(predicted_colm1[:-12],'and',predicted_colm2[:-12], 'actual differentiation',a)
        print(predicted_colm1[:-12], 'model differentiation',b)
        print(predicted_colm2[:-12], 'model differentiation',c)
    means(predicted_colm1,predicted_colm2,actual_colm)
    plt.title(predicted_colm1 + ' (green) ' + predicted_colm2 + ' (red)')
    plt.ylim(y_min,y_max)
plt.subplots(1,3,figsize=(20,4))
plt.subplot(1,3,1)
double_lift_chart('Random Forest Predictions','glm predictions','ClaimNb_x','Exposure_x',10)
plt.subplot(1,3,2)
double_lift_chart('Random Forest Predictions','gbm predictions','ClaimNb_x','Exposure_x',10)
plt.subplot(1,3,3)
double_lift_chart('glm predictions','gbm predictions','ClaimNb_x','Exposure_x',10)

predictions_df


#datasets for pdp
#rf_model
rf_X_train = pd.read_pickle('/kaggle/input/alex-farquharson-rf-dataframe/Alex_Farquharson_X_train_dataframe.gzip')
rf_y_train = pd.read_pickle('/kaggle/input/alex-farquharson-rf-dataframe/Alex_Farquharson_y_train_dataframe.gzip')
rf_features = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'LogDensity', 'B', 'C', 'D','E', 'F', 'B10', 'B11', 'B12', 'B13', 'B14', 'B2', 'B3', 'B4', 'B5','B6', 'Regular',
               'R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R31', 'R41','R42', 'R43', 'R52', 'R53', 'R54', 'R72', 'R73', 'R74', 'R82', 'R83','R91', 'R93', 'R94']
rf_pdp_dataset = pd.concat((rf_X_train,rf_y_train),axis=1)
xgb.plot_importance(gbm_model)
xgb.plot_importance(gbm_model, importance_type="cover")
xgb.plot_importance(gbm_model, importance_type="gain")
importances_rf=pd.Series(data=rf_model.feature_importances_,
                      index=rf_X_train.columns).sort_values(ascending=False).iloc[:20]
importances_rf.plot(kind='bar')
plt.title('Influence of Each Feature on the Model')
#Using pdpbox module for PDP plots

from pdpbox import pdp, info_plots
#first plot target plot

feature_to_be_analysed = 'DrivAge'
feature_name = 'DrivAge'
target_feature = 'ClaimNb'

fig, axes, summary_df = info_plots.target_plot(df=rf_pdp_dataset,
                                               feature=feature_to_be_analysed,
                                               feature_name=feature_name,
                                               target= target_feature,
                                               num_grid_points = 20,
                                               show_percentile=True,figsize= (20,8))






list_of_features = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'LogDensity', 'B', 'C', 'D','E', 'F', 'B10', 'B11', 'B12', 'B13', 'B14', 'B2', 'B3', 'B4', 'B5','B6', 'Regular','R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R31', 'R41','R42', 'R43', 'R52', 'R53', 'R54', 'R72', 'R73', 'R74', 'R82', 'R83','R91', 'R93', 'R94']

#get model partial dependency values
pdp_features = pdp.pdp_isolate(model=rf_model, dataset=rf_pdp_dataset, model_features=list_of_features, feature='DrivAge', num_grid_points=20)

#plot values
fig, axes = pdp.pdp_plot(pdp_features, feature_name = 'DrivAge', plot_lines=True, frac_to_plot=50, plot_pts_dist=False,figsize=(20,8))
axes['pdp_ax'].set_ylim(-0.025, 0.010)
sns.heatmap(rf_X_train[['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'LogDensity']].corr())
#two way plots
fig,axes,summary_df = info_plots.target_plot_interact(df=rf_pdp_dataset,
                                               features=['DrivAge','BonusMalus'],
                                               feature_names=['DrivAge','BonusMalus'],
                                               target= target_feature,
                                               show_percentile=True,figsize= (20,8))
pdp_features_interact = pdp.pdp_interact(model=rf_model, dataset=rf_pdp_dataset, model_features=list_of_features,
                                         features=['DrivAge','BonusMalus'])

#plot values
fig, axes = pdp.pdp_interact_plot(pdp_features_interact, feature_names = ['DrivAge','BonusMalus'],figsize=(20,8))
#get datasets for pdp and model 
#model is rf_model
from pdpbox import pdp, info_plots
rf_X_train = pd.read_pickle('/kaggle/input/alex-farquharson-rf-dataframe/Alex_Farquharson_X_train_dataframe.gzip')
rf_y_train = pd.read_pickle('/kaggle/input/alex-farquharson-rf-dataframe/Alex_Farquharson_y_train_dataframe.gzip')
rf_features = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'LogDensity', 'B', 'C', 'D','E', 'F', 'B10', 'B11', 'B12', 'B13', 'B14', 'B2', 'B3', 'B4', 'B5','B6', 'Regular',
               'R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R31', 'R41','R42', 'R43', 'R52', 'R53', 'R54', 'R72', 'R73', 'R74', 'R82', 'R83','R91', 'R93', 'R94']
rf_pdp_dataset = pd.concat((rf_X_train,rf_y_train),axis=1)
def rf_pdp_plot(features,feature_name,fraction_to_plot = 10, plot_dist = False,fig_size=(20,8)):
    pdp_features = pdp.pdp_isolate(model=rf_model, dataset=rf_pdp_dataset, model_features=rf_features, feature=features)
    fig, axes = pdp.pdp_plot(pdp_features, feature_name = feature_name, plot_lines=True, frac_to_plot=fraction_to_plot, plot_pts_dist=plot_dist,figsize=fig_size)
    
def rf_target_plot(features, feature_name,fig_size = (20,8)):
    fig, axes, summary_df = info_plots.target_plot(df=rf_pdp_dataset, feature=features,
                                               feature_name=feature_name, target='ClaimNb', show_percentile=True,figsize=fig_size)
fig, axes, summary_df = info_plots.target_plot(df=rf_pdp_dataset, feature='BonusMalus',feature_name='BonusMalus', target='ClaimNb', num_grid_points=15, show_percentile=True)

pdp_features = pdp.pdp_isolate(model=rf_model, dataset=rf_pdp_dataset, model_features=rf_features, feature='BonusMalus',  num_grid_points=15)
fig, axes = pdp.pdp_plot(pdp_features, feature_name = 'BonusMalus', plot_lines=True, frac_to_plot=10)
sns.distplot(rf_pdp_dataset['BonusMalus'],kde=False)
axes['pdp_ax'].set_xlim(50,125)
axes['pdp_ax'].set_ylim(0,0.05)
fig
rf_target_plot(features = ['B10', 'B11', 'B12', 'B13', 'B14', 'B2', 'B3', 'B4', 'B5','B6'],feature_name = 'VehType')
rf_pdp_plot(features = ['B10', 'B11', 'B12', 'B13', 'B14', 'B2', 'B3', 'B4', 'B5','B6'],feature_name = 'VehType')
rf_target_plot(features = 'LogDensity',feature_name = 'LogDensity')
rf_pdp_plot(features = 'LogDensity',feature_name = 'LogDensity')
rf_X_validation = pd.read_pickle('/kaggle/input/alex-farquharson-rf-dataframe/Alex_Farquharson_X_validation_dataframe.gzip')
gbm_X_test = gbm_df_with_columns.drop(['Exposure','ClaimNb','IDpol','freq','pred_ClaimNb','gbm predictions'],axis=1)
gbm_y_test = gbm_df_with_columns['ClaimNb']
import shap
#for jupyter notebooks
shap.initjs()
#xgboost model
D_test = xgb.DMatrix(gbm_X_test,gbm_y_test)
explainer = shap.TreeExplainer(gbm_model)
shap_values = explainer.shap_values(D_test)
#rf model
explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf  = explainer_rf.shap_values(rf_X_validation)
shap.force_plot(explainer.expected_value,shap_values[10,:],gbm_X_test.iloc[10,:])
shap.force_plot(explainer_rf.expected_value,shap_values_rf[10,:],rf_X_validation.iloc[10,:])
plt.title('SHAP summary plot - XGBoost')
shap.summary_plot(shap_values,gbm_X_test)

plt.title('SHAP summary plot - Random Forest')
shap.summary_plot(shap_values_rf,rf_X_validation)
shap.dependence_plot('DrivAge',shap_values,gbm_X_test)
shap.dependence_plot('DrivAge',shap_values_rf,rf_X_validation, interaction_index = 'BonusMalus')
shap.dependence_plot('BonusMalus',shap_values,gbm_X_test, interaction_index = 'VehBrand_B12')
shap.dependence_plot('BonusMalus',shap_values_rf,rf_X_validation, interaction_index = 'B12')
#pdp with glm model
gbm_pdp = gbm_df_with_columns.drop(['IDpol','Exposure','freq','pred_ClaimNb','gbm predictions'],axis=1)
gbm_features = list(gbm_pdp.columns)
gbm_features.remove('ClaimNb')

def glm_pdp_plot(features,feature_name,fraction_to_plot = 10, plot_dist = False,fig_size=(20,8)):
    pdp_features = pdp.pdp_isolate(model=glm_model, dataset=glm_pdp, model_features=glm_features, feature=features)
    fig, axes = pdp.pdp_plot(pdp_features, feature_name = feature_name, plot_lines=True, frac_to_plot=fraction_to_plot, plot_pts_dist=plot_dist,figsize=fig_size)
    
def gbm_target_plot(features, feature_name,fig_size = (20,8)):
    fig, axes, summary_df = info_plots.target_plot(df=glm_pdp, feature=features,
                                               feature_name=feature_name, target='Freq', show_percentile=True,figsize=fig_size)
glm_df
#deviance
def poisson_deviance(y,p):
    d = -2*np.where(y==0,-(y-p),(y*np.log(y/p))-(y-p))
    deviance = sum(d)
    return(deviance)

def proportion_deviance_explained(y,p,w):
    deviance = poisson_deviance(y,p)
    null_deviance = poisson_deviance(y, w * np.sum(y) / np.sum(w))
    propn_dev_expl  = (null_deviance - deviance) / null_deviance
    return (deviance, null_deviance, propn_dev_expl)
dev_list=[]
for x in ['Random Forest Predictions','glm predictions','gbm predictions']:
    dev_list.append(proportion_deviance_explained(predictions_df['ClaimNb_x'],predictions_df[x],predictions_df['Exposure_x']))

lst = ['Random Forest Predictions','glm predictions','gbm predictions']
pd.DataFrame(data=dev_list, index = lst,columns = ('deviance', 'null_deviance', 'propn_dev_expl'))


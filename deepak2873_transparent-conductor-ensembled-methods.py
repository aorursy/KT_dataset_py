# import libraries and Load data  

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import export_graphviz

%matplotlib inline



from sklearn import preprocessing

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_predict, train_test_split

from sklearn.metrics import r2_score, mean_squared_error  #, mean_squared_log_error, mean_absolute_error



#from sklearn.linear_model import LinearRegression, Ridge,  RANSACRegressor

#from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost.sklearn import XGBRegressor

#from scipy.stats import randint

#import scipy.stats as st



# load data

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

train_data.head()

train_data.shape
test_data.head()
test_data.shape
train_data.columns[(train_data == 0).all()]
unique_values_distribution = []

def unique_col_values(df):

    for column in df:

        unique_values_distribution.append ((df[column].name, len(df[column].unique()), df[column].dtype ))

        

unique_col_values(train_data)



columns_heading  = ['Header Name','Unique Count','Data Type']



data_distribution = pd.DataFrame.from_records(unique_values_distribution, columns=columns_heading)

data_distribution
train_data["spacegroup"].unique()
train_data["number_of_total_atoms"].unique()
#correlation matrix

corrmat = train_data.corr()

plt.figure(figsize=(10, 10))

sns.heatmap(corrmat, cmap='viridis');
#correlation matrix

corrmat = train_data.corr()



plt.figure(figsize=(12, 12))



sns.heatmap(corrmat[(corrmat >= 0.4) | (corrmat <= -0.4)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 11}, square=True);
# 1. define the columns for train_data



train_data = train_data[[ 'spacegroup',                 'number_of_total_atoms', 

                          'percent_atom_al',            'percent_atom_ga',            'percent_atom_in', 

                          'lattice_vector_1_ang',       'lattice_vector_2_ang',       'lattice_vector_3_ang', 

                          'lattice_angle_alpha_degree', 'lattice_angle_beta_degree',  'lattice_angle_gamma_degree',

                          'formation_energy_ev_natom',  'bandgap_energy_ev'

                        ]]



train_data.columns = [    'spacegroup',                 'number_of_total_atoms', 

                          'percent_atom_al',            'percent_atom_ga',            'percent_atom_in', 

                          'lattice_vector_1_ang',       'lattice_vector_2_ang',       'lattice_vector_3_ang', 

                          'lattice_angle_alpha_degree', 'lattice_angle_beta_degree',  'lattice_angle_gamma_degree',

                          'formation_energy_ev_natom',  'bandgap_energy_ev'

                        ]



# 2. define the columns for test_data



test_data = test_data[[   'spacegroup',                 'number_of_total_atoms', 

                          'percent_atom_al',            'percent_atom_ga',            'percent_atom_in', 

                          'lattice_vector_1_ang',       'lattice_vector_2_ang',       'lattice_vector_3_ang', 

                          'lattice_angle_alpha_degree', 'lattice_angle_beta_degree',  'lattice_angle_gamma_degree'

                        ]]



test_data.columns = [     'spacegroup',                 'number_of_total_atoms', 

                          'percent_atom_al',            'percent_atom_ga',            'percent_atom_in', 

                          'lattice_vector_1_ang',       'lattice_vector_2_ang',       'lattice_vector_3_ang', 

                          'lattice_angle_alpha_degree', 'lattice_angle_beta_degree',  'lattice_angle_gamma_degree'

                        ]
# 3. Separate the target from train_data and split the train_data into training and testing data.

X_train = train_data.drop([ "formation_energy_ev_natom", "bandgap_energy_ev"], axis = 1)



Y_formation_energy = train_data['formation_energy_ev_natom']

Y_bandgap_energy   = train_data['bandgap_energy_ev']



# 

fX_train_data, fX_test_data, fy_train_target, fy_test_target  = train_test_split(X_train, Y_formation_energy, 

                                                                                 test_size=0.25, random_state=101)

bX_train_data, bX_test_data, by_train_target, by_test_target  = train_test_split(X_train, Y_bandgap_energy, 

                                                                                 test_size=0.25, random_state=101)
# 4. Feature Importance



import xgboost as xgb



xgr = XGBRegressor()

classifier = xgb.sklearn.XGBRegressor(nthread=-1, seed=42)

# Feature Importance : for Formation Energy

xgr.fit(fX_train_data, fy_train_target)

plt.figure(figsize=(20,15))

xgb.plot_importance(xgr, ax=plt.gca())
# Feature Importance : for Bandgap Energy

xgr.fit(bX_train_data, by_train_target)

plt.figure(figsize=(20,15))

xgb.plot_importance(xgr, ax=plt.gca())


plt.figure(figsize=(15,15))

xgb.plot_tree(xgr, num_trees=7,ax=plt.gca())

#  Based on the important features for both Formation Energy and Bandgap Energy, we can remove 'spacegroup',  

#  'number_of_total_atoms' and 'percent_atom_ga'.



# And split the train data for Formation and Bandgap energy



# 5. Re-define the columns for train_data



train_data = train_data[[ 'percent_atom_al',            'percent_atom_in', 

                          'lattice_vector_1_ang',       'lattice_vector_2_ang',       'lattice_vector_3_ang', 

                          'lattice_angle_alpha_degree', 'lattice_angle_beta_degree',  'lattice_angle_gamma_degree',

                          'formation_energy_ev_natom',  'bandgap_energy_ev'

                        ]]



train_data.columns = [    'percent_atom_al',            'percent_atom_in', 

                          'lattice_vector_1_ang',       'lattice_vector_2_ang',       'lattice_vector_3_ang', 

                          'lattice_angle_alpha_degree', 'lattice_angle_beta_degree',  'lattice_angle_gamma_degree',

                          'formation_energy_ev_natom',  'bandgap_energy_ev'

                        ]



# 6. Re-define the columns for test_data



test_data = test_data[[   'percent_atom_al',            'percent_atom_in', 

                          'lattice_vector_1_ang',       'lattice_vector_2_ang',       'lattice_vector_3_ang', 

                          'lattice_angle_alpha_degree', 'lattice_angle_beta_degree',  'lattice_angle_gamma_degree'

                        ]]



test_data.columns = [     'percent_atom_al',            'percent_atom_in', 

                          'lattice_vector_1_ang',       'lattice_vector_2_ang',       'lattice_vector_3_ang', 

                          'lattice_angle_alpha_degree', 'lattice_angle_beta_degree',  'lattice_angle_gamma_degree'

                        ]



# 7. Separate the target from train_data and split the train_data into training and testing data.

X_train = train_data.drop([ "formation_energy_ev_natom", "bandgap_energy_ev"], axis = 1)



Y_formation_energy = train_data['formation_energy_ev_natom']

Y_bandgap_energy   = train_data['bandgap_energy_ev']



# 

fX_train_data, fX_test_data, fy_train_target, fy_test_target  = train_test_split(X_train, Y_formation_energy, 

                                                                                 test_size=0.25, random_state=101)

bX_train_data, bX_test_data, by_train_target, by_test_target  = train_test_split(X_train, Y_bandgap_energy, 

                                                                                 test_size=0.25, random_state=101)


dtr = DecisionTreeRegressor()

rfr = RandomForestRegressor()

gbr = GradientBoostingRegressor()

abr = AdaBoostRegressor()        

bgr = BaggingRegressor()

etr = ExtraTreesRegressor()

xgr = XGBRegressor()  





regressors = {  'ABR' : abr,  'BGR' : bgr, 'DTR' : dtr, 'ETR': etr, 'GBR' : gbr, 'XGR' : xgr  }  



param ={}

random_state = 101

reg = DecisionTreeRegressor(criterion='mse', max_depth=3, random_state = random_state, min_samples_leaf= 40)





def hyper_parameters(var):

    

    if  var == 'DTR':

        param = { #'decisiontreeregressor__criterion': ['mse','mae'],

                  'decisiontreeregressor__max_depth': [3],

                  #'decisiontreeregressor__max_features': ['auto', 'sqrt', 'log2'],   

                  #'decisiontreeregressor__max_leaf_nodes': [250] ,

                  'decisiontreeregressor__min_samples_split':  [10],

                  'decisiontreeregressor__min_samples_leaf': [40 ],

                  #'decisiontreeregressor__splitter',

                  #'decisiontreeregressor__min_impurity_decrease':[50],

                  'decisiontreeregressor__random_state': [random_state]

                }   



    elif var == 'RFR':

        param = {#'randomforestregressor__max_features' : ['auto'],

                 'randomforestregressor__max_depth': [3],

                 'randomforestregressor__n_estimators': [300],

                 'randomforestregressor__min_samples_split':  [10],

                 'randomforestregressor__min_samples_leaf': [40 ],

                 #'randomforestregressor__min_impurity_decrease':[50],

                 'randomforestregressor__random_state' :[random_state],

                 'randomforestregressor__oob_score':[True]

                }

        

    elif var == 'GBR':

        param = {'gradientboostingregressor__n_estimators': [300],

                 'gradientboostingregressor__learning_rate': [0.1],

                 'gradientboostingregressor__max_depth': [3],

                 #'gradientboostingregressor__loss': ['ls'],

                 'gradientboostingregressor__min_samples_split':  [10],

                 #'gradientboostingregressor__max_leaf_nodes'

                 'gradientboostingregressor__min_samples_leaf': [40 ],

                 'gradientboostingregressor__max_features': ['auto'],

                 #'gradientboostingregressor__alpha'

                 'gradientboostingregressor__random_state' :[random_state]

                } 

        

    elif var == 'ABR':

        param = {  'adaboostregressor__random_state': [random_state],  

                   'adaboostregressor__base_estimator': [reg],   

                   'adaboostregressor__n_estimators': [300] , 

                   'adaboostregressor__loss': ['exponential'],   #['linear', 'square', 'exponential']  

                   'adaboostregressor__learning_rate': [0.1] 

                    }

    elif var == 'BGR':

        param = { 'baggingregressor__n_estimators': [300], 

                  #'baggingregressor__max_features': [9], #[7,8,9],

                  'baggingregressor__random_state': [random_state],

                  'baggingregressor__n_jobs':[-1],

                  'baggingregressor__bootstrap':[True],

                  #'baggingregressor__base_estimator':[rfr],

                  'baggingregressor__oob_score':[True],



                  'baggingregressor__max_samples': [325]

                 }

    elif var == 'ETR':

        param =  { 'extratreesregressor__random_state': [random_state],   

                   #'extratreesregressor__criterion': ['mse','friedman_mse'],

                   #'extratreesregressor__max_features': ['auto', 'sqrt', 'log2'], 

                   'extratreesregressor__max_depth':[3],

                   'extratreesregressor__n_estimators':[300],

                   'extratreesregressor__min_samples_split' :[10],

                   'extratreesregressor__oob_score':[True],                 

                    'extratreesregressor__bootstrap':[True],

                    'extratreesregressor__min_samples_leaf':[40]

                     }    

    elif var == 'XGR':     

        param = { 'xgbregressor__max_depth': [3],

                  'xgbregressor__learning_rate': [0.1],

                  'xgbregressor__n_estimators': [300],

                  #'xgbregressor__n_jobs': [-1],

                  'xgbregressor__reg_lambda': [0.5],

                  'xgbregressor__max_delta_step': [0.3],

                  #'xgbregressor__min_child_weight': [1,2],

                  'xgbregressor__seed': [42],

                  'xgbregressor__random_state':  [random_state]

                 

                 }

       

    return param
import  time

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

def root_mean_squared_log_error(h, y): 

    """

    Compute the Root Mean Squared Log Error for hypthesis h and targets y

    Args:

        h - numpy array containing predictions with shape (n_samples, n_targets)

        y - numpy array containing targets with shape (n_samples, n_targets)

    """

    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())





def collect_error_score(target, prediction):

    meansquare_error = mean_squared_error (target, prediction)                 # Mean Squared Error  

    r2square_error = r2_score(target, prediction)                              # R Squared  

    rmslog_error = root_mean_squared_log_error(prediction, target)             # Root Mean Square Log Error  

    #meanabsolute_error = mean_absolute_error (target, prediction)              # Absolute Mean Error 

    #msle = mean_squared_log_error(target, prediction)

    

    return ( meansquare_error, r2square_error, rmslog_error)

    

########    

def predict_evaluate(train_feature, train_target, test_feature, test_target):

    

    train_reg = []           # to collect the trained regressors

    test_error_scores = []   # to collect the error scores

    train_error_scores = []   # to collect the error scores

    

    print ("==== Start training  Regressors ====")

    t = time.time()

    for i, model in regressors.items():

        it = time.time()

        pipe = make_pipeline(preprocessing.PolynomialFeatures(degree = 4), model)    #StandardScaler, MinMaxScaler

        hyperparameters = hyper_parameters(i)

        trainedmodel = GridSearchCV(pipe, hyperparameters, n_jobs = -1, verbose = 1, scoring = 'r2', cv=5)

        # Fit and predict train data

        #---------------------------

        trainedmodel.fit(train_feature, train_target)

        

        print (i,' trained best score :: ',trainedmodel.best_score_)

        print (":::::::::::::::::::::::::::")

        

        #print (i,' - ',trainedclfs.best_params_)

        #print (trainedmodel.best_estimator_)

        

         # predict train data

        pred_train = trainedmodel.predict(train_feature)

        

        # Get error scores on train data

        tmse, tr2, trmsle = collect_error_score(train_target, pred_train)

        train_error_scores.append ((i,  tmse, tr2, trmsle))

        # predict test data

        pred_test = trainedmodel.predict(test_feature)

        

        # Get error scores on test data

        mse, r2, rmsle = collect_error_score(test_target, pred_test)



        test_error_scores.append ((i,  mse, r2, rmsle))

        train_reg.append ((i, trainedmodel))

        print (i, " :  Training time :  ({0:.3f} s)\n".format(time.time() - it) )

    print ("==== Finished training  Regressors ====\n")    

    print (" Total training time :  ({0:.3f} s)\n".format(time.time() - t) )

    return ( train_error_scores,test_error_scores, train_reg)

    

def error_table (score, labels, sort_col ):

    #labels  = ['Clf','mean absolute error','mean square error','R2 squared', 'Mean Sq Log Error', 'Root Mean Sq Log Error']

    scored_df = pd.DataFrame.from_records(score, columns=labels, index = None)

    sorted_scored = scored_df.sort_values(by = sort_col, ascending=False)

    return sorted_scored

    

# Call "predict_evaluate" for Formation Energy

# pass training and test data for Formation energy to "predict_evaluate"

# "predict_evaluate" will return 

#      1. the classifier short initials ( like 'ETR' for ExtraTreesRegressor(), DTR for  DecisionTreeRegressor() etc...)

#      2. training data error scores ( like mean square error  R2 squared  Root Mean Sq Log Error etc.. ) and

#      3. test data error scores ( like mean square error  R2 squared  Root Mean Sq Log Error etc.. )    



train_form_error_scores,form_error_scores, trained_pred_form = predict_evaluate(fX_train_data, fy_train_target, fX_test_data, fy_test_target)   

labels  = ['train Regr','train MSE', 'train R2', 'train RMSLE']



#############

print("Formation Energy scores : on test data - ordered by train R Squared : \n")

train_formation_energy_score = error_table (train_form_error_scores, labels,  'train R2' )

train_formation_energy_score

#train_error_score_df = pd.DataFrame.from_records(train_formation_energy_score, columns=['train Regr','train mean square error', 'train R Squared', 'train Root Mean Sq Log Error' ], index = None)

#train_error_score_df
labels  = ['test Regr','test MSE', 'test R2', 'test RMSLE']

#############

print("Formation Energy scores : on test data - ordered by test R Squared : \n")

test_formation_energy_score = error_table (form_error_scores, labels,  'test R2' )

test_formation_energy_score

score_df = pd.concat ([train_formation_energy_score, test_formation_energy_score], axis = 1)

score_df

# diff is the difference between train R2 and Test R2

diff = score_df['train R2']- score_df['test R2']

score_df = pd.concat ([score_df, diff], axis = 1)

score_df = score_df.rename(columns={0:'R2 Diff'})

score_df = score_df.sort_values(by = 'test R2', ascending=False)

score_df
R2score = score_df[['train Regr','train R2','test R2']]



R2score.plot(kind='bar', ylim=None, figsize=(10,4), align='center', colormap="jet") 

plt.xticks(np.arange(6), R2score['train Regr']) 

plt.ylabel('Error Score') 

plt.title('Formation Energy : R2 Score - Distribution by Regressor') 

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)

MSE_score =score_df[['train Regr','train MSE','test MSE']]



MSE_score.plot(kind='bar', ylim=None, figsize=(10,4), align='center', colormap="copper")

plt.xticks(np.arange(6), MSE_score['train Regr'])

plt.ylabel('Error Score')

plt.title('Formation Energy : MSE - Distribution by Regressor')

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)


RMSLE_score =score_df[['train Regr','train RMSLE','test RMSLE']]



RMSLE_score.plot(kind='bar', ylim=None, figsize=(10,4), align='center', colormap="tab20")

plt.xticks(np.arange(6), RMSLE_score['train Regr'])

plt.ylabel('RMSLE Error Score')

plt.title('RMSLE - Distribution by Regressor')

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
##############

# Call "predict_evaluate" for Bandgap Energy

# pass training and test data for Formation energy to "predict_evaluate"

# "predict_evaluate" will return 

#      1. the classifier short initials

#      2. training data error scores and

#      3. test data error scores



train_band_error_scores,test_band_error_scores, trained_pred_band   = predict_evaluate(bX_train_data, by_train_target, bX_test_data, by_test_target  )
labels  = ['train Regr','train MSE', 'train R2', 'train RMSLE']



print("Bandgap Energy error scores on test data - ordered by Train R2 : \n")

train_bandgap_energy_score = error_table (train_band_error_scores, labels, 'train R2')

train_bandgap_energy_score

labels  = ['test Regr','test MSE', 'test R2', 'test RMSLE']

#############

print("Bandgap Energy error scores on test data - ordered by test R Squared : \n")

test_bandgap_energy_score = error_table (test_band_error_scores, labels,  'test R2' )

test_bandgap_energy_score

score_df = pd.concat ([train_bandgap_energy_score, test_bandgap_energy_score], axis = 1)

score_df = score_df.sort_values(by = 'test R2', ascending=False)

score_df
R2score = score_df[['train Regr','train R2','test R2']]



R2score.plot(kind='bar', ylim=None, figsize=(10,4), align='center', colormap="jet") 

plt.xticks(np.arange(6), R2score['train Regr']) 

plt.ylabel('Error Score') 

plt.title('Bandgap Energy : R2 Score - Distribution by Regressor') 

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)

MSE_score =score_df[['train Regr','train MSE','test MSE']]



MSE_score.plot(kind='bar', ylim=None, figsize=(10,4), align='center', colormap="copper")

plt.xticks(np.arange(6), MSE_score['train Regr'])

plt.ylabel('Error Score')

plt.title('Bandgap Energy : MSE - Distribution by Regressor')

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)


RMSLE_score =score_df[['train Regr','train RMSLE','test RMSLE']]



RMSLE_score.plot(kind='bar', ylim=None, figsize=(10,4), align='center', colormap="tab20")

plt.xticks(np.arange(6), RMSLE_score['train Regr'])

plt.ylabel('RMSLE Error Score')

plt.title('Bandgap Energy : RMSLE - Distribution by Regressor')

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
# select the best regressor

X_test_data = test_data #.drop(['id', 'number_of_total_atoms'], axis = 1)



def select_regressor(score_data, predictor) :

    print (" The Best prediction regressor with minimum MSE prediction \n ")

    print (" --------------------------------------------------------- \n ")

    """

    # find the regressor initial such as 'GBR','XGR','BGRD' etc.,from "formation_scored_tuned" for which MSE is lowest.

    # argmin() checks for the minimum value in the column 'mean square error' and returns the corresponing index value of the row

    """

    #val = score_data.loc[score_data['mean square error'].argmin(), 'Regr']

    val = score_data.loc[score_data['test R2'].argmax(), 'test Regr']

    # iterate through the items in train_reg collection and compare with the minimum MSE regressor initials extracted above 

    # to select the best trained regressor

    #val ='ETR'

    for i in range(len(predictor)):

        if predictor[i][0] == val:

            print (predictor[i])

            selected_reg = predictor[i][1]            

    return selected_reg

# Call the select_regressor function to get the regressor and

# predict the formation energy for our original test_data (test.csv)



selected_regressor_form = select_regressor(test_formation_energy_score, trained_pred_form)



test_pred_form = selected_regressor_form.predict(X_test_data)
len(test_pred_form)
# Call the select_regressor function to get the regressor and

# predict the bandgap energy for our original test_data (test.csv)

selected_regressor_band = select_regressor(test_bandgap_energy_score, trained_pred_band)

#

test_pred_band = selected_regressor_band.predict(X_test_data)

len(test_pred_band)


test_csv = pd.read_csv('../input/test.csv')
## Save the the output to "submission.csv" file formation_energy_ev_natom	bandgap_energy_ev



id=(test_csv['id'])           # 'id' from test_data of test.csv



submission_id = pd.DataFrame({ 'id' : id})

submission_form = pd.DataFrame({ 'formation_energy_ev_natom': test_pred_form})  # dataframe for predict formation energy

submission_band = pd.DataFrame({ 'bandgap_energy_ev': test_pred_band})          # dataframe for predict bandgap energy

submission_df =  pd.concat([submission_form,submission_band],axis=1)            

submission_df =  pd.concat([submission_id,submission_df],axis=1)                #dataframe with 'id', formation and bandgap energy 

# save into submission.csv

# submission_df.to_csv('..\submission.csv', index=False)
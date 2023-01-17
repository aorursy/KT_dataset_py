# import libraries and Load data  

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn import preprocessing

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_predict, train_test_split

from sklearn.metrics import r2_score, mean_squared_error  #, mean_squared_log_error, mean_absolute_error



from sklearn.linear_model import LinearRegression, Ridge,  RANSACRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost.sklearn import XGBRegressor

#from scipy.stats import randint

#import scipy.stats as st



# load data

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

#train_data = pd.read_csv('D:\DataScienceCourse\TensorFlow Bootcamp/Kaggle_Predicting_Transparent_Conductors/train.csv')

#test_data = pd.read_csv('D:\DataScienceCourse\TensorFlow Bootcamp/Kaggle_Predicting_Transparent_Conductors/test.csv/test.csv')

train_data.head()

train_data.shape
test_data.head()
test_data.shape
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
# 1. define the columns for train_data



train_data = train_data[[ 'spacegroup', #'number_of_total_atoms', 

                         'percent_atom_al', 'percent_atom_ga',    'percent_atom_in', 

                         'lattice_vector_1_ang',     'lattice_vector_2_ang','lattice_vector_3_ang',

                         'lattice_angle_alpha_degree','lattice_angle_beta_degree','lattice_angle_gamma_degree',

                         'formation_energy_ev_natom','bandgap_energy_ev']]



train_data.columns = [ 'spacegroup', #'number_of_total_atoms',                        

                       'percent_atom_al', 'percent_atom_ga',     'percent_atom_in',  

                       'lattice_vector_1_ang',     'lattice_vector_2_ang', 'lattice_vector_3_ang', 

                       'lattice_angle_alpha_degree', 'lattice_angle_beta_degree', 'lattice_angle_gamma_degree', 

                       'formation_energy_ev_natom','bandgap_energy_ev']



# 2. Separate the target from train_data and split the train_data into training and testing data.

X_train = train_data.drop([ "formation_energy_ev_natom", "bandgap_energy_ev"], axis = 1)



Y_formation_energy = train_data['formation_energy_ev_natom']

Y_bandgap_energy   = train_data['bandgap_energy_ev']



# 

fX_train_data, fX_test_data, fy_train_target, fy_test_target  = train_test_split(X_train, Y_formation_energy, 

                                                                                 test_size=0.25, random_state=42)

bX_train_data, bX_test_data, by_train_target, by_test_target  = train_test_split(X_train, Y_bandgap_energy, 

                                                                                 test_size=0.25, random_state=42)
lrg = LinearRegression()

svr = SVR()

rrg = RANSACRegressor()

rid = Ridge()

dtr = DecisionTreeRegressor()

rfr = RandomForestRegressor()

gbr = GradientBoostingRegressor()

abr = AdaBoostRegressor()        

bgr = BaggingRegressor()

etr = ExtraTreesRegressor()

xgr = XGBRegressor(nthread=-1)





regressors = {  'DTR' : dtr, 'RRG' : rrg, 'SVR': svr, 'RID' : rid, 'ABR' : abr,

                'BGR' : bgr, 'RFR' : rfr, 'ETR': etr, 'GBR' : gbr, 'XGR' : xgr,

                'DTRD' : dtr, 'SVRD': svr, 'RIDD' : rid, 'ABRD' : abr,

                'BGRD' : bgr, 'RFRD' : rfr, 'ETRD': etr, 'GBRD' : gbr, 'XGRD' : xgr} 
param ={}

def hyper_parameters(var):

    

    if var == 'SVR':

        param = { 'svr__gamma': ['auto'],                                  #[0.0001, 0.001, 0.005, 0.01, 0.1]    

                  'svr__epsilon': [0.1],      

                  'svr__tol': [0.001],       

                  'svr__cache_size': [200,250,300,500] 

                }

    elif var == 'RRG':

        param = { 'ransacregressor__base_estimator': [lrg], 

                  'ransacregressor__min_samples':[ 5,10,15],

                  'ransacregressor__residual_threshold': [1.0,5.0,10.0],

                  'ransacregressor__max_trials': [50,100,200] 

                }

    elif var == 'RID':

        param = { 'ridge__max_iter': [None],  

                  'ridge__solver': ['auto'], 

                  'ridge__alpha': [1.0, 0.5,1.5],      

                  'ridge__normalize': [False],       

                  'ridge__tol': [1.0,0.001,0.01,0.1] 

                }

    elif var == 'DTR':

        param = { 'decisiontreeregressor__criterion': ['mse','mae'],

                  'decisiontreeregressor__max_depth': [7],

                  'decisiontreeregressor__max_features': ['auto', 'sqrt', 'log2'],   

                  'decisiontreeregressor__max_leaf_nodes': [200] ,

                  'decisiontreeregressor__min_samples_split':  [20],

                  'decisiontreeregressor__min_samples_leaf': [7, 10,50 ]

                } 



    elif var == 'RFR':

        param = {'randomforestregressor__max_features' : ['auto'],

                 'randomforestregressor__max_depth': [7],

                 'randomforestregressor__n_estimators': [90,100],

                 'randomforestregressor__min_samples_split':  [6,7,8,9]

                }

        

    elif var == 'GBR':

        param = {'gradientboostingregressor__n_estimators': [90],

                 'gradientboostingregressor__learning_rate': [0.1],

                 'gradientboostingregressor__max_depth': [3,7]

                 #'gradientboostingregressor__loss': ['ls']

                } 

        

    elif var == 'ABR':

        param = { #'adaboostregressor__random_state': [None],  

                  #'adaboostregressor__base_estimator': [None],   

                   'adaboostregressor__n_estimators': [160,170,180,190,200] , 

                   'adaboostregressor__loss': ['exponential'],   #['linear', 'square', 'exponential']  

                   'adaboostregressor__learning_rate': [0.1] 

                    }

    elif var == 'BGR':

        param = { 'baggingregressor__n_estimators': [50,51,52], 

                  'baggingregressor__max_features': [9], #[7,8,9],

                  #'baggingregressor__random_state': [None, 10,100],

                  'baggingregressor__max_samples': [300]

                 }

    elif var == 'ETR':

        param =  { #'extratreesregressor__random_state': [None,1,5],   #

                   'extratreesregressor__criterion': ['mse'],

                   'extratreesregressor__max_features': ['auto', 'sqrt', 'log2'], 

                   'extratreesregressor__n_estimators':[70,80,90]

                     }    

    elif var == 'XGR':     

        param = { 'xgbregressor__max_depth': [3],

                  'xgbregressor__learning_rate': [0.1],

                  'xgbregressor__n_estimators': [80],

                  #'xgbregressor__n_jobs': [dep]

                  'xgbregressor__reg_lambda': [0.5],

                  'xgbregressor__max_delta_step': [0.3]

                  #'xgbregressor__min_child_weight': [1,2]

                 }

    # regressor with default parameters   

    elif var == 'SVRD':

            param = { }

    elif var == 'RIDD':

            param = { }

    elif var == 'DTRD':

            param = { }

    elif var == 'RFRD':

            param = { }

    elif var == 'GBRD':

            param = { }

    elif var == 'ABRD':

            param = { }

    elif var == 'BGRD':

            param = { }

    elif var == 'ETRD':

            param =  { }

    elif var == 'XGRD':     

            param = { }

       

    return param
import  time

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

    

    print ("==== Start training  Regressors ====")

    t = time.time()

    for i, model in regressors.items():

       

        pipe = make_pipeline(preprocessing.StandardScaler(), model)

        hyperparameters = hyper_parameters(i)

        trainedmodel = GridSearchCV(pipe, hyperparameters, cv=15)

    

        # Fit and predict train data

        #---------------------------

        trainedmodel.fit(train_feature, train_target)

        

        print (i,' trained best score :: ',trainedmodel.best_score_)

        print (":::::::::::::::::::::::::::")

        

        #print (i,' - ',trainedclfs.best_params_)

        #print (trainedmodel.best_estimator_)

        

        # predict test data

        pred_test = trainedmodel.predict(test_feature)

        

        # Get error scores on test data

        mse, r2, rmsle = collect_error_score(test_target, pred_test)



        test_error_scores.append ((i,  mse, r2, rmsle))

        train_reg.append ((i, trainedmodel))

        

    print ("==== Finished training  Regressors ====")    

    print (" Total training time :  ({0:.3f} s)\n".format(time.time() - t) )

    return ( test_error_scores, train_reg)

    

def error_table (score, labels, sort_col ):

    #labels  = ['Clf','mean absolute error','mean square error','R2 squared', 'Mean Sq Log Error', 'Root Mean Sq Log Error']

    scored_df = pd.DataFrame.from_records(score, columns=labels, index = None)

    sorted_scored = scored_df.sort_values(by = sort_col, ascending=True)

    return sorted_scored

    

# Call "predict_evaluate" for Formation Energy

# pass training and test data for Formation energy to "predict_evaluate"

# "predict_evaluate" will return 

#      1. the classifier short initials ( like 'ETR' for ExtraTreesRegressor(), DTR for  DecisionTreeRegressor() etc...)

#      2. training data error scores ( like mean square error  R2 squared  Root Mean Sq Log Error etc.. ) and

#      3. test data error scores ( like mean square error  R2 squared  Root Mean Sq Log Error etc.. )    



form_error_scores, trained_pred_form = predict_evaluate(fX_train_data, fy_train_target, fX_test_data, fy_test_target)   

labels  = ['Regr','mean square error', 'R Squared', 'Root Mean Sq Log Error']

#############

print("Formation Energy scores : on test data - ordered by Mean Square Error : \n")

formation_energy_score = error_table (form_error_scores, labels,  'mean square error' )

formation_energy_score

# Select top 10 scores

formation_energy_score_10 = formation_energy_score[0:10]

formation_energy_score_10
formation_energy_score_10.plot(kind='bar', ylim=(-0.20,1.0), figsize=(12,4), align='center', colormap="tab20")

plt.xticks(np.arange(10), formation_energy_score_10.Regr)

plt.ylabel('Error Score')

plt.title('Error Score Distribution by Regressor')

plt.legend(bbox_to_anchor=(1.3, 0.9), loc=5, borderaxespad=0.5)
error_score_df = pd.DataFrame.from_records(formation_energy_score_10, columns=['Regr','mean square error','R Squared', 'Root Mean Sq Log Error' ], index = None)



error_score_df
RMSLEscore = error_score_df[['Regr','Root Mean Sq Log Error']]



RMSLEscore.plot(kind='bar', ylim=None, figsize=(10,4), align='center', colormap="jet") 

plt.xticks(np.arange(10), RMSLEscore.Regr) 

plt.ylabel('Error Score') 

plt.title('Root Mean Square Logistic Error - Distribution by Regressor') 

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)

MSEscore = error_score_df[['Regr','mean square error']]



MSEscore.plot(kind='bar', ylim=None, figsize=(10,4), align='center', colormap="rainbow") 

plt.xticks(np.arange(10), MSEscore.Regr) 

plt.ylabel('Error Score') 

plt.title('Mean Square Error - Distribution by Regressor') 

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
RSquarescore =error_score_df[['Regr','R Squared']]



RSquarescore.plot(kind='bar', ylim=None, figsize=(10,4), align='center', colormap="Spectral")

plt.xticks(np.arange(20), RSquarescore.Regr)

plt.ylabel('Error Score')

plt.title('R Squared - Distribution by Regressor')

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
MSE_RMSLEscore =error_score_df[['Regr','mean square error','Root Mean Sq Log Error']]



MSE_RMSLEscore.plot(kind='bar', ylim=None, figsize=(10,4), align='center', colormap="copper")

plt.xticks(np.arange(10), MSE_RMSLEscore.Regr)

plt.ylabel('Error Score')

plt.title('MSE and RMSLE - Distribution by Regressor')

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
# select the best regressor

def select_regressor(score_data, predictor) :

    print (" The Best prediction regressor with minimum MSE prediction \n ")

    print (" --------------------------------------------------------- \n ")

    """

    # find the regressor initial such as 'GBR','XGR','BGRD' etc.,from "formation_scored_tuned" for which MSE is lowest.

    # argmin() checks for the minimum value in the column 'mean square error' and returns the corresponing index value of the row

    """

    #val = score_data.loc[score_data['mean square error'].argmin(), 'Regr']

    val = score_data.loc[score_data['R Squared'].argmax(), 'Regr']

    # iterate through the items in train_reg collection and compare with the minimum MSE regressor initials extracted above 

    # to select the best trained regressor

    

    for i in range(len(predictor)):

        if predictor[i][0] == val:

            print (predictor[i])

            selected_reg = predictor[i][1]            

    return selected_reg



X_test_data = test_data.drop(['id', 'number_of_total_atoms'], axis = 1)
# Call the select_regressor function to get the regressor and

# predict the formation energy for our original test_data (test.csv)



selected_regressor_form = select_regressor(formation_energy_score, trained_pred_form)



test_pred_form = selected_regressor_form.predict(X_test_data)
len (test_pred_form), test_pred_form.size 
##############

# Call "predict_evaluate" for Bandgap Energy

# pass training and test data for Formation energy to "predict_evaluate"

# "predict_evaluate" will return 

#      1. the classifier short initials

#      2. training data error scores and

#      3. test data error scores



test_error_scores, trained_pred_band   = predict_evaluate(bX_train_data, by_train_target, bX_test_data, by_test_target  )
labels  = ['Regr','mean square error', 'R Squared', 'Root Mean Sq Log Error']



print("bandgap energy error scores on test data - ordered by mean square error : \n")

bandgap_energy_score = error_table (test_error_scores, labels, 'mean square error')

bandgap_energy_score
# Select top 10 scores

bandgap_energy_score_10 = bandgap_energy_score[0:10]

bandgap_energy_score_10

bandgap_energy_score_10.plot(kind='bar', ylim=None, figsize=(12,4), align='center', colormap="tab20")

plt.xticks(np.arange(10), bandgap_energy_score_10.Regr)

plt.ylabel('Error Score')

plt.title('Error Score Distribution by Regressor')

plt.legend(bbox_to_anchor=(1.3, 0.9), loc=5, borderaxespad=0.5)
error_score_df = pd.DataFrame.from_records(bandgap_energy_score_10, columns=['Regr','mean square error','R Squared', 'Root Mean Sq Log Error' ], index = None)



error_score_df
RMSLEscore = error_score_df[['Regr','Root Mean Sq Log Error']]



RMSLEscore.plot(kind='bar', ylim=None, figsize=(10,3), align='center', colormap="jet") 

plt.xticks(np.arange(10), RMSLEscore.Regr) 

plt.ylabel('Error Score') 

plt.title('Root Mean Square Logistic Error - Distribution by Regressor') 

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)

print("bandgap energy error scores on test data - ordered by mean square error : \n")

MSEscore = error_score_df[['Regr','mean square error']]



MSEscore.plot(kind='bar', ylim=None, figsize=(10,4), align='center', colormap="rainbow") 

plt.xticks(np.arange(10), MSEscore.Regr) 

plt.ylabel('Error Score') 

plt.title('Mean Square Error - Distribution by Regressor') 

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
# Call the select_regressor function to get the regressor and

# predict the bandgap energy for our original test_data (test.csv)

selected_regressor_band = select_regressor(bandgap_energy_score, trained_pred_band)

#

test_pred_band = selected_regressor_band.predict(X_test_data)

len(test_pred_band)


## Save the the output to "submission.csv" file formation_energy_ev_natom	bandgap_energy_ev



id=(test_data['id'])           # 'id' from test_data of test.csv



submission_id = pd.DataFrame({ 'id' : id})

submission_form = pd.DataFrame({ 'formation_energy_ev_natom': test_pred_form})  # dataframe for predict formation energy

submission_band = pd.DataFrame({ 'bandgap_energy_ev': test_pred_band})          # dataframe for predict bandgap energy

submission_df =  pd.concat([submission_form,submission_band],axis=1)            

submission_df =  pd.concat([submission_id,submission_df],axis=1)                #dataframe with 'id', formation and bandgap energy 

# save into submission.csv

#submission_df.to_csv('D:\DataScienceCourse\TensorFlow Bootcamp\Kaggle_Predicting_Transparent_Conductors\submission.csv', index=False)
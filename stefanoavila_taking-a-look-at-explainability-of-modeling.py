# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import shap



from sklearn.model_selection import train_test_split



import scikitplot as skplt

import os



import matplotlib.pyplot as plt

from matplotlib.colors import Normalize



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import GridSearchCV



import lightgbm as lgb



from sklearn import metrics

from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc, classification_report



from sklearn.preprocessing import StandardScaler





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
# load data

data = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
# number of Patient ID x unique Patient ID

print('number of Patient ID:', data.shape[0], '\nnumber of unique Patient ID:',data['Patient ID'].nunique())
# drop Patient ID

data.drop(columns='Patient ID', inplace = True)
# number of missing values

null_columns_percent_by_positive = (data[data['SARS-Cov-2 exam result']=='positive'].isnull().sum()/data.shape[0]).sort_values(ascending=True)

null_columns_percent_by_total = (data.isnull().sum()/data.shape[0]).sort_values(ascending=True)

null_columns_percent_by_negative = (data[data['SARS-Cov-2 exam result']=='negative'].isnull().sum()/data.shape[0]).sort_values(ascending=True)



sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(20,30))



# Plot the total of null values

sns.set_color_codes("pastel")

sns.barplot(x=null_columns_percent_by_total.values, y=null_columns_percent_by_total.index, label="Total", color="b")



# Plot the negative case number with null values

sns.set_color_codes("muted")

sns.barplot(x=null_columns_percent_by_negative.values, y=null_columns_percent_by_negative.index, label="Negative case", color="b")



# Plot the positive case number with null values

sns.set_color_codes("dark")

sns.barplot(x=null_columns_percent_by_positive.values, y=null_columns_percent_by_positive.index, label="Positive case", color="b")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="upper right", frameon=True)

ax.set(xlim=(0, 1), ylabel="", xlabel="Percent of missing values")

sns.despine(left=True, bottom=True)
# how these missing values are distributed

plt.figure( figsize=(20,30))

sns.heatmap(data.isnull().T, cbar=False)
# distribution patient age quantile

plt.figure( figsize=(20,5))

negative = data[data["SARS-Cov-2 exam result"] == 'negative']

positive = data[data["SARS-Cov-2 exam result"] == 'positive']

sns.distplot(negative['Patient age quantile'], label="Negative", bins=20, kde=False)

sns.distplot(positive['Patient age quantile'], label="Positive", bins=20, kde=False)

plt.legend()
# Drop columns with 100% of missing values

cols2drop = ['Partial thromboplastin time\xa0(PTT)\xa0','Urine - Sugar','Mycoplasma pneumoniae', 'D-Dimer', 'Prothrombin time (PT), Activity']

data.drop(columns=cols2drop, inplace = True)
# get only not null rown of target

data = data[data['SARS-Cov-2 exam result'].notnull()]
# fill missing values

data.fillna(-999, inplace=True)
def object_to_number(dataframe):

    categorical_var = list(dataframe.select_dtypes(include='object').columns)

    for categorical in categorical_var:

        dataframe[categorical] = dataframe[categorical].astype('category')

        dataframe[categorical+'_cat'] = dataframe[categorical].cat.codes

        dataframe.drop(categorical, axis=1, inplace=True)

    return dataframe
# categorical to number

data = object_to_number(data)
# rename columns

data.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in data.columns]
# get label

target = 'SARS_Cov_2_exam_result_cat'

train = data.drop(columns=target)

train_labels = data[target]
def train_lgbm(train, train_labels):

    

    from imblearn.over_sampling import SMOTE, ADASYN

    x_SMOTE, y_SMOTE = SMOTE().fit_sample(x_train, y_train)

    

    # Dados de treino para lightGBM

    train_data = lgb.Dataset(x_SMOTE, label = y_SMOTE)

#     train_data = lgb.Dataset(train, label = train_labels)



    # Selecionando os Hyperparameters

    params = {'boosting_type': 'gbdt',

            'max_depth' : -1,

            'objective': 'binary',

            'nthread': 5,

            'num_leaves': 64,

            'learning_rate': 0.07,

            'max_bin': 512,

            'subsample_for_bin': 200,

            'subsample': 1,

            'subsample_freq': 1,

            'colsample_bytree': 0.8,

            'reg_alpha': 1.2,

            'reg_lambda': 1.2,

            'min_split_gain': 0.5,

            'min_child_weight': 1,

            'min_child_samples': 5,

            'scale_pos_weight': 1,

            'num_class' : 1,

            'metric' : 'binary_error'

            }



    # criando os parametros para busca

    gridParams = {'max_depth' : [-1],

                'learning_rate': [0.09],

                'n_estimators': [100],

                'num_leaves': [100],

                'boosting_type' : ['gbdt'],

                'objective' : ['binary'],

                'random_state' : [0], 

                'colsample_bytree' : [0.63],

                'subsample' : [0.7],

                #'class_weight':[{0: 1.0, 1: 2.0},{0: 1.0, 1: 20.0},{0: 1.0, 1: 200.0},{0: 1.0, 1: 2000.0}]

                #'reg_alpha' : [1, 1.2],

                #'reg_lambda' : [ 1.2, 1.4],

                }



    # Criando o classificador

    mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',

                            objective = 'binary',

                            num_class = params['num_class'],

                            n_jobs = -2, 

                            silent = True,

                            max_depth = params['max_depth'],

                            max_bin = params['max_bin'],

                            subsample_for_bin = params['subsample_for_bin'],

                            subsample = params['subsample'],

                            subsample_freq = params['subsample_freq'],

                            min_split_gain = params['min_split_gain'],

                            min_child_weight = params['min_child_weight'],

                            min_child_samples = params['min_child_samples'],

                            scale_pos_weight = params['scale_pos_weight']

                            )



    # View the default model params:

    mdl.get_params().keys()

    

    # Create the grid

    grid = GridSearchCV(mdl, gridParams, verbose=2, cv=3, n_jobs=-2)



    # Run the grid

    grid.fit(train, train_labels)



    # Print the best parameters found

    print(grid.best_params_)

    print(grid.best_score_)



    # Using parameters already set above, replace in the best from the grid search

    params['colsample_bytree'] = grid.best_params_['colsample_bytree']

    params['learning_rate'] = grid.best_params_['learning_rate']

    params['num_leaves'] = grid.best_params_['num_leaves']

    params['subsample'] = grid.best_params_['subsample']

    params['n_estimators'] = grid.best_params_['n_estimators']

    # params['max_bin'] = grid.best_params_['max_bin']

    #params['reg_alpha'] = grid.best_params_['reg_alpha']

    #params['reg_lambda'] = grid.best_params_['reg_lambda']

    #params['class_weight'] = grid.best_params_['class_weight']

    # params['subsample_for_bin'] = grid.best_params_['subsample_for_bin']



    print('Fitting with params: ')

    print(params)



    #Train model on selected parameters and number of iterations

    lgbm = lgb.train(params,

                    train_data,

                    280,

                    #early_stopping_rounds= 40,

                    verbose_eval= 4

                    )

    

    return lgbm, lgb
# splitting the data for modeling

x_train, x_test, y_train, y_test = train_test_split(train, train_labels, test_size=0.20, random_state=42, shuffle = True)
# traning on train data

lgbm, lgb= train_lgbm(x_train, y_train)
def predict_lgbm(model, lightgbm, test, test_labels = None, validation = True):

   

    if validation == False:

        #Predict on test set

        predictions_lgbm_prob = model.predict(test)

        predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output



    else:

        predictions_lgbm_prob = model.predict(test)

        predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output



        #Print accuracy

        acc_lgbm = accuracy_score(test_labels,predictions_lgbm_01)

        print('Overall accuracy of Light GBM model:', acc_lgbm)

        

        #Plot Variable Importances

        lightgbm.plot_importance(model, max_num_features=21, importance_type='split')

        

        #Classification report

        print(classification_report(test_labels, predictions_lgbm_01))



        #Print Confusion Matrix

        plt.figure()

        cm = confusion_matrix(test_labels, predictions_lgbm_01)

        labels = ['0', '1']

        plt.figure(figsize=(7,4))

        sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2);

        plt.title('Confusion Matrix')

        plt.ylabel('True Class')

        plt.xlabel('Predicted Class')

        plt.show()

        

        #Print Area Under Curve

        plt.figure()

        plt.figure(figsize=(7,4))

        false_positive_rate, recall, thresholds = roc_curve(test_labels, predictions_lgbm_prob)

        roc_auc = auc(false_positive_rate, recall)

        plt.title('Receiver Operating Characteristic (ROC)')

        plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)

        plt.legend(loc='lower right')

        plt.plot([0,1], [0,1], 'r--')

        plt.xlim([0.0,1.0])

        plt.ylim([0.0,1.0])

        plt.ylabel('Recall')

        plt.xlabel('Fall-out (1-Specificity)')

        plt.show()

        print('AUC score:', roc_auc)

        

    return predictions_lgbm_prob, predictions_lgbm_01
# predictions on test data

predictions_lgbm_prob, predictions_lgbm_01 =  predict_lgbm(lgbm, lgb, x_test, y_test, validation = True)
# Create object that can calculate shap values

explainer = shap.TreeExplainer(lgbm)
%time shap_values = explainer.shap_values(x_train)
# summarize the effects of all the features

shap.summary_plot(shap_values, x_train)
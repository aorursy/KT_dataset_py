

#   .o88b.  .d88b.  db    db d888888b d8888b.         db .d888b. 

#  d8P  Y8 .8P  Y8. 88    88   `88'   88  `8D        o88 88' `8D 

#  8P      88    88 Y8    8P    88    88   88         88 `V8o88' 

#  8b      88    88 `8b  d8'    88    88   88 C8888D  88    d8'  

#  Y8b  d8 `8b  d8'  `8bd8'    .88.   88  .8D         88   d8'   

#   `Y88P'  `Y88P'     YP    Y888888P Y8888D'         VP  d8'    

#                                                                

# This machine learning pipeline is a fork 

#   from a work in progress (to be published in 2021):

#   BRHIM - Base de Registros Hospitalares 

#   para Informações e Metadados - By: Vaz, Dora, Lamb e Camey

#***************************************************************

# The COVID-19 Supicius Cases Classifier is an analysis done 

#   with the help of a multidisciplinary team  

#   from Hospital de Clínicas de Porto Alegre (HCPA), 

#   that holds a permanent data science study group 

#

#   Contact to authors   

#

#           mhbarbian@gmail.com , 

#           scamey@hcpa.edu.br, 

#           tvaz@hcpa.edu.br , 

#           vhirakata@hcpa.edu.br 

#

#

#

#  DATA CULTURE                // ______________________________________________

#     SCIENCE                 // _____________________________________________

#       MULTIDISCIPLINARITY  // ____________________________________________

#

#=

import pandas as pd

import numpy as np 

import seaborn as sns 



#from sklearn.model_selection import 

#from sklearn import metrics



from sklearn import datasets

from sklearn import preprocessing

#from sklearn.preprocessing import LabelEncoder

#from sklearn.preprocessing import OneHotEncoder

#from sklearn.preprocessing import label_binarize



from sklearn import model_selection



from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold

from sklearn.feature_selection import RFE, VarianceThreshold

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso,  LogisticRegression, LinearRegression

from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn import metrics

from sklearn.metrics import (accuracy_score, brier_score_loss, precision_score, recall_score,f1_score, classification_report, confusion_matrix, roc_auc_score, log_loss)

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer



import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

#from matplotlib.ticker import StrMethodFormatter





pd.options.display.max_columns = 50
df = pd.read_csv('/kaggle/input/covideinsteinn598-prep/dataset_n598.xlsx - BD_COVID_N598.tsv', 

                  encoding='utf8' ,

                  index_col=0, 

                  sep='\t',  lineterminator='\r', decimal=",", 

                  keep_default_na=False,

                  na_values=['-1.#IND', '1.#QNAN', '1.#IND','-1.#QNAN', '#N/A','N/A', '#NA', 'NA','#NULL!', 'NaN', '-NaN', 'nan', '-nan'])

df.reset_index(inplace=True)

df.rename(columns={'SARSCov2examresult':'outcome'}, inplace=True)

df.rename(columns={'PatientID':'id'}, inplace=True)

df.rename(columns={'Patientaddmitedtoregularward1yes0no':'regular'}, inplace=True)

df.rename(columns={'Patientaddmitedtosemiintensiveunit1yes0no':'semi'}, inplace=True)

df.rename(columns={'Patientaddmitedtointensivecareunit1yes0no':'icu'}, inplace=True)

df.columns = df.columns.str.replace(" ", "_")

df.rename(columns={'Patientagequantile':'age'}, inplace=True)

target_combo = ['outcome','regular','semi', 'icu']

print('Dataset shape:', df.shape)

#print('Dataset shape:', df.dtypes)





df
df.describe()
#declare dataframe for features

X = df.copy()



#declare dataframe for targets

#task 1

y= X['outcome']

#task 2 : not implemented

y_r= X['regular']

y_s= X['semi']

y_i= X['icu']



print('Quali work')

quali = ['object']

quali_columns = list(X.select_dtypes(include=quali).columns)

quali_columns.remove('id')

print("Sample and feature selection done during descriptive analysis.")

for col in quali_columns:

    X[col] = X[col].astype('float64')

    #X[col] = X[col].cat.add_categories('Unknown')

    #X[col].fillna('Unknown', inplace =True)

X.drop(labels='id', axis=1, inplace=True)





print('Quanti work')

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'uint8']

X = X.round(decimals=12)

print('Cleanning ...empty and duplicates')

X = X.drop_duplicates()

X.dropna(axis=1, how='all', thresh=None, subset=None, inplace=True)

print('Cleanning ...constants')

constant_filter = VarianceThreshold(threshold=0)

constant_filter.fit(X.select_dtypes(include=numerics))

len(X.select_dtypes(include=numerics).columns[constant_filter.get_support()])

constant_columns = [column for column in X.select_dtypes(include=numerics).columns

                    if column not in X.select_dtypes(include=numerics).columns[constant_filter.get_support()]]

X.drop(labels=constant_columns, axis=1, inplace=True)

print(constant_columns)

print('Cleanning ...quasi-constants')

qconstant_filter = VarianceThreshold(threshold=0.0001)

qconstant_filter.fit(X.select_dtypes(include=numerics))

len(X.select_dtypes(include=numerics).columns[qconstant_filter.get_support()])

qconstant_columns = [column for column in X.select_dtypes(include=numerics).columns

                    if column not in X.select_dtypes(include=numerics).columns[qconstant_filter.get_support()]]

X.drop(labels=qconstant_columns, axis=1, inplace=True)

print(qconstant_columns)



print('Cleanning ... correlated :off')

correlated_features = set()

correlation_matrix = X.select_dtypes(include=numerics).corr()

for i in range(len(correlation_matrix.columns)):

    for j in range(i):

        if abs(correlation_matrix.iloc[i, j]) > 0.99999:

            colname = correlation_matrix.columns[i]

            correlated_features.add(colname)

#X.drop(labels=correlated_features, axis=1, inplace=True)

print(correlated_features)



sns.heatmap(X.isnull(), cbar=False)

print('Count missings')

print(X.select_dtypes(include=numerics).isnull().sum().sum())

print('Plot missings ... heatmap: marron are missing values')

#prepare targets for task 1

y= X['outcome']

del X['outcome']



#task 2 (maybe)

#y_r= df_prepared['regular'].astype('category')

#y_s= df_prepared['semi'].astype('category')

#y_i= df_prepared['icu'].astype('category')



#prepare datasate with meaningfull features



experiment_selection = ['age', 'Hematocrit',

       'Hemoglobin', 'Platelets', 'Meanplateletvolume', 'RedbloodCells',

       'Lymphocytes', 'MeancorpuscularhemoglobinconcentrationMCHC',

       'Leukocytes', 'Basophils', 'MeancorpuscularhemoglobinMCH',

       'Eosinophils', 'MeancorpuscularvolumeMCV', 'Monocytes',

       'RedbloodcelldistributionwidthRDW']



experiment_imputation = ['age', 'Hematocrit',

       'Hemoglobin', 'Platelets', 'Meanplateletvolume', 'RedbloodCells',

       'Lymphocytes', 'MeancorpuscularhemoglobinconcentrationMCHC',

       'Leukocytes', 'Basophils', 'MeancorpuscularhemoglobinMCH',

       'Eosinophils', 'MeancorpuscularvolumeMCV', 'Monocytes',

       'RedbloodcelldistributionwidthRDW', 'Neutrophils', 'Urea',

       'ProteinaCreativamgdL', 'Creatinine', 'Potassium', 'Sodium']



experiment_complete = ['age', 'regular', 'semi', 'icu', 'Hematocrit',

       'Hemoglobin', 'Platelets', 'Meanplateletvolume', 'RedbloodCells',

       'Lymphocytes', 'MCHC',

       'Leukocytes', 'Basophils', 'MeancorpuscularhemoglobinMCH',

       'Eosinophils', 'MeancorpuscularvolumeMCV', 'Monocytes',

       'RedbloodcelldistributionwidthRDW', 'Neutrophils', 'Urea',

       'ProteinaCreativamgdL', 'Creatinine', 'Potassium', 'Sodium',

       'RespiratorySyncytialVirus_n', 'InfluenzaA_n', 'InfluenzaB_n',

       'Parainfluenza1_n', 'CoronavirusNL63_n', 'RhinovirusEnterovirus_n',

       'CoronavirusHKU1_n', 'Parainfluenza3_n', 'Chlamydophilapneumoniae_n',

       'Adenovirus_n', 'Parainfluenza4_n', 'Coronavirus229E_n',

       'CoronavirusOC43_n', 'InfAH1N12009_n', 'Bordetellapertussis_n',

       'Metapneumovirus_n', 'Parainfluenza2_n', 'InfluenzaBrapidtest_n',

       'InfluenzaArapidtest_n', 'StreptoA_n']



X_experiment = X[experiment_imputation]
X_train,X_test,y_train,y_test = train_test_split(X_experiment, y , test_size=0.25,random_state=0, stratify=y)

print('  X_train  x_test: ', len(X_train), len(X_test))

print('  y_train  y_test: ', len(y_train), len(y_test))



#print('Somethin missing?')

#print(X_train.select_dtypes(include=numerics).isnull().sum().sum())



print('impute mean:')

X_experiment =  X_experiment.select_dtypes(include=numerics).apply(lambda x: x.fillna(x.mean())) 

X_train =    X_train.select_dtypes(include=numerics).apply(lambda x: x.fillna(x.mean())) 

X_test =    X_test.select_dtypes(include=numerics).apply(lambda x: x.fillna(x.mean())) 



print('Count missings: ')

print(X_experiment.select_dtypes(include=numerics).isnull().sum().sum())



sns.heatmap(X_experiment.isnull(), cbar=False)
# calculate the correlation matrix

corr = X_experiment.corr()



# plot the heatmap

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
#Classifier Showdown



#time and details matters

import time

start_time = time.time()



#select classifier candidates

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('QDA', QuadraticDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('ABC', AdaBoostClassifier()))

models.append(('RF', RandomForestClassifier()))

models.append(('RFb', RandomForestClassifier(class_weight='balanced')))

models.append(('GBC', GradientBoostingClassifier()))

#models.append(('NVC', NuSVC(probability=True)))

#models.append(('NB', GaussianNB()))

#models.append(('SVM', SVC()))



splits = 3



print("AUC-ROC")

# evaluate each model in turn

results = []

names = []

scoring = 'roc_auc'

for name, model in models:

    kfold = model_selection.StratifiedKFold(n_splits=splits)    

    #cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    cv_results = model_selection.cross_val_score(model, X_experiment, y,  cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)



# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison (AUC-ROC)')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()





# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

print("ACCURACY")

for name, model in models:

    kfold = model_selection.StratifiedKFold(n_splits=splits)    

    #cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    cv_results = model_selection.cross_val_score(model, X_experiment, y,  cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)



fig = plt.figure()

fig.suptitle('Algorithm Comparison (ACCURACY)')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()


results = []

names = []

scoring = 'f1_weighted'

print("F1-SCORE")



for name, model in models:

    kfold = model_selection.StratifiedKFold(n_splits=splits)    

    #cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    cv_results = model_selection.cross_val_score(model, X_experiment, y,  cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)



fig = plt.figure()

fig.suptitle('Algorithm Comparison (F1-SCORE)')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()


results = []

names = []

scoring = 'recall_weighted'

print("SENSIBILITY")



for name, model in models:

    kfold = model_selection.StratifiedKFold(n_splits=splits)    

    #cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    cv_results = model_selection.cross_val_score(model, X_experiment, y,  cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)



fig = plt.figure()

fig.suptitle('Algorithm Comparison (SENSIBILITY)')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()


results = []

names = []

scoring = 'precision_weighted'

print("PRECISION")



for name, model in models:

    kfold = model_selection.StratifiedKFold(n_splits=splits)    

    #cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    cv_results = model_selection.cross_val_score(model, X_experiment, y,  cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)



# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison (PRECISION)')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
#time and details matters

import time

start_time = time.time()



for name, model in models:

    if name =='GBC':  clf_champ = model



#FIT with TRAIN DATA 

clf_champ.fit(X_train,y_train)





#VALIDATION OF SCORES WITH ALL DATA (AS WE DONT HAVE A VALIDATION SET)

y_validation = y

y_validation_prob=clf_champ.predict_proba(X_experiment)

y_validation_pred=clf_champ.predict(X_experiment)



print("Validation")

print('Accuracy: ', metrics.accuracy_score(y_validation, y_validation_pred))

print('F1-score: ',f1_score(y_validation, y_validation_pred, pos_label=1, average='weighted'))

print('Recall: ',recall_score(y_validation, y_validation_pred, pos_label=1, average='weighted'))

print('Precision: ',precision_score(y_validation, y_validation_pred, average='weighted'))

print('AUC-ROC: ',roc_auc_score(y_validation, y_validation_prob[:,1]))



fpr, tpr, threshold = metrics.roc_curve(y_validation, y_validation_prob[:,1],pos_label=1)

roc_auc = metrics.auc(fpr, tpr)



# method I: plt

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()



#OPTIMIZA TRESHOLD

#sensibility = tpr = 1-fnr

#specificity= tnr = 1-fpr



best_threshold = threshold[np.argmax(tpr + (1-fpr) -1)]  



y_validation_pred_treshold = y_validation_prob[:,1] >= best_threshold



print("Best threshold ROC-AUC: %.2f%%" % (round(best_threshold, 3)))     

#print(y_validation_pred_treshold)



confusion_matrix_validation = pd.crosstab(y_validation, y_validation_pred, rownames=['Actual'], colnames=['Predicted'])

confusion_matrix_treshold= pd.crosstab(y_validation, y_validation_pred_treshold, rownames=['Actual'], colnames=['Predicted'])





#plot results

plt.rc("font", size=11)

plt.rcParams['font.family'] = "cursive"

sns.set_style("darkgrid")

sns.set(color_codes=True)

sns.set(style="whitegrid", font_scale=0.8)

sns.set_context("paper")





plt.title('Confusion Matrix')

plt.xlabel('')

g1 = sns.heatmap(confusion_matrix_validation, annot=True,cmap='Blues', fmt='d')

g1.set_ylabel('Total Nbr. of Patients')





confusion_matrix_treshold

plt.show()





plt.title('Confusion Matrix (Youden index )')

plt.xlabel('')

g1 = sns.heatmap(confusion_matrix_treshold, annot=True,cmap='Blues', fmt='d')

g1.set_ylabel('Total Nbr. of Patients')

plt.show()





print('todo: random search and grid optimization')

print("--- %s seconds ---" % (time.time() - start_time))
#feature_importances = pd.DataFrame(clf.feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance',ascending=False)

feat_importances = pd.Series(clf_champ.fit(X_train,y_train).feature_importances_, index=X_train.columns)

feat_importances.nlargest(21).plot(kind='barh')

#print(feature_importances)
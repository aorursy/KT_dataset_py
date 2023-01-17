import pandas as pd

import numpy as np
df = pd.read_excel('../input/covid19/dataset.xlsx', encoding='utf8')
df.head()
print("This dataset has {:.1f}".format(100*df.isna().to_numpy().sum()/(df.shape[0]*df.shape[1])) + "% missing values")
#Encode the addmission columns into one

df2 = df.copy()



def unit_room(x):

    if x['Patient addmited to regular ward (1=yes, 0=no)']>0:

        return 'regular ward'

    elif x['Patient addmited to semi-intensive unit (1=yes, 0=no)']>0:

        return 'semi-intensive'

    elif x['Patient addmited to intensive care unit (1=yes, 0=no)']>0:

        return 'icu'

    else:

        return 'elective'



df2['unit'] = df2.apply(unit_room, axis=1)
df2[['Patient addmited to regular ward (1=yes, 0=no)', 

     'Patient addmited to semi-intensive unit (1=yes, 0=no)',

     'Patient addmited to intensive care unit (1=yes, 0=no)',

     'unit']].head(10)
#Conversion from categorical to numerical

mask = {'positive': 1, 

        'negative': 0,

        'detected': 1, 

        'not_detected': 0,

        'not_done': np.NaN,

        'Não Realizado': np.NaN,

        'absent': 0, 

        'present': 1,

        'detected': 1, 

        'not_detected': 0,

        'normal': 1,

        'light_yellow': 1, 

        'yellow': 2, 

        'citrus_yellow': 3, 

        'orange': 4,

        'clear': 1, 

        'lightly_cloudy': 2, 

        'cloudy': 3, 

        'altered_coloring': 4,

        '<1000': 1000,

        'Ausentes': 0, 

        'Urato Amorfo --+': 1, 

        'Oxalato de Cálcio +++': 1,

        'Oxalato de Cálcio -++': 1, 

        'Urato Amorfo +++': 1}



df3 = df2.copy()

df3 = df2.replace(mask)



df3['Urine - pH'] = df3['Urine - pH'].astype('float')

df3['Urine - Leukocytes'] = df3['Urine - Leukocytes'].astype('float')
#Removal of exams not performed on positive cases

exams_cols = df3.columns.to_list()[6:-1]



is_null_columns_covid = df3[df3['SARS-Cov-2 exam result']==1][exams_cols].apply(lambda col: col.isnull().all(), axis=0)

all_null_columns_covid = is_null_columns_covid[is_null_columns_covid==True]
print("Exams not performed on positive cases, so they are uninformative:\n")

print(all_null_columns_covid.index)
df4 = df3[set(df3.columns)-set(all_null_columns_covid.index)].copy()

exams_cols = set(exams_cols) - set(all_null_columns_covid.index)
exams_performed = (-df4[exams_cols].isnull()).astype('uint8')
exams_performed.head()
import seaborn as sns

import matplotlib.pyplot as plt
exams_cols_list = list(exams_cols)

freqExams = df4.shape[0] - df4[exams_cols_list].isnull().sum()



covY = (df4[df4['SARS-Cov-2 exam result']==1].shape[0]

        -df4[df4['SARS-Cov-2 exam result']==1][exams_cols_list].isnull().sum())/(df4.shape[0] - df4[exams_cols_list].isnull().sum())

covN = (df4[df4['SARS-Cov-2 exam result']==0].shape[0]

        -df4[df4['SARS-Cov-2 exam result']==0][exams_cols_list].isnull().sum())/(df4.shape[0] - df4[exams_cols_list].isnull().sum())



examsICU = (df4[df4['unit']=='icu'].shape[0]

        -df4[df4['unit']=='icu'][exams_cols_list].isnull().sum())/(df4.shape[0] - df4[exams_cols_list].isnull().sum())

examsSemiIntesive = (df4[df4['unit']=='semi-intensive'].shape[0]

        -df4[df4['unit']=='semi-intensive'][exams_cols_list].isnull().sum())/(df4.shape[0] - df4[exams_cols_list].isnull().sum())

examsRegular = (df4[df4['unit']=='regular ward'].shape[0]

        -df4[df4['unit']=='regular ward'][exams_cols_list].isnull().sum())/(df4.shape[0] - df4[exams_cols_list].isnull().sum())

examsElective = (df4[df4['unit']=='elective'].shape[0]

        -df4[df4['unit']=='elective'][exams_cols_list].isnull().sum())/(df4.shape[0] - df4[exams_cols_list].isnull().sum())



fig, axs = plt.subplots(3, 1, figsize=(25,8))

fig.patch.set_facecolor('white')



#Frequency plot

pFreq = axs[0].bar(exams_cols_list, freqExams, color='orange', )  

axs[0].set_title("(a) Frequency of tests performed")

axs[0].get_xaxis().set_ticks([])



#Admission units over total tests performed

pElective = axs[1].bar(exams_cols_list, examsElective, color='dodgerblue')

pRegular = axs[1].bar(exams_cols_list, examsRegular, bottom=examsElective,  color='lightcoral')

pSemiIntesive = axs[1].bar(exams_cols_list, examsSemiIntesive, bottom=(examsElective+examsRegular), color='orangered')

pICU = axs[1].bar(exams_cols_list, examsICU, bottom=(examsSemiIntesive+examsElective+examsRegular), color='darkred')

axs[1].set_title("(b) Patient admission unit representativeness by test performed")

axs[1].legend(["Elective*", "Regular ward", "Semi-Intensive", "ICU"], loc="lower right")

axs[1].get_xaxis().set_ticks([])



#Percentage of COVID cases over total tests performed

pCovY = axs[2].bar(exams_cols_list, covY, color='red')

pCovN = axs[2].bar(exams_cols_list, covN, bottom=covY, color='grey')

plt.xticks(exams_cols_list, exams_cols_list, rotation='vertical')

axs[2].set_title("(c) Percentage of COVID cases over total tests performed")

axs[2].legend(["COVID POSITIVE", "COVID NEGATIVE"])



plt.xticks(exams_cols_list, exams_cols_list, rotation='vertical')



plt.subplots_adjust(hspace=0.2) 

fig.suptitle("Figure 1 - Analysis of variables with respect to frequency, patient admission unit and COVID status")

plt.plot()
plt.figure(figsize=(15,10))

sns.heatmap(exams_performed, cbar=False)

sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

plt.title("Figure 2 - Presence (white) or absense (black) of exams")
from sklearn.decomposition import PCA
pca_model = PCA(random_state=42).fit(exams_performed)
%matplotlib inline

variances_explained = pca_model.explained_variance_ratio_.cumsum()

fig = plt.figure(figsize=(15,5))

fig.patch.set_facecolor('white')

plt.step(list(range(1,len(variances_explained)+1)),variances_explained)

plt.title("Figure 3 - Cummulative explained variance of Principal Components")
variances_explained
pca_model = PCA(n_components=6, random_state=42).fit(exams_performed)
fig, axs = plt.subplots(2, 3, figsize=(20,10))

fig.patch.set_facecolor((1,1,1))

plt.subplots_adjust(wspace=0.5) 

pc=-1

for i in list(range(2)): #plot lines

  for j in list(range(3)): #plot cols

    pc+=1

    component_feature_explanation = pd.Series(pca_model.components_[pc], index=list(exams_cols)).sort_values()

    axis_truncated = [txt[:20] for txt in component_feature_explanation[-20:].index]

    axs[i, j].barh(axis_truncated, component_feature_explanation[-20:])

    axs[i, j].set_title('PC '+str(pc+1))

fig.suptitle("Figure 4 - Principal Components (PC) explained")
pcs_vars = {'respiratory': ['Influenza B', 'Respiratory Syncytial Virus', 'Influenza A',

                            'Metapneumovirus', 'Parainfluenza 1', 'Inf A H1N1 2009',

                            'Bordetella pertussis', 'Chlamydophila pneumoniae', 'Coronavirus229E',

                            'Parainfluenza 2', 'Parainfluenza 3', 'CoronavirusNL63',

                            'Rhinovirus/Enterovirus', 'CoronavirusOC43', 'Coronavirus HKU1',

                            'Adenovirus', 'Parainfluenza 4'],

            'regular_blood': ['Proteina C reativa mg/dL',

                              'Neutrophils', 'Mean platelet volume ', 'Monocytes',

                              'Red blood cell distribution width (RDW)', 'Red blood Cells',

                              'Platelets', 'Eosinophils', 'Basophils', 'Leukocytes',

                              'Mean corpuscular hemoglobin (MCH)', 'Mean corpuscular volume (MCV)',

                              'Mean corpuscular hemoglobin concentration\xa0(MCHC)', 'Lymphocytes',

                              'Hemoglobin', 'Hematocrit'],

            'liver_kidney_gas': ['Creatine phosphokinase\xa0(CPK)\xa0', 

                                 'International normalized ratio (INR)', 

                                 'Alkaline phosphatase', 'Gamma-glutamyltransferase\xa0',

                                 'Alanine transaminase', 'Aspartate transaminase',

                                 'HCO3 (venous blood gas analysis)',

                                 'Hb saturation (venous blood gas analysis)',

                                 'Total CO2 (venous blood gas analysis)',

                                 'pCO2 (venous blood gas analysis)', 'pH (venous blood gas analysis)',

                                 'pO2 (venous blood gas analysis)',

                                 'Base excess (venous blood gas analysis)', 'Total Bilirubin',

                                 'Direct Bilirubin', 'Indirect Bilirubin',

                                 'Sodium', 'Potassium', 'Urea', 'Creatinine'],

            'urine': ['Urine - Ketone Bodies', 'Urine - Esterase', 'Urine - Protein',

                      'Urine - Hyaline cylinders', 'Urine - Urobilinogen',

                      'Urine - Bile pigments', 'Urine - Hemoglobin', 'Urine - pH',

                      'Urine - Granular cylinders', 'Urine - Aspect', 'Urine - Density',

                      'Urine - Color', 'Urine - Red blood cells', 'Urine - Leukocytes',

                      'Urine - Yeasts', 'Urine - Crystals'],

            'bone_narrow_cells': ['Metamyelocytes', 'Myelocytes', 'Promyelocytes', 'Rods #',

                                  'Myeloblasts', 'Segmented'],

            'influenza_rapid': ['Influenza B, rapid test', 'Influenza A, rapid test']}



vars_analyzed = [var for pc in pcs_vars for var in pcs_vars[pc]]

#len(vars_analyzed) == len(set(vars_analyzed)) #sanity check must return true

#len(vars_analyzed) #Output: 77
df5 = df4.copy()

#Here we create columns to flag the group of tests performed

for pc in pcs_vars.keys():

  df5[pc+"_tests"] = df5.apply(lambda x: 0 if x[pcs_vars[pc]].isnull().all() else 1,axis=1)
exams_not_included = list(set(exams_performed.columns) - set(vars_analyzed))



freqExams = df5.shape[0] - df5[exams_not_included].isnull().sum()

covY = (df5[df5['SARS-Cov-2 exam result']==1].shape[0]

        -df5[df5['SARS-Cov-2 exam result']==1][exams_not_included].isnull().sum())/(df5.shape[0] - df5[exams_not_included].isnull().sum())

covN = (df5[df5['SARS-Cov-2 exam result']==0].shape[0]

        -df5[df5['SARS-Cov-2 exam result']==0][exams_not_included].isnull().sum())/(df5.shape[0] - df5[exams_not_included].isnull().sum())





fig, axs = plt.subplots(2, 1, figsize=(15,5))





pFreq = axs[0].bar(exams_not_included, freqExams, color='orange', )  

pCovY = axs[1].bar(exams_not_included, covY, color='red')

pCovN = axs[1].bar(exams_not_included, covN, bottom=covY, color='grey')

axs[0].get_xaxis().set_ticks([])

plt.xticks(exams_not_included, exams_not_included, rotation='vertical')

axs[0].set_title("Frequency of tests performed")

axs[1].set_title("Percentage of COVID cases over total tests perform")

plt.legend(["COVID POSITIVE", "COVID NEGATIVE"])

plt.subplots_adjust(hspace=0.2) 



fig.suptitle("Figure 5 - Analysis of variables not included in the 5 proposed groups")

plt.plot()
inv_pcs_vars = {}

for pc in pcs_vars:

    for var in pcs_vars[pc]:

        inv_pcs_vars[var]=pc



cols_sorted = pd.DataFrame(data = exams_performed.sum(), columns=["count"]).merge(

    pd.DataFrame(data = pd.Series(inv_pcs_vars), columns=["pc"]), how="left", left_index=True, 

    right_index=True).sort_values(by=["pc","count"],ascending=False)



rows_order_by = cols_sorted.reset_index().groupby("pc").first().sort_values(by=["count"],ascending=False)["index"].to_list()

#rows_order_by = ['Respiratory Syncytial Virus', 'Hematocrit', 'Influenza B, rapid test','Creatinine', 'Segmented', 'Urine - Red blood cells']



exams_performed_sorted = exams_performed[cols_sorted.index].sort_values(by=rows_order_by, ascending=False)



plt.figure(figsize=(15,10))

sns.heatmap(exams_performed_sorted, cbar=False)

sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

plt.title("Figure 6 - Figure 2 reordered by group of exams")
def cooc_matrix(dataset):

  cooc =  dataset.T.dot(dataset)/dataset.shape[0]

  cooc_format = cooc.style.format("{:.1%}")

  return cooc_format



tests_cols = ['respiratory_tests', 'regular_blood_tests', 

              'liver_kidney_gas_tests', 'urine_tests', 'bone_narrow_cells_tests',

              'influenza_rapid_tests']



cooc_tests_covid_neg = cooc_matrix(df5[df5['SARS-Cov-2 exam result']==0][tests_cols])

cooc_tests_covid_poz = cooc_matrix(df5[df5['SARS-Cov-2 exam result']==1][tests_cols])

cooc_tests_icu = cooc_matrix(df5[df5['unit']=='icu'][tests_cols])

cooc_tests_semi = cooc_matrix(df5[df5['unit']=='semi-intensive'][tests_cols])

cooc_tests_reg = cooc_matrix(df5[df5['unit']=='regular ward'][tests_cols])

cooc_tests_ele = cooc_matrix(df5[df5['unit']=='elective'][tests_cols])
cooc_tests_covid_neg
cooc_tests_covid_poz
cooc_tests_icu
cooc_tests_semi
cooc_tests_reg
cooc_tests_ele
#Data split and hyperparameter search

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



#Metrics

from sklearn.metrics import classification_report

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score



#Classifiers

from sklearn.dummy import DummyClassifier

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from tensorflow.python.keras.layers import Dense

from tensorflow.python.keras import Sequential

from keras.wrappers.scikit_learn import KerasClassifier
selected_vars = ['SARS-Cov-2 exam result', 'unit', 'Patient age quantile'] + pcs_vars['regular_blood']

df_red_blood = df5[selected_vars].dropna() #Shape: (420, 19)



X = df_red_blood[pcs_vars['regular_blood']+['Patient age quantile']]

y = df_red_blood['SARS-Cov-2 exam result']

strat = df_red_blood['SARS-Cov-2 exam result'].astype(str) + df_red_blood['unit'] #stratification



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=strat, shuffle=True, random_state=42)

#X_train.shape: (294, 17)

#X_test.shape: (126, 17)
blood_with_respiratory = df5[(df5['regular_blood_tests']==1)&(df5['SARS-Cov-2 exam result']==0)][pcs_vars['respiratory']].dropna()



plt.figure(figsize=(8,8))

sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

sns.heatmap(blood_with_respiratory.T.dot(blood_with_respiratory),annot=True, cbar=False)

plt.title("Figure 8 - Coocurrence of patients diagnosed with respiratory diseases, but SARS-CoV-2 Negative")
df_InfB = df5[(df5['Influenza B']==1) & (df5['regular_blood_tests']==1) & (df5['SARS-Cov-2 exam result']==0)][pcs_vars['regular_blood']+['Patient age quantile', 'SARS-Cov-2 exam result']].dropna()

X_InfB = df_InfB[pcs_vars['regular_blood']+['Patient age quantile']]

y_InfB = df_InfB['SARS-Cov-2 exam result']



df_H1N1 = df5[(df5['Inf A H1N1 2009']==1) & (df5['regular_blood_tests']==1) & (df5['SARS-Cov-2 exam result']==0)][pcs_vars['regular_blood']+['Patient age quantile', 'SARS-Cov-2 exam result']].dropna()

X_H1N1 = df_H1N1[pcs_vars['regular_blood']+['Patient age quantile']]

y_H1N1 = df_H1N1['SARS-Cov-2 exam result']



df_Rhino = df5[(df5['Rhinovirus/Enterovirus']==1) & (df5['regular_blood_tests']==1) & (df5['SARS-Cov-2 exam result']==0)][pcs_vars['regular_blood']+['Patient age quantile', 'SARS-Cov-2 exam result']].dropna()

X_Rhino = df_Rhino[pcs_vars['regular_blood']+['Patient age quantile']]

y_Rhino = df_Rhino['SARS-Cov-2 exam result']
dummy_most_freq_clf = DummyClassifier(strategy="most_frequent")

dummy_most_freq_clf.fit(X_train, y_train)

y_dummy_most_freq_pred = dummy_most_freq_clf.predict(X_test)



print("Classification report for Dummy 'Most Frequent' Classifier")

print(classification_report(y_test, y_dummy_most_freq_pred))
dummy_strat_clf = DummyClassifier(strategy="stratified")

dummy_strat_clf.fit(X_train, y_train)

y_dummy_strat_pred = dummy_strat_clf.predict(X_test)



print("Classification report for Dummy Stratified Classifier")

print(classification_report(y_test, y_dummy_strat_pred))
parameters = {'kernel':('linear', 'rbf'), 

              'C':[0.1, 1, 10],

              'class_weight': [{1: 1}, {1: 2}, {1: 5}, {1: 10}, {1: 15}]

              }



clf = svm.SVC(random_state=42)

grid_search = GridSearchCV(clf, parameters, cv=5, scoring='f1_macro')

grid_search.fit(X_train, y_train)



print("Best estimator is:"+str(grid_search.best_params_))

print("F1-Score (macro avg) on train: "+"{0:.2%}".format(grid_search.best_score_))





y_pred = grid_search.best_estimator_.predict(X_test)



print("\nClassification report for SVM on test")

print(classification_report(y_test, y_pred))
parameters = {'n_estimators': [25,50, 100, 150, 200], 

              'max_depth': [3, 5, 10, 15,20,25],

              'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001]

              }

model = GradientBoostingClassifier(random_state=42)

grid_search = GridSearchCV(model, parameters, cv=5, scoring='f1_macro')

grid_search.fit(X_train, y_train)



print("Best estimator is:"+str(grid_search.best_params_))

print("F1-Score (macro avg) on train: "+"{0:.2%}".format(grid_search.best_score_))



y_pred = grid_search.best_estimator_.predict(X_test)



print("\nClassification report for Gradient Boosting Classifier on test")

print(classification_report(y_test, y_pred))
pd.DataFrame(grid_search.best_estimator_.feature_importances_,

             index = X_train.columns, columns=['importance']).sort_values('importance',ascending=False)
#Influenza B

y_pred = grid_search.best_estimator_.predict(X_InfB)

print("\n Accuracy of Gradient Boosting Classifier on Influenza B patients: {:.1f}".format(accuracy_score(y_InfB, y_pred)*100)+"%")
#H1N1

y_pred = grid_search.best_estimator_.predict(X_H1N1)

print("\n Accuracy of Gradient Boosting Classifier on H1N1 patients: {:.1f}".format(accuracy_score(y_H1N1, y_pred)*100)+"%")
#Rhinovirus 

y_pred = grid_search.best_estimator_.predict(X_Rhino)

print("\n Accuracy of Gradient Boosting Classifier on Rhinovirus patients: {:.1f}".format(accuracy_score(y_Rhino, y_pred)*100)+"%")
parameters = {'n_estimators': [50, 100, 200], 

              'max_depth': [3, 5, 10, 15],

              'max_features': [0.6, 0.8, 1.0],

              'class_weight': [{1: 1}, {1: 2}, {1: 5}, {1: 10}, {1: 15}]

              }



clf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(clf, parameters, cv=5, scoring='f1_macro')

grid_search.fit(X_train, y_train)



print("Best estimator is:"+str(grid_search.best_params_))

print("F1-Score (macro avg) on train: "+"{0:.2%}".format(grid_search.best_score_))



y_pred = grid_search.best_estimator_.predict(X_test)



print("\nClassification report for Random Forest on test")

print(classification_report(y_test, y_pred))
parameters = {'n_estimators': [5, 10, 20, 30], 

              'base_estimator': [DecisionTreeClassifier(max_depth=1),

                                 DecisionTreeClassifier(max_depth=3),

                                 DecisionTreeClassifier(max_depth=1, class_weight="balanced"),

                                 DecisionTreeClassifier(max_depth=3, class_weight="balanced")],

              'learning_rate': [0.01, 0.1, 1.0]

              }



clf = AdaBoostClassifier(random_state=42)

grid_search = GridSearchCV(clf, parameters, cv=5, scoring='f1_macro')

grid_search.fit(X_train, y_train)



print("Best estimator is:"+str(grid_search.best_params_))

print("F1-Score (macro avg) on train: "+"{0:.2%}".format(grid_search.best_score_))



y_pred = grid_search.best_estimator_.predict(X_test)



print("\nClassification report for AdaBoost on test")

print(classification_report(y_test, y_pred))
importances = grid_search.best_estimator_.feature_importances_

std = np.std([tree.feature_importances_ for tree in grid_search.best_estimator_.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Plot the feature importances of the forest

plt.figure(figsize=(10,5))

plt.title("Figure 9 - AdaBoost Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="r", align="center")

plt.xticks(range(X.shape[1]), X_train.columns, rotation='vertical')

plt.xlim([-1, X.shape[1]])

plt.show()



# Print the feature ranking

print("Feature ranking:")



for f in range(X.shape[1]):

    print("%d. %s (%f)" % (f + 1, X_train.columns[f], importances[indices[f]]))

#Influenza B

y_pred = grid_search.best_estimator_.predict(X_InfB)

print("\n Accuracy of Ada Boost Classifier on Influenza B patients: {:.1f}".format(accuracy_score(y_InfB, y_pred)*100)+"%")
#H1N1

y_pred = grid_search.best_estimator_.predict(X_H1N1)

print("\n Accuracy of Ada Boost Classifier on H1N1 patients: {:.1f}".format(accuracy_score(y_H1N1, y_pred)*100)+"%")
#Rhinovirus 

y_pred = grid_search.best_estimator_.predict(X_Rhino)

print("\n Accuracy of Ada Boost Classifier on Rhinovirus patients: {:.1f}".format(accuracy_score(y_Rhino, y_pred)*100)+"%")
selected_vars = ['SARS-Cov-2 exam result', 'unit', 'Patient age quantile'] + pcs_vars['respiratory']

df_respiratory = df5[selected_vars].dropna() #Shape: (420, 19)



X = df_respiratory[pcs_vars['respiratory']+['Patient age quantile']]

y = df_respiratory['SARS-Cov-2 exam result']

strat = df_respiratory['SARS-Cov-2 exam result'].astype(str) + df_respiratory['unit'] #stratification



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=strat, shuffle=True, random_state=42)

#X_train.shape: (946, 18)

#X_test.shape: (406, 18)
dummy_most_freq_clf = DummyClassifier(strategy="most_frequent")

dummy_most_freq_clf.fit(X_train, y_train)

y_dummy_most_freq_pred = dummy_most_freq_clf.predict(X_test)



print("Classification report for Dummy 'Most Frequent' Classifier")

print(classification_report(y_test, y_dummy_most_freq_pred))
dummy_strat_clf = DummyClassifier(strategy="stratified")

dummy_strat_clf.fit(X_train, y_train)

y_dummy_strat_pred = dummy_strat_clf.predict(X_test)



print("Classification report for Dummy Stratified Classifier")

print(classification_report(y_test, y_dummy_strat_pred))
parameters = {'kernel':('linear', 'rbf'), 

              'C':[0.1, 1, 10],

              'class_weight': [{1: 1}, {1: 2}, {1: 5}, {1: 10}, {1: 15}]

              }



clf = svm.SVC(random_state=42)

grid_search = GridSearchCV(clf, parameters, cv=5, scoring='f1_macro')

grid_search.fit(X_train, y_train)



print("Best estimator is:"+str(grid_search.best_params_))

print("F1-Score (macro avg) on train: "+"{0:.2%}".format(grid_search.best_score_))



y_pred = grid_search.best_estimator_.predict(X_test)



print("\nClassification report for SVM on test")

print(classification_report(y_test, y_pred))
parameters = {'n_estimators': [25,50, 100, 150, 200], 

              'max_depth': [3, 5, 10, 15,20,25],

              'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001]

              }

model = GradientBoostingClassifier()

grid_search = GridSearchCV(model, parameters, cv=5, scoring='f1_macro')

grid_search.fit(X_train, y_train)



print("Best estimator is:"+str(grid_search.best_params_))

print("F1-Score (macro avg) on train: "+"{0:.2%}".format(grid_search.best_score_))



y_pred = grid_search.best_estimator_.predict(X_test)



print("\nClassification report for Gradient Boosting Classifier on test")

print(classification_report(y_test, y_pred))
parameters = {'n_estimators': [50, 100, 200], 

              'max_depth': [3, 5, 10, 15],

              'max_features': [0.6, 0.8, 1.0],

              'class_weight': [{1: 1}, {1: 2}, {1: 5}, {1: 10}, {1: 15}]

              }



clf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(clf, parameters, cv=5, scoring='f1_macro')

grid_search.fit(X_train, y_train)



print("Best estimator is:"+str(grid_search.best_params_))

print("F1-Score (macro avg) on train: "+"{0:.2%}".format(grid_search.best_score_))



y_pred = grid_search.best_estimator_.predict(X_test)



print("\nClassification report for Random Forest on test")

print(classification_report(y_test, y_pred))
pd.DataFrame(grid_search.best_estimator_.feature_importances_,

             index = X_train.columns, columns=['importance']).sort_values('importance',ascending=False)
parameters = {'n_estimators': [5, 10, 20, 30], 

              'base_estimator': [DecisionTreeClassifier(max_depth=1),

                                 DecisionTreeClassifier(max_depth=3),

                                 DecisionTreeClassifier(max_depth=1, class_weight="balanced"),

                                 DecisionTreeClassifier(max_depth=3, class_weight="balanced")],

              'learning_rate': [0.01, 0.1, 1.0]

              }



clf = AdaBoostClassifier(random_state=42)

grid_search = GridSearchCV(clf, parameters, cv=5, scoring='f1_macro')

grid_search.fit(X_train, y_train)



print("Best estimator is:"+str(grid_search.best_params_))

print("F1-Score (macro avg) on train: "+"{0:.2%}".format(grid_search.best_score_))



y_pred = grid_search.best_estimator_.predict(X_test)



print("\nClassification report for Random Forest on test")

print(classification_report(y_test, y_pred))
selected_vars = ['SARS-Cov-2 exam result', 'unit', 'Patient age quantile'] + pcs_vars['influenza_rapid']

df_influenza = df5[selected_vars].dropna() #Shape: (820, 5)



X = df_influenza[pcs_vars['influenza_rapid']+['Patient age quantile']]

y = df_influenza['SARS-Cov-2 exam result']

strat = df_influenza['SARS-Cov-2 exam result'].astype(str) + df_influenza['unit'] #stratification



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=strat, shuffle=True, random_state=42)

#X_train.shape: (574, 3)

#X_test.shape: (246, 3)
dummy_most_freq_clf = DummyClassifier(strategy="most_frequent")

dummy_most_freq_clf.fit(X_train, y_train)

y_dummy_most_freq_pred = dummy_most_freq_clf.predict(X_test)



print("Classification report for Dummy 'Most Frequent' Classifier")

print(classification_report(y_test, y_dummy_most_freq_pred))
dummy_strat_clf = DummyClassifier(strategy="stratified")

dummy_strat_clf.fit(X_train, y_train)

y_dummy_strat_pred = dummy_strat_clf.predict(X_test)



print("Classification report for Dummy Stratified Classifier")

print(classification_report(y_test, y_dummy_strat_pred))
parameters = {'kernel':('linear', 'rbf'), 

              'C':[0.01, 0.1, 1, 10],

              'class_weight': [{1: 1}, {1: 2}, {1: 5}, {1: 10}, {1: 15}]

              }



clf = svm.SVC(random_state=42)

grid_search = GridSearchCV(clf, parameters, cv=5, scoring='f1_macro')

grid_search.fit(X_train, y_train)



print("Best estimator is:"+str(grid_search.best_params_))

print("F1-Score (macro avg) on train: "+"{0:.2%}".format(grid_search.best_score_))



y_pred = grid_search.best_estimator_.predict(X_test)



print("\nClassification report for SVM on test")

print(classification_report(y_test, y_pred))
parameters = {'n_estimators': [25,50, 100, 150, 200], 

              'max_depth': [3, 5, 10, 15,20,25],

              'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001]

              }

model = GradientBoostingClassifier()

grid_search = GridSearchCV(model, parameters, cv=5, scoring='f1_macro')

grid_search.fit(X_train, y_train)



print("Best estimator is:"+str(grid_search.best_params_))

print("F1-Score (macro avg) on train: "+"{0:.2%}".format(grid_search.best_score_))



y_pred = grid_search.best_estimator_.predict(X_test)



print("\nClassification report for Gradient Boosting Classifier on test")

print(classification_report(y_test, y_pred))
parameters = {'n_estimators': [50, 100, 200], 

              'max_depth': [3, 5, 10, 15],

              'max_features': [0.6, 0.8, 1.0],

              'class_weight': [{1: 1}, {1: 2}, {1: 5}, {1: 10}, {1: 15}]

              }



clf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(clf, parameters, cv=5, scoring='f1_macro')

grid_search.fit(X_train, y_train)



print("Best estimator is:"+str(grid_search.best_params_))

print("F1-Score (macro avg) on train: "+"{0:.2%}".format(grid_search.best_score_))



y_pred = grid_search.best_estimator_.predict(X_test)



print("\nClassification report for Random Forest on test")

print(classification_report(y_test, y_pred))
parameters = {'n_estimators': [5, 10, 20, 30], 

              'base_estimator': [DecisionTreeClassifier(max_depth=1),

                                 DecisionTreeClassifier(max_depth=3),

                                 DecisionTreeClassifier(max_depth=1, class_weight="balanced"),

                                 DecisionTreeClassifier(max_depth=3, class_weight="balanced")],

              'learning_rate': [0.01, 0.1, 1.0]

              }



clf = AdaBoostClassifier(random_state=42)

grid_search = GridSearchCV(clf, parameters, cv=5, scoring='f1_macro')

grid_search.fit(X_train, y_train)



print("Best estimator is:"+str(grid_search.best_params_))

print("F1-Score (macro avg) on train: "+"{0:.2%}".format(grid_search.best_score_))



y_pred = grid_search.best_estimator_.predict(X_test)



print("\nClassification report for Random Forest on test")

print(classification_report(y_test, y_pred))
from IPython.display import HTML



HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/gxAaO2rsdIs" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
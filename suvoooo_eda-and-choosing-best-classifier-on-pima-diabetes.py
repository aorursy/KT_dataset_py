import time

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
diab_df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

diab_df.head(6)
print ('dataframe shape: ', diab_df.shape)

print ('null values in the entire dataframe: ', diab_df.isnull().values.any()) # check for NaNs

print ('total number of null values: ', diab_df.isnull().sum().sum()) # total number of NaNs

print ('number of positive (1) and negative (0) diabetic patients: ', '\n', diab_df.Outcome.value_counts())
diab_df.rename(columns={'DiabetesPedigreeFunction': 'DiabPedgFunct'}, inplace=True)



print ('check dataframe columns :', diab_df.columns)
glucose_val_count0 = diab_df['Glucose'].value_counts()[0]

# print (glucose_val_count0)



bp_val_count0 = diab_df['BloodPressure'].value_counts()[0]

# print (bp_val_count0)



skin_th_count0 = diab_df['SkinThickness'].value_counts()[0]

# print (skin_th_count0)



insulin_count0 = diab_df['Insulin'].value_counts()[0]

# print(insulin_count0)



BMI_count0 = diab_df['BMI'].value_counts()[0]

# print(BMI_count0)



# Age_count0 = diab_df['Age'].value_counts()[30]

# print (Age_count0) # no zero values for age, gives a keyerror 



val_list0 = [glucose_val_count0/diab_df.shape[0], bp_val_count0/diab_df.shape[0], skin_th_count0/diab_df.shape[0], 

             insulin_count0/diab_df.shape[0], BMI_count0/diab_df.shape[0]]



labels0 = ['Glucose', 'BP', 'SkinThick', 'Insulin', 'BMI'] 

x = np.arange(len(labels0))



fig = plt.figure(figsize=(6, 5))

plt.bar(x, height=val_list0, width=0.4, align='center', color='magenta', alpha=0.7)

plt.xticks(ticks=x, labels=labels0, fontsize=12)

plt.title('Fraction of Missing Values', fontsize=14)

plt.show()
fig = plt.figure(figsize=(12, 7))

fig.add_subplot(241)

plt.hist(diab_df['Age'], bins=int(np.sqrt(diab_df.shape[0])), density= True, color='lime', alpha=0.6)

plt.xlabel('Age', fontsize=12)



fig.add_subplot(242)

plt.hist(diab_df['BloodPressure'], bins=int(np.sqrt(diab_df.shape[0])), density= True, color='lime', alpha=0.6)

plt.xlabel('Blood Pressure', fontsize=12)



fig.add_subplot(243)

plt.hist(diab_df['BMI'], bins=int(np.sqrt(diab_df.shape[0])), density= True, color='lime', alpha=0.6)

plt.xlabel('BMI', fontsize=12)



fig.add_subplot(244)

plt.hist(diab_df['DiabPedgFunct'], bins=int(np.sqrt(diab_df.shape[0])), density= True, color='lime', alpha=0.6)

plt.xlabel('Diabetes Pedigree Function', fontsize=12)



fig.add_subplot(245)

plt.hist(diab_df['Glucose'], bins=int(np.sqrt(diab_df.shape[0])), density= True, color='lime', alpha=0.6)

plt.xlabel('Glucose', fontsize=12)



fig.add_subplot(246)

plt.hist(diab_df['Insulin'], bins=int(np.sqrt(diab_df.shape[0])), density= True, color='lime', alpha=0.6)

plt.xlabel('Insulin', fontsize=12)



fig.add_subplot(247)

plt.hist(diab_df['Pregnancies'], bins=int(np.sqrt(diab_df.shape[0])), density= True, color='lime', alpha=0.6)

plt.xlabel('Pregnancies', fontsize=12)



fig.add_subplot(248)

plt.hist(diab_df['SkinThickness'], bins=int(np.sqrt(diab_df.shape[0])), density= True, color='lime', alpha=0.6)

plt.xlabel('Skin  Thickness', fontsize=12)



plt.tight_layout()

plt.show()
diab_df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diab_df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.NaN)
# replace using mean 



diab_df['BMI_New'] = diab_df['BMI'].replace(np.NaN, diab_df['BMI'].mean())

diab_df['BloodPressure_New'] = diab_df['BloodPressure'].replace(np.NaN, diab_df['BloodPressure'].mean())

diab_df['Glucose_New'] = diab_df['Glucose'].replace(np.NaN, diab_df['Glucose'].mean())

diab_df['Insulin_New'] = diab_df['Insulin'].replace(np.NaN, diab_df['Insulin'].mean())

diab_df['SkinThickness_New'] = diab_df['SkinThickness'].replace(np.NaN, diab_df['SkinThickness'].mean())
fig = plt.figure(figsize=(10, 8))



fig.add_subplot(231)

plt.hist(diab_df['BMI_New'], density=True, bins=int(np.sqrt(diab_df.shape[0])), color='lime', alpha=0.7)

plt.xlabel('BMI', fontsize=12)



fig.add_subplot(232)

plt.hist(diab_df['BloodPressure_New'], density=True, bins=int(np.sqrt(diab_df.shape[0])), color='lime', alpha=0.7)

plt.xlabel('BP', fontsize=12)



fig.add_subplot(233)

plt.hist(diab_df['Glucose_New'], density=True, bins=int(np.sqrt(diab_df.shape[0])), color='lime', alpha=0.7)

plt.xlabel('Glucose', fontsize=12)



fig.add_subplot(234)

plt.hist(diab_df['Insulin_New'], density=True, bins=int(np.sqrt(diab_df.shape[0])), color='lime', alpha=0.7)

plt.xlabel('Insulin', fontsize=12)



fig.add_subplot(235)

plt.hist(diab_df['SkinThickness_New'], density=True, bins=int(np.sqrt(diab_df.shape[0])), color='lime', alpha=0.7)

plt.xlabel('Skin Thickness', fontsize=12)



plt.tight_layout()

plt.show()
diab_df = diab_df.astype(float)

diab_df['SkinThickness_New1'] = diab_df.SkinThickness.interpolate(method='linear', limit=400, limit_direction='both')

diab_df['Insulin_New1'] = diab_df.Insulin.interpolate(method='linear', limit=600, limit_direction='both')





fig = plt.figure(figsize=(6, 5))



fig.add_subplot(121)

plt.hist(diab_df['SkinThickness_New1'], density=True, color='lime', alpha=0.6)

plt.xlabel('Skin Thickness', fontsize=12)



fig.add_subplot(122)

plt.hist(diab_df['Insulin_New1'], density=True, color='lime', alpha=0.7)

plt.xlabel('Insulin_New1', fontsize=12)



plt.tight_layout()

plt.show()
### check the min and max values of the above 2 features. 



print ('Insulin max and min; ', max(diab_df['Insulin_New1']) , min(diab_df['Insulin_New1']))

print ('Skin Thickness max and min; ', max(diab_df['SkinThickness_New1']), min(diab_df['SkinThickness_New1']))
#### select the relevant features

diab_df_selected = diab_df[['Pregnancies', 'Glucose_New', 'BloodPressure_New', 'SkinThickness_New1', 'Insulin_New1',

       'BMI_New', 'DiabPedgFunct', 'Age', 'Outcome']]   
features = diab_df_selected.drop(['Outcome'], axis=1)



features_arr = features.to_numpy()

feature_names_list = features.columns.to_list()



positive_diab = features_arr[diab_df_selected.Outcome==1]

negative_diab = features_arr[diab_df_selected.Outcome==0]



fig,axes =plt.subplots(4,2, figsize=(10, 8))

ax = axes.ravel()



for i in range(8):

    _,bins= np.histogram(features_arr[:, i], bins=int(np.sqrt(768)) )

    # plt.close()

    ax[i].hist(positive_diab[:, i], bins=bins, histtype='stepfilled', edgecolor='red', linewidth=1.2, fill=False, alpha=0.8,)

    ax[i].hist(negative_diab[:, i], bins=bins, color='green', alpha=0.6)

    ax[i].set_title(feature_names_list[i],fontsize=12)

ax[0].legend(['Positive','Negative'],loc='best',fontsize=11)

plt.tight_layout()

plt.show() 
# features = list(diab_df_selected.columns[0:7])

# feature_corr = diab_df_selected[features].corr() # alternate way 





feature_corr = features.corr() 



fig = plt.figure(figsize=(10, 7))

g1 = sns.heatmap(feature_corr, cmap='coolwarm', vmin=0., vmax=1., )

g1.set_xticklabels(g1.get_xticklabels(), rotation=40, fontsize=10)

g1.set_yticklabels(g1.get_yticklabels(), rotation=40, fontsize=10)

plt.title('Correlation Plot of Features', fontsize=12)

plt.show()
sns.set(font_scale=1.2)

sns.pairplot(diab_df_selected, hue='Outcome', palette='Set2')
f, axes = plt.subplots(2, 3, figsize=(12, 8))



sns.set(font_scale=0.8)

sns.boxplot(x=diab_df_selected['Age'], ax=axes[0][0],)

sns.boxplot(x=diab_df_selected['Glucose_New'], ax=axes[1][0])

sns.boxplot(x=diab_df_selected['Insulin_New1'], ax=axes[1][1]) ### many outliers !!!

sns.boxplot(x=diab_df_selected['BMI_New'], ax=axes[0][1])

sns.boxplot(x=diab_df_selected['BloodPressure_New'], ax=axes[0][2])

sns.boxplot(diab_df_selected['Pregnancies'], ax=axes[1][2])
print (diab_df_selected['Insulin_New1'].describe())
diab_df_selected_Zscore = diab_df_selected[(np.abs(stats.zscore(diab_df_selected)) < 3).all(axis=1)]



print ('check new dataframe shape after rejecting outliers: ', diab_df_selected_Zscore.shape) # 50 rows are gone
f, axes = plt.subplots(2, 3, figsize=(12, 8))



sns.set(font_scale=0.8)

sns.boxplot(x=diab_df_selected_Zscore['Age'], ax=axes[0][0],)

sns.boxplot(x=diab_df_selected_Zscore['Glucose_New'], ax=axes[1][0])

sns.boxplot(x=diab_df_selected_Zscore['Insulin_New1'], ax=axes[1][1]) 

sns.boxplot(x=diab_df_selected_Zscore['BMI_New'], ax=axes[0][1])

sns.boxplot(x=diab_df_selected_Zscore['BloodPressure_New'], ax=axes[0][2])

sns.boxplot(diab_df_selected_Zscore['Pregnancies'], ax=axes[1][2])
Outcome_arr = diab_df_selected_Zscore['Outcome'].to_numpy()

features_Zscore = diab_df_selected_Zscore.drop(['Outcome'], axis=1)

features_Zscore_arr = features_Zscore.to_numpy()



print ('check shapes for features and outcome: ', features_Zscore_arr.shape, Outcome_arr.shape)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(features_Zscore_arr, Outcome_arr, test_size=0.20, random_state=42, shuffle=True, stratify=Outcome_arr)



print ('check shape of training data: ', X_train.shape, y_train.shape)

print ('check shape of test data: ', X_test.shape, y_test.shape)
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
pipelines = [ [('scaler', StandardScaler()), ('SVM', SVC())], [('scaler', StandardScaler()), ('LR', LogisticRegression())], 

             [ ('RF', RandomForestClassifier())],  [('ADB', AdaBoostClassifier(DecisionTreeClassifier(max_depth=2)))], 

             [('scaler', StandardScaler()), ('GNB', GaussianNB())]]
svm_param_grid = {'SVM__C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 40, 50, 75, 100, 200],  

                  'SVM__kernel': ['linear']}



LR_param_grid = {'LR__C': [0.01, 0.05, 0.1, 0.5, 1., 2., 5., 10.], 'LR__class_weight':['balanced']}



RF_param_grid = {'RF__criterion': ['gini', 'entropy'], 'RF__n_estimators': [30, 50, 75, 100, 125, 150, 200], 'RF__max_depth': [2, 3, 4]}



ADB_param_grid = {'ADB__n_estimators': [20, 40, 50, 75, 100, 200], 'ADB__learning_rate': [0.01, 0.05, 0.1, 0.5, 1., 2]}



GNB_param_grid = {'GNB__priors': [[0.35, 0.65], [0.4, 0.6]], 'GNB__var_smoothing': [1e-9, 1e-8]}



all_param_grid = [svm_param_grid, LR_param_grid, RF_param_grid, ADB_param_grid, GNB_param_grid]
all_pipelines = []



for p in pipelines:

    all_pipelines.append(Pipeline(p))

print ('check one of the pipelines: ', all_pipelines[4])  
from tqdm.notebook import tqdm



grid_scores = []

grid_best_params = []



time1 = time.time()



for x in tqdm(range(len(all_pipelines))):

    grid = GridSearchCV(all_pipelines[x], param_grid=all_param_grid[x], cv=5)

    grid.fit(X_train, y_train)

    score = grid.score(X_test, y_test)

    grid_scores.append(score)

    grid_best_params.append(grid.best_params_)

print ('!!!!! out of the loop !!!!!')  

print ('time taken: ', time.time() - time1, 'seconds')
print ('Below are the Selected Best Hyperparameter for Each Classifier: ')

print ('\n')

for x in range(len(grid_best_params)):

    print (grid_best_params[x])
models = ['SVM', 'Logistic Reg', 'Random Forest', 'AdaBoost', 'GaussianNB']

score_dict = dict(zip(models, grid_scores))

print ('check score for each model : \n', score_dict)
score_df = pd.DataFrame(score_dict.items(), columns=['Model', 'Score'])
score_df
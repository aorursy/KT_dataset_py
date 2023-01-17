import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
heart_data_df = pd.read_csv('../input/heart.csv')

heart_data_df.head()
heart_data_df.rename(columns={'sex':'sex(numeric)',

                              'cp':'chest_pain_type(numeric)', 

                              'trestbps':'resting_blood_pressure', 

                              'chol':'serum_cholestrol', 

                              'fbs':'fasting_blood_sugar(numeric)',

                              'restecg':'resting_ecg_results(numeric)',

                              'thalach':'max_heart_rate', 

                              'exang':'exercise_angina(numeric)',

                              'ca':'n_major_colored_vessels(numeric)',

                              'slope':'slope(numeric)',

                              'thal':'thal(numeric)',

                              'target':'condition(numeric)'}, inplace=True)

heart_data_df.head()
heart_data_df['sex(categorical)'] = heart_data_df['sex(numeric)'].astype('category')

heart_data_df['sex(categorical)'].cat.categories = ['female', 'male']



heart_data_df['fasting_blood_sugar(categorical)'] = heart_data_df['fasting_blood_sugar(numeric)'].astype('category')

heart_data_df['fasting_blood_sugar(categorical)'].cat.categories = ['normal', 'high']



heart_data_df['exercise_angina(categorical)'] = heart_data_df['exercise_angina(numeric)'].astype('category')

heart_data_df['exercise_angina(categorical)'].cat.categories = ['no', 'yes']



heart_data_df['condition(categorical)'] = heart_data_df['condition(numeric)'].astype('category')

heart_data_df['condition(categorical)'].cat.categories = ['healthy', 'diseased']



heart_data_df.dtypes
sns.set(palette='tab10')
plot = sns.countplot(data=heart_data_df, x='condition(categorical)')

plot.set_ylim((0, 225))



counts = heart_data_df[['age', 'condition(categorical)']].groupby(['condition(categorical)']).count().rename(columns={'age':'count'}).reset_index()

total = counts['count'].sum()



counts_iter = 0

for barIndex in range(len(counts['condition(categorical)'].unique())):

    plot.text(barIndex,

              counts.loc[counts_iter, 'count'] + 8,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1
plt.figure(figsize=(7,5))

plot = sns.countplot(data=heart_data_df, x='condition(categorical)', hue='sex(categorical)')

plot.set_ylim((0, 160))



counts = heart_data_df[['age', 'sex(categorical)', 'condition(categorical)']].groupby(['condition(categorical)', 'sex(categorical)']).count().rename(columns={'age':'count'}).reset_index()

total = counts['count'].sum()

counts_iter = 0

for barIndex in range(len(counts['condition(categorical)'].unique())):

    plot.text(barIndex - plot.patches[0].get_width() / 2,

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1

    

    plot.text(barIndex + plot.patches[0].get_width() / 2,

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1
plt.figure(figsize=(7,5))

plot = sns.countplot(data=heart_data_df, x='condition(categorical)', hue='fasting_blood_sugar(categorical)')

plot.set_ylim((0, 225))



counts = heart_data_df[['age', 'fasting_blood_sugar(categorical)', 'condition(categorical)']].groupby(['condition(categorical)', 'fasting_blood_sugar(categorical)']).count().rename(columns={'age':'count'}).reset_index()

total = counts['count'].sum()

counts_iter = 0

for barIndex in range(len(counts['condition(categorical)'].unique())):

    plot.text(barIndex - plot.patches[0].get_width() / 2,

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1

    

    plot.text(barIndex + plot.patches[0].get_width() / 2,

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1
plt.figure(figsize=(7,5))

plot = sns.countplot(data=heart_data_df, x='condition(categorical)', hue='exercise_angina(categorical)')

plot.set_ylim((0,200))



counts = heart_data_df[['age', 'exercise_angina(categorical)', 'condition(categorical)']].groupby(['condition(categorical)', 'exercise_angina(categorical)']).count().rename(columns={'age':'count'}).reset_index()

total = counts['count'].sum()

counts_iter = 0

for barIndex in range(len(counts['condition(categorical)'].unique())):

    plot.text(barIndex - plot.patches[0].get_width() / 2,

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1

    

    plot.text(barIndex + plot.patches[0].get_width() / 2,

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1
plt.figure(figsize=(13,5))

plot = sns.countplot(data=heart_data_df, x='condition(categorical)', hue='chest_pain_type(numeric)')

plot.set_ylim((0,140))



counts = heart_data_df[['age', 'chest_pain_type(numeric)', 'condition(categorical)']].groupby(['condition(categorical)', 'chest_pain_type(numeric)']).count().rename(columns={'age':'count'}).reset_index()

total = counts['count'].sum()

counts_iter = 0

for barIndex in range(len(counts['condition(categorical)'].unique())):

    plot.text(barIndex - 1.5 * plot.patches[0].get_width(),

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1

    

    plot.text(barIndex - 0.5 * plot.patches[0].get_width(),

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1

    

    plot.text(barIndex + 0.5 * plot.patches[0].get_width(),

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1

    

    plot.text(barIndex + 1.5 * plot.patches[0].get_width(),

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1
plt.figure(figsize=(10,5))

plot = sns.countplot(data=heart_data_df, x='condition(categorical)', hue='resting_ecg_results(numeric)')

plot.set_ylim((0, 160))



counts = heart_data_df[['age', 'resting_ecg_results(numeric)', 'condition(categorical)']].groupby(['condition(categorical)', 'resting_ecg_results(numeric)']).count().rename(columns={'age':'count'}).reset_index()

total = counts['count'].sum()

counts_iter = 0

for barIndex in range(len(counts['condition(categorical)'].unique())):

    plot.text(barIndex - plot.patches[0].get_width(),

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1

    

    plot.text(barIndex,

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1

    

    plot.text(barIndex + plot.patches[0].get_width(),

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1
plt.figure(figsize=(13,5))

plot = sns.countplot(data=heart_data_df, x='condition(categorical)', hue='thal(numeric)')

plot.set_ylim((0,180))



counts = heart_data_df[['age', 'thal(numeric)', 'condition(categorical)']].groupby(['condition(categorical)', 'thal(numeric)']).count().rename(columns={'age':'count'}).reset_index()

total = counts['count'].sum()

counts_iter = 0

for barIndex in range(len(counts['condition(categorical)'].unique())):

    plot.text(barIndex - 1.5 * plot.patches[0].get_width(),

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1

    

    plot.text(barIndex - 0.5 * plot.patches[0].get_width(),

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1

    

    plot.text(barIndex + 0.5 * plot.patches[0].get_width(),

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1

    

    plot.text(barIndex + 1.5 * plot.patches[0].get_width(),

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1
plt.figure(figsize=(13,5))

plot = sns.countplot(data=heart_data_df, x='condition(categorical)', hue='slope(numeric)')

plot.set_ylim((0,180))



counts = heart_data_df[['age', 'slope(numeric)', 'condition(categorical)']].groupby(['condition(categorical)', 'slope(numeric)']).count().rename(columns={'age':'count'}).reset_index()

total = counts['count'].sum()

counts_iter = 0

for barIndex in range(len(counts['condition(categorical)'].unique())):

    plot.text(barIndex - plot.patches[0].get_width(),

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1

    

    plot.text(barIndex,

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1

    

    plot.text(barIndex + plot.patches[0].get_width(),

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1
plt.figure(figsize=(16,5))

plot = sns.countplot(data=heart_data_df, x='condition(categorical)', hue='n_major_colored_vessels(numeric)')

plot.set_ylim((0,180))



counts = heart_data_df[['age', 'n_major_colored_vessels(numeric)', 'condition(categorical)']].groupby(['condition(categorical)', 'n_major_colored_vessels(numeric)']).count().rename(columns={'age':'count'}).reset_index()

total = counts['count'].sum()

counts_iter = 0

for barIndex in range(len(counts['condition(categorical)'].unique())):

    plot.text(barIndex - 2 * plot.patches[0].get_width(),

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1

    

    plot.text(barIndex - plot.patches[0].get_width(),

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1

    

    plot.text(barIndex,

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1

    

    plot.text(barIndex + plot.patches[0].get_width(),

              counts.loc[counts_iter, 'count'] + 2.5,

              str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

              color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1

    

    plot.text(barIndex + 2 * plot.patches[0].get_width(),

          counts.loc[counts_iter, 'count'] + 2.5,

          str(counts.loc[counts_iter, 'count']) + '\n(' + str(round((counts.loc[counts_iter, 'count'] / total)*100, 2)) + '%)',

          color='black', horizontalAlignment='center', fontsize=15)

    counts_iter += 1
scatter_mat = sns.pairplot(data=heart_data_df[['condition(categorical)', 'age', 'resting_blood_pressure', 'serum_cholestrol', 'max_heart_rate', 'oldpeak']],

             diag_kind='kde',

             hue='condition(categorical)',

             markers='o')

upper_triangle_indices = np.triu_indices_from(scatter_mat.axes, 1)

row_indices, column_indices = upper_triangle_indices[0], upper_triangle_indices[1]

for i,j in zip(row_indices, column_indices):

    scatter_mat.axes[i, j].set_visible(False)
experiments_df = pd.DataFrame(columns=['classifier', 'best_acc_params', 'best_prec_params', 'best_recall_params', 'best_f1_params', 'Scaled', 'n_PCA_comps', 'n_MDS_comps'])

feature_cols = ['age', 'sex(numeric)', 'fasting_blood_sugar(numeric)', 'chest_pain_type(numeric)', 'resting_blood_pressure', 'serum_cholestrol', 'resting_ecg_results(numeric)', 'max_heart_rate', 'exercise_angina(numeric)', 'oldpeak', 'slope(numeric)', 'n_major_colored_vessels(numeric)', 'thal(numeric)']

X = heart_data_df[feature_cols]

y = heart_data_df['condition(numeric)']
C_values = [round(n, 2) for n in list(np.arange(0.010, 1.1, 0.010))] + [round(n, 2) for n in list(np.arange(1.1, 2.1, 0.10)) + [*range(3, 31)]]

kernels = ['linear', 'rbf', 'sigmoid']

n_cores = -1
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier



from sklearn.model_selection import GridSearchCV



import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

warnings.filterwarnings('ignore', category=DeprecationWarning)
def get_best_hyper_params(classifier_type, input_features, labels, kernel, cv, n_jobs, param_grid):

    if(classifier_type == 'LogisticReg'):

        classifier = LogisticRegression(random_state=0)

    elif (classifier_type == 'SVM'):

        if(kernel not in kernels):

            return

        classifier = SVC(kernel=kernel, random_state=0)

    elif (classifier_type == 'kNN'):

        classifier = KNeighborsClassifier()

    elif (classifier_type == 'RandomForest'):

        classifier = RandomForestClassifier(random_state=0)

    else:

        return

        

    acc_grid = GridSearchCV(classifier, cv=cv, n_jobs=n_jobs, param_grid=param_grid)

    _ = acc_grid.fit(input_features, labels)

    best_acc_params = acc_grid.best_params_.copy()

    best_acc_params['score'] = acc_grid.best_score_

    

    prec_grid = GridSearchCV(classifier, cv=cv, n_jobs=n_jobs, param_grid=param_grid, scoring='precision')

    _ = prec_grid.fit(input_features, labels)

    best_prec_params = prec_grid.best_params_.copy()

    best_prec_params['score'] = prec_grid.best_score_

    

    recall_grid = GridSearchCV(classifier, cv=cv, n_jobs=n_jobs, param_grid=param_grid, scoring='recall')

    _ = recall_grid.fit(input_features, labels)

    best_recall_params = recall_grid.best_params_.copy()

    best_recall_params['score'] = recall_grid.best_score_

    

    f1_grid = GridSearchCV(classifier, cv=cv, n_jobs=n_jobs, param_grid=param_grid, scoring='f1')

    _ = f1_grid.fit(input_features, labels)

    best_f1_params = f1_grid.best_params_.copy()

    best_f1_params['score'] = f1_grid.best_score_

    

    return best_acc_params, best_prec_params, best_recall_params, best_f1_params
# (best_acc_params,

#  best_prec_params,

#  best_recall_params,

#  best_f1_params) = get_best_hyper_params(classifier_type='LogisticReg',

#                                              input_features=X,

#                                              labels=y, 

#                                              kernel=None,

#                                              cv=4,

#                                              n_jobs=n_cores, 

#                                              param_grid={'C':C_values})



# experiments_df = experiments_df.append({'classifier':'LogisticReg',

#                                         'best_prec_params':best_prec_params,

#                                         'best_recall_params':best_recall_params,

#                                         'best_acc_params':best_acc_params,

#                                         'best_f1_params':best_f1_params,

#                                         'Scaled':False,

#                                         'n_PCA_comps':0, 'n_MDS_comps':0}, ignore_index=True)

# experiments_df
gammas = C_values
# (best_acc_params,

#  best_prec_params,

#  best_recall_params,

#  best_f1_params) = get_best_hyper_params(classifier_type='SVM', 

#                                              input_features=X,

#                                              labels=y, 

#                                              kernel='linear',

#                                              cv=4,

#                                              n_jobs=n_cores, 

#                                              param_grid={'C':C_values})



# experiments_df = experiments_df.append({'classifier':'SVM_linear',

#                                         'best_prec_params':best_prec_params,

#                                         'best_recall_params':best_recall_params,

#                                         'best_acc_params':best_acc_params,

#                                         'best_f1_params':best_f1_params,

#                                         'Scaled':False,

#                                         'n_PCA_comps':0, 'n_MDS_comps':0}, ignore_index=True)

# experiments_df
# (best_acc_params,

#  best_prec_params,

#  best_recall_params,

#  best_f1_params) = get_best_hyper_params(classifier_type='SVM',

#                                              input_features=X,

#                                              labels=y, 

#                                              kernel='rbf',

#                                              cv=4,

#                                              n_jobs=n_cores, 

#                                              param_grid={'C':C_values, 'gamma':gammas})



# experiments_df = experiments_df.append({'classifier':'SVM_rbf',

#                                         'best_prec_params':best_prec_params,

#                                         'best_recall_params':best_recall_params,

#                                         'best_acc_params':best_acc_params,

#                                         'best_f1_params':best_f1_params,

#                                         'Scaled':False,

#                                         'n_PCA_comps':0, 'n_MDS_comps':0}, ignore_index=True)

# experiments_df
# (best_acc_params,

#  best_prec_params,

#  best_recall_params,

#  best_f1_params) = get_best_hyper_params(classifier_type='SVM',

#                                              input_features=X,

#                                              labels=y, 

#                                              kernel='sigmoid',

#                                              cv=4,

#                                              n_jobs=n_cores, 

#                                              param_grid={'C':C_values, 'gamma':gammas})



# experiments_df = experiments_df.append({'classifier':'SVM_sigmoid',

#                                         'best_prec_params':best_prec_params,

#                                         'best_recall_params':best_recall_params,

#                                         'best_acc_params':best_acc_params,

#                                         'best_f1_params':best_f1_params,

#                                         'Scaled':False,

#                                         'n_PCA_comps':0, 'n_MDS_comps':0}, ignore_index=True)

# experiments_df
n_neighbors = [*range(1, 21)]
# (best_acc_params,

#  best_prec_params,

#  best_recall_params,

#  best_f1_params) = get_best_hyper_params(classifier_type='kNN',

#                                              input_features=X,

#                                              labels=y,

#                                              kernel=None,

#                                              cv=4,

#                                              n_jobs=n_cores, 

#                                              param_grid={'n_neighbors':n_neighbors})



# experiments_df = experiments_df.append({'classifier':'kNN',

#                                         'best_prec_params':best_prec_params,

#                                         'best_recall_params':best_recall_params,

#                                         'best_acc_params':best_acc_params,

#                                         'best_f1_params':best_f1_params,

#                                         'Scaled':False,

#                                         'n_PCA_comps':0, 'n_MDS_comps':0}, ignore_index=True)

# experiments_df
n_estimators = [*range(4, 51)]

max_features = [*range(1, 14)]

max_leaf_nodes = [*range(2, 101)]
# (best_acc_params,

#  best_prec_params,

#  best_recall_params,

#  best_f1_params) = get_best_hyper_params(classifier_type='RandomForest',

#                                           input_features=X,

#                                           labels=y,

#                                           kernel=None,

#                                           cv=4,

#                                           n_jobs=n_cores, 

#                                           param_grid={'n_estimators':n_estimators, 'max_features':max_features, 

#                                                       'max_leaf_nodes':max_leaf_nodes})



# experiments_df = experiments_df.append({'classifier':'RandomForest',

#                                         'best_prec_params':best_prec_params,

#                                         'best_recall_params':best_recall_params,

#                                         'best_acc_params':best_acc_params,

#                                         'best_f1_params':best_f1_params,

#                                         'Scaled':False,

#                                         'n_PCA_comps':0, 'n_MDS_comps':0}, ignore_index=True)

# experiments_df
from sklearn.preprocessing import StandardScaler

scaled_X = pd.DataFrame(StandardScaler().fit_transform(X))
# (best_acc_params,

#  best_prec_params,

#  best_recall_params,

#  best_f1_params) = get_best_hyper_params(classifier_type='LogisticReg',

#                                              input_features=scaled_X,

#                                              labels=y, 

#                                              kernel=None,

#                                              cv=4,

#                                              n_jobs=n_cores, 

#                                              param_grid={'C':C_values})



# experiments_df = experiments_df.append({'classifier':'LogisticReg',

#                                         'best_prec_params':best_prec_params,

#                                         'best_recall_params':best_recall_params,

#                                         'best_acc_params':best_acc_params,

#                                         'best_f1_params':best_f1_params,

#                                         'Scaled':True,

#                                         'n_PCA_comps':0, 'n_MDS_comps':0}, ignore_index=True)

# experiments_df
# (best_acc_params,

#  best_prec_params,

#  best_recall_params,

#  best_f1_params) = get_best_hyper_params(classifier_type='SVM',

#                                              input_features=scaled_X,

#                                              labels=y,

#                                              kernel='linear',

#                                              cv=4,

#                                              n_jobs=n_cores,

#                                              param_grid={'C':C_values})



# experiments_df = experiments_df.append({'classifier':'SVM_linear',

#                                         'best_prec_params':best_prec_params,

#                                         'best_recall_params':best_recall_params,

#                                         'best_acc_params':best_acc_params,

#                                         'best_f1_params':best_f1_params,

#                                         'Scaled':True,

#                                         'n_PCA_comps':0, 'n_MDS_comps':0}, ignore_index=True)

# experiments_df
# (best_acc_params,

#  best_prec_params,

#  best_recall_params,

#  best_f1_params) = get_best_hyper_params(classifier_type='SVM',

#                                              input_features=scaled_X,

#                                              labels=y,

#                                              kernel='rbf',

#                                              cv=4,

#                                              n_jobs=n_cores,

#                                              param_grid={'C':C_values, 'gamma':gammas})



# experiments_df = experiments_df.append({'classifier':'SVM_rbf',

#                                         'best_prec_params':best_prec_params,

#                                         'best_recall_params':best_recall_params,

#                                         'best_acc_params':best_acc_params,

#                                         'best_f1_params':best_f1_params,

#                                         'Scaled':True,

#                                         'n_PCA_comps':0, 'n_MDS_comps':0}, ignore_index=True)

# experiments_df
# (best_acc_params,

#  best_prec_params,

#  best_recall_params,

#  best_f1_params) = get_best_hyper_params(classifier_type='SVM',

#                                              input_features=scaled_X,

#                                              labels=y,

#                                              kernel='sigmoid',

#                                              cv=4,

#                                              n_jobs=n_cores,

#                                              param_grid={'C':C_values, 'gamma':gammas})



# experiments_df = experiments_df.append({'classifier':'SVM_sigmoid',

#                                         'best_prec_params':best_prec_params,

#                                         'best_recall_params':best_recall_params,

#                                         'best_acc_params':best_acc_params,

#                                         'best_f1_params':best_f1_params,

#                                         'Scaled':True,

#                                         'n_PCA_comps':0, 'n_MDS_comps':0}, ignore_index=True)

# experiments_df
n_neighbors = [*range(1, 21)]
# (best_acc_params,

#  best_prec_params,

#  best_recall_params,

#  best_f1_params) = get_best_hyper_params(classifier_type='kNN',

#                                              input_features=scaled_X,

#                                              labels=y,

#                                              kernel=None,

#                                              cv=4,

#                                              n_jobs=n_cores,

#                                              param_grid={'n_neighbors':n_neighbors})



# experiments_df = experiments_df.append({'classifier':'kNN',

#                                         'best_prec_params':best_prec_params,

#                                         'best_recall_params':best_recall_params,

#                                         'best_acc_params':best_acc_params,

#                                         'best_f1_params':best_f1_params,

#                                         'Scaled':True,

#                                         'n_PCA_comps':0, 'n_MDS_comps':0}, ignore_index=True)

# experiments_df
n_estimators = [*range(4, 51)]

max_features = [*range(1, 14)]

max_leaf_nodes = [*range(2, 101)]
# (best_acc_params,

#  best_prec_params,

#  best_recall_params,

#  best_f1_params) = get_best_hyper_params(classifier_type='RandomForest',

#                                              input_features=scaled_X,

#                                              labels=y,

#                                              kernel=None,

#                                              cv=4,

#                                              n_jobs=n_cores,

#                                              param_grid={'n_estimators':n_estimators, 'max_features':max_features,

#                                                       'max_leaf_nodes':max_leaf_nodes})



# experiments_df = experiments_df.append({'classifier':'RandomForest',

#                                         'best_prec_params':best_prec_params,

#                                         'best_recall_params':best_recall_params,

#                                         'best_acc_params':best_acc_params,

#                                         'best_f1_params':best_f1_params,

#                                         'Scaled':True,

#                                         'n_PCA_comps':0, 'n_MDS_comps':0}, ignore_index=True)

# experiments_df
# experiments_df.to_msgpack('experiments.msgpack')
from sklearn.decomposition import PCA
# for n_comps in range(1, 13):

    

#     pca = PCA(n_components=n_comps, random_state=0)

#     pca_X = pd.DataFrame(pca.fit_transform(scaled_X))



#     (best_acc_params,

#      best_prec_params,

#      best_recall_params,

#      best_f1_params) = get_best_hyper_params(classifier_type='LogisticReg',

#                                                  input_features=pca_X,

#                                                  labels=y, 

#                                                  kernel=None,

#                                                  cv=4,

#                                                  n_jobs=n_cores, 

#                                                  param_grid={'C':C_values})



#     experiments_df = experiments_df.append({'classifier':'LogisticReg',

#                                             'best_prec_params':best_prec_params,

#                                             'best_recall_params':best_recall_params,

#                                             'best_acc_params':best_acc_params,

#                                             'best_f1_params':best_f1_params,

#                                             'Scaled':True,

#                                             'n_PCA_comps':n_comps, 'n_MDS_comps':0}, ignore_index=True)



    

#     (best_acc_params,

#      best_prec_params,

#      best_recall_params,

#      best_f1_params) = get_best_hyper_params(classifier_type='SVM',

#                                                  input_features=pca_X,

#                                                  labels=y,

#                                                  kernel='linear',

#                                                  cv=4,

#                                                  n_jobs=n_cores,

#                                                  param_grid={'C':C_values})



#     experiments_df = experiments_df.append({'classifier':'SVM_linear',

#                                             'best_prec_params':best_prec_params,

#                                             'best_recall_params':best_recall_params,

#                                             'best_acc_params':best_acc_params,

#                                             'best_f1_params':best_f1_params,

#                                             'Scaled':True,

#                                             'n_PCA_comps':n_comps, 'n_MDS_comps':0}, ignore_index=True)



    

#     (best_acc_params,

#      best_prec_params,

#      best_recall_params,

#      best_f1_params) = get_best_hyper_params(classifier_type='SVM',

#                                                  input_features=pca_X,

#                                                  labels=y,

#                                                  kernel='rbf',

#                                                  cv=4,

#                                                  n_jobs=n_cores,

#                                                  param_grid={'C':C_values, 'gamma':gammas})



#     experiments_df = experiments_df.append({'classifier':'SVM_rbf',

#                                             'best_prec_params':best_prec_params,

#                                             'best_recall_params':best_recall_params,

#                                             'best_acc_params':best_acc_params,

#                                             'best_f1_params':best_f1_params,

#                                             'Scaled':True,

#                                             'n_PCA_comps':n_comps, 'n_MDS_comps':0}, ignore_index=True)



    

#     (best_acc_params,

#      best_prec_params,

#      best_recall_params,

#      best_f1_params) = get_best_hyper_params(classifier_type='SVM',

#                                                  input_features=pca_X,

#                                                  labels=y,

#                                                  kernel='sigmoid',

#                                                  cv=4,

#                                                  n_jobs=n_cores,

#                                                  param_grid={'C':C_values, 'gamma':gammas})



#     experiments_df = experiments_df.append({'classifier':'SVM_sigmoid',

#                                             'best_prec_params':best_prec_params,

#                                             'best_recall_params':best_recall_params,

#                                             'best_acc_params':best_acc_params,

#                                             'best_f1_params':best_f1_params,

#                                             'Scaled':True,

#                                             'n_PCA_comps':n_comps, 'n_MDS_comps':0}, ignore_index=True)



    

#     (best_acc_params,

#      best_prec_params,

#      best_recall_params,

#      best_f1_params) = get_best_hyper_params(classifier_type='kNN',

#                                                  input_features=pca_X,

#                                                  labels=y,

#                                                  kernel=None,

#                                                  cv=4,

#                                                  n_jobs=n_cores,

#                                                  param_grid={'n_neighbors':n_neighbors})



#     experiments_df = experiments_df.append({'classifier':'kNN',

#                                             'best_prec_params':best_prec_params,

#                                             'best_recall_params':best_recall_params,

#                                             'best_acc_params':best_acc_params,

#                                             'best_f1_params':best_f1_params,

#                                             'Scaled':True,

#                                             'n_PCA_comps':n_comps, 'n_MDS_comps':0}, ignore_index=True)



    

#     max_features = list(set([1] + [*range(1, n_comps)]))

#     (best_acc_params,

#      best_prec_params,

#      best_recall_params,

#      best_f1_params) = get_best_hyper_params(classifier_type='RandomForest',

#                                                  input_features=pca_X,

#                                                  labels=y,

#                                                  kernel=None,

#                                                  cv=4,

#                                                  n_jobs=n_cores,

#                                                  param_grid={'n_estimators':n_estimators, 'max_features':max_features,

#                                                           'max_leaf_nodes':max_leaf_nodes})



#     experiments_df = experiments_df.append({'classifier':'RandomForest',

#                                             'best_prec_params':best_prec_params,

#                                             'best_recall_params':best_recall_params,

#                                             'best_acc_params':best_acc_params,

#                                             'best_f1_params':best_f1_params,

#                                             'Scaled':True,

#                                             'n_PCA_comps':n_comps, 'n_MDS_comps':0}, ignore_index=True)



#     experiments_df.to_msgpack('experiments.msgpack')
from sklearn.manifold import MDS
# for n_comps in range(1, 13):

#     mds = MDS(n_components=n_comps, random_state=0)

#     mds_X = pd.DataFrame(mds.fit_transform(scaled_X))

    

#     (best_acc_params,

#      best_prec_params,

#      best_recall_params,

#      best_f1_params) = get_best_hyper_params(classifier_type='LogisticReg',

#                                                  input_features=mds_X,

#                                                  labels=y, 

#                                                  kernel=None,

#                                                  cv=4,

#                                                  n_jobs=n_cores, 

#                                                  param_grid={'C':C_values})



#     experiments_df = experiments_df.append({'classifier':'LogisticReg',

#                                             'best_prec_params':best_prec_params,

#                                             'best_recall_params':best_recall_params,

#                                             'best_acc_params':best_acc_params,

#                                             'best_f1_params':best_f1_params,

#                                             'Scaled':True,

#                                             'n_PCA_comps':0, 'n_MDS_comps':n_comps}, ignore_index=True)



    

#     (best_acc_params,

#      best_prec_params,

#      best_recall_params,

#      best_f1_params) = get_best_hyper_params(classifier_type='SVM',

#                                                  input_features=mds_X,

#                                                  labels=y,

#                                                  kernel='linear',

#                                                  cv=4,

#                                                  n_jobs=n_cores,

#                                                  param_grid={'C':C_values})



#     experiments_df = experiments_df.append({'classifier':'SVM_linear',

#                                             'best_prec_params':best_prec_params,

#                                             'best_recall_params':best_recall_params,

#                                             'best_acc_params':best_acc_params,

#                                             'best_f1_params':best_f1_params,

#                                             'Scaled':True,

#                                             'n_PCA_comps':0, 'n_MDS_comps':n_comps}, ignore_index=True)



    

#     (best_acc_params,

#      best_prec_params,

#      best_recall_params,

#      best_f1_params) = get_best_hyper_params(classifier_type='SVM',

#                                                  input_features=mds_X,

#                                                  labels=y,

#                                                  kernel='rbf',

#                                                  cv=4,

#                                                  n_jobs=n_cores,

#                                                  param_grid={'C':C_values, 'gamma':gammas})



#     experiments_df = experiments_df.append({'classifier':'SVM_rbf',

#                                             'best_prec_params':best_prec_params,

#                                             'best_recall_params':best_recall_params,

#                                             'best_acc_params':best_acc_params,

#                                             'best_f1_params':best_f1_params,

#                                             'Scaled':True,

#                                             'n_PCA_comps':0, 'n_MDS_comps':n_comps}, ignore_index=True)



    

#     (best_acc_params,

#      best_prec_params,

#      best_recall_params,

#      best_f1_score) = get_best_hyper_params(classifier_type='SVM',

#                                                  input_features=mds_X,

#                                                  labels=y,

#                                                  kernel='sigmoid',

#                                                  cv=4,

#                                                  n_jobs=n_cores,

#                                                  param_grid={'C':C_values, 'gamma':gammas})



#     experiments_df = experiments_df.append({'classifier':'SVM_sigmoid',

#                                             'best_prec_params':best_prec_params,

#                                             'best_recall_params':best_recall_params,

#                                             'best_acc_params':best_acc_params,

#                                             'best_f1_params':best_f1_params,

#                                             'Scaled':True,

#                                             'n_PCA_comps':0, 'n_MDS_comps':n_comps}, ignore_index=True)



    

#     (best_acc_params,

#      best_prec_params,

#      best_recall_params,

#      best_f1_score) = get_best_hyper_params(classifier_type='kNN',

#                                                  input_features=mds_X,

#                                                  labels=y,

#                                                  kernel=None,

#                                                  cv=4,

#                                                  n_jobs=n_cores,

#                                                  param_grid={'n_neighbors':n_neighbors})



#     experiments_df = experiments_df.append({'classifier':'kNN',

#                                             'best_prec_params':best_prec_params,

#                                             'best_recall_params':best_recall_params,

#                                             'best_acc_params':best_acc_params,

#                                             'best_f1_params':best_f1_params,

#                                             'Scaled':True,

#                                             'n_PCA_comps':0, 'n_MDS_comps':n_comps}, ignore_index=True)



    

#     max_features = list(set([1] + [*range(1, n_comps)]))

#     (best_acc_params,

#      best_prec_params,

#      best_recall_params,

#      best_f1_score) = get_best_hyper_params(classifier_type='RandomForest',

#                                                  input_features=mds_X,

#                                                  labels=y,

#                                                  kernel=None,

#                                                  cv=4,

#                                                  n_jobs=n_cores,

#                                                  param_grid={'n_estimators':n_estimators, 'max_features':max_features,

#                                                           'max_leaf_nodes':max_leaf_nodes})



#     experiments_df = experiments_df.append({'classifier':'RandomForest',

#                                             'best_prec_params':best_prec_params,

#                                             'best_recall_params':best_recall_params,

#                                             'best_acc_params':best_acc_params,

#                                             'best_f1_params':best_f1_params,

#                                             'Scaled':True,

#                                             'n_PCA_comps':0, 'n_MDS_comps':n_comps}, ignore_index=True)



#     experiments_df.to_msgpack('experiments.msgpack')
!pip install wget
# #  this cell downloads the experiments.pkl pickle to the current directory on Kaggle cloud.

import wget

url = 'https://res.cloudinary.com/code-sage-cloud/raw/upload/v1553486810/experiments.msgpack'

msgpack_filename = wget.download(url, '')

msgpack_filename
experiments_df = pd.read_msgpack(msgpack_filename)
def get_acc_score(x):

    return dict(x['best_acc_params'])['score']

def get_prec_score(x):

    return dict(x['best_prec_params'])['score']

def get_recall_score(x):

    return dict(x['best_recall_params'])['score']

def get_f1_score(x):

    return dict(x['best_f1_params'])['score']
rows_without_dimen_reduction = experiments_df[(experiments_df['n_PCA_comps'] == 0) & (experiments_df['n_MDS_comps'] == 0)]
acc_without_dimen_reduction = rows_without_dimen_reduction[['classifier', 'best_acc_params']]

plt.figure(figsize=(16,6))

x = acc_without_dimen_reduction['classifier']

y = acc_without_dimen_reduction.apply(get_acc_score, axis=1)

barPlot = sns.barplot(x=x,y=y, hue=experiments_df['Scaled'])

_ = plt.ylim((0.5, 1))

_ = plt.title("Classifiers' Accuracy Scores Comparison (no-scaling vs. scaled)")

_ = plt.legend(title='Scaling', loc='lower right')

num_classifier_types = len(x.unique())

for barIndex in range(num_classifier_types):

    barPlot.text(barIndex - barPlot.patches[0].get_width() / 2,

                 y[barIndex] - 0.023,

                 str(round(y[barIndex], 5)),

                 color='w', horizontalalignment='center', fontsize=12)

    

    barPlot.text(barIndex + barPlot.patches[0].get_width() / 2,

                 y[barIndex + num_classifier_types] - 0.023,

                 str(round(y[barIndex + num_classifier_types], 5)),

                 color='w', horizontalalignment='center', fontsize=12)
prec_without_dimen_reduction = rows_without_dimen_reduction[['classifier', 'best_prec_params']]

plt.figure(figsize=(16,6))

x = prec_without_dimen_reduction['classifier']

y = prec_without_dimen_reduction.apply(get_prec_score, axis=1)

barPlot = sns.barplot(x=x,y=y, hue=experiments_df['Scaled'])

_ = plt.ylim((0.5, 1))

_ = plt.title("Classifiers' Precision Scores Comparison (no-scaling vs. scaled)")

_ = plt.legend(title='Scaling', loc='lower right')

num_classifier_types = len(x.unique())

for barIndex in range(num_classifier_types):

    barPlot.text(barIndex - barPlot.patches[0].get_width() / 2,

                 y[barIndex] - 0.023,

                 str(round(y[barIndex], 5)),

                 color='w', horizontalalignment='center', fontsize=12)

    

    barPlot.text(barIndex + barPlot.patches[0].get_width() / 2,

                 y[barIndex + 6] - 0.023,

                 str(round(y[barIndex + 6], 5)),

                 color='w', horizontalalignment='center', fontsize=12)
recall_without_dimen_reduction = rows_without_dimen_reduction[['classifier', 'best_recall_params']]

plt.figure(figsize=(16,6))

x = recall_without_dimen_reduction['classifier']

y = recall_without_dimen_reduction.apply(get_recall_score, axis=1)

barPlot = sns.barplot(x=x,y=y, hue=experiments_df['Scaled'])

_ = plt.ylim((0.5, 1.0))

_ = plt.title("Classifiers' Recall Scores Comparison (no-scaling vs. scaled)")

_ = plt.legend(title='Scaling', loc='lower right')

num_classifier_types = len(x.unique())

for barIndex in range(num_classifier_types):

    barPlot.text(barIndex - barPlot.patches[0].get_width() / 2,

                 y[barIndex] - 0.023,

                 str(round(y[barIndex], 5)),

                 color='w', horizontalalignment='center', fontsize=12)

    

    barPlot.text(barIndex + barPlot.patches[0].get_width() / 2,

                 y[barIndex + 6] - 0.023,

                 str(round(y[barIndex + 6], 5)),

                 color='w', horizontalalignment='center', fontsize=12)
f1_without_dimen_reduction = rows_without_dimen_reduction[['classifier', 'best_f1_params']]

plt.figure(figsize=(16,6))

x = f1_without_dimen_reduction['classifier']

y = f1_without_dimen_reduction.apply(get_f1_score, axis=1)

barPlot = sns.barplot(x=x,y=y, hue=experiments_df['Scaled'])

_ = plt.ylim((0.5, 1))

_ = plt.title("Classifiers' F1-Scores Comparison (no-scaling vs. scaled)")

_ = plt.legend(title='Scaling', loc='lower right')

num_classifier_types = len(x.unique())

for barIndex in range(num_classifier_types):

    barPlot.text(barIndex - barPlot.patches[0].get_width() / 2,

                 y[barIndex] - 0.023,

                 str(round(y[barIndex], 5)),

                 color='w', horizontalalignment='center', fontsize=12)

    

    barPlot.text(barIndex + barPlot.patches[0].get_width() / 2,

                 y[barIndex + 6] - 0.023,

                 str(round(y[barIndex + 6], 5)),

                 color='w', horizontalalignment='center', fontsize=12)
scaled_rows_without_reduc = rows_without_dimen_reduction[rows_without_dimen_reduction['Scaled'] == True]

pca_rows = scaled_rows_without_reduc.append(experiments_df[(experiments_df['n_PCA_comps'] > 0)])

mds_rows = scaled_rows_without_reduc.append(experiments_df[(experiments_df['n_MDS_comps'] > 0)])
plt.figure(figsize=(26,9))

plt.subplot(1,2,1)

x = pca_rows['n_PCA_comps']

y = pca_rows.apply(get_acc_score, axis=1)

y.index = pca_rows.index.copy()

hue = pca_rows['classifier']

_ = sns.lineplot(x=x, y=y, hue=hue)

_ = plt.legend(loc='lower right')

_ = plt.title("Classifiers' Accuracy Scores across 0 to 12 PCA components")

maxIndices = list(y.nlargest(3).index)

scatterXs = x[maxIndices]

scatterYs = y[maxIndices]

plt.scatter(x=scatterXs, y=scatterYs, s=120, facecolors='none', edgecolors='r')

for i in maxIndices:

    plt.text(scatterXs.loc[i] - 0.2,

             scatterYs.loc[i]+0.0022,

             str(round(scatterYs.loc[i], 5)), horizontalalignment='center')

max_pca_rows = pca_rows.loc[maxIndices, ['classifier', 'n_PCA_comps']].copy()

max_pca_rows['score'] =  pca_rows.loc[maxIndices].apply(get_acc_score, axis=1)

print('top 3 in PCA accuracy:\n', max_pca_rows, '\n\n')

    

plt.subplot(1,2,2)

x = mds_rows['n_MDS_comps']

y = mds_rows.apply(get_acc_score, axis=1)

y.index = mds_rows.index.copy()

hue = mds_rows['classifier']

_ = sns.lineplot(x=x, y=y, hue=hue)

_ = plt.title("Classifiers' Accuracy Scores across 0 to 12 MDS components")

maxIndices = list(y.nlargest(3).index)

scatterXs = x[maxIndices]

scatterYs = y[maxIndices]

plt.scatter(x=scatterXs, y=scatterYs, s=120, facecolors='none', edgecolors='r')

for i in maxIndices:

    plt.text(scatterXs.loc[i] - 0.2,

             scatterYs.loc[i]+0.0022,

             str(round(scatterYs.loc[i], 5)), horizontalalignment='center')

max_mds_rows = mds_rows.loc[maxIndices, ['classifier', 'n_MDS_comps']].copy()

max_mds_rows['score'] =  mds_rows.loc[maxIndices].apply(get_acc_score, axis=1)

print('top 3 in MDS accuracy:\n', max_mds_rows)
plt.figure(figsize=(26,9))

plt.subplot(1,2,1)

x = pca_rows['n_PCA_comps']

y = pca_rows.apply(get_prec_score, axis=1)

y.index = pca_rows.index.copy()

hue = pca_rows['classifier']

_ = sns.lineplot(x=x, y=y, hue=hue)

_ = plt.title("Classifiers' Precision Scores across 0 to 12 PCA components")

maxIndices = list(y.nlargest(5).index)

scatterXs = x[maxIndices]

scatterYs = y[maxIndices]

plt.scatter(x=scatterXs, y=scatterYs, s=120, facecolors='none', edgecolors='r')

for i in maxIndices:

    plt.text(scatterXs.loc[i] - 0.2,

             scatterYs.loc[i]+0.0022,

             str(round(scatterYs.loc[i], 5)), horizontalalignment='center')

max_pca_rows = pca_rows.loc[maxIndices, ['classifier', 'n_PCA_comps']].copy()

max_pca_rows['score'] =  pca_rows.loc[maxIndices].apply(get_prec_score, axis=1)

    

plt.subplot(1,2,2)

x = mds_rows['n_MDS_comps']

y = mds_rows.apply(get_prec_score, axis=1)

y.index = mds_rows.index.copy()

hue = mds_rows['classifier']

_ = sns.lineplot(x=x, y=y, hue=hue)

_ = plt.title("Classifiers' Precision Scores across 0 to 12 MDS components")

maxIndices = list(y.nlargest(5).index)

scatterXs = x[maxIndices]

scatterYs = y[maxIndices]

plt.scatter(x=scatterXs, y=scatterYs, s=120, facecolors='none', edgecolors='r')

for i in maxIndices:

    plt.text(scatterXs.loc[i] - 0.2,

             scatterYs.loc[i]+0.0022,

             str(round(scatterYs.loc[i], 5)), horizontalalignment='center')

max_mds_rows = mds_rows.loc[maxIndices, ['classifier', 'n_MDS_comps']].copy()

max_mds_rows['score'] =  mds_rows.loc[maxIndices].apply(get_prec_score, axis=1)

plt.show()



print('top 5 in PCA precision:\n\n', max_pca_rows, '\n\n')

print('top 5 in MDS precision:\n', max_mds_rows)
plt.figure(figsize=(26,9))

plt.subplot(1,2,1)

x = pca_rows['n_PCA_comps']

y = pca_rows.apply(get_recall_score, axis=1)

y.index = pca_rows.index.copy()

hue = pca_rows['classifier']

_ = sns.lineplot(x=x, y=y, hue=hue)

_ = plt.title("Classifiers' Recall Scores across 0 to 12 PCA components")

maxIndices = list(y.nlargest(5).index)

scatterXs = x[maxIndices]

scatterYs = y[maxIndices]

plt.scatter(x=scatterXs, y=scatterYs, s=120, facecolors='none', edgecolors='r')

for i in maxIndices:

    plt.text(scatterXs.loc[i] - 0.2,

             scatterYs.loc[i]+0.0022,

             str(round(scatterYs.loc[i], 5)), horizontalalignment='center')

max_pca_rows = pca_rows.loc[maxIndices, ['classifier', 'n_PCA_comps']].copy()

max_pca_rows['score'] =  pca_rows.loc[maxIndices].apply(get_recall_score, axis=1)

    

plt.subplot(1,2,2)

x = mds_rows['n_MDS_comps']

y = mds_rows.apply(get_recall_score, axis=1)

y.index = mds_rows.index.copy()

hue = mds_rows['classifier']

_ = sns.lineplot(x=x, y=y, hue=hue)

_ = plt.title("Classifiers' Recall Scores across 0 to 12 MDS components")

maxIndices = list(y.nlargest(5).index)

scatterXs = x[maxIndices]

scatterYs = y[maxIndices]

plt.scatter(x=scatterXs, y=scatterYs, s=120, facecolors='none', edgecolors='r')

for i in maxIndices:

    plt.text(scatterXs.loc[i] - 0.2,

             scatterYs.loc[i]+0.0022,

             str(round(scatterYs.loc[i], 5)), horizontalalignment='center')

max_mds_rows = mds_rows.loc[maxIndices, ['classifier', 'n_MDS_comps']].copy()

max_mds_rows['score'] =  mds_rows.loc[maxIndices].apply(get_recall_score, axis=1)

plt.show()



print('top 5 in PCA recall:\n\n', max_pca_rows, '\n\n')

print('top 5 in MDS recall:\n\n', max_mds_rows)
recall_of_SVM_sigmoid_pca = pca_rows[pca_rows['classifier'] == 'SVM_sigmoid'].apply(get_recall_score, axis=1).reset_index(drop=True)

recall_of_SVM_rbf_pca = pca_rows[pca_rows['classifier'] == 'SVM_rbf'].apply(get_recall_score, axis=1).reset_index(drop=True)

print('recall_of_SVM_rbf_pca == recall_of_SVM_sigmoid_pca:\n', recall_of_SVM_rbf_pca == recall_of_SVM_sigmoid_pca, '\n')



recall_of_SVM_sigmoid_mds = mds_rows[mds_rows['classifier'] == 'SVM_sigmoid'].apply(get_recall_score, axis=1).reset_index(drop=True)

recall_of_SVM_rbf_mds = mds_rows[mds_rows['classifier'] == 'SVM_rbf'].apply(get_recall_score, axis=1).reset_index(drop=True)

print('recall_of_SVM_rbf_mds == recall_of_SVM_sigmoid_mds:\n', recall_of_SVM_rbf_mds == recall_of_SVM_sigmoid_mds)
plt.figure(figsize=(28,9))

plt.subplot(1,2,1)

x = pca_rows['n_PCA_comps']

y = pca_rows.apply(get_f1_score, axis=1)

y.index = pca_rows.index.copy()

hue = pca_rows['classifier']

_ = sns.lineplot(x=x, y=y, hue=hue)

_ = plt.title("Classifiers' F1-Scores across 0 to 12 PCA components")

maxIndices = list(y.nlargest(5).index)

scatterXs = x[maxIndices]

scatterYs = y[maxIndices]

plt.scatter(x=scatterXs, y=scatterYs, s=120, facecolors='none', edgecolors='r')

for i in maxIndices:

    plt.text(scatterXs.loc[i] - 0.2,

             scatterYs.loc[i]+0.0022,

             str(round(scatterYs.loc[i], 5)), horizontalalignment='center')

max_pca_rows = pca_rows.loc[maxIndices, ['classifier', 'n_PCA_comps']].copy()

max_pca_rows['score'] =  pca_rows.loc[maxIndices].apply(get_f1_score, axis=1)

    

plt.subplot(1,2,2)

x = mds_rows['n_MDS_comps']

y = mds_rows.apply(get_f1_score, axis=1)

y.index = mds_rows.index.copy()

hue = mds_rows['classifier']

_ = sns.lineplot(x=x, y=y, hue=hue)

_ = plt.title("Classifiers' F1-Scores across 0 to 12 MDS components")

maxIndices = list(y.nlargest(5).index)

scatterXs = x[maxIndices]

scatterYs = y[maxIndices]

plt.scatter(x=scatterXs, y=scatterYs, s=120, facecolors='none', edgecolors='r')

for i in maxIndices:

    plt.text(scatterXs.loc[i] - 0.2,

             scatterYs.loc[i]+0.0022,

             str(round(scatterYs.loc[i], 5)), horizontalalignment='center')

max_mds_rows = mds_rows.loc[maxIndices, ['classifier', 'n_MDS_comps']].copy()

max_mds_rows['score'] =  mds_rows.loc[maxIndices].apply(get_f1_score, axis=1)

plt.show()



print('top 5 in PCA F1-score:\n\n', max_pca_rows, '\n\n')

print('top 5 in MDS F1-score:\n\n', max_mds_rows)
f1_of_SVM_rbf_mds = mds_rows[mds_rows['classifier'] == 'SVM_rbf'].apply(get_f1_score, axis=1).reset_index(drop=True)



f1_of_SVM_sigmoid_mds = mds_rows[mds_rows['classifier'] == 'SVM_sigmoid'].apply(get_f1_score, axis=1).reset_index(drop=True)



f1_of_kNN_mds = mds_rows[mds_rows['classifier'] == 'kNN'].apply(get_f1_score, axis=1).reset_index(drop=True)



f1_of_RandomForest_mds = mds_rows[mds_rows['classifier'] == 'RandomForest'].apply(get_f1_score, axis=1).reset_index(drop=True)



(f1_of_SVM_rbf_mds == f1_of_SVM_sigmoid_mds) & (f1_of_SVM_sigmoid_mds == f1_of_kNN_mds) &(f1_of_SVM_sigmoid_mds == f1_of_RandomForest_mds)
y = heart_data_df['condition(numeric)']
best_prec_experiment = dict(*experiments_df[(experiments_df['classifier'] == 'RandomForest') & (experiments_df['n_MDS_comps'] == 4)]['best_prec_params'])

prec_oriented_classifier = RandomForestClassifier(n_estimators=best_prec_experiment['n_estimators'],

                                                  max_features=best_prec_experiment['max_features'],

                                                  max_leaf_nodes=best_prec_experiment['max_leaf_nodes'],

                                                  random_state=0)
best_recall_experiment = dict(*experiments_df[(experiments_df['classifier'] == 'SVM_rbf') & (experiments_df['n_MDS_comps'] == 4)]['best_recall_params'])

recall_oriented_classifier = SVC(kernel='rbf',

                                 C=best_recall_experiment['C'],

                                 gamma=best_recall_experiment['gamma'],

                                 probability=True,

                                 random_state=0)
best_f1_experiment = dict(experiments_df[(experiments_df['classifier'] == 'RandomForest') & (experiments_df['n_PCA_comps'] == 0) & (experiments_df['n_MDS_comps'] == 0)]['best_f1_params'][5])

f1_oriented_classifier = RandomForestClassifier(n_estimators=best_f1_experiment['n_estimators'],

                                                  max_features=best_f1_experiment['max_features'],

                                                  max_leaf_nodes=best_f1_experiment['max_leaf_nodes'],

                                                  random_state=0)
from sklearn.base import BaseEstimator, ClassifierMixin

global Y

Y = y.copy()
class MyClassifier(BaseEstimator, ClassifierMixin):

    

    def __init__(self, prec=None, f1=None, recall=None, l=0, m=0, n=0):

#         self.threshold = threshold

        self.l, self.m, self.n = l,m,n

        self.prec = prec

        self.f1 = f1

        self.recall = recall

    

    def fit(self, X, y):

        mds_X = MDS(n_components=4, random_state=0).fit_transform(X)

        

        self.prec.fit(mds_X, y)

        self.f1.fit(X, y)

        self.recall.fit(mds_X,y)

        

        return self

    

    def predict(self, X):

        index = X.index.copy()

        mds_X = pd.DataFrame(MDS(n_components=4, random_state=0).fit_transform(X), index=index.copy())

        

        prec_results = self.prec.predict(mds_X)

        f1_results = self.f1.predict(X)

        recall_results = self.recall.predict(mds_X)

        

        prec_proba = self.prec.predict_proba(mds_X)

        f1_proba = self.f1.predict_proba(X)

        recall_proba = self.recall.predict_proba(mds_X)

        

        heuristic = self.l * prec_proba[:,1] * prec_results + self.m * f1_proba[:,1] * f1_results + self.n * recall_proba[:,1] * recall_results

        heuristic_df = pd.DataFrame(heuristic, columns=['heuristic'])

        

        heuristic_df['labels'] = 0

        heuristic_df['label_str'] = 'low'

        heuristic_df.loc[:,'labels'][heuristic_df['heuristic'] > 0.5] = 1

        heuristic_df.loc[:,'label_str'][heuristic_df['heuristic'] > 0] = 'medium'

        heuristic_df.loc[:,'label_str'][heuristic_df['heuristic'] > 0.5] = 'high'

        global Y

        heuristic_df['actual_label'] = [*Y.loc[index]]

    

        return heuristic_df[['labels', 'label_str', 'actual_label', 'heuristic']]

#         return heuristic_df['labels']

    

    def get_params(self, deep=True):

        return {"prec":self.prec, 'l':self.l, 'm':self.m, 'n':self.n,

                "f1":self.f1, "recall":self.recall}



    def set_params(self, **parameters):

        for parameter, value in parameters.items():

            setattr(self, parameter, value)

        return self
weights = [{'l':[l], 'm':[m] , 'n':[n]} for l in np.arange(0,1,0.01) for m in np.arange(0,1,0.01) for n in np.arange(0,1,0.01) if (l + m + n == 1.0) and (l != 0) and (m != 0) and (l >= m) and (m >= n)]
from sklearn.model_selection import cross_val_score, cross_val_predict
# classifier = MyClassifier(prec_oriented_classifier, f1_oriented_classifier, recall_oriented_classifier)

# # thresholds = [*np.arange(0, 1, 0.001)]

# heuristic_grid = GridSearchCV(classifier, cv=4, n_jobs=n_cores, param_grid=weights)
# heuristic_grid.fit(scaled_X, y)
# heuristic_grid.best_params_
# heuristic_grid.best_score_
# import pickle

# with open('with_0_recall_and_highest_score.pkl', "wb") as pkl_file:

#     pickle.dump(heuristic_grid, pkl_file)
# cross_val_score(classifier, scaled_X, y, cv=4, n_jobs=n_cores)
classifier = MyClassifier(prec_oriented_classifier, f1_oriented_classifier, recall_oriented_classifier, 0.5, 0.5)

cvp_result = cross_val_predict(classifier, scaled_X, y, cv=4, n_jobs=n_cores)
cvp = pd.DataFrame(cvp_result, columns=['label', 'label_str', 'actual_label', 'heuristic'])
cvp.groupby(['label_str', 'actual_label']).count()
count = cvp.groupby(['label_str', 'actual_label', 'label']).count()

# count['label'] = (count['label'] / count['label'].sum()) * 100

count
len(cvp['heuristic'].unique())
cvp.groupby(['label_str']).count()
# sns.distplot([*cvp['heuristic'][cvp.actual_label == 0]], bins=20)

# # sns.distplot?

# sns.distplot([*cvp['heuristic'][cvp.actual_label == 1]], bins=20)

plt.hist([*cvp['heuristic'][cvp.actual_label == 0]], bins=20, fc=(0,1,0,0.5))

plt.hist([*cvp['heuristic'][cvp.actual_label == 1]], bins=20, fc=(1,0,0,0.5))



plt.legend(['healthy', 'diseased'])

plt.xlabel('heuristic')

plt.ylabel('count')
cvp.dtypes
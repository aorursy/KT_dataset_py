# Basics

import pandas as pd

import numpy as np



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Preprocessing

from sklearn.preprocessing import StandardScaler, MinMaxScaler, binarize, LabelEncoder



# Model Selection

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV



# Model

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier



# Metrics 

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, accuracy_score



# Feature Selection

from sklearn.feature_selection import SelectKBest, chi2



# Warnings

import warnings as ws

ws.filterwarnings('ignore')



pd.pandas.set_option('display.max_columns', None)

sns.set_style("whitegrid")
# Load Dataset

data_hr = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

data_hr.head()
print( 'DataSet Shape {}'.format(data_hr.shape))



data_hr.columns
# Summary

def summary(data):

    df = {

     'Count' : data.shape[0],

     'NA values' : data.isna().sum(),

     '% NA' : round((data.isna().sum()/data.shape[0]) * 100, 2),

     'Unique' : data.nunique(),

     'Dtype' : data.dtypes

    } 

    return(pd.DataFrame(df))



print('Shape is :', data_hr.shape)

summary(data_hr)
data = data_hr.copy()

data.drop(['EmployeeCount', 'Over18', 'StandardHours','EmployeeNumber'], axis = 1, inplace = True)
num_var = [var for var in data if data[var].dtypes != 'O']

cat_var = [var for var in data if data[var].dtypes == 'O']
data[num_var].hist(bins = 25, figsize = (20,20))

plt.show()
cont_var = [var for var in num_var if len(data[var].unique()) > 10]

disc_var = [var for var in num_var if len(data[var].unique()) <= 10]
plt.figure(figsize = (13,15))

i = 0

for cont in cont_var:

    

    mu_yes = data[cont][data['Attrition'] == 'Yes'].mean()

    mu_no = data[cont][data['Attrition'] == 'No'].mean()

    

    plt.subplot(5,3,i+1)

    sns.kdeplot(data[cont][data['Attrition'] == 'Yes'], label = 'Yes (mean: {:.2f})'.format(mu_yes))

    sns.kdeplot(data[cont][data['Attrition'] == 'No'], label = 'No (mean: {:.2f})'.format(mu_no))

    plt.tight_layout()

    plt.title('{} vs Attrition'.format(cont))

    i+=1

plt.show()
plt.figure(figsize = (13,15))

i = 0

for disc in disc_var:

    

    j=0

    col = ['Fields', '% of Leavers']

    df_field = pd.DataFrame(columns = col)

    

    for field in list(data[disc].unique()):    

        ratio = data[(data[disc] == field ) & (data['Attrition'] == 'Yes')].shape[0]/data[data[disc] == field].shape[0]

        df_field.loc[j] = [field, ratio * 100]

        j+=1

    

    plt.subplot(5,3,i+1)

    sns.barplot(x = 'Fields', y = '% of Leavers', data = df_field)

    plt.tight_layout()

    plt.title('{} vs Attrition'.format(disc))

    i+=1

plt.show()
plt.figure(figsize = (13,25))

i = 0

for cat in cat_var[1:]:

    

    j=0

    col = ['Fields', '% of Leavers']

    df_field = pd.DataFrame(columns = col)

    

    for field in list(data[cat].unique()):    

        ratio = data[(data[cat] == field ) & (data['Attrition'] == 'Yes')].shape[0]/data[data[cat] == field].shape[0]

        df_field.loc[j] = [field, ratio * 100]

        j+=1

    

    plt.subplot(5,3,i+1)

    sns.barplot(x = 'Fields', y = '% of Leavers', data = df_field)

    plt.tight_layout()

    plt.xticks(rotation = 90)

    plt.title('{} vs Attrition'.format(cat))

    i+=1

plt.show()
# Attrition Rate

sns.countplot(x = 'Attrition', data = data)
data.columns
data['Target'] = data['Attrition'].replace({'No':0,'Yes':1})



# Find correlations with the target and sort

corr = data.corr()['Target'].sort_values()



print('-'*25)

print('Top 5 Positive Correlation')

print('-'*25)

print(corr.tail(5))



print('-'*25)

print('Top 5 Negative Correlation')

print('-'*25)

print(corr.head(5))
# Correlation Map

corr = data.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

# Heatmap

plt.figure(figsize=(15, 10))

sns.heatmap(corr, annot = True, fmt = '.2f', mask = mask, linewidths = 2, cmap="YlGnBu", vmax = 0.5 )

plt.plot()
data.drop('Target',axis = 1, inplace = True)
label_var = [var for var in cat_var if len(data[var].unique()) <=2]



le = LabelEncoder()

for label in label_var:

    data[label] = le.fit_transform(data[label])

    

print('{} columns were Label Encoded'.format(label_var))
data = pd.get_dummies(data, drop_first = True)

print('Shape of the data is {}'.format(data.shape))
# Scaling Data (MinMaxScaler)

scale = MinMaxScaler(feature_range = (0,5))

HR_col = list(data.columns)

HR_col.remove('Attrition')

for col in HR_col:

    data[col] = data[col].astype(float)

    data[[col]] = scale.fit_transform(data[[col]])

data['Attrition'] = pd.to_numeric(data['Attrition'], downcast='float')

data.head()
X = data.drop('Attrition', axis = 1)

Y = data['Attrition']
# Train Test Split

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.25, random_state = 7)



print('X_Train Shape : {}'.format(x_train.shape))

print('X_Test Shape : {}'.format(x_test.shape))

print('Y_Train Shape : {}'.format(y_train.shape))

print('Y_Test Shape : {}'.format(y_test.shape))
models = []

models.append(('LR',LogisticRegression(class_weight = 'balanced')))

models.append(('RF',RandomForestClassifier(n_estimators=100, random_state=42, class_weight = 'balanced')))

models.append(('SVM', SVC(gamma='auto', random_state=7)))

models.append(('KNN', KNeighborsClassifier()))

models.append(('DT', DecisionTreeClassifier(random_state=7)))
acc_score = []

auc_score = []

names = []



col = ['Model', 'ROC AUC Mean','ROC AUC Std', 'ACC Mean','ACC Std']

result = pd.DataFrame(columns = col)



i = 0

for name, model in models:

    kfold = StratifiedKFold(n_splits = 5, random_state = 42)

    cv_acc_score = cross_val_score(model, x_train, y_train, cv = kfold, scoring = 'accuracy')

    cv_auc_score = cross_val_score(model, x_train, y_train, cv = kfold, scoring = 'roc_auc')

    

    acc_score.append(cv_acc_score)

    auc_score.append(cv_auc_score)

    names.append(name)

    

    result.loc[i] = [name,cv_auc_score.mean(), cv_auc_score.std(), cv_auc_score.mean(), cv_auc_score.std()]

    i+=1



result = result.sort_values('ROC AUC Mean', ascending = False)

result
plt.figure(figsize = (10,5))

plt.subplot(1,2,1)

sns.boxplot(x = names, y = auc_score)

plt.title('ROC AUC Score')



plt.subplot(1,2,2)

sns.boxplot(x = names, y = acc_score)

plt.title('Accuracy Score')

plt.show()
# Normalized Confusion Matrix

def get_norm_cnf_matrix(y_test, y_pred):



    # Noramalized Confusion Matrix

    y_test_0 = y_test.value_counts()[0]

    y_test_1 = y_test.value_counts()[1]    

    cnf_norm_matrix = np.array([[1.0 / y_test_0,1.0/y_test_0],[1.0/y_test_1,1.0/y_test_1]])

    norm_cnf_matrix = np.around(confusion_matrix(y_test, y_pred) * cnf_norm_matrix,3)

    

    return(norm_cnf_matrix)



# Confusion Matrix

def plt_cnf_matrix(y_test,y_pred):

    

    # Confusion Matrix`

    cnf_matrix = confusion_matrix(y_test, y_pred)    

    

    # Normalized Confusion Matrix

    norm_cnf_matrix = get_norm_cnf_matrix(y_test, y_pred)

    

    # Confusion Matrix plot

    plt.figure(figsize = (15,3))

    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

    plt.subplot(1,2,1)

    sns.heatmap(cnf_matrix, annot = True, fmt = 'g', cmap = plt.cm.Blues)

    plt.xlabel('Predicted Label')

    plt.ylabel('True Label')

    plt.title('Confusion Matrix')

    

    # Noramalized Confusion Matrix Plot

    plt.subplot(1,2,2)

    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

    sns.heatmap(norm_cnf_matrix, annot = True, fmt = 'g', cmap = plt.cm.Blues)

    plt.xlabel('Predicted Label')

    plt.ylabel('True Label')  

    plt.title('Normalized Confusion Matrix')

    plt.show()

    

    print('-'*25)

    print('Classification Report')

    print('-'*25)

    print(classification_report(y_test, y_pred))

    

    
hyper_parameter = {

    'C' : np.arange(0.0001, 2, 0.01)

}



lm_model = LogisticRegression(solver='liblinear',class_weight = 'balanced')

randomized_model = RandomizedSearchCV(lm_model,hyper_parameter, n_jobs=-1, cv = 10, verbose = 1,  scoring='roc_auc')

randomized_model.fit(x_train, y_train)
print('='*25)

print('Best Estimertor')

print('='*25)

print(randomized_model.best_estimator_)

print('\n')



print('='*25)

print('Best Parameter')

print('='*25)

print(randomized_model.best_params_)

print('\n')



print('='*25)

print('Best Score')

print('='*25)

print(randomized_model.best_score_)
lm_final_model = randomized_model.best_estimator_

y_pred_lm = lm_final_model.predict(x_test)

plt_cnf_matrix(y_test, y_pred_lm)
probs = lm_final_model.predict_proba(x_test) # predict probabilities

probs = probs[:, 1] # we will only keep probabilities associated with the employee leaving

logit_roc_auc = roc_auc_score(y_test, probs) # calculate AUC score using test dataset

print('AUC score for Logistic Regression : %.3f' % logit_roc_auc)
# Parameters

n_estimators = [50, 75, 100, 125, 150, 175]

max_depth = [5, 10, 15, 20, 25]

min_samples_split = [2,4,6,8,10]

min_samples_leaf = [1, 2, 3, 4]

max_features = ['auto', 'sqrt']
random_grid = {

    'n_estimators' : n_estimators,

    'max_depth' : max_depth,

    'min_samples_split' : min_samples_split,

    'min_samples_leaf' : min_samples_leaf

}



rm = RandomForestClassifier(class_weight = 'balanced',random_state = 7)

rm_random_model = RandomizedSearchCV(rm, random_grid, n_jobs=-1, cv = 10, verbose = 1, random_state = 42, scoring = 'roc_auc', iid = True)

rm_random_model.fit(x_train, y_train)
print('='*25)

print('Best Estimertor')

print('='*25)

print(rm_random_model.best_estimator_)

print('\n')



print('='*25)

print('Best Parameter')

print('='*25)

print(rm_random_model.best_params_)

print('\n')



print('='*25)

print('Best Score')

print('='*25)

print(rm_random_model.best_score_)
rm_final_model = rm_random_model.best_estimator_

y_pred_rm = rm_final_model.predict(x_test)

plt_cnf_matrix(y_test, y_pred_rm)
rm_probs = rm_final_model.predict_proba(x_test) # predict probabilities

rm_probs = rm_probs[:, 1] # we will only keep probabilities associated with the employee leaving

rm_roc_auc = roc_auc_score(y_test, rm_probs) # calculate AUC score using test dataset

print('AUC score for Logistic Regression : %.3f' % rm_roc_auc)
importances = rm_final_model.feature_importances_

indices = np.argsort(importances)[::-1]

names = [x_train.columns[i] for i in indices]



fea = pd.DataFrame({

    'Names' : names,

    'Score' : importances[indices]    

})



plt.figure(figsize = (10,5))

sns.barplot(x = 'Names', y = 'Score', data = fea, color = 'darkBlue')

plt.xticks(rotation = 90)
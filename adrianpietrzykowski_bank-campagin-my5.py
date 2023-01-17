# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

sns.set(style='whitegrid', palette='muted', font_scale=1.5)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
MAIN_PATH = '../input/'

df = pd.read_csv('../input/bank-marketing-dataset/bank.csv')

# term_deposits = df.copy()

# Have a grasp of how our data looks.

df.head()

df.describe()

df['y'] = df['deposit']
# Build a function to show categorical values disribution

def plot_bar(column):

    # temp df 

    temp_1 = pd.DataFrame()

    # count categorical values

    temp_1['No_deposit'] = df[df['y'] == 'no'][column].value_counts()

    temp_1['Yes_deposit'] = df[df['y'] == 'yes'][column].value_counts()

    temp_1.plot(kind='bar')

    plt.xlabel(f'{column}')

    plt.ylabel('Number of clients')

    plt.title('Distribution of {} and deposit'.format(column))

    plt.show();
# Build a function to show categorical values disribution

def plot_bar_stacked(column):

    # temp df 

    temp_1 = pd.DataFrame()

    # count categorical values

    temp_1['Open Deposit'] = df[df['y'] == 'yes'][column].value_counts()/(df[column].value_counts())

    temp_1.plot(kind='bar')

    plt.xlabel(f'{column}')

    plt.ylabel('Reponse Rate %')

    plt.title('Reponse Rate on offer'.format(column))

    plt.show();
plot_bar('job'), plot_bar_stacked('job')
plot_bar('marital'), plot_bar_stacked('marital')
plot_bar('education'), plot_bar_stacked('education')
plot_bar('contact'), plot_bar_stacked('contact')
plot_bar('poutcome'), plot_bar_stacked('poutcome')
plot_bar('loan'), plot_bar_stacked('loan')
plot_bar('housing'), plot_bar_stacked('housing')
plot_bar('contact'), plot_bar_stacked('contact')
# Convert target variable into numeric

df.y = df.y.map({'no':0, 'yes':1}).astype('uint8')
df.describe()
# Build correlation matrix

corr = df.corr()

corr.style.background_gradient(cmap='PuBu')
#Verifying null values

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
df.isna().any()
df.isna().sum()
# Replacing values with binary ()

df.contact = df.contact.map({'cellular': 1, 'telephone': 0, 'unknown':0}).astype('uint8') #0 will means other

df.loan = df.loan.map({'yes': 1, 'unknown': 0, 'no' : 0}).astype('uint8')

df.housing = df.housing.map({'yes': 1, 'unknown': 0, 'no' : 0}).astype('uint8')

df.default = df.default.map({'no': 1, 'unknown': 0, 'yes': 0}).astype('uint8')

df.pdays = df.pdays.replace(999, 0) # replace with 0 if not contact 

df.previous = df.previous.apply(lambda x: 1 if x > 0 else 0).astype('uint8') # binary has contact or not



# binary if were was an outcome of marketing campane

df.poutcome = df.poutcome.map({'unknown':0, 'failure':0,'other':0, 'success':1}).astype('uint8')  #what mean unknow - not in campaign?
df.info()
corr = df.corr()

corr.style.background_gradient(cmap='PuBu')
'''Convert Duration Call into 5 category'''

def duration(data):

    data.loc[data['duration'] <= 102, 'duration'] = 1

    data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration'] = 2

    data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration'] = 3

    data.loc[(data['duration'] > 319) & (data['duration'] <= 645), 'duration'] = 4

    data.loc[data['duration']  > 645, 'duration'] = 5

    return data





duration(df);

df.head()
df_a = df.drop(columns=['day','month'],axis = 1) #drop features which shouldn't influent outcome of campaign and remove targer feature - y
#get dummies data for job,education and marital status

job_dum = pd.get_dummies(df_a['job']).rename(columns=lambda x: 'job_' + str(x))

education_dum = pd.get_dummies(df['education']).rename(columns=lambda x: 'education_' + str(x))

marital_dum = pd.get_dummies(df['marital']).rename(columns=lambda x: 'marital' + str(x))

## dummies for age and pdays - zrob buckety */
#create dataset with dummies variables

df_b = pd.concat([df_a,job_dum,education_dum,marital_dum],axis=1)
df_b.head()
df_target = df_b['y']

df_feat =  df_b.drop(columns=['job','marital','education','y','deposit'],axis = 1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df_feat)

scaled_features = scaler.transform(df_feat)

df_feat_sc = pd.DataFrame(scaled_features,columns=df_feat.columns)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_feat, df_target, test_size=0.30, random_state=101)

X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(df_feat_sc, df_target, test_size=0.30, random_state=101)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



clf = RandomForestClassifier(n_estimators = 50, max_depth = 4)



scores = []

num_features = len(df_feat_sc.columns)

for i in range(num_features):

    col = df_feat_sc.columns[i]

    score = np.mean(cross_val_score(clf, df_feat_sc[col].values.reshape(-1,1), df_target, cv=10))

    scores.append((float(score*100), col))



print(sorted(scores, reverse = True))
df_target = df_b['y']

df_feat = df_feat[['duration','pdays','previous','poutcome','housing','contact','age','balance','campaign','maritalsingle']]
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from sklearn.metrics import classification_report

from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import GridSearchCV



## Logistic Regression

from sklearn.linear_model import LogisticRegression

classifier_LR = LogisticRegression(random_state = 0, penalty = 'l1')

classifier_LR.fit(X_train_sc, y_train_sc)



# Predicting Test Set

y_pred = classifier_LR.predict(X_test_sc)



acc = accuracy_score(y_test_sc, y_pred)

prec = precision_score(y_test_sc, y_pred)

rec = recall_score(y_test_sc, y_pred)

f1 = f1_score(y_test_sc, y_pred)



y_pred_prob_lr = classifier_LR.predict_proba(X_test_sc)[:,1]

fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test_sc, y_pred_prob_lr)

roc_auc_lr = auc(fpr_lr, tpr_lr)

precision_lr, recall_lr, th_lr = precision_recall_curve(y_test_sc, y_pred_prob_lr)

precision_recall_auc_lr = auc(recall_lr, precision_lr)



results = pd.DataFrame([['Logistic Regression', acc, prec, rec, f1, roc_auc_lr, precision_recall_auc_lr]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC auc','Precision recall auc'])

results
## Logistic Regression with grid search

penalty = ['l1', 'l2']

# Create regularization hyperparameter space

C = [0.01,0.1,0.5,0.8,1.2,1.5]



hyperparameters = dict(C=C, penalty=penalty)

# Create grid search using 5-fold cross validation

clf = GridSearchCV(classifier_LR, hyperparameters, cv=5, verbose=0,scoring='roc_auc')

# Fit grid search

LR_best_model = clf.fit(X_train_sc, y_train_sc)



# Predicting Test Set

y_pred = LR_best_model.predict(X_test_sc)

acc = accuracy_score(y_test_sc, y_pred)

prec = precision_score(y_test_sc, y_pred)

rec = recall_score(y_test_sc, y_pred)

f1 = f1_score(y_test_sc, y_pred)





y_pred_prob_LR_best_model = LR_best_model.predict_proba(X_test_sc)[:,1]

fpr_LR_best_model, tpr_LR_best_model, thresholds_LR_best_model = roc_curve(y_test_sc, y_pred_prob_LR_best_model)

roc_auc_LR_best_model = auc(fpr_LR_best_model, tpr_LR_best_model)

precision_LR_best_model, recall_LR_best_model, th_LR_best_model = precision_recall_curve(y_test_sc, y_pred_prob_LR_best_model)

precision_recall_auc_LR_best_model = auc(recall_LR_best_model, precision_LR_best_model)







model_results = pd.DataFrame([['Linear Regression Best Model', acc, prec, rec, f1, roc_auc_LR_best_model, precision_recall_auc_LR_best_model]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC auc','Precision recall auc'])



results = results.append(model_results, ignore_index = True)

## Naive Bayes

from sklearn.naive_bayes import GaussianNB

classifier_NaiveB = GaussianNB()

classifier_NaiveB.fit(X_train_sc, y_train_sc)



# Predicting Test Set

y_pred = classifier_NaiveB.predict(X_test_sc)

acc = accuracy_score(y_test_sc, y_pred)

prec = precision_score(y_test_sc, y_pred)

rec = recall_score(y_test_sc, y_pred)

f1 = f1_score(y_test_sc, y_pred)





y_pred_prob_NaiveB = classifier_NaiveB.predict_proba(X_test_sc)[:,1]

fpr_NaiveB, tpr_NaiveB, thresholds_NaiveB = roc_curve(y_test_sc, y_pred_prob_NaiveB)

roc_auc_NaiveB = auc(fpr_NaiveB, tpr_NaiveB)

precision_NaiveB, recall_NaiveB, th_NaiveB = precision_recall_curve(y_test_sc, y_pred_prob_NaiveB)

precision_recall_auc_NaiveB = auc(recall_NaiveB, precision_NaiveB)







model_results = pd.DataFrame([['Naive Bayes (Gaussian)', acc, prec, rec, f1, roc_auc_NaiveB, precision_recall_auc_NaiveB]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC auc','Precision recall auc'])



results = results.append(model_results, ignore_index = True)

## Random Forest Gini (n=100)

from sklearn.ensemble import RandomForestClassifier

classifier_RandomForest100 = RandomForestClassifier(random_state = 0, n_estimators = 100,

                                    criterion = 'gini')

classifier_RandomForest100.fit(X_train_sc, y_train_sc)



# Predicting Test Set

y_pred = classifier_RandomForest100.predict(X_test_sc)

acc = accuracy_score(y_test_sc, y_pred)

prec = precision_score(y_test_sc, y_pred)

rec = recall_score(y_test_sc, y_pred)

f1 = f1_score(y_test_sc, y_pred)



y_pred_prob_RandomForest100 = classifier_RandomForest100.predict_proba(X_test_sc)[:,1]

fpr_RandomForest100, tpr_RandomForest100, thresholds_RandomForest100 = roc_curve(y_test_sc, y_pred_prob_RandomForest100)

roc_auc_RandomForest100 = auc(fpr_RandomForest100, tpr_RandomForest100)

precision_RandomForest100, recall_RandomForest100, th_RandomForest100 = precision_recall_curve(y_test_sc, y_pred_prob_RandomForest100)

precision_recall_auc_RandomForest100 = auc(recall_RandomForest100, precision_RandomForest100)



model_results = pd.DataFrame([['Random Forest Gini (n=100)', acc, prec, rec, f1, roc_auc_RandomForest100 , precision_recall_auc_RandomForest100 ]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC auc','Precision recall auc'])



results = results.append(model_results, ignore_index = True)



## Random Forest Gini (n=200)

from sklearn.ensemble import RandomForestClassifier

classifier_RandomForest200 = RandomForestClassifier(random_state = 0, n_estimators = 200,

                                    criterion = 'gini')

classifier_RandomForest200.fit(X_train_sc, y_train_sc)



# Predicting Test Set

y_pred = classifier_RandomForest200.predict(X_test_sc)

acc = accuracy_score(y_test_sc, y_pred)

prec = precision_score(y_test_sc, y_pred)

rec = recall_score(y_test_sc, y_pred)

f1 = f1_score(y_test_sc, y_pred)





y_pred_prob_RandomForest200 = classifier_RandomForest200.predict_proba(X_test_sc)[:,1]

fpr_RandomForest200, tpr_RandomForest200, thresholds_RandomForest200 = roc_curve(y_test_sc, y_pred_prob_RandomForest200)

roc_auc_RandomForest200 = auc(fpr_RandomForest200, tpr_RandomForest200)

precision_RandomForest200, recall_RandomForest200, th_RandomForest200 = precision_recall_curve(y_test_sc, y_pred_prob_RandomForest200)

precision_recall_auc_RandomForest200 = auc(recall_RandomForest200, precision_RandomForest200)



model_results = pd.DataFrame([['Random Forest Gini (n=200)', acc, prec, rec, f1, roc_auc_RandomForest200 , precision_recall_auc_RandomForest200 ]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC auc','Precision recall auc'])



results = results.append(model_results, ignore_index = True)

## Random Forest with GridSearch

# I use f1-score because unbalance classification



rf_model = RandomForestClassifier()

rf_params = {

 "n_estimators": [10,50,100,400],

 "max_depth": (2,5,10),

 "min_samples_split": (2,5,10,15),

 "min_samples_leaf": (1,5,10,15)

 }

rf_grid = GridSearchCV(rf_model,

 rf_params,

scoring='roc_auc',

 cv=5,

 verbose=1,

 n_jobs=-1)



rf_grid.fit(X_train_sc, y_train_sc)



# Predicting Test Set

y_pred = rf_grid.predict(X_test_sc)

acc = accuracy_score(y_test_sc, y_pred)

prec = precision_score(y_test_sc, y_pred)

rec = recall_score(y_test_sc, y_pred)

f1 = f1_score(y_test_sc, y_pred)





y_pred_prob_rf_grid = rf_grid.predict_proba(X_test_sc)[:,1]

fpr_rf_grid, tpr_rf_grid, thresholds_rf_grid = roc_curve(y_test_sc, y_pred_prob_rf_grid)

roc_auc_rf_grid = auc(fpr_rf_grid, tpr_rf_grid)

precision_rf_grid, recall_rf_grid, th_rf_grid = precision_recall_curve(y_test_sc, y_pred_prob_rf_grid)

precision_recall_auc_rf_grid = auc(recall_rf_grid, precision_rf_grid)



model_results = pd.DataFrame([['Random Forest Grid Search', acc, prec, rec, f1, roc_auc_rf_grid , precision_recall_auc_rf_grid ]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC auc','Precision recall auc'])



results = results.append(model_results, ignore_index = True)

## SVM (rbf)

from sklearn.svm import SVC

classifier_SVM = SVC(random_state = 0, kernel = 'rbf', probability= True)

classifier_SVM.fit(X_train_sc, y_train_sc)



# Predicting Test Set

y_pred = classifier_SVM.predict(X_test_sc)

acc = accuracy_score(y_test_sc, y_pred)

prec = precision_score(y_test_sc, y_pred)

rec = recall_score(y_test_sc, y_pred)

f1 = f1_score(y_test_sc, y_pred)



y_pred_prob_SVM = classifier_SVM.predict_proba(X_test_sc)[:,1]

fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(y_test, y_pred_prob_SVM)

roc_auc_SVM = auc(fpr_SVM, tpr_SVM)

precision_SVM, recall_SVM, th_SVM = precision_recall_curve(y_test_sc, y_pred_prob_SVM)

precision_recall_auc_SVM = auc(recall_SVM, precision_SVM)



model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1, roc_auc_SVM , precision_recall_auc_SVM ]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC auc','Precision recall auc'])



results = results.append(model_results, ignore_index = True)

## SVM with GridSearch



param_grid_svm = {'C': [0.1, 1, 10, 100, 1000],  

              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 

              'kernel': ['rbf']}  

  

svm_grid_rbf = GridSearchCV(SVC(probability= True), param_grid_svm, refit = True, verbose = 3, cv = 2, scoring='roc_auc') 

  

# fitting the model for grid search 

svm_grid_rbf.fit(X_train_sc, y_train_sc) 



# Predicting Test Set

y_pred = svm_grid_rbf.predict(X_test_sc)

acc = accuracy_score(y_test_sc, y_pred)

prec = precision_score(y_test_sc, y_pred)

rec = recall_score(y_test_sc, y_pred)

f1 = f1_score(y_test_sc, y_pred)



y_pred_prob_svm_grid_rbf = svm_grid_rbf.predict_proba(X_test_sc)[:,1]

fpr_svm_grid_rbf, tpr_svm_grid_rbf, thresholds_svm_grid_rbf = roc_curve(y_test_sc, y_pred_prob_svm_grid_rbf)

roc_auc_SVM_grid_rbf = auc(fpr_svm_grid_rbf, tpr_svm_grid_rbf)

precision_svm_grid_rbf, recall_svm_grid_rbf, th_svm_grid_rbf = precision_recall_curve(y_test_sc, y_pred_prob_svm_grid_rbf)

precision_recall_auc_SVM_grid_rbf = auc(recall_svm_grid_rbf, precision_svm_grid_rbf)



model_results = pd.DataFrame([['SVM (RBF) with Grid', acc, prec, rec, f1, roc_auc_SVM_grid_rbf , precision_recall_auc_SVM_grid_rbf ]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC auc','Precision recall auc'])

model_results

results = results.append(model_results, ignore_index = True)

## SVM (Linear) 

from sklearn.svm import SVC

classifier_SVM = SVC(random_state = 0, kernel = 'linear', probability= True)

classifier_SVM.fit(X_train_sc, y_train_sc)



# Predicting Test Set

y_pred = classifier_SVM.predict(X_test_sc)

acc = accuracy_score(y_test_sc, y_pred)

prec = precision_score(y_test_sc, y_pred)

rec = recall_score(y_test_sc, y_pred)

f1 = f1_score(y_test_sc, y_pred)





y_pred_prob_SVM = classifier_SVM.predict_proba(X_test)[:,1]

fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(y_test, y_pred_prob_SVM)

roc_auc_SVM = auc(fpr_SVM, tpr_SVM)

precision_SVM, recall_SVM, th_SVM = precision_recall_curve(y_test, y_pred_prob_SVM)

precision_recall_auc_SVM = auc(recall_SVM, precision_SVM)



model_results = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1, roc_auc_SVM , precision_recall_auc_SVM ]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC auc','Precision recall auc'])



results = results.append(model_results, ignore_index = True)

# KNN

from sklearn.neighbors import KNeighborsClassifier

Classifier_KNN = KNeighborsClassifier(n_neighbors=2)

Classifier_KNN.fit(X_train_sc, y_train_sc)



# Predicting Test Set

y_pred = Classifier_KNN.predict(X_test_sc)

acc = accuracy_score(y_test_sc, y_pred)

prec = precision_score(y_test_sc, y_pred)

rec = recall_score(y_test_sc, y_pred)

f1 = f1_score(y_test_sc, y_pred)



y_pred_prob_KNN = Classifier_KNN.predict_proba(X_test_sc)[:,1]

fpr_KNN, tpr_KNN, thresholds_KNN = roc_curve(y_test_sc, y_pred_prob_KNN)

roc_auc_KNN = auc(fpr_KNN, tpr_KNN)

precision_KNN, recall_KNN, th_KNN = precision_recall_curve(y_test_sc, y_pred_prob_KNN)

precision_recall_auc_KNN = auc(recall_KNN, precision_KNN)



model_results = pd.DataFrame([['KNeighborsClassifier', acc, prec, rec, f1, roc_auc_KNN , precision_recall_auc_KNN ]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC auc','Precision recall auc'])



model_results

results = results.append(model_results, ignore_index = True)

#KNN model with grid

Classifier_KNN_grid = KNeighborsClassifier()



knn_param_grid = {'n_neighbors':[2,4,5,6,7,19],

              'leaf_size':[1,3,5],

              'algorithm':['auto', 'kd_tree'],

              'n_jobs':[-1]}



#Fit the model

KNN_grid = GridSearchCV(Classifier_KNN_grid, knn_param_grid, cv=3,scoring='roc_auc')

KNN_grid.fit(X_train_sc, y_train_sc)



# Predicting Test Set

y_pred = Classifier_KNN.predict(X_test_sc)

acc = accuracy_score(y_test_sc, y_pred)

prec = precision_score(y_test_sc, y_pred)

rec = recall_score(y_test_sc, y_pred)

f1 = f1_score(y_test_sc, y_pred)



y_pred_prob_KNN = Classifier_KNN.predict_proba(X_test_sc)[:,1]

fpr_KNN, tpr_KNN, thresholds_KNN = roc_curve(y_test_sc, y_pred_prob_KNN)

roc_auc_KNN = auc(fpr_KNN, tpr_KNN)

precision_KNN, recall_KNN, th_KNN = precision_recall_curve(y_test_sc, y_pred_prob_KNN)

precision_recall_auc_KNN = auc(recall_KNN, precision_KNN)



model_results = pd.DataFrame([['KNeighborsClassifier Grid', acc, prec, rec, f1, roc_auc_KNN , precision_recall_auc_KNN ]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC auc','Precision recall auc'])



model_results

results = results.append(model_results, ignore_index = True)





# Neural Networks 

from sklearn.neural_network import MLPClassifier

Classifie_MLP = MLPClassifier(hidden_layer_sizes=(30,60,90,120),

 learning_rate='adaptive',

 batch_size=30,

 learning_rate_init=0.01,

 shuffle=True)



Classifie_MLP.fit(X_train_sc, y_train_sc)





# Predicting Test Set

y_pred = Classifie_MLP.predict(X_test_sc)

acc = accuracy_score(y_test_sc, y_pred)

prec = precision_score(y_test_sc, y_pred)

rec = recall_score(y_test_sc, y_pred)

f1 = f1_score(y_test_sc, y_pred)



y_pred_prob_MLP = Classifie_MLP.predict_proba(X_test_sc)[:,1]

fpr_MLP, tpr_MLP, thresholds_MLP = roc_curve(y_test_sc, y_pred_prob_MLP)

roc_auc_MLP = auc(fpr_MLP, tpr_MLP)

precision_MLP, recall_MLP, th_MLP = precision_recall_curve(y_test_sc, y_pred_prob_MLP)

precision_recall_auc_MLP = auc(recall_MLP, precision_MLP)



model_results = pd.DataFrame([['MLP - Neural Networks', acc, prec, rec, f1, roc_auc_MLP, precision_recall_auc_MLP ]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC auc','Precision recall auc'])



results = results.append(model_results, ignore_index = True)

#Xgboost

from xgboost import XGBClassifier

xgb_model = XGBClassifier()

xgb_model.fit(X_train_sc, y_train_sc)



# performance

y_pred = xgb_model.predict(X_test_sc)

acc = accuracy_score(y_test_sc, y_pred)

prec = precision_score(y_test_sc, y_pred)

rec = recall_score(y_test_sc, y_pred)

f1 = f1_score(y_test_sc, y_pred)



y_pred_prob_xgb_model = xgb_model.predict_proba(X_test_sc)[:,1]

fpr_xgb_model, tpr_xgb_model, thresholds_xgb_model = roc_curve(y_test_sc, y_pred_prob_xgb_model)

roc_auc_xgb_model = auc(fpr_xgb_model, tpr_xgb_model)

precision_xgb_model, recall_xgb_model, th_xgb_model = precision_recall_curve(y_test_sc, y_pred_prob_xgb_model)

precision_recall_auc_xgb_model = auc(recall_xgb_model, precision_xgb_model)



model_results = pd.DataFrame([['XGBoost', acc, prec, rec, f1, roc_auc_xgb_model, precision_recall_auc_xgb_model ]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC auc','Precision recall auc'])



results = results.append(model_results, ignore_index = True)







#Xgboost with grid

XGB_model = XGBClassifier()



XGB_params = {

 "n_estimators": [10,20,50],

 "max_depth": (5,15),

 "min_samples_split": (2,5,10,15),

 "min_samples_leaf": (1,5,10,15)

 }









XGB_grid = GridSearchCV(XGB_model,

 XGB_params,

scoring='f1',

 cv=5,

 verbose=1,

 n_jobs=-1)



XGB_grid.fit(X_train_sc, y_train_sc)





# performance

y_pred = XGB_grid.predict(X_test_sc)

acc = accuracy_score(y_test_sc, y_pred)

prec = precision_score(y_test_sc, y_pred)

rec = recall_score(y_test_sc, y_pred)

f1 = f1_score(y_test_sc, y_pred)



y_pred_prob_XGB_grid = XGB_grid.predict_proba(X_test_sc)[:,1]

fpr_XGB_grid, tpr_XGB_grid, thresholds_xgb_model = roc_curve(y_test_sc, y_pred_prob_XGB_grid)

roc_auc_XGB_grid = auc(fpr_XGB_grid, tpr_XGB_grid)

precision_XGB_grid, recall_XGB_grid, th_XGB_grid = precision_recall_curve(y_test_sc, y_pred_prob_XGB_grid)

precision_recall_auc_XGB_grid = auc(recall_XGB_grid, precision_XGB_grid)



model_results = pd.DataFrame([['XGBoost Grid', acc, prec, rec, f1, roc_auc_XGB_grid, precision_recall_auc_XGB_grid ]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC auc','Precision recall auc'])



results = results.append(model_results, ignore_index = True)
results
# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_svm_grid_rbf, tpr_svm_grid_rbf, label='SVM RBF with GridSearch (area = %0.3f)' % roc_auc_LR_best_model)

plt.plot(fpr_xgb_model, tpr_xgb_model, label='Xbboost with deafult(area = %0.3f)' % roc_auc_xgb_model)

plt.plot(fpr_rf_grid, tpr_rf_grid, label='Random Forest with GridSearch (area = %0.3f)' % roc_auc_rf_grid )

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC curves from the investigated models')

plt.legend(loc='best')

plt.show()
#Precision - recall curve

plt.plot([1, 0], [0, 1], 'k--')

plt.plot(recall_svm_grid_rbf, precision_svm_grid_rbf, label='SVM RBF with Grid')

plt.plot(recall_xgb_model, precision_xgb_model, label='XGB with default')

plt.plot(recall_rf_grid, precision_rf_grid, label='Random Forest with Grid')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.legend(loc='best')

plt.show()
print(classification_report(y_test_sc, xgb_model.predict(X_test_sc)))
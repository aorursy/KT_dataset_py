#! pip install git+https://www.github.com/keras-team/keras-contrib.git

! wget https://github.com/jganzabal/santander_kaggle_solutions_tests/raw/master/santander_helper.py
#Import base libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import warnings

from tqdm import tqdm_notebook as tqdm

import random

from scipy import interp

import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)

plt.style.use('ggplot')

%matplotlib inline

#Import sklearn libraries

from sklearn.metrics import accuracy_score,roc_auc_score

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import KFold,cross_val_predict,cross_validate,cross_val_score,train_test_split

from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler

#Help functions



def plot_2d_space(X, y, label='Classes'):

    plt.figure(figsize=(8,8))

    markers = ['o', 's']

    for l, m in zip(np.unique(y), markers):

        plt.scatter(

            X[y==l, 0],

            X[y==l, 1]

            , label=l, marker=m

            )

    plt.title(label)

    plt.legend(loc='upper right')

    plt.show()

#ROC/AUC

def plot_roc_curve(fpr, tpr):  

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
# GET INDICIES OF REAL TEST DATA FOR FE

#######################

# TAKE FROM YAG320'S KERNEL

# https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split



test_path = '../input/test.csv'

train_path = '../input/train.csv'



df_test = pd.read_csv(test_path)

df_test.drop(['ID_code'], axis=1, inplace=True)

df_test = df_test.values



unique_samples = []

unique_count = np.zeros_like(df_test)

for feature in range(df_test.shape[1]):

    _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)

    unique_count[index_[count_ == 1], feature] += 1



# Samples which have unique values are real the others are fake

real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]

synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]



print('Found',len(real_samples_indexes),'real test')

print('Found',len(synthetic_samples_indexes),'fake test')



###################



d = {}

for i in range(200): d['var_'+str(i)] = 'float32'

d['target'] = 'uint8'

d['ID_code'] = 'object'



train = pd.read_csv('../input/train.csv', dtype=d)

test = pd.read_csv('../input/test.csv', dtype=d)



print('Loaded',len(train),'rows of train')

print('Loaded',len(test),'rows of test')

print('Found',len(real_samples_indexes),'real test')

print('Found',len(synthetic_samples_indexes),'fake test')



###################



d = {}

for i in range(200): d['var_'+str(i)] = 'float32'

d['target'] = 'uint8'

d['ID_code'] = 'object'



train = pd.read_csv(train_path, dtype=d)

test = pd.read_csv(test_path, dtype=d)



print('Loaded',len(train),'rows of train')

print('Loaded',len(test),'rows of test')
# FREQUENCY ENCODE

def encode_FE(df,col,test):

    cv = df[col].value_counts()

    nm = col+'_FE'

    df[nm] = df[col].map(cv)

    test[nm] = test[col].map(cv)

    test[nm].fillna(0,inplace=True)

    if cv.max()<=255:

        df[nm] = df[nm].astype('uint8')

        test[nm] = test[nm].astype('uint8')

    else:

        df[nm] = df[nm].astype('uint16')

        test[nm] = test[nm].astype('uint16')        

    return



test['target'] = -1

comb = pd.concat([train,test.loc[real_samples_indexes]],axis=0,sort=True)

for i in range(200): 

    encode_FE(comb,'var_'+str(i),test)

train = comb[:len(train)]; del comb

print('Added 200 new magic features!')
del df_test, real_samples_indexes, synthetic_samples_indexes, unique_count, unique_samples, d
#Create DataFrames



# Load data with counts saved in the previous cells

df_train_data = train.drop(columns=['ID_code'])

df_train_X = df_train_data.drop(columns=['target'])



reverse_columns = True

if reverse_columns:

    reverse_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 15, 16, 18, 19, 22, 24, 25, 26,

                    27, 29, 32, 35, 37, 40, 41, 47, 48, 49, 51, 52, 53, 55, 60, 61,

                    62, 65, 66, 67, 69, 70, 71, 74, 78, 79, 82, 84, 89, 90, 91, 94,

                    95, 96, 97, 99, 103, 105, 106, 110, 111, 112, 118, 119, 125, 128,

                    130, 133, 134, 135, 137, 138, 140, 144, 145, 147, 151, 155, 157,

                    159, 161, 162, 163, 164, 167, 168, 170, 171, 173, 175, 176, 179,

                    180, 181, 184, 185, 187, 189, 190, 191, 195, 196, 199,

                    ]



    for j in reverse_list:

        df_train_X[f'var_{j}'] *= -1
#Define X,y

y = df_train_data['target'].values

X = df_train_X.values



print(df_train_X.shape)
#Check the distribution of target classes

df_train_data.groupby(by="target")["target"].count().plot(kind="bar",title="Distribution Of Classes")
# Normalize data

rs = RobustScaler()

X_scaled_rs = rs.fit_transform(X)
#Let`s plot a 2d scatter plot to see the distribution of 2 random dimensions

n = random.randrange(start=0,stop=198)

print(n)

plot_2d_space(X_scaled_rs[:,n:n+2], y[:], 'Imbalanced dataset')
from sklearn.metrics import accuracy_score,classification_report

from sklearn.metrics import confusion_matrix,roc_curve, auc,roc_auc_score

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import KFold,cross_validate,cross_val_predict,cross_val_score,StratifiedKFold

import seaborn as sns



ExtraTreeModel = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',

                                   max_depth=None, max_features='auto', max_leaf_nodes=None,

                                   min_impurity_decrease=0.0, min_impurity_split=None,

                                   min_samples_leaf=1, min_samples_split=2,

                                   min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,

                                   oob_score=False, random_state=None, verbose=0, warm_start=False)
#Train with cross validation

cv = StratifiedKFold(n_splits=5)



tprs = []

aucs = []

mean_fpr = np.linspace(0, 1, 100)



i = 0

for k_train, k_test in tqdm(cv.split(X_scaled_rs, y)):

    probas_ = ExtraTreeModel.fit(X_scaled_rs[k_train], y[k_train]).predict_proba(X_scaled_rs[k_test])

    # Compute ROC curve and area the curve

    fpr, tpr, thresholds = roc_curve(y[k_test], probas_[:, 1])

    tprs.append(interp(mean_fpr, fpr, tpr))

    tprs[-1][0] = 0.0

    roc_auc = auc(fpr, tpr)

    aucs.append(roc_auc)

    plt.plot(fpr, tpr, lw=1, alpha=0.3,

             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))



    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',

         label='Chance', alpha=.8)



mean_tpr = np.mean(tprs, axis=0)

mean_tpr[-1] = 1.0

mean_auc = auc(mean_fpr, mean_tpr)

std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color='b',

         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),

         lw=2, alpha=.8)



std_tpr = np.std(tprs, axis=0)

tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,

                 label=r'$\pm$ 1 std. dev.')



plt.xlim([-0.05, 1.05])

plt.ylim([-0.05, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
#Import NB model

from sklearn.naive_bayes import GaussianNB



def create_NB_Model():

    model = GaussianNB()

    return model



def fit_NB_Model(Model,X,y,cv_splits=5): 

    cv = KFold(n_splits=cv_splits)

    tprs = []

    aucs = []

    mean_fpr = np.linspace(0, 1, 100)

    

    for train,test in cv.split(X, y):

        probas_ = Model.fit(X[train], y[train]).predict_proba(X[test])

        # Compute ROC curve and area the curve

        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])

        tprs.append(interp(mean_fpr, fpr, tpr))

        tprs[-1][0] = 0.0

        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)

        

    mean_tpr = np.mean(tprs, axis=0)

    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)

    return Model,mean_auc



def get_NB_results(X,y,cv_splits=5):

    model_aucs  = []

    models = {}

    for k in tqdm(range(0,int(X.shape[1]/2))):

        Model = create_NB_Model()

        X_train = X[:,[k,k+200]]#.reshape(-1,1)

        model,auc = fit_NB_Model(Model,X_train,y,cv_splits=cv_splits)

        #Append model

        models["Model_"+str(k)] =model

        model_aucs.append(auc)

        print(str(auc)+str(" AUC Average for Model_")+str(k))

        #Fit each Model

    return models,np.array(model_aucs)



def make_NB_predictions(X_train,y_train,cv,X_test):

    #Create a new array to store results

    train_predictions = np.zeros_like(X_train)

    test_predictions = np.zeros_like(X_test)

    #Fit model

    models,model_aucs = get_NB_results(X_train,y_train,cv_splits=cv)

    #Iterate over the X

    for k in range(0,int(X_train.shape[1]/2)):

        X_train_pred = X_train[:,[k,k+200]]#.reshape(-1,1)

        train_probas_ = models.get("Model_"+str(k)).predict_proba(X_train_pred)[:, 1]

        train_predictions[:, k] = train_probas_

        

    for j in range(0,int(X_test.shape[1]/2)):

        X_test_pred = X_test[:,[j,j+200]]#.reshape(-1,1)

        test_probas_ = models.get("Model_"+str(j)).predict_proba(X_test_pred)[:, 1]

        test_predictions[:, j] = test_probas_

        

    return test_predictions,train_predictions



def save_submit_file(predictions, filename, test_filename=test_path, index_column='ID_code', target_column = 'target'):

    df_test_submit = pd.read_csv(test_filename).set_index(index_column)

    df_test_submit[target_column] = predictions

    df_test_submit[[target_column]].to_csv(filename)

    return 
_,train_predictions = make_NB_predictions(X_train = X_scaled_rs,y_train = y,cv = 5,X_test=X_scaled_rs)
from sklearn.linear_model import LogisticRegression

ensemble_model = LogisticRegression()

ensemble_model.fit(X = train_predictions, y = y)
fpr, tpr, thresholds = roc_curve(y, ensemble_model.predict_proba(train_predictions)[:, 1])

plot_roc_curve(fpr, tpr)
# Normalize test data

df_test = test.drop(columns=['ID_code', 'target'])

X_test = df_test.values

X_test_scaled_rs = rs.fit_transform(X_test)
test_predictions,train_predictions = make_NB_predictions(X_train = X_scaled_rs,y_train = y,cv = 5,X_test=X_test_scaled_rs)
int(X.shape[1]/2)
ensemble_model.fit(X = train_predictions, y = y)

predictions = ensemble_model.predict(test_predictions)

save_submit_file(predictions, "submission_rev4.csv")
from IPython.display import FileLink

FileLink('submission_rev4.csv')
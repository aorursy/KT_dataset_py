import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Preprocessing

from sklearn.preprocessing import StandardScaler



# Model Selection

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV



#Handing Imbalance Dataset

from imblearn.under_sampling import NearMiss

from imblearn.over_sampling import RandomOverSampler



# Model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



# Metrics

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, accuracy_score, f1_score



# Warnings

import warnings as ws

ws.filterwarnings('ignore')



pd.pandas.set_option('display.max_columns',None)
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')

data.head(5)
data.shape
print(np.round(data['Class'].value_counts()/data.shape[0] * 100,2))



plt.figure(figsize = (2,5))

sns.set_style('darkgrid')

sns.countplot(x = 'Class', data = data)

plt.title('Class')

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



print('Shape is :', data.shape)

summary(data)

pd.set_option('display.float_format', lambda x: '%.5f' % x)

data.describe()
df = data.copy()



# Standardize Amount Value

scale = StandardScaler()

df['Amount'] = scale.fit_transform(df[['Amount']])
# Train Test Ratio

def train_test_ratio(y_train):

    

    class_0 = np.round(y_train.value_counts()[0]/len(y_train),3)

    class_1 = np.round(y_train.value_counts()[1]/len(y_train),3)

    

    len_class_0 = y_train.value_counts()[0]

    len_class_1 = y_train.value_counts()[1]

    

    print('-'*25)

    print('Train - Test Ratio :')

    print('-'*25)

    print('class 0 : {} : {} %'.format(len_class_0,class_0))

    print('class 1 : {} : {} %'.format(len_class_1,class_1))
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
X = data.drop('Class', axis = 1)

Y = data['Class']
x_train_base, x_test_base, y_train_base, y_test_base = train_test_split(X, Y, test_size = 0.3, random_state = 7)

print(train_test_ratio(Y))



base_model = LogisticRegression()

base_model.fit(x_train_base, y_train_base)



y_pred_base = base_model.predict(x_test_base)

plt_cnf_matrix(y_test_base, y_pred_base)
under_sampling = NearMiss()

X_under, Y_under = under_sampling.fit_sample(X,Y)

 

print(train_test_ratio(Y_under))



x_train_under, x_test_under, y_train_under, y_test_under = train_test_split(X_under, Y_under, test_size = 0.3, random_state = 7)



under_sampling_model = LogisticRegression()

under_sampling_model.fit(x_train_under, y_train_under)



y_pred_under = under_sampling_model.predict(x_test_under)

plt_cnf_matrix(y_test_under, y_pred_under)
over_sampling = RandomOverSampler(random_state = 42)

X_over, Y_over = over_sampling.fit_sample(X,Y)

 

print(train_test_ratio(Y_over))



x_train_over, x_test_over, y_train_over, y_test_over = train_test_split(X_over, Y_over, test_size = 0.3, random_state = 7)



over_sampling_model = LogisticRegression()

over_sampling_model.fit(x_train_over, y_train_over)



y_pred_over = over_sampling_model.predict(x_test_over)

plt_cnf_matrix(y_test_over, y_pred_over)
X_bal, Y_bal = X,Y



print(train_test_ratio(Y_over))



x_train_bal, x_test_bal, y_train_bal, y_test_bal = train_test_split(X_bal, Y_bal, test_size = 0.3, random_state = 42)



balanced_model = LogisticRegression(class_weight='balanced')

balanced_model.fit(x_train_bal, y_train_bal)



y_pred_bal = balanced_model.predict(x_test_bal)

plt_cnf_matrix(y_test_bal, y_pred_bal)
models = []

models.append(('Base Model', base_model))

models.append(('Under Sampling Model', under_sampling_model))

models.append(('Over Sampling Model', over_sampling_model))

models.append(('Balanced Model', balanced_model))
def model_evaluation(y_train, y_pred):

    

    # Confusion Matrix

    cnf_matrix = confusion_matrix(y_train, y_pred)

    

    # Confusion Matrix Parameters

    tp = cnf_matrix[1,1] # True Poistive

    tn = cnf_matrix[0,0] # True Negative

    fn = cnf_matrix[1,0] # False Negative

    fp = cnf_matrix[0,1] # False Positive

    

    # True Positive Rate

    """ Howmany of the True correctly classified as True """

    tpr = tp/(tp+fn)

    

    # True Negative Rate

    """ Howmany of the False correctly classifier as False """

    tnr = tn/(tn+fp)

    

    # Precision (for True)

    """ Howmany of the predicted True actually True """

    precision = tp/(tp+fp)

    

    # F1 Score

    """ Weighted avg of Precision and Recall """

    fscore = f1_score(y_train, y_pred)

    

    # Accuracy

    acc = (tp+tn)/(tp+tn+fp+fn)



    return(tp,tn,fp,fn,tpr,tnr,precision,fscore,acc)
col = ['Model','TP','TN','FP','FN','TPR','TNR','Precision','F1 Score','Accuracy']

result = pd.DataFrame(columns = col)



i = 0

for name, model in models:

   

    Y_pred = model.predict(X)

    tp,tn,fp,fn,tpr,tnr,precision,fscore,acc = model_evaluation(Y, Y_pred)

    

    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    result.loc[i] = [name,tp,tn,fp,fn,tpr,tnr,precision,fscore,acc]

    i+=1

    

result
estimater = LogisticRegression(class_weight='balanced')

hyperparameters = {

    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],

    'penalty' : ['l1','l2','elasticnet', 'none']

}



x_train_bal, x_test_bal, y_train_bal, y_test_bal = train_test_split(X_bal, Y_bal, test_size = 0.3, random_state = 42)



rm_model = RandomizedSearchCV(estimater, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)

best_model = rm_model.fit(x_train_bal, y_train_bal)
rm_best_model = best_model.best_estimator_

rm_best_model
y_rm_pred = rm_best_model.predict(X)



tp,tn,fp,fn,tpr,tnr,precision,fscore,acc = model_evaluation(Y, y_rm_pred)

result.loc[5] = ['Tunned Model',tp,tn,fp,fn,tpr,tnr,precision,fscore,acc]

result
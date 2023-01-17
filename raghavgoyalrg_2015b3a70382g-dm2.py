#IMPORTING ALL THE NEEDED LIBRARIES



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



from imblearn.over_sampling import RandomOverSampler



from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score



from math import sqrt

from math import log



from sklearn.ensemble.forest import RandomForestClassifier

from sklearn.ensemble.weight_boosting import AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import GridSearchCV

#IMPORTING THE DATASET AND CONVERTING ALL '?' TO NA VALUES



train_raw = pd.read_csv('../input/train.csv', sep = ',', na_values = '?')
train_raw.head()
train_raw.info(verbose=True, null_counts=True)
train_raw.describe()
train_raw.nunique()
#UNDERSTANDING THE DATASET AND LOOKING FOR ANAMOLIES



for col in train_raw.columns:

        print("Column Name: "+col + " Values: ",train_raw[col].unique())

        print()
#CORRELATION HEAT MAP



f, ax = plt.subplots(figsize=(10, 8))

corr = train_raw.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot = True);
#REMOVING FEATURES WITH NUMBER OF MISSING VALUES > 50,000 OR 0.5 OF THE DATASET



train_raw  = train_raw[train_raw.columns[train_raw.isnull().mean() < 0.5]]
train_raw.describe()
train_raw
train_raw.info(verbose=True, null_counts=True)

train_raw.nunique()
#DROPPING ALL THE TUPLES CONTAINING NA VALUES



train_raw.dropna(inplace = True)
train_raw
train_raw.columns
#DROPPING 'ID' COLUMN



train_raw.drop('ID', inplace = True, axis =1)
#LISTING THE NUMERICAL AND CATEGORICAL COLUMNS SEPARATELY



cols = train_raw.columns

num_cols = train_raw._get_numeric_data().columns

print ('Numerical Columns = ', list(num_cols))

cat_cols = list(set(cols) - set(num_cols))

print ('Categorical Columns = ',cat_cols)
train_raw.head()
#CHECKING FOR COUNT OF CLASSES

#NOTICED THAT CLASSES ARE HIGHLY IMBALANCED



train_raw['Class'].value_counts()
#REMOVING CLASS COLUMN



target = 'Class'



X = train_raw.drop(target, axis = 1)

y = train_raw[target]
#SPLITTING THE TRAINING DATA INTO TRAINING AND VALIDATION SET USING TRAIN_TEST_SPLIT



X_train,X_cv,y_train,y_cv = train_test_split(X, y, test_size=0.2, random_state=1,shuffle=True)

train = pd.concat([X_train,y_train],axis=1)

cv = pd.concat([X_cv,y_cv],axis=1)
X_train
train['Class'].value_counts()
cv['Class'].value_counts()
#LABEL ENCODING THE TRAINING SET



le = LabelEncoder()



for col in X_train.columns:

    if(X_train[col].dtype == np.object):

        le.fit(X_train[col])

        X_train[col] = le.transform(X_train[col])
#LABEL ENCODING THE VALIDATION SET



le = LabelEncoder()





for col in X_cv.columns:

    if(X_cv[col].dtype == np.object):

        le.fit(X_cv[col])

        X_cv[col] = le.transform(X_cv[col])
#PERFORMING OVERSAMPLING TO FIX THE IMBALANCED CLASS PROBLEM

#ROS AND SMOTE HAVE BEEN USED, ROS GIVES BETTER RESULTS



ros = RandomOverSampler(random_state = 42)

X_train_res, y_train_res = ros.fit_sample(X_train, y_train)



#from imblearn.over_sampling import SMOTE

#sm = SMOTE(random_state=12, ratio = 1)

#X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

#CONVERTING THE RESULTANT LISTS TO DATAFRAMES



X_train_res = pd.DataFrame(X_train_res)

y_train_res = pd.DataFrame(y_train_res)
#PERFORMING MIN_MAX_NORMALIZATION



min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X_train_res)

X_train_res = pd.DataFrame(np_scaled)

np_scaled_val = min_max_scaler.transform(X_cv)

X_cv = pd.DataFrame(np_scaled_val)

X_train_res.head()

#WRITING A FUNCTION TO DEFINE THE PERFORMANCE METRICS, NAMELY, ACCURACY, RECALL, PRECISION, F1 SCORE AND ROC



def performance_metrics(y_true,y_pred):

    acc = accuracy_score(y_true, y_pred)

    rec = recall_score(y_true, y_pred)

    pre = precision_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    roc = roc_auc_score(y_true, y_pred)

    

    return acc,rec,pre,f1,roc
#DEFINING A LIST OF ALGORITHMS TO RUN OUR DATASET ON AND THEN USING A FOR LOOP TO RUN EACH MODEL ON THE DATASET

#AND OUTPUT THE CORRESPONDING METRIC SCORES AGAINST EACH CLASSIFIER



names = [

    'RandomForestClassifier',

    'AdaBoostClassifier',

    'KNeighborsClassifier',

    'DecisionTreeClassifier',

    'GaussianNB',

    'LogisticRegression'

]



classifiers = [

    RandomForestClassifier(),

    AdaBoostClassifier(),

    KNeighborsClassifier(),

    DecisionTreeClassifier(),

    GaussianNB(),

    LogisticRegression()

]

displ = []

i = 0



listmodels =[]

for model in classifiers:

    print(names[i])

    model.fit(X_train_res,y_train_res)

    listmodels.append(model)

    em = []

    em.append(names[i])

    y_pred = model.predict(X_cv)

    y_pred=list(y_pred)

    acc,rec,pre,f1,roc = performance_metrics(y_cv,y_pred)

    em.append(acc)

    em.append(rec)

    em.append(pre)

    em.append(f1)

    em.append(roc)

    displ.append(em)

    i = i + 1

    

output_class = pd.DataFrame(displ,columns=['Classifier Name','Accuracy','Recall','Precision','F1_Score','Area Under ROC'])

output_class
#Adaboost Hyperparameter tuning - learning rate, with base estimator DTC(maax depth = 1)



'''

a=[]

b=[]

for i in [0.3,1,1.1,1.2,1.3,1.5]:

    print(i)

    a.append(i)

    model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=1),learning_rate=i)

    model.fit(X_train_res,y_train_res)

    em = []

    em.append('Ada Boost Classifier')

    y_pred = model.predict(X_cv)

    y_pred=list(y_pred)

    acc,rec,pre,f1,roc = performance_metrics(y_cv,y_pred)

    b.append(roc)

plt.plot(np.log(a),b)

plt.show()

res = pd.DataFrame()

res['learning Rate'] = a

res['Area under ROC'] = b

res

'''
#Adaboost Hyperparameter tuning - learning rate, with base estimator DTC(maax depth = 2)

'''

a=[]

b=[]

for i in [0.3,1,1.1,1.2,1.3,1.5]:

    print(i)

    a.append(i)

    model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2),learning_rate=i)

    model.fit(X_train_res,y_train_res)

    em = []

    em.append('Ada Boost Classifier')

    y_pred = model.predict(X_cv)

    y_pred=list(y_pred)

    acc,rec,pre,f1,roc = performance_metrics(y_cv,y_pred)

    b.append(roc)

plt.plot(np.log(a),b)

plt.show()

res = pd.DataFrame()

res['learning Rate'] = a

res['Area under ROC'] = b

res

'''
#Adaboost Hyperparameter tuning - n_estimators, with base estimator DTC(maax depth = 1)(by default)

'''

a=[]

b=[]

for i in range(1,30):

    print(i)

    a.append(i*25)

    model = AdaBoostClassifier(n_estimators = i * 25)

    model.fit(X_train_res,y_train_res)

    em = []

    em.append('Ada Boost Classifier')

    y_pred = model.predict(X_cv)

    y_pred=list(y_pred)

    acc,rec,pre,f1,roc = performance_metrics(y_cv,y_pred)

    b.append(roc)

plt.plot(a,b)

plt.show()

res = pd.DataFrame()

res['n_estimators'] = a

res['Area under ROC'] = b

res

'''
#Adaboost Hyperparameter tuning - n_estimators, with base estimator DTC(maax depth = 2)

'''

a=[]

b=[]

for i in range(1,30):

    print(i)

    a.append(i*25)

    model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2),n_estimators = i * 25)

    model.fit(X_train_res,y_train_res)

    em = []

    em.append('Ada Boost Classifier')

    y_pred = model.predict(X_cv)

    y_pred=list(y_pred)

    acc,rec,pre,f1,roc = performance_metrics(y_cv,y_pred)

    b.append(roc)

plt.plot(a,b)

plt.show()

res = pd.DataFrame()

res['n_estimators'] = a

res['Area under ROC'] = b

res

'''
#Adaboost Hyperparameter tuning of base estimator i.e. Decision Tree

#DTC max-depth

'''

a=[]

b=[]

for i in [1,2,3,4,5,6,7,8,9,10]:

    print(i)

    a.append(i)

    model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=i))

    model.fit(X_train_res,y_train_res)

    em = []

    em.append('Ada Boost Classifier')

    y_pred = model.predict(X_cv)

    y_pred=list(y_pred)

    acc,rec,pre,f1,roc = performance_metrics(y_cv,y_pred)

    b.append(roc)

plt.plot(np.log(a),b)

plt.show()

res = pd.DataFrame()

res['DTC max-depth'] = a

res['Area under ROC'] = b

res

'''
#Adaboost Hyperparameter tuning of base estimator i.e. Decision Tree

#DTC min_sample_split

'''

a=[]

b=[]

for i in range(2,10):

    print(i)

    a.append(i)

    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1,min_samples_split = i))

    model.fit(X_train_res,y_train_res)

    em = []

    em.append('Ada Boost Classifier')

    y_pred = model.predict(X_cv)

    y_pred=list(y_pred)

    acc,rec,pre,f1,roc = performance_metrics(y_cv,y_pred)

    b.append(roc)

plt.plot(a,b)

plt.show()

res = pd.DataFrame()

res['DTC min_sample_split'] = a

res['Area under ROC'] = b

res

'''
#Running GridSearchCV to find optimal parameters

'''

rf_temp = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))        #Initialize the classifier object



parameters = {'n_estimators':[475,500,525],'learning_rate':[0.8,0.9,1]}    #Dictionary of parameters





grid_obj = GridSearchCV(rf_temp, parameters,scoring = 'roc_auc')         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train_res, y_train_res)        #Fit the gridsearch object with X_train,y_train



best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



print(grid_fit.best_params_)

'''
#AFTER FINDING THE OPTIMAL PARAMETERS (DTC WITH MAX DEPTH 1, N_ESTIMATORS OF ADABOOST AS 575, LEARNING RATE=1)



model = AdaBoostClassifier(base_estimator =DecisionTreeClassifier(max_depth=1), n_estimators=575, learning_rate=1)

model.fit(X_train_res,y_train_res)

y_pred = model.predict(X_cv)

y_pred=list(y_pred)

acc,rec,pre,f1,roc = performance_metrics(y_cv,y_pred)

print(roc)
#IMPORTING TEST SET



test_raw = pd.read_csv('../input/test.csv', sep = ',', na_values = '?')

test_raw.describe()
test_raw.info()
tst_idx = test_raw['ID']
#DROPPING 'ID' COLUMN FROM TEST SET



test_raw.drop(['ID'],axis=1,inplace=True)
test_raw.head()
#REMOVING FEATURES WITH NUMBER OF MISSING VALUES > 0.5 OF THE TEST SET





test_raw  = test_raw[test_raw.columns[test_raw.isnull().mean() < 0.5]]
test_raw.info()
#FILLING NA VALUES OF CATEGORICAL COLUMNS WITH MODE



a = ['Hispanic','COB FATHER','COB MOTHER','COB SELF']

for i in a:

    test_raw[i].fillna(test_raw[i].mode()[0] ,inplace= True)

    
#LABEL ENCODING



le = LabelEncoder()



for col in test_raw.columns:

    if(test_raw[col].dtype == np.object):

        le.fit(test_raw[col])

        test_raw[col] = le.transform(test_raw[col])
#Performing Min_Max Normalization



min_max_scaler = preprocessing.MinMaxScaler()

np_scaled_test = min_max_scaler.fit_transform(test_raw)

test_raw = pd.DataFrame(np_scaled_test)

test_raw.head()
test_raw.describe()
#RUNNING THE MODEL WITH OPTIMAL PARAMETERS CHOSEN AND PREDICTING CLASSES OF THE TEST SET



model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators = 575,learning_rate=1)

model.fit(X_train_res,y_train_res)

y_pred_tst = model.predict(test_raw)



#CONVERTING THE RESULT INTO A LIST



y_pred_tst = list(y_pred_tst)
#ADDING CLASS COLUMN IN TEST SET



test_raw['Class'] = y_pred_tst
#CREATING SUBMISSION FILE WITH TEST 'ID' COLUMN AND PREDICTED CLASS COLUMN



pd.concat([tst_idx,test_raw['Class']],axis = 1).to_csv('DM_FINAL_SUB.csv',index = False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(pd.concat([tst_idx,test_raw['Class']],axis = 1))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import average_precision_score

from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

import xgboost

from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import RandomizedSearchCV

import warnings

warnings.filterwarnings("ignore")

plt.rcParams['figure.figsize'] = (8.5,7.0)
train = pd.read_csv("../input/datasetshr/train.csv")

train.head()
print("Shape of train data is:", train.shape)
test = pd.read_csv("../input/datasetshr/test.csv")

test.head()
print('Shape of test data is:', test.shape)
train.dtypes
print("Null values in train data:")

train.isna().sum()
print("Null values in test data:")

test.isna().sum()
corr = train.iloc[:,1:].corr()

sns.heatmap(corr, linewidth = 0.1, linecolor = "black", cmap = sns.color_palette("Pastel1"),

           annot = True)

plt.autoscale(enable=True, axis='y')

plt.xticks(rotation = 45)

plt.yticks(rotation = 360)

plt.title("Correlation between features")

plt.show()
summary = train.loc[:, ['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'avg_training_score']].describe()

sns.heatmap(summary.transpose(), annot = True, linewidth = 0.1, linecolor = 'black', cmap = sns.color_palette("YlGnBu"))

plt.autoscale(enable=True, axis='y')

plt.title("Summary of the data")

plt.show()
ax = sns.boxplot(data=train.loc[:, ['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'avg_training_score']], orient="h", palette="Set2")

plt.title("Boxplot")
cols = ['department', 'region','education', 'gender', 'no_of_trainings',

        'previous_year_rating', 'recruitment_channel', 'KPIs_met >80%', 'awards_won?']



def countplots(i):

    plt.figure(figsize = (8,40))

    plt.subplot(9,1,i+1)

    sns.countplot(train[cols[i]])

    plt.xticks(rotation = 90)

    plt.show()



for x in range(len(cols)):

    countplots(x)
#Number of 'not promoted' and 'promoted' employees

train['is_promoted'].value_counts()
#Bar chart

train['is_promoted'].value_counts()

ax = sns.countplot(y = train['is_promoted'])

plt.title("Proportion of target variable in the dataset")

plt.show()
#Pie chart

plt.figure(figsize = (8,5))

plt.pie(train['is_promoted'].value_counts(), autopct = '%1.1f%%', labels = ['not promoted', 'is promoted'],

       shadow = True, textprops = {'fontsize':15})

plt.title("Proportion of target variable in the dataset")

plt.show()
train['education'] = train['education'].fillna(train['education'].mode()[0])
train['previous_year_rating'] = train['previous_year_rating'].fillna(-999)
train.isna().sum()
def feature_generator(data):  #generates feature for given data

                              #Returns none

    #Feature

    data['total_training_score'] = data['avg_training_score']*data['no_of_trainings']



    #feature

    data['joining_age'] = data['age'] - data['length_of_service'] 



    #feature

    data['kpi_dept'] = data['department'].astype('str') + '_' + data['KPIs_met >80%'].astype('str') 



    #train = train.set_index('employee_id')



    data['kpi_dept'] = data['kpi_dept'].astype('category')



    #feature

    data['kpi_ats_cat'] = data['avg_training_score'].astype('str') + '_' + data['KPIs_met >80%'].astype('str') 



    #feature

    data['kpi_ats_num'] = data['avg_training_score'] *data['KPIs_met >80%'].astype('int64') 



    #specify categories

    cols = ['department', 'region', 'education', 'gender', 'recruitment_channel', 

            'KPIs_met >80%', 'awards_won?','kpi_dept', 'kpi_ats_cat']

    for i in cols:

        data[i] = data[i].astype('category')



    data['previous_year_rating'] = data['previous_year_rating'].astype('int')
def split_XY(data, smote): #Uses smote to generate new samples if True, else simply splits X and y components of data

                           #Returns X, y

    X = data.drop(['is_promoted'], axis = 1)

    y = data['is_promoted']

    

    #specify categories

    cols = ['department', 'region', 'education', 'gender', 'recruitment_channel', 

            'KPIs_met >80%', 'awards_won?']

    for i in cols:

        data[i] = data[i].astype('category')

        

    if smote == True:

        from imblearn.over_sampling import SMOTENC

        smote_nc = SMOTENC(categorical_features = np.where(X.dtypes == 'category')[0], 

                           random_state=5)

        X_resampled, y_resampled = smote_nc.fit_resample(X, y)

        return(X_resampled, y_resampled)

    

    elif smote == False:

        return(X, y)
def normalize_standardize(data, operation): #If operation = 'normalize', then normalizes data (train)

                                            #If operation = 'standardize', then standardizes data

                                            #Returns none

    from sklearn.preprocessing import StandardScaler, Normalizer

    num_cols = ['no_of_trainings', 'age', 

                      'previous_year_rating', 'length_of_service', 'avg_training_score', 

                      'total_training_score', 'joining_age', 'kpi_ats_num']

    if operation == 'standardize':

        data[num_cols] = StandardScaler().fit_transform(data[num_cols])

    

    if operation == 'normalize':

        data[num_cols] = Normalizer().fit_transform(data[num_cols])

        

    if operation == None:

        data[num_cols] = StandardScaler().fit_transform(data[num_cols])

        
def encoding(data, method): #Encodes the data as per selected 'method' parameter

                            #method : 'mean', 'label', 'onehot'

                            #Returns none

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    le = LabelEncoder()

    #ohe = OneHotEncoder(drop = 'first')



    cat_cols =  ['department', 'region', 

                 'education', 'gender', 'recruitment_channel', 'KPIs_met >80%', 'awards_won?','kpi_dept', 'kpi_ats_cat']

    

    num_cols = ['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'avg_training_score', 

                      'total_training_score', 'joining_age', 'kpi_ats_num']

    

    if method == 'mean':

        for i in cat_cols:

            mean_encoded_subject = data.groupby(i)['is_promoted'].mean().to_dict() 

            data[i] =  data[i].map(mean_encoded_subject)

          



    if method == 'label':

        for i in cat_cols:

            data[i] = le.fit_transform(data[i])

            data[i] = data[i].astype('category')        

    

    if method == 'onehot':

        for i in cat_cols:

            data[i] = le.fit_transform(data[i])

            data[i] = data[i].astype('category')

        data = pd.get_dummies(data, drop_first = True, columns = cat_cols)

        

    return(data)

     
def data_split(data, X , y): #Splits the X and y components of data into training and cross validation set

                             #data if data is unsplit(into X and y) ; X,y if data is split into X and y

                             #Returns X_train, X_cv, y_train, y_cv

    if X is None:

        X_train, X_cv, y_train, y_cv = train_test_split(data.drop('is_promoted', axis = 1), data.is_promoted, test_size = 0.2, shuffle = True)

    elif data is None:

        X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.2, shuffle = True)

        

    print('Shape of X_train',X_train.shape)

    print('Shape of X_cv', X_cv.shape)

    print('Shape of y_train', y_train.shape)

    print('Shape of y_cv',y_cv.shape)

    

    return(X_train, X_cv, y_train, y_cv)
def model_building(algo, X_train, y_train, X_cv, y_cv, learning_rate, folds): 

    #Builds and evaluates model

    #algo : 'xgb', 'lgm', 'etc', 'rfc', 'cbc'

    #learning_rate : the rate at which model is trained

    #folds : number of cross validation folds

    

    from sklearn.metrics import confusion_matrix

    from sklearn.metrics import classification_report

    from sklearn.metrics import average_precision_score



    if algo == 'lgb':

        print("Training LightGBM model..........................................................................")

        param_grid = {

        'learning_rate': np.arange(0.01, 0.11, 0.01),

        'max_depth': np.arange(3, 10, 1),

        'scale_pos_weight': np.arange(5, 50, 5),

        'reg_lambda': np.arange(100, 1000, 100)

                 }



        lgb = LGBMClassifier()

        model = RandomizedSearchCV(

        estimator = lgb,

        param_distributions = param_grid,

        scoring = 'f1',

        verbose = 100,

        n_jobs = -1,

        cv = folds,

        random_state = 3)

        

        model.fit(X_train, y_train, eval_set = (X_cv, y_cv))

        print(model.best_score_)

        print(model.best_estimator_.get_params())

        #lgb_pred = lgb.predict(X_cv)

        

        return(model)

    

    elif algo == 'xgb':

        print("Training XGBoost model..........................................................................")        

        param_grid = {

        'learning_rate': np.arange(0.01, 0.11, 0.01),

        'max_depth': np.arange(3, 10, 1),

        'scale_pos_weight': np.arange(5, 50, 5),

        'n_estimators' : np.arange(100, 1000, 100),

        'colsample_bytree' : np.arange(0.1, 1, 0.1),

        'max_delta_step' : np.arange(0.1, 1, 0.1)

                 }



        xgb = XGBClassifier()

        model = RandomizedSearchCV(

        estimator = xgb,

        param_distributions = param_grid,

        scoring = 'f1',

        verbose = 100,

        n_jobs = -1,

        cv = folds,

        random_state = 3)

        

        model.fit(X_train, y_train)

        print(model.best_score_)

        print(model.best_estimator_.get_params())

        print('Model trained.')

        return(model)

    

    elif algo == 'rfc':

        print("Training Random Forest Classifier model..........................................................................")

        rfc = RandomForestClassifier(random_state = 45)

        param_grid = {

        'min_samples_split': np.arange(2, 15, 1),

        'max_depth': np.arange(3, 10, 1),

        'class_weight': ['balanced', 'balanced_subsample'],

        'criterion' : ['gini', 'entropy'],

        'n_estimators' : np.arange(100, 1000, 100),

                 }

        

        model = RandomizedSearchCV(

        estimator = rfc,

        param_distributions = param_grid,

        scoring = 'f1',

        verbose = 100,

        n_jobs = -1,

        cv = folds,

        random_state = 3)

        

        model.fit(X_train, y_train)

        print(model.best_score_)

        print(model.best_estimator_.get_params())

        print('Model trained.')

        return(model)

    

        

    elif algo == 'etc':

        print("Training Extra Trees model..........................................................................")

        etc = ExtraTreesClassifier()

        param_grid = {

        'min_samples_split': np.arange(2, 15, 1),

        'max_depth': np.arange(3, 10, 1),

        'class_weight': ['balanced', 'balanced_subsample'],

        'criterion' : ['gini', 'entropy'],

        'n_estimators' : np.arange(100, 1000, 100),

                 }

        

        model = RandomizedSearchCV(

        estimator = etc,

        param_distributions = param_grid,

        scoring = 'f1',

        verbose = 100,

        n_jobs = -1,

        cv = folds,

        random_state = 3)

        

        model.fit(X_train, y_train)

        print(model.best_score_)

        print(model.best_estimator_.get_params())

        print('Model trained.')

        return(model)

    

    elif algo == 'cbc':

        print("Training CatBoost model..........................................................................")

        cbc = CatBoostClassifier()



        param_grid = {

        'iterations': np.arange(500, 4000, 500),

        'learning_rate': np.arange(0.01, 0.11, 0.01),

        'max_depth': np.arange(3, 10, 1),

        'l2_leaf_reg': np.arange(1000, 100000, 500),

        'random_strength': np.arange(1, 10),

        'class_weights' :[[4,10], [3, 7], [1,9], [7, 12]]

                 }



        model = RandomizedSearchCV(

        estimator = cbc,

        param_distributions = param_grid,

        scoring = 'f1',

        verbose = 1000,

        n_jobs = -1,

        cv = folds,

        random_state = 3)

        

        model.fit(X_train, y_train, cat_features = np.where(X_train.dtypes == 'category')[0],

                  eval_set = (X_cv, y_cv), plot = True)

        print('Model trained.')

        print(model.best_score_)

        print(model.best_estimator_.get_params())



        return(model)

    

   

def evaluation(model, X_cv, y_cv):

    pred = model.predict(X_cv)

    print("Classification Report:", classification_report(y_cv, pred))

    

    '''

    # define evaluation procedure

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # evaluate model

    scores = cross_val_score(model, X_cv.values, y_cv.values, scoring='f1', cv=cv, n_jobs=-1)

    # summarize performance

    print('Mean F1 score: %.5f' % np.mean(scores))

    '''    

    

    plt.figure(figsize = (5,5))

    sns.heatmap(confusion_matrix(y_cv, pred), annot = True,

                fmt = "d", linecolor = "k", linewidths = 3)

    print("Precision is:", precision_score(y_cv, pred))

    print("Recall is:", recall_score(y_cv, pred))

    plt.autoscale(enable=True, axis='y')

    plt.show()

#Step 1

X, y = split_XY(train, smote = False)
#Step 2

X_train, X_cv, y_train, y_cv = data_split(data = None,X = X, y = y)
#Step 3

feature_generator(X_train)

feature_generator(X_cv)
#Step 4

normalize_standardize(data = X_train, operation = 'normalize')

normalize_standardize(data = X_cv, operation = 'normalize')
#Step 5

#For catboost algorithm, the features don't need to be encoded, so we will create a copy 

#of the data and then apply encoding



X_train_enc = X_train.copy()

X_cv_enc = X_cv.copy()



X_train_enc = encoding(data = X_train_enc, method = 'onehot')



X_cv_enc = encoding(data = X_cv_enc, method = 'onehot')

print(X_train.columns)

print(X_train_enc.columns)

print(X_train_enc.shape)
y_train.value_counts().plot(kind = 'pie', autopct = '%1.1f%%')
y_cv.value_counts().plot(kind = 'pie', autopct = '%1.1f%%')
if 'employee_id' in X_train.columns and 'employee_id' in X_train_enc.columns and 'employee_id' in X_cv.columns and 'employee_id' in X_cv_enc.columns:

    X_train = X_train.drop('employee_id', axis = 1)

    X_train_enc = X_train_enc.drop('employee_id', axis = 1)

    X_cv = X_cv.drop('employee_id', axis = 1)

    X_cv_enc = X_cv_enc.drop('employee_id', axis = 1)
print(list(set(X_train_enc) - set(X_cv_enc)))
#Since cv data doesnt contain these 2 features, we will drop them from train data 



#if 'kpi_ats_cat_119' in X_train_enc.columns:

#    X_train_enc = X_train_enc.drop('kpi_ats_cat_119', axis = 1)

    

#if 'kpi_ats_cat_120' in X_train_enc.columns:

    #X_train_enc = X_train_enc.drop('kpi_ats_cat_120', axis = 1)

    

X_train_enc = X_train_enc.drop(list(set(X_train_enc) - set(X_cv_enc)), axis = 1)
set(X_train_enc.columns) - set(X_cv_enc.columns)
xgb = model_building('xgb', X_train_enc, y_train, X_cv_enc, y_cv, learning_rate = 0.05, folds = 3)

evaluation(xgb, X_cv_enc, y_cv)
lgb = model_building('lgb', X_train_enc, y_train, X_cv_enc, y_cv, learning_rate = 0.05, folds = 3)

evaluation(lgb, X_cv_enc, y_cv)
etc = model_building('etc', X_train_enc, y_train, X_cv_enc, y_cv, learning_rate = 0.05, folds = 3)

evaluation(etc, X_cv_enc, y_cv)
rfc = model_building('rfc', X_train_enc, y_train, X_cv_enc, y_cv, learning_rate = 0.05, folds = 3)

evaluation(rfc, X_cv_enc, y_cv)
cbc = model_building('cbc', X_train, y_train, X_cv, y_cv, learning_rate = 0.05, folds = 3)

evaluation(cbc, X_cv, y_cv)
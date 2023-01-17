# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import skew

from sklearn.metrics import log_loss

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

#from statsmodels.discrete.discrete_model import Logit

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df





def import_data(file):

    """create a dataframe and optimize its memory usage"""

    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)

    df = reduce_mem_usage(df)

    return df
print('-' * 80)

print('train')

df = import_data('../input/heart.csv')

#df = pd.read_csv("../input/heart.csv")
df.info()
df.apply(lambda x: sum(x.isnull()))
df.head(3)
df.describe()
plt.figure(figsize=(14,10))

sns.heatmap(df.corr(), annot=True, linewidths=0.5, cmap='magma')
# 2 categories

df['sex'] = df['sex'].replace({1: 'male',

                              0: 'female'})

df['exang'] = df['exang'].replace({1: 'yes',

                                 0: 'no'})

df['fbs'] = df['fbs'].replace({1: 'over 120',

                              0: 'under 120'})

# more than 2 categories

df['cp_cat'] = df['cp'].replace({0: '1st type',

                            1: '2st type',

                            2: '3st type',

                            3: '4st type'})



df['restecg_cat'] = df['restecg'].replace({0: 'normal',

                                      1: 'ST-T wave abnormality',

                                      2: 'left ventricular hypertrophy'})

df['slope_cat'] = df['slope'].replace({0: 'upsloping',

                                  1: 'flat',

                                  2: 'downsloping'})

#df['target'] = df['target'].replace({0: 'no', 1: 'yes'})

# category

df['ca_cat'] = df['ca'].astype('category')

df['thal_cat'] = df['thal'].astype('category')

df['sex'] = df['sex'].astype('category')

df['cp_cat'] = df['cp_cat'].astype('category')

df['fbs'] = df['fbs'].astype('category')

df['restecg_cat'] = df['restecg_cat'].astype('category')

df['exang'] = df['exang'].astype('category')

df['slope_cat'] = df['slope_cat'].astype('category')

old_columns = ['cp', 'restecg', 'slope', 'ca', 'thal']
def bar_plot(columns):

    v = df[columns][df['target'] == 1].value_counts(normalize=True)

    f, (ax1,ax2) = plt.subplots(1,2, figsize=(12,6))

    sns.barplot(x=v.index, y=v.values, ax=ax1, palette='Set1')

    ax1.set_ylim(0,1)

    sns.countplot(x=df[columns], ax=ax2, palette='Set1')

    plt.suptitle('Category Column: {}'.format(columns), size=20)
def category_plot(column, order=None):

    v = df[column].value_counts(normalize=True)

    f, (ax1,ax2) = plt.subplots(1,2, figsize=(10,6))

    #1

    sns.countplot(x=column, data=df, order=order, label='Whole population',

                  color='blue', ax=ax1)

    sns.countplot(x=column, data=df[df['target'] == 1], order=order,

                  label='Participation of 1 target population', color='red', ax=ax1)

    #2

    sns.barplot(x=v.index, y=v.values, ax=ax2, palette='Set1', order=order)

    ax1.tick_params(rotation=60)

    ax2.tick_params(rotation=60)

    ax2.set_ylim(0,1)

    ax1.legend(loc='best')

    ax1.set_xlabel('')

    ax1.set_title('Appearance of heart diseases')

    ax2.set_title('Distribution of category')

    plt.suptitle('Category column: {}'.format(column))

    plt.show()
cat_col = df.select_dtypes(include='category')

for c in cat_col:

    category_plot(c)
num_cols = df.select_dtypes(include='number').columns

num_cols = [c for c in num_cols if c in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]
def plot_hist(column):

    plt.figure(figsize=(14,10))

    plt.hist(df[column], bins=20, edgecolor='black', label='Whole group', color='black')

    plt.hist(df[column][df['target'] == 1], bins=20, color='red', edgecolor='black', histtype='stepfilled', alpha=0.8,label='1 target group')

    plt.hist(df[column][df['target'] == 0], bins=20, color='yellow', edgecolor='black', histtype='stepfilled', alpha=0.8,label='0 target group')

    plt.title('Histogram of {}'.format(column))

    plt.legend()

    skeww = round(skew(df[column]),4)

    #plt.text(65, 25, s='Skewness: {}'.format(skeww), size=12)

    print('Skewness:',column, round(skew(df[column]),4))
for n in num_cols:

    plot_hist(n)
for o in old_columns:

    plot_hist(o)
def prepare_data(test_size, scale=None, get_dummies=None, drop_first=None):

    # X and y

    y = df['target']

    X = df.drop(col_to_drop, axis=1)

    

    # dummies

    if get_dummies:

        X = pd.get_dummies(X, drop_first=drop_first)

        

    # TRAIN TEST

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, y, test_size=test_size, random_state=1)

    # SCALING

    if scale:

        scaler = StandardScaler()

        X_TRAIN = scaler.fit_transform(X_TRAIN)

        X_TEST = scaler.fit_transform(X_TEST)

        

    return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
def model_perfomance(model, text, coef=None, importance=None):

    model.fit(X_TRAIN, Y_TRAIN)

    Y_PRED = model.predict(X_TEST)

    Y_PRED_PROBA = model.predict_proba(X_TEST)

    cnf = confusion_matrix(Y_TEST,Y_PRED)

    cnf = pd.DataFrame(cnf)

    acc = accuracy_score(Y_TEST, Y_PRED)

    rec = recall_score(Y_TEST, Y_PRED)

    prec = precision_score(Y_TEST, Y_PRED)

    roc = roc_auc_score(Y_TEST, Y_PRED_PROBA[:,1])

    print('Accuracy: ', accuracy_score(Y_TEST, Y_PRED))

    print('Recall: ', recall_score(Y_TEST, Y_PRED))

    print('Precision: ', precision_score(Y_TEST, Y_PRED))

    print('ROC AUC', roc)

    bar = pd.DataFrame(index=['Measure', 'Value'],

                      data=[['Accuracy', 'Recall', 'Precision'],[acc,rec,prec]])

    bar = bar.T

    plt.figure(figsize=(15,15))

    plt.suptitle(text, fontsize=18)

    plt.subplot(311)

    sns.heatmap(cnf.T, annot=True)

    plt.xlabel('Actual')

    plt.ylabel('Predicted')

    plt.subplot(312)

    sns.barplot(y=bar['Measure'], x=bar['Value'], edgecolor='black')

    if coef:

        coef = pd.DataFrame(columns=X_TRAIN.columns, data=model.coef_)

        print(model.coef_.shape)

        plt.subplot(313)

        sns.barplot(x=coef.T.index, y=coef.T[0])

        plt.ylabel('Coefficients')

        plt.xticks(rotation=90)

    if importance:

        #coef = pd.DataFrame(columns=X_TRAIN.columns, data=np.transpose(model.feature_importances_))

        print(np.transpose(model.feature_importances_).shape)

        plt.subplot(313)

        sns.barplot(x=X_TRAIN.columns,y=model.feature_importances_)

        plt.xticks(rotation=90)

    return model

        
def grid(estimator, params, cv):

    #data

    y = df['target']

    X = df.drop(col_to_drop, axis=1)

    # dummies

    X = pd.get_dummies(X, drop_first=True)



    cv = GridSearchCV(estimator=estimator, param_grid=params, cv=cv)

    cv.fit(X,y)

    #

    print(cv.best_params_)

    return cv.best_estimator_

    
def estimation():

    logit = LogisticRegression(max_iter=10000, solver='lbfgs', random_state=1)

    rf1 = RandomForestClassifier(random_state=1)



    params_l = {'C': np.linspace(0.1,1,10)}

    params_r = {'max_depth' : [2,3,4,5],

               'max_features' : [2, 3, 5],

               'n_estimators': [400,600,800]}



    model_logit = grid(logit, params_l, 3)

    model_rf = grid(rf1, params_r, 3)

    print('Logit model')

    logit_clf = model_perfomance(model_logit, 'Logistic regression', coef=True, importance=False)

    print('Random forest model')

    rf_clf = model_perfomance(model_rf, 'Random forest model', coef=False, importance=True)
col_to_drop = ['target', 'cp', 'restecg','slope','ca', 'thal']

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = prepare_data(0.33, scale=False, get_dummies=True, drop_first=True)

estimation()
col_to_drop = ['target', 'cp_cat', 'restecg_cat','slope_cat','ca_cat', 'thal_cat']

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = prepare_data(0.33, scale=False, get_dummies=True, drop_first=True)

estimation()
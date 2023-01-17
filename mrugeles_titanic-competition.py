# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import warnings

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

from category_encoders.ordinal import OrdinalEncoder    

from sklearn.externals import joblib



plt.style.use('seaborn-whitegrid')

%matplotlib inline

warnings.filterwarnings('ignore')



dataset = pd.read_csv('/kaggle/input/titanic/train.csv')

print(dataset.shape)

dataset.head()

dataset['Fare'].plot(kind = 'hist')
dataset[['Age', 'Fare']].plot(kind = 'scatter', x = 'Age', y = 'Fare');
import re

from enum import Enum

regex = re.compile('[^\sa-zA-Z]')

age_mean = {}

'''

(['Mr', 'Mrs', 'Miss', 'Master', 'Planke', 'Don', 'Rev', 'Billiard',

       'der', 'Walle', 'Dr', 'Pelsmaeker', 'Mulder', 'y', 'Steen',

       'Carlo', 'Mme', 'Impe', 'Ms', 'Major', 'Gordon', 'Messemaeker',

       'Mlle', 'Col', 'Capt', 'Velde', 'the', 'Shawah', 'Jonkheer',

       'Melkebeke', 'Cruyssen']

Miss = Mlle, Ms 

Mr = Mme, Planke, Don, Rev, Dr, Major, Col, Capt



'''

class ImputerStrategy(Enum):

    MEAN = 'mean'

    MEDIAN = 'median'

    MODE = 'mode'

    CONSTANT = 'constant'

    REGRESSOR_MODEL = 'regressor_model'

    CLASSIFICATION_MODEL = 'clasification_model'

    

fill_mean = lambda col: col.fillna(col.mean())

fill_median = lambda col: col.fillna(col.median())

fill_mode = lambda col: col.fillna(col.mode()[0])



impute_strategies = {

    ImputerStrategy.MEAN: fill_mean,

    ImputerStrategy.MEDIAN: fill_median,

    ImputerStrategy.MODE: fill_mode



}



    

def impute(dataset, impute_strategy):

    if impute_strategy in [ImputerStrategy.MEAN, ImputerStrategy.MEDIAN, ImputerStrategy.MODE]:

        return dataset.apply(impute_strategies[impute_strategy], axis=0)

    else:

        return dataset



def process_title(title):

    if(title in ['Mr', 'Mme', 'Planke', 'Don', 'Rev', 'Dr', 'Major', 'Col', 'Capt']):

        return 'Mr'

    if(title in ['Miss', 'Mlle', 'Ms']):

        return 'Miss'

    if(title in ['Master', 'Mrs']):

        return title

    return '999'

    

def one_hot_encode(df, column):

    df = df.join(pd.get_dummies(df[column], prefix=column))

    return df.drop([column], axis = 1)  



def label_encode(df, column):

    le = preprocessing.LabelEncoder()

    values = list(df[column].values)

    le.fit(values)

    df[column] = le.transform(values)

    return df



def ordinal_encode(df, column, map_list):

    mapping_dict = [{'col': column, 'mapping': map_list}]

    enc = OrdinalEncoder(mapping = mapping_dict)



    return enc.fit_transform(df[column])



def scale_normalize(df, columns):

    df[columns] = MinMaxScaler().fit_transform(df[columns])

    for column in columns:

        df[column] = df[column].apply(lambda x: np.log(x + 1))

    return df



def categorize(df, column, bins):

    data = pd.cut(np.array(df[column]),  bins=bins)

    data = pd.Series(data)

    data = pd.DataFrame(data, columns=[f'{column}_Range'])

    data = data[f'{column}_Range'].apply(lambda value: str(value).replace('(', '').replace(']', '').replace(', ', '_'))



    df = df.join(pd.DataFrame(data, columns=[f'{column}_Range']))

    df = df.join(pd.get_dummies(df[f'{column}_Range']))

    df = df.drop([column], axis = 1)

    return df.drop([f'{column}_Range'], axis = 1)



def clean_ticket(text):

    text = text.lower()

    return re.sub(r"[^a-zA-Z0-9 ]", "", text)



def impute_age(df, row):

    age_mean = df.groupby(['SibSp', 'Parch'])['Age'].mean().to_dict()

    age = row['Age']

    if(np.isnan(age)):

        age = age_mean[(row['SibSp'], row['Parch'])]

    return age   



def categorize_age(df):

    data = pd.cut(df['Age'].values,  bins=[0, 10, 20, 30, 40,50, 60, 70, 80, 90, 100])

    data = pd.Series(data)

    age_distribution = pd.DataFrame(data, columns=['AgeRange'])

    age_distribution['Age'] = df['Age']

    return age_distribution['AgeRange']



def scale_range(X):

    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))



def scale_max_clipping(X, clip):

    return X.apply(lambda x: x if x < clip else clip)



def scale_min_clipping(X, clip):

    return X.apply(lambda x: x if x > clip else clip)



def scale_log(X):

    return X.apply(lambda x: np.log(x + 1))



def scale_z_score(X):

    return X.apply(lambda x: (x - X.mean() / X.std()))



def normalizers(X, clip = 10):

    fig, ax = plt.subplots(1, 5, figsize=(17, 2))

    X_range = scale_range(X)

    X_clipped = scale_max_clipping(X, clip)

    X_log_scaled = scale_log(X)

    X_z_score = scale_z_score(X)



    ax[0].hist(X)

    ax[1].hist(X_range)

    ax[1].set_title('Scale Range')

    ax[2].hist(X_clipped)

    ax[2].set_title('Clipped')

    ax[3].hist(X_log_scaled)

    ax[3].set_title('Log')

    ax[4].hist(X_z_score)

    ax[4].set_title('Z-Score')

    

def process_fare(fare):

    #[(-0.001, 7.91] < (7.91, 14.454] < (14.454, 31.0] < (31.0, 512.329]]

    if(fare <= 7.91):

        return 0

    if(fare > 7.91 and fare <= 14.454):

        return 1

    if(fare > 14.454 and fare <= 31):

        return 2

    if(fare > 31 and fare <= 99):

        return 3

    if(fare > 99 and fare <= 250):

        return 4

    return 5



def get_ticket_count(df):

    ticket_count_df = df.groupby('Ticket')['PassengerId'].count().to_frame(name = 'TicketCount')

    return ticket_count_df.reset_index(level = 0, inplace = True)





def pre_process(dataset):

    df = dataset.copy()

    df = df.drop(['PassengerId', 'Cabin'], axis = 1)

    df['Age'] = impute(df[['Age']], ImputerStrategy.MEAN)

    df['Fare'] = impute(df[['Fare']], ImputerStrategy.MEAN)

    df['Relatives'] = df['SibSp'] + df['Parch']

    

    df = categorize(df, 'Age', [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])



    df = one_hot_encode(df, 'Embarked')

    df = one_hot_encode(df, 'Sex')

    

    df['Relatives'] = df['SibSp'] + df['Parch']

    #df['PclassMale'] = df['Pclass'] + df['Sex_male']

    #df['PclassFemale'] = df['Pclass'] + df['Sex_female']



    df['title'] = df['Name'].apply(lambda name: name.split(' ')[1])



    df = label_encode(df, 'title')

    df = label_encode(df, 'Ticket')



    df = df.drop(['Name'], axis = 1)

    if('Survived' in list(df.columns)):

        df = df.drop_duplicates()



    columns = ['Pclass', 'SibSp', 'Parch', 'Ticket', 'Fare', 'title', 'Relatives']

    df = scale_normalize(df, columns)

    return df





def plot_missing_values(dataset):

    zero_df = dataset.isnull().mean().to_frame(name = 'missing')

    non_zero_df = dataset.notnull().mean().to_frame(name = 'not_missing')

    missing_values = pd.concat([zero_df, non_zero_df], axis=1, sort=False)

    missing_values['total'] = missing_values['missing'] + missing_values['not_missing'] 

    print(zero_df[zero_df['missing'] > 0])

    

    plt.figure(figsize=(15,5))



    features = list(missing_values.index)

    width = .7

    ind = np.arange(len(features)) 



    plot_missing = plt.bar(ind, missing_values['missing'], width, color = 'red', alpha = 0.3)

    plot_not_missing = plt.bar(ind, missing_values['not_missing'], width, color = 'green', alpha = 0.3, bottom=missing_values['missing'])



    plt.ylabel('%')

    plt.title('% Missing values')

    plt.xticks(ind, tuple(features))

    plt.yticks(np.arange(0, .8, 1))

    plt.legend((plot_missing[0], plot_not_missing[0]), ('Missing', 'Not missing'), bbox_to_anchor=(1.05, 1), borderaxespad=0., loc='upper right')

    plt.xticks(rotation=45, ha='right')



    plt.show()
processed_df = pre_process(dataset)

display(processed_df.head())





from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.linear_model import RidgeClassifier 

from sklearn.linear_model import RidgeClassifierCV 

from sklearn.linear_model import SGDClassifier 

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import ExtraTreeClassifier 

from sklearn.neighbors import KNeighborsClassifier

from mlxtend.classifier import MultiLayerPerceptron



from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import ExtraTreesClassifier

    

from time import time

from sklearn.metrics import accuracy_score, confusion_matrix

import xgboost as xgb



def init_classifiers(seed):

    return {

        'AdaBoostClassifier': AdaBoostClassifier(random_state = seed),

        'BaggingClassifier': BaggingClassifier(random_state = seed),

        'ExtraTreesClassifier': ExtraTreesClassifier(random_state = seed),

        'GradientBoostingClassifier': GradientBoostingClassifier(random_state = seed),

        'RandomForestClassifier': RandomForestClassifier(random_state = seed),

        'HistGradientBoostingClassifier': HistGradientBoostingClassifier(random_state = seed),

        'XGBClassifier': xgb.XGBClassifier(),

        'LogisticRegression': LogisticRegression(random_state = seed),

        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(random_state = seed),

        'RidgeClassifier': RidgeClassifier(random_state = seed),

        'RidgeClassifierCV': RidgeClassifierCV(),

        'SGDClassifier': SGDClassifier(random_state = seed),

        'KNeighborsClassifier': KNeighborsClassifier(),

        #'RadiusNeighborsClassifier': RadiusNeighborsClassifier(),

        'MLPClassifier': MLPClassifier(random_state = seed),

        'DecisionTreeClassifier': DecisionTreeClassifier(random_state = seed),

        'ExtraTreeClassifier': ExtraTreeClassifier(random_state = seed),

    }



def train_predict(learner, beta_value, X_train, y_train, X_test, y_test, dfResults):

    start = time()

    learner = learner.fit(X_train, y_train)

    end = time()



    train_time = end - start



    start = time()

    predictions_test = learner.predict(X_test)

    predictions_train = learner.predict(X_train)

    end = time() # Get end time



    pred_time = end - start



    f_train = accuracy_score(y_train, predictions_train)



    f_test =  accuracy_score(y_test, predictions_test)

    

    dfResults = dfResults.append({'learner': learner.__class__.__name__, 'train_time': train_time, 'pred_time': pred_time, 'f_test': f_test, 'f_train':f_train}, ignore_index=True)

    return learner, dfResults, predictions_test



def tune_classifier(clf, parameters, X_train, X_test, y_train, y_test):

    scorer = make_scorer(accuracy_score)

    grid_obj = GridSearchCV(clf, param_grid=parameters, scoring=scorer, iid=False)

    grid_fit = grid_obj.fit(X_train, y_train)

    best_clf = grid_fit.best_estimator_

    

    predictions = (clf.fit(X_train, y_train)).predict(X_test)

    best_predictions = best_clf.predict(X_test)



    default_score = accuracy_score(y_test, predictions)

    tuned_score = accuracy_score(y_test, best_predictions)



    cnf_matrix = confusion_matrix(y_test, best_predictions)

    

    return best_clf, default_score, tuned_score, cnf_matrix
seed = 300

test_size = 0.25

    

def split_data(df):

    from sklearn.model_selection import train_test_split

    features = processed_df.drop(['Survived'], axis = 1)

    labels = processed_df['Survived']





    return train_test_split(features, labels, test_size = test_size, random_state = seed, stratify=labels)



def train_models(X_train, X_test, y_train, y_test):

    classifiers = init_classifiers(seed)



    # Collect results on the learners

    df = pd.DataFrame(columns=['learner', 'train_time', 'pred_time', 'f_test', 'f_train'])

    predictions = y_test.to_frame('predictions')



    for clf in list(classifiers.values()):

        clf, df, predictions_test = train_predict(clf, 2, X_train, y_train, X_test, y_test, df)

        joblib.dump(clf, f'{clf.__class__.__name__}.joblib')

        predictions[clf.__class__.__name__] = predictions_test



    return df.sort_values(by=['f_test'], ascending = False), predictions





X_train, X_test, y_train, y_test = split_data(processed_df)

results, predictions = train_models(X_train, X_test, y_train, y_test)

display(results)
colors = ['#fdbb6c', '#b3df72'] 
from sklearn.ensemble import GradientBoostingClassifier



classifier = GradientBoostingClassifier(random_state = seed)

print(classifier)



parameters = {

   

}



classifier, default_score, tunned_score, mmatrix = tune_classifier(classifier, parameters, X_train, X_test, y_train, y_test)



print("Unoptimized model score: {:.4f}".format(default_score))

print("Optimized model score: {:.4f}".format(tunned_score))

#joblib.dump(ml_classifier, 'ml_classifier.joblib')

print(classifier)
processed_df.head()
from mlxtend.classifier import StackingCVClassifier

from sklearn import model_selection



clf1 = BaggingClassifier(random_state = seed)

clf2 = GradientBoostingClassifier(random_state = seed)

clf3 = RandomForestClassifier()

lr = LogisticRegression()



X = processed_df.drop(['Survived'], axis = 1)

y = processed_df['Survived']



sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],

                            use_probas=True,

                            meta_classifier=lr,

                            random_state=42)



df = pd.DataFrame(columns=['learner', 'train_time', 'pred_time', 'f_test', 'f_train'])



sclf, df, predictions_test = train_predict(sclf, 2, X_train, y_train, X_test, y_test, df)



df
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

test = pre_process(test_df)



predictions = sclf.predict(test)



predictions_df = test_df[['PassengerId']]

predictions_df['Survived'] = predictions



predictions_df.to_csv('predictions.csv', index = False)

from IPython.display import FileLink

FileLink('predictions.csv')
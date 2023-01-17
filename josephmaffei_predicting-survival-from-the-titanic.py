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
#look over training data

training_data = pd.read_csv('/kaggle/input/titanic/train.csv')

display(training_data.head())



#inspect class distribution in training data

print('Number of entries in either survived \'1\' or didn\'t survive \'0\' group')

print(training_data.groupby('Survived').count()['Pclass'])
print('Number of missing entries for each feature')

print(training_data.isnull().sum())
print('Number of unique entries for each feature')

print(training_data.nunique())
interesting_feats = ['Age', 'PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin']

training_data[interesting_feats].iloc[:15, :]
print('Value Counts for feature \'Cabin\'')

print(training_data.Cabin.value_counts()[:50])

print('\nValue Counts for feature \'Ticket\'')

print(training_data.Ticket.value_counts()[:50])

import re

#create list of letters that cabin identifiers begin with

Cabin_Letter = training_data.Cabin.fillna('').astype('str')

unique_letters = re.findall("[A-Z]", "".join(set("".join(Cabin_Letter))))



#create new feature "Cabin_Letter" that shows the letter the cabin room starts with.  

for char in range(len(unique_letters)):

    mask = Cabin_Letter.str.startswith(unique_letters[char])

    Cabin_Letter.where(~mask, unique_letters[char], inplace = True)



Cabin_Letter.replace('', 'Unknown', inplace = True)    

training_data["Cabin_Letter"] = Cabin_Letter

print('First 10 rows of new feature \'Cabin Letter\'')

print(training_data.Cabin_Letter[:10])

print('\nValue counts for feature \'Cabin Letter\'')

print(training_data.Cabin_Letter.value_counts())

print('\nUnique entries for feature \'Cabin Letter\'')

print(training_data.Cabin_Letter.unique())
training_data["Ticket_Letter"] = training_data.Ticket.str.contains('[a-zA-Z]').astype(int)

print('Number of entries that do not contain \'0\' and do contain \'1\' a letter in the ticket field')

print(training_data.Ticket_Letter.value_counts())
#find symbols and letters in ticket names

symbols = ''.join(re.findall("[\WA-Za-z]", "".join(set("".join(training_data.Ticket)))))

print("Symbols and letters contained in ticket names:", symbols)



#remove symbols and letters from ticket names

Stripped_Ticket = training_data.Ticket.str.replace("[\WA-Za-z]", '')



#replace empty ticket fields with zero to allow for int conversion

m = Stripped_Ticket == ''

Stripped_Ticket.where(~m, '0', inplace = True)



#check to make sure there are no remaining non-numeric characters

print('Remaining number of entries that contain non-numeric character:',

      Stripped_Ticket.str.contains("[\WA-Za-z]").sum())



#convert data type to int

training_data["Stripped_Ticket"] = Stripped_Ticket.astype(int)



#replace zero values with median ticket number

median_value = np.median(training_data.Stripped_Ticket[training_data.Stripped_Ticket != 0])

training_data.Stripped_Ticket.replace(0, median_value, inplace = True)
#create new feature measuring which quantile the ticket number belongs in

cut = pd.qcut(training_data.Stripped_Ticket, 5, labels = False)

training_data["Ticket_Quant"] = cut



#view new set of features

training_data.head()
#set passegerId to index

training_data.set_index(['PassengerId'], inplace = True)



#create new series to hold survival outcomes

survived = training_data.Survived



#drop unwanted features

training_data_pared = training_data.drop(['Name', 'Ticket', 'Cabin', 'Stripped_Ticket', 'Survived'], axis = 1)



#reformat sex feature with binary value

training_data_pared.replace({'female': 0, 'male': 1}, inplace = True)



training_data_pared.head()
!pip install category_encoders;
# import sklearn modules for generating the model

import warnings

warnings.filterwarnings("ignore", category= FutureWarning)

import category_encoders as ce

from sklearn.impute import SimpleImputer

from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.feature_selection import RFECV

from sklearn.preprocessing import StandardScaler, normalize, Normalizer

from sklearn.metrics import classification_report, f1_score, roc_curve, roc_auc_score, accuracy_score

from sklearn.svm import LinearSVC

from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline, Pipeline
#PCA decompose data into 2 principle components and then plot them on a scatter plot



#Confirm format of null values in data set for imputation

print(training_data_pared.Embarked[training_data_pared.Embarked.isnull()])

print(training_data_pared.Age[training_data_pared.Age.isnull()])
#Quick encoding of categorical variables (potentially not optimal but we need to pick something for this overview of the data)

encoder = ce.TargetEncoder(cols = ['Embarked', 'Cabin_Letter'], return_df = True)

encoded_data = encoder.fit_transform(training_data_pared, survived)



#impute missing values for 'Age' -- the encoding takes care of the missing values for 'Embarked'

imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')

enc_imp_data = pd.DataFrame(imp.fit_transform(encoded_data), columns = encoded_data.columns, 

                            index = encoded_data.index)



#standarize data to avoid over-contribution from 'Age' feature

sc = StandardScaler()

stand_data = pd.DataFrame(sc.fit_transform(enc_imp_data), columns = enc_imp_data.columns,

                          index = enc_imp_data.index)



#PCA transform

pca = PCA(n_components = 2, svd_solver = 'full', random_state = 50)

pca_data = pd.DataFrame(pca.fit_transform(stand_data), columns = ['Component_1', 'Component_2'],

                        index = stand_data.index)

pca_data = pd.concat([pca_data, survived], axis = 1)

print("Number of components = ", pca.n_components_)

print("Variance explained = ", np.sum(pca.explained_variance_ratio_))

print("\n",pca_data.head())
#import plotting libraries

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline



#plot PCA decomposed data

sns.scatterplot(x = pca_data.iloc[:,0], y = pca_data.iloc[:,1], hue = pca_data.iloc[:,2], data = pca_data);

plt.title("PCA Decomposed Titanic Data", fontsize = 18);
#instantiate encoders

onehot = ce.OneHotEncoder(cols = ['Embarked', 'Cabin_Letter'], return_df = True)

loo = ce.LeaveOneOutEncoder(cols = ['Embarked', 'Cabin_Letter'], return_df = True)



#instantiate normalizer and imputer

norm = Normalizer()

imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')



#instantiate cross-validation strategy

skf = StratifiedKFold(n_splits = 7)



#instantiate estimators

kmeans = KMeans(n_clusters = 2, random_state = 42)

forest = RandomForestClassifier(random_state = 100, max_features = None, min_samples_split = 6)

svm = LinearSVC(dual = False, random_state = 50)



#create lists of encoders and models

model = [kmeans, forest, svm]

encode = [onehot, loo]
#Design function that will create dictionary of parameters for each run through GridSearchCV



def make_param_dict(encoder, model_type):

    param_dict = {}        

    

    if type(model_type).__name__ == 'KMeans':

        param_dict['kmeans__n_init'] = np.arange(10, 30, 1)

        

    elif type(model_type).__name__ == 'RandomForestClassifier':

        param_dict['randomforestclassifier__n_estimators'] = np.arange(10, 100, 10)

        param_dict['randomforestclassifier__min_samples_leaf'] = np.arange(1, 10, 1)

    

    elif type(model_type).__name__ == 'LinearSVC':

        param_dict['linearsvc__C'] = np.arange(0.01, 10, 0.5)

    

    return param_dict
grid_results = []



for enc in encode:

    for mod in model:

        param_grid = make_param_dict(enc, mod)



        pipe = make_pipeline(enc, imp, norm, mod)



        gscv = GridSearchCV(pipe, param_grid = param_grid, scoring = 'accuracy', cv = skf, 

                           return_train_score = True)



        print("Working on finding best parameters for encoder: {} and model: {}".format(

            type(enc).__name__, type(mod).__name__))



        gscv.fit(training_data_pared, survived)

        

        print("Done")

        

        grid_results.append(gscv)

scores = [s.best_score_ for s in grid_results]

gscv_df = pd.DataFrame(grid_results[np.argmax(scores)].cv_results_)

mask = gscv_df.rank_test_score == 1

display(gscv_df[mask])
nes_mask = (gscv_df.param_randomforestclassifier__min_samples_leaf == 5)



nes_scores_train = gscv_df[nes_mask].mean_train_score

nes_scores_test = gscv_df[nes_mask].mean_test_score

nes_values = gscv_df[nes_mask].param_randomforestclassifier__n_estimators



msl_mask = (gscv_df.param_randomforestclassifier__n_estimators == 50)



msl_scores_train = gscv_df[msl_mask].mean_train_score

msl_scores_test = gscv_df[msl_mask].mean_test_score

msl_values = gscv_df[msl_mask].param_randomforestclassifier__min_samples_leaf



fig, ax = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize = [15,5]);

ax[0].plot(nes_values, nes_scores_train, '-o', label = 'train');

ax[0].plot(nes_values, nes_scores_test, '-^', label = 'test');

ax[0].set_xlabel('number of estimators');

ax[0].set_ylabel('Accuracy');

ax[0].legend();

ax[1].plot(msl_values, msl_scores_train, '-o', label = 'train');

ax[1].plot(msl_values, msl_scores_test, '-^', label = 'test');

ax[1].set_xlabel('min samples per leaf');

ax[1].legend();
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head()
#create list of letters that cabin identifiers begin with

Cabin_Letter_test = test_data.Cabin.fillna('').astype('str')

unique_letters = re.findall("[A-Z]", "".join(set("".join(Cabin_Letter_test))))



#create new feature "Cabin_Letter" that shows the letter the cabin room starts with.  

for char in range(len(unique_letters)):

    mask = Cabin_Letter_test.str.startswith(unique_letters[char])

    Cabin_Letter_test.where(~mask, unique_letters[char], inplace = True)



Cabin_Letter_test.replace('', 'Unknown', inplace = True)    

test_data["Cabin_Letter"] = Cabin_Letter_test
test_data["Ticket_Letter"] = test_data.Ticket.str.contains('[a-zA-Z]').astype(int)
#find symbols and letters in ticket names

symbols = ''.join(re.findall("[\WA-Za-z]", "".join(set("".join(test_data.Ticket)))))

print("Symbols and letters contained in ticket names:", symbols)



#remove symbols and letters from ticket names

Stripped_Ticket = test_data.Ticket.str.replace("[\WA-Za-z]", '')



#replace empty ticket fields with zero to allow for int conversion

m = Stripped_Ticket == ''

Stripped_Ticket.where(~m, '0', inplace = True)



#check to make sure there are no remaining non-numeric characters

print('Remaining number of entries that contain non-numeric character:',

      Stripped_Ticket.str.contains("[\WA-Za-z]").sum())



#convert data type to int

test_data["Stripped_Ticket"] = Stripped_Ticket.astype(int)



#replace zero values with median ticket number

median_value = np.median(test_data.Stripped_Ticket[test_data.Stripped_Ticket != 0])

test_data.Stripped_Ticket.replace(0, median_value, inplace = True)
#create new feature measuring which quantile the ticket number belongs in

cut = pd.qcut(test_data.Stripped_Ticket, 5, labels = False)

test_data["Ticket_Quant"] = cut



#view new set of features

test_data.head()
#set passegerId to index

test_data.set_index(['PassengerId'], inplace = True)



#drop unwanted features

test_data_pared = test_data.drop(['Name', 'Ticket', 'Cabin', 'Stripped_Ticket'], axis = 1)



#reformat sex feature with binary value

test_data_pared.replace({'female': 0, 'male': 1}, inplace = True)
test_data_pared.head()
gscv_fitter = grid_results[np.argmax(scores)]

pred_df = pd.DataFrame(gscv_fitter.predict(test_data_pared), index = test_data_pared.index,

                       columns = ['Survived'])

display(pred_df.head())

print('Number of predicted survivors: ', pred_df.sum().values)
pred_df.to_csv('predictions.csv')
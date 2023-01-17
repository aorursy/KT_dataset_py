#Data Manipulation

import pandas as pd

import numpy as np

import os



#Visualization

import matplotlib.pyplot as plt

import seaborn as sns





# Other Packages

import missingno as msno



# Set a few plotting defaults

%matplotlib inline

plt.style.use('fivethirtyeight')

plt.rcParams['font.size'] = 18

plt.rcParams['patch.edgecolor'] = 'k'
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train_samp = train.sample(frac=.3)
y = train_samp['Target']

y_full = train['Target']
y.value_counts(normalize=True)
print(f'Train shape: {train_samp.shape}')

print(f'Test shape: {test.shape}')
print(train.info())

train.columns[1::]
train_samp.select_dtypes('object')

len(train.columns)
msno.matrix(train)
train_samp.isnull().sum()

#v2a1, v18q1, 

train_samp.columns[train_samp.isnull().any()]
train_samp.select_dtypes('int64')

train_samp.get_dtype_counts()
mapping = {"yes": 1, "no": 0}



# Apply same operation to both train and test

for df in [train, test]:

    # Fill in the values with the correct mapping

    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)

    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)

    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)



train[['dependency', 'edjefa', 'edjefe']].describe()
train_samp = train.sample(frac=.3)
train.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color = 'blue', 

                                                                             figsize = (8, 6),

                                                                            edgecolor = 'k', linewidth = 2);

plt.xlabel('Number of Unique Values'); plt.ylabel('Count');

plt.title('Count of Unique Values in Integer Columns');
#Plot densities of float columns

from collections import OrderedDict



plt.figure(figsize = (20, 16))

plt.style.use('fivethirtyeight')



# Color mapping

colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})

poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})



# Iterate through the float columns

for i, col in enumerate(train.select_dtypes('float')):

    ax = plt.subplot(6, 2, i + 1)

    # Iterate through the poverty levels

    for poverty_level, color in colors.items():

        # Plot each poverty level as a separate line

        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(), 

                    ax = ax, color = color, label = poverty_mapping[poverty_level])

        

    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')



plt.subplots_adjust(top = 2)
miss_cols = train_samp.columns[train.isnull().any()]

miss_cols
train_samp.rez_esc.value_counts()
train_samp.isnull().sum()
#Fill values for v2a1

train_samp['v2a1'] = train_samp['v2a1'].fillna(train_samp['v2a1'].mode()[0])



#Fill values for v18q1

train_samp['v18q1'] = train_samp['v18q1'].fillna(train_samp['v18q1'].mean())



#Fill values for rez_esc

train_samp['rez_esc'] = train_samp['rez_esc'].fillna(train_samp['rez_esc'].mode()[0])



#Fill values for meaneduc

train_samp['meaneduc'] = train_samp['meaneduc'].fillna(train_samp['meaneduc'].mode()[0])



#Fill values for SQBmeaned

train_samp['SQBmeaned'] = train_samp['SQBmeaned'].fillna(train_samp['SQBmeaned'].mode()[0])
#Fill values for v2a1

train['v2a1'] = train['v2a1'].fillna(train['v2a1'].mode()[0])



#Fill values for v18q1

train['v18q1'] = train['v18q1'].fillna(train['v18q1'].mean())



#Fill values for rez_esc

train['rez_esc'] = train['rez_esc'].fillna(train['rez_esc'].mode()[0])



#Fill values for meaneduc

train['meaneduc'] = train['meaneduc'].fillna(train['meaneduc'].mode()[0])



#Fill values for SQBmeaned

train['SQBmeaned'] = train['SQBmeaned'].fillna(train['SQBmeaned'].mode()[0])
train_samp.columns[train_samp.isnull().any()]
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import classification_report, roc_auc_score, f1_score, make_scorer, precision_recall_fscore_support



# Custom scorer for cross validation

scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')
#Drop Columns from dataset

X = train_samp.drop(['Id', 'Target', 'idhogar'], axis=1).copy()

X_full = train.drop(['Id', 'Target', 'idhogar'], axis=1).copy()
#Using the sample data set (train_samp) we drop the Id, Target, and idhogar to split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

x_tr, x_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=42)



#Split full data set

X_trfull, X_tefull, y_trfull, y_tefull = train_test_split(X_full, y_full, test_size=0.20, random_state=42)
#pd.concat(y_tefull['idhogar'])

#pd.merge(type_df, y_tefull, left_index=True)

#y_tefull.head()



n_classes = y_full.unique().max()

n_classes
#Full training and test set split

print(f'X_train: {X_trfull.shape}')

print(f'X_test: {X_tefull.shape}')

print(f'y_train: {y_trfull.shape}')

print(f'y_test: {y_tefull.shape}')



#Training and Test set split

print(f'X_train: {X_train.shape}')

print(f'X_test: {X_test.shape}')

print(f'y_train: {y_train.shape}')

print(f'y_test: {y_test.shape}')



#Sample of our training set

print(f'Train Sample: {train_samp.shape}')



#Split the data a second time

print(f'x_tr: {x_tr.shape}')

print(f'y_tr: {y_tr.shape}')

print(f'x_val: {y_val.shape}')

print(f'y_val: {y_val.shape}')
param_dictionary = {"n_estimators": [1000]}

clf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=3)

# Press Shift-Tab to look at what the arguments are for a function, as well as the defaults for each argument

gs = GridSearchCV(clf, param_dictionary, n_jobs=1, verbose=2, cv=2)

gs.fit(X_trfull, y_trfull)

# max depth 5, n estimators 500
val_predictions = gs.predict(X_trfull)

cr = classification_report(y_trfull, val_predictions)

#roc_auc = roc_auc_score(y_val, val_predictions)

print('Validation Scores:')

print(cr)

print('-'*50)

#print("ROC AUC Score: {}".format(roc_auc))
feat_imports = sorted(list(zip(X_train.columns, gs.best_estimator_.feature_importances_)), key=lambda x:x[1], reverse=True)

feat_imports[0:10]
clf = RandomForestClassifier(n_jobs=-1, max_depth=5, n_estimators=1000, class_weight='balanced', verbose=1)

clf.fit(X_train, y_train)
# Dataframe to hold results

model_results = pd.DataFrame(columns = ['model', 'cv_mean', 'cv_std'])



def cv_model(train, train_labels, model, name, model_results=None):

    """Perform 10 fold cross validation of a model"""

    

    cv_scores = cross_val_score(model, train, train_labels, cv = 10, scoring=scorer, n_jobs = -1)

    print(f'10 Fold CV Score: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}')

    

    if model_results is not None:

        model_results = model_results.append(pd.DataFrame({'model': name, 

                                                           'cv_mean': cv_scores.mean(), 

                                                            'cv_std': cv_scores.std()},

                                                           index = [0]),

                                             ignore_index = True)



        return model_results
from sklearn.tree import DecisionTreeClassifier



model_results = cv_model(X_tefull, y_tefull, 

                         DecisionTreeClassifier(),

                         'DT', model_results)
from sklearn.ensemble import ExtraTreesClassifier



model_results = cv_model(X_tefull, y_tefull, 

                         ExtraTreesClassifier(n_estimators = 100, random_state = 10),

                         'EXT', model_results)
model_results = cv_model(X_tefull, y_tefull, 

                         RandomForestClassifier(n_estimators = 100, random_state = 10),

                         'RF', model_results)
for n in [5, 10, 20]:

    print(f'\nKNN with {n} neighbors\n')

    model_results = cv_model(X_tefull, y_tefull, 

                             KNeighborsClassifier(n_neighbors = n),

                             f'knn-{n}', model_results)
model_results.set_index('model', inplace = True)

model_results['cv_mean'].plot.bar(color = 'orange', figsize = (8, 6),

                                  yerr = list(model_results['cv_std']),

                                  edgecolor = 'k', linewidth = 2)

plt.title('Model F1 Score Results');

plt.ylabel('Mean F1 Score (with error bar)');

model_results.reset_index(inplace = True)
def pred_and_score(model, train, train_labels, test, test_ids):

    """Train and test a model on the dataset"""

    

    # Train on the data

    model.fit(train, train_labels)

    

    predictions = model.predict(test)

    predictions = pd.DataFrame({'idhogar': test_ids,

                               'Target': predictions})

    #Compute the mean accuracy

    scores = model.score(test, test_ids)

    

    #Get most important features

    imp_feats = sorted(list(zip(X_train.columns, clf.feature_importances_)), key=lambda x:x[1], reverse=True)

    imp_feats = imp_feats[0:10]



    return predictions, test_ids, scores, imp_feats
#I need to put the idhogar back on
predictions, true_values, scores, imp_feats = pred_and_score(ExtraTreesClassifier(n_estimators = 100, random_state = 10), 

                         X_trfull, y_trfull, X_tefull, y_tefull)


cr = precision_recall_fscore_support(predictions['Target'], true_values, average='macro')

#roc_auc = roc_auc_score(y_val, val_predictions)

print('Validation Scores:')

print('-'*50)



print(f'precision: {cr[0]}')

print(f'recall: {cr[1]}')

print(f'f1-score: {cr[2]}')
print(f'Accuracy Score: {scores}')

print(f'Important Features: {imp_feats}')

#print(f'Evaluation Metrics: {true_values})
clf = ExtraTreesClassifier(n_estimators = 100, random_state = 10)

clf.fit(X_trfull, y_trfull)

clf.score(X_tefull, y_tefull)
predictions.head()
#I want to match idhogars from the full dataset to the predicted values

submission = pd.merge(train['idhogar'].to_frame(), predictions, left_index=True, right_index=True)
submission = submission.drop('idhogar_y', axis=1)

submission.head()
submission.columns = ['Id', 'Target']

submission.head()
# Fill in households missing a head

submission['Target'] = submission['Target'].fillna(4).astype(np.int8)
submission.to_csv('Costa_Rica_Predictions.csv', index=False)
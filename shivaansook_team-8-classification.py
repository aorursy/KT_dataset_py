# Project packages.

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

import numpy as np



# Visualisations.

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns



# NLP.

import nltk  

from nltk.corpus import words

import re



# Machine Learning.

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



# Metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

from sklearn.metrics import roc_auc_score



# Warnings.

import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")

    
# Importing the datasets.

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Saving the 'id' column required for submission.

test_id = test['id']



# Combining the training and testing data to perform uniform data cleaning.

train_test_combined = pd.concat([train,test])
# finding the unique values from the personality column

p_type = np.unique(np.array(train["type"]))

set(p_type)
train['type'].value_counts().plot(kind = 'bar')

plt.xlabel("Personality Types", size = 13)

plt.show()
# Plotting number of posts per personality indicator.

total_posts = train.groupby(["type"]).count()*50

plt.figure(figsize = (10, 5))

plt.bar(np.array(total_posts.index), height = total_posts["posts"],)

plt.xlabel("Personality Types", size = 13)

plt.ylabel("Posts", size = 13)

plt.title("Total Posts per Personality Type")
df = train.copy()

df["Words_Per_Post"] = df['posts'].apply(lambda x: len(x.split()))

plt.figure(figsize=(15,10))

sns.swarmplot("type", "Words_Per_Post", data=df)
def remove_url(string):

    """

    Removes url links found in user posts by replacing them with a single space.

    

    Args:

        string : Text to perform removal on.

        

    Returns:

        string : Transformed text.

    """

    pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

    string = re.sub(pattern_url, " ", string)

    return string



# Applying the url removal element-wise across the combined dataset.

train_test_combined['posts'] = train_test_combined['posts'].apply(remove_url)
def remove_numbers(string):

    """

    Removes numerical characters found in user posts by replacing them with a single space.

    

    Args:

        string : Text to perform removal on.

        

    Returns:

        string : Transformed text.

    """

    pattern_number = r'[0-9]*'

    string = re.sub(pattern_number, "", string)

    return string



# Applying the numerical character removal element-wise across the combined dataset.

train_test_combined['posts'] = train_test_combined['posts'].apply(remove_numbers)
#train_test_combined['posts'] = train_test_combined['posts'].str.lower()
import string

def remove_punc(strings):

    """

    Removes punctuation found in user posts by replacing them with a single space.

    

    Args:

        strings : Text to perform removal on.

        

    Returns:

        string : Transformed text.

    """

    for punctuation in string.punctuation:

        strings = strings.replace(punctuation, ' ')

    return strings



# Applying the punctuation removal element-wise across the combined dataset.

train_test_combined['posts'] = train_test_combined['posts'].apply(remove_punc)
def remove_extra_spacing(string):

    """

    Removes extra spacing found in user posts by replacing them with a single space.

    

    Args:

        string : Text to perform removal on.

        

    Returns:

        string : Transformed text.

    """

    string = re.sub('\s+', ' ', string)

    return string



# Applying the space removal element-wise across the combined dataset.

train_test_combined['posts'] = train_test_combined['posts'].apply(remove_extra_spacing)
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords 



# Defining a set of stop words in cache.

stopwords_list = set(stopwords.words("english"))

# Initializing a Lemmatizer.

lemmatiser = WordNetLemmatizer()



def remove_stopwords(string):

    """

    Removes stop words found in user posts.

    Lemmatizes all remaining words if their length is below 15 characters.

    

    Args:

        string : Text to perform removal on.

        

    Returns:

        string : Transformed text.

    """

    string = " ".join([lemmatiser.lemmatize(word) for word in string.split() if (word not in stopwords_list)&(len(word) < 15)])

    return string



# Applying the stop word removal and lemmatization , element-wise across the combined dataset.

train_test_combined['posts'] = train_test_combined['posts'].apply(remove_stopwords)
def generate_targets(data):

    """

    Creates target binary values for each personality profile.

    - Mind: I = 0, E = 1 

    - Energy: S = 0, N = 1 

    - Nature: F = 0, T = 1 

    - Tactics: P = 0, J = 1

    

    Args:

        data : Dataframe to generate targets from.

        

    Returns:

        dataframe : Transformed dataframe.

    """

    df = data.copy()

    df['mind'] = df['type'].apply(lambda x: x[0] == 'E').astype('int')

    df['energy'] = df['type'].apply(lambda x: x[1] == 'N').astype('int')

    df['nature'] = df['type'].apply(lambda x: x[2] == 'T').astype('int')

    df['tactics'] = df['type'].apply(lambda x: x[3] == 'J').astype('int')

    return df[['mind','energy','nature','tactics']] 



# Generating target labels to be used for classification training.

y = generate_targets(train)



# Example of generated target labels.

y.head()
from sklearn.feature_extraction.text import TfidfVectorizer



# Initializing vectorizer.

tfidvectorizer = TfidfVectorizer(stop_words='english',min_df=1,max_df=0.95) 



# Performing vectorization on training material (user posts).

tfidfvectorized_X = tfidvectorizer.fit_transform(train_test_combined['posts'])
# Separating the submission post-vectorization data from the training data.

X = tfidfvectorized_X[:len(y)]

X_submission = tfidfvectorized_X[len(y):]



# Checking to see that data was separated correctly.

print(X.shape, y.shape, X_submission.shape)



# Seperating the training data into a training and validation set.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=22)
# Initializing model.

LR_model = LogisticRegression()



# Creating dictionary to store results

results_index = ['Accuracy','Precision','Recall','F1 Score']

LR_model_results = pd.DataFrame(index = results_index, columns=list(y.columns))

LR_default_submission_dict = {'id':test['id']}



# Perform fitting and validation across all attributes.

print(" Logistic Regression (Default Parameters) ")  

print('-'*len(" Logistic Regression (Default Parameters) "))

for attribute in list(y.columns):

    LR_model.fit(X_train, y_train[attribute])

    y_pred = LR_model.predict(X_test)

    LR_default_submission_dict[attribute] = LR_model.predict(X_submission)

    LR_model_results.loc['Accuracy',attribute] = accuracy_score(y_test[attribute],y_pred)

    LR_model_results.loc['Precision',attribute] = precision_score(y_test[attribute],y_pred)

    LR_model_results.loc['Recall',attribute] = recall_score(y_test[attribute],y_pred)

    LR_model_results.loc['F1 Score',attribute] = f1_score(y_test[attribute],y_pred)

    LR_model_results.loc['Log-loss',attribute] = log_loss(y_test[attribute],y_pred)

    LR_model_results.loc['ROC_AUC',attribute] = roc_auc_score(y_test[attribute],y_pred)

    print(f'Confusion Matrix ({attribute}): \n' + str(pd.DataFrame(confusion_matrix(y_test[attribute],y_pred,labels=[0,1]))))

print('\n')

print(" Logistic Regression Results (Default Parameters) ")  

print("-"*len(" Logistic Regression Results (Default Parameters) "))

print(LR_model_results)   

# Initializing model.

LR_model = LogisticRegression()



# Parameters to use during GridSearch.

grid_parameters = {"C":[0.05,0.5,1,5,10,20,25,30], 

                   "solver" : ['lbfgs', 'liblinear', 'sag', 'saga'],

                   "class_weight": ['balanced']}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)



# Creating dictionary to store results.

results_index = ['Accuracy','Precision','Recall','F1 Score']

LR_model_GSCV_results = pd.DataFrame(index = results_index, columns=list(y.columns))

LR_submission_dict = {'id':test['id']}



# Perform fitting and validation across all attributes.

print(" Logistic Regression (Grid Searched Parameters) ")

print("-"*len(" Logistic Regression(Grid Searched Parameters "))

print(" (Best parameters per attribute) ")

print(" ")

for attribute in list(y.columns):

    LR_model_GSCV = GridSearchCV(LR_model, param_grid = grid_parameters, scoring = 'f1', cv = kfold)

    LR_model_GSCV.fit(X_train, y_train[attribute])

    print(f'{attribute}:',LR_model_GSCV.best_params_)

    print(" ")

    best_model = LR_model_GSCV.best_estimator_

    y_pred = best_model.predict(X_test)

    LR_submission_dict[attribute] = best_model.predict(X_submission)

    LR_model_GSCV_results.loc['Accuracy',attribute] = accuracy_score(y_test[attribute],y_pred)

    LR_model_GSCV_results.loc['Precision',attribute] = precision_score(y_test[attribute],y_pred)

    LR_model_GSCV_results.loc['Recall',attribute] = recall_score(y_test[attribute],y_pred)

    LR_model_GSCV_results.loc['F1 Score',attribute] = f1_score(y_test[attribute],y_pred)

    LR_model_GSCV_results.loc['Log-loss',attribute] = log_loss(y_test[attribute],best_model.predict(X_test))

    LR_model_GSCV_results.loc['ROC_AUC',attribute] = roc_auc_score(y_test[attribute],y_pred)

    print(f'Confusion Matrix ({attribute}): \n' + str(pd.DataFrame(confusion_matrix(y_test[attribute],y_pred,labels=[0,1]))))

    print(" ")

    

print("-"*len(" Logistic Regression Results (Grid Searched Parameters) "))

print(" Logistic Regression Results (Grid Searched Parameters) ")  

print("-"*len(" Logistic Regression Results (Grid Searched Parameters) "))

print(LR_model_GSCV_results)   

LR_submission_df = pd.DataFrame(LR_submission_dict)

# Initializing model.

KNN_model = KNeighborsClassifier(weights='distance', n_neighbors=3)



# Creating dictionary to store results

results_index = ['Accuracy','Precision','Recall','F1 Score']

KNN_model_results = pd.DataFrame(index = results_index, columns=list(y.columns))

KNN_submission_dict = {'id':test['id']}



# Perform fitting and validation across all attributes.

print(" K-Nearest-Neighbors Results ")

print('-'*len(" K-Nearest-Neighbors Results "))

for attribute in list(y.columns):

    KNN_model.fit(X_train, y_train[attribute])

    y_pred = KNN_model.predict(X_test)

    KNN_submission_dict[attribute] = KNN_model.predict(X_submission)

    KNN_model_results.loc['Accuracy',attribute] = accuracy_score(y_test[attribute],y_pred)

    KNN_model_results.loc['Precision',attribute] = precision_score(y_test[attribute],y_pred)

    KNN_model_results.loc['Recall',attribute] = recall_score(y_test[attribute],y_pred)

    KNN_model_results.loc['F1 Score',attribute] = f1_score(y_test[attribute],y_pred)

    KNN_model_results.loc['Log-loss',attribute] = log_loss(y_test[attribute],y_pred)

    KNN_model_results.loc['ROC_AUC',attribute] = roc_auc_score(y_test[attribute],y_pred)

    print(f'Confusion Matrix ({attribute}): \n' + str(pd.DataFrame(confusion_matrix(y_test[attribute],y_pred,labels=[0,1]))))

print('\n')

print("            K-Nearest-Neighbors Results ")  

print("-"*49)

print(KNN_model_results)
# Initializing model.

RFC_model = RandomForestClassifier()



# Parameters to use during GridSearch.

grid_parameters = {"n_estimators":[100,200], 

                   "class_weight": ['balanced']}



kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)



# Creating dictionary to store results

results_index = ['Accuracy','Precision','Recall','F1 Score']

RFC_model_GSCV_results = pd.DataFrame(index = results_index, columns=list(y.columns))

RFC_submission_dict = {'id':test['id']}



# Perform fitting and validation across all attributes.

print(" Random Forest Classifier (Grid Searched Parameters) ")

print("-"*len(" Random Forest Classifier (Grid Searched Parameters) "))

print(" (Best parameters per attribute) ")

print(" ")

for attribute in list(y.columns):

    RFC_model_GSCV = GridSearchCV(RFC_model, param_grid = grid_parameters, scoring = 'neg_log_loss', cv = kfold)

    RFC_model_GSCV.fit(X_train, y_train[attribute])

    print(f'{attribute}:',RFC_model_GSCV.best_params_)

    print(" ")

    best_model = RFC_model_GSCV.best_estimator_

    y_pred = best_model.predict(X_test)

    RFC_submission_dict[attribute] = best_model.predict(X_submission)

    RFC_model_GSCV_results.loc['Accuracy',attribute] = accuracy_score(y_test[attribute],y_pred)

    RFC_model_GSCV_results.loc['Precision',attribute] = precision_score(y_test[attribute],y_pred)

    RFC_model_GSCV_results.loc['Recall',attribute] = recall_score(y_test[attribute],y_pred)

    RFC_model_GSCV_results.loc['F1 Score',attribute] = f1_score(y_test[attribute],y_pred)

    RFC_model_GSCV_results.loc['Log-loss',attribute] = log_loss(y_test[attribute],y_pred)

    RFC_model_GSCV_results.loc['ROC_AUC',attribute] = roc_auc_score(y_test[attribute],y_pred)

    print(f'Confusion Matrix ({attribute}): \n' + str(pd.DataFrame(confusion_matrix(y_test[attribute],y_pred,labels=[0,1]))))

    print(" ")

    

print("-"*len(" Random Forest Classifier Results (Grid Searched Parameters) "))

print(" Random Forest Classifier Results (Grid Searched Parameters) ")  

print("-"*len(" Random Forest Classifier Results (Grid Searched Parameters) "))

print(RFC_model_GSCV_results)
# Combining each models results for comparison.

LR_model_results['model'] = 'Logistic_Default'

LR_model_GSCV_results['model'] = 'Logistic_GridSearch' 

KNN_model_results['model'] = 'K_Nearest_Neighbors'

RFC_model_GSCV_results['model'] = 'Random_Forest' 

results_df = LR_model_results.append(LR_model_GSCV_results).append(KNN_model_results).append(RFC_model_GSCV_results)

results_df = results_df.reset_index()

results_df = results_df.rename(columns = {'index':'metric'})

results_df = results_df.set_index(['model'])

results_df = results_df.sort_values(by = 'metric')



# Viewing the results.

results_df
# Preparing submission file.

LR_submission_df = pd.DataFrame(LR_submission_dict)

LR_submission_df.to_csv("LR_submission_df.csv",index=False)
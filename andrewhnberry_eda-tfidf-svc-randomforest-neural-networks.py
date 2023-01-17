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
#importing the holy trinity of data science packages
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#Other visualization packages
import seaborn as sns

#Importing NLP plugins
from nltk.corpus import stopwords 
stop_words = stopwords.words('english')
from nltk.stem import WordNetLemmatizer 
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#Importing our Sklearn Plugins
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#importing our models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#Model Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
df = pd.read_csv("../input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv")
df.shape
df.head(3)
#Checking our Data Types
df.info()
#Check percentage of data missing for each feature/column
df.isna().sum()/len(df)
#Checking for unique elements for each column
df.nunique()
# Always good practice to make a copy of your dataframe ever so often,
# so you can roll back your mistakes much easier than rerunning your whole kernal again.
df_2 = df.copy()
df_2 = df_2.drop(labels = ['job_id','salary_range',
                    'department','benefits',
                    'company_profile'], axis = 1) #axis = 1 to refer droping columns
df_2.tail(3)
df_2['employment_type'] = df_2['employment_type'].bfill(axis=0)
df_2['required_experience'] = df_2['required_experience'].bfill(axis = 0)
df_2['required_education'] = df_2['required_education'].bfill(axis = 0)
df_2['industry'] = df_2['industry'].bfill(axis = 0)
df_2['function'] = df_2['function'].bfill(axis = 0)
# Make Dataframe copy
df_3 = df_2.copy()

# Keeping non NaN rows in my new dataframe
df_3 = df_3[df_3['description'].notna()]

# Replacing NaNs with an empty string.
#df_3 = df_3.replace(np.nan, '', regex = True)
# For good measure let's drop any other Nans 
df_3 = df_3.dropna(axis = 0, how = 'any')
print(f'We currenlty have {len(df_3)} rows. However, let\'s drop duplicates and compare.')
# drop duplicates
df_3 = df_3.drop_duplicates(keep = 'first')
df_3.isna().sum()/len(df)
print(f'After dropping duplicates we have {len(df_3)} rows left. It seems there were 178 duplicate rows.')
# Make copy
df_4 = df_3.copy()

#concatenating our description and requirments columns
df_4['description'] = df_4['description'] + ' ' + df_4['requirements']
del df_4['requirements']
#Clean DataFrame
df_clean = df_4.copy()

display(df_clean.head(7))
print(df_clean.shape)
#Ploting the Target variable
plt.figure(figsize = (10,5))
sns.countplot(x = df.fraudulent, data = df,palette="Set3")
plt.title('Fradulent (Target Variable) Count')
plt.show()
#Stylistic Set
sns.set(style="whitegrid")

plt.figure(figsize = (14,11))

#fig 1
plt.subplot(2,2,1)
sns.countplot(y = df.employment_type, data = df,palette="Set3", 
              order = df.employment_type.value_counts().index)
plt.title("Employment Type Count")
plt.ylabel("")

#fig2
plt.subplot(2,2,2)
#matplotlib version
#df.required_experience.value_counts().plot(kind='barh')
#sns version
sns.countplot(y = df.required_experience, data = df,palette="Set3",
             order = df.required_experience.value_counts().index)
plt.title("Required Experience Count")
plt.ylabel("")

#fig 3
plt.subplot(2,2,3)
sns.countplot(y = df.required_education, data = df,palette="Set3",
             order = df.required_education.value_counts().index)
plt.title("Required Education Count")
plt.ylabel("")

plt.tight_layout()
plt.show()
industry = df.industry.value_counts()[:10]
function = df.function.value_counts()[:10]

plt.figure(figsize = (12,12))

plt.subplot(2,1,1)
industry.plot(kind = 'barh')
plt.title('Top 10 Industries Represented in this Dataset.')
plt.xlabel('Count')

plt.subplot(2,1,2)
function.plot(kind = 'barh')
plt.title('Top 10 Business Functions Represented in this Dataset.')
plt.xlabel('Count')

plt.tight_layout()
plt.show()
#Make Copy
df_5 = df_clean.copy()

# One Hot Encoding using Pandas get dummies function
columns_to_1_hot = ['employment_type','required_experience','required_education',
                   'industry', 'function']

for column in columns_to_1_hot:
    encoded = pd.get_dummies(df_5[column])
    df_5 = pd.concat([df_5, encoded], axis = 1)

columns_to_1_hot += ['title', 'location']
    
#droping the original columns that we just one hot encoded from
df_5 = df_5.drop(columns_to_1_hot, axis = 1)
df_5.head()
def tokenizer(text):
    
    #All characters in this string will be converted to lowercase
    text = text.lower()
    
    #Removing sentence punctuations
    for punctuation_mark in string.punctuation:
        text = text.replace(punctuation_mark,'')
    
    #Creating our list of tokens
    list_of_tokens = text.split(' ')
    #Creating our cleaned tokens list 
    cleaned_tokens = []
    #Intatiating our Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #Removing Stop Words in our list of tokens and any tokens that happens to be empty strings
    for token in list_of_tokens:
        if (not token in stop_words) and (token != ''):
            #lemmatizing our token
            token_lemmatized = lemmatizer.lemmatize(token)
            #appending our finalized cleaned token
            cleaned_tokens.append(token_lemmatized)
    
    return cleaned_tokens
df_6 = df_5.copy()

#Instatiating our tfidf vectorizer
tfidf = TfidfVectorizer(tokenizer = tokenizer, min_df = 0.05, ngram_range=(1,3))
#Fit_transform our description 
tfidf_features = tfidf.fit_transform(df_6['description']) #this will create a sparse matrix
#I want to append this sparse matrix to the original pandas Dataframe
tfidf_vect_df = pd.DataFrame(tfidf_features.todense(), columns = tfidf.get_feature_names())

df_tfidf = pd.concat([df_6, tfidf_vect_df], axis = 1)

#Minor Cleaning steps after appending our tfidf results to our Dataframe, we will need to drop the description column. 
df_tfidf = df_tfidf.drop(['description'], axis = 1)
df_tfidf = df_tfidf.dropna()
df_tfidf.head(3)
#Instatiating our CountVectorizer
count_vect = CountVectorizer(tokenizer = tokenizer, min_df = 0.05, ngram_range=(1,3))
#Fit_transform our description 
count_vect_features = count_vect.fit_transform(df_6['description']) #this will create a sparse matrix

count_vect_df = pd.DataFrame(count_vect_features.todense(), columns = count_vect.get_feature_names())

df_count_vect = pd.concat([df_6, count_vect_df], axis = 1)
df_count_vect = df_count_vect.drop(['description'], axis = 1)
df_count_vect = df_count_vect.dropna()
df_count_vect.head(3)
target = df_tfidf.fraudulent
features = df_tfidf.drop(['fraudulent'], axis = 1)

#Spliting our Data into train and holdout sets to test our models
X_train, X_hold, y_train, y_hold = train_test_split(features, target, test_size = 0.1,
                                                    stratify = target, random_state = 42)
#Intatiating our Logistic Regression Model
log_reg = LogisticRegression()
#I want to optimze the C-Value and penalty
c_values = [.00001, .0001, .001, .1, 1, 10, 100, 1000, 10000]
penalty_options = ['l1','l2']

param_grid = dict(C = c_values, penalty = penalty_options)
grid_tfidf = GridSearchCV(log_reg, param_grid= param_grid, cv = 10, scoring = 'roc_auc', n_jobs = -1)
grid_tfidf.fit(X_train, y_train)
print(grid_tfidf.best_score_)
print(grid_tfidf.best_params_)
log_reg_tfidf_pred = grid_tfidf.predict(X_hold)
print(roc_auc_score(y_hold, log_reg_tfidf_pred))
print(classification_report(y_hold, log_reg_tfidf_pred))
target_2 = df_count_vect.fraudulent
features_2 = df_count_vect.drop(['fraudulent'], axis = 1)

#Spliting our Data into train and holdout sets to test our models
X_train_2, X_hold_2, y_train_2, y_hold_2 = train_test_split(features_2, target_2, test_size = 0.1,
                                                    stratify = target_2, random_state = 42)

#Intiatiating our previous logistic regression model, using the count vectorizer dataset
grid_count_vect = GridSearchCV(log_reg, param_grid= param_grid, cv = 10, scoring = 'roc_auc', n_jobs = -1)
grid_count_vect.fit(X_train_2, y_train_2)
print(grid_count_vect.best_score_)
print(grid_count_vect.best_params_)
log_reg_pred_2 = grid_count_vect.predict(X_hold_2)
print(roc_auc_score(y_hold_2, log_reg_pred_2))
print(classification_report(y_hold_2, log_reg_pred_2))
# Model - KNearestNeighbors
knn = KNeighborsClassifier()

#The parameters we would like to optimize for
k_range = list(np.arange(2,23,2))
param_grid_knn = dict(n_neighbors=k_range)
print(param_grid_knn)
#Intatiate our knn gridsearch
grid_knn = GridSearchCV(knn, param_grid_knn, cv=10, scoring='roc_auc',
                        n_jobs = -1)

#Fit our grid_knn
grid_knn.fit(X_train, y_train)
print(grid_knn.best_score_)
print(grid_knn.best_params_)
#predicting on our holdout data
knn_pred = grid_knn.predict(X_hold)
#Printing out our evaluation metrics
print(roc_auc_score(y_hold, knn_pred))
print(classification_report(y_hold, knn_pred))
#Intatiating our SVM model
svc = SVC(kernel = 'linear', gamma = 'auto' )

# I wont use a gridsearch because SVMs usually take a long looong time. I will just use a simple SVC
# and see how it plays out
svc.fit(X_train, y_train)
#predicting our holdout data
svc_pred = svc.predict(X_hold)

#Printing out our evaluation metrics
print(roc_auc_score(y_hold, svc_pred))
print(classification_report(y_hold, svc_pred))
#Instatiating our random forest

rf = RandomForestClassifier()

#The parameters we want to tune with our random forest
n_estimators_range = [1, 2, 4, 8, 16, 32, 64, 100, 200]

param_grid_rf = dict(n_estimators=n_estimators_range)

grid_rf = GridSearchCV(rf, param_grid_rf, cv=10, scoring='roc_auc',
                        n_jobs = -1)
grid_rf.fit(X_train, y_train)
print(grid_rf.best_score_)
print(grid_rf.best_params_)
rf_pred = grid_rf.predict(X_hold)
#Printing out our evaluation metrics
print(roc_auc_score(y_hold, rf_pred))
print(classification_report(y_hold, rf_pred))
#Instatiatie our MLPClassifier
mlp = MLPClassifier(solver='lbfgs', 
                    activation = 'relu',
                   hidden_layer_sizes = (100,50,30), 
                    max_iter = 1000)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_hold)

#Printing out our evaluation metrics
print(roc_auc_score(y_hold, mlp_pred))
print(classification_report(y_hold, mlp_pred))
#Instatiatie our MLPClassifier
mlp = MLPClassifier(solver='adam', 
                    activation = 'relu',
                   hidden_layer_sizes = (100,50,30), 
                    max_iter = 1000)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_hold)

#Printing out our evaluation metrics
print(roc_auc_score(y_hold, mlp_pred))
print(classification_report(y_hold, mlp_pred))

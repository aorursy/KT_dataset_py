# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import re
import string
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import vstack
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
data_file = pd.read_csv('/kaggle/input/phishing-data/combined_dataset.csv')
data_file.head(5)
#Checking for null values
data_file.isna().any()
#Checking the distribution of data
print(data_file['label'].value_counts())
print(data_file.shape)

def remove_characters(row):
    chars = re.escape(string.punctuation)
    return re.sub(r'['+chars+']', ' ',row)

data_file['domain'] = data_file['domain'].apply(remove_characters)
data_file.head(5)
#Creating string for Wordcloud of 'domain' tokens
comment_words = '' 
stopwords = set(STOPWORDS) 
for val in data_file['domain']: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords=stopwords,
                min_font_size = 10).generate(comment_words)
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.show() 
data_file.shape
data_file_x = data_file.drop(['label'], axis=1)
data_file_y = data_file['label']
#Dividing the dataset into train, val and test datasets
train_df_x = data_file_x[:60000]
train_df_y = data_file_y[:60000]
val_df_x = data_file_x[60000:78000]
val_df_y = data_file_y[60000:78000]
test_df_x = data_file_x[78000:]
test_df_y = data_file_y[78000:]
train_domain = train_df_x['domain']
val_domain = val_df_x['domain']
test_domain = test_df_x['domain']
#Drop 'domain' from dataset since we are converting it into bag of words
train_df_x = train_df_x.drop(['domain'], axis=1)
val_df_x = val_df_x.drop(['domain'], axis=1)
test_df_x = test_df_x.drop(['domain'], axis=1)
print(train_df_x.shape, val_df_x.shape, test_df_x.shape)
print(train_df_y.shape, val_df_y.shape, test_df_y.shape)
count_vect = CountVectorizer()
X_train_bow = count_vect.fit_transform(train_domain)
X_val_bow = count_vect.transform(val_domain)
X_test_bow = count_vect.transform(test_domain)
feature_names_bow = count_vect.get_feature_names()
print(X_train_bow.shape)
print(X_val_bow.shape)
print(X_test_bow.shape)
#Stacking the BoW features and other features from dataset
bow_final_train_x = hstack((X_train_bow, train_df_x))
bow_final_val_x = hstack((X_val_bow, val_df_x))
bow_final_test_x = hstack((X_test_bow, test_df_x))
#Final shape of dataset will be 75926 features of domain plus 10 features of dataset. Therefore 75936 features
bow_final_train_x.shape
bow_final_val_x.shape
bow_final_test_x.shape
X_train_val = vstack((bow_final_train_x, bow_final_val_x))
Y_train_val = pd.concat([train_df_y, val_df_y], axis= 0)
param_grid = {
 'max_depth': [4, 8, 16, 32],
 'n_estimators': [1, 2, 5, 10, 50, 100, 200]
}
t1 = datetime.datetime.now()
rf = RandomForestClassifier(n_jobs=-1)
clf = GridSearchCV(estimator = rf, param_grid = param_grid, scoring = 'roc_auc')
clf.fit(X_train_val,Y_train_val)
print("time required = ", datetime.datetime.now() - t1)
clf.best_params_
rf_clf = RandomForestClassifier(max_depth = clf.best_params_['max_depth'], 
                                n_estimators=clf.best_params_['n_estimators'])
rf_clf.fit(X_train_val,Y_train_val)
bow_test_proba = rf_clf.predict_proba(bow_final_test_x)
bow_train_proba = rf_clf.predict_proba(X_train_val)
print("Train proba", bow_train_proba)
print("Test proba", bow_test_proba)
print("Top 20 Important Features")
d = sorted(list(zip(count_vect.get_feature_names(), rf_clf.feature_importances_ )), key=lambda x: x[1], reverse=True)[:20]
features_list = []
for (i,j) in d:
    features_list.append(i)
print(features_list)
#calculatinf the AUC
bow_fpr_train, bow_tpr_train, _ = roc_curve(Y_train_val, bow_train_proba[:, 1])
bow_fpr_test, bow_tpr_test, _ = roc_curve(test_df_y, bow_test_proba[:, 1])
bow_test_auc = auc(bow_fpr_test, bow_tpr_test)
bow_train_auc = auc(bow_fpr_train, bow_tpr_train)
print("Train AUC", bow_train_auc)
print("Test AUC", bow_test_auc)
import pylab
plt.figure(figsize=(13, 10))
plt.plot([0,1], [0,1], color='black', lw=2, linestyle='--')
plt.plot(bow_fpr_test, bow_tpr_test, label="Test, auc="+str(bow_test_auc), color = 'red')
plt.plot(bow_fpr_train, bow_tpr_train, label="Train, auc="+str(bow_train_auc), color = 'green')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

plt.show()
#Making predictions
bow_test_conf = rf_clf.predict(bow_final_test_x)
bow_train_conf = rf_clf.predict(X_train_val)
#Confusion Matrix and classification report
bow_train_conf_matrix = confusion_matrix(Y_train_val, bow_train_conf)
bow_test_conf_matrix = confusion_matrix(test_df_y, bow_test_conf)
class_report = classification_report(test_df_y, bow_test_conf)
print(bow_test_conf_matrix)
print(class_report)
ax= plt.subplot()
sns.heatmap(bow_train_conf_matrix, annot=True, ax = ax, fmt='g')
ax.set_ylabel('Predicted labels')
ax.set_xlabel('True labels')
ax.set_title('Train Confusion Matrix') 
ax.xaxis.set_ticklabels(['negative', 'positive']) 
ax.yaxis.set_ticklabels(['negative', 'positive'])
ax= plt.subplot()
sns.heatmap(bow_test_conf_matrix, annot=True, ax = ax, fmt='g')
ax.set_ylabel('Predicted labels')
ax.set_xlabel('True labels')
ax.set_title('Train Confusion Matrix') 
ax.xaxis.set_ticklabels(['negative', 'positive']) 
ax.yaxis.set_ticklabels(['negative', 'positive'])
from prettytable import PrettyTable
    
x = PrettyTable()
x.field_names = ["Algorithm", "Max_depth", "n_estimators",  "Vectorizer", "Train", "Test"]

x.add_row(["Random Forest", 32, 200, "BoW", 0.98914, 0.9875])
print(x)
#Importing necessary Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns
#Reading the dataset

data=pd.read_csv('../input/imdb-5000-movie-dataset/movie_metadata.csv')

data.head()
#Categorising the target varible 

bins = [ 1, 3, 6, 10]

labels = ['FLOP', 'AVG', 'HIT']

data['imdb_binned'] = pd.cut(data['imdb_score'], bins=bins, labels=labels)
data.groupby(['imdb_binned']).size().plot(kind="bar",fontsize=14)

plt.xlabel('Categories')

plt.ylabel('Number of Movies')

plt.title('Categorization of Movies')
#Checking the new column

data.head(5)
#Shape of the dataset

data.shape
#Total null values present in each column

data.isnull().sum()
#Droping the samples that have missing values

data.dropna(inplace=True)
#Final shape of the data after Droping missing values

data.shape
#List of variables in the datset

data.columns
data.shape
#Describing the categorical data

data.describe(include='object')
#Dropping 2 columns

data.drop(columns=['movie_title','movie_imdb_link'],inplace=True)
#Label encoding the categorical columns

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cat_list=['color', 'director_name', 'actor_2_name',

        'genres', 'actor_1_name',

        'actor_3_name',

        'plot_keywords',

        'language', 'country', 'content_rating',

       'title_year', 'aspect_ratio']

data[cat_list]=data[cat_list].apply(lambda x:le.fit_transform(x))
#A sample of data after label encoding

data.head()
#Finding Correlation between variables

corr = data.corr()

mask = np.zeros(corr.shape, dtype=bool)

mask[np.triu_indices(len(mask))] = True

plt.subplots(figsize=(20,15))

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,cmap='RdYlGn',annot=True,mask = mask)
#Removing few columns due to multicollinearity

data.drop(columns=['cast_total_facebook_likes','num_critic_for_reviews'],inplace=True)
#Removing the column "imdb_score" since we have "imdb_binned"

data.drop(columns=['imdb_score'],inplace=True)
data.shape
#Independent Variables

X = data.iloc[:, 0:23].values

#Dependent/Target Variable

y = data.iloc[:, 23].values

y
#Spliting the data into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0,stratify = y)

print(X_train.shape)

print(y_train.shape)
#Scaling the dependent variables

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#Performing Recursive Feauture Elimation with Cross Validation

#Using Random forest for RFE-CV and logloss as scoring

from sklearn.feature_selection import RFECV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import log_loss

clf_rf=RandomForestClassifier(random_state=0)

rfecv=RFECV(estimator=clf_rf, step=1,cv=5,scoring='neg_log_loss')

rfecv=rfecv.fit(X_train,y_train)
#Optimal number of features

X_train = pd.DataFrame(X_train)

X_test = pd.DataFrame(X_test)

print('Optimal number of features :', rfecv.n_features_)

print('Best features :', X_train.columns[rfecv.support_])
#Feauture Ranking

clf_rf = clf_rf.fit(X_train,y_train)

importances = clf_rf.feature_importances_



std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]

#Logloss vs Number of features

import matplotlib.pyplot as plt

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score of number of selected features")

plt.title("Log loss vs Number of fetures")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
#Selecting the Important Features

X_opt = X_train.iloc[:,X_train.columns[rfecv.support_]]

X_test = X_test.iloc[:,X_test.columns[rfecv.support_]]
#Creating anew dataframe with column names and feature importance

dset = pd.DataFrame()

data1 = data

data1.drop(columns=['imdb_binned'],inplace=True)

dset['attr'] = data1.columns



dset['importance'] = clf_rf.feature_importances_

#Sorting with importance column

dset = dset.sort_values(by='importance', ascending=True)



#Barplot indicating Feature Importance

plt.figure(figsize=(16, 14))

plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')

plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)

plt.xlabel('Importance', fontsize=14, labelpad=20)

plt.show()
#Training the Random Forest Classifer on Train data

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)

classifier.fit(X_opt, y_train)

#Predicting the target variable

y_pred = classifier.predict(X_test)
#Confusion Matrix

from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test,y_pred)

cm
#Classification Report

from sklearn.metrics import classification_report

cr = classification_report(y_test,y_pred)

print(cr)
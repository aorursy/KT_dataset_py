import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.ensemble import GradientBoostingClassifier

pd.set_option('display.max_columns', 50)
raw_data = pd.read_csv('the_zebra_movie_data.csv')
raw_data.head()
raw_data = raw_data.drop_duplicates()
numericalCols = list(raw_data.select_dtypes(include=['float64']).columns) + list(raw_data.select_dtypes(include=['int64']).columns)

for each in numericalCols:

    raw_data[each] = raw_data.groupby('movie_title')[each].transform('mean')
raw_data = raw_data.groupby('movie_title', as_index=False).first()
raw_data['movie_title'].value_counts().head()
#If you would like to see all plots you can uncomment the line below. I am showing three features as to

#not overcrowd the notebook.



for each in numericalCols[:3]:

#for each in numericalCols:

    plt.hist(raw_data[each])

    plt.xlabel(each)

    plt.ylabel('Count')

    plt.show()
raw_data.describe() 
raw_data["roi"]  = (raw_data['gross']/raw_data['budget'])*100

#raw_data["net_profit"]  = raw_data['gross']-raw_data['budget'] -This could be another option to predict on instead of ROI
cols_nan = pd.DataFrame({'Total Null': raw_data.isnull().sum().sort_values(ascending=False).values,\

                        'Total NonNull': raw_data.count().sort_values().values,\

                        'Percent Null': (raw_data.isnull().sum()/raw_data.shape[0]).sort_values(ascending =False).values},\

                       index = raw_data.count().sort_values().index)

cols_nan.head()
raw_data = raw_data.dropna(axis=0, subset=['roi', 'movie_score'])
raw_data['success'] = np.where((raw_data['roi']>=0) &(raw_data['movie_score']>=7), 1, 0)
labels = "Successful", "Non-Successful"

explode = (.03, 0.0)

sizes = [raw_data[raw_data['success']==1].shape[0] ,raw_data[raw_data['success']==0].shape[0]]

plt.pie(sizes, explode=explode, labels =labels, autopct='%1.f%%')

plt.title("Distribution of Successful Movies")

plt.show()

categoryCols = list(raw_data.select_dtypes(include=['object']).columns)

categoryCols
for columns in categoryCols:

    if len(raw_data[columns].value_counts()) <= 5:

        dummies = pd.get_dummies(raw_data[columns], prefix = columns)

        raw_data = pd.concat([raw_data, dummies], axis=1)

        raw_data = raw_data.drop([columns], axis = 1)
raw_data = raw_data.drop(['director_name','actor_1_name','actor_2_name','actor_3_name','plot_keywords'], axis=1)
raw_data['genres'] = raw_data['genres'].str.split('|')

raw_data = raw_data.drop('genres', 1).join(raw_data.genres.str.join('|').str.get_dummies())
raw_data['language'] = np.where(raw_data['language'].str.lower()=='english', 1, 0)

raw_data['country']= np.where(raw_data['country'].str.lower()=='usa', 1, 0)
raw_data['sequel'] = np.where(raw_data['movie_title'].str[-2:-1].str.isnumeric(), 1, 0)

raw_data = raw_data.drop('movie_title', axis = 1)                     
raw_data['content_rating'].value_counts()
content_map = {'R': 2, 'PG-13': 1, 'PG': 0, 'Not Rated' : 2, 'G': 0, 'Unrated' : 2, 'Approved' : 0,\

               'X': 2, 'Passed': 1,'NC-17':2, 'GP': 0 ,'M': 2}
raw_data['content_rating'] = raw_data['content_rating'].map(content_map)
#If you would like to see all plots you can uncomment the line below. I am showing three features as to

#not overcrown the notebook.



#for each in raw_data.columns:

for each in ['director_facebook_likes','Short','Thriller']:

    sns.factorplot('success', each, data=raw_data, kind='bar',ci= None)

    plt.show()
#I will fill in null data with the respective column's average. This is something to look further into.

#In the case of facebook likes, a null might represent the fact that the actor does not have a facebook so they actually have 0 likes.

raw_data = raw_data.fillna(raw_data.mean())
from sklearn.model_selection import train_test_split

#We need to remove the dependent variable and the features that were used to calcuate it's value

X = raw_data.drop(['success','movie_score','roi', 'gross','budget'], axis=1)

y = raw_data['success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
clf = GradientBoostingClassifier()

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

predictions_train = clf.predict(X_train)



print("Confusion Matrix: ")

print(confusion_matrix(y_test, predictions))



print("[[True Non-Success, False Non-Success]")

print("[False Success, True Success]]")



print("Classification Report")

print(classification_report(y_test, predictions))
print(roc_auc_score(y_test, predictions))

print(roc_auc_score(y_train, predictions_train))
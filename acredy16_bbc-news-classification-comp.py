# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
TRAIN_PATH = os.path.join("../input/", "BBC News Train.csv")



df = pd.read_csv(TRAIN_PATH)
df.head()
df['category_id'] = df['Category'].factorize()[0]



df['category_id'][0:10]
category_id_df = df[['Category', 'category_id']].drop_duplicates().sort_values('category_id')
category_id_df
category_to_id = dict(category_id_df.values)

id_to_category = dict(category_id_df[['category_id', 'Category']].values)
id_to_category
category_to_id
df.groupby('Category').category_id.count()
df.groupby('Category').category_id.count().plot.bar(ylim=0)
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')



features = tfidf.fit_transform(df.Text).toarray() 

labels = df.category_id                         
features.shape
category_to_id.items()
sorted(category_to_id.items())
from sklearn.feature_selection import chi2



N = 3





for Category, category_id in sorted(category_to_id.items()):

  features_chi2 = chi2(features, labels == category_id)                   

  indices = np.argsort(features_chi2[0])                                 

  feature_names = np.array(tfidf.get_feature_names())[indices]           

  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]         

  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]          

  print("# '{}':".format(Category))

  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:]))) 

  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:]))) 
features_chi2
from sklearn.manifold import TSNE



# Sampling a subset of our dataset because t-SNE is computationally expensive

SAMPLE_SIZE = int(len(features) * 0.3)

np.random.seed(0)

indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)          # Randomly select 30 % of samples

projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices]) # Array of all projected features of 30% of Randomly chosen samples 
projected_features.shape
my_id = 0 # Select a category_id

projected_features[(labels[indices] == my_id).values]
colors = ['pink', 'green', 'midnightblue', 'orange', 'darkgrey']



# Find points belonging to each category and plot them

for category, category_id in sorted(category_to_id.items()):

    points = projected_features[(labels[indices] == category_id).values]

    plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[category_id], label=category)

plt.title("tf-idf feature vector for each article, projected on 2 dimensions.",

          fontdict=dict(fontsize=15))

plt.legend()
features.shape
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB



from sklearn.model_selection import cross_val_score





models = [

    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),

    MultinomialNB(),

    LogisticRegression(random_state=0),

]



CV = 5  # Cross Validate with 5 different folds of 20% data ( 80-20 split with 5 folds )



#Create a data frame that will store the results for all 5 trials of the 3 different models

cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = [] # Initially all entries are empty

#For each Algorithm 

for model in models:

  model_name = model.__class__.__name__

  # create 5 models with different 20% test sets, and store their accuracies

  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)

  # Append all 5 accuracies into the entries list ( after all 3 models are run, there will be 3x5 = 15 entries)

  for fold_idx, accuracy in enumerate(accuracies):

    entries.append((model_name, fold_idx, accuracy))
# Store the entries into the results dataframe and name its columns    

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
import seaborn as sns



sns.boxplot(x='model_name', y='accuracy', data=cv_df)

sns.stripplot(x='model_name', y='accuracy', data=cv_df, 

              size=8, jitter=True, edgecolor="gray", linewidth=2)
# Mean accuracy of each algorithm

cv_df.groupby('model_name').accuracy.mean()
cv_df
from sklearn.model_selection import train_test_split



model = LogisticRegression(random_state=0)



#Split Data 

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)



#Train Algorithm

model.fit(X_train, y_train)



# Make Predictions

y_pred_proba = model.predict_proba(X_test)

y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix

import seaborn as sns



conf_mat = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_mat, annot=True, fmt='d',

            xticklabels=category_id_df.Category.values, yticklabels=category_id_df.Category.values)

plt.ylabel('Actual')

plt.xlabel('Predicted')
from IPython.display import display



for predicted in category_id_df.category_id:

  for actual in category_id_df.category_id:

    if predicted != actual and conf_mat[actual, predicted] >= 2:

      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))

      display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]]['Text'])

      print('')
model.fit(features, labels)
# model.coef_ contains the importance of each feature for each category

model.coef_
from sklearn.feature_selection import chi2



N = 5

for Category, category_id in sorted(category_to_id.items()):

  indices = np.argsort(model.coef_[category_id])   # This time using the model co-eficients / weights

  feature_names = np.array(tfidf.get_feature_names())[indices]

  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]

  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]

  print("# '{}':".format(Category))

  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))

  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))
texts = ["Hooli stock price soared after a dip in PiedPiper revenue growth.",

         "Captain Tsubasa scores a magnificent goal for the Japanese team.",

         "Merryweather mercenaries are sent on another mission, as government oversight groups call for new sanctions.",

         "Beyoncé releases a new album, tops the charts in all of south-east Asia!",

         "You won't guess what the latest trend in data analysis is!"]

text_features = tfidf.transform(texts)

predictions = model.predict(text_features)

for text, predicted in zip(texts, predictions):

  print('"{}"'.format(text))

  print("  - Predicted as: '{}'".format(id_to_category[predicted]))

  print("")
import os

print(os.listdir("../input/"))
TEST_PATH = os.path.join("../input/", "BBC News Test.csv")



#Load the data using pandas : Create a DataFrame

test_df = pd.read_csv(TEST_PATH)
test_features = tfidf.transform(test_df.Text.tolist())



Y_pred = model.predict(test_features)

Y_pred
# Since all predictions are in terms of "Category IDs (numbers)", need to convert back to Category name

Y_pred_name =[]

for cat_id in Y_pred :

    Y_pred_name.append(id_to_category[cat_id])
Y_pred_name
#Create Submission Dataframe

submission = pd.DataFrame({

        "ArticleId": test_df["ArticleId"],

        "Category": Y_pred_name

    })
# Convert submission dataframe to csv 

# you could use any filename. We choose submission here

submission.to_csv('submission.csv', index=False)
ls
pwd
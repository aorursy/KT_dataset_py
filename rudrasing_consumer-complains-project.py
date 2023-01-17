# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.pipeline import Pipeline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

complains = pd.read_csv("../input/consumer_complaints/Consumer_Complaints.csv",parse_dates = True,low_memory = False)
complains.groupby('State').sum()
complains['complains_true'] = complains['Consumer Complaint'].apply(lambda x : 0 if pd.isna(x) else 1)
complains.groupby('State').sum()['complains_true'].plot()
complains.groupby('Company').sum()['complains_true'].sort_values(ascending = False)[:10]
np.random.seed(10)

remove_n = 1000000

drop_indices = np.random.choice(complains.index, remove_n, replace=False)

df = complains.drop(drop_indices)

df.shape

# Remove above code after mem optimization

df.head()
df = df[['Product','Consumer Complaint']]

df = df[pd.notnull(df['Consumer Complaint'])]

#pd.notnull(df['Consumer Complaint'])
df['category_id'] = df['Product'].factorize()[0]
category_to_id = dict(df[['Product','category_id']].drop_duplicates().values)
id_to_category = dict(df[['category_id','Product']].drop_duplicates().values)
df.groupby('Product').count()['Consumer Complaint'].sort_values(ascending = False)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf = True,min_df = 5,norm = 'l2',encoding = 'latin-1',ngram_range = (1,2),stop_words = 'english')

features = tfidf.fit_transform(df['Consumer Complaint'])

labels = df['category_id']
from sklearn.feature_selection import chi2

for Product,category_id in sorted(category_to_id.items()):

    features_chi2 = chi2(features,labels == category_id)

    index = np.argsort(features_chi2[0])

    feature_names = np.array(tfidf.get_feature_names())[index]

    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]

    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

    print(Product)

    print('most correlated unigrams-->',' '.join(unigrams[-2:]))

    print('most correlated bigrams-->',' '.join(bigrams[-2:]))
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
X_train,X_test,y_train,y_test = train_test_split(df['Consumer Complaint'],df['category_id'],test_size = 0.2)

count_vec = CountVectorizer()

X_train_count = count_vec.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

naive_bays = MultinomialNB()

clf = naive_bays.fit(X_train_tfidf,y_train)
clf.predict(tfidf_transformer.transform(count_vec.transform(["This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."])))
clf.predict(tfidf_transformer.transform(count_vec.transform(["I am disputing the inaccurate information the Chex-Systems has on my credit report. I initially submitted a police report on XXXX/XXXX/16 and Chex Systems only deleted the items that I mentioned in the letter and not all the items that were actually listed on the police report. In other words they wanted me to say word for word to them what items were fraudulent. The total disregard of the police report and what accounts that it states that are fraudulent. If they just had paid a little closer attention to the police report I would not been in this position now and they would n't have to research once again. I would like the reported information to be removed : XXXX XXXX XXXX"])))
# testing with other classifiers with help of cross validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score

models = [RandomForestClassifier(n_estimators = 200,max_depth = 5),

          LogisticRegression(max_iter = 500),

          LinearSVC(),

          MultinomialNB()

         ]

array = []

for model in  models:

    scores = cross_val_score(model,features,labels,cv=5,verbose = 10,n_jobs = -1)

    for cv_num,scores in enumerate(scores):

        model_name = model.__class__.__name__

        array.append((model_name,cv_num,scores))

        score_df = pd.DataFrame(array,columns = ['model_name','cv_name','scores'])

        

    
score_df
X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2)
clf = LinearSVC(C = 0.6)

classifier = clf.fit(X_train,y_train)

pred = classifier.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(pred,y_test))
import sklearn

sklearn.metrics.confusion_matrix(pred,y_test)
model = LinearSVC(C = 0.6)

model.fit(features,labels)

for Product,category_id in sorted(category_to_id.items()):

    indices = np.argsort(model.coef_[category_id])  # in this line when the model is trained we are sorting the coefficients of the model according to the category

    # and the value we get after applying the argsort is the weights distributed in ascenging order with terms with highest weight at the end of the array

    feature_names = np.array(tfidf.get_feature_names())[indices]

    unigrams = [v for v in feature_names if len(v.split(' ')) == 1][-2:]

    bigrams = [v for v in feature_names if len(v.split(' ')) == 2][-2:]

    print(Product)

    print('Unigrams with highest weight for the category ',Product, 'are',unigrams)

    print('bigrams with highest weight for the category ',Product, 'are',bigrams,'\n')
indices
np.array(tfidf.get_feature_names())[np.argsort(model.coef_[0])]
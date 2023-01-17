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


"""

As is data model

I am starrting my code as explained in quick guide without manipulating data.

Based on test results, I am keeping on changing approaches.



Assumptions & Considerations:

    1) Only text column is considered for training the model. 

    keyword & location are ignored.

    2) Stop words are not removed

    3) 100% of training data is taken for training

    

"""



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, metrics, model_selection





train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")



count_vectorizer = feature_extraction.text.CountVectorizer()





train_vectors = count_vectorizer.fit_transform(train_df["text"])

### we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)

print(train_vectors.todense().shape)

#print(count_vectorizer.get_feature_names()) # if curious to see features



## Our vectors are really big, so we want to push our model's weights

## toward 0 without completely discounting different words - ridge regression 

## is a good way to do this.

clf = linear_model.RidgeClassifier()

#scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=2, scoring="f1")

clf.fit(train_vectors, train_df["target"])



# Based on ths model I am predicting results for all train set

pred_1 = clf.predict(train_vectors)

f1_1 = metrics.f1_score(train_df["target"], pred_1, labels=None, pos_label=1, average='binary', sample_weight=None)



# Got a good f1 score, checking the records which our algorithm could not predict correctly

train1=pd.concat([train_df,pd.DataFrame(pred_1)], axis=1)

Mismatching=train1[train1.target!=train1[0]]



'''

EDA

if we observe a sample keyword "bioterror" from Mismatching id 898

and go to train1 to analyze all rows with keyword bioterror,

we can observe that even correctly predicted records dont make sense. 

For example both rows of ids: 834 & 836 have same content with different urls.

The text in url does not really help algorithm to predict target unless we open that webpage & scrape that content.

So, even though our algorithm predicted correctly, it is overfit.

As a next step, we can remove http address & analyze.



Validations of assumptions:

1) Keyword is only a word of text. We can observe a lot of entries with different texts & targets for same keyword "bioterror"

   So, we can ignore keyword. 

   In th eabove example, if we open 2 urls from ids 834 & 836, there is no much difference.

   But target is 1 for a row & 0 for other. I see this data as noice.

   In this case even location does not substantiate choosing target so.

   So, we can ignore location

2) Removing stop words is reducing accuracy. 

   Keeping Stop words in tis case seem to help taking right decision besides the fact we can treat url address to be a stop word & remove

   AFter removing these urls, we may see less accuracy in our model, but at least we are not overfit. 

3) After seeing noice of data, it is better to remove Mismatching items from input dataset    



'''



### remove https

# we can observe in all cases, any info after url address is not useful, so we cab truncate from http word

train_df.text=train_df.text.apply(lambda x:x.split(" http")[0])

test_df.text=test_df.text.apply(lambda x:x.split(" http")[0])

### remove https



# Repeat same code after removing urls

train_vectors = count_vectorizer.fit_transform(train_df["text"])

print(train_vectors.todense().shape)

clf = linear_model.RidgeClassifier()

clf.fit(train_vectors, train_df["target"])

pred_1 = clf.predict(train_vectors)

f1_1 = metrics.f1_score(train_df["target"], pred_1, labels=None, pos_label=1, average='binary', sample_weight=None)

train1=pd.concat([train_df,pd.DataFrame(pred_1)], axis=1)

Mismatching1=train1[train1.target!=train1[0]]

# Repeat same code after removing urls



# Remove bad data Mismatching1 & model

train_df=train_df.loc[~train_df.index.isin(Mismatching1.index)].reset_index()

train_vectors = count_vectorizer.fit_transform(train_df["text"])

print(train_vectors.todense().shape)

clf = linear_model.RidgeClassifier()

clf.fit(train_vectors, train_df["target"])

pred_1 = clf.predict(train_vectors)

f1_1 = metrics.f1_score(train_df["target"], pred_1, labels=None, pos_label=1, average='binary', sample_weight=None)

train2=pd.concat([train_df,pd.DataFrame(pred_1)], axis=1)

Mismatching2=train2[train2.target!=train2[0]]

# Remove bad data Mismatching1 & model





# Final step is to trandform the model to test file to find target values

# note that we're NOT using .fit_transform() here. Using just .transform() makes sure

# that the tokens in the train vectors are the only ones mapped to the test vectors - 

# i.e. that the train and test vectors use the same set of tokens.

test_vectors = count_vectorizer.transform(test_df["text"])

pred_2 = clf.predict(test_vectors)

sample_submission=pd.concat([test_df.id, pd.DataFrame(pred_2)], axis=1)

sample_submission.columns=['id','target']

sample_submission.to_csv("/kaggle/working/sample_submission.csv", index=False)







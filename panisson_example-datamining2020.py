%pylab inline

import pandas as pd
train_data = pd.read_csv("../input/datamining2020/train_data.csv", encoding="utf8")
train_data.head()
train_data.author.unique().shape
target = pd.read_csv("../input/datamining2020/train_target.csv")
target.head()
subreddits = train_data.subreddit.unique()

subreddits_map = pd.Series(index=subreddits, data=arange(subreddits.shape[0]))
from scipy import sparse
def extract_features(group):

    group_subreddits = group['subreddit'].values

    idxs = subreddits_map[group_subreddits].values

    v = sparse.dok_matrix((1, subreddits.shape[0]))

    for idx in idxs:

        if not np.isnan(idx):

            v[0, idx] = 1

    return v.tocsr()



extract_features(train_data[train_data.author=='RedThunder90'])
features_dict = {}



for author, group in train_data.groupby('author'):

    features_dict[author] = extract_features(group)
X = sparse.vstack([features_dict[author] for author in target.author])

X
y = target.gender
def extract_text(group):

    group_text = group['body'].values

    return " ".join(group_text)



extract_text(train_data[train_data.author=='RedThunder90'])
text_dict = {}



for author, group in train_data.groupby('author'):

    text_dict[author] = extract_text(group)
author_text = [text_dict[author] for author in target.author]

author_text[0][:100]
# YOUR CODE HERE



class Model():

    def predict_proba(self, X):

        return np.zeros((X.shape[0], 2))

    

model = Model()
test_data = pd.read_csv("../input/datamining2020/test_data.csv", encoding="utf8")
features_dict = {}



for author, group in test_data.groupby('author'):

    features_dict[author] = extract_features(group)
X_test = sparse.vstack([features_dict[author] for author in test_data.author.unique()])

X_test
y_pred = model.predict_proba(X_test)[:,1]
solution = pd.DataFrame({"author":test_data.author.unique(), "gender":y_pred})

solution.head()
# solution.to_csv("solution.csv", index=False)
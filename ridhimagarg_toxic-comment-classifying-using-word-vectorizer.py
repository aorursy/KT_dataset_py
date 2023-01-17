import pandas as pd
data_train_file  = "../input/train.csv"
train_data = pd.read_csv(data_train_file)
data_test_file  = "../input/test.csv"
test_data = pd.read_csv(data_test_file)

print(train_data)
print(train_data.columns)
print(len(train_data))
def remove_punctuation(text):
    import string
    text = text.lower()
    translator =str.maketrans('', '', string.punctuation)
    return (text.translate(translator))

train_data['comment'] = train_data['comment_text'].apply(remove_punctuation)
train_data['comment'] = train_data['comment'].replace('\n','', regex=True)
print(train_data['comment'])
test_data['comment'] = test_data['comment_text'].apply(remove_punctuation)
test_data['comment'] = test_data['comment'].replace('\n','', regex=True)
print(test_data)

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer

# Taking both train data and test data to calculate features at once
data = train_data.append(test_data)
#fetching comment for imput
Comment = data['comment']
Comment = Comment.tolist()
#fetching the output labels for each comment
y_train = train_data.iloc[:,2:8]
y_train = np.array(y_train)



## using scikit learn to calculate count vector for each comment
from sklearn.feature_extraction.text import CountVectorizer
## Creating the vectorizer
vectorizer = CountVectorizer()
vectorizer


## Calculating count vector of dataset and to be used as an input for my model.
X = vectorizer.fit_transform(Comment)
X_train = X[0:159571]
X_test = X[159571:]



## Creating classifer one vs rest through SVM for training my model
classif = OneVsRestClassifier(SGDClassifier(loss="hinge", penalty="l2"))
classif.fit(X_train, y_train)
y_predict = classif.predict(X_test)
classif.score(X[3000:159571], y_train[3000:])

from collections import OrderedDict
id = test_data['id']
id.tolist()
toxic = y_predict[0:,0]
toxic.tolist()
severe_toxic = y_predict[0:,1]
severe_toxic.tolist()
obscene = y_predict[0:,2]
obscene.tolist()
threat = y_predict[0:,3]
threat.tolist()
insult = y_predict[0:,4]
insult.tolist()
identity_hate = y_predict[0:,5]
identity_hate.tolist()


df = pd.DataFrame( OrderedDict({'id':id,'toxic':toxic,'severe_toxic':severe_toxic,'obscene':obscene,'threat':threat,'insult':insult,'identity_hate':identity_hate}  ) )
print(df)


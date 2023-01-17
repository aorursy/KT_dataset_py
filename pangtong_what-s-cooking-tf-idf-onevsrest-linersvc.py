# Import the requried libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import json
import os
print(os.listdir('../input'))

# Dataser Prepare
print('Read Dataset...')
def read_data(path):
    return json.load(open(path))
train = read_data('../input/train.json')
test = read_data('../input/test.json')

# Text Data Feature
print('Prepare text data of train and test...')
def generate_text(data):
    text_data = [' '.join(doc['ingredients']).lower() for doc in data]
    return text_data
train_text = generate_text(train)
test_text = generate_text(test)
target = [doc['cuisine'] for doc in train]

# Feature Engineering
print('TF-IDF on text data...')
tfidf = TfidfVectorizer(binary=True)
def tfidf_features(txt, flag):
    if flag == 'train':
        x = tfidf.fit_transform(txt)
    else:
        x = tfidf.transform(txt)
    x = x.astype('float64')
    return x
X = tfidf_features(train_text, flag='train')
X_test = tfidf_features(test_text, flag='test')

# Label Encoding - Target
print('Label Encode The Target Variable...')
lb = LabelEncoder()
y = lb.fit_transform(target)

# Model Training
# classifier = LinearSVC(dual=False,
#                        random_state=0)
# model = OneVsRestClassifier(classifier, n_jobs=-1)
classifier = DecisionTreeClassifier(criterion='entropy',
                                    )
model = OneVsRestClassifier(classifier, n_jobs=-1)
model.fit(X, y)

# Prediction
print('Predict On Test Data...')
y_test = model.predict(X_test)
y_pred = lb.inverse_transform(y_test)

# Submission
print('Generate Submission File...')
test_id = [doc['id'] for doc in test]
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('output.csv', index=False)
print(sub.head())
import numpy as np
import pandas as pd
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def read_json_to_dataframe(filepath, test_file=False):
    with open(filepath) as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    df['url'] = df.images.map(lambda x: x['url'][0])
    if not test_file:
        df['image_id'] = df.annotations.map(lambda x: x['image_id'])
        df['label_id'] = df.annotations.map(lambda x: x['label_id'])
        df.drop(columns=['annotations', 'images'], inplace=True)
    return df
train_data = read_json_to_dataframe('../input/train.json', test_file=False)
validation_data = read_json_to_dataframe('../input/validation.json', test_file=False)
test_data = read_json_to_dataframe('../input/test.json', test_file=True)
print("Train size: ", train_data.shape)
print("Validation size: ", validation_data.shape)
print("Test size: ", test_data.shape)
tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=2500, lowercase=False)
tfidf.fit(train_data.url)
train_features = tfidf.transform(train_data.url)
validation_features = tfidf.transform(validation_data.url)
test_features = tfidf.transform(test_data.url)
np.random.seed(0)
random_ids = np.random.choice(np.arange(len(train_data)), size=7500, replace=False)
%%time
lr = LogisticRegression(C=10.0)
lr.fit(train_features[random_ids], train_data.label_id.values[random_ids])
print("Validation error: %.3f" % (1 - accuracy_score(validation_data.label_id, lr.predict(validation_features))))

submission = pd.DataFrame({
    'id': 1 + np.arange(len(test_data)),
    'predicted': lr.predict(test_features),
})
submission.head()
submission.to_csv('submission.csv', index=False)
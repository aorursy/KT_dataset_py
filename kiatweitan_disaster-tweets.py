import pandas as pd
raw_data = pd.read_csv('../input/disaster/train.csv')
raw_data.head()
import spacy
import numpy as np
nlp = spacy.load('en_core_web_lg')
def vectorize(data):
    with nlp.disable_pipes():
        return np.array([nlp(data.text).vector for _, data in data.iterrows()])
raw_test = pd.read_csv('../input/disaster/test.csv')
raw_combined = pd.concat([raw_data, raw_test])
keyword = pd.get_dummies(raw_combined.keyword, dummy_na=True, drop_first=True)
location = pd.get_dummies(raw_combined.location, dummy_na=True, drop_first=True)
vectors = vectorize(raw_combined)
combined = pd.concat((keyword, location), axis=1)
combined = np.concatenate((combined.to_numpy(), vectors), axis=1)
X = combined[:len(raw_data)]
y = raw_data.target.to_numpy()
test = combined[len(raw_data):]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
from sklearn.svm import LinearSVC

model = LinearSVC(random_state=1, dual=False)
model.fit(X_train, y_train)

print(f'Model test accuracy: {model.score(X_test, y_test)*100:.3f}%')
result = model.predict(test)
import csv

with open ('../working/disaster.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'target'])
    for r in zip(raw_test.id, result):
        writer.writerow(r)
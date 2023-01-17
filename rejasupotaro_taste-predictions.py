import json
import re
import unidecode
import numpy as np
import pandas as pd
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from tqdm import tqdm
tqdm.pandas()

from lime import lime_text
from lime.lime_text import LimeTextExplainer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_json('../input/train.json')
df = df.head(10000)

label_encoder = LabelEncoder()
df['y'] = label_encoder.fit_transform(df['cuisine'].values)
class_names = label_encoder.classes_.tolist()

df['num_ingredients'] = df['ingredients'].apply(len)
df = df[df['num_ingredients'] > 1]

df['ingredients'] = df['ingredients'].apply(lambda x: ' '.join(x))

lemmatizer = WordNetLemmatizer()
def preprocess(ingredients_text):
    ingredients_text = ingredients_text.lower()
    ingredients_text = ingredients_text.replace('-', ' ')
    words = []
    for word in ingredients_text.split():
        if re.findall('[0-9]', word): continue
        if len(word) <= 2: continue
        if '’' in word: continue
        word = re.sub('[.,()!]', '', word)
        word = lemmatizer.lemmatize(word)
        if len(word) > 0: words.append(word)
    return ' '.join(words)

vectorizer = make_pipeline(
    FunctionTransformer(lambda x: [preprocess(ingredients) for ingredients in x], validate=False),
    TfidfVectorizer(sublinear_tf=True, stop_words='english'),
    FunctionTransformer(lambda x: x.astype('float16'), validate=False)
)
train, val = train_test_split(df, random_state=42)

x_train = vectorizer.fit_transform(train['ingredients'].values)
x_val = vectorizer.transform(val['ingredients'].values)
y_train = train['y'].values
y_val = val['y'].values
estimator = SVC(
    C=50,
    kernel='rbf',
    gamma=1.4,
    coef0=1,
    cache_size=500,
    probability=True
)
classifier = OneVsRestClassifier(estimator, n_jobs=-1)
classifier.fit(x_train, y_train)
y_pred = label_encoder.inverse_transform(classifier.predict(x_val))
val['pred'] = y_pred
y_true = label_encoder.inverse_transform(y_val)

print(f'accuracy score on train data: {accuracy_score(y_true, y_pred)}')

def report2dict(cr):
    rows = []
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0: rows.append(parsed_row)
    measures = rows[0]
    classes = defaultdict(dict)
    for row in rows[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            classes[class_label][m.strip()] = float(row[j + 1].strip())
    return classes
report = classification_report(y_true, y_pred)
pd.DataFrame(report2dict(report)).T
explainer = LimeTextExplainer(class_names=class_names)
classifier_fn = make_pipeline(vectorizer, classifier).predict_proba
def explain(recipe):
    display(recipe[['id', 'cuisine', 'pred', 'ingredients']])
    text_instance = recipe['ingredients'].values[0]
    explainer.explain_instance(
        text_instance,
        classifier_fn,
        top_labels=1,
        num_features=6
    ).show_in_notebook()
val[val['cuisine'] == 'japanese'].head(10)[['id', 'cuisine', 'pred', 'ingredients']]
explain(val[val['id'] == 26634])
explain(val[val['id'] == 17628])
explain(val[val['id'] == 36372])
explain(val[val['id'] == 11331])
explain(val[val['id'] == 49040])
my_recipe = '''
1. Add the chicken and sweet peppers to the cooker.
2. Leave the tortillas and chedder cheese.
3. Finish with sprinkled olives and a dollop of sour cream on top.
'''
explainer.explain_instance(
    my_recipe,
    classifier_fn,
    top_labels=1,
    num_features=6
).show_in_notebook()
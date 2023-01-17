import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# text feature stuff
from nltk import word_tokenize, download
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

# classifiers
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import ExtraTreesClassifier
train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")

y_tr = train.cuisine
train.head()
lemmatizer = WordNetLemmatizer()

def lemmatize_ingredients(df):
    all_ingredients = set()
    ingredients_list = []

    for i in range(len(df.ingredients)):
        lemmatized_list = []
        for ingr in df.ingredients[i]:
            split_ingr = ingr.split(" ")
            lemmatized = []
            for word in split_ingr:
                lemmatized.append(lemmatizer.lemmatize(word.lower()))
            all_ingredients.add(" ".join(lemmatized))
            lemmatized_list.append(" ".join(lemmatized))
        ingredients_list.append(lemmatized_list)        
        
    df['ingredients_lem'] = ingredients_list    
    
    return df, all_ingredients
train, all_ingredients = lemmatize_ingredients(train)
test, _ = lemmatize_ingredients(test)
test.head()
def arraytotext(records): return [" ".join(record).lower() for record in records]
tfidf = TfidfVectorizer(binary=True)
train_tfidf_features = tfidf.fit_transform(arraytotext(train.ingredients_lem))
test_tfidf_features= tfidf.transform(arraytotext(test.ingredients_lem))
et = ExtraTreesClassifier(n_estimators=25, max_depth=300, min_samples_split=5, min_samples_leaf=1, random_state=None, min_impurity_decrease=1e-7)
model = OneVsRestClassifier(et)
model.fit(train_tfidf_features, y_tr)
predictions = model.predict(test_tfidf_features)
submission = pd.DataFrame()
submission['id'] = test.id
submission['cuisine'] = predictions
submission.to_csv('20180718_ova_etc_4.csv', index=False)
# et2 = ExtraTreesClassifier(n_estimators=25, max_depth=300, min_samples_split=2, min_samples_leaf=1, random_state=None, min_impurity_decrease=1e-7)
# et2.fit(train_tfidf_features, y_tr)
# predictions_2 = et2.predict(test_tfidf_features)

# submission2 = pd.DataFrame()
# submission2['id'] = test.id
# submission2['cuisine'] = predictions
# submission2.to_csv('20180718_etc_1.csv', index=False)

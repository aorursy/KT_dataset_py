import pandas as pd
from pprint import pprint
from time import time
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
sns.set_context('talk')
train = pd.read_json('../input/train.json', orient='columns')
test = pd.read_json('../input/test.json', orient='columns')
f, ax = plt.subplots(figsize=(5,6))
sns.countplot(y = 'cuisine', 
                   data = train,
                  order = train.cuisine.value_counts(ascending=False).index)
ingredients_individual = Counter([ingredient for ingredient_list in train.ingredients for ingredient in ingredient_list])
ingredients_individual = pd.DataFrame.from_dict(ingredients_individual,orient='index').reset_index()
ingredients_individual = ingredients_individual.rename(columns={'index':'Ingredient', 0:'Count'})
sns.barplot(x = 'Count', 
            y = 'Ingredient',
            data = ingredients_individual.sort_values('Count', ascending=False).head(10))
f, ax = plt.subplots(figsize=(15,10))
sns.barplot(x='number_ingredients_meal',
            y='number_meals',
            data= (train.ingredients.map(lambda l: len(l))
                    .value_counts()
                    .sort_index()
                    .reset_index()
                    .rename(columns={'index':'number_ingredients_meal', 'ingredients':'number_meals'}))
            )
f, ax = plt.subplots(figsize=(32,15))
sns.boxplot(x='cuisine',
            y='number_ingredients',
            data= (pd.concat([train.cuisine,train.ingredients.map(lambda l: len(l))], axis=1)
                    .rename(columns={'ingredients':'number_ingredients'}))
            )
train_ingredients_text = train.ingredients.apply(lambda s: ' '.join(w.lower() for w in s)).str.replace('[^\w\s]','')
test_ingredients_text = test.ingredients.apply(lambda s: ' '.join(w.lower() for w in s)).str.replace('[^\w\s]','')
lb = LabelEncoder()
train_y = lb.fit_transform(train.cuisine)
#pipeline = Pipeline([
#    ('tfidf', TfidfVectorizer()),
#    ('clf', RandomForestClassifier())
#])
#parameters = {
#    'tfidf__use_idf': (True, False),
#    'tfidf__norm': ('l1', 'l2'),
#    'clf__n_estimators': (50,150)
#}
#if __name__ == '__main__':
#    grid_search = GridSearchCV(pipeline, parameters, cv=5,
#                               n_jobs=-1, verbose=1)

#    print("Performing grid search...")
#    print("pipeline:", [name for name, _ in pipeline.steps])
#    print("parameters:")
#    pprint(parameters)
#    t0 = time()
#    grid_search.fit(train_ingredients_text, train_y)
#    print("done in %0.3fs" % (time() - t0))
#    print()

#    print("Best score: %0.3f" % grid_search.best_score_)
#    print("Best parameters set:")
#    best_parameters = grid_search.best_estimator_.get_params()
#    for param_name in sorted(parameters.keys()):
#        print("\t%s: %r" % (param_name, best_parameters[param_name]))

#pred_y = grid_search.predict(test_ingredients_text)
clf = RandomForestClassifier(n_estimators=150)
vectorizer = TfidfVectorizer(norm='l2',use_idf=True)
train_x = vectorizer.fit_transform(train_ingredients_text)
test_x = vectorizer.transform(test_ingredients_text)
clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)
test_id = [test_id for test_id in test.id]
sub = pd.DataFrame({'id': test_id, 'cuisine': lb.inverse_transform(pred_y)}, columns=['id', 'cuisine'])
sub.to_csv('predicitions.csv', index=False)
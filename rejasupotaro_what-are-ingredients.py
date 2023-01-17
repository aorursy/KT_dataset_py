import json
import langdetect
import re
import time
import unidecode
import ipywidgets as widgets
import numpy as np
import pandas as pd
from ipywidgets import interact
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, MultiLabelBinarizer
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
train.head()
df = pd.concat([train, test], sort=False)
df['ingredients_text'] = df['ingredients'].apply(lambda x: ', '.join(x))
df['num_ingredients'] = df['ingredients'].apply(lambda x: len(x))
raw_ingredients = [ingredient for ingredients in df.ingredients.values for ingredient in ingredients]
df.head()
from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(figsize=(16,4))
sns.countplot(x='num_ingredients', data=df)
df[df['num_ingredients'] <= 1]
[ingredient for ingredient in raw_ingredients if len(ingredient) <= 2]
' '.join(sorted([char for char in set(' '.join(raw_ingredients)) if re.findall('[^A-Za-z]', char)]))
list(set([ingredient for ingredient in raw_ingredients if re.findall('[A-Z]+', ingredient)]))[:5]
list(set([ingredient for ingredient in raw_ingredients if '’' in ingredient]))
list(set([ingredient for ingredient in raw_ingredients if re.findall('-', ingredient)]))[:5]
list(set([ingredient for ingredient in raw_ingredients if re.findall('[0-9]', ingredient)]))[:5]
units = ['inch', 'oz', 'lb', 'ounc', '%'] # ounc is a misspelling of ounce?

@interact(unit=units)
def f(unit):
    ingredients_df = pd.DataFrame([ingredient for ingredient in raw_ingredients if unit in ingredient], columns=['ingredient'])
    return ingredients_df.groupby(['ingredient']).size().reset_index(name='count').sort_values(['count'], ascending=False)
keywords = [
    # It indicates the cusine directly
    'american', 'greek', 'filipino', 'indian', 'jamaican', 'spanish', 'italian', 'mexican', 'chinese', 'thai',
    'vietnamese', 'cajun', 'creole', 'french', 'japanese', 'irish', 'korean', 'moroccan', 'russian',
    # Region names I found in the dataset
    'tokyo', 'shaoxing', 'california'
]

@interact(keyword=keywords)
def f(keyword):
    ingredients_df = pd.DataFrame([ingredient for ingredient in raw_ingredients if keyword in ingredient], columns=['ingredient'])
    return ingredients_df.groupby(['ingredient']).size().reset_index(name='count').sort_values(['count'], ascending=False)
accents = ['â', 'ç', 'è', 'é', 'í', 'î', 'ú']

@interact(accent=accents)
def f(accent):
    ingredients_df = pd.DataFrame([ingredient for ingredient in raw_ingredients if accent in ingredient], columns=['ingredient'])
    return ingredients_df.groupby(['ingredient']).size().reset_index(name='count').sort_values(['count'], ascending=False)
lemmatizer = WordNetLemmatizer()
def preprocess(ingredients):
    ingredients = ' '.join(ingredients).lower().replace('-', ' ')
    ingredients = re.sub("\d+", "", ingredients)
    return [lemmatizer.lemmatize(ingredient) for ingredient in ingredients.split()]

ingredients_df = df.groupby(['cuisine'])['ingredients'].sum().apply(lambda ingredients: preprocess(ingredients)).reset_index()
unique_ingredients = []
for cuisine in ingredients_df['cuisine'].unique():
    target = set(ingredients_df[ingredients_df['cuisine'] == cuisine]['ingredients'].values[0])
    others = set(ingredients_df[ingredients_df['cuisine'] != cuisine]['ingredients'].sum())
    unique_ingredients.append({
        'cuisine': cuisine,
        'ingredients': target - others
    })
pd.DataFrame(unique_ingredients, columns=['cuisine', 'ingredients'])
text_languages = []
for text in [
    'ein, zwei, drei, vier',
    'purée',
    'taco',
    'tofu',
    'tangzhong',
    'xuxu',
]:
    text_languages.append({
        'text': text,
        'detected language': langdetect.detect(text)
    })
pd.DataFrame(text_languages, columns=['text', 'detected language'])
from IPython.display import clear_output

ingredients = ['romaine lettuce', 'Eggs', 'Beef demi-glace', 'Sugar 10g', 'Pumpkin purée', 'Kahlúa']
labels = [widgets.Label(ingredient) for ingredient in ingredients]

lower_checkbox = widgets.Checkbox(value=False, description='lower', indent=False)
lemmatize_checkbox = widgets.Checkbox(value=False, description='lemmatize', indent=False)
remove_hyphens_checkbox = widgets.Checkbox(value=False, description='remove hyphens', indent=False)
remove_numbers_checkbox = widgets.Checkbox(value=False, description='remove numbers', indent=False)
strip_accents_checkbox = widgets.Checkbox(value=False, description='strip accents', indent=False)

lemmatizer = WordNetLemmatizer()
def lemmatize(sentence):
    return ' '.join([lemmatizer.lemmatize(word) for word in sentence.split()])
assert lemmatize('eggs') == 'egg'

def remove_numbers(sentence):
    words = []
    for word in sentence.split():
        if re.findall('[0-9]', word): continue
        if len(word) > 0: words.append(word)
    return ' '.join(words)

def update_ingredients(widget):
    for i, ingredient in enumerate(ingredients):
        processed = ingredient
        if lower_checkbox.value: processed = processed.lower()
        if lemmatize_checkbox.value: processed = lemmatize(processed)
        if remove_hyphens_checkbox.value: processed = processed.replace('-', ' ')
        if remove_numbers_checkbox.value: processed = remove_numbers(processed)
        if strip_accents_checkbox.value: processed = unidecode.unidecode(processed)
        if processed == ingredient:
            labels[i].value = ingredient
        else:
            labels[i].value = f'{ingredient} => {processed}'

lower_checkbox.observe(update_ingredients)
lemmatize_checkbox.observe(update_ingredients)
remove_hyphens_checkbox.observe(update_ingredients)
remove_numbers_checkbox.observe(update_ingredients)
strip_accents_checkbox.observe(update_ingredients)

display(widgets.VBox([
    widgets.Box([lower_checkbox, lemmatize_checkbox, remove_hyphens_checkbox, remove_numbers_checkbox, strip_accents_checkbox]),
    widgets.VBox(labels)
]))
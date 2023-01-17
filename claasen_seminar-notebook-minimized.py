import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.multiclass import OneVsRestClassifier
nltk.download('wordnet')
def char_preprocess_ingredients(ingredient):
    ingredient = ingredient.lower() #Kleinschreibung anwenden
    ingredient = re.sub("[^a-zA-Z] ","",ingredient) #Sonderzeichen und Zahlen entfernen
    ingredient = re.sub((r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b'), ' ', ingredient) #Gewichtseinheiten entfernen
    ingredient = re.sub(r'\s+', ' ', ingredient) #Doppelte Lerrzeichen entfernen
    ingredient = " ".join(ingredient.split())
    return ingredient
def porter_stem_ingredients(ingredient):
    porter_stemmer = PorterStemmer()
    tokens = ingredient.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    ingredient = ' '.join(stemmed_tokens)
    return ingredient
def snowball_stem_ingredients(ingredient):
    snowball_stemmer = SnowballStemmer('english')
    tokens = ingredient.split()
    stemmed_tokens = [snowball_stemmer.stem(token) for token in tokens]
    ingredient = ' '.join(stemmed_tokens)
    return ingredient
def wordnet_lemmatizer_ingredients(ingredient):
    lemmatizer = WordNetLemmatizer()
    tokens = ingredient.split()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    ingredient = ' '.join(lemmatized_tokens)
    return ingredient
def main_preprocessing(ingredient):
    ingredient = char_preprocess_ingredients(ingredient)
    ingredient = wordnet_lemmatizer_ingredients(ingredient)
    ingredient = snowball_stem_ingredients(ingredient)
    ingredient = porter_stem_ingredients(ingredient)
    return ingredient
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_json('../input/train.json')

df_train['ingredients'] = df_train['ingredients'].apply(lambda x : [main_preprocessing(y) for y in x])
df_train['all_ingredients'] = df_train['ingredients'].map(";".join)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word", 
    ngram_range = (1,1),
    binary = True,
    tokenizer = None,    
    preprocessor = None, 
    stop_words = None,  
    max_df = 0.99)
X = vectorizer.fit_transform(df_train['all_ingredients'].values)
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(df_train.cuisine)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=1, solver='warn',
          tol=0.1, verbose=0, warm_start=False)
logistic.fit(X_train, y_train)
print('Accuracy: %.5f' % logistic.score(X_test, y_test))
ovrLR = OneVsRestClassifier(estimator=logistic, n_jobs=1)
ovrLR.fit(X_train, y_train)
print('Accuracy: %.5f' % ovrLR.score(X_test, y_test))
from sklearn.svm import LinearSVC
lsvc = LinearSVC(C=0.2, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=100,
     multi_class='ovr', penalty='l2', random_state=1, tol=0.0001,
     verbose=0)
lsvc.fit(X_train, y_train)
print('Accuracy: %.5f' % lsvc.score(X_test, y_test))
ovrLSVC = OneVsRestClassifier(estimator=lsvc, n_jobs=1)
ovrLSVC.fit(X_train, y_train)
print('Accuracy: %.5f' % ovrLSVC.score(X_test, y_test))
from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
           max_depth=None, max_features=0.25, max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=None,
           oob_score=False, random_state=1, verbose=0, warm_start=False)
etc.fit(X_train, y_train)
print('Accuracy: %.5f' % etc.score(X_test, y_test))
ovrETC = OneVsRestClassifier(estimator=etc, n_jobs=1)
ovrETC.fit(X_train, y_train)
print('Accuracy: %.5f' % ovrETC.score(X_test, y_test))
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100, 100), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=1, shuffle=True, solver='adam', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
mlp.fit(X_train, y_train)
print('Accuracy: %.5f' % mlp.score(X_test, y_test))
ovrMLP = OneVsRestClassifier(estimator=mlp, n_jobs=1)
ovrMLP.fit(X_train, y_train)
print('Accuracy: %.5f' % ovrMLP.score(X_test, y_test))
from sklearn.ensemble import VotingClassifier
vc = VotingClassifier(n_jobs=1, estimators=[
    ('clf1', ovrMLP),
    ('clf2', ovrETC),
    ('clf3', ovrLSVC),
    ('clf4', ovrLR)
])
vc.fit(X_train, y_train)
print('Accuracy: %.5f' % vc.score(X_test, y_test))
df_test = pd.read_json('../input/test.json')
df_test['ingredients'] = df_test['ingredients'].apply(lambda x : [main_preprocessing(y) for y in x])
df_test['all_ingredients'] = df_test['ingredients'].map(";".join)
#vectorizer = CountVectorizer(vocabulary = features)
X = vectorizer.transform(df_test['all_ingredients'].values)
y_pred = vc.predict(X)
y_pred = enc.inverse_transform(y_pred)
y_pred = pd.DataFrame({'cuisine' : y_pred , 'id' : df_test.id }, columns=['id', 'cuisine'])
y_pred.to_csv('submission.csv', index = False)
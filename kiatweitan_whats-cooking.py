import zipfile
import os

input_dir = '../input/whats-cooking'
for f in os.listdir(input_dir):
    with zipfile.ZipFile(os.path.join(input_dir, f), 'r') as zip_ref:
        zip_ref.extractall()

import numpy as np
import pandas as pd
raw_train = pd.read_json('train.json')
raw_test = pd.read_json('test.json')
raw_combined = pd.concat((raw_train, raw_test))
n_cuisine = raw_combined.cuisine.nunique()

corpus = [' '.join(r) for r in raw_combined.ingredients]
raw_train.head()
raw_combined['wc'] = raw_combined.ingredients.apply(lambda l: len(l))
raw_combined.groupby('cuisine')['wc'].agg(np.median)
### BoW
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
vec = cv.fit_transform(corpus)
vec_arr = vec.toarray()
### Top 10 ingredients by cuisine
df = pd.DataFrame(vec_arr[:len(raw_train),], columns=cv.get_feature_names())
df['cuisine']  = raw_train.cuisine
agg = df.groupby(['cuisine']).sum()
v = df.var().sort_values(ascending=False)
feat = []
for v, i in zip(v, v.index):
    if v > 0.01:
        feat.append(i)
raw_combined.filter(feat)
for i in agg.index:
    top = agg[agg.index.isin([i])].sort_values(i, axis=1, ascending=False)
    print(i.upper(), '\t', ' '.join([l for l in top.columns[:10]]))
### Tf Idf
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer()
vec = tv.fit_transform(corpus)
vec_arr = vec.toarray()
### Word2Vec
import spacy
nlp = spacy.load('en_core_web_lg')
with nlp.disable_pipes():
    vec_arr = np.array([
        nlp(' '.join(data.ingredients)).vector 
        for _, data in raw_combined.iterrows()
    ])
### PCA
from sklearn.decomposition import PCA

pca = PCA()
vec_arr = pca.fit_transform(vec_arr)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)
## standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
vec_arr = scaler.fit_transform(vec_arr)
### LDA
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=1000, random_state=0)
vec_arr = lda.fit_transform(vec_arr)
### SVD
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=100)
vec_arr = svd.fit_transform(vec_arr)
print(svd.explained_variance_ratio_.sum())
X = vec_arr[:len(raw_train)]
raw_train.cuisine = raw_train.cuisine.astype('category')
y = raw_train.cuisine.cat.codes
test = vec_arr[len(raw_train):]
### F score select
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

selector = SelectKBest(f_classif, k=100)
X = selector.fit_transform(X, y)
test = selector.transform(test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
### Ridge Reg
from sklearn.linear_model import RidgeClassifier

model_ridge = RidgeClassifier(random_state=1)
model_ridge.fit(X_train, y_train)

print(f'Model test accuracy: {model_ridge.score(X_test, y_test)*100:.3f}%')
result_ridge = model_ridge.predict(test)
### SGD
from sklearn.linear_model import SGDClassifier

model_sgd = SGDClassifier(random_state=1)
model_sgd.fit(X_train, y_train)

print(f'Model test accuracy: {model_sgd.score(X_test, y_test)*100:.3f}%')
result_sgd = model_sgd.predict(test)
### SVC
from sklearn.svm import LinearSVC

model_svc = LinearSVC(random_state=1, dual=False)
model_svc.fit(X_train, y_train)

print(f'Model test accuracy: {model_svc.score(X_test, y_test)*100:.3f}%')
result_svc = model_svc.predict(test)
### Decision Tree
from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)

print(f'Model test accuracy: {model_dt.score(X_test, y_test)*100:.3f}%')
result_dt = model_dt.predict(test)
### Multinomial NB
from sklearn.naive_bayes import MultinomialNB

model_mnb = MultinomialNB()
model_mnb.fit(X_train, y_train)

print(f'Model test accuracy: {model_mnb.score(X_test, y_test)*100:.3f}%')
result_mnb = model_mnb.predict(test)
### L1 feature select
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

l1 = SelectFromModel(model_logit, prefit=True)
X_train = l1.transform(X_train)
X_test = l1.transform(X_test)
test = l1.transform(test)
### Logistic
from sklearn.linear_model import LogisticRegression

model_logit = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000)
model_logit.fit(X_train, y_train)

print(f'Model test accuracy: {model_logit.score(X_test, y_test)*100:.3f}%')
result_logit = model_logit.predict(test)
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(X_train, feature_names=cv.get_feature_names(), discretize_continuous=True)
idx = 0
exp = explainer.explain_instance(X_test[idx,], model_logit.predict_proba, num_features=X_test.shape[1])
exp.show_in_notebook(show_table=True, show_all=False)
exp.show_in_notebook(show_table=True, show_all=False)
### Voting
from sklearn.ensemble import VotingClassifier

estimators = [('logit', model_logit), ('svc', model_svc), ('mnb', model_mnb), ('ridge', model_ridge), ('sgd', model_sgd)]
model_vote = VotingClassifier(estimators=estimators)
model_vote.fit(X_train, y_train)

print(f'Model test accuracy: {model_vote.score(X_test, y_test)*100:.3f}%')
result_vote = model_vote.predict(test)
### Stacked
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.ensemble import StackingClassifier
estimators = [('logit', model_logit), ('svc', model_svc), ('mnb', model_mnb), ('sgd', model_sgd), ('ridge', model_ridge)]
model_stack = StackingClassifier(estimators=estimators, 
                                 final_estimator=LogisticRegression(
                                     random_state=0, 
                                     solver='lbfgs', 
                                     multi_class='multinomial',
                                     max_iter=3000)
                                )
model_stack.fit(X_train, y_train)

print(f'Model test accuracy: {model_stack.score(X_test, y_test)*100:.3f}%')
result_stack = model_stack.predict(test)
### ADA Boost
from sklearn.ensemble import AdaBoostClassifier

model_ada = AdaBoostClassifier(base_estimator=model_logit, n_estimators=100, random_state=0)
model_ada.fit(X_train, y_train)

print(f'Model test accuracy: {model_ada.score(X_test, y_test)*100:.3f}%')
result_ada = model_ada.predict(test)
### Bagging Logit
from sklearn.ensemble import BaggingClassifier

model_bag = BaggingClassifier(base_estimator=model_logit, n_estimators=5, random_state=0)
model_bag.fit(X_train, y_train)

print(f'Model test accuracy: {model_bag.score(X_test, y_test)*100:.3f}%')
result_bag = model_bag.predict(test)
### MLP
from sklearn.neural_network import MLPClassifier

model_mlp = MLPClassifier(random_state=0, max_iter=300)
model_mlp.fit(X_train, y_train)

print(f'Model test accuracy: {model_mlp.score(X_test, y_test)*100:.3f}%')
result_mlp = model_mlp.predict(test)
### K-means
from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier(n_neighbors=n_cuisine)
model_knn.fit(X_train, y_train)

print(f'Model test accuracy: {model_knn.score(X_test, y_test)*100:.3f}%')
result_knn = model_knn.predict(test)
### LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model_lda = LinearDiscriminantAnalysis()
model_lda.fit(X_train, y_train)

print(f'Model test accuracy: {model_lda.score(X_test, y_test)*100:.3f}%')
result_lda = model_lda.predict(test)
### XGB
import xgboost as xgb
from sklearn.metrics import accuracy_score

xgb_model = xgb.XGBRegressor(objective="multi:softmax", num_class=n_cuisine, random_state=42)
xgb_model.fit(X_train, y_train)
### XGB (cont'd)
y_pred = [round(pred) for pred in xgb_model.predict(X_test)]
print(f'Model test accuracy: {accuracy_score(y_pred, y_test)*100:.3f}%')
result_xgb = xgb_model.predict(test)
### NN
import numpy as np
from keras import backend
from keras import layers
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

backend.clear_session()
    
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

encoder = LabelEncoder()
encoder.fit(y)
def process_y(y):
    return np_utils.to_categorical(encoder.transform(y))

input_dim = X_train.shape[1]
y_train_enc = process_y(y_train)
y_test_enc = process_y(y_test)

skf = StratifiedKFold(n_splits = 3, random_state = 42, shuffle = True)
fold = 1

for idx_train, idx_val in skf.split(np.zeros(len(X_train)), y_train):
    model_nn = Sequential()
#     model_nn.add(layers.Embedding(input_dim, 300))
    model_nn.add(layers.Dense(30, activation='relu'))
    model_nn.add(layers.Dropout(0.1))
#     model_nn.add(layers.Dense(10, activation='relu'))
    model_nn.add(layers.Dense(n_cuisine, activation='softmax'))
    model_nn.compile(optimizer='Nadam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
#     model_nn.summary()

    
    train_x = X_train[idx_train]
    val_x = X_train[idx_val]
    train_y = y_train_enc[idx_train]
    val_y = y_train_enc[idx_val]

    checkpoint = ModelCheckpoint(f'model_{fold}.h5', 
                                 monitor='val_accuracy', verbose=0, 
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history = model_nn.fit(train_x, train_y,
                        epochs=50,
                        verbose=False,
                        validation_data=(val_x, val_y),
                        callbacks=callbacks_list,
                        batch_size=int(len(idx_train)/30))
    model_nn.load_weights(f"model_{fold}.h5")

    print('Fold ', fold)
    results = model_nn.evaluate(val_x, val_y, verbose=False)
    results = dict(zip(model_nn.metrics_names,results))
    print('/t', 'val:', results)
    results = model_nn.evaluate(X_test, y_test_enc, verbose=False)
    results = dict(zip(model_nn.metrics_names,results))
    print('/t', 'test:', results)
    backend.clear_session()
    fold += 1

result_nn = model_nn.predict_classes(test)
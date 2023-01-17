import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, cross_val_score

import statsmodels.discrete.discrete_model as sm
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

import time
chunksize = 10**5
chunks = pd.read_csv('../input/xtrain.csv',chunksize=chunksize, iterator=True)

X = pd.concat(chunks)
X.head()
print('Initial size: {}'.format(X.shape))
print('After NaN omit size: {}'.format(X.dropna().shape))
X = X.fillna(method='bfill').fillna(method='ffill')
X.head()
chunks = pd.read_csv('../input/ytrain.csv',chunksize=chunksize, iterator=True)

y = pd.concat(chunks)
y.describe()
y = np.array(y).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
model = sm.Logit(y_train, X_train)
result = model.fit()
print(result.summary())
sig_columns = [i for i,x in enumerate(result.pvalues.ravel()) if x<=0.1]

X_train.iloc[:,sig_columns].head()
model = sm.Logit(y_train, X_train.iloc[:,sig_columns])
result = model.fit()
print(result.summary())
pca = PCA()

pca.fit(X_train)
plt.figure(figsize=(15, 8))
features = range(pca.n_components_)
plt.plot(features, pca.explained_variance_ratio_.cumsum(),'--o', label='cumulative explained variance ratio')
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.legend()
plt.show()
plt.figure(figsize=(15,8))
plt.plot(range(0,58),X_train.var().cumsum(),'--o', label="cumulative feature variance")
plt.legend()
plt.show()
scaler = StandardScaler()

pca = PCA()

pipeline = make_pipeline(scaler, pca)

pipeline.fit(X_train)

plt.figure(figsize=(15, 8))
features = range(pca.n_components_)
plt.plot(features, pca.explained_variance_ratio_.cumsum(),'--o', label='normalized cumulative explained variance ratio')
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.legend()
plt.show()
pca_X_train = pipeline.transform(X_train)
pca_X_test = pipeline.transform(X_test)

print('pca_X_train shape: {}'.format(pca_X_train.shape))
print('pca_X_test shape: {}'.format(pca_X_test.shape))
def compute_models(X_train, y_train, X_test, y_test):
    results={}
    def test_model(model):
        start_time = time.time()
        
        model.fit(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        model_probs = model.predict_proba(X_test)
        test_log_loss = log_loss(y_test, model_probs)
        cv_acc = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5)
        
        scores= [cv_acc.mean(), test_accuracy, test_log_loss, (time.time() - start_time)]
        return scores
    m = LogisticRegression()
    results['Logistic Regression'] = test_model(m)
    
    for i in range(6, 15, 4):
        m = DecisionTreeClassifier(max_depth=3, min_samples_leaf=i)
        results['Decision tree {}'.format(i)] = test_model(m)
    
    for i in range(60,150,40):
        m = RandomForestClassifier(n_estimators=i)
        results['Random forest {}'.format(i)] = test_model(m)
    
    for i in range(6,15,4):
        m = KNeighborsClassifier(n_neighbors = i)
        results['KNN {}'.format(i)] = test_model(m)
        
    m = SVC(probability=True)
    results['SVM'] = test_model(m)
    
    results = pd.DataFrame.from_dict(results,orient='index')
    results.columns=["Train mean accuracy", "Test accuracy", "Test log loss", "Calculation time (sec)"] 
    results=results.sort_values(by=["Test accuracy","Test log loss"],ascending=[False,True])

    return results
limit = 10**3*5
no_pca_results = compute_models(X_train[:limit], y_train[:limit], X_test[:limit], y_test[:limit])
no_pca_results
pca_results = compute_models(pca_X_train[:limit], y_train[:limit], pca_X_test[:limit], y_test[:limit])
pca_results
pca = PCA(n_components=1)

pca.fit(X_train)

pca1_X_train = pca.transform(X_train)
pca1_X_test = pca.transform(X_test)

print('pca_X_train shape: {}'.format(pca1_X_train.shape))
print('pca_X_test shape: {}'.format(pca1_X_test.shape))
pca_results = compute_models(pca1_X_train[:limit], y_train[:limit], pca1_X_test[:limit], y_test[:limit])
pca_results
def draw_models(model_str,X_train, y_train, X_test, y_test, range_):
    test_accuracy = [0]*len(range_)
    test_log_loss = [0]*len(range_)
    def test_model(model):
        model.fit(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        model_probs = model.predict_proba(X_test)
        test_log_loss = log_loss(y_test, model_probs)
        return test_accuracy, test_log_loss
    if (model_str=='DT'):
        for i,x in enumerate(range_):
            m = DecisionTreeClassifier(max_depth=3, min_samples_leaf=x)
            test_accuracy[i], test_log_loss[i] = test_model(m)
    else:
        for i,x in enumerate(range_):
            m = KNeighborsClassifier(n_neighbors = x)
            test_accuracy[i], test_log_loss[i] = test_model(m)
    
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.plot(range_, test_accuracy,'--o', label='Test accuracy')
    plt.legend()
    plt.subplot(122)
    plt.plot(range_, test_log_loss,'--o', label='Test log loss')
    plt.legend()
    plt.show()
limit = 10**5
draw_models('DT',pca1_X_train[:limit], y_train[:limit], pca1_X_test[:limit], y_test[:limit],range(2,20,4))
draw_models('KNN',pca1_X_train[:limit], y_train[:limit], pca1_X_test[:limit], y_test[:limit],range(10,19))

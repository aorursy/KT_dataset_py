# tools
import numpy as np # linear algebra
np.random.seed(42)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
# models
from sklearn.naive_bayes import BernoulliNB # Naive Bayes
from sklearn.neighbors import KNeighborsClassifier # KNN Classifier
from sklearn import ensemble # RandomForestClassifier(), GradientBoostingClassifier()
from sklearn import linear_model # LogisticRegression(penalty = 'l1') OLS, RidgeClassifier(), LogisticRegression(penalty = 'l2') Lasso Classifier
from sklearn import svm # SVC()
data_path = '../input/data.csv'
df = pd.read_csv(data_path)
df.drop(['Unnamed: 32', 'id'], axis = 1, inplace = True)

df['diagnosis'].loc[df['diagnosis'] == 'M'] = 0
df['diagnosis'].loc[df['diagnosis'] == 'B'] = 1
df['diagnosis'] = df['diagnosis'].astype(int)
display(df.shape)
display(df.head())
'''
for col in df.columns:
    df[col].hist()
    plt.title(col)
    plt.show()
    ''';
# skewed towards 0: concavity, radius_se, perimeter_se, area_se, compactness_se, concavity_se, fractal_dimension_se, area_worst, concavity_worst
# min-max normalize
df_norm = (df - df.min()) / (df.max() - df.min())
# correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(df_norm.corr())
# take care of correlated features
X = df_norm.drop('diagnosis', axis = 1)
Y = df_norm['diagnosis']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
df_eval = pd.DataFrame()
def evaluate(model, model_name):
    model.fit(X_train,Y_train)
    model_pred = model.predict(X_test)
    evals = np.zeros(2)
    # accuracy
    evals[0] = (pd.DataFrame([model_pred,Y_test]).all(axis = 0).sum() + pd.DataFrame([1 - model_pred,1 - Y_test]).all(axis = 0).sum()) / len(Y_test.index)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, model_pred, pos_label=1)
    evals[1] = metrics.auc(fpr, tpr)
    df_eval[model_name] = evals
# runtime ~ 0 seconds (default train_test_split)
#### Baseline: Guess malignant = 0
start = time.time()
base_pred = np.zeros(len(Y_test) - 1)
base_pred = np.append(base_pred, 1)
print ("Runtime %0.2f" % (time.time() - start))
df_eval['Baseline'] = [(pd.DataFrame([base_pred,Y_test]).all(axis = 0).sum() + pd.DataFrame([1 - base_pred,1 - Y_test]).all(axis = 0).sum()) / len(Y_test.index), 0]
# runtime ~ 0.24 seconds (default train_test_split)
#### Naive Bayes
start = time.time()
bnb = BernoulliNB()
evaluate(bnb,'Naive Bayes')
print ("Runtime %0.2f" % (time.time() - start))
# runtime ~ 447.93 seconds (default train_test_split)
#### KNN Classifier
start = time.time()
knn_model = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', n_jobs = 3)
evaluate(knn_model, 'KNN')
print ("Runtime %0.2f" % (time.time() - start))
# runtime ~ 7.58 seconds (default train_test_split)
#### RandomForestClassifier 
start = time.time()
rfc = ensemble.RandomForestClassifier(n_jobs = 3)
evaluate(rfc, 'Random Forest')
print ("Runtime %0.2f" % (time.time() - start))
# runtime ~ 4.91 seconds (default train_test_split)
#### Logistic Regression
start = time.time()
log_reg_model = linear_model.LogisticRegression(penalty = 'l2', C=1e9)
evaluate(log_reg_model, 'Logistic Regression')
print ("Runtime %0.2f" % (time.time() - start))
# runtime ~ 0.21 seconds (default train_test_split)
#### Ridge Classifier
start = time.time()
ridge_model = linear_model.RidgeClassifier(alpha = 1e-4)
evaluate(ridge_model, 'Ridge Regression')
print ("Runtime %0.2f" % (time.time() - start))
# runtime ~ 1.21 seconds (default train_test_split)
#### Lasso Classifier
start = time.time()
lasso_model = linear_model.LogisticRegression(penalty = 'l1', C = 10, tol = 1e-6)
evaluate(lasso_model, 'Lasso Regression')
print ("Runtime %0.2f" % (time.time() - start))
# runtime ~ 13.47 seconds (default train_test_split)
#### SVClassifier
start = time.time()
svc_model = svm.SVC(C = 100, kernel = 'sigmoid')
evaluate(svc_model, 'SVM')
print ("Runtime %0.2f" % (time.time() - start))
# runtime ~ 840.21 seconds (default train_test_split)
# n_estimators = 1000, max_depth = 4, subsample, 0.5, learning_rate = 0.001
#### Gradient Boost Classifier
start = time.time()
params = {'n_estimators': 500,
          'max_depth': 4,
          'learning_rate': 0.1,
          'loss': 'exponential'}
gbc = ensemble.GradientBoostingClassifier(**params)
evaluate(gbc, 'Gradient Boost')
print ("Runtime %0.2f" % (time.time() - start))
df_eval.rename(index={0: 'Accuracy', 1: 'AUC Score'}, inplace = True)
# Accuracy scores, highlight highest in each row
df_eval.style.highlight_max(axis = 1)
lasso_params = pd.Series(lasso_model.coef_[0], index = X_test.columns)
display(lasso_params[lasso_params != 0].sort_values())

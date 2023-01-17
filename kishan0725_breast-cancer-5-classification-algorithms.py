import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data.head(10)
data.shape
data.columns
data.isna().sum()
# removing id and unnamed: 32 column which is not necessary for our model

data = data.drop(['id','Unnamed: 32'],axis = 1)
data.head(10)
# As our dataset is balanced (around 60-40 ratio), there is no need to balance our data

data.diagnosis.value_counts(normalize = True)
# Mapping our target variable to 1 and 0

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['diagnosis'] = le.fit_transform(data['diagnosis'])
data.diagnosis.value_counts(normalize=True)
# Finding correlation among features using sns' heatmap

plt.figure(figsize=(20,20))

sns.heatmap(data.corr(),annot=True,cmap='coolwarm')
# removing features that are less correlated with our target variable

data.corr().diagnosis[data.corr().diagnosis<=0.2]
less_corr = data.corr().diagnosis[data.corr().diagnosis<=0.2].index
data = data.drop(less_corr,axis=1)
data.shape
# Standardizing our features except target variable

from sklearn.compose import make_column_transformer

from sklearn.preprocessing import StandardScaler

stand_scale = data.drop(['diagnosis'],axis = 1)

col_trans = make_column_transformer(

            (StandardScaler(), stand_scale.columns),

            remainder = 'passthrough')
from sklearn.model_selection import train_test_split

X = data.drop(['diagnosis'], axis = 1)

y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
col_trans.fit_transform(X_train)
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline

logreg = LogisticRegression(solver='lbfgs')

pipe = make_pipeline(col_trans,logreg)
from sklearn.model_selection import cross_val_score

print('Accuracy score on Train data: {}'.format(cross_val_score(pipe, X_train, y_train, cv=10, scoring='accuracy').mean()*100))
pipe = make_pipeline(col_trans,logreg)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

from sklearn import metrics

print('Accuracy score on Test data: {}'.format(metrics.accuracy_score(y_test,y_pred)*100))
from sklearn.neighbors import KNeighborsClassifier

knn_scores = []

for k in range(1,31):

    knn_classifier = KNeighborsClassifier(n_neighbors = k)

    pipe = make_pipeline(col_trans,knn_classifier)

    knn_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())
plt.figure(figsize=(16,16))

plt.plot([k for k in range(1, 31)], knn_scores, color = 'red')

for i in range(1,31):

    plt.text(i, knn_scores[i-1], (i, round(knn_scores[i-1]*100,2)))

plt.xticks([i for i in range(1, 31)])

plt.xlabel('Number of Neighbors (K)')

plt.ylabel('Scores')

plt.title('K Neighbors Classifier scores for different K values')
print('Accuracy score on Train data: {}'.format(knn_scores[4]*100))
knn_classifier = KNeighborsClassifier(n_neighbors = 4)

pipe = make_pipeline(col_trans,knn_classifier)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('Accuracy score on Test Data: {}'.format(metrics.accuracy_score(y_test,y_pred)*100))
from sklearn.svm import SVC

svc_scores = []

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for i in range(len(kernels)):

    svc_classifier = SVC(kernel = kernels[i])

    pipe = make_pipeline(col_trans,svc_classifier)

    svc_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())
from matplotlib.cm import rainbow

import numpy as np

colors = rainbow(np.linspace(0, 1, len(kernels)))

plt.figure(figsize=(10,7))

plt.bar(kernels, svc_scores, color = colors)

for i in range(len(kernels)):

    plt.text(i, svc_scores[i], svc_scores[i])

plt.xlabel('Kernels')

plt.ylabel('Scores')

plt.title('Support Vector Classifier scores for different kernels')
print('Accuracy score on Train data: {}'.format(svc_scores[2]*100))
svc_classifier = SVC(kernel = 'rbf')

pipe = make_pipeline(col_trans,svc_classifier)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('Accuracy score on Test data: {}'.format(metrics.accuracy_score(y_test,y_pred)*100))
from sklearn.tree import DecisionTreeClassifier

dt_scores = []

for i in range(1, len(X.columns) + 1):

    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)

    pipe = make_pipeline(col_trans,dt_classifier)

    dt_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())
plt.figure(figsize=(10,10))

plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green')

for i in range(1, len(X.columns) + 1):

    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))

plt.xticks([i for i in range(1, len(X.columns) + 1)])

plt.xlabel('Max features')

plt.ylabel('Scores')

plt.title('Decision Tree Classifier scores for different number of maximum features')
print('Accuracy score on Train data: {}'.format(dt_scores[3]*100))
dt_classifier = DecisionTreeClassifier(max_features = 4, random_state = 0)

pipe = make_pipeline(col_trans,dt_classifier)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('Accuracy  score on Test data: {}'.format(metrics.accuracy_score(y_test,y_pred)*100))
from sklearn.ensemble import RandomForestClassifier

rf_scores = []

estimators = [10, 100, 200, 500, 1000]

for i in estimators:

    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)

    pipe = make_pipeline(col_trans,rf_classifier)

    rf_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())
plt.figure(figsize=(10,7))

colors = rainbow(np.linspace(0, 1, len(estimators)))

plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)

for i in range(len(estimators)):

    plt.text(i, rf_scores[i], round(rf_scores[i],5))

plt.xticks(ticks = [i for i in range(len(estimators))], labels = [str(estimator) for estimator in estimators])

plt.xlabel('Number of estimators')

plt.ylabel('Scores')

plt.title('Random Forest Classifier scores for different number of estimators')
print('Accuracy score on Train data: {}'.format(rf_scores[4]*100))
rf_classifier = RandomForestClassifier(n_estimators = 1000, random_state = 0)

pipe = make_pipeline(col_trans,rf_classifier)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('Accuracy score on Test data: {}'.format(metrics.accuracy_score(y_test,y_pred)*100))
pipe = make_pipeline(col_trans,logreg)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print('Confusion Matrix - Training Dataset')

print(pd.crosstab(y_train, pipe.predict(X_train), rownames = ['True'], colnames = ['Predicted'], margins = True))
4/165
print('Confusion Matrix - Testing Dataset')

print(pd.crosstab(y_test, y_pred, rownames = ['True'], colnames = ['Predicted'], margins = True))
# Checking False Negative Rate

2/47
from sklearn.metrics import precision_score,recall_score,f1_score

print('Precision Score: {}'.format(precision_score(y_test,y_pred)));

print('Recall Score: {}'.format(recall_score(y_test,y_pred)))

print('F1 Score: {}'.format(f1_score(y_test,y_pred)))
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(y_test, y_pred)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt

from inspect import signature



precision, recall, _ = precision_recall_curve(y_test, y_pred)



# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument

step_kwargs = ({'step': 'post'}

               if 'step' in signature(plt.fill_between).parameters

               else {})

plt.step(recall, precision, color='b', alpha=0.2,

         where='post')

plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(

          average_precision))
from sklearn.metrics import roc_auc_score

print('ROC AUC Score: {}'.format(roc_auc_score(y_test, y_pred)))
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test,y_pred)

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve for breast cancer prediction')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
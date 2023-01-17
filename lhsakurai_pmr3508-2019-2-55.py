import pandas as pd

import sklearn

import matplotlib.pyplot as plt

from sklearn import preprocessing
train_data = "../input/uci-data/adult_data.csv"

test_data = "../input/uci-data/adult_test.csv"
adult = pd.read_csv(train_data, skipinitialspace = True, na_values = "?")
adult.shape
adult.head()
nadult = adult.dropna()
all_attributes = ['age',

 'workclass',

 'fnlwgt',

 'education',

 'education_num',

 'marital_status',

 'occupation',

 'relationship',

 'race',

 'sex',

 'capital_gain',

 'capital_loss',

 'hours_per_week',

 'native_country',

 'target']
fit_adult = nadult[all_attributes].apply(preprocessing.LabelEncoder().fit_transform)
fit_adult.head()
nadult.describe(include="all")
fit_adult.describe(include='all')
low_income_df = nadult[nadult['target'] == '<=50K']

high_income_df = nadult[nadult['target'] == '>50K']
low_income_df.head()
def plot_compare(attribute, type_plot):

    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(12,6))

    low_income_df[attribute].value_counts().plot(ax = axes[0],kind=type_plot, subplots=True)

    high_income_df[attribute].value_counts().plot(ax = axes[1],kind=type_plot, subplots=True)

    axes[0].title.set_text('<=50k')

    axes[1].title.set_text('>50k')

    plt.show()
plot_compare('workclass', 'bar')
plot_compare('education', 'bar')
plot_compare('education_num', 'bar')
plot_compare('marital_status', 'bar')
plot_compare('occupation', 'bar')
plot_compare('relationship', 'bar')
plot_compare('race', 'pie')
plot_compare('sex', 'bar')
plot_compare("capital_gain", "bar")
plot_compare("capital_loss", "bar")
plot_compare("hours_per_week", "bar")
nadult.shape
test_adult = pd.read_csv(test_data, skipinitialspace = True, na_values="?")
test_adult.shape
ntest_adult = test_adult.dropna()
ntest_adult.shape
attributes = ["sex", "workclass", "education_num", "occupation", "marital_status"]
train_adult_x = fit_adult[attributes]
train_adult_y = nadult.target
test_adult_x = ntest_adult[attributes].apply(preprocessing.LabelEncoder().fit_transform)
test_adult_y = ntest_adult.target
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
# columns = ['neighbors', 'cross', 'accuracy']

# results = [columns]

# for n in range (30, 60):

#     neighbors = n

#     cross = 10

#     knn = KNeighborsClassifier(n_neighbors = neighbors)

#     scores = cross_val_score(knn, train_adult_x, train_adult_y, cv = cross)

# #     scores

#     knn.fit(train_adult_x, train_adult_y)

#     test_pred_y = knn.predict(test_adult_x)

# #     test_pred_y

#     acc = accuracy_score(test_adult_y, test_pred_y)

#     results.append([neighbors,cross, acc])
# results
neighbors = 53

cross = 10

knn = KNeighborsClassifier(n_neighbors = neighbors)

scores = cross_val_score(knn, train_adult_x, train_adult_y, cv = cross)

scores
knn.fit(train_adult_x, train_adult_y)

test_pred_y = knn.predict(test_adult_x)

test_pred_y
accuracy_score(test_adult_y, test_pred_y)
from time import time

train_time = {}

acc = {}
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(solver="newton-cg")

start = time()

logisticRegr.fit(train_adult_x, train_adult_y)

end = time()

train_time['log_regr'] = end-start
log_regr_pred = logisticRegr.predict(test_adult_x)

log_reg_acc = accuracy_score(log_regr_pred, test_pred_y)

acc['log_regr'] = log_reg_acc
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 50, max_depth = 10)

start = time()

rfc.fit(train_adult_x, train_adult_y)

end = time()

train_time['rfc'] = end-start
rfc_pred = rfc.predict(test_adult_x)

rfc_acc = accuracy_score(rfc_pred, test_pred_y)

acc['rfc'] = rfc_acc
from sklearn import svm
svm_clf = svm.SVC(gamma="auto")

start = time()

svm_clf.fit(train_adult_x, train_adult_y)

end = time()

train_time['svm'] = end-start
svm_clf_pred = svm_clf.predict(test_adult_x)

svm_clf_acc = accuracy_score(svm_clf_pred, test_pred_y)

acc['svm'] = svm_clf_acc
train_time
acc
kaggle_data = '../input/kaggle-data/test_data.csv'

kaggle_adult = pd.read_csv(kaggle_data)

kaggle_adult_x = kaggle_adult[attributes].apply(preprocessing.LabelEncoder().fit_transform)

kaggle_pred_y = logisticRegr.predict(kaggle_adult_x)

id_index = pd.DataFrame({'Id' : list(range(len(kaggle_pred_y)))})

income = pd.DataFrame({'income' : kaggle_pred_y})

result = id_index.join(income)
result.to_csv("submission.csv", index = False)
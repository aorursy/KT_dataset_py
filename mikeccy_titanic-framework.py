import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/test.csv')
dataset.columns
dataset.head()
dataset.dtypes
dataset.describe()
def dist(dat, col):
    dat[col].hist()
    plt.ylabel('number')
    plt.xlabel(col)
    plt.ylabel('%s dist' % col)
    plt.show()

    dat[dat.Survived == 0][col].hist()
    plt.ylabel('number')
    plt.xlabel(col)
    plt.title('%s dist of ppl died' % col)
    plt.show()

    dat[dat.Survived == 1][col].hist()
    plt.ylabel('number')
    plt.xlabel(col)
    plt.title('%s dist of ppl survived' % col)
    plt.show()
df = pd.DataFrame(dict(map(
    lambda s: (s, dataset.Survived[dataset.Sex == s].value_counts()),
    dataset.Sex.unique()
)))
df.plot(kind = 'bar', stacked = True)
plt.title('survived by sex')
plt.xlabel('survived')
plt.ylabel('count')
plt.show()
dist(dataset, 'Age')
dist(dataset, 'Fare')
dataset.Pclass.hist()
plt.show()
print(dataset.Pclass.isnull().values.any())

df = pd.DataFrame(dict(map(
    lambda pc: (pc, dataset.Survived[dataset.Pclass == pc].value_counts()),
    dataset.Pclass.unique()
)))
df.plot(kind = 'bar', stacked = True)
plt.title('suvived by pclass')
plt.xlabel('pclass')
plt.ylabel('count')
plt.show()
df = pd.DataFrame(dict(map(
    lambda em: (em, dataset.Survived[dataset.Embarked == em].value_counts()),
    dataset.Embarked.unique()
)))
df.plot(kind = 'bar', stacked = True)
plt.title("survived by embarked")
plt.xlabel("survival") 
plt.ylabel("count")
plt.show()
label = dataset.loc[:, 'Survived']
data = dataset.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
testdata = testset.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
print(data.shape)
print(data)
def fill_nan(data):
    data_copy = data.copy(deep = True)
    cols = ['Age', 'Fare', 'Pclass',]
    for col in cols:
        data_copy.loc[:, col] = data_copy[col].fillna(data[col].median())
    data_copy.loc[:,'Sex'] = data_copy['Sex'].fillna('female')
    data_copy.loc[:,'Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy

data_no_nan = fill_nan(data)
testdata_no_nan = fill_nan(testdata)

print(testdata.isnull().values.any())    
print(testdata_no_nan.isnull().values.any())
print(data.isnull().values.any())   
print(data_no_nan.isnull().values.any())

print(data_no_nan)

print(data_no_nan['Sex'].isnull().values.any())

def transfer_sex(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1
    return data_copy

data_after_sex = transfer_sex(data_no_nan)
testdata_after_sex = transfer_sex(testdata_no_nan)
print(testdata_after_sex)
def transfer_embark(data):
    data_copy = data.copy(deep = True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy

data_after_embarked = transfer_embark(data_after_sex)
testdata_after_embarked = transfer_embark(testdata_after_sex)
print(testdata_after_embarked)
from sklearn.model_selection import train_test_split

data_now = data_after_embarked
testdata_now = testdata_after_embarked

train, val, train_labels, val_labels = train_test_split(data_now, label, random_state = 0, test_size = 0.2)
print(train.shape, val.shape, train_labels.shape, val_labels.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
k_range = range(1, 51)
k_scores = []
for K in k_range:
    clf = KNeighborsClassifier(n_neighbors = K)
    clf.fit(train, train_labels)
    print('K = %d' % K)
    predictions = clf.predict(val)
    score = accuracy_score(val_labels, predictions)
    print(score)
    k_scores.append(score)
    print(classification_report(val_labels, predictions))

plt.plot(k_range, k_scores)
plt.xlabel('k for knn')
plt.ylabel('acc on validation set')
plt.show()
print(np.array(k_scores).argsort())
# 预测
clf = KNeighborsClassifier(n_neighbors = 33)
clf.fit(data_now, label)
result = clf.predict(testdata_now)
# 检测模型precision， recall 等各项指标

from sklearn.model_selection import cross_val_score

k_range = range(1, 51)
k_scores = []
k_errors = []
for K in k_range:
    clf = KNeighborsClassifier(n_neighbors = K)
    scores = cross_val_score(clf, data_now, label, cv = 10)
    k_scores.append(scores.mean())
    k_errors.append(scores.std() * 2)

plt.errorbar(k_range, k_scores, k_errors, linestyle='None', marker='^')
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.estimator_checks import check_estimator

class knn(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors = 3):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.classes_ = unique_labels(y)
        return self
    
    def predict(self, X):
        # Input validation
        X = check_array(X)

        dataset = self.X_
        labels = self.y_
        dataset_size = dataset.shape[0]
        
        results = np.empty(0)
        for x in X:
            diffMat = np.tile(x, (dataset_size, 1)) - dataset
            sqDiffMat = diffMat ** 2
            sumDiffMat = sqDiffMat.sum(axis = 1)
            distances = sumDiffMat ** 0.5
            sortedDistances = distances.argsort()

            classCount = {}
            for i in range(self.n_neighbors):
                vote = labels[sortedDistances[i]]
                classCount[vote] = classCount.get(vote, 0) + 1

            max = 0
            ans = 0
            for k, v in classCount.items():
                if (v > max):
                    ans = k
                    max = v
            results = np.append(results, ans)
        return results

# check_estimator(knn)

k_range = range(1, 51)
k_scores = []
k_errors = []
for K in k_range:
    clf = knn(K)
    scores = cross_val_score(clf, data_now, label, cv = 10)
    k_scores.append(scores.mean())
    k_errors.append(scores.std() * 2)

plt.errorbar(k_range, k_scores, k_errors, linestyle='None', marker='^')
# 预测
clf = KNeighborsClassifier(n_neighbors = 23)
clf.fit(data_now, label)
result = clf.predict(testdata_now)
print(result)
df = pd.DataFrame({
    'PassengerId': testset.PassengerId,
    'Survived': result,
})
df.to_csv('submission.csv', header = True, index = False)
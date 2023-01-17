import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.core.display import HTML 

pd.options.mode.chained_assignment = None  # default='warn'
print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')
X_test = pd.read_csv('../input/test.csv')

y_train = df_train['Survived']
X_train = df_train.drop('Survived', axis=1, inplace=False)

print('number of rows:', X_train.shape[0] + X_test.shape[0])
print('train / test: ', X_train.shape[0], X_test.shape[0])
print('number of cols:', X_train.shape[1])
X_train.head(10)
print('Train columns with null values:\n', X_train.isnull().sum())
print('Test/Validation columns with null values:\n', X_test.isnull().sum())
datasets = [X_train, X_test]
X_train.describe(include='all')
drop_col = ['PassengerId', 'Cabin', 'Ticket']

for dataset in datasets:
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
    dataset.drop(drop_col, axis=1, inplace=True)

print(X_train.isnull().sum())
print(X_test.isnull().sum())
plt.hist(X_train['Fare'])
plt.title('Exponential Distribution of Fares')
plt.xlabel('Fare')
plt.ylabel('# of occurances')
plt.show()
for dataset in datasets:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset.drop(['SibSp','Parch'], axis=1, inplace=True)
    dataset['IsAlone'] = 'yes'
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 'no'
    dataset['Title'] = dataset['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    dataset['Child'] = dataset['Age'] < 16
    dataset.drop(['Name'], axis=1, inplace=True)
    title_names = (dataset['Title'].value_counts() < 10)
    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    dataset['Fare'] = dataset['Fare'].apply(lambda x: 0 if x==0 else np.log(x))
x = np.arange(0.1, 5, 0.1)
y = np.log(x)
plt.plot(x, y)
plt.xlabel('x', size=14)
plt.ylabel('log(x)',size=14)
y = np.repeat(0, len(x))
plt.plot(x, y)
plt.show()

plt.hist(X_train['Fare'])
plt.title('Right skewed distribution of log(Fares)')
plt.xlabel('Fare')
plt.ylabel('# of occurances')
plt.show()

plt.hist(X_train['FamilySize'])
plt.title('Potential outliers in family size')
plt.xlabel('Family Size')
plt.ylabel('# of occurances')
plt.show()
X_train.head(10)
sc = StandardScaler()

for i, dataset in enumerate(datasets):

    df_discrete = dataset.filter(['Pclass','Sex','Embarked','IsAlone','Title','Child'], axis=1)
    df_continuous = dataset.filter(['Age','Fare','FamilySize'], axis=1)
    
    #hot encode discrete variables
    df_discrete = df_discrete.astype(str)
    df_discrete = pd.get_dummies(df_discrete).astype(int)
    
    #standardize continuous variables
    if i == 0:
        std_features = sc.fit_transform(df_continuous.values)
        df_continuous = pd.DataFrame(std_features,
                                     index=df_continuous.index, columns=df_continuous.columns)
        X_train = pd.concat([df_discrete, df_continuous], axis=1)
    elif i == 1:
        std_features = sc.transform(df_continuous.values)
        df_continuous = pd.DataFrame(std_features,
                                     index=df_continuous.index, columns=df_continuous.columns)
        X_test = pd.concat([df_discrete, df_continuous], axis=1)

print(X_train.columns)
X_train.sample(10)
mask = y_train==0
plt.scatter(X_train['Fare'].loc[mask], y_train.loc[mask], label='Not Survived')
plt.scatter(X_train['Fare'].loc[np.logical_not(mask)], y_train.loc[np.logical_not(mask)],
            label='Survived')
plt.xlabel('Fare (std)', size=14)
plt.ylabel('Survived', size=14)
plt.legend(bbox_to_anchor = [1.5, 0.65], fontsize=13)
plt.show()
class LinearRegressionGD:

    def __init__(self, eta=0.0001, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)
lr = LinearRegressionGD()
X_fare = X_train['Fare'].values.reshape(-1,1)
lr.fit(X_fare, y_train)
y_pred = lr.predict(X_fare)
plt.scatter(X_train['Fare'].loc[mask], y_train.loc[mask], label='Not Survived')
plt.scatter(X_train['Fare'].loc[np.logical_not(mask)], y_train.loc[np.logical_not(mask)],
            label='Survived')
plt.scatter(X_train['Fare'], y_pred, label='LR Predictions')
plt.legend(bbox_to_anchor = [1.5, 0.65], fontsize=13)
plt.xlabel('Fare (std)', size=14)
plt.ylabel('Survived', size=14)
plt.show()
plt.plot(lr.cost_)
plt.title('Linear regression GD error', size=14)
plt.ylabel('SSE error', size=14)
plt.xlabel('Number of epochs', size=14)
plt.show()
p = np.arange(0.01, 1, 0.01)
odds = p / (1-p)
plt.plot(p, odds)
plt.xlabel('probability of a positive event', size=14)
plt.ylabel('odds-ratio', size=14)
plt.show()
logit = np.log(odds)
plt.plot(p, logit)
plt.xlabel('probability', size=14)
plt.ylabel('logit(p)', size=14)
plt.show()
def sigmoid(z):
    return 1.0 / (1.0+np.exp(-z))
z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.title('Sigmoid function', size=14)
plt.xlabel('z', size=14)
plt.ylabel('$\phi {z}$', size=14)
plt.plot(z, phi_z)
plt.show()
cost = -np.log(phi_z)
plt.plot(phi_z, cost, label='y = 1')
cost = -np.log(1-phi_z)
plt.plot(phi_z, cost, label='y = 0')
plt.ylabel('J(w)', size=14)
plt.xlabel('$\phi {z}$', size=14)
plt.legend()
plt.show()
class LogisticRegressionGD:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []       
        for i in range(self.n_iter):
            y_val = self.activation(X)
            errors = (y - y_val)
            neg_grad = X.T.dot(errors)
            self.w_[1:] += self.eta * neg_grad
            self.w_[0] += self.eta * errors.sum()
            self.cost_.append(self._logit_cost(y, self.activation(X)))
        return self

    def _logit_cost(self, y, y_val):
        logit = -y.dot(np.log(y_val)) - ((1 - y).dot(np.log(1 - y_val)))
        return logit
    
    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        z = self.net_input(X)
        return self._sigmoid(z)
    
    def predict_proba(self, X):
        return self.activation(X)

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.5, 1, 0)
lr = LogisticRegressionGD()
X_fare = X_train['Fare'].values.reshape(-1,1)
lr.fit(X_fare, y_train)
y_pred = lr.predict_proba(X_fare)
plt.scatter(X_train['Fare'].loc[mask], y_train.loc[mask], label='Not Survived')
plt.scatter(X_train['Fare'].loc[np.logical_not(mask)], y_train.loc[np.logical_not(mask)],
            label='Survived')
plt.scatter(X_train['Fare'], y_pred, label='LR Probabilities')
plt.legend(bbox_to_anchor = [1.5, 0.65], fontsize=13)
plt.xlabel('Fare (std)', size=14)
plt.ylabel('Survived', size=14)
plt.show()
plt.plot(lr.cost_)
plt.title('Logistic regression GD error', size=14)
plt.ylabel('NLL error', size=14)
plt.xlabel('Number of epochs', size=14)
plt.show()
from sklearn.metrics import accuracy_score

y_pred = lr.predict(X_fare)
accuracy = accuracy_score(y_true=y_train, y_pred=y_pred)
print('Logistic Regression Accuracy: {} '.format(accuracy))
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
sfs = SequentialFeatureSelector(lr,
                                k_features=1,
                                forward=False,
                                floating=False,
                                verbose=0,
                                scoring='accuracy',
                                cv=4)

sfs.fit(X_train.values, y_train.values)
fig_sfs = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
plt.title('Squential Backward Selection (w. StdDev)')
plt.grid()
plt.show()
sbs_results = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
feature_idx = list(sbs_results['feature_idx'][8])
feat_regularization = feature_idx
print(X_train.columns[feature_idx])
lr = LogisticRegression()
sfs = SequentialFeatureSelector(lr,
                                k_features=X_train.shape[1],
                                forward=True,
                                floating=True,
                                verbose=0,
                                scoring='accuracy',
                                cv=4)

sfs.fit(X_train.values, y_train.values)
fig_sfs = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
plt.title('Sequential Floating Forward Selection (w. StdDev)')
plt.grid()
plt.show()
sffs_results = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
feature_idx = list(sffs_results['feature_idx'][4])
print(X_train.columns[feature_idx])
def plot_regularization(penalty='l2'):
    weights, params = [], []
    for c in range(-5,5):
        lr = LogisticRegression(C=10**c, penalty=penalty, random_state=0)
        lr.fit(X_train.values, y_train.values)
        weights.append(lr.coef_[0])
        params.append(10**c)
    weights = np.array(weights)
    for i, c in enumerate(feat_regularization):
        plt.plot(params, weights[:, i], label=X_train.columns[c])
    plt.ylabel('weight coefficient', size=13)
    plt.xlabel('C', size=13)
    plt.xscale('log')
    plt.legend(bbox_to_anchor = [1.4, 1.0], fontsize=12)

plot_regularization(penalty='l2')
plt.title('L2 Regularization')
plt.show()
plot_regularization(penalty='l1')
plt.title('L1 Regularization')
plt.show()
from sklearn.feature_selection import SelectFromModel

lr = LogisticRegression(C=0.1, penalty='l1')
lr.fit(X_train.values, y_train.values)
model = SelectFromModel(lr, prefit=True)
feat_mask = model.get_support()
coefficients = np.array(lr.coef_).reshape(-1,1)
print('Selected Weights:')
print(coefficients[feat_mask])
print('Selected Features:')
print(X_train.columns[feat_mask])
print('Not Selected Weights:')
print(coefficients[np.logical_not(feat_mask)])
from mlxtend.plotting import plot_decision_regions

print(sffs_results['feature_idx'][:2])
print(X_train.columns[feature_idx])
X = X_train.iloc[:,[3,19]].values
y = y_train.values

lr = LogisticRegression()
lr.fit(X, y)
plot_decision_regions(X, y, clf=lr)
plt.title('Logistic Regression', size=14)
plt.xlabel('Sex_female', size=14)
plt.ylabel('FamilySize', size=14)
plt.show()

y_pred = lr.predict(X)
accuracy = accuracy_score(y_true=y, y_pred=y_pred)
print('Logistic Regression Accuracy: {} '.format(accuracy))
from sklearn.decomposition import PCA

print(X_train.columns[feature_idx])
X = X_train.iloc[:,[0,3,10,19]].values
pca = PCA(n_components=2)
lr = LogisticRegression()
X = pca.fit_transform(X)
lr.fit(X, y)
plot_decision_regions(X, y, clf=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()
print('Captured {0:.2f}% variance'.format(pca.explained_variance_ratio_.sum()))

y_pred = lr.predict(X)
accuracy = accuracy_score(y_true=y, y_pred=y_pred)
print('Logistic Regression 2D Accuracy: {} '.format(accuracy))

lr = LogisticRegression()
X = X_train.iloc[:,[0,3,10,19]].values
lr.fit(X, y)
y_pred = lr.predict(X)
accuracy = accuracy_score(y_true=y, y_pred=y_pred)
print('Logistic Regression 4D Accuracy: {} '.format(accuracy))
from sklearn.neighbors import KNeighborsClassifier

X = X_train.iloc[:,[0,3,10,19]].values
pca = PCA(n_components=2)
X = pca.fit_transform(X)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
plot_decision_regions(X, y, clf=knn)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()
print('Captured {0:.2f}% variance'.format(pca.explained_variance_ratio_.sum()))

y_pred = knn.predict(X)
accuracy = accuracy_score(y_true=y, y_pred=y_pred)
print('K Nearest Neighbors 2D Accuracy: {} '.format(accuracy))

knn = KNeighborsClassifier(n_neighbors=5)
X = X_train.iloc[:,[0,3,10,19]].values
knn.fit(X, y)
y_pred = knn.predict(X)
accuracy = accuracy_score(y_true=y, y_pred=y_pred)
print('K Nearest Neighbors 4D Accuracy: {} '.format(accuracy))
X = X_train.values
pca = PCA(n_components=2)
lr = LogisticRegression()
X = pca.fit_transform(X)
lr.fit(X, y)
plot_decision_regions(X, y, clf=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()
print('Captured {0:.2f}% variance'.format(pca.explained_variance_ratio_.sum()))

y_pred = lr.predict(X)
accuracy = accuracy_score(y_true=y, y_pred=y_pred)
print('Logistic Regression Accuracy: {} '.format(accuracy))
def build_submission(y_pred):
    df_submission = pd.DataFrame(columns=['PassengerId', 'Survived'])
    df_submission['PassengerId'] = list(range(892, 1310))
    df_submission['Survived'] = y_pred
    return df_submission

def fit_logistic(feat_idx):
    lr = LogisticRegression()
    X = X_train.iloc[:, feat_idx].values
    lr.fit(X, y_train)
    y_pred = lr.predict(X)
    accuracy = accuracy_score(y_true=y_train, y_pred=y_pred)
    
    X = X_test.iloc[:, feat_idx].values
    y_pred = lr.predict(X)
    df_submission = build_submission(y_pred)
    
    return accuracy, df_submission
feat_idx = list(sbs_results['feature_idx'][8])

accuracy, submission = fit_logistic(feat_idx)
print('SBS Training Acc: {0:.5f}'.format(accuracy))

#submission.to_csv('sbs_submission.csv', index=False)
print('SBS Test Acc: 0.78947')
feat_idx = list(sffs_results['feature_idx'][4])

accuracy, submission = fit_logistic(feat_idx)
print('SFFS Training Acc: {0:.5f}'.format(accuracy))
print('SFFS Test Acc: 0.78947')
feat_idx = [c for c in range(len(feat_mask)) if feat_mask[c] == True]

accuracy, submission = fit_logistic(feat_idx)
print('L1 Regularization Training Acc: {0:.5f}'.format(accuracy))
print('L1 Regularization Test Acc: 0.78947')
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

lr = LogisticRegression()
sfs = SequentialFeatureSelector(estimator=lr,
                                k_features=10,
                                forward=True,
                                floating=True,
                                scoring='accuracy',
                                cv=4)

pipe = Pipeline([('sfs', sfs),
                 ('lr', lr)])

param_grid = [
    {'sfs__k_features': [3, 4, 5, 6, 7, 8, 9, 10],
     'sfs__estimator__C': [1000.0, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]}
    ]

gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  n_jobs=1, 
                  cv=4,  
                  refit=True)

gs = gs.fit(X_train.values, y_train.values)
print('Best score:', gs.best_score_)
print("Best parameters via GridSearch", gs.best_params_)
feat_idx = list(gs.best_estimator_.steps[0][1].k_feature_idx_)
print('Best features:', feat_idx)
print(X_train.columns[feat_idx])

lr = LogisticRegression(C=1000.0)
X = X_train.iloc[:, feat_idx].values
lr.fit(X, y_train)
y_pred = lr.predict(X)
accuracy = accuracy_score(y_true=y_train, y_pred=y_pred)
print('Logistic Regression(C=1,000) Train Accuracy: {0:.5f}'.format(accuracy))

y_pred = lr.predict(X_test.iloc[:, feat_idx].values)
#df_submission = build_submission(y_pred)
print('Logistic Regression(C=1,000) Test Accuracy: 0.77990')
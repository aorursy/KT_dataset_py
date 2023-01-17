import pandas as pd
PATH = '../input/lending-club-loan-data/loan.csv'
import warnings
warnings.filterwarnings('ignore')
raw_data = pd.read_csv(PATH)
pd.options.display.max_columns = 2000
raw_data.head()
data = raw_data[['loan_amnt', 'term', 'int_rate', 'installment', 
                 'grade', 'sub_grade', 'emp_length', 
                 'home_ownership', 'annual_inc', 'verification_status', 
                 'purpose', 'dti', 'delinq_2yrs', 'delinq_amnt', 
                 'chargeoff_within_12_mths',  'tax_liens',  
                 'acc_now_delinq', 'inq_last_12m', 'open_il_24m', 
                 'loan_status']]
data.head()
data.isnull().sum()
data = data.dropna(axis = 'index', 
                   subset = ['annual_inc', 'dti', 'delinq_2yrs', 
                             'delinq_amnt', 'chargeoff_within_12_mths', 
                             'tax_liens', 'acc_now_delinq'])
data.isnull().sum()
data = data.fillna(value = {'emp_length' : 'no_info', 
                            'inq_last_12m' : 'no_info', 
                            'open_il_24m':'no_info'}) 
import seaborn as sn
import matplotlib.pyplot as plt
%matplotlib notebook

data_for_corr = data.assign(term = data.term.astype('category').cat.codes,
                            grade = data.grade.astype('category').cat.codes,
                            sub_grade = data.sub_grade.astype('category').cat.codes,
                            emp_length = data.emp_length.astype('category').cat.codes,
                            home_ownership = data.home_ownership.astype('category').cat.codes,
                            verification_status = data.verification_status.astype('category').cat.codes,
                            purpose = data.purpose.astype('category').cat.codes,
                            loan_status = data.loan_status.astype('category').cat.codes,
                            inq_last_12m = data.inq_last_12m.astype('category').cat.codes,
                            open_il_24m = data.open_il_24m.astype('category').cat.codes
                            )

corr_matrix = data_for_corr.corr()
plt.figure(figsize=(18,14))
sn.heatmap(corr_matrix, annot=True, cmap = 'Blues',vmin=-0.1, vmax=1)
plt.title('Correlation matrix')
plt.show()
data = data.drop(columns = ['installment', 'grade', 'sub_grade', 'open_il_24m'])
data_inq_nan = data[data.inq_last_12m == 'no_info']
pd.DataFrame({'where inq_last_12m=NaN': data_inq_nan['loan_status'].value_counts()/len(data_inq_nan), 
              'all dataset'           : data['loan_status'].value_counts()/len(data)}
             ).style.format('{:.2f}')
data_3_statuses = data[(data.loan_status == 'Fully Paid') | (data.loan_status == 'Charged Off') | 
                 (data.loan_status == 'Default')]
data_inq_nan = data_3_statuses[data_3_statuses.inq_last_12m == 'no_info']
pd.DataFrame({'where inq_last_12m=NaN': 
              data_inq_nan['loan_status'].value_counts()/len(data_inq_nan), 
              'all dataset': 
              data_3_statuses['loan_status'].value_counts()/len(data_3_statuses)}
             ).style.format('{:.2f}')
data = data.drop(columns = 'inq_last_12m')
data['loan_status'].value_counts()
data = data[(data.loan_status == 'Fully Paid') | (data.loan_status == 'Charged Off') | 
                 (data.loan_status == 'Default')]
data['loan_status'] = data['loan_status'].replace(to_replace = ['Fully Paid', 'Charged Off', 'Default'], 
                                                       value = [0, 1, 1])
data['loan_status'].value_counts()
data.describe().style.format('{:.2f}')
X = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values
from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [4, 6, 7])],
                       remainder='passthrough')
X = ct.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
def prepare_data(data):
    X = data.iloc[:, 0:-1].values
    y = data.iloc[:, -1].values
    
    labelencoder_X = LabelEncoder()
    X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
    X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
    
    ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [4, 6, 7])],
                       remainder='passthrough')
    X = ct.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test
#X_train, X_test, y_train, y_test = prepare_data(data)
from sklearn.naive_bayes import GaussianNB

classifier_naive_bayes = GaussianNB()
classifier_naive_bayes.fit(X_train, y_train)
y_pred = classifier_naive_bayes.predict(X_test)
from sklearn.metrics import confusion_matrix

def build_conf_matrix(title):
    conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred, normalize='pred'))
    sn.heatmap(conf_matrix, annot=True, cmap='Blues',vmin=0, vmax=1)
    plt.title(title)
    plt.show()
    
build_conf_matrix(title='Naive Bayes, imbalanced dataset')
data_loan_status_1 = data[data['loan_status'] == 1]
data_loan_status_0 = data[data['loan_status'] == 0].sample(n=len(data_loan_status_1))
data_balanced = data_loan_status_1.append(data_loan_status_0) 
X_train, X_test, y_train, y_test = prepare_data(data_balanced)
classifier_naive_bayes = GaussianNB()
classifier_naive_bayes.fit(X_train, y_train)
y_pred = classifier_naive_bayes.predict(X_test)
build_conf_matrix(title='Naive Bayes, balanced dataset')
from sklearn.ensemble import RandomForestClassifier

classifier_rand_forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier_rand_forest.fit(X_train, y_train)
y_pred = classifier_rand_forest.predict(X_test)
build_conf_matrix(title='Random Forest, 10 trees')
from sklearn.model_selection import GridSearchCV

parameters = [ {'n_estimators':[10, 50, 100, 200], 'criterion':['entropy', 'gini']}]
grid_search = GridSearchCV(estimator = classifier_rand_forest, 
                                 param_grid = parameters,
                                 scoring = 'accuracy',
                                 cv = 2,
                                 n_jobs = -1,
                                 verbose = 5)
grid_search = grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_score_
classifier_rand_forest = RandomForestClassifier(n_estimators = 200, criterion = 'gini')
classifier_rand_forest.fit(X_train, y_train)
y_pred = classifier_rand_forest.predict(X_test)
build_conf_matrix(title='Random Forest, 200 trees, gini')
from sklearn.linear_model import LogisticRegression

classifier_log_reg = LogisticRegression()
classifier_log_reg.fit(X_train, y_train)
y_pred = classifier_log_reg.predict(X_test)
build_conf_matrix(title='Logistic Regression')
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier_naive_bayes, X = X_train, y = y_train, cv = 5)
('Naive Bayes',{'accuracy':accuracies.mean(), 'std':accuracies.std()})
accuracies = cross_val_score(estimator = classifier_log_reg, X = X_train, y = y_train, cv = 5)
('Logistic Regression',{'accuracy':accuracies.mean(), 'std':accuracies.std()})
accuracies = cross_val_score(estimator = classifier_rand_forest, X = X_train, y = y_train, cv = 2)
('Random Forest',{'accuracy':accuracies.mean(), 'std':accuracies.std()})
data_loan_status_1 = data[data['loan_status'] == 1].sample(n=10000)
data_loan_status_0 = data[data['loan_status'] == 0].sample(n=len(data_loan_status_1))
data_small = data_loan_status_1.append(data_loan_status_0)
X = data_small.iloc[:, [0, 2]].values
y = data_small.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier_log_reg = LogisticRegression()
classifier_log_reg.fit(X_train, y_train)
y_pred = classifier_log_reg.predict(X_test)
build_conf_matrix(title='Logistic Regression, small dataset with 2 columns')
from matplotlib.colors import ListedColormap
import numpy as np
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 0.3, stop = X_set[:, 0].max() + 0.3, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 0.3, stop = X_set[:, 1].max() + 0.3, step = 0.01))
plt.contourf(X1, X2, classifier_log_reg.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('#024000', '#600000')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('#4dcb49', '#ff3b3b'))(i), label = j, s=5)
plt.xlabel('loan_amnt')
plt.ylabel('int_rate')
plt.show()
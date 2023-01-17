import pandas as pd
train =  pd.read_csv('../input/train.csv')
train = train[['BORO','GRADE','SCORE','CRITICAL FLAG', 'ZIPCODE','CAMIS']]
def score_to_numeric(x):
    if x=='BROOKLYN':
        return 1
    if x=='QUEENS':
        return 2
    if x=='MANHATTAN':
        return 3
    if x=='BRONX':
        return 4
    if x=='STATEN ISLAND':
        return 5
    if x=='Missing':
        return 6
train['BORO'] = train['BORO'].apply(score_to_numeric)
train = train[train['GRADE'].notnull()]
train = train[train['BORO'].notnull()]
train = train[train['SCORE'].notnull()]
train = train[train['CRITICAL FLAG'].notnull()]
train = train[train['ZIPCODE'].notnull()]
test = test[test['CAMIS'].notnull()]

def crit_to_val(x):
    if x=='Critical':
        return 1
    else:
        return 0
train['CRITICAL FLAG'] = train['CRITICAL FLAG'].apply(crit_to_val)

def grade_to_numeric(x):
    if x=='A':
        return 1
    if x=='B':
        return 2
    if x=='C':
        return 3
    if x=='Z':
        return 4
    if x=='P':
        return 5
    if x=='Not Yet Graded':
        return 6
train['GRADE'] = train['GRADE'].apply(grade_to_numeric)



# train.GRADE.unique()
train.head()
test = pd.read_csv('../input/test.csv')
test = test[['BORO','GRADE','SCORE', 'ZIPCODE','CAMIS']]
def score_to_numeric(x):
    if x=='BROOKLYN':
        return 1
    if x=='QUEENS':
        return 2
    if x=='MANHATTAN':
        return 3
    if x=='BRONX':
        return 4
    if x=='STATEN ISLAND':
        return 5
    if x=='Missing':
        return 6
test['BORO'] = test['BORO'].apply(score_to_numeric)
test = test[test['GRADE'].notnull()]
test = test[test['BORO'].notnull()]
test = test[test['SCORE'].notnull()]
test = test[test['ZIPCODE'].notnull()]
test = test[test['CAMIS'].notnull()]


def grade_to_numeric(x):
    if x=='A':
        return 1
    if x=='B':
        return 2
    if x=='C':
        return 3
    if x=='Z':
        return 4
    if x=='P':
        return 5
    if x=='Not Yet Graded':
        return 6
test['GRADE'] = test['GRADE'].apply(grade_to_numeric)



# test.GRADE.unique()
test.head()


from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

x_test=test.values[:, 0:3]
y_test=test.values[:,3]
X = train.values[:, 0:3]
Y = train.values[:,3]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
#                                max_depth=3, min_samples_leaf=5)
# clf_gini.fit(X_train, y_train)
# y_pred = clf_gini.predict(X_test)
# print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)


rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_en = rf.predict(x_test)
y_pred_en
print ("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)
print ("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)
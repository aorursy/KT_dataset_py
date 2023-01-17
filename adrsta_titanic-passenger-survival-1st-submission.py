import pandas as pd
trainset = pd.read_csv('../input/train.csv', index_col='PassengerId')
trainset.shape
trainset.columns
trainset.head()
trainset.describe()
trainset.isnull().sum()
def preprocess(dataset):

    try:

        X = dataset.drop(['Survived'], axis=1)

        y = dataset['Survived']

    except ValueError:

        X = dataset

        y = 0

    ### TODO Drop difficult features for now

    X = X.drop(['Name', 'Cabin', 'Fare', 'Ticket', 'Embarked'], axis=1)

    ### Fill in missing ages

    from sklearn.preprocessing import Imputer

    imputer = Imputer(strategy = 'median')

    X['Age'] = imputer.fit_transform(X[['Age']])

    # Encode categorical data

    X = pd.merge(X, pd.get_dummies(X['Sex']), left_index=True, right_index=True)

    X = X.drop(['Sex', 'female'], axis=1)

    X = X.rename(columns={'male':'Male'})

    X = pd.merge(X, pd.get_dummies(X['Pclass']), left_index=True, right_index=True)

    X = X.drop(['Pclass', 1], axis=1)

    X = X.rename(columns={2:'Pclass2', 3:'Pclass3'})

    return X, y
X, y = preprocess(trainset)

X.head()
y.head()
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()

reg.fit(X, y)

(reg.intercept_, reg.coef_)
reg.score(X, y)
testset = pd.read_csv('../input/test.csv', index_col='PassengerId')

X_test, _ = preprocess(testset)

X_test.describe()
y_pred = reg.predict(X_test)
output = pd.DataFrame()

output['PassengerId'] = X_test.index

output['Survived'] = y_pred

output = output.set_index('PassengerId')

output.to_csv('out.csv')
with open('out.csv', 'r') as fin:

    print(fin.read())
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
def plot_confusion_matrix(y_test, y_pred):

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)

    import seaborn as sn

    df_cm = pd.DataFrame(cm, index = ['Dead', 'Alive'], columns = ['Dead', 'Alive'])

    sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='d', vmin=0, vmax=len(y_pred))

plot_confusion_matrix(y_test, y_pred)    
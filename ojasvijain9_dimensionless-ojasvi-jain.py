import pandas as pd
train = pd.read_csv('/kaggle/input/credit-risk-modeling-case-study/CRM_TrainData.csv')

test = pd.read_csv('/kaggle/input/credit-risk-modeling-case-study/CRM_TestData.csv')
train.head()
test.head()
train.dropna(inplace=True)
train.shape
train.drop(['Loan ID', 'Customer ID'], axis=1, inplace=True)

train.head()
x_train = train.drop(['Loan Status', 'Years in current job', 'Monthly Debt', 'Maximum Open Credit','Purpose', 'Home Ownership','Term'], axis=1)

y_train = train['Loan Status']
x_train.head()
y_train.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y_t = le.fit_transform(y_train)

y_tr = pd.Series(y_t)

y_tr.head()
test.head()
x_test = test.drop(['Customer ID','Unnamed: 2', 'Years in current job', 'Monthly Debt', 'Maximum Open Credit','Purpose', 'Home Ownership','Term'], axis=1)
x_test.head()
x_test.dropna(inplace=True)
x_te = x_test.drop(['Loan ID'], axis=1)
x_te.head()
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()

model = log.fit(x_train, y_tr)

y_pred = model.predict(x_te)

print(y_pred)
final = pd.DataFrame({'Loan ID': x_test['Loan ID'],

                      'Loan Status': y_pred})

final.head()
final.to_csv("/content/submission.csv")
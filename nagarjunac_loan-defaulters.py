import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
data = pd.read_excel(open('../input/default of credit card clients.xls','rb'), sheetname='Data', skiprows=1)
# check the data of first 5 rows using head function.
data.head()
data.shape
data.columns
data.isnull().values.any()
# check if there is any null data 
data[data.isnull().any(axis=1)] 
# add a clumn to insert row numbers for entire dateframe for easy smalping of observations
data ['a'] = pd.DataFrame({'a':range(30001)})
    

# check if the colun a is added 
data.columns
sampled_df = data[(data['a'] % 10) == 0]
sampled_df.shape
sampled_df_remaining = data[(data['a'] % 10) != 0]
sampled_df_remaining.shape
y = sampled_df['default payment next month'].copy()
loan_features = ['LIMIT_BAL','SEX','EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
x = sampled_df[loan_features].copy()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=324)
loan_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
loan_classifier.fit(X_train, y_train)
predictions = loan_classifier.predict(X_test)
predictions[:20]
accuracy_score(y_true = y_test, y_pred = predictions)
X1 = sampled_df_remaining[loan_features].copy()
y1 = sampled_df_remaining['default payment next month'].copy()
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.33, random_state=324)
loan_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
loan_classifier.fit(X1_train, y1_train)
predictions1 = loan_classifier.predict(X1_test)
predictions1[:20]
accuracy_score(y_true = y1_test, y_pred = predictions1)

import numpy as np 

import pandas as pd

from sklearn.model_selection import train_test_split



from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier



from sklearn.metrics import accuracy_score,classification_report,f1_score



from sklearn.impute import SimpleImputer





from sklearn.preprocessing import LabelEncoder



from sklearn.model_selection import GridSearchCV

from xgboost.sklearn import XGBRegressor

print('Compelete Imports')
#File Path Full

file_path_train='../input/titanic/train.csv'

file_path_test='../input/titanic/test.csv'



# Read the train data

df_train = pd.read_csv(file_path_train,index_col='PassengerId')

# Read the test data

df_test=pd.read_csv(file_path_test,index_col='PassengerId')





print('Read Done!')

df_test.columns
# Remove rows with missing target, separate target from predictors



df_train.dropna(axis=0, subset=['Survived'], inplace=True)

# define target

y = df_train.Survived

# Remove target colum

X = df_train.drop(['Survived'], axis=1)

X_test=df_test

X.head()
X_test.head()
category_colums=[colum for colum in X.columns if X[colum].dtype == 'object']

len(category_colums)
X.drop(columns=['Name', 'Cabin', 'Ticket'],inplace=True)

X_test.drop(columns=['Name', 'Cabin', 'Ticket'],inplace=True)
# Get list of categorical variables

s = (X.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
cols_with_missing = [col for col in X.columns

                     if X[col].isnull().any()]

len(cols_with_missing)
# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestClassifier(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return accuracy_score(y_valid, preds)
cols_with_missing = [col for col in X.columns

                     if X[col].isnull().any()]

cols_with_missing
# Imputation

my_imputer = SimpleImputer(strategy='most_frequent')

imputed_X = pd.DataFrame(my_imputer.fit_transform(X))

imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))



# Imputation removed column names; put them back

imputed_X.columns = X.columns

imputed_X_test.columns = X_test.columns

imputed_X

imputed_X_test
cols_with_missing_imputed_X = [col for col in imputed_X.columns

                     if imputed_X[col].isnull().any()]

cols_with_missing_imputed_X
# Make copy to avoid changing original data 

imputed_X_lable = imputed_X.copy()

imputed_X_test_lable = imputed_X_test.copy()



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    imputed_X_lable[col] = label_encoder.fit_transform(imputed_X[col])

    imputed_X_test_lable[col] = label_encoder.transform(imputed_X_test[col])
imputed_X_lable
imputed_X_lable.isna().sum()
y
X_train, X_valid, y_train, y_valid=train_test_split(imputed_X_lable,y)
max_depth =list(range(2, 20,1))

max_depth
results = {}

for i in max_depth:

    dt = DecisionTreeClassifier(max_leaf_nodes=i, random_state=1)

    dt.fit(X_train, y_train)

    preds = dt.predict(X_valid)

    acc = accuracy_score(y_true=y_valid, y_pred=preds)

    f1 = f1_score(y_true=y_valid, y_pred=preds)

    print(i)

    print(classification_report(y_true=y_valid, y_pred=preds))

    print('-----------------------')

    results[i] = f1
results
max(results, key=results.get)
results[max(results, key=results.get)]
best_max_leaf_node = max(results, key=results.get)

best_max_leaf_node
final_model = DecisionTreeClassifier(max_leaf_nodes=best_max_leaf_node)

final_model.fit(imputed_X_lable, y)
imputed_X_test_lable
imputed_X_test_lable.isna().sum()
my_preds_test=final_model.predict(imputed_X_test_lable)

my_preds_test
my_preds_test.shape
imputed_X_test_lable.shape
test_out = pd.DataFrame({

    'PassengerId': X_test.index, 

    'Survived': my_preds_test

})
test_out.to_csv('submission.csv', index=False)
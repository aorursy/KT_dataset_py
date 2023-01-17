import numpy as np
import pandas as pd
#Chosen path of testing and training data. 
dir_test = '../input/titanic/test.csv'
dir_train = '../input/titanic/train.csv'
#Read data
df_test_full = pd.read_csv(dir_test)
df_train_full = pd.read_csv(dir_train)
df_train_full.head()
df_test_full.head()
print("Rows X Columns in datasets:")
print(df_train_full.shape)
print(df_test_full.shape)
#Percentage of missing values by column
miss_by_col_train = df_train_full.isnull().sum() * 100/ len(df_train_full)
miss_by_col_test = df_test_full.isnull().sum() * 100/ len(df_test_full)
print("Percentage of missing values by column: \n")
print("Training data: \n\n", miss_by_col_train, "\n")
print("Testing data: \n\n", miss_by_col_test)
# All categorical columns
object_cols = [col for col in df_train_full.columns if df_train_full[col].dtype == "object"]

#Basic search of number of unique values by categorical column.
object_nunique = list(map(lambda col: df_train_full[col].nunique(), object_cols)) 

d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])
#Sex and Embarked have low cardinality. Cabin, Name and Ticket have high cardinality so we'll drop them altogether
#PassengerId is dropped because of having no value to our basic prediction. It is not derived from cardinality or amount of missing values
#Drop columns chosen. This columns require some feature engineering for them to be valuable. For the sake of simplicity, we will remove them
to_drop_train = df_train_full[['Cabin', 'Ticket', 'Name', 'PassengerId']]
to_drop_test  = df_train_full[['Cabin', 'Ticket', 'Name']]
#Leave 'PassegnerId' in testing data so the number of columns match between test and training data.
#We will later drop it, but we have to extract the values first in order to arrange submition file

#We will encode 'Sex' and 'Embarked'. We will impute 'Age'. 'Fare' will only be imputed in testing data(where it's actually missing)
# Drop corresponding columns
df_train = (df_train_full.drop(to_drop_train, axis = 1)).copy()
df_test = (df_test_full.drop(to_drop_test, axis = 1)).copy()

print(df_train.shape, '\n')
print(df_test.shape)

#We manually impute 'Age' with the mean of the training set values.
#Note that we use in both test and train data the training set mean value. 
age_mean = round(df_train['Age'].mean())
df_train['Age'] = df_train['Age'].replace(np.nan, age_mean)
df_test['Age'] = df_test['Age'].replace(np.nan,age_mean)
#Manually impute 'Fare' in test data
fare_mean =  round(df_train['Fare'].mean())
df_test['Fare'] = df_test['Fare'].replace(np.nan,fare_mean)

#Separate training set and target
y = df_train.Survived
X = df_train.drop('Survived', axis=1)
#Check data before splitting
print(X.head(), '\n')
print(y.head()) 

#Split train and validation data. 
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                            train_size=0.8, test_size=0.2,
                                                            random_state=0)


# Apply one-hot encoder to each column with categorical data
X_train_dummied = pd.get_dummies(X_train)
X_valid_dummied = pd.get_dummies(X_valid)
df_test_dummied = pd.get_dummies(df_test)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# Define the model
dtr_model = DecisionTreeRegressor(random_state = 1)
rfr_model = RandomForestRegressor(n_estimators=100, random_state = 1)
logreg = LogisticRegression()
# fit and predict DecisionTreeRegressor
dtr_model.fit(X_train_dummied, y_train)
dtr_pred = dtr_model.predict(X_valid_dummied)
# fit and predict RandomForestRegressor
rfr_model.fit(X_train_dummied, y_train)
rfr_pred = rfr_model.predict(X_valid_dummied)
# fit and predict LogisticRegression
logreg.fit(X_train_dummied, y_train)
logreg_pred = logreg.predict(X_valid_dummied)

# Calculate the score of your model on the validation data
dtr_score = dtr_model.score(X_train_dummied, y_train)
rfr_score = rfr_model.score(X_train_dummied, y_train)
logreg_score = logreg.score(X_train_dummied, y_train)

print('Score of DecisionTreeRegressor model: ', dtr_score)
print('Score of RandomForestRegressor model: ', rfr_score)
print('Score of LogisticRegression model: ', logreg_score)
#Finally extract the 'PassengerId' column to submit a result. Here i will use RFR
ids = df_test['PassengerId']
pred_test_data = rfr_model.predict(df_test_dummied.drop('PassengerId', axis=1)).astype(int)
out_rfr = pd.DataFrame({'PassengerId': ids, 'Survived': pred_test_data})
print (out_rfr.head()) 


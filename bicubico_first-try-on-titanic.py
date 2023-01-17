import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
# Datasources
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

print("### TRAIN DATA COLUMNS ###")
print(train_data.columns)

print("### TEST DATA COLUMNS ###")
print(test_data.columns)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


train_data.head(20)
test_id = test_data.PassengerId
#Prediction Target
#Single column on train data that contains the prediction
target = train_data.Survived
#Identify columns with missing values to erase them from the train set
#finding columns with missing values
cols_with_missing_values = [col for col in train_data.columns
                                   if train_data[col].isnull().any()]
cols_with_missing_values=['Age','Cabin']
print(cols_with_missing_values)
# Excluding non valuable columns, like PassengerId, Survived, and columns with missing data
# Predictor Columns
candidate_train_predictors = train_data.drop(['PassengerId','Survived']+cols_with_missing_values, axis=1)
candidate_test_predictors = test_data.drop(['PassengerId']+cols_with_missing_values, axis=1)
# Categorical values. 
# Choosing only those columns for on hot encodding where the categorical value for any attribute is not more than 10
low_cardinality_cols = [cname for cname in candidate_train_predictors.columns
                                       if candidate_train_predictors[cname].nunique()< 10 and
                                       candidate_train_predictors[cname].dtype=="object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns
                               if candidate_train_predictors[cname].dtype in ['int64', 'float64']]

useful_cols = low_cardinality_cols + numeric_cols
train_predictors = candidate_train_predictors[useful_cols]
test_predictors = candidate_test_predictors[useful_cols]
print(train_predictors.columns)
tempdf = candidate_train_predictors
df = pd.DataFrame(tempdf)#, columns = useful_cols)
df['y'] = target
sns.pairplot(df,hue='y')
print(train_predictors.columns)
def feature_engineering(df):
    #df['FamilySize'] = df['SibSp']+df['Parch']
    df['CuicoHijoUnico'] = (4-df['Pclass'])/(df['Parch']+1)
    #df['FarePerPerson'] = df['Fare']/(df['FamilySize']+1)
    #df['AgeClass'] = df['Age']*df['Pclass']
    #df=df.drop(['Parch','SibSp'],axis=1)
    return df

train_predictors = feature_engineering(train_predictors)
test_predictors = feature_engineering(test_predictors)
train_predictors.head(20)
# Adding dummy columns to categorical data
# HotEncoding
one_hot_encoded_train_data = pd.get_dummies(train_predictors)
one_hot_encoded_test_data = pd.get_dummies(test_predictors)

one_hot_encoded_train_data.describe()
one_hot_encoded_test_data.describe()
#tempdf = one_hot_encoded_train_data
#df = pd.DataFrame(tempdf)#, columns = useful_cols)
#df['y'] = target
#sns.pairplot(df,hue='y')
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
one_hot_encoded_train_data = my_imputer.fit_transform(one_hot_encoded_train_data)
one_hot_encoded_test_data = my_imputer.fit_transform(one_hot_encoded_test_data)
pd.DataFrame(one_hot_encoded_train_data).head()
pd.DataFrame(one_hot_encoded_test_data).head()
#Model Selection
#from sklearn.ensemble import RandomForestRegressor
#model = RandomForestRegressor()


from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=1000,learning_rate=0.05)
#Model Fit to Data
model.fit(one_hot_encoded_train_data, target,verbose=False)

#Get Predictions
test_predictions = np.around(model.predict(one_hot_encoded_test_data),0)
test_predictions = test_predictions.astype(np.int64)
#Submit predictions
my_submission = pd.DataFrame({'PassengerId': test_id, 'Survived': test_predictions})
my_submission.describe()

my_submission.head(10)
my_submission.to_csv('submission.csv', index=False)


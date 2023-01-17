import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
df.columns
df.columns[df.isna().any()].tolist()
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Survived'].describe()
df.head(5)
def get_model(process_df, used_columns):
    train_df, validation_df = train_test_split(process_df, test_size=0.2)
    x_train = train_df[used_columns]
    y_train = train_df['Survived']
    x_validation = validation_df[used_columns]
    y_validation = validation_df['Survived']
    
    model = XGBClassifier()
    model.fit(x_train, y_train)
    print(accuracy_score(model.predict(x_validation),y_validation))
    
    return model
used_columns = ['Pclass', 'Age', 'SibSp','Parch', 'Fare']
model = get_model(df, used_columns)
used_columns = ['Pclass', 'Age', 'SibSp','Parch']
model = get_model(df, used_columns)
def make_one_hot(df, columns):
    for column in columns:
        temp = pd.get_dummies(df[column], prefix=column)
        df = df.join(temp)
        
    return df
one_hoted_df = make_one_hot(df, ['Pclass'])
print(one_hoted_df.columns)
print(one_hoted_df[['Pclass', 'Pclass_1', 'Pclass_2','Pclass_3']].head(10)) 
one_hoted_df = make_one_hot(df, ['Pclass', 'Sex', 'Embarked'])
used_columns = ['Age', 'SibSp','Parch', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 
                'Embarked_C', 'Embarked_Q', 'Embarked_S']
model = get_model(one_hoted_df, used_columns)
df['Age'].plot.hist(bins=20)
age_group = [14, 20, 30, 35, 45, 55, 80]
previous_age_group = 0
for idx, age_group in enumerate(age_group):
    df['Age_' + str(idx)] = np.where((df['Age'] > previous_age_group) & (df['Age'] <= age_group), 1, 0)
    previous_age_group = age_group
df[['Age', 'Age_0', 'Age_1', 'Age_2', 'Age_3', 'Age_4', 'Age_5', 'Age_6']].head(10)
one_hoted_df = make_one_hot(df, ['Pclass', 'Sex', 'Embarked'])
used_columns = ['Age_0', 'Age_1', 'Age_2', 'Age_3', 'Age_4', 'Age_5', 'Age_6', 'SibSp','Parch', 
                'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'Embarked_C', 
                'Embarked_Q', 'Embarked_S']
model = get_model(one_hoted_df, used_columns)
for column in ['PassengerId', 'Pclass', 'Age', 'SibSp','Parch', 'Fare']:
    print(column, np.corrcoef(df[column], df['Survived'])[0, 1])
used_columns = ['Pclass', 'Fare']
model = get_model(df, used_columns)
one_hoted_df = make_one_hot(df, ['Pclass', 'Sex', 'Embarked'])
multipled_columns = ['Age_', 'Pclass_', 'Sex_', 'Embarked_']
for column in multipled_columns:
    columns = [col for col in list(one_hoted_df.columns) if col.startswith(column)]
    survived_columns = ['Survived' for _ in range(len(columns))]
    print(column, np.corrcoef(one_hoted_df[columns], one_hoted_df[survived_columns])[0, 1])
def create_age_bucket(df, new_age_bucket):
    for col in df.columns:
        if col.startswith('Age_'):
            del df[col]
    previous_age_group = 0
    for idx, age_group in enumerate(new_age_bucket):
        df['Age_' + str(idx)] = np.where((df['Age'] > previous_age_group) & (df['Age'] <= age_group), 1, 0)
        previous_age_group = age_group
    return df
def count_age_correlation(df, new_age_bucket):
    df = create_age_bucket(df, new_age_bucket)
    columns = [col for col in list(df.columns) if col.startswith('Age_')]
    survived_columns = ['Survived' for _ in range(len(columns))]
    print('Age', np.corrcoef(df[columns], df[survived_columns])[0, 1])
age_group = [15, 35, 50, 80]
count_age_correlation(df, age_group)
one_hoted_df = make_one_hot(df, ['Pclass', 'Sex', 'Embarked'])
used_columns = ['Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Embarked_C', 
                'Embarked_Q', 'Embarked_S'] \
                + [col for col in list(df.columns) if col.startswith('Age_')]
model = get_model(one_hoted_df, used_columns)
def run_predict(test_df, model, used_columns):
    x_test = test_df[used_columns]
    predict = model.predict(x_test)
    predict_df = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predict
    })
    
    return predict_df
new_test_df = create_age_bucket(test_df, age_group)
one_hoted_df = make_one_hot(new_test_df, ['Pclass', 'Sex', 'Embarked'])
predict_df = run_predict(one_hoted_df, model, used_columns)
predict_df.to_csv('prediction.csv', index=False)

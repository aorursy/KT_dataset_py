import os



import pandas as pd

import numpy as np



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer
df = pd.read_csv('../input/titanic/train.csv')

df.head()
def preprocessing(proc_df):

    

    feature_cols = proc_df.columns.to_list()

    

    try:

        feature_cols.remove('Survived')

    except ValueError:

        pass

    

    feature_cols.remove('Name')

    feature_cols.remove('Ticket')

    feature_cols.remove('PassengerId')

    

    proc_df = proc_df[feature_cols]

    

    # Convert Cabin to binary

    proc_df['Cabin'] = proc_df['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)

    

    # Chane categorical values to numerical values

    le = LabelEncoder()

    proc_df[['Sex', 'Embarked']] = proc_df[['Sex', 'Embarked']].apply(lambda col: le.fit_transform(col.to_list()))

    

    # Handle missing values

    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    

    imputer.fit(proc_df['Age'].values.reshape(-1, 1))

    proc_df['Age'] = imputer.transform(proc_df['Age'].values.reshape(-1, 1))

    

    imputer.fit(proc_df['Fare'].values.reshape(-1, 1))

    proc_df['Fare'] = imputer.transform(proc_df['Fare'].values.reshape(-1, 1))

    

    return proc_df
y = df['Survived']

x = preprocessing(df)

x.head()
# Run the Random Forest Classifier

rf = RandomForestClassifier(n_estimators=100)

rf.fit(x, y)



print(rf.score(x, y))
# Now lets predict for test csv given to us

x_test = pd.read_csv('../input/titanic/test.csv')



passenger_id = x_test['PassengerId']

x_test = preprocessing(x_test)



x_test.head()
rf.predict(x_test)
final_df = pd.concat([passenger_id, pd.DataFrame(rf.predict(x_test))], axis=1)

final_df.columns = ['PassengerId', 'Survived']

final_df.to_csv('submission.csv', index=False)



final_df.head()
final_df.Survived.value_counts()
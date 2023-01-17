import numpy as np
import pandas as pd
%matplotlib inline
np.set_printoptions(suppress=True)
x = pd.read_csv('../input/train.csv')
x_test_raw = pd.read_csv('../input/test.csv')
x_train = x.copy()
x_test = x_test_raw.copy()
def name_to_num(data):
    status = (('Dona.', 0), ('Don.', 1), ('Jonkheer.', 1), 
              ('Countess.', 0), ('Sir.', 1), ('Lady.', 0),
              ('Rev.', 1), ('Dr.', 1), ('Major', 1), ('Col.', 1), 
              ('Master.', 1), ('Capt.', 1), ('Mr.', 2), ('Mrs.', 3), 
              ('Miss.', 4), ('Ms.', 5), ('Mlle.', 4), ('Mme.', 5))

    for ix, name in enumerate(data['Name']):
        for value, number in status:
            if value in name:
                data.loc[ix, 'Title'] = number
    
    return data
x_train = name_to_num(x_train)
x_test = name_to_num(x_test)
x_train['Embarked'].fillna(method='ffill', inplace=True)
x_test['Embarked'].fillna(method='ffill', inplace=True)
x_train.drop(['Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)
x_test.drop(['Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)
from sklearn.preprocessing import OrdinalEncoder

def ordinal_encoder_to_df(model, columns):
    cat = model[list(columns)]
    ordinal_encoder = OrdinalEncoder()
    cat_encoded = ordinal_encoder.fit_transform(cat)
    model_num = pd.DataFrame(cat_encoded, columns=columns)

    return model_num

x_train_num = ordinal_encoder_to_df(x_train, ['Sex', 'Embarked', 'Ticket'])
x_train = x_train_num.combine_first(x_train)

x_test_num = ordinal_encoder_to_df(x_test, ['Sex', 'Embarked', 'Ticket'])
x_test = x_test_num.combine_first(x_test)
from sklearn.impute import SimpleImputer

def imputer(df):
    simple_imputer = SimpleImputer(strategy='mean')
    x = simple_imputer.fit_transform(df)
    df = pd.DataFrame(x, columns=df.columns, index=list(df.index.values))
    
    return df

x_train = imputer(x_train)
x_test = imputer(x_test)
corr_matrix = x_train.corr()
corr_matrix['Survived'].sort_values(ascending=False)
def labels_drop(df, labels):
    df_labels = df[labels].copy()
    df.drop(labels, axis=1, inplace=True)
    
    return df, df_labels
x_train, y_train = labels_drop(x_train, 'Survived')
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.astype(np.float64))
x_test = scaler.fit_transform(x_test.astype(np.float64))
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.model_selection import ValidationCurve
viz = ValidationCurve(RandomForestClassifier(), 
                      param_name='n_estimators', 
                      param_range=np.arange(5, 180, 5), 
                      cv=10)
viz.fit(x_train, y_train)
viz = ValidationCurve(RandomForestClassifier(n_estimators=100), 
                      param_name='max_leaf_nodes', 
                      param_range=np.arange(5, 180, 5), 
                      cv=10)
viz.fit(x_train, y_train)
viz = ValidationCurve(RandomForestClassifier(n_estimators=100), 
                      param_name='min_samples_leaf', 
                      param_range=np.arange(1, 11, 1), 
                      cv=10)
viz.fit(x_train, y_train)
from sklearn.model_selection import GridSearchCV
params = {
    'n_estimators': np.arange(25, 55, 5),
    'max_leaf_nodes': np.arange(5, 110, 5),
    'min_samples_leaf': np.arange(1, 6, 1),
}

grid_search = GridSearchCV(RandomForestClassifier(), 
                           params, cv=10)
grid_search.fit(x_train, y_train)
grid_search.best_params_
grid_search.best_score_
from yellowbrick.model_selection import LearningCurve
viz = LearningCurve(grid_search.best_estimator_, cv=5)
viz.fit(x_train, y_train)
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(x_test)

final = pd.DataFrame(
    final_predictions.astype(np.int64), columns=['Survived'], 
    index=x_test_raw['PassengerId'].astype(np.int64)
)
final.to_csv('gender_submisson15.csv')
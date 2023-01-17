import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
%matplotlib inline
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
tr = pd.concat([df_train.drop(['Survived'],axis=1), df_test])
tr.index = tr['PassengerId']
tr.head()
tr.columns
# Identify empty columns.
tr[tr.columns[tr.isnull().any()]].isnull().sum()
# Make a title.
def create_name_title(tr):
    tr['NameTitle'] = tr['Name'].str.extract(r'([A-Za-z]+)\.')
    tr['NameTitle'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
                            'Rare',inplace=True)
    tr['NameTitle'].replace(['Mlle', 'Ms', 'Mme'],
                            ['Miss', 'Miss', 'Mrs'],inplace=True)
create_name_title(df_train)
create_name_title(tr)
df_train.groupby('NameTitle')['Survived'].agg(["mean", "size"])
# Fill the empty age.
tr_age_mean = tr.groupby('NameTitle')['Age'].mean()
tr_age_null = tr['Age'].isnull() 

tr_age_mean_1 = tr_age_mean[tr.loc[tr_age_null, 'NameTitle']]
tr_age_mean_1.index = tr[tr_age_null].index
tr_age_mean_1
tr.loc[tr_age_null, 'Age'] = tr_age_mean_1
tr[tr['Age'].isnull()]
# Make age group features.
band = [0,9,18,27,36,45,53,62,71,100]
df_train['AgeBand'] = pd.cut(df_train['Age'],band)
tr['AgeBand'] = pd.cut(tr['Age'], band)
df_train.groupby('AgeBand')['Survived'].agg(["mean", "size"])
# Fill the empty Fare.
tr.loc[tr['Fare'].isnull(), 'Fare'] = tr.query('Age>60 & Pclass==3')['Fare'].mean()
# Fill the empty Embarked.
display(tr.loc[tr['Embarked'].isnull()])     
display(tr.groupby(['Embarked']).apply(lambda x: pd.Series(dict(
    p_1 = (x.Pclass == 1).sum(),
    p_2 = (x.Pclass == 2).sum(),
    p_3 = (x.Pclass == 3).sum(),
    cabin = (x.Cabin == 'B28' ).sum(),
    fare_down = (x.Fare < 80).sum(),
    fare_up = (x.Fare > 80).sum(),
    SibSp = (x.SibSp == 0).sum(),
    Parch = (x.Parch == 0).sum()
))))
tr.loc[tr['Embarked'].isnull(), 'Embarked'] = 'S'    
# PCA Method
from sklearn.decomposition import PCA

def dummy_to_pca(tr, column_name:str, features) :
    max_seq = 300
    max_d = 15
    col_count = tr.groupby(column_name)[column_name].count()
    if len(col_count) > max_seq:
        tops = col_count.sort_values(ascending=False)[0:max_seq].index
        f =tr.loc[tr[column_name].isin(tops)][['PassengerId', column_name]]
    else:
        tops = col_count.index
        f =tr[['PassengerId', column_name]]
    f = pd.get_dummies(f, columns=[column_name])  # This method performs One-hot-encoding
    f = f.groupby('PassengerId').mean()
    if len(tops) < max_d:
        max_d = len(tops)
    pca = PCA(n_components=max_d)
    pca.fit(f)
    cumsum = np.cumsum(pca.explained_variance_ratio_) #분산의 설명량을 누적합
    #print(cumsum)
    num_d = np.argmax(cumsum >= 0.99) + 1 # 분산의 설명량이 99%이상 되는 차원의 수
    if num_d == 1:
        num_d = max_d
    pca = PCA(n_components=num_d)    
    result = pca.fit_transform(f)
    result = pd.DataFrame(result)
    result.columns = [column_name + '_' + str(column) for column in result.columns]
    result.index = f.index
    return pd.concat([features, result], axis=1, join_axes=[features.index])
# Mean Encoding
def mean_encoding(tr, feature_name):
    mean = df_train.groupby(feature_name)['Survived'].mean()
    tr.loc[:,feature_name] = tr[feature_name].map(mean)
    tr.loc[tr[feature_name].isnull(), feature_name] = df_train['Survived'].mean()
    #print(tr[feature_name+'Mean'])
# Creates a ticket label variable.
def create_ticket_label(tr):
    tr['TicketLabel'] = tr['Ticket'].str.extract(r'([A-Za-z0-9/.]+) ')
    tr['TicketLabel'] = tr['TicketLabel'].str.replace("\.", "")
    tr['TicketLabel'] = tr['TicketLabel'].str.replace("/", "")
    tr['TicketLabel'] = tr['TicketLabel'].str.upper()
    tr['TicketLabel'].replace(['CASOTON','SCOW', 'AQ3', 'AQ4', 'SOP', 'STONOQ', 'STONO2', 'SCA3', 'A'],
                               ['CA', 'SC', 'AQ', 'AQ', 'SOPP', 'SOTONOQ', 'SOTONO2', 'SC', 'A4'],inplace=True)
    tr['TicketLabel'].fillna('NaN', inplace=True)

create_ticket_label(df_train)
create_ticket_label(tr)
df_train.groupby('TicketLabel')['Survived'].agg(["mean", "size"])
# Creates a ticket label variable.
def create_ticket_a(tr):
    tr['TicketA'] = tr['TicketLabel'].str[:2]
    tr['TicketA'].fillna('NaN', inplace=True)
create_ticket_a(tr)
create_ticket_a(df_train)
df_train.groupby('TicketA')['Survived'].agg(["mean", "size"])
# Refine the Cabin variable.
def create_cabin_a(tr):
    tr['CabinA'] = tr['Cabin'].str[:1]
    tr['CabinB'] = tr['Cabin'].str[:2]
    tr['CabinA'].fillna('NaN', inplace=True)
    tr['CabinB'].fillna('NaN', inplace=True)
create_cabin_a(tr)
create_cabin_a(df_train)
df_train.groupby(['CabinA','CabinB'])['Survived'].agg(["mean", "size"])
# Create a family number variable.
def create_family_size(tr):
    tr['FamilySize'] = tr['SibSp'] + tr['Parch'] + 1
    tr['IsAlone'] = tr['FamilySize'] == 1
create_family_size(tr)
create_family_size(df_train)
df_train.groupby('FamilySize')['Survived'].agg(["mean", "size"])
#df_train.groupby('IsAlone')['Survived'].agg(["mean", "size"])
# Create a feature for learning. : mean_encoding
import sklearn.preprocessing as pp

def get_features_mean():
    f = tr[['PassengerId','Pclass','Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
            'FamilySize', 'IsAlone', 'NameTitle','CabinA', 'CabinB', 'AgeBand', 
            'Sex', 'TicketA', 'TicketLabel']]
    f.index = f['PassengerId']
    mean_encoding(f, 'Sex')
    mean_encoding(f, 'IsAlone')
    mean_encoding(f, 'AgeBand')
    mean_encoding(f, 'Embarked')
    mean_encoding(f, 'NameTitle')
    mean_encoding(f, 'CabinA')
    mean_encoding(f, 'CabinB')
    mean_encoding(f, 'TicketA')
    mean_encoding(f, 'TicketLabel')

    scaler = pp.StandardScaler()
    f = pd.DataFrame(scaler.fit_transform(f), columns=f.columns)
    f['PassengerId'] = tr.index
    #display(f.head())
    return f    
# Create a feature for learning. : PCA
import sklearn.preprocessing as pp

def get_features_pca():
    f = tr[['PassengerId','Pclass','Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
            'FamilySize', 'IsAlone', 'NameTitle','CabinA', 'Sex', 'TicketA']]
    f.index = f['PassengerId']

    f = pd.get_dummies(f, columns=['Sex', 'IsAlone','Embarked', 'NameTitle', 'CabinA', 'TicketA'])
    f = dummy_to_pca(tr, 'TicketLabel', f)
    f = dummy_to_pca(tr, 'AgeBand', f) 
    f = dummy_to_pca(tr, 'CabinB', f) 
    #f.columns
    scaler = pp.StandardScaler()
    f = pd.DataFrame(scaler.fit_transform(f), columns=f.columns)
    f['PassengerId'] = tr.index
    #display(f.head())
    return f
# Save the result.
def split_train_data(f):
    X_train = df_train[['PassengerId']]
    X_train = pd.merge(X_train, f, how='left')
    #display(X_train.head())
    y_train = df_train.Survived

    X_test = df_test[['PassengerId']]
    X_test = pd.merge(X_test, f, how='left')
    #display(X_test.tail())

    X_train.drop(['PassengerId'], axis=1, inplace=True)
    X_test.drop(['PassengerId'], axis=1, inplace=True)
    return X_train, y_train, X_test
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def train(X_train, y_train):    
    clf_mlp = MLPClassifier()
    clf_xb = XGBClassifier()
    clf_rf = RandomForestClassifier()
    clfs = [
        ('xgb', clf_xb),#0.8272
        ('rf', clf_rf), #0.8284
        ('mlp', clf_mlp),
    ]
    clf_eb = VotingClassifier(estimators=clfs, voting='soft')
    parameters = {
        'xgb__max_depth':[4], 'xgb__min_child_weight':[4], 'xgb__gamma':[0.2],
        'xgb__subsample':[0.9], 'xgb__colsample_bytree':[0.84],
        'xgb__reg_alpha':[0.01], 'xgb__learning_rate':[0.2], 
        "rf__n_estimators":[45], "rf__max_depth":[20], "rf__min_samples_leaf":[3],
        'mlp__solver':['adam'], 'mlp__max_iter':[1000], 'mlp__early_stopping':[True], 
        'mlp__hidden_layer_sizes':[(128,64,32)],'mlp__activation':['logistic'],
    }
    clf = GridSearchCV(clf_eb, parameters, n_jobs=-1, cv=5)
    clf.fit(X_train, y_train)
    #print(clf.best_params_)
    score = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (score.mean(), score.std(), "eb"))
    return score.mean(), clf
def create_data_n_train(data_type):
    if (data_type == 'PCA'):
        features = get_features_pca()
    else:
        features = get_features_mean()
    X_train, y_train, X_test = split_train_data(features)
    score, clf = train(X_train, y_train) 
    pred = clf.fit(X_train, y_train).predict(X_test)
    return score, pred     
score_pca, pred_pca = create_data_n_train('PCA')
score_mean, pred_mean = create_data_n_train('MEAN')
print('PCA Score:', score_pca, ' Mean Encoding Score: ', score_mean)
submission_pca = pd.concat([df_test['PassengerId'], pd.Series(pred_pca, name="Survived")] ,axis=1)
submission_pca.to_csv('submission_pca.csv', index=False)

submission_mean = pd.concat([df_test['PassengerId'], pd.Series(pred_mean, name="Survived")] ,axis=1)
submission_mean.to_csv('submission_mean.csv', index=False)

display(submission_pca.head())
display(submission_mean.head())
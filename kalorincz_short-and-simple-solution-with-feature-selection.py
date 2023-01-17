import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



import warnings

warnings.filterwarnings("ignore")



from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectFromModel



from sklearn.preprocessing import LabelEncoder



import xgboost as xgb
import os

print(os.listdir("../input"))
df_train=pd.read_csv("../input/train.csv")

df_test=pd.read_csv('../input/test.csv')
print('Size of the training set',df_train.shape)

print('Size of the test set',df_test.shape)

missing =  pd.DataFrame(data = {'Missing in training set': df_train.isnull().sum(), 

                        'Missing in test set': df_test.isnull().sum()})

missing
s = df_train.shape[0]

y = df_train['Survived']

df_train = df_train.drop(columns = 'Survived')



X = pd.concat([df_train, df_test], ignore_index=True)
X['Title'] = X['Name'].apply(lambda x: x.partition(',')[-1].split()[0])



X['Title'][(X['Title'] != 'Mr.') & (X['Title'] != 'Mrs.')  &

           (X['Title'] != 'Master.') & (X['Title'] != 'Miss.') & (X['Sex'] == 'male')] = 'Mr.'

X['Title'][(X['Title'] != 'Mr.') & (X['Title'] != 'Mrs.')  & 

           (X['Title'] != 'Master.') & (X['Title'] != 'Miss.') & (X['Sex'] == 'female')] = 'Mrs.'
ind=X['Cabin'][X['Cabin'].notnull()].index.values

X['Cabin'][ind] = X['Cabin'][ind].apply(lambda x:  x[0])



groups = X.groupby('Ticket').count()

groups_with_cabin = groups[groups['Cabin'] != 0]

groups_with_cabin['Diff'] = groups_with_cabin['Fare'] - groups_with_cabin['Cabin']



ind=groups_with_cabin[(groups_with_cabin['Diff']>0)].index

cabins = X[(X['Ticket'].isin(ind))].sort_values('Ticket')

cabins = cabins.groupby('Ticket')['Cabin'].transform(lambda x: x.fillna(x.mode()[0]))

X.set_value(cabins.index,'Cabin', cabins.values)

X['Cabin'].fillna('N',inplace=True);
boy = np.round(X['Age'][(X['Name'] == 'Master.')].mean(),decimals = 1)

X['Age'][(X['Name'] == 'Master.')] = X['Age'][(X['Name'] == 'Master.')].fillna(boy)



girl = np.round(X['Age'][(X['Parch'] > 0) & (X['Name'] == 'Miss.')].mean(),decimals = 1)

X['Age'][(X['Parch'] > 0) & (X['Name'] == 'Miss.')] = X['Age'][(X['Parch'] > 0) & (X['Name'] == 'Miss.')].fillna(girl)

       

unmarried_female = np.round(X['Age'][(X['Parch'] == 0) & (X['Name'] == 'Miss.')].mean(),decimals = 1)

X['Age'][(X['Parch'] == 0) & (X['Name'] == 'Miss.')] = X['Age'][(X['Parch'] == 0) & (X['Name'] == 'Miss.')].fillna(unmarried_female)

        

#Fill the remaining missing values with the average age of male and female pessangers

mean_age = X[(X['Name'] != 'Miss.') & (X['Name'] != 'Master.')].groupby('Sex').mean()

X['Age'][(X['Sex'] == 'female')] = X['Age'][(X['Sex'] == 'female')].fillna(np.round(mean_age['Age']['female'], decimals = 1))

X['Age'][(X['Sex'] == 'male')] = X['Age'][(X['Sex'] == 'male')].fillna(np.round(mean_age['Age']['male'], decimals = 1))
X['Embarked'].fillna(X['Embarked'].value_counts().idxmax(),inplace=True)
X['Fare'] = X.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.mean()))
X.drop(columns=['PassengerId', 'Name','Cabin', 'Ticket'], inplace=True)
X.head()
def label_encoding(df):

    obj = [var for var in df.columns if df[var].dtype=='object']

    for c in obj:

        le=LabelEncoder()

        df[c]=le.fit_transform(df[c])

    return df
X = label_encoding(X)

X.head(3)
X_train = X.iloc[:s, :]

X_test = X.iloc[s:, :]
param = {'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 500, 'objective': 'binary:logistic',

        'reg_alpha': 0, 'reg_lambda': 0, 'seed': 1}



xgb_model = xgb.XGBClassifier(**param)



# fit model on all training data

xgb_model.fit(X_train, y)

sc = cross_val_score(xgb_model, X_train, y, cv = 10)

print("Accuracy: %.2f%%" % (sc.mean() * 100.0))



# Fit model using each importance as a threshold

thresholds = np.sort(xgb_model.feature_importances_)

for thresh in thresholds:

    # select features using threshold

    selection = SelectFromModel(xgb_model, threshold=thresh, prefit = True)

    select_X_train = selection.transform(X_train)

    # train model

    selection_model =xgb.XGBClassifier(**param)

    sc = cross_val_score(selection_model, select_X_train, y, cv = 10) 

    print("Thresh=%.3f, number of features=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], sc.mean()*100.0))
feat_imp = pd.DataFrame(data = {'Feature': X_train.columns, 'Importance': xgb_model.feature_importances_})

feat_imp.sort_values(by=['Importance'], ascending = False, inplace = True)

selected_features = feat_imp['Feature'][:5].values

print(selected_features)

feat_imp
X_train = X_train[selected_features]

X_test = X_test[selected_features] 
xgb_model.fit(X_train, y)

survival = xgb_model.predict(X_test)
submission = pd.DataFrame(data = {'PassengerId': df_test['PassengerId'], 'Survived': survival})

submission.head(5)
submission.to_csv('mysubmission.csv', index = False)
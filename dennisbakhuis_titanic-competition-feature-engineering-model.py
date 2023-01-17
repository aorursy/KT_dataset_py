import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

def proc_fare_testonly(inputdf):
    """
    Filling in the single missing value with the median of the Fare of the same Pclass
    """
    df = inputdf.copy()
    df['Fare'] = df['Fare'].fillna(8.05)
    return df

def proc_cabin(traindf, testdf):
    """
    Extract first letter from Cabin column as 'deck' indicator
    Move most luxerious T cabin as A, as it is a single one.
    """
    traindf = traindf.copy()
    testdf = testdf.copy()
    for df in [traindf, testdf]:
        df['deck'] = 'N'
        df.loc[~df.Cabin.isna(), 'deck'] = df.loc[~df.Cabin.isna(), 'Cabin'].apply(lambda x: list([y[0] for y in x.split()])[0])
        df.loc[df.deck=='T', 'deck'] = 'A'
    return traindf, testdf

def proc_embarked_trainonly(inputdf):
    """
    There are two missing Embarked values in the train set. Filled it in with S as tickets before and after 
    were also from Southampton (by ticket number)
    """
    df = inputdf.copy()
    df.loc[[829, 61], 'Embarked'] = 'S'
    return df

def proc_age_combined(traindf, testdf):
    """
    Fill in missing ages with median of the same Pclass+Sex groups
    """
    traindf = traindf.copy()
    testdf = testdf.copy()
    df1 = traindf[['Sex', 'Pclass', 'Age']].copy()
    df2 = testdf[['Sex', 'Pclass', 'Age']].copy()
    df1['source'] = 'train'
    df2['source'] = 'test'
    combined = pd.concat([df1, df2]).reset_index(drop=True)
    combined['Age'] = combined.groupby(['Sex', 'Pclass'])['Age'].apply(lambda group: group.fillna(group.median()))
    traindf['Age'] = combined.loc[combined.source=='train', 'Age'].values
    testdf['Age']  = combined.loc[combined.source=='test', 'Age'].values
    return traindf, testdf

def proc_tickets(inputdf):
    """
    Split ticket number and prefix
    """
    df = inputdf.copy()
    ticketSplit = df.Ticket.apply(lambda x: x.split()).values
    df['ticketnumber'] = pd.array([int(x[-1]) if x[-1].isdigit() else None for x in ticketSplit], dtype=pd.Int64Dtype())
    df['ticketprefix'] = [np.nan if len(x)==1 else ' '.join(x[:len(x)-1]) for x in ticketSplit]
    return df

def proc_names(inputdf):
    df = inputdf.copy()
    df['title'] = df.Name.str.split().apply(lambda x: [y for y in x if y[-1] =='.'][0])
    df['buyersurname'] = df.Name.apply(lambda x: x.split(',')[0])
    df['surname'] = df.buyersurname # by default it is the same person
    df['maidenname'] = None
    
    boughtForOthers = df.loc[(df.Name.str.contains('\(')) & ~(df.Name.str.contains('\"'))] # bought for others
    # first the none wifes:
    names = boughtForOthers.loc[boughtForOthers.title !='Mrs.'].Name.apply(lambda x: x.split('(')[1].split(')')[0].split()[-1])
    df.loc[names.index, 'surname'] = names
    # now the maiden names of the wifes (this is not perfect and there are some errors)
    names = boughtForOthers.loc[boughtForOthers.title =='Mrs.'].Name.apply(lambda x: x.split('(')[1].split(')')[0])
    names = names.loc[names.apply(lambda x: len(x.split())>1)]
    df.loc[names.index, 'maidenname'] = names
    return df


def importDataset():
    train = pd.read_csv('/kaggle/input/titanic/train.csv')
    test = pd.read_csv('/kaggle/input/titanic/test.csv')
    
    train = proc_names(train)
    test = proc_names(test)
    
    train = proc_tickets(train)
    test = proc_tickets(test)
    
    train, test = proc_age_combined(train, test)
    train, test = proc_cabin(train, test)
    
    train = proc_embarked_trainonly(train)
    test = proc_fare_testonly(test)
    return train, test

train, test = importDataset()
test['Survived'] = None
test['source'] = 'test'
train['source'] = 'train'
combined = pd.concat([train, test]).copy()

combined.title.unique()
combined['married'] = 'unknown'
combined.loc[(combined.title=='Mrs.') | (combined.title=='Countess.') | (combined.title=='Lady.'), 'married'] = 'yes'
combined.loc[(combined.title=='Miss.') | (combined.title=='Master.') | (combined.title=='Mlle.'), 'married'] = 'no'
_ = sns.countplot(x='married', hue='Survived', data=combined)
combined['title'] = combined['title'].replace(['Mlle.', 'Ms.'], 'Miss.')
combined['title'] = combined['title'].replace(['Mme.', 'Countess.', 'Lady.', 'Dona.'], 'Mrs.')
combined['title'] = combined['title'].replace(['Don.', 'Rev.', 'Dr.', 'Major.', 'Sir.', 'Col.', 'Capt.', 'Jonkheer.'], 'Other')

_ = sns.countplot(x='title', hue='Survived', data=combined)

combined['Age'] = pd.qcut(combined['Age'], 10)
_ = sns.countplot(x='Age', hue='Survived', data=combined)
combined['Fare'] = pd.qcut(combined['Fare'], 10)
_ = sns.countplot(x='Fare', hue='Survived', data=combined)

combined['familysize'] = combined['Parch'] + combined['SibSp']

sns.countplot(x='familysize', hue='Survived', data=combined)
combined['family'] = 'alone'
combined.loc[(combined.familysize > 0) & (combined.familysize < 4), 'family'] = 'small'
combined.loc[(combined.familysize > 4), 'family'] = 'large'

_ = sns.countplot(x='family', hue='Survived', data=combined)

combined['surnamefrequency'] = combined['surname'].map(combined['surname'].value_counts())
_ = sns.countplot(x='surnamefrequency', hue='Survived', data=combined)
combined['surnamefreq'] = 'alone'
combined.loc[(combined.surnamefrequency > 1) & (combined.surnamefrequency <= 4), 'surnamefreq'] = 'small'
combined.loc[(combined.surnamefrequency > 4), 'surnamefreq'] = 'large'

_ = sns.countplot(x='surnamefreq', hue='Survived', data=combined)

combined['ticketfrequency'] = combined['ticketnumber'].map(combined['ticketnumber'].value_counts())
_ = sns.countplot(x='ticketfrequency', hue='Survived', data=combined)
combined['ticketfreq'] = 'single'
combined.loc[(combined.ticketfrequency > 1) & (combined.ticketfrequency <= 4), 'ticketfreq'] = 'small'
# combined.loc[(combined.ticketfrequency > 1) & (combined.ticketfrequency <= 2), 'ticketfreq'] = 'small'
# combined.loc[(combined.ticketfrequency > 2) & (combined.ticketfrequency <= 4), 'ticketfreq'] = 'medium'
combined.loc[(combined.ticketfrequency > 4), 'ticketfreq'] = 'large'

_ = sns.countplot(x='ticketfreq', hue='Survived', data=combined)

_ = sns.countplot(x='deck', hue='Survived', data=combined)
combined.deck.value_counts()
combined.loc[combined.deck=='G', 'deck'] = 'F'
combined.to_csv('Titanic_raw_processed.csv', index=False)

# df = combined[['Survived', 'Pclass', 'Sex', 'Age', 'Fare','Embarked', 'title', 'deck',
#        'source', 'married', 'family', 
#        'surnamefreq', 'ticketfreq']].copy()
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# selectedfeatures = ['Sex', 'Age', 'Fare','Embarked', 'title', 'deck', 'married', 'family', 'surnamefreq', 'ticketfreq']

# for feature in selectedfeatures:        
#     df[feature] = LabelEncoder().fit_transform(df[feature])

# df
# selectedfeatures = ['Pclass', 'Sex', 'Embarked', 'title', 'deck', 'married', 'family', 'surnamefreq', 'ticketfreq']
# encoded = []

# for feature in selectedfeatures:
#     encode = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
#     n = df[feature].nunique()
#     cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
#     encodeddf = pd.DataFrame(encode, columns=cols)
#     encodeddf.index = df.index
#     encoded.append(encodeddf)

# final = pd.concat([df[['Survived', 'source']], *encoded], axis=1)

# final
# train = final.loc[final.source=='train'].drop('source', axis=1).copy()
# test = final.loc[final.source=='test'].drop(['source', 'Survived'], axis=1).copy()

final = combined[['Survived', 'Pclass', 'Sex', 'Age', 'Fare','Embarked', 'title', 'deck',
       'source', 'married', 'family', 
       'surnamefreq', 'ticketfreq']].copy()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
selectedfeatures = ['Age', 'Fare']

for feature in selectedfeatures:        
    final[feature] = LabelEncoder().fit_transform(final[feature])

train = final.loc[final.source=='train'].drop('source', axis=1).copy()
test = final.loc[final.source=='test'].drop(['source', 'Survived'], axis=1).copy()

from catboost import CatBoostClassifier, Pool
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import Pool, CatBoostClassifier, cv

target = train.Survived.values
train = train.drop('Survived', axis=1)
categoryFeatureIdx = np.where(train.dtypes != float)[0]
target = target.astype(np.uint8)
xtrain, xtest, ytrain, ytest = train_test_split(train, target, train_size=.85, random_state=42)
model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True, random_seed=42, loss_function='Logloss')
model.fit(xtrain, ytrain, cat_features=categoryFeatureIdx, eval_set=(xtest,ytest), )
cv_data = cv(Pool(train,target,cat_features=categoryFeatureIdx), model.get_params(), fold_count=10)
print('the best cv accuracy is :{}'.format(np.max(cv_data["test-Accuracy-mean"])))
print('the test accuracy is :{:.6f}'.format(accuracy_score(ytest,model.predict(xtest))))
cv_data
print('the test accuracy is :{:.6f}'.format(accuracy_score(ytest,model.predict(xtest))))
model2 = CatBoostClassifier(eval_metric='Accuracy',use_best_model=False, random_seed=42, loss_function='Logloss', iterations=2000)
model2.fit(train, target, cat_features=categoryFeatureIdx)

print('the test accuracy is :{:.6f}'.format(accuracy_score(ytest,model2.predict(xtest))))

pred = model2.predict(test)
pred = pred.astype(np.int)
submission = combined.loc[combined.source=='test', ['PassengerId',]].copy()
submission['Survived'] = pred
submission.to_csv('titanic_submission.csv', index=False)

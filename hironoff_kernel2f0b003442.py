import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import log_loss, accuracy_score

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

import time



import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



train_x = train.drop(['Survived'], axis = 1)

train_y = train['Survived']



test_x = test.copy()

train_x.head()
train_x.info()
def prepare_data(input_df_0):

    input_df = input_df_0.copy()

    input_df['Salutation'] = input_df.Name.str.extract('([A-Za-z]+).', expand = False) #この形の文字列を抜き出す

    input_df['Salutation'] = input_df['Salutation'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    input_df['Salutation'] = input_df['Salutation'].replace('Mile', 'Miss')

    input_df['Salutation'] = input_df['Salutation'].replace('Ms', 'Miss')

    input_df['Salutation'] = input_df['Salutation'].replace('Mme', 'Mrs')

    

    input_df['Age'] = input_df.groupby('Salutation')['Age'].transform(lambda x : x.fillna(x.mean()))

    

    input_df['Age'].fillna(input_df.Age.mean(), inplace = True)

    input_df['Ticket_Lett'] = input_df['Ticket'].apply(lambda x : str(x)[0])

    input_df['Ticket_Lett'] = np.where((input_df['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']),

                                       input_df['Ticket_Lett'], 

                                           np.where((input_df['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), 

                                                '0',

                                                '0')) 

    input_df['Ticket_Num'] = train_x['Ticket'].apply(lambda x: len(x.split()[-1]))

  

    #input_df['Cabin_Lett'] = input_df['Cabin'].apply(lambda x: str(x)[0]) #最初の一文字だけとってくる

    #input_df['Cabin_Lett'] = np.where((input_df['Cabin_Lett']).isin([ 'F', 'E', 'D', 'C', 'B', 'A']),input_df['Cabin_Lett'], np.nan)

    

    input_df.drop(['PassengerId', 'Cabin', 'Ticket', 'Name', 'Salutation'], axis = 1,inplace = True)

    

    for label in ['Embarked', 'Sex', 'Ticket_Lett']:

        le = LabelEncoder() 

        le.fit(input_df[label].fillna('NA'))

        input_df[label] = le.transform(input_df[label].fillna('NA'))

    

    input_df['Fare'].fillna(input_df.Fare.mean(), inplace = True) ##一考の余地あり

    

    return input_df

    
#SibSpとParchを加算したdfを提供

def createFamilySize(input_df_0):

    input_df = input_df_0.copy()

    input_df['FamilySize'] = input_df['SibSp'] + input_df['Parch']

    input_df.drop(['SibSp', 'Parch'], axis = 1, inplace = True)

    return input_df
train_x_cleaned = prepare_data(train_x)

print(train_x_cleaned.info())

train_x_cleaned.head()
train_x_cleaned2 = createFamilySize(train_x_cleaned)

print(train_x_cleaned2.info())

train_x_cleaned2.head()
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train_x_cleaned2.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
#チューニング



stime = time.time()

parameters = { #過学習のないモデルにするには

    "n_estimators":[230,240,250], #よくわかんない

    "criterion":["gini","entropy"], #どうでもいい

    "max_depth":[6,7,8], #小さくなってほしい

     'min_samples_split': [7,8,9], #大きくなってほしい

}



clf1 = GridSearchCV(RandomForestClassifier(), parameters, cv = 5)

clf1.fit(train_x_cleaned2, train_y)

 

print(clf1.best_estimator_)



#これチューニングしたやつ

va_pred = clf1.best_estimator_.predict_proba(train_x_cleaned2)[:, 1]



logloss = log_loss(train_y, va_pred)

accuracy = accuracy_score(train_y, va_pred > 0.5)



print(logloss)

print(accuracy)



etime = time.time()



print('経過時間：' + str(etime - stime))
test_x.info()
test_x_result = prepare_data(test_x)

test_x_result = createFamilySize(test_x_result)

test_x_result.info()
pred_result = clf1.best_estimator_.predict(test_x_result)



df_result = pd.DataFrame({'PassengerId': test_x['PassengerId'], 'Survived' : pred_result})



df_result.set_index('PassengerId', inplace = True)



df_result.to_csv("submission_clf1.csv")
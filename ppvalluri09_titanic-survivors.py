import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler



df = pd.read_csv('../input/titanic/train.csv')

df.head()
def prepareData(df, type='train'):

    # Removing un-necessary data

    p_id = df['PassengerId']

    if type == 'train':

        df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Survived']]    # We removed Cabin because out of 891 entries 687 were null (> 15%)

    elif type=='test':

        df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']]

    df['Cabin'] = df['Cabin'].fillna(0)

    df['Fare'] = df['Fare'] / (df['SibSp'] + df['Parch'] + 1)

    df = df.drop(['SibSp', 'Parch', 'Cabin'], axis=1)

    # df.head()



    # Conversion of String data to categorical data



    # Gender

    df['Sex'] = pd.get_dummies(df['Sex'])['female']    # We are taking female as 1 and male as 0(since it would be easy while submission)

    

    # Class

    p_class = pd.get_dummies(df['Pclass'])

    df['Class1'], df['Class2'], df['Class3'] = p_class[1], p_class[2], p_class[3]

    df = df.drop(['Pclass'], axis=1)



    sns.boxplot(df['Age'])

    plt.show()

    sns.boxplot(df['Fare'])

    plt.show()



    # Checking for NaN

    if df['Age'].isna().sum() > 0 or df['Fare'].isna().sum > 0:

        df['Age'].fillna(df['Age'].mean(), inplace=True)

        df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Filling NaN with the mean of the Age (mean can be used since the pulling by the outliers isn't much)

    # But for Fare we replace with median since the outliers are pulling and making the distribution skew



    # Feature Scaling for faster loss convergence

    scaler = StandardScaler()

    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])



    return df, p_id
df, p_id = prepareData(df)

df.head()
from sklearn.metrics import f1_score, classification_report, roc_auc_score, confusion_matrix, accuracy_score



def model_training(clf, x_t, y_t, x_v=None, y_v=None, model='binary:logistic'):

    clf.fit(x_t, y_t)

    print('Training Accuracy: ', clf.score(x_t, y_t))



    if model=='binary:logistic':

      print('Validation Accuracy', clf.score(x_v,y_v))

      print('Validation f1_score',f1_score(clf.predict(x_v),y_v))

      print('Validation roc_auc score',roc_auc_score(y_v,clf.predict_proba(x_v)[::,-1]))

      print('Confusion Matrix \n',confusion_matrix(y_v, clf.predict(x_v)))

    

    if model=='reg:linear':

        if x_v!=None:

            print('Validation r2_score', clf.score(x_v,y_v))

            print('Validation MSE',mean_squared_error(clf.predict(x_v),y_v))



            

    return clf
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier



x_t, x_v, y_t, y_v = train_test_split(df.iloc[:, [0, 1, 2, 4, 5, 6]], df['Survived'], test_size=0.2, random_state=42)



lgr = LogisticRegression()

xgb = XGBClassifier(n_estimators=500, max_depth=5,learning_rate=0.1,scale_pos_weight=1.4266790777602751)

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 16), random_state=1)
model = model_training(xgb, x_t, y_t, x_v, y_v)    # Training the XGBoost model
# Testing the model with the test data

test = pd.read_csv('../input/titanic/test.csv')

test, ids = prepareData(test, type='test')

test.head()
y_pred = model.predict(test)



# Saving the predictions to csv

result = pd.DataFrame()

result['PassengerId'] = pd.Series(ids)

result['Survived'] = pd.Series(y_pred)

result.to_csv('submission.csv', index=False)
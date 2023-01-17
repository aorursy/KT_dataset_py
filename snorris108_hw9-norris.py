# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Libraries to utilize

%matplotlib inline

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.

test_file = pd.read_csv("../input/test.csv")

X = pd.read_csv("../input/train.csv")
X.head()
def clean_df(df, df_name):

    # drop columns not useful for prediction

    df.drop(labels=['Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)

    df.rename(columns = lambda x: x.strip(), inplace = True);

    

    print("DataFrame ", df_name + ':')

    print(df.isnull().sum())

    # for ['Age'], fill NaNs with mean age

    df['Age'].fillna(df['Age'].mean(), inplace=True)

    # for ['Fare'], fill NaNs with median fare

    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # for ['Embarked'], fill NaNs with mode data

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    pct = df['Cabin'].isnull().sum() / df.shape[0] * 100

    print(df_name + "['Cabin'] is {}% NaN, \nso we will drop it.\n".format(round(pct)))

    df.drop(labels=['Cabin'], axis=1, inplace=True)

    

    print(df.isnull().sum())

    print('-'*42)
clean_df(test_file, 'test_file')

clean_df(X, 'X')
def ohe(df, cols):

    df = pd.concat([df, pd.get_dummies(df[ohe_cols])], axis=1)

    df.drop(labels=ohe_cols, axis=1, inplace=True)

    df = df.astype('float64')

    return df
ohe_cols = ['Pclass', 'Sex', 'Embarked']

test_file = ohe(test_file, ohe_cols)

X = ohe(X, ohe_cols)
# get target series y

y = X['Survived']

X.drop('Survived', axis=1, inplace=True)
X.head()
test_file.head()
X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size=0.3,

                                                    random_state=1,

                                                    stratify=y)

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
orders = [10**i for i in range(-3, 1)]

for magnitude in orders:

    pipe = make_pipeline(StandardScaler(),

                         LogisticRegression(random_state=1, solver='lbfgs',

                                            C=magnitude))

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    print('\nInverse of Regularization Strength: ', magnitude)

    print('Prediction Accuracy: %.2f' %(pipe.score(X_test, y_test)*100), '%', sep='')

    confmat = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)

    

    for i in range(confmat.shape[0]):

        for j in range(confmat.shape[1]):

            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('Predicted label')

    plt.ylabel('True label')

    plt.tight_layout()

    #plt.savefig('images/06_09.png', dpi=300)

    plt.show()
survived_list = []

for index, row in test_file.iterrows():

    row_data = []

    for element in row:

        row_data.append(element)

    survived_list.append('Survived' if pipe.predict([row_data]) == 1 else 'Died')

survived_series = pd.Series(survived_list)
test_file['Predicted Survival'] = survived_series

test_file.head()
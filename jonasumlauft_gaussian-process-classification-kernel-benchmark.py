import pandas  as pd

import numpy as np

from sklearn.model_selection import cross_validate

from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, RationalQuadratic, DotProduct,Matern

from sklearn.gaussian_process import GaussianProcessClassifier

# load data

train_df = pd.read_csv('../input/titanic/train.csv')

test_df  = pd.read_csv('../input/titanic/test.csv')



for test_set,df in enumerate([train_df, test_df]): 

    # Extract title from name and summarize

    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',

        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df["Title"] = df["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, 

                                   "Mlle":1, "Mrs":1, "Mr":2, "Rare":3}).fillna(3).astype(int)



    # Convert sex and embarked port to numerical value

    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).fillna(2).astype(int)

    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).fillna(3).astype(int)



    # Replace number by their category

    df["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in df['Cabin'] ])

    df['Cabin'] = df['Cabin'].map( {'A': 0, 'B': 1, 'C': 2,'D': 3, 'E': 4,

                                    'F': 5,'G': 6, 'T': 7, 'X': 8} ).fillna(9).astype(int)

    # Remove Name and PassengerID and Ticket number

    df = df.drop(['Name', 'PassengerId', 'Ticket'], axis=1)

    

    # Fill in missing Fare values

    index_NaN = list(df["Fare"][df["Fare"].isnull()].index)

    for i in index_NaN:

        df.loc[i,'Fare'] = df.Fare[df['Pclass'] == df.loc[i,"Pclass"]].median()



    # Fill in missing Age values

    index_NaN = list(df["Age"][df["Age"].isnull()].index)

    for i in index_NaN:

        idx = ((df['SibSp'] == df.iloc[i]["SibSp"]) &

                (df['Parch'] == df.iloc[i]["Parch"]) & 

                (df['Pclass'] == df.iloc[i]["Pclass"]))

        # print(idx)

        age_pred = df.Age[idx].median()

        if np.isnan(age_pred) :

            df.loc[i,'Age'] = df["Age"].median()

        else:

            df.loc[i,'Age'] = age_pred

            

    # 

    if test_set == True:

        X_test = df

    else:

        X_train = df.drop(['Survived'], axis=1)

        y_train = df["Survived"]







print(X_train.head(5))

print(X_test.head(5))

print(y_train.head(5))



dx = X_train.shape[1]



ls = np.ones((dx,))

kernels = [ConstantKernel() * RBF(ls) + WhiteKernel(),

            ConstantKernel() * Matern(ls) + WhiteKernel(),

           # ConstantKernel() * DotProduct() + WhiteKernel(),

           # ConstantKernel() * RationalQuadratic()+ WhiteKernel() ,

          ]

           

best_score = 0

for kernel in kernels:

    classifier = GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=5)

    classifier.fit(X_train, y_train)

    result = cross_validate(classifier,X_train, y_train,cv=3)

    score = result['test_score'].mean()

    print('Average result: ' + str(score) + ', Training time: ' + str(result['fit_time'].mean()))

    if score > best_score: 

        best_score = score

        best_classifier = classifier

predictions = best_classifier.predict(X_test).astype(int)

output = pd.DataFrame({'PassengerId': test_df.PassengerId, 

                       'Survived': predictions})

print(best_classifier.kernel)

print(output)

output.to_csv('submission.csv', index=False)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



df_train = pd.read_csv("../input/titanic/train.csv")



df_predict = pd.read_csv("../input/titanic/test.csv")



labels = df_train['Survived'].to_numpy()



df_predict['Survived'] = np.nan



passengersId = df_predict['PassengerId'].to_numpy()



def cleanData(df):

             

    df.fillna(0, inplace=True)



    ############### Fare ################



    def classify_fare(fare):

        if fare <= 7.91:

            return 0

        elif fare > 7.91 and fare <= 14.454:

            return 1

        elif fare > 14.454 and fare <= 31:

            return 2

        elif fare > 31:

            return 3

        else:

            return 4



    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    df['FareCat'] = df['Fare'].apply(classify_fare)

    df['Fare'] = df['Fare'].astype(int)



    ############### Title ################



    label = LabelEncoder()



    df['Title'] = df['Name'].str.split(", ", expand=True)[

        1].str.split(".", expand=True)[0]



    stat_min = 10



    title_names = (df['Title'].value_counts() < stat_min)



    df['Title'] = df['Title'].apply(

        lambda x: 'Misc' if title_names.loc[x] == True else x)



    df['Title_Code'] = label.fit_transform(df['Title'])



    

    ############### Age ################



    df['Age'].fillna(df['Age'].median(), inplace=True)



    df['AgeCat'] = df['Age'].astype(int)



    df.loc[df['AgeCat'] <= 12, 'AgeCat'] = 0

    df.loc[(df['AgeCat'] > 12) & (df['AgeCat'] <= 18), 'AgeCat'] = 1

    df.loc[(df['AgeCat'] > 18) & (df['AgeCat'] <= 22), 'AgeCat'] = 2

    df.loc[(df['AgeCat'] > 22) & (df['AgeCat'] <= 27), 'AgeCat'] = 3

    df.loc[(df['AgeCat'] > 27) & (df['AgeCat'] <= 33), 'AgeCat'] = 4

    df.loc[(df['AgeCat'] > 33) & (df['AgeCat'] <= 40), 'AgeCat'] = 5

    df.loc[(df['AgeCat'] > 40) & (df['AgeCat'] <= 66), 'AgeCat'] = 6

    df.loc[df['AgeCat'] > 66, 'AgeCat'] = 7



    ############### Is Kid ################



    df['IsKid'] = 0

    df['IsKid'] = np.where((df['AgeCat'] == 0), 1, 0)



    ############## Male / Female Kid ########



    df['IsFemaleKid'] = 0

    df['IsFemaleKid'] = np.where(

        (df['Sex'] == 'female') & (df['AgeCat'] == 1), 1, 0)

    

    df['IsMaleKid'] = 0

    df['IsMaleKid'] = np.where(

        (df['Sex'] == 'male') & (df['AgeCat'] == 1), 1, 0)

    

    ########################################

    

    df['FirstClassFemale'] = 0

    df['FirstClassFemale'] = np.where(

        (df['Sex'] == 'female') & (df['Pclass'] == 1), 1, 0)



    ############### Embarked ################



    df.loc[df.Embarked == 'Q', 'Embarked'] = 1

    df.loc[df.Embarked == 'S', 'Embarked'] = 2

    df.loc[df.Embarked == 'C', 'Embarked'] = 3



    df.loc[df.Embarked == 0, 'Embarked'] = 2



    ############### Sex ################



    df.loc[df.Sex == 'male', 'Sex'] = 1

    df.loc[df.Sex == 'female', 'Sex'] = 0



    ############### Family ################



    df['Family'] = df['SibSp'] + df['Parch'] + 1

    

    df['IsAlone'] = 0

    df['IsAlone'].loc[df['SibSp'] + df['Parch'] == 0] = 1



    df['IsSmallFamily'] = 0

    df['IsSmallFamily'].loc[df['SibSp'] + df['Parch'] < 5] = 1



    df['IsLargeFamily'] = 0

    df['IsLargeFamily'].loc[df['SibSp'] + df['Parch'] >= 5] = 1

        

    ############### Cabin ###################

    

    df["Cabin"].isnull().apply(lambda x: not x)



    df['HadCabin'] = df['Cabin'].notna()

    df['HadCabin'] = df['HadCabin'].astype(int)  



    df["Deck"] = df["Cabin"].str.slice(0,1)

    df["Room"] = df["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")



    df["Deck"] = df["Deck"].fillna("N")

    df["Room"] = df["Room"].fillna(df["Room"].mean())

    

    

    ############### Columns Drop ################



    if 'Survived' in df.columns:

        df.drop(['Survived'], axis=1, inplace=True)    

        

    df.drop(['PassengerId'], axis=1, inplace=True)



    df.drop(['Name'], axis=1, inplace=True)

    df.drop(['Ticket'], axis=1, inplace=True)

    df.drop(['Cabin'], axis=1, inplace=True)

    df.drop(["Title"], axis=1, inplace=True)

    df.drop(['Fare'], axis=1, inplace=True)

    df.drop(['Room'], axis=1, inplace=True)

    df.drop(['Deck'], axis=1, inplace=True)

    df.drop(['HadCabin'], axis=1, inplace=True)

    df.drop(['Family'], axis=1, inplace=True)

    df.drop(['IsAlone'], axis=1, inplace=True)

    df.drop(['FareCat'], axis=1, inplace=True)

    return df



def dummy_data(data, columns):

    for column in columns:

        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)

        data = data.drop(column, axis=1)

    return data



def dropColumns(df, columns):

    for column in columns:

        df.drop([column], axis=1, inplace=True)



data = cleanData(df_train)



predict_data = cleanData(df_predict)



data = dummy_data(data, ["Pclass", "Embarked", "AgeCat",])



predict_data = dummy_data(predict_data, ["Pclass", "Embarked", "AgeCat"])



#### Drop Columns after getting data from feature_importances



dropColumns(data, ["Embarked_3"]) 



dropColumns(predict_data,  ["Embarked_3"])



#### Split Data for training and evaluation ####



train_data, test_data, train_labels, test_labels = train_test_split(

    data, labels, test_size=.2)



count = 1



last_final_score = 0



importances = ''



feature_importances = ''

while count < 150:

    

    print(count)



    rf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',

            max_depth=10, max_features='auto', max_leaf_nodes=10,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=4,

            min_weight_fraction_leaf=0.0, n_estimators=count, n_jobs=1,

            oob_score=False, random_state=True, verbose=0,

            warm_start=False)



    rf.fit(data, labels)



    score = rf.score(test_data, test_labels)



    feature_importances = pd.DataFrame(rf.feature_importances_, index = train_data.columns, columns=['importance']).sort_values('importance',ascending=False)



    prediction = rf.predict(predict_data)

    

    print("N_stimators: %s" % count)

    print("Final Score: %s" % score)

    

    if score > last_final_score:

        last_final_score = score

        print("N_stimators: %s" % count)

        print("Final Score: %s" % score)

        print(feature_importances)

        result = {'PassengerId': passengersId, 'Survived': prediction}

        new_df = pd.DataFrame(result)

        new_df.to_csv('submission.csv', index = None, header=True)     

    count = count + 1



print('Final Score: ',last_final_score)

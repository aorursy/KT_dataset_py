#Olivia Alexander

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
# women = train_data.loc[train_data.Sex == 'female']["Survived"]

# rate_women = sum(women)/len(women)



# # print("% of women who survived:", rate_women)



# men = train_data.loc[train_data.Sex == 'male']["Survived"]

# rate_men = sum(men)/len(men)



# # print("% of men who survived:", rate_men)



# class_three = train_data.loc[train_data.Pclass == 3]["Survived"]

# rate_classy_three = sum(class_three)/len(class_three)



# print("% of 3rd class folk who survived:", rate_classy_three)



from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier



y = train_data['Survived']



features = ["Pclass", "Sex", "Parch", "Fare", "Age", "Embarked"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



my_imputer = SimpleImputer()

imputed_X = pd.DataFrame(my_imputer.fit_transform(X))

imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))

imputed_X.columns = X.columns

imputed_X_test.columns = X_test.columns





model = RandomForestClassifier(n_estimators=80, max_depth=7, random_state=1)

model.fit(imputed_X, y)

predictions = model.predict(imputed_X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

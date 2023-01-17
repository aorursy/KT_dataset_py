# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df2 = pd.get_dummies(df, columns=['Pclass','Sex','Parch','Embarked'])

df3 = df2.dropna()
df_test = pd.get_dummies(df_test, columns=['Pclass','Sex','Parch','Embarked'])

df_test = df_test.fillna(0)
logi = LogisticRegression()
logi.fit(df3[['Age', 'SibSp', 'Fare',

       'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5',

       'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']], df3['Survived'])
result = logi.predict(df_test[['Age', 'SibSp', 'Fare',

       'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5',

       'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']])
solution = pd.DataFrame(result,  columns=['Survived'])

solution.index = df_test['PassengerId']

solution.to_csv('solution.csv')
solution.to_csv('solution.csv')
logi_cv = LogisticRegressionCV()
logi_cv_fit = logi_cv.fit(df3[['Age', 'SibSp', 'Fare',

       'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5',

       'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']], df3['Survived'])
logi_cv_fit.score(df3[['Age', 'SibSp', 'Fare',

       'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5',

       'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']], df3['Survived'])
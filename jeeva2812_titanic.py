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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



merge = pd.concat([train,test])



merge.info()
def handle_non_numerical_data(df):

    columns = df.columns.values

    for column in columns:

        text_digit_vals = {}

        def convert_to_int(val):

            return text_digit_vals[val]



        if df[column].dtype != np.int64 and df[column].dtype != np.float64:

            column_contents = df[column].values.tolist()

            unique_elements = set(column_contents)

            x = 0

            for unique in unique_elements:

                if unique not in text_digit_vals:

                    text_digit_vals[unique] = x

                    x+=1



            df[column] = list(map(convert_to_int, df[column]))



    return df



dfX = merge.drop(['Name','PassengerId','Survived'], 1)

y = np.array(train['Survived'])



dfX.fillna(-99999, inplace=True)

dfX = handle_non_numerical_data(dfX)







X = np.array(dfX.iloc[:train.shape[0]])

testData = np.array(dfX.iloc[-test.shape[0]:])



dfX.head()
#Just for calculating accuracy



from sklearn import preprocessing, model_selection, svm



X = preprocessing.scale(X)



X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = 0.2)



clf = svm.SVC()

clf.fit(X_train,y_train);



accuracy = clf.score(X_test,y_test)

print(accuracy)
clf1 = svm.SVC()

clf1.fit(X,y)



testData = preprocessing.scale(testData)



predictions = clf1.predict(testData)

submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})



predictions
submission.to_csv('submission1.csv',index=False)
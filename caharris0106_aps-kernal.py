# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
testing = "../input/aps-failure-at-scania-trucks-data-set/aps_failure_test_set.csv"

training = "../input/aps-failure-at-scania-trucks-data-set/aps_failure_training_set.csv"

prepared = "prepared_aps_data.csv"





def preprocess_data(file):

    # Read the Data from the training Csv

    df = pd.read_csv(file, header=0)



    # Change datatype from string to np.nan for missing values

    def change_na_to_numpy_nan(x):

        # New Series has NaN instead of text "na"

        x.replace("na", np.nan, inplace=True)

        return x



    df = df.apply(lambda x: change_na_to_numpy_nan(x))



    # Number of instances when dropping all NaN values

    # print(df.isnull().sum())

    # There are a lot of attirbutes with thousands of NaN vlaues in the data



    # Change the target classes to 1/0 instead of pos/neg

    def change_target_classes(cl):

        if cl=="pos":

            return 1

        else:

            return 0



    df['class'] = df['class'].apply(lambda x: change_target_classes(x))



    df = df.apply(lambda x: pd.to_numeric(x,errors='ignore'))



    df = df.fillna(value='0')

#     df.to_csv(file[:-5]+"cleaned.csv")

    return df





df = preprocess_data(training)

# Try doing some machine learning to classify

T = df['class'].values

X = df[df.columns].values[:, 1:]



# Dimensions of X

N, D = X.shape



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)



## Logistic Regression on X/T

from sklearn.linear_model import LogisticRegression



reg = LogisticRegression()



reg.fit(X,T)



test_df = preprocess_data(testing)



T_test = test_df['class'].values

X_test = test_df[test_df.columns].values[:, 1:]

scaler = StandardScaler()

X_test = scaler.fit_transform(X_test)

Y_test = reg.predict(X_test)

# print("Score for testing: ", reg.score(X_test,T_test))

submission = pd.DataFrame({'Id': test_df.index, 'SalePrice': Y_test})

from sklearn.metrics import classification_report, confusion_matrix



# print(confusion_matrix(T_test, Y_test))



for i,j in zip(T_test, Y_test):

    if i!=j:

        print("Real: ", i , " Pred: ", j)



submission.to_csv('submission.csv', index=False)
c1 = 0

c2 = 0

for i,j in zip(T_test, Y_test):

    if i == 1 and j ==0:

        print("Real: ", i , " Pred: ", j)

        c1+=10

    if i == 0 and j ==1:

        print("Real: ", i , " Pred: ", j)

        c1+=500

print(c1, c2)
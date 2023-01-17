# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

from sklearn import model_selection

from sklearn import preprocessing, cross_validation, svm

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVC



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read data

dataframe = pd.read_csv('../input/train.csv')

#dataframe = dataframe.drop(['project_id', 'name', 'desc', 'keywords', 'currency'], axis=1)

dataframe = dataframe.drop(['project_id', 'name', 'desc', 'keywords', 'currency','backers_count'], axis=1)

dataframe.head()
def handle_non_numerical_data(df):

    columns = df.columns.values

    

    #text_digit_vals = {}



    for column in columns:

        #print df[column].values

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
df = handle_non_numerical_data(dataframe)

print(df.head())

#df = preprocessing.normalize(df)



X = np.array(df.drop(['final_status'],1))

y = np.array(df['final_status'])



print('Shape X: ', X.shape)

print('Shape y: ', y.shape)



#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#print('Train Shapes: ', X_train.shape, y_train.shape)

#print('Test Shapes: ', X_test.shape, y_test.shape)


#Models

seed = 7



# prepare models

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

#models.append(('SVM', SVC())) #taking long time to execute



# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

clf = DecisionTreeClassifier()

clf = clf.fit(X, y)


# read data

dataframe_original = pd.read_csv('../input/test.csv')

dataframe = dataframe_original.drop(['project_id', 'name', 'desc', 'keywords', 'currency'], axis=1)

#dataframe['backers_count'] = 0

dataframe.head()
df = handle_non_numerical_data(dataframe)

print(df.head())



X_test = np.array(df)

print('Shape X_test: ', X_test.shape)
prediction = clf.predict(X_test)

print('prediction: ', prediction)

type(prediction)
# np.savetxt("file_name.csv", np.column_stack((data1, data2)), delimiter=",", fmt='%s', header=header)

df = dataframe_original['project_id']

project_ids = np.array(df)

print(project_ids.shape)

print(prediction.shape)



# file = np.savetxt("submission.txt", np.column_stack((project_ids, prediction)), delimiter=",", fmt='%s', header='project_id, final_status')



submission = pd.DataFrame({ 'project_ids': project_ids,

                            'final_status': prediction })

submission.to_csv("submission.csv", index=False)
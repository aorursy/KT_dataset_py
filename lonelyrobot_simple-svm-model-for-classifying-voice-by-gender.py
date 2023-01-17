import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import svm # our model
########################################

# read data from the csv, convert the labels to boolean

voice_dataframe = pd.read_csv('../input/voice.csv')

voice_dataframe['label'] = voice_dataframe['label'] == 'male'



########################################

# only keep the most significant metrics

voice_dataframe = voice_dataframe[['IQR', 'meanfun', 'label']]



########################################

# randomly shuffle the dataframe rows

voice_dataframe = voice_dataframe.sample(frac=1)



########################################

# split the data set into training, CV, and test sets

voice_dataframe_train = voice_dataframe.head(3000)

voice_dataframe_cv = voice_dataframe_train.tail(100)

voice_dataframe_train = voice_dataframe_train.head(2900)

voice_dataframe_test = voice_dataframe.tail(100)
########################################

# Create training set

y_train = voice_dataframe_train['label'].data

X_train = voice_dataframe_train.drop('label', 1).values



# Create test set

y_test = voice_dataframe_test['label'].data

X_test = voice_dataframe_test.drop('label', 1).values



# Create cross validation set

y_cv = voice_dataframe_cv['label'].data

X_cv = voice_dataframe_cv.drop('label', 1).values
########################################

# run cross-validation on the data for possible values of c

for c in range(1, 11):

    model = svm.LinearSVC(C=c*1.)

    model.fit(X_train, y_train)

    y_cv_pred = model.predict(X_cv)

    print (c, sum(y_cv_pred == y_cv))
# use our best found model (in this case, with C=1) to make the final prediction

model = svm.LinearSVC(C=1.)

model.fit(X_train, y_train)



# make the prediction

y_predicted = model.predict(X_test)



# And find the final test error

print(sum(y_predicted == y_test))
# coefficients of our decision function

# the actual decision function would be something like (h.x + c >= 0) ? true : false

print(model.coef_, model.intercept_)
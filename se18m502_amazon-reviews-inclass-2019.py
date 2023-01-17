import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import mean_squared_error as MSE

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_validate

from sklearn.utils import shuffle

from sklearn import neighbors

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn import tree

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn import ensemble

from sklearn import svm

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.feature_extraction.text import TfidfTransformer

pd.set_option('display.max_colwidth', -1)
train_df = pd.read_csv("/kaggle/input/mse-3-bb-ds-ws19-amazonreviews/amazon_review_ID.shuf.lrn.csv")

test_df = pd.read_csv("/kaggle/input/mse-3-bb-ds-ws19-amazonreviews/amazon_review_ID.shuf.tes.csv")
train_df.describe()
train_df.Class.unique()
test_id = pd.DataFrame(test_df['ID'])

test_df = test_df.drop(['ID'], axis=1)

target = pd.DataFrame(train_df['Class'])

train_df = train_df.drop(['ID', 'Class'], axis=1)
train_shuffle, target_shuffle = shuffle(train_df, target)
classifiers = [

    neighbors.KNeighborsClassifier(5),

    neighbors.KNeighborsClassifier(15),

    neighbors.KNeighborsClassifier(20),

    GaussianNB(),

    Perceptron(),

    tree.DecisionTreeClassifier(),

    ensemble.RandomForestClassifier(n_estimators=100),

    svm.SVC(),

    svm.LinearSVC()

]



classifiers_name = [

    '5-NN',

    '15-NN',

    '20-NN',

    'Naive Bayes',

    'Perceptron',

    'Full Decision Tree',

    'Random Forest (n=100)',

    'SVC',

    'LinearSVC'

]
df_results = pd.DataFrame([],columns = ['Classifier', 'Accuracy', 'std'])

scoring = ['precision_micro', 'balanced_accuracy']

for indexClassifier, classifier in enumerate(classifiers):

    scores = cross_validate(classifier, train_shuffle, target_shuffle.values.ravel(), cv=5, scoring=scoring)

    # Compute metrics

    acc = scores['test_balanced_accuracy']

    pre = scores['test_precision_micro']

    accuracy_str = str(round(acc.mean(),4)) 

    precision_str = str(round(pre.mean(),4))

    acc_std = acc.std() * 2

    df = pd.DataFrame([(classifiers_name[indexClassifier], accuracy_str, acc_std)], columns = ['Classifier', 'Accuracy', 'std'])

    df_results = df_results.append(df)
df_results
C= list(np.arange(10,50, 10))

param_grid = { 

    'C': C,

}



cl = svm.LinearSVC(max_iter=10000)

CV_rfc = GridSearchCV(estimator=cl, param_grid=param_grid, cv= 5)

CV_rfc.fit(train_shuffle, target_shuffle.values.ravel())

pd.DataFrame(CV_rfc.cv_results_).loc[:, ['mean_test_score', 'std_test_score', 'rank_test_score', 'params']].sort_values(by='rank_test_score')
criterion = ['gini', 'entropy']

max_depth = list(np.arange(10,1000, 100))

param_grid2 = { 

    'criterion': criterion,

    'max_depth': max_depth

}



cl3 = tree.DecisionTreeClassifier()

CV_rfc2 = GridSearchCV(estimator=cl3, param_grid=param_grid2, cv= 10)

CV_rfc2.fit(train_shuffle, target_shuffle.values.ravel())

pd.DataFrame(CV_rfc2.cv_results_).loc[:, ['mean_test_score', 'std_test_score', 'rank_test_score', 'params']].sort_values(by='rank_test_score')
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn import preprocessing, model_selection

from keras.layers import Dense, Dropout, Activation

encoder = LabelEncoder()

encoder.fit(target_shuffle)

encoded_Y = encoder.transform(target_shuffle.values.ravel())

# convert integers to dummy variables (i.e. one hot encoded)

dummy_y = np_utils.to_categorical(encoded_Y)
train_x, test_x, train_y, test_y = model_selection.train_test_split(train_shuffle,dummy_y,test_size = 0.1, random_state = 0)
# define baseline model

num_classes=50

attributes=10000

def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(num_classes * 20 , input_dim=attributes,  activation = 'relu' )) 





    #Output layer

    model.add(Dense(num_classes, activation = 'softmax'))

    

    

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
model = baseline_model()

model.fit(train_x, train_y, epochs = 15, batch_size = 20)

scores = model.evaluate(test_x, test_y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model = baseline_model()

model.fit(train_shuffle, dummy_y, epochs = 15, batch_size = 20)

predictions = model.predict_classes(test_df)

prediction_ = np.argmax(np_utils.to_categorical(predictions), axis = 1)

prediction_ = encoder.inverse_transform(prediction_)
prediction_df = pd.DataFrame(prediction_)

merge_nn = pd.concat([test_id, prediction_df], axis=1)

merge_nn.columns = ['ID', 'Class']
tfidf = TfidfTransformer()

tfidf_train = tfidf.fit_transform(train_shuffle)

tfidf_test = tfidf.fit_transform(test_df)
param_grid = { 

    'C': [1,2,3,4],

}

cl = svm.LinearSVC(max_iter=10000)

CV_tdidf = GridSearchCV(estimator=cl, param_grid=param_grid, cv= 5)

CV_tdidf.fit(tfidf_train, target_shuffle.values.ravel())

pd.DataFrame(CV_tdidf.cv_results_).loc[:, ['mean_test_score', 'std_test_score', 'rank_test_score', 'params']].sort_values(by='rank_test_score')
train_x, test_x, train_y, test_y = model_selection.train_test_split(tfidf_train,dummy_y,test_size = 0.1, random_state = 0)
model = baseline_model()

model.fit(train_x, train_y, epochs = 10, batch_size = 10)

scores = model.evaluate(test_x, test_y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model = baseline_model()

model.fit(tfidf_train, dummy_y, epochs = 10, batch_size = 10)

predictions = model.predict_classes(tfidf_test)

prediction_ = np.argmax(np_utils.to_categorical(predictions), axis = 1)

prediction_ = encoder.inverse_transform(prediction_)
prediction_df = pd.DataFrame(prediction_)

merge_nn_tf = pd.concat([test_id, prediction_df], axis=1)

merge_nn_tf.columns = ['ID', 'Class']
# Test with LinearSVC 1 - Default Parameters

c = svm.LinearSVC()

c.fit(train_shuffle, target_shuffle.values.ravel())

# predict

y_test_predicted1 = c.predict(test_df)
# Test with Perceptron - Default Parameters

c = Perceptron()

c.fit(train_shuffle, target_shuffle.values.ravel())

# predict

y_test_predicted2 = c.predict(test_df)
# Test with Decision Tree - Default Parameters

c = tree.DecisionTreeClassifier()

c.fit(train_shuffle, target_shuffle.values.ravel())

# predict

y_test_predicted3 = c.predict(test_df)
# Test with Decision Tree - Gini & max_deptht=510

c = tree.DecisionTreeClassifier(criterion='gini', max_depth=510)

c.fit(train_shuffle, target_shuffle.values.ravel())

# predict

y_test_predicted4 = c.predict(test_df)
# Test with td-dif and LinearSVC

c = svm.LinearSVC(C=4)

c.fit(tfidf_train, target_shuffle.values.ravel())

# predict

y_test_predicted5 = c.predict(tfidf_test)
df_result1 = pd.DataFrame(y_test_predicted1)

merge1 = pd.concat([test_id, df_result1], axis=1)

merge1.columns = ['ID', 'Class']

df_result2 = pd.DataFrame(y_test_predicted2)

merge2 = pd.concat([test_id, df_result2], axis=1)

merge2.columns = ['ID', 'Class']

df_result3 = pd.DataFrame(y_test_predicted3)

merge3 = pd.concat([test_id, df_result3], axis=1)

merge3.columns = ['ID', 'Class']

df_result4 = pd.DataFrame(y_test_predicted4)

merge4 = pd.concat([test_id, df_result4], axis=1)

merge4.columns = ['ID', 'Class']

df_result5 = pd.DataFrame(y_test_predicted5)

merge5 = pd.concat([test_id, df_result5], axis=1)

merge5.columns = ['ID', 'Class']
merge1.to_csv('LinearSVC_default.csv', index=False)

merge2.to_csv('Perceptron_default.csv', index=False)

merge3.to_csv('DT_default.csv', index=False)

merge4.to_csv('DT_Gini_510.csv', index=False)

merge5.to_csv('LSVC_td_idf.csv', index=False)
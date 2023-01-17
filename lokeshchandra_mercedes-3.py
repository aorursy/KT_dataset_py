# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn import model_selection, preprocessing, ensemble

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



# fix random seed for reproducibility

seed = 7

np.random.seed(seed)



%matplotlib inline



pd.options.mode.chained_assignment = None  # default='warn'

pd.options.display.max_columns = 999





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ", train_df.shape)

print("Test shape : ", test_df.shape)
#train_df.head()
#test_df
train_y = train_df['y']

#train_y.shape

train_x1 = train_df.ix[:,2:378]

train_id = train_df.ix[:,0]



train_df1 = train_df

test_df1 = test_df



categorical = ["X0",  "X1",  "X2", "X3", "X4",  "X5", "X6", "X8"]

for f in categorical:

        if train_df[f].dtype=='object':

            print(f)

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(train_df[f].values) + list(test_df[f].values))

            train_df1[f] = lbl.transform(list(train_df[f].values))

            test_df1[f] = lbl.transform(list(test_df[f].values))



nptrain_y = np.array(train_y.as_matrix())

#nptrain_y



nptrain_x = np.array(train_df.as_matrix())

nptest_x= np.array(test_df.as_matrix())

2#nptrain_x = nptrain_x[:,2:378]



#nptrain_id = nptrain_x[:,0]

#nptrain_x1 = np.c(nptrain_x, ones())

#nptrain_x.shape[1]



#nptrain_x1 = np.empty([nptrain_x.shape[0],nptrain_x.shape[1]+1])

#nptrain_x1[:,1:377] = nptrain_x



nptrain_xId = nptrain_x[:,0]

#nptrain_xId

nptrain_xCl = nptrain_x[:,2:10]

#nptrain_xCl

nptrain_xNu = nptrain_x[:,11:]

nptrain_xNu.shape
nptest_xId = nptest_x[:,0]

#nptrain_xId

nptest_xCl = nptest_x[:,1:9]

#nptrain_xCl

nptest_xNu = nptest_x[:,10:]

nptest_xNu.shape
def baseline_model():



 model = Sequential()

 model.add(Dense(1000, input_dim=367, kernel_initializer='normal', activation='relu'))

 model.add(Dense(50,  kernel_initializer='normal', activation='relu'))

 model.add(Dense(1, kernel_initializer='normal'))

 model.compile(loss='mean_squared_error', optimizer='adam')



 return model
#estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)



#estimators = []

#estimators.append(('standardize', StandardScaler()))

#estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=200, verbose=0)))

#pipeline = Pipeline(estimators)

#kfold = KFold(n_splits=4, random_state=seed)

#results = cross_val_score(pipeline, nptrain_xNu, nptrain_y, cv=kfold)

#print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))



#kfold = KFold(n_splits=10, random_state=seed)

#results = cross_val_score(estimator, nptrain_xNu, nptrain_y, cv=kfold)

#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
scale = StandardScaler()

X_trainNu = scale.fit_transform(nptrain_xNu)

X_testNu = scale.fit_transform(nptest_xNu)



clf = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=50,verbose=0)



#estimators = []

#estimators.append(('standardize', StandardScaler()))

#estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=200, verbose=0)))

#pipeline = Pipeline(clf)



clf.fit(X_trainNu,nptrain_y)



kfold = KFold(n_splits=4, random_state=seed)

results = cross_val_score(clf, X_trainNu, nptrain_y, cv=kfold)

print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))





res = clf.predict(X_testNu)

res.shape

res


 # predefine or use append





#for num in range(0, res.shape[0]):

#    label = nptest_xId[num]

#    pred = res[num]

#    rows[num] = "%d,%d\n"%(label,pred)



#np.savetxt("foo.csv", res, delimiter=",")
sub = pd.DataFrame()

sub['ID'] = nptest_xId

sub['y'] = res

sub.to_csv('output.csv', index=False)

sub
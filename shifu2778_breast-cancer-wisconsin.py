# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data.csv",header=0)
print(data.head(5))
data.info()
data.drop("Unnamed: 32",axis=1,inplace=True)
#we don't need id either

data.drop("id", axis=1, inplace=True)
#let's divide the data into 3 parts i.e. mean, se and worst

mean= list(data.columns[1:11])

se= list(data.columns[11:21])

worst= list(data.columns[21:31])

print(mean)

print(se)

print(worst)
#let's start with the mean part

data['diagnosis']= data['diagnosis'].map({'M':1, 'B':0})
import seaborn as sns

sns.countplot(data['diagnosis'], label="Count")
data1 = data[mean]

vals= data1.values

X = vals[:, 0:10]

y = data['diagnosis'].values
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()

model.fit(X, y)

print(model.feature_importances_)
data1.head(2)
# for random forest

predrfc = ['radius_mean','perimeter_mean','area_mean','concavity_mean','concave points_mean']
#for logistic regression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

rfe = RFE(model, 5)

fit = rfe.fit(X, y)

print("Num Features: %d" % (fit.n_features_))

print("Selected Features: %s" % (fit.support_))

print("Feature Ranking: %s" % (fit.ranking_))

predlog = ['radius_mean','perimeter_mean','concavity_mean','concave points_mean','symmetry_mean']
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.3)
from sklearn import metrics

from sklearn.cross_validation import KFold

def classification_model(model, data, predictors, outcome):

    model.fit(data[predictors],data[outcome])

    predictions = model.predict(data[predictors])

    accuracy = metrics.accuracy_score(predictions,data[outcome])

    print("Accuracy : %s" % "{0:.3%}".format(accuracy))



    #Perform k-fold cross-validation with 5 folds

    kf = KFold(data.shape[0], n_folds=5)

    error = []

    for train, test in kf:

        

        # Filter training data

        train_predictors = (data[predictors].iloc[train,:])

    

        # The target we're using to train the algorithm.

        train_target = data[outcome].iloc[train]

    

        # Training the algorithm using the predictors and target.

        model.fit(train_predictors, train_target)

    

        #Record error from each cross-validation run

        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))

    

        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    

    #Fit the model again so that it can be refered outside the function:

    model.fit(data[predictors],data[outcome]) 
predictor_var = predlog

outcome_var='diagnosis'

model=LogisticRegression()

classification_model(model,train,predictor_var,outcome_var)
predictor_var = ['radius_mean']

model=LogisticRegression()

classification_model(model,train,predictor_var,outcome_var)
from sklearn.tree import DecisionTreeClassifier

predictor_var = predrfc

outcome_var='diagnosis'

model=DecisionTreeClassifier()

classification_model(model,train,predictor_var,outcome_var)
predictor_var = ['radius_mean']

model=DecisionTreeClassifier()

classification_model(model,train,predictor_var,outcome_var)
from sklearn.ensemble import RandomForestClassifier

predictor_var = predrfc

model = RandomForestClassifier(n_estimators=100,min_samples_split=25, max_depth=7, max_features=2)

classification_model(model, train,predictor_var,outcome_var)
# using only one predictor



predictor_var = ['radius_mean']

model = RandomForestClassifier(n_estimators=100)

classification_model(model, train,predictor_var,outcome_var)
# using all the features



predictor_var = mean

model = RandomForestClassifier(n_estimators=100,min_samples_split=25, max_depth=7, max_features=2)

classification_model(model, train,predictor_var,outcome_var)
predictor_var = mean

model = RandomForestClassifier(n_estimators=100,min_samples_split=25, max_depth=7, max_features=2)

classification_model(model, test,predictor_var,outcome_var)
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
import pandas as pd
sampleSubmissionDf= pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

#sampleSubmissionDf.head()

testDf= pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

#testDf.head()

trainDf= pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

trainDf.head()
dir(trainDf)
trainDf.index +=1

trainDf.index
trainDf.head()
trainDf.drop(columns=["label"]) #, axis=1



from sklearn.ensemble import RandomForestClassifier

import numpy as np

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(trainDf.drop(columns=["label"]), trainDf.label, test_size=0.2)

model = RandomForestClassifier(n_jobs=2, random_state=0)



model.fit(X_train, y_train)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_split=1e-07, min_samples_leaf=1,

            min_samples_split=2, min_weight_fraction_leaf=0.0,

            n_estimators=10, n_jobs=2, oob_score=False, random_state=0,

            verbose=0, warm_start=False)
preds = model.predict(X_test)
def evaluate_model(preds, y_test):

    print(pd.crosstab(y_test, preds, rownames=['Actual Result'], colnames=['Predicted Result']))

    # calculate accuracy

    from sklearn import metrics

    print("\nAccuracy is " + str(metrics.accuracy_score(y_test, preds)))

    #print(metrics.confusion_matrix(y_test, preds))
evaluate_model(preds, y_test)
#Run Model on the whole data set



model.fit(trainDf.drop(columns=["label"]), trainDf.label)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_split=1e-07, min_samples_leaf=1,

            min_samples_split=2, min_weight_fraction_leaf=0.0,

            n_estimators=10, n_jobs=2, oob_score=False, random_state=0,

            verbose=0, warm_start=False)
preds = model.predict(testDf)
type(preds)
pd.DataFrame(data=preds[:,],    # values

index=preds[:,],    # 1st column as index

columns=["Label"])  # 1st row as the column names
submissionDf=pd.DataFrame({'Label': preds[:, ]})

submissionDf.index +=1
submissionDf["ImageId"] = submissionDf.index

submissionDf=submissionDf[["ImageId","Label"]]
submissionDf.to_csv('submission.csv', index=False)
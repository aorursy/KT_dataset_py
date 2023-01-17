# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys # Imported this for some printing functionality 

import xgboost as xgb

# Panda is going to read my CSV data



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# The below code, # List the files we have available to work with

# The subprocess module allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes.

# In computing, ls is a command to list files in Unix and Unix-like operating systems.



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8")) 

# TF-8 is a character encoding capable of encoding all possible characters, or code points

# Other coders have not included the above submission, not sure what the requirement is. 



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', header=0)

# header=0 denotes the first line of data rather than the first line of the file.

# Explicitly pass header=0 to be able to replace existing names. 

# Another way, if you want to explicitly state their names

# ('params.csv', names=paramnames)

test = pd.read_csv('../input/test.csv', header=0)
type(train)

type(test)
# I think if you add the below, the print function below will look better

pd.set_option('display.max_columns', 7)



print (train.values)

# It seems that python 3 requires paranthesis. 

# To convert the data from the panda datafram to the numpy-array matrix representation do this:

# The above did not run when I ran the type command.

# train.columns

# type(train)

# The below store the names of the columns, paramdata.index

print(train.index)

# The below one works fairly good, you can actually see what you are printing.

train.to_csv(sys.stdout)



#.base just seems to be a subset of the sklearn database

# In computing, NaN, standing for not a number,

# The below code are so replicatable that it is a straight copy from stackoverflow

from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],

            index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.fill)

    

# At the moment the above is just generic behavioural code, below follows the executory code

# You still have to define on which of the variables you would like the imputation to occur



feature_pre = ['Pclass','Sex','Age','Fare','Parch']



feature_append = train[feature_pre].append(test[feature_pre])

# Append stacks it below eachother, here we join the train and 



# Below is the executory imputation code

feature_post = DataFrameImputer().fit_transform(feature_append)



# Importantly we want to xboost as the method to predict whether someone would be saved or not

# xboost would have to be catagorical given the catagorical nature of the test

# xboost however has an issue with catagorical datatypes ironically



# XGBoost doesn't (yet) handle categorical features automatically, so we need to change

# them to columns of integer values.

# See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more

# details and options, nan being non numeric value



nan_post = ['Sex']



# below follows the encoding that is required to flip cat to int

from sklearn import preprocessing as pre



le = pre.LabelEncoder()

for blue in nan_post:  # For all identifier in the nan_post column, transform them to integer

    feature_post[blue] = le.fit_transform(feature_post[blue])

  # Now feature post is perfectly updated    













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
train_X = feature_post[0:train.shape[0]].as_matrix()

# Notice what we are doing, we are jsut stealing the shape from the previosu variable to reshape to

# training and testing data, i.e. splitting it back into order.

test_X = feature_post[train.shape[0]::].as_matrix()

# Sometimes it is the best to at the very beginning, know exactely know the target variable you are

# working towards, in our case, we are working towards survived.

train_y = train['Survived']
# Now finally we can use xGboost to run the predictions

# You can experiment with many other options here, using the same .fit() and .predict()

# methods; see http://scikit-learn.org

# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost



gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)

predictions = gbm.predict(test_X)

# Kaggle needs the submission to have a certain format;

# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv

# for an example of what it's supposed to look like.

submission = pd.DataFrame({ 'PassengerId': test['PassengerId'],

                            'Survived': predictions })

submission.to_csv("submission.csv", index=False)


    

   
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
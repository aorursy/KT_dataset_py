# Importing libraries 

from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split



import pandas as pd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
data=pd.read_csv('../input/train.csv')

data.head()
# Splitting data into training and test set

X_train, X_test, y_train, y_test = train_test_split(data.drop(['id','target'],axis=1),data['target'],train_size=0.75, test_size=0.25)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
tpot = TPOTClassifier(generations=8, population_size=50, verbosity=2)

tpot.fit(X_train, y_train)

print("Accuracy is {}%".format(tpot.score(X_test, y_test)*100))
data_test=pd.read_csv('../input/test.csv')

data_test.head()
tpot.export('tpot_exported_pipeline.py')
# import module we'll need to import our custom module

#from shutil import copyfile



# copy our file into the working directory (make sure it has .py suffix)

#copyfile(src = "../input/tpot_exported_pipeline.py", dst = "../working/tpot_exported_pipeline.py")



# import all our functions

#from tpot_exported_pipeline import *
tpot.predict(data_test.drop(['id'],axis=1))
submission=pd.DataFrame({'id': data_test['id'], 'target':tpot.predict(data_test.drop(['id'],axis=1)) })

submission.to_csv('submission.csv',index=False)
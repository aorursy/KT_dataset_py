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
#Validate working directory

os.getcwd() 

print(os.getcwd())

#Validate Current Path and create Path to data

from pathlib import Path

INPUT = Path("../input/digit-recognizer")

os.listdir(INPUT)
#Import CSV into Pandas dataframe and test shape of file 

train_df = pd.read_csv(INPUT/"train.csv")

train_df.shape

#Split training data 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['label'], 

                                                                  axis=1), 

                                                    train_df["label"], 

                                                    shuffle = True, 

                                                    train_size = .6, 

                                                    random_state = 1)

# Check the shape of the trainig data set array

print('Shape of X_train_data:', X_train.shape)

print('Shape of y_train_data:', y_train.shape)

print('Shape of X_test_data:', X_test.shape)

print('Shape of y_test_data:', y_test.shape)
from sklearn.ensemble import RandomForestClassifier as rfc



rf = rfc(random_state = 1234, 

         n_jobs = -1, 

         n_estimators=100)



rf.fit(X_train, y_train)

print("Training Accuracy: ", rf.score(X_train, y_train))

print("Testing Accuracy: ", rf.score(X_test, y_test))
#possible parameters to adjust

print(rf.get_params())
from sklearn.model_selection import RandomizedSearchCV

criterion = ['entropy','gini']

n_estimators = [50, 100, 250, 500]

random_grid = {'criterion': criterion, 

               'n_estimators': n_estimators}
rf_random = RandomizedSearchCV(estimator = rf, 

                               param_distributions = random_grid, 

                               n_iter = 10, 

                               cv = 2, 

                               verbose = 2, 

                               random_state = 12, 

                               n_jobs = -1)



rf_random.fit(X_train, y_train)
rf_random.best_params_
rf = rfc(n_estimators = 500, 

       criterion = 'gini', 

       bootstrap = True)
from datetime import datetime

start=datetime.now()

rf.fit(X_train, y_train)

end=datetime.now()

print("time:", end-start)
print("Training Accuracy: ", rf.score(X_train, y_train))

print("Testing Accuracy: ", rf.score(X_test, y_test))
test_df = pd.read_csv(INPUT/"test.csv")

test_df.shape
prediction = rf.predict(test_df)

print (prediction)

ImageID = range(1, len(prediction)+1)

output = {'ImageId': ImageID, 'Label': prediction}

Output = pd.DataFrame.from_dict(output)

Output.set_index('ImageId', inplace = True)

print (Output)

Output.to_csv('jwmyers82_digit_recognizer_rfc.csv')
from sklearn.metrics import f1_score as f1

mypred_train = rf.predict(X_train)

mypred_test = rf.predict(X_test)

print("Training F1 Accuracy: ", f1(mypred_train, y_train, average = 'macro'))

print("Testing F1 Accuracy: ", f1(mypred_test, y_test, average = 'macro'))
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



x = np.concatenate((X_train, X_test), axis = 0)

x = StandardScaler().fit_transform(x)

pca = PCA(.98)

pca.fit(x)

totimages = pca.transform(x)

pca.n_components_
trainimages = totimages[0:25200,:] 

testimages = totimages[25200:42000,:]
trainimages = trainimages.astype(int)

testimages = testimages.astype(int)
rf2 = rfc(n_estimators=500, 

          criterion='gini', 

          bootstrap=True)
start=datetime.now()

rf2.fit(trainimages, y_train)

end=datetime.now()

print(end - start)
print("Accuracy: ", rf2.score(testimages, y_test))
mypred2 = rf2.predict(testimages)

print("F1 Accuracy: ", f1(mypred2, y_test, average='macro'))
prediction2 = rf2.predict(pca.transform(test_df))

print (prediction2)

ImageID = range(1, len(prediction2)+1)

output2 = {'ImageId': ImageID, 'Label': prediction2}

Output2 = pd.DataFrame.from_dict(output2)

Output2.set_index('ImageId', inplace = True)

print (Output2)

Output2.to_csv('jwmyers82_digit_recognizer_rfc_pca.csv')
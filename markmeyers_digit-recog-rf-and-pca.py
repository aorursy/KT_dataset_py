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



# Any results you write to the current directory are saved as output.

#Validate working directory

os.getcwd()

print(os.getcwd())

#Validate Current Path and create Path to data

from pathlib import Path

INPUT = Path("../input/digit-recognizer")

os.listdir(INPUT)



#Import CSV into Pandas dataframe and test shape of file

train_df = pd.read_csv(INPUT/"train.csv")

train_df.head(3)

train_df.shape



#Split training data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['label'], axis=1), train_df["label"], shuffle=True,

train_size=.75, random_state=1)

# Check the shape of the trainig data set array

print('Shape of X_train_data:', X_train.shape)

print('Shape of y_train_data:', y_train.shape)

print('Shape of X_test_data:', X_test.shape)

print('Shape of y_test_data:', y_test.shape)
import matplotlib.pyplot as plt

import matplotlib

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn import preprocessing
dx = ()

dx = train_df

dx = dx.drop(['label'], axis = 1)

dx.shape
i = 42

ax = dx.iloc[i].values

some_digit = ax

some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,

           interpolation="nearest")

plt.axis("off")

plt.title(print("This should be showing a(n)",train_df.label[i], "."))

plt.show()
train_df.label.value_counts().sort_index().plot(kind="bar")

plt.title("Distribution of Digits in dataset")

plt.show()
Reduction = y_train.value_counts().sort_index() / train_df.label.value_counts().sort_index()



#visualize represented label quantity to ensure adequate representation of ALL labels

Reduction.plot(kind="bar")

plt.title("Reduction via train/test split of Digits in original vs training")

plt.show()
# Check the shape of the trainig data set array

print('Shape of X_train_data:', X_train.shape)

print('Shape of y_train_data:', y_train.shape)

print('Shape of X_test_data:', X_test.shape)

print('Shape of y_test_data:', y_test.shape)
from sklearn.ensemble import RandomForestClassifier as rfc

from datetime import datetime

rf = rfc(n_estimators=200, criterion = 'gini', bootstrap=True)

start=datetime.now()

rf.fit(X_train, y_train)

end=datetime.now()

print("Success! Time to calc:", end-start)
print("Accuracy:", rf.score(X_test, y_test))
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from datetime import datetime

start=datetime.now()

x = np.concatenate((X_train,X_test), axis = 0)

x = StandardScaler().fit_transform(x)

pca = PCA(.95)

pca.fit(x)

totimages = pca.transform(x)

end=datetime.now()

print("Standardized dataset")

print("Qty components:", pca.n_components_)

print("Time to calc:", end-start)
X_reduced = totimages[0:31500,:]

X_test_reduced = totimages[31500:42000,:]

X_reduced = X_reduced.astype(int)

X_test_reduced = X_test_reduced.astype(int)

X_reduced.shape
rf2 = rfc(n_estimators=200, criterion = 'gini', bootstrap=True)

start=datetime.now()

rf2.fit(X_reduced, y_train)

end=datetime.now()

print("Success! Time to calc:", end-start)
print("Accuracy:", rf2.score(X_test_reduced, y_test))
test_df = pd.read_csv(INPUT/"test.csv")

test_df.head(3)
y_test_rf = rf.predict(test_df)
y_test_rf.shape
test_df_pca = pca.transform(test_df)

test_df_pca.shape
y_test_rf2 = rf2.predict(test_df_pca)
y_test_rf2.shape
sample_df = pd.read_csv(INPUT/"sample_submission.csv")

sample_df.head(3)

submission1 = ()

submission1 = pd.DataFrame({'Imageid' : np.arange(1, len(y_test_rf) + 1), 'Label' : y_test_rf})

submission1.head(3)
submission1.to_csv("RF_Classifier.csv", index=False)
submission2 = ()

submission2 = pd.DataFrame({'Imageid' : np.arange(1, len(y_test_rf2) + 1), 'Label' : y_test_rf2})

submission2.head(3)
submission2.to_csv("PCA_RF_Classifier.csv", index=False)
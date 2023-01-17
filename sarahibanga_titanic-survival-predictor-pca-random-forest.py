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
train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("../input/titanic/test.csv")

test_data.head()
# Split data into class labels

# Drop Survived from X label since that is what we will try to predict. 

# Drop categorical columns Name and Ticket.

# Also exclude PassengerId

train_X = train_data.drop(['Survived', 'Name', 'Ticket', 'PassengerId'], axis = 1)

train_Y = train_data['Survived']
columns_to_exclude = train_data.columns[train_data.isnull().any()]

train_X.drop(columns_to_exclude, axis = 1, inplace = True)
import pandas as pd

from sklearn.decomposition import PCA as PCA



def perform_pca(data, num_components=None, display_pc_matrix=False):

    """ Specify the number of components (by default all), perform PCA, and fit model to data."""

    pca = PCA() if num_components is None else PCA(num_components)

    pca.fit(data)

    

    # Output the principal components in order of explained variance ratio

    pc_indices = ['PC-%s'% str(a + 1) for a in range(len(pca.components_))]

    display([(pc_indices[ind], b) for ind,b in enumerate(pca.explained_variance_ratio_)])

    

    # Output the matrix comparing features to components

    if display_pc_matrix:

        display(pd.DataFrame(abs(pca.components_), columns=data.columns, index = pc_indices))

        display("Least inportant feature by principal component:", pd.DataFrame(abs(pca.components_), columns=data.columns).idxmin(axis=1))

        



one_hot_encoded_training_data = pd.get_dummies(train_X)

perform_pca(one_hot_encoded_training_data)
perform_pca(one_hot_encoded_training_data, num_components=4, display_pc_matrix=True)
from sklearn.ensemble import RandomForestClassifier



features = ["Pclass", "SibSp", "Parch", "Sex"]

X = pd.get_dummies(train_X[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=1)

model.fit(X, train_Y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
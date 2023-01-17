#Let us import some modules

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split # Train-Test-Split
df = pd.read_json(open("../input/train.json", "r"))



#Encode label

from sklearn import preprocessing



lbl = preprocessing.LabelEncoder()

lbl.fit(list(df['manager_id'].values))

df['manager_id'] = lbl.transform(list(df['manager_id'].values))





X = df.drop(["interest_level"], axis = 1)

y = df["interest_level"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)



X_train.head()
#Our feature construction class will inherit from these two base classes of sklearn.

from sklearn.base import BaseEstimator

from sklearn.base import TransformerMixin



class manager_skill(BaseEstimator, TransformerMixin):

    """

    Adds the column "manager_skill" to the dataset, based on the Kaggle kernel

    "Improve Perfomances using Manager features" by den3b. The function should

    be usable in scikit-learn pipelines.

    

    Parameters

    ----------

    threshold : Minimum count of rental listings a manager must have in order

                to get his "own" score, otherwise the mean is assigned.



    Attributes

    ----------

    mapping : pandas dataframe

        contains the manager_skill per manager id.

        

    mean_skill : float

        The mean skill of managers with at least as many listings as the 

        threshold.

    """

    def __init__(self, threshold = 5):

        

        self.threshold = threshold

        

    def _reset(self):

        """Reset internal data-dependent state of the scaler, if necessary.

        

        __init__ parameters are not touched.

        """

        # Checking one attribute is enough, becase they are all set together

        # in fit        

        if hasattr(self, 'mapping_'):

            

            self.mapping_ = {}

            self.mean_skill_ = 0.0

        

    def fit(self, X,y):

        """Compute the skill values per manager for later use.

        

        Parameters

        ----------

        X : pandas dataframe, shape [n_samples, n_features]

            The rental data. It has to contain a column named "manager_id".

            

        y : pandas series or numpy array, shape [n_samples]

            The corresponding target values with encoding:

            low: 0.0

            medium: 1.0

            high: 2.0

        """        

        self._reset()

        

        temp = pd.concat([X.manager_id,pd.get_dummies(y)], axis = 1).groupby('manager_id').mean()

        temp.columns = ['low_frac', 'medium_frac', 'high_frac']

        temp['count'] = X.groupby('manager_id').count().iloc[:,1]

        

        print(temp.head())

        

        temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']

        

        mean = temp.loc[temp['count'] >= self.threshold, 'manager_skill'].mean()

        

        temp.loc[temp['count'] < self.threshold, 'manager_skill'] = mean

        

        self.mapping_ = temp[['manager_skill']]

        self.mean_skill_ = mean

            

        return self

        

    def transform(self, X):

        """Add manager skill to a new matrix.

        

        Parameters

        ----------

        X : pandas dataframe, shape [n_samples, n_features]

            Input data, has to contain "manager_id".

        """        

        X = pd.merge(left = X, right = self.mapping_, how = 'left', left_on = 'manager_id', right_index = True)

        X['manager_skill'].fillna(self.mean_skill_, inplace = True)

        

        return X
#Initialize the object

trans = manager_skill()

#First, fit it to the training data:

trans.fit(X_train, y_train)

#Now transform the training data

X_train_transformed = trans.transform(X_train)

#You can also do fit and transform in one step:

X_train_transformed = trans.fit_transform(X_train, y_train)

X_train_transformed.head()
X_val_transformed = trans.transform(X_val)

X_val_transformed.head()
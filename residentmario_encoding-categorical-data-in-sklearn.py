import numpy as np

import pandas as pd

import pandas as pd

from sklearn import preprocessing

import category_encoders as ce





def get_cars_data():

    """

    Load the cars dataset, split it into X and y, and then call the label encoder to get an integer y column.

    :return:

    """

    df = pd.read_csv(

        'https://raw.githubusercontent.com/scikit-learn-contrib/categorical-encoding/master/examples/source_data/cars/car.data.txt'

    )

    X = df.reindex(columns=[x for x in df.columns.values if x != 'class'])

    y = df.reindex(columns=['class'])

    y = preprocessing.LabelEncoder().fit_transform(y.values.reshape(-1, ))



    mapping = [

        {'col': 'buying', 'mapping': [('vhigh', 0), ('high', 1), ('med', 2), ('low', 3)]},

        {'col': 'maint', 'mapping': [('vhigh', 0), ('high', 1), ('med', 2), ('low', 3)]},

        {'col': 'doors', 'mapping': [('2', 0), ('3', 1), ('4', 2), ('5more', 3)]},

        {'col': 'persons', 'mapping': [('2', 0), ('4', 1), ('more', 2)]},

        {'col': 'lug_boot', 'mapping': [('small', 0), ('med', 1), ('big', 2)]},

        {'col': 'safety', 'mapping': [('high', 0), ('med', 1), ('low', 2)]},

    ]



    return X, y, mapping



X, y, mapping = get_cars_data()
bus_or_taxi = pd.DataFrame(

    np.where(np.random.random((1, 1000)) > 0.5, 'bus', 'taxi').T,

    columns=['mode']

)
encoder = ce.OneHotEncoder()

encoder.fit_transform(bus_or_taxi).head()
encoder.category_mapping
encoder = ce.OneHotEncoder(use_cat_names=True)

encoder.fit_transform(bus_or_taxi).head()
encoder = ce.OrdinalEncoder()

encoder.fit_transform(bus_or_taxi).head()
X.head()
import category_encoders as ce

X_trans = ce.PolynomialEncoder().fit_transform(X, y)

X_trans.head()
import category_encoders as ce

X_trans = ce.HelmertEncoder().fit_transform(X, y)

X_trans.head()
import category_encoders as ce

X_trans = ce.BinaryEncoder().fit_transform(X)

X_trans.head()
import category_encoders as ce

X_trans = ce.HashingEncoder().fit_transform(X)

X_trans.head()
import category_encoders as ce

X_trans = ce.BaseNEncoder(base=2).fit_transform(X)

X_trans.head()

# up to three columns per class
import category_encoders as ce

X_trans = ce.BaseNEncoder(base=4).fit_transform(X)

X_trans.head()

# up to two columns per class
import category_encoders as ce

X_trans = ce.BackwardDifferenceEncoder().fit_transform(X)

X_trans.head()
import category_encoders as ce

X_trans = ce.TargetEncoder().fit_transform(X, y)

X_trans.head()
import category_encoders as ce

X_trans = ce.TargetEncoder().fit_transform(X, y)

X_trans.head()
import category_encoders as ce

X_trans = ce.LeaveOneOutEncoder().fit_transform(X, y)

X_trans.head()
import category_encoders as ce

X_trans = ce.CatBoostEncoder().fit_transform(X, y)

X_trans.head()
import category_encoders as ce

X_trans = ce.MEstimateEncoder().fit_transform(X, y)

X_trans.head()
import category_encoders as ce

X_trans = ce.SumEncoder().fit_transform(X, y)

X_trans.head()
import category_encoders as ce

X_trans = ce.WOEEncoder().fit_transform(X, (y > 1).astype(int))

X_trans.head()
import category_encoders as ce

X_trans = ce.JamesSteinEncoder().fit_transform(X, y)

X_trans.head()
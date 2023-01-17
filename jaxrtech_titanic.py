import numpy as np

import pandas as pd
def read_csv(path):

    hints = {"PassengerId": np.uint32, "Age": np.float64}

    return pd.read_csv(path, dtype=hints).set_index('PassengerId')



train_raw = read_csv("../input/train.csv")

test_raw = read_csv("../input/test.csv")
# columns that contain NaN

pd.isnull(test_raw).any()
# average survival rate

train_raw['Survived'].sum() / len(train_raw)
# note that essentially everything is working on a view unless you use `df.copy()`



def clean(df):

    dfx = df

    included_dtypes = ['int', 'float']

    

    dfx['Gender'] = dfx['Sex'].map({'female': -1, 'male': 1}).astype(int)

    dfx = dfx.select_dtypes(include=included_dtypes)

    dfx = dfx.dropna()

    

    return dfx

        



train = clean(train_raw.copy())

test = clean(test_raw.copy())
from random import random



def predict(classifier):

    train_input = train.drop('Survived', axis=1).values

    train_target = train['Survived'].values



    model = classifier.fit(train_input, train_target)

    

    test_input = test.values

    predicted = model.predict(test_input)

    output = pd.DataFrame(data={'Survived': predicted}, index=test.index)

    

    return output

    

def write_csv(output_path, output):

    # TODO: it's inefficent to throw away other columns since one is NaN

    # it would be better to create another "limited" model using only non-NaN columns

    def replace_nan(x):

        if not pd.isnull(x):

            return x



        p = 0.38

        return 1 if random() >= p else 0

    

    csv = test_raw[[]].join(output, how='left')

    csv['Survived'] = csv['Survived'].map(replace_nan).astype(np.uint8) # joining will produce NaN which requires a float

    

    csv_result = csv[['Survived']]

    csv_result.to_csv(output_path, index=True)

    

    return csv_result



def run(output_path, classifier):

    output = predict(classifier)

    return write_csv(output_path, output)

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier



classifiers = {

    'random_forest_output.csv':

        RandomForestClassifier(n_estimators = 100),

    

    'nearest_neighbors.csv':

        KNeighborsClassifier(n_neighbors = 5),

    

    'neural_net.csv':

        MLPClassifier(solver='lbfgs', alpha=0.01, hidden_layer_sizes=(5, 2), random_state=1),

}
[run(path, classifier) for path, classifier in classifiers.items()]
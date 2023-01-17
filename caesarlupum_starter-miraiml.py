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
!pip install miraiml
# Read the data

data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

data = data[['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd', 'SalePrice']]
from sklearn.model_selection import train_test_split



train_data, test_data = train_test_split(data, test_size=0.2)
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import  LinearRegression

from sklearn.preprocessing import QuantileTransformer

from sklearn.preprocessing import StandardScaler



from miraiml import SearchSpace

from miraiml.pipeline import compose



Pipeline = compose(

    [('scaler', StandardScaler), ('linear_reg', LinearRegression)]

)



search_spaces = [

    SearchSpace(

        id='k-NeighborsRegressor',

        model_class=KNeighborsRegressor,

        parameters_values=dict(

            n_neighbors=range(2, 9),

            weights=['uniform', 'distance'],

            p=range(2, 5)

        )

    ),

    SearchSpace(

        id='Pipeline',

        model_class=Pipeline,

        parameters_values=dict(

            scaler__with_mean=[True, False],

            scaler__with_std=[True, False],

            lin_reg__fit_intercept=[True, False]

        )

    )

]
from sklearn.metrics import r2_score



from miraiml import Config



config = Config(

    local_dir='miraiml_local',

    problem_type='regression',

    score_function=r2_score,

    search_spaces=search_spaces,

    ensemble_id='Ensemble'

)
from miraiml import Engine



def on_improvement(status):

    scores = status.scores

    for key in sorted(scores.keys()):

        print('{}: {}'.format(key, round(scores[key], 3)), end='; ')

    print()



engine = Engine(config=config, on_improvement=on_improvement)
engine.load_train_data(train_data, 'SalePrice')

engine.load_test_data(test_data)
from time import sleep



engine.restart()



sleep(1)



print('\nShuffling train data')

engine.shuffle_train_data(restart=True)



sleep(1)



engine.interrupt()
status = engine.request_status()
print(status.build_report(include_features=True))
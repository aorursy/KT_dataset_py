import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
raw = pd.read_csv("../input/train.csv")

raw.isnull().sum()
raw.head(5)
from datetime import datetime

pickup = []

dropoff = []

diffTrip = []

for i in range(raw.shape[0]):

    pickup.append(datetime.strptime(raw["pickup_datetime"][i], "%Y-%m-%d %H:%M:%S"))

    dropoff.append(datetime.strptime(raw["dropoff_datetime"][i], "%Y-%m-%d %H:%M:%S"))

    diffTrip.append(dropoff[i] - pickup[i])
val = diffTrip[0]

val.seconds
for i in range(len(diffTrip)):

    diffTrip[i] = diffTrip[i].seconds

se = pd.Series(diffTrip)

raw['diffTrip'] = se.values

raw.head(5)
x = raw.drop(["id", "pickup_datetime", "store_and_fwd_flag", "dropoff_datetime"],1)

y = raw["trip_duration"]
var_dummy = pd.get_dummies(raw["store_and_fwd_flag"])

x_catnum= pd.concat([x, var_dummy], axis = 1)

x_catnum.head(5)
X= x_catnum

Y= y

# try normalize data, maybe improve accuration

from sklearn.preprocessing import StandardScaler

# get column name because we lose it after standarization

data_columns = X.columns

# initiate standarscaler

scaler = StandardScaler()

# fitting and transform to dataframe feature data

#X = scaler.fit_transform(X)

scaler.fit(X)

normal_X = pd.DataFrame(scaler.transform(X))

# get column name back

normal_X.columns = data_columns

# check data after standardize

normal_X.head(5)
# split data train and data testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

  X,

  Y,

  test_size=0.2,

  random_state = 42 )
from sklearn import metrics

from sklearn.model_selection import cross_val_score, KFold

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import RANSACRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor





classifiers = [

    LinearRegression(),

    Ridge(),

    Lasso(),

    RandomForestRegressor(n_jobs=-1),

    DecisionTreeRegressor(),

    RANSACRegressor(LinearRegression(), 

                     max_trials=100, 

                     min_samples=50, 

                     loss='absolute_loss', 

                     residual_threshold=5.0, 

                     random_state=0)]







for clf in classifiers:

    

    clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    

    ########################## perform split validation ######################

    train_predictions = clf.predict(X_test)

    rmse = np.sqrt( metrics.mean_squared_error( y_test, train_predictions ) )

    print("RMSE: {}".format(rmse))

    ##########################################################################

    

    ########################## perform 10 fold validation ######################

    kf = KFold(n_splits=10)

    scorelist = []

    for train_index, test_index in kf.split(X.values):

        clf.fit(X.values[train_index], Y.values[train_index])

        p = clf.predict(X.values[test_index])

        RMSE = metrics.mean_squared_error(Y.values[test_index], p)**0.5

        scorelist.append(rmse)

    

    print("MeanCVScore: {}".format(sum(scorelist)/len(scorelist)))

    print("10FoldCVScore: {}".format(scorelist))

    #############################################################################

    

print("="*30)
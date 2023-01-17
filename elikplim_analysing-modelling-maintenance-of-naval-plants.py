import numpy

import pandas



from sklearn.feature_selection import RFE

from sklearn.ensemble import ExtraTreesRegressor





import matplotlib.pyplot as plt





from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import explained_variance_score

#from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error





from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.utils import np_utils

from keras.constraints import maxnorm



# fix random seed for reproducibility

seed = 7

numpy.random.seed(seed)
# load dataset

dataframe = pandas.read_csv(r"../input/navalplantmaintenance.csv", delim_whitespace=True, header=None)



dataframe = dataframe.round(3)



# Assign names to Columns

dataframe.columns = ['lever_position', 'ship_speed', 'gt_shaft', 'gt_rate', 'gg_rate', 'sp_torque', 'pp_torque', 'hpt_temp', 'gt_c_i_temp', 'gt_c_o_temp', 'hpt_pressure', 'gt_c_i_pressure', 'gt_c_o_pressure', 'gt_exhaust_pressure', 'turbine_inj_control', 'fuel_flow', 'gt_c_decay',  'gt_t_decay']



dataframe = dataframe.dropna()



dataframe = dataframe.drop('gt_c_i_temp', axis=1)
print("Head:", dataframe.head())



print("Statistical Description:", dataframe.describe())



print("Shape:", dataframe.shape)



print("Data Types:", dataframe.dtypes)
dataset = dataframe.values





X = dataset[:,0:15]

Y = dataset[:,15]

Y2 = dataset[:,16] 
#Feature Selection

model = ExtraTreesRegressor()

rfe = RFE(model, 3)

fit = rfe.fit(X, Y)



print("Number of Features: ", fit.n_features_)

print("Selected Features: ", fit.support_)

print("Feature Ranking: ", fit.ranking_) 
#Feature Selection

model = ExtraTreesRegressor()

rfe = RFE(model, 3)

fit = rfe.fit(X, Y2)



print("Number of Features: ", fit.n_features_)

print("Selected Features: ", fit.support_)

print("Feature Ranking: ", fit.ranking_) 
dataframe.hist()
# Split Data to Train and Test

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2)



num_instances = len(X)



models = []

models.append(('LiR', LinearRegression()))

models.append(('Ridge', Ridge()))

models.append(('Bag_Re', BaggingRegressor()))

models.append(('RandomForest', RandomForestRegressor()))

models.append(('ExtraTreesRegressor', ExtraTreesRegressor()))

models.append(('KNN', KNeighborsRegressor()))

models.append(('CART', DecisionTreeRegressor()))



# Evaluations

results = []

names = []

scoring = []



for name, model in models:

    # Fit the model

    model.fit(X_Train, Y_Train)

    

    predictions = model.predict(X_Test)

    

    # Evaluate the model

    score = explained_variance_score(Y_Test, predictions)

    mae = mean_absolute_error(predictions, Y_Test)

    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    results.append(mae)

    names.append(name)

    

    msg = "%s: %f (%f)" % (name, score, mae)

    print(msg)
# Split Data to Train and Test

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y2, test_size=0.2)



num_instances = len(X)



models = []

models.append(('LiR', LinearRegression()))

models.append(('Ridge', Ridge()))

models.append(('Bag_Re', BaggingRegressor()))

models.append(('RandomForest', RandomForestRegressor()))

models.append(('ExtraTreesRegressor', ExtraTreesRegressor()))

models.append(('KNN', KNeighborsRegressor()))

models.append(('CART', DecisionTreeRegressor()))



# Evaluations

results = []

names = []

scoring = []



for name, model in models:

    # Fit the model

    model.fit(X_Train, Y_Train)

    

    predictions = model.predict(X_Test)

    

    # Evaluate the model

    score = explained_variance_score(Y_Test, predictions)

    mae = mean_absolute_error(predictions, Y_Test)

    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    results.append(mae)

    names.append(name)

    

    msg = "%s: %f (%f)" % (name, score, mae)

    print(msg)
# Split Data to Train and Test

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3)









# create model

model = Sequential()

model.add(Dense(6, input_dim=15, init='uniform', activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(4, init='uniform', activation='relu', kernel_constraint=maxnorm(3)))

model.add(Dropout(0.2))

model.add(Dense(2, init='uniform', activation='relu'))

model.add(Dense(1, init='uniform', activation='relu'))



# Compile model

model.compile(loss='mean_absolute_error', optimizer='adam')



# Fit the model

model.fit(X_Train, Y_Train, epochs=100, batch_size=10)



# Evaluate the model

scores = model.evaluate(X_Test, Y_Test)

print("score: %.2f%%" % (100-scores))
# Split Data to Train and Test

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y2, test_size=0.3)









# create model

model = Sequential()

model.add(Dense(6, input_dim=15, init='uniform', activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(4, init='uniform', activation='relu', kernel_constraint=maxnorm(3)))

model.add(Dropout(0.2))

model.add(Dense(2, init='uniform', activation='relu'))

model.add(Dense(1, init='uniform', activation='relu'))



# Compile model

model.compile(loss='mean_absolute_error', optimizer='adam')



# Fit the model

model.fit(X_Train, Y_Train, epochs=100, batch_size=10)



# Evaluate the model

scores = model.evaluate(X_Test, Y_Test)

print("score: %.2f%%" % (100-scores))
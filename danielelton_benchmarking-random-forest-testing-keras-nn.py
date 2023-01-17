import pandas as pd

import numpy as np



train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
def scale_features(X): 

    ''' scale the features columnwise to 0 mean and normalize by range

    '''

    for i in range(len(X[1,:])):

        X[:,i] = (X[:,i] - X[:,i].mean())/(X[:,i].max()-X[:,i].min())

    return X



def fill_nan(X):

    '''replace NaNs with mean for each column

    ''' 

    for i in range(len(X[1,:])):

        mean = np.nanmean(X[:,i])

        mask = np.isnan(X[:,i])

        X[mask,i] = mean

    return X





def strings_to_num(df):

    '''Input: df (Pandas dataframe)

       Ouput: numpy array with categorical (string) columns transformed into numerical

    '''

    #transform dataframe categories to numbers

    return df.apply(lambda x: pd.factorize(x)[0]).values



def df_to_numpy_array(df):

    ''' Seperate categorical and numerical columns of dataframe

        Input df: Pandas dataframe

        Output: numpy array

    '''

    #get names of numerical columns

    num_columns = list(df.select_dtypes(include=['float64', 'int64']).columns.values)

    

    #get numerical values into NumPy array

    num_values = df[num_columns].values

    

    #fill NaN in numerical features

    num_values = fill_nan(num_values)

    

    #scale numerical features

    num_values = scale_features(num_values)   

    

    #get categorical columns 

    cat_columns = list(df.select_dtypes(include=['object']).columns.values)



    #transform categorical columns into numpy array

    cat_values = strings_to_num(df[cat_columns])

    

    return np.concatenate((cat_values,num_values),axis=1)





feature_names = train_df.drop(["SalePrice","Id"],axis=1).columns.tolist()

num_features = len(feature_names)    

    

X = df_to_numpy_array(train_df.drop(["SalePrice","Id"],axis=1))     #training data

Xtest = df_to_numpy_array(test_df.drop(["Id"],axis=1)) #test data

y = train_df['SalePrice'].values #target

test_ids = test_df["Id"].values
from sklearn import linear_model

from sklearn import cross_validation



lm = linear_model.LinearRegression()

lm = lm.fit(X,y)



scores = cross_validation.cross_val_score(lm, X, y, cv=4, n_jobs=-1)

print("scores = ",scores)

print("mean   = %4.2f" % (scores.mean()))
%matplotlib inline 

from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt



rf = RandomForestRegressor(n_estimators=200)

rf = rf.fit( X, y )



scores = cross_validation.cross_val_score(rf, X, y, cv=3, n_jobs=-1)

print("accuracies     = ",scores)

print("mean accuracy = %4.2f" % (scores.mean()))
from sklearn.learning_curve import validation_curve



num_to_try = [5,10,25,50,75,100,200,400,800]

train_scores, valid_scores = validation_curve(RandomForestRegressor(), 

                                              X, y, 

                                              "n_estimators",

                                               num_to_try, n_jobs=-1)



f, ax = plt.subplots(figsize=(10,4))

ax.semilogx(num_to_try, np.mean(valid_scores, axis=1),"-*")

f.show()
f, ax = plt.subplots(figsize=(10,4))



bar_placements = range(len(rf.feature_importances_))

ax.bar(bar_placements, rf.feature_importances_)

ax.set_title("Feature Importances")

ax.set_xticks([tick + .5 for tick in bar_placements])

ax.set_xticklabels(feature_names)



f.show()
#create dictionary

f_imps = {}

for i in range(num_features):

    f_imps[feature_names[i]] = rf.feature_importances_[i]

    

#sort dictionary 

sorted_feature_names = sorted(f_imps, key=f_imps.__getitem__, reverse=True)

sorted_values = sorted(f_imps.values(), reverse=True)



num_to_print = 20

for i in range(num_to_print):

    print("%15s %4.3f" % (sorted_feature_names[i],sorted_values[i]))
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.cross_validation import cross_val_score



# define base mode

def model():

    # create model

    model = Sequential()

    model.add(Dense(60, input_dim=num_features, init='normal', activation='relu'))

    model.add(Dense(20, init='normal', activation='relu'))

    model.add(Dense(1, init='normal'))

    # Compile model

    model.compile(loss='msle', optimizer='adam')

    return model



# evaluate model with standardized dataset

kr = KerasRegressor(build_fn=model, nb_epoch=100, batch_size=5, verbose=0)



scores = cross_val_score(kr, X, y, cv=4)

print("msle = %4.2f std = %4.2f" % (scores.mean(),scores.std()))
kr = KerasRegressor(build_fn=model, nb_epoch=500, batch_size=5, verbose=1)

kr = kr.fit(X,y) 
predictions = kr.model.predict(Xtest)

 

to_save = np.concatenate((np.reshape(test_ids,(len(test_ids),1)), predictions),axis=1)

np.savetxt("predictions.csv",to_save,header="Id,SalePrice",delimiter=',',comments="",fmt=["%8i", "%8.6f"],)
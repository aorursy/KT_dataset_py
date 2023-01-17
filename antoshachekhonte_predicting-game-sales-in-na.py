import numpy as np 

import pandas as pd

from IPython.display import display

%matplotlib inline



try:

    data = pd.read_csv('../input/vgsales.csv')

    print ('Dataset loaded...')

except:

    print ('Unable to load dataset...') 
display(data[:10])   
data = data[np.isfinite(data['Year'])]
naSales = data['NA_Sales']

features = data.drop(['Name', 'Global_Sales', 'NA_Sales'], axis = 1)



# Displaying our features and target columns... 

display(naSales[:5])

display(features[:5])
# Firstly, I am dividing the features data set into two as follows. 



salesFeatures = features.drop(['Rank', 'Platform', 'Year', 'Genre', 'Publisher'], 

                              axis = 1)

otherFeatures = features.drop(['EU_Sales', 'JP_Sales', 'Other_Sales', 'Rank'], 

                              axis = 1)



# Secondly, I am obtaining the PCA transformed features...



from sklearn.decomposition import PCA

pca = PCA(n_components = 1)

pca.fit(salesFeatures)

salesFeaturesTransformed = pca.transform(salesFeatures)



# Finally, I am merging the new transfomed salesFeatures 

# (...cont) column back together with the otherFeatures columns...



salesFeaturesTransformed = pd.DataFrame(data = salesFeaturesTransformed, 

                                        index = salesFeatures.index, 

                                        columns = ['Sales'])

rebuiltFeatures = pd.concat([otherFeatures, salesFeaturesTransformed], 

                            axis = 1)



display(rebuiltFeatures[:5])
# This code is inspired by udacity project 'student intervention'.

temp = pd.DataFrame(index = rebuiltFeatures.index)



for col, col_data in rebuiltFeatures.iteritems():

    

    if col_data.dtype == object:

        col_data = pd.get_dummies(col_data, prefix = col)

        

    temp = temp.join(col_data)

    

rebuiltFeatures = temp

display(rebuiltFeatures[:5])
# Dividing the data into training and testing sets...

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(rebuiltFeatures, 

                                                    naSales, 

                                                    test_size = 0.2, 

                                                    random_state = 2)
# Creating & fitting a Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor



regDTR = DecisionTreeRegressor(random_state = 4)

regDTR.fit(X_train, y_train)

y_regDTR = regDTR.predict(X_test)



from sklearn.metrics import r2_score

print ('The following is the r2_score on the DTR model...')

print (r2_score(y_test, y_regDTR))



# Creating a K Neighbors Regressor

from sklearn.neighbors import KNeighborsRegressor



regKNR = KNeighborsRegressor()

regKNR.fit(X_train, y_train)

y_regKNR = regKNR.predict(X_test)



print ('The following is the r2_score on the KNR model...')

print (r2_score(y_test, y_regKNR))
# This code is inspired by udacity project 'student intervention'

from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV

from sklearn.cross_validation import ShuffleSplit

cv_sets = ShuffleSplit(X_train.shape[0], n_iter = 10, 

                       test_size = 0.2, random_state = 2)

regressor = DecisionTreeRegressor(random_state = 4)

params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 

          'splitter': ['best', 'random']}

scoring_func = make_scorer(r2_score)

    

grid = GridSearchCV(regressor, params, cv = cv_sets, 

                    scoring = scoring_func)

grid = grid.fit(X_train, y_train)



optimizedReg = grid.best_estimator_

y_optimizedPrediction = optimizedReg.predict(X_test)



print ('The r2_score of the optimal regressor is:')

print (r2_score(y_test, y_optimizedPrediction))
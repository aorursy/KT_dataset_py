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
# About the data.



# The study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. Each observation is a 30m x 30m patch. You are asked to predict an integer classification for the forest cover type. The seven types are:



# 1 - Spruce/Fir

# 2 - Lodgepole Pine

# 3 - Ponderosa Pine

# 4 - Cottonwood/Willow

# 5 - Aspen

# 6 - Douglas-fir

# 7 - Krummholz



# The training set (15120 observations) contains both features and the Cover_Type. The test set contains only the features. You must predict the Cover_Type for every row in the test set (565892 observations).

# Data Fields



# Elevation - Elevation in meters

# Aspect - Aspect in degrees azimuth

# Slope - Slope in degrees

# Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features

# Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features

# Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway

# Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice

# Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice

# Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice

# Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points

# Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation

# Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation

# Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation



# The wilderness areas are:



# 1 - Rawah Wilderness Area

# 2 - Neota Wilderness Area

# 3 - Comanche Peak Wilderness Area

# 4 - Cache la Poudre Wilderness Area



# The soil types are:



# 1 Cathedral family - Rock outcrop complex, extremely stony.

# 2 Vanet - Ratake families complex, very stony.

# 3 Haploborolis - Rock outcrop complex, rubbly.

# 4 Ratake family - Rock outcrop complex, rubbly.

# 5 Vanet family - Rock outcrop complex complex, rubbly.

# 6 Vanet - Wetmore families - Rock outcrop complex, stony.

# 7 Gothic family.

# 8 Supervisor - Limber families complex.

# 9 Troutville family, very stony.

# 10 Bullwark - Catamount families - Rock outcrop complex, rubbly.

# 11 Bullwark - Catamount families - Rock land complex, rubbly.

# 12 Legault family - Rock land complex, stony.

# 13 Catamount family - Rock land - Bullwark family complex, rubbly.

# 14 Pachic Argiborolis - Aquolis complex.

# 15 unspecified in the USFS Soil and ELU Survey.

# 16 Cryaquolis - Cryoborolis complex.

# 17 Gateview family - Cryaquolis complex.

# 18 Rogert family, very stony.

# 19 Typic Cryaquolis - Borohemists complex.

# 20 Typic Cryaquepts - Typic Cryaquolls complex.

# 21 Typic Cryaquolls - Leighcan family, till substratum complex.

# 22 Leighcan family, till substratum, extremely bouldery.

# 23 Leighcan family, till substratum - Typic Cryaquolls complex.

# 24 Leighcan family, extremely stony.

# 25 Leighcan family, warm, extremely stony.

# 26 Granile - Catamount families complex, very stony.

# 27 Leighcan family, warm - Rock outcrop complex, extremely stony.

# 28 Leighcan family - Rock outcrop complex, extremely stony.

# 29 Como - Legault families complex, extremely stony.

# 30 Como family - Rock land - Legault family complex, extremely stony.

# 31 Leighcan - Catamount families complex, extremely stony.

# 32 Catamount family - Rock outcrop - Leighcan family complex, extremely stony.

# 33 Leighcan - Catamount families - Rock outcrop complex, extremely stony.

# 34 Cryorthents - Rock land complex, extremely stony.

# 35 Cryumbrepts - Rock outcrop - Cryaquepts complex.

# 36 Bross family - Rock land - Cryumbrepts complex, extremely stony.

# 37 Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.

# 38 Leighcan - Moran families - Cryaquolls complex, extremely stony.

# 39 Moran family - Cryorthents - Leighcan family complex, extremely stony.

# 40 Moran family - Cryorthents - Rock land complex, extremely stony.
# Data Science workflow.



# 1. The problem.

# 2. The data.

# 3. Prepare data.

# 4. Perfom exploratory analysis.

# 5. Model Data.

# 6. Validate and implement data model.

# 7. Optimize and Strategize.
# 1. The problem.

# To predict an integer classification for the forest cover type.
# Import libraries for data analysis.



import pandas as pd

import seaborn as sns

import numpy as np

%matplotlib inline
# 2. The data.



training_data = pd.read_csv('/kaggle/input/learn-together/train.csv')

test_data = pd.read_csv('/kaggle/input/learn-together/test.csv')

submission = pd.read_csv('/kaggle/input/learn-together/sample_submission.csv')
training_data.describe()
training_data.shape
training_data.Cover_Type.value_counts()
# Checking for missing values.



missing_values = [x for x in training_data.columns if training_data[x].isnull().any()]

missing_values
# Data Types

training_data.dtypes
columns = training_data.columns
# 3. Prepare data.

from sklearn.preprocessing import StandardScaler
# Scaling data.



scaler = StandardScaler()



scaled_training_data = training_data.copy()



# We scale continuous variables

for col in columns[1:11]:

    scaled_data = scaler.fit_transform(training_data[col].values.reshape(-1,1))

    scaled_training_data[col] = scaled_data
# 4. Perform exploratory analysis. 



# Exploraotry analysis for numerical features.



for col in columns[1:11]:

    a = sns.FacetGrid(training_data, hue = 'Cover_Type', aspect=4 )

    a.map(sns.kdeplot, col , shade= True )

    a.set(xlim=(0 , training_data[col].max()))

    a.add_legend()
# Exploraotry analysis for categorical features.



for x in columns[11:-1]:

    a = sns.FacetGrid(training_data, hue = x, aspect=4)

    a.map(sns.countplot,'Cover_Type',order=training_data['Cover_Type'].value_counts().index)

    a.add_legend()
# Unimportant features. 



# Droping features with no predictive value.



features_to_drop = []



for x in columns[11:-1]:

    vals = training_data[x].value_counts().values

    percent = vals[0]/sum(vals)*100

    

    if 98 < percent or percent < 5 :

        

        #print("Feature {} uniformity {} percent".format(x,round(percent,2)))

        

        features_to_drop.append(x)

        

    else:

        pass



print(features_to_drop)



drop_training_data = training_data.copy()



drop_training_data = drop_training_data.drop(features_to_drop,axis=1)
# 5. Model Data.
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from eli5.sklearn import PermutationImportance

import eli5
# RandomForestClassifier



X_train, X_test, y_train, y_test = train_test_split(training_data[columns[1:-1]], training_data['Cover_Type'])

rf_clf = RandomForestClassifier(criterion="entropy", random_state=7)

rf_clf.fit(X_train, y_train)



rf_prediction = rf_clf.predict(X_test)

rf_score = sum([a==b for a,b in zip(y_test, rf_prediction)])/len(rf_prediction)

print("Test data score: {}".format(rf_score))



cv_score= cross_val_score(rf_clf,X_train,y_train,cv=5)

print("Cross validation score: {}".format(cv_score))

print("Average cross valiation score: {}".format(sum(cv_score)/len(cv_score)))



# Mispredictions

mispredictions = [[a,b] for a,b in zip(y_test, rf_prediction) if a!=b]

rf_miss = pd.DataFrame(mispredictions, columns=['value', 'prediction'])



for x in list(range(1,8)):

    miss_x = rf_miss[rf_miss['value'] == x]

    a = sns.FacetGrid(miss_x, aspect=4, palette="rainbow")

    a.map(sns.countplot,'prediction',order=miss_x['prediction'].value_counts().index)

    a.set(title="Cover type-{} miss predictions with other cover types".format(x))

    a.add_legend()
# Tuning max_depth hyperparameter.



def max_depth_tuner(data, depth):

    

    columns = data.columns

    depth_range = [x+1 for x in range(depth)]

    train_X, test_X, train_y, test_y = train_test_split(data[columns[1:-1]], data['Cover_Type'], random_state=0)

    

    scores_list = []

    

    for x in depth_range:

        

        model = RandomForestClassifier(criterion="entropy", max_depth=x, random_state=7)

        model.fit(train_X, train_y)

        cv_score = cross_val_score(model,train_X,train_y,cv=5)

        cv_score_avg = sum(cv_score)/len(cv_score)

        

        scores_list.append(cv_score_avg)

    

    g = sns.lineplot(x=depth_range, y=scores_list)

    g.set(xlabel ="Depth of the tree",ylabel="Score",title="Random Forest Classifier tuning of max_depth hyperparameter")

    



max_depth_tuner(training_data, 30)
# Tuning min_samples_split hyperparameter.



def min_samples_split_tuner(data):

    

    columns = data.columns

    min_samples_splits = np.linspace(2, 20, 10, dtype=int)

    train_X, test_X, train_y, test_y = train_test_split(data[columns[1:-1]], data['Cover_Type'], random_state=0)

    

    scores_list = []

    

    for x in min_samples_splits:

        

        model = RandomForestClassifier(criterion="entropy", min_samples_split=x, random_state=7)

        model.fit(train_X, train_y)

        cv_score = cross_val_score(model,train_X,train_y,cv=5)

        cv_score_avg = sum(cv_score)/len(cv_score)

        

        scores_list.append(cv_score_avg)

    

    g = sns.lineplot(x=min_samples_splits, y=scores_list)

    g.set(xlabel ="Min samples splits",ylabel="Score",title="Random Forest Classifier tuning of min_samples_split hyperparameter")

    



min_samples_split_tuner(training_data)
# Tuning min_samples_leaf hyperparameter.



def min_samples_leaf_tuner(data):

    

    columns = data.columns

    min_samples_leaf = np.linspace(2, 20, 10, dtype=int)

    train_X, test_X, train_y, test_y = train_test_split(data[columns[1:-1]], data['Cover_Type'], random_state=0)

    

    scores_list = []

    

    for x in min_samples_leaf:

        

        model = RandomForestClassifier(criterion="entropy", min_samples_leaf=x, random_state=7)

        model.fit(train_X, train_y)

        cv_score = cross_val_score(model,train_X,train_y,cv=5)

        cv_score_avg = sum(cv_score)/len(cv_score)

        

        scores_list.append(cv_score_avg)

    

    g = sns.lineplot(x=min_samples_leaf, y=scores_list)

    g.set(xlabel ="Min samples leaf",ylabel="Score",title="Random Forest Classifier tuning of min_samples_leaf hyperparameter")

    



min_samples_leaf_tuner(training_data)
# Tuning max_feature hyperparameter.



def max_feature_tuner(data):

    

    columns = data.columns

    max_features = list(range(1,52))

    

    train_X, test_X, train_y, test_y = train_test_split(data[columns[1:-1]], data['Cover_Type'], random_state=0)

    

    scores_list = []

    

    for x in max_features:

        

        model = RandomForestClassifier(criterion="entropy", max_features=x, random_state=7)

        model.fit(train_X, train_y)

        cv_score = cross_val_score(model,train_X,train_y,cv=5)

        cv_score_avg = sum(cv_score)/len(cv_score)

        

        scores_list.append(cv_score_avg)

    

    g = sns.lineplot(x=max_features, y=scores_list)

    g.set(xlabel ="Max features",ylabel="Score",title="Random Forest Classifier tuning of max_features hyperparameter")

    



max_feature_tuner(training_data)
# Tuning max_leaf_nodes hyperparameter.



def max_leaf_nodes_tuner(data):

    

    columns = data.columns

    max_leaf_nodes = np.linspace(10, 1000, 20, dtype=int)

    

    train_X, test_X, train_y, test_y = train_test_split(data[columns[1:-1]], data['Cover_Type'], random_state=0)

    

    scores_list = []

    

    for x in max_leaf_nodes:

        

        model = RandomForestClassifier(criterion="entropy", max_leaf_nodes=x, random_state=7)

        model.fit(train_X, train_y)

        cv_score = cross_val_score(model,train_X,train_y,cv=5)

        cv_score_avg = sum(cv_score)/len(cv_score)

        

        scores_list.append(cv_score_avg)

    

    g = sns.lineplot(x=max_leaf_nodes, y=scores_list)

    g.set(xlabel ="Max leaf nodes",ylabel="Score",title="Random Forest Classifier tuning of max_leaf_nodes hyperparameter")

    



max_leaf_nodes_tuner(training_data)
# Tuning n_estimators hyperparameter.



def n_estimators_tuner(data):

    

    columns = data.columns

    n_est = np.linspace(10, 500, 10, dtype=int)

    train_X, test_X, train_y, test_y = train_test_split(data[columns[1:-1]], data['Cover_Type'], random_state=0)

    

    scores_list = []

    

    for x in n_est:

        

        model = RandomForestClassifier(n_estimators=x, random_state=0)

        model.fit(train_X, train_y)

        cv_score = cross_val_score(model,train_X,train_y,cv=5)

        cv_score_avg = sum(cv_score)/len(cv_score)

        

        scores_list.append(cv_score_avg)

    

    g = sns.lineplot(x=n_est, y=scores_list)

    g.set(xlabel ="Number of estimators",ylabel="Score",title="Decision Tree Regressor tuning of n_estimators parameter")

    



n_estimators_tuner(training_data)
# 6. Validate and implement final model.



# Final model 



#RandomForestClassifier



X_train, X_test, y_train, y_test = train_test_split(training_data[columns[1:-1]], training_data['Cover_Type'])

rf_clf = RandomForestClassifier(criterion="entropy", n_estimators=250, bootstrap=False, max_depth=20, random_state=7)

rf_clf.fit(X_train, y_train)

rf_prediction = rf_clf.predict(X_test)

#print(dt_prediction)

rf_score = sum([a==b for a,b in zip(y_test, rf_prediction)])/len(rf_prediction)

print("Test data score: {}".format(rf_score))



cv_score= cross_val_score(rf_clf,X_train,y_train,cv=10)

print("Cross validation score: {}".format(cv_score))

print("Average cross valiation score: {}".format(sum(cv_score)/len(cv_score)))



# Mispredictions

mispredictions = [[a,b] for a,b in zip(y_test, rf_prediction) if a!=b]

rf_miss = pd.DataFrame(mispredictions, columns=['value', 'prediction'])



for x in list(range(1,8)):

    miss_x = rf_miss[rf_miss['value'] == x]

    a = sns.FacetGrid(miss_x, aspect=4, palette="rainbow")

    a.map(sns.countplot,'prediction',order=miss_x['prediction'].value_counts().index)

    a.set(title="Cover type-{} miss predictions with other cover types".format(x))

    a.add_legend()

# Looking at feature importance in model. 



perm = PermutationImportance(rf_clf, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
# The most important feature in this model. 



a = sns.FacetGrid(scaled_training_data, hue = 'Cover_Type', aspect=4 )

a.map(sns.kdeplot, 'Elevation' , shade= True )

a.set(ylim=(0 , scaled_training_data['Elevation'].max()),xlim=(-2,2))

a.add_legend()
# Final model with scaled data.



#Random Forest Classifier



X_train, X_test, y_train, y_test = train_test_split(scaled_training_data[columns[1:-1]], scaled_training_data['Cover_Type'])

rf_clf = RandomForestClassifier(criterion="entropy", n_estimators=250, bootstrap=False, max_depth=20, random_state=7)

rf_clf.fit(X_train, y_train)

rf_prediction = rf_clf.predict(X_test)

#print(dt_prediction)

rf_score = sum([a==b for a,b in zip(y_test, rf_prediction)])/len(rf_prediction)

print("Test data score: {}".format(rf_score))



cv_score= cross_val_score(rf_clf,X_train,y_train,cv=5)

print("Cross validation score: {}".format(cv_score))

print("Average cross valiation score: {}".format(sum(cv_score)/len(cv_score)))



# Mispredictions

mispredictions = [[a,b] for a,b in zip(y_test, rf_prediction) if a!=b]

rf_miss = pd.DataFrame(mispredictions, columns=['value', 'prediction'])



for x in list(range(1,8)):

    miss_x = rf_miss[rf_miss['value'] == x]

    a = sns.FacetGrid(miss_x, aspect=4, palette="rainbow")

    a.map(sns.countplot,'prediction',order=miss_x['prediction'].value_counts().index)

    a.set(title="Cover type-{} miss predictions with other cover types".format(x))

    a.add_legend()

    
# Final model with droped non informative features.



#Random Forest Classifier



left_features = drop_training_data.columns



X_train, X_test, y_train, y_test = train_test_split(drop_training_data[left_features[1:-1]], drop_training_data['Cover_Type'])

rf_clf = RandomForestClassifier(criterion="entropy", n_estimators=250, bootstrap=False, max_depth=20, random_state=7)

rf_clf.fit(X_train, y_train)

rf_prediction = rf_clf.predict(X_test)

#print(dt_prediction)

rf_score = sum([a==b for a,b in zip(y_test, rf_prediction)])/len(rf_prediction)

print("Test data score: {}".format(rf_score))



cv_score= cross_val_score(rf_clf,X_train,y_train,cv=5)

print("Cross validation score: {}".format(cv_score))

print("Average cross valiation score: {}".format(sum(cv_score)/len(cv_score)))



# Mispredictions

mispredictions = [[a,b] for a,b in zip(y_test, rf_prediction) if a!=b]

rf_miss = pd.DataFrame(mispredictions, columns=['value', 'prediction'])



for x in list(range(1,8)):

    miss_x = rf_miss[rf_miss['value'] == x]

    a = sns.FacetGrid(miss_x, aspect=4, palette="rainbow")

    a.map(sns.countplot,'prediction',order=miss_x['prediction'].value_counts().index)

    a.set(title="Cover type-{} miss predictions with other cover types".format(x))

    a.add_legend()
# 7. Optimize and Strategize.



# It seems that there are no difference if you scale data or drop non informative columns.

# Tell me what do you think. Which direction we should move in the future.

# You can contact me: jonas.salys@gmail.com



# Team members:

# Michiel

# Ashok

# Jielin

# Aakash

# Preparing test data. 



# Final model 



# Random Forest Classifier



test_columns = test_data.columns



test_data_final = test_data[test_columns[1:]]



X_train, y_train = training_data[columns[1:-1]], training_data['Cover_Type']

rf_clf_final_t = RandomForestClassifier(criterion="entropy", n_estimators=250, bootstrap=False, max_depth=20, random_state=7)

rf_clf_final_t.fit(X_train, y_train)



rf_prediction_final_t = rf_clf_final_t.predict(test_data_final)

print(rf_prediction_final_t)

test_data.Id.values
sub = submission.copy()
# Save test predictions to file

output = pd.DataFrame({'Id': test_data.Id.values,

                       'Cover_Type': rf_prediction_final_t})

output.to_csv('submission.csv', index=False)
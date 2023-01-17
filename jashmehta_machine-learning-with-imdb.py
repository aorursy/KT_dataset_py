# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline 

from sklearn.neighbors import KNeighborsRegressor

from sklearn.cross_validation import cross_val_score, ShuffleSplit, train_test_split

from sklearn.datasets import load_boston

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn import svm

from sklearn import linear_model

import pandas as pd

import numpy as np



%matplotlib inline

#import seaborn as sns

#sns.set_style("white")

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import warnings

warnings.filterwarnings("ignore")
import pandas as pd

datasource = pd.read_csv('../input/movie_metadata.csv', delimiter=',')



datasource = datasource.dropna(how='any')  #Removing all the rows with Null/NaN values. 

#This step is done because NaN values in features will give an error when we try to fit linear regression model.
datasource.head()

datasource.shape[0] #Actual Dataset from Kaggle had 5043 rows, after removing the NaN values there are 3756 rows in the dataframe.
datasource.shape[1] # 28 columns in the dataframe
Features=datasource[[2,3,4,5,7,8,12,13,18,22,24,26,27]] 
type(Features)
Features.head()
Label=datasource[[25]]
LabelMatrix = Label.as_matrix()

FeaturesMatrix = Features.as_matrix() #This Matrix will be used in finding best features using Random Forest
IMDB_Features_df = pd.DataFrame(Features, columns= ["num_critic_for_reviews","duration","director_facebook_likes",

                                                    "actor_3_facebook_likes","actor_1_facebook_likes","gross","num_voted_users",

                                                    "cast_total_facebook_likes","num_user_for_reviews","budget",

                                                     "actor_2_facebook_likes","aspect_ratio","movie_facebook_likes"])



IMDB_Label_df = pd.DataFrame(Label, columns= ["imdb_score"])
IMDB_df = IMDB_Features_df
IMDB_df = pd.concat((IMDB_Features_df, IMDB_Label_df), axis=1)
print ("Number of observations: {}\nNumber of features {}".\

    format(IMDB_df.shape[0], IMDB_df.shape[1]))
IMDB_df.head()
# baseline - what's the score of all the features? Answer: Not very good. we can do better.

scores = cross_val_score(linear_model.LinearRegression(), IMDB_Features_df, IMDB_Label_df, scoring='r2') # mean_squared_error

print("Linear Regression Accuracy all Features: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



#We are going to further select best features and then try to find out the accuracy again
IMDB_df.corr(method='pearson') # We are getting similar results for - pearson, kendall, spearman

# Top 3 Features:

#num_voted_users

#duration

#num_critic_for_reviews
# Return the 2 features with highest variance 

IMDB_best2 = SelectKBest(f_regression, k=2).fit_transform(IMDB_Features_df, IMDB_Label_df) 
IMDB_best2 

#Top 2 features 

# duration

# num_voted_users
names1=["num_critic_for_reviews","duration","director_facebook_likes","actor_3_facebook_likes","actor_1_facebook_likes","gross","num_voted_users",

              "cast_total_facebook_likes","num_user_for_reviews","budget","actor_2_facebook_likes",

          "aspect_ratio","movie_facebook_likes"]
# univariate feature selection using Random Forest Regressor

# this is a different approach, but confirms our best two features

rf = RandomForestRegressor(n_estimators=20, max_depth=4)

scores1 = []

for i in range(FeaturesMatrix.shape[1]):

     score = cross_val_score(rf, FeaturesMatrix[:, i:i+1], LabelMatrix, scoring="r2",  #'mean_squared_error' sklearn impl is negative.

                              cv=ShuffleSplit(n=len(FeaturesMatrix), n_iter=10, test_size=.1))

     scores1.append((round(np.mean(score), 10), names1[i]))

scores1_df = pd.DataFrame(scores1, columns = ['score', 'feature'])

scores1_df.sort_values(['score'], ascending=False)
from sklearn.cross_validation import train_test_split
X_train_validation, X_test, y_train_validation, y_test = train_test_split(IMDB_Features_df, 

                                                                          IMDB_Label_df, 

                                                                          test_size = 0.1, 

                                                                          random_state = 0)
X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, 

                                                                y_train_validation, 

                                                                test_size = 0.1, 

                                                                random_state = 0)

X_train_validation1, X_test1, y_train_validation1, y_test1 = train_test_split(IMDB_Features_df, 

                                                                          IMDB_Label_df, 

                                                                          test_size = 0.2, 

                                                                          random_state = 0)
X_train1, X_validation1, y_train1, y_validation1 = train_test_split(X_train_validation1, 

                                                                y_train_validation1, 

                                                                test_size = 0.1, 

                                                                random_state = 0)
X_train_validation2, X_test2, y_train_validation2, y_test2 = train_test_split(IMDB_Features_df, 

                                                                          IMDB_Label_df, 

                                                                          test_size = 0.4, 

                                                                          random_state = 0)


X_train2, X_validation2, y_train2, y_validation2 = train_test_split(X_train_validation2, 

                                                                y_train_validation2, 

                                                                test_size = 0.15, 

                                                                random_state = 0)
X_train_validation3, X_test3, y_train_validation3, y_test3 = train_test_split(IMDB_Features_df, 

                                                                          IMDB_Label_df, 

                                                                          test_size = 0.6, 

                                                                          random_state = 0)


X_train3, X_validation3, y_train3, y_validation3 = train_test_split(X_train_validation3, 

                                                                y_train_validation3, 

                                                                test_size = 0.2, 

                                                                random_state = 0)
X_train_validation4, X_test4, y_train_validation4, y_test4 = train_test_split(IMDB_Features_df, 

                                                                          IMDB_Label_df, 

                                                                          test_size = 0.8, 

                                                                          random_state = 0)


X_train4, X_validation4, y_train4, y_validation4 = train_test_split(X_train_validation4, 

                                                                y_train_validation4, 

                                                                test_size = 0.2, 

                                                                random_state = 0)
FS1 = pd.DataFrame(IMDB_Features_df["num_voted_users"])
columns= ['num_voted_users','duration']

FS2 = pd.DataFrame(IMDB_Features_df, columns=columns)



#columns= ['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']

columns= ["num_voted_users", "movie_facebook_likes", "duration", "director_facebook_likes", "num_user_for_reviews", "num_critic_for_reviews"]

FS3 = pd.DataFrame(IMDB_Features_df, columns=columns)
# Let's assess the accuracy of some other models.

from sklearn.neighbors import KNeighborsRegressor

from sklearn import tree

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

nn3_model = KNeighborsRegressor(n_neighbors=3)

nn5_model = KNeighborsRegressor(n_neighbors=5)

nn7_model = KNeighborsRegressor(n_neighbors=7)

nn9_model = KNeighborsRegressor(n_neighbors=9)

regr = linear_model.LinearRegression()

decision_tree = tree.DecisionTreeRegressor()

poly2 = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', LinearRegression(fit_intercept=False))])

poly3 = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression(fit_intercept=False))])

poly4 = Pipeline([('poly', PolynomialFeatures(degree=4)),('linear', LinearRegression(fit_intercept=False))])

poly5 = Pipeline([('poly', PolynomialFeatures(degree=5)),('linear', LinearRegression(fit_intercept=False))])

rf = RandomForestRegressor()

models = [ {'name' : '3-Nearest Neighbors', 'estimator' : nn3_model},

          {'name' : '5-Nearest Neighbors', 'estimator' : nn5_model},

          {'name' : '7-Nearest Neighbors', 'estimator' : nn7_model},

          {'name' : '9-Nearest Neighbors', 'estimator' : nn9_model},

          {'name' : 'Linear Regression', 'estimator' : regr},

          {'name' : 'Decision Tree', 'estimator' : decision_tree},

          {'name' : 'Random Forest', 'estimator' : rf}

         ]

for model in models:

    scores = cross_val_score(model['estimator'], FS1, IMDB_Label_df, cv=ShuffleSplit(n=len(FS1), n_iter=10, test_size=.1), scoring='r2' )

    print(" %s Accuracy: %0.2f (+/- %0.2f)" % (model['name'], scores.mean(), scores.std() * 2))
# Let's assess the accuracy of some other models.

from sklearn.neighbors import KNeighborsRegressor

from sklearn import tree

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

nn3_model = KNeighborsRegressor(n_neighbors=3)

nn5_model = KNeighborsRegressor(n_neighbors=5)

nn7_model = KNeighborsRegressor(n_neighbors=7)

nn9_model = KNeighborsRegressor(n_neighbors=9)

regr = linear_model.LinearRegression()

decision_tree = tree.DecisionTreeRegressor()

poly2 = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', LinearRegression(fit_intercept=False))])

poly3 = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression(fit_intercept=False))])

poly4 = Pipeline([('poly', PolynomialFeatures(degree=4)),('linear', LinearRegression(fit_intercept=False))])

poly5 = Pipeline([('poly', PolynomialFeatures(degree=5)),('linear', LinearRegression(fit_intercept=False))])

rf = RandomForestRegressor()

models = [ {'name' : '3-Nearest Neighbors', 'estimator' : nn3_model},

          {'name' : '5-Nearest Neighbors', 'estimator' : nn5_model},

          {'name' : '7-Nearest Neighbors', 'estimator' : nn7_model},

          {'name' : '9-Nearest Neighbors', 'estimator' : nn9_model},

          {'name' : 'Linear Regression', 'estimator' : regr},

          {'name' : 'Decision Tree', 'estimator' : decision_tree},

          {'name' : 'Random Forest', 'estimator' : rf}

         ]

for model in models:

    scores = cross_val_score(model['estimator'], FS2, IMDB_Label_df, cv=ShuffleSplit(n=len(FS2), n_iter=10, test_size=.1), scoring='r2' )

    print(" %s Accuracy: %0.2f (+/- %0.2f)" % (model['name'], scores.mean(), scores.std() * 2))
# Let's assess the accuracy of some other models.

from sklearn.neighbors import KNeighborsRegressor

from sklearn import tree

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

nn3_model = KNeighborsRegressor(n_neighbors=3)

nn5_model = KNeighborsRegressor(n_neighbors=5)

nn7_model = KNeighborsRegressor(n_neighbors=7)

nn9_model = KNeighborsRegressor(n_neighbors=9)

regr = linear_model.LinearRegression()

decision_tree = tree.DecisionTreeRegressor()

poly2 = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', LinearRegression(fit_intercept=False))])

poly3 = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression(fit_intercept=False))])

poly4 = Pipeline([('poly', PolynomialFeatures(degree=4)),('linear', LinearRegression(fit_intercept=False))])

poly5 = Pipeline([('poly', PolynomialFeatures(degree=5)),('linear', LinearRegression(fit_intercept=False))])

rf = RandomForestRegressor()

models = [ {'name' : '3-Nearest Neighbors', 'estimator' : nn3_model},

          {'name' : '5-Nearest Neighbors', 'estimator' : nn5_model},

          {'name' : '7-Nearest Neighbors', 'estimator' : nn7_model},

          {'name' : '9-Nearest Neighbors', 'estimator' : nn9_model},

          {'name' : 'Linear Regression', 'estimator' : regr},

          {'name' : 'Decision Tree', 'estimator' : decision_tree},

          {'name' : 'Random Forest', 'estimator' : rf}

         ]

for model in models:

    scores = cross_val_score(model['estimator'], FS3, IMDB_Label_df, cv=ShuffleSplit(n=len(FS3), n_iter=10, test_size=.1), scoring='r2' )

    print(" %s Accuracy: %0.2f (+/- %0.2f)" % (model['name'], scores.mean(), scores.std() * 2))
from sklearn.linear_model import LinearRegression
model1 = LinearRegression().fit(X_train[['num_voted_users']], y_train)

model2 = LinearRegression().fit(X_train[['num_voted_users', 'duration']], y_train)

model3 = LinearRegression().fit(X_train[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']], y_train)







model4 = LinearRegression().fit(X_train1[['num_voted_users']], y_train1)

model5 = LinearRegression().fit(X_train1[['num_voted_users', 'duration']], y_train1)

model6 = LinearRegression().fit(X_train1[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']], y_train1)







model7 = LinearRegression().fit(X_train2[['num_voted_users']], y_train2)

model8 = LinearRegression().fit(X_train2[['num_voted_users', 'duration']], y_train2)

model9 = LinearRegression().fit(X_train2[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']], y_train2)





model10 = LinearRegression().fit(X_train3[['num_voted_users']], y_train3)

model11 = LinearRegression().fit(X_train3[['num_voted_users', 'duration']], y_train3)

model12 = LinearRegression().fit(X_train3[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']], y_train3)



model13 = LinearRegression().fit(X_train4[['num_voted_users']], y_train4)

model14 = LinearRegression().fit(X_train4[['num_voted_users', 'duration']], y_train4)

model15 = LinearRegression().fit(X_train4[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']], y_train4)

# Let's compute the training MSE

model1_train_err = ((model1.predict(X_train[['num_voted_users']]) - y_train)**2).mean()[0]

model2_train_err = ((model2.predict(X_train[['num_voted_users', 'duration']]) - y_train)**2).mean()[0]

model3_train_err = ((model3.predict(X_train[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']]) - y_train)**2).mean()[0]



model4_train_err = ((model4.predict(X_train1[['num_voted_users']]) - y_train1)**2).mean()[0]

model5_train_err = ((model5.predict(X_train1[['num_voted_users', 'duration']]) - y_train1)**2).mean()[0]

model6_train_err = ((model6.predict(X_train1[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']]) - y_train1)**2).mean()[0]



model7_train_err = ((model7.predict(X_train2[['num_voted_users']]) - y_train2)**2).mean()[0]

model8_train_err = ((model8.predict(X_train2[['num_voted_users', 'duration']]) - y_train2)**2).mean()[0]

model9_train_err = ((model9.predict(X_train2[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']]) - y_train2)**2).mean()[0]



model10_train_err = ((model10.predict(X_train3[['num_voted_users']]) - y_train3)**2).mean()[0]

model11_train_err = ((model11.predict(X_train3[['num_voted_users', 'duration']]) - y_train3)**2).mean()[0]

model12_train_err = ((model12.predict(X_train3[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']]) - y_train3)**2).mean()[0]





model13_train_err = ((model10.predict(X_train4[['num_voted_users']]) - y_train4)**2).mean()[0]

model14_train_err = ((model11.predict(X_train4[['num_voted_users', 'duration']]) - y_train4)**2).mean()[0]

model15_train_err = ((model12.predict(X_train4[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']]) - y_train4)**2).mean()[0]

[model1_train_err, model2_train_err, model3_train_err] # Models Trainded with 80% Training size
[model4_train_err, model5_train_err, model6_train_err] #Models Trainded with 70% Training size
[model7_train_err, model8_train_err, model9_train_err] #Models Trainded with 50% Training size
[model10_train_err, model11_train_err, model12_train_err] #Models Trainded with 30% Training size
[model13_train_err, model14_train_err, model15_train_err] #Models Trainded with 15% Training size
model1_test_err = ((model1.predict(X_test[['num_voted_users']]) - y_test)**2).mean()[0]

model2_test_err = ((model2.predict(X_test[['num_voted_users','duration']]) - y_test)**2).mean()[0]

model3_test_err = ((model3.predict(X_test[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']]) - y_test)**2).mean()[0]





model4_test_err = ((model4.predict(X_test1[['num_voted_users']]) - y_test1)**2).mean()[0]

model5_test_err = ((model5.predict(X_test1[['num_voted_users','duration']]) - y_test1)**2).mean()[0]

model6_test_err = ((model6.predict(X_test1[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']]) - y_test1)**2).mean()[0]



model7_test_err = ((model7.predict(X_test2[['num_voted_users']]) - y_test2)**2).mean()[0]

model8_test_err = ((model8.predict(X_test2[['num_voted_users','duration']]) - y_test2)**2).mean()[0]

model9_test_err = ((model9.predict(X_test2[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']]) - y_test2)**2).mean()[0]



model10_test_err = ((model10.predict(X_test3[['num_voted_users']]) - y_test3)**2).mean()[0]

model11_test_err = ((model11.predict(X_test3[['num_voted_users','duration']]) - y_test3)**2).mean()[0]

model12_test_err = ((model12.predict(X_test3[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']]) - y_test3)**2).mean()[0]



model13_test_err = ((model13.predict(X_test4[['num_voted_users']]) - y_test4)**2).mean()[0]

model14_test_err = ((model14.predict(X_test4[['num_voted_users','duration']]) - y_test4)**2).mean()[0]

model15_test_err = ((model15.predict(X_test4[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']]) - y_test4)**2).mean()[0]
[model1_test_err,model2_test_err, model3_test_err]
[model4_test_err,model5_test_err, model6_test_err]
[model7_test_err,model8_test_err, model9_test_err]
[model10_test_err,model11_test_err, model12_test_err]
[model13_test_err,model14_test_err, model15_test_err]
model9_test_err - model9_train_err
model12_test_err - model12_train_err
performance_train = [model3_train_err,model6_train_err, model9_train_err, model12_train_err,model15_train_err]
performance_test = [ model3_test_err,  model6_test_err, model9_test_err, model12_test_err, model15_test_err]
l_set = [80,70,50,30,15]
plt.plot(l_set,performance_test, 'b', label='train (Test)')

plt.plot(l_set,performance_train, 'g', label='train (Train)')



#plt.legend();

plt.xlabel('Training size');

plt.ylabel('MSE');

numpyMatrix = y_train.as_matrix()



numpyMatrix1 = y_train1.as_matrix()



numpyMatrix2 = y_train2.as_matrix()



numpyMatrix3 = y_train3.as_matrix()



numpyMatrix4 = y_train4.as_matrix()

model_rf1 = RandomForestRegressor().fit(X_train[["num_voted_users"]],y_train)

model_rf2 = RandomForestRegressor().fit(X_train[["num_voted_users","duration"]],y_train)

model_rf3 = RandomForestRegressor().fit(X_train[["num_voted_users","movie_facebook_likes","duration","num_critic_for_reviews","num_user_for_reviews","director_facebook_likes"]],y_train)





model_rf4 = RandomForestRegressor().fit(X_train1[["num_voted_users"]],y_train1)

model_rf5 = RandomForestRegressor().fit(X_train1[["num_voted_users","duration"]],y_train1)

model_rf6 = RandomForestRegressor().fit(X_train1[["num_voted_users","movie_facebook_likes","duration","num_critic_for_reviews","num_user_for_reviews","director_facebook_likes"]],y_train1)



model_rf7 = RandomForestRegressor().fit(X_train2[["num_voted_users"]],y_train2)

model_rf8 = RandomForestRegressor().fit(X_train2[["num_voted_users","duration"]],y_train2)

model_rf9 = RandomForestRegressor().fit(X_train2[["num_voted_users","movie_facebook_likes","duration","num_critic_for_reviews","num_user_for_reviews","director_facebook_likes"]],y_train2)



model_rf10 = RandomForestRegressor().fit(X_train3[["num_voted_users"]],y_train3)

model_rf11 = RandomForestRegressor().fit(X_train3[["num_voted_users","duration"]],y_train3)

model_rf12 = RandomForestRegressor().fit(X_train3[["num_voted_users","movie_facebook_likes","duration","num_critic_for_reviews","num_user_for_reviews","director_facebook_likes"]],y_train3)



model_rf13 = RandomForestRegressor().fit(X_train4[["num_voted_users"]],y_train4)

model_rf14 = RandomForestRegressor().fit(X_train4[["num_voted_users","duration"]],y_train4)

model_rf15 = RandomForestRegressor().fit(X_train4[["num_voted_users","movie_facebook_likes","duration","num_critic_for_reviews","num_user_for_reviews","director_facebook_likes"]],y_train4)







temp1 = model_rf1.predict(X_train[["num_voted_users"]])

temp2 = model_rf2.predict(X_train[["num_voted_users","duration"]])

temp3 = model_rf3.predict(X_train[["num_voted_users","movie_facebook_likes","duration","num_critic_for_reviews","num_user_for_reviews","director_facebook_likes"]])



temp4 = model_rf4.predict(X_train[["num_voted_users"]])

temp5 = model_rf5.predict(X_train[["num_voted_users","duration"]])

temp6 = model_rf6.predict(X_train[["num_voted_users","movie_facebook_likes","duration","num_critic_for_reviews","num_user_for_reviews","director_facebook_likes"]])



temp7 = model_rf7.predict(X_train[["num_voted_users"]])

temp8 = model_rf8.predict(X_train[["num_voted_users","duration"]])

temp9 = model_rf9.predict(X_train[["num_voted_users","movie_facebook_likes","duration","num_critic_for_reviews","num_user_for_reviews","director_facebook_likes"]])



temp10 = model_rf10.predict(X_train[["num_voted_users"]])

temp11 = model_rf11.predict(X_train[["num_voted_users","duration"]])

temp12 = model_rf12.predict(X_train[["num_voted_users","movie_facebook_likes","duration","num_critic_for_reviews","num_user_for_reviews","director_facebook_likes"]])



temp13 = model_rf13.predict(X_train[["num_voted_users"]])

temp14 = model_rf14.predict(X_train[["num_voted_users","duration"]])

temp15 = model_rf15.predict(X_train[["num_voted_users","movie_facebook_likes","duration","num_critic_for_reviews","num_user_for_reviews","director_facebook_likes"]])







model_rf1_train_err = ((temp1 - numpyMatrix)**2).mean()

model_rf2_train_err = ((temp2 - numpyMatrix)**2).mean()

model_rf3_train_err = ((temp3 - numpyMatrix)**2).mean()



model_rf4_train_err = ((temp4 - numpyMatrix1)**2).mean()

model_rf5_train_err = ((temp5 - numpyMatrix1)**2).mean()

model_rf6_train_err = ((temp6 - numpyMatrix1)**2).mean()



model_rf7_train_err = ((temp7 - numpyMatrix2)**2).mean()

model_rf8_train_err = ((temp8 - numpyMatrix2)**2).mean()

model_rf9_train_err = ((temp9 - numpyMatrix2)**2).mean()



model_rf10_train_err = ((temp10 - numpyMatrix3)**2).mean()

model_rf11_train_err = ((temp11 - numpyMatrix3)**2).mean()

model_rf12_train_err = ((temp12 - numpyMatrix3)**2).mean()



model_rf13_train_err = ((temp13 - numpyMatrix4)**2).mean()

model_rf14_train_err = ((temp14 - numpyMatrix4)**2).mean()

model_rf15_train_err = ((temp15 - numpyMatrix4)**2).mean()
[model_rf1_train_err, model_rf2_train_err, model_rf3_train_err]
[model_rf4_train_err, model_rf5_train_err, model_rf6_train_err]
[model_rf7_train_err, model_rf8_train_err, model_rf9_train_err]
[model_rf10_train_err, model_rf11_train_err, model_rf12_train_err]
[model_rf13_train_err, model_rf14_train_err, model_rf15_train_err]
Matrix = y_test.as_matrix()

Matrix1 = y_test1.as_matrix()

Matrix2 = y_test2.as_matrix()

Matrix3 = y_test3.as_matrix()

Matrix4 = y_test4.as_matrix()

temp_test1 = model_rf1.predict(X_test[["num_voted_users"]])

temp_test2 = model_rf2.predict(X_test[["num_voted_users","duration"]])

temp_test3 = model_rf3.predict(X_test[["num_voted_users","movie_facebook_likes","duration","num_critic_for_reviews","num_user_for_reviews","director_facebook_likes"]])



temp_test4 = model_rf1.predict(X_test1[["num_voted_users"]])

temp_test5 = model_rf2.predict(X_test1[["num_voted_users","duration"]])

temp_test6 = model_rf3.predict(X_test1[["num_voted_users","movie_facebook_likes","duration","num_critic_for_reviews","num_user_for_reviews","director_facebook_likes"]])



temp_test7 = model_rf1.predict(X_test2[["num_voted_users"]])

temp_test8 = model_rf2.predict(X_test2[["num_voted_users","duration"]])

temp_test9 = model_rf3.predict(X_test2[["num_voted_users","movie_facebook_likes","duration","num_critic_for_reviews","num_user_for_reviews","director_facebook_likes"]])



temp_test10 = model_rf1.predict(X_test3[["num_voted_users"]])

temp_test11 = model_rf2.predict(X_test3[["num_voted_users","duration"]])

temp_test12 = model_rf3.predict(X_test3[["num_voted_users","movie_facebook_likes","duration","num_critic_for_reviews","num_user_for_reviews","director_facebook_likes"]])



temp_test13 = model_rf1.predict(X_test[["num_voted_users"]])

temp_test14 = model_rf2.predict(X_test[["num_voted_users","duration"]])

temp_test15 = model_rf3.predict(X_test[["num_voted_users","movie_facebook_likes","duration","num_critic_for_reviews","num_user_for_reviews","director_facebook_likes"]])

model_rf1_test_err = ((temp_test1 - Matrix)**2).mean()

model_rf2_test_err = ((temp_test2 - Matrix)**2).mean()

model_rf3_test_err = ((temp_test3 - Matrix)**2).mean()



model_rf4_test_err = ((temp_test4 - Matrix1)**2).mean()

model_rf5_test_err = ((temp_test5 - Matrix1)**2).mean()

model_rf6_test_err = ((temp_test6 - Matrix1)**2).mean()



model_rf7_test_err = ((temp_test7 - Matrix2)**2).mean()

model_rf8_test_err = ((temp_test8 - Matrix2)**2).mean()

model_rf9_test_err = ((temp_test9 - Matrix2)**2).mean()



model_rf10_test_err = ((temp_test10 - Matrix3)**2).mean()

model_rf11_test_err = ((temp_test11 - Matrix3)**2).mean()

model_rf12_test_err = ((temp_test12 - Matrix3)**2).mean()



model_rf13_test_err = ((temp_test13 - Matrix4)**2).mean()

model_rf14_test_err = ((temp_test14 - Matrix4)**2).mean()

model_rf15_test_err = ((temp_test15- Matrix4)**2).mean()
[model_rf1_test_err, model_rf2_test_err, model_rf3_test_err]
[model_rf4_test_err, model_rf5_test_err, model_rf6_test_err]
[model_rf7_test_err, model_rf8_test_err, model_rf9_test_err]
[model_rf10_test_err, model_rf11_test_err, model_rf12_test_err]
[model_rf13_test_err, model_rf14_test_err, model_rf15_test_err]
performance_train_rf = [model_rf3_train_err, model_rf6_train_err, model_rf9_train_err, model_rf12_train_err,model_rf15_train_err]
performance_test_rf = [ model_rf3_test_err,  model_rf6_test_err, model_rf9_test_err, model_rf12_test_err, model_rf15_test_err]
plt.plot(l_set,performance_test_rf, 'b', label='train (Test)')

plt.plot(l_set,performance_train_rf, 'g', label='train (Train)')



#plt.legend();

plt.xlabel('Training size');

plt.ylabel('MSE');
LR_set= X_validation3[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']]
P = y_validation3.as_matrix()

for i in range(0,301):

    XYZ = np.array(LR_set.iloc[i])

    lr_dp = model12.predict(XYZ)

    print("Linear Regression Prediction: ",lr_dp, "Actual: ", P[i:(i+1)])
model12_validation_err = ((model12.predict(X_validation3[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']]) - y_validation3)**2).mean()[0]

model12_validation_err
RF_set= X_validation2[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']]
P = y_validation2.as_matrix()

for i in range(0,338):

    XYZ = np.array(RF_set.iloc[i])

    rf_dp = model_rf9.predict(XYZ)

    print(" RFR Prediction: ",rf_dp, "Actual: ", P[i:(i+1)])


temp_rf_val = model_rf9.predict(X_validation2[['num_voted_users','movie_facebook_likes','duration','num_critic_for_reviews','num_user_for_reviews','director_facebook_likes']])

MatVal = y_validation2.as_matrix()

model_rf9_validation_err = ((temp_rf_val - MatVal)**2).mean()

model_rf9_validation_err
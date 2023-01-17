import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
## creating the datata for plynolimal regression
np.random.seed(0)

n = 15

x = np.linspace(0,10,n) + np.random.randn(n)/5

y = np.sin(x)+x/6 + np.random.randn(n)/10
x
y
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
def part1_scatter():

    import matplotlib.pyplot as plt

    %matplotlib notebook

    plt.figure()

    plt.scatter(X_train, y_train, label='training data')

    plt.scatter(X_test, y_test, label='test data')

    plt.legend(loc=4);

   
part1_scatter()
def part1_plot():

    import matplotlib.pyplot as plt

    %matplotlib notebook

    plt.figure()

    plt.plot(x,y)

    plt.legend(loc=4);
part1_plot()
## create a polynomial regression
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
X_train.reshape(11,1)

## you have to convert this array into 2D array because polynomial feture wants the data that way
print(X_test)

## 4 elements to amke 2d array we ned to do 4,1 reshape

print (X_test.reshape(4,1))


## we are iterating this for creating the poly nolimal function for different degree

## like first degree polynomial third degree polynoimial etc



data_we_wil_try_predict = np.linspace(0,10,100)

#print(data_we_wil_try_predict)

## but to fit it we need to reshape just like we reshape the train data

data_we_wil_try_predict = data_we_wil_try_predict.reshape(100,1)

res = np.zeros((4, 100))



pr=[]

for i,degree in enumerate([1,3,6,9]):

    pol = PolynomialFeatures(degree)

    #print (pol)

    X_poly = pol.fit_transform(X_train.reshape(11,1))

    ## you have to reshape this just like we did in deep learnng in nural network

    #print (X_poly)

    ##now for every degree we nee to predict

    ## the value and store in a array

    linreg = LinearRegression().fit(X_poly, y_train)

    #print(linreg)

    test = pol.fit_transform(data_we_wil_try_predict)

    y = linreg.predict(test)

    print(y.shape)

    pr.append(y)

pr = np.array(pr)

pr.shape
np.array(pr).shape
def plot_one(degree_predictions):

    import matplotlib.pyplot as plt

    %matplotlib notebook

    plt.figure(figsize=(10,5))

    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)

    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)

    for i,degree in enumerate([1,3,6,9]):

        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))

    plt.ylim(-1,2.5)

    plt.legend(loc=4)
plot_one(pr)
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics.regression import r2_score



def two():

    r2_train = []

    r2_test = []



    for i in range(10):

        pol = PolynomialFeatures(degree=i)



        X_poly = pol.fit_transform(X_train.reshape(11,1))

        linreg = LinearRegression().fit(X_poly, y_train)        

        r2_train.append(linreg.score(X_poly, y_train))



        X_test_poly = pol.fit_transform(X_test.reshape(4,1))

        r2_test.append(linreg.score(X_test_poly, y_test))

    print(np.array(r2_train).shape)

    print(np.array(r2_test).shape)



    return (np.array(r2_train), np.array(r2_test))
two()
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Lasso, LinearRegression

from sklearn.metrics.regression import r2_score



poly = PolynomialFeatures(degree=12)



X_train_poly = poly.fit_transform(X_train.reshape(11,1))

X_test_poly = poly.fit_transform(X_test.reshape(4,1))



lin = LinearRegression()

lin_fit = lin.fit(X_train_poly, y_train)

lin_test = lin_fit.score(X_test_poly, y_test)

print(lin_test)
las = Lasso(alpha=0.01, max_iter = 10000)

las_fit = las.fit(X_train_poly, y_train)

las_test = las_fit.score(X_test_poly, y_test)
las_test
!wget https://raw.githubusercontent.com/msivalenka/Mushroom-Dataset/master/mushrooms.csv
import pandas as pd
df = pd.read_csv('mushrooms.csv')
df.head()
df2 = pd.get_dummies(df)
df2
X_df2 = df2.drop('class_p',axis=1)
y_df2 = df2[['class_p']]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_df2, y_df2, random_state=0)
X_subset = X_test2

y_subset = y_test2
from sklearn.tree import DecisionTreeClassifier



dtc = DecisionTreeClassifier(random_state=0)

tree = dtc.fit(X_train2, y_train2)

f_names = []



for i, importance in enumerate(tree.feature_importances_):

    f_names.append([importance, X_train2.columns[i]])



f_names.sort(reverse=True)

f_names = np.array(f_names)

#f_names = f_names[:5,1]

f_names.tolist()
from sklearn.svm import SVC

from sklearn.model_selection import validation_curve



svc = SVC(kernel='rbf', C=1, random_state=0)

gamma = np.logspace(-4,1,6)

train_scores, test_scores = validation_curve(

                        svc, X_subset, y_subset,

                        param_name='gamma',

                        param_range=gamma,

                        scoring='accuracy'

                        )



#return (train_scores.mean(axis=1), test_scores.mean(axis=1))
train_scores.mean(axis=1)
test_scores.mean(axis=1)
import matplotlib.pyplot as plt
plt.plot(train_scores.mean(axis=1),test_scores.mean(axis=1))
def answer_seven():

        

    return (0.0001, 10.0, 0.1)
import pandas as pd

import numpy  as np

import matplotlib.pyplot as plt

from termcolor import colored

import warnings

warnings.filterwarnings('ignore')

warnings.simplefilter(action='ignore', category=FutureWarning)



# Make the dataset:

# Create a noisy parabola

x_parabola = 50 * np.random.default_rng(100).random((50,))

y_parabola = ((x_parabola-15)**2) + (np.random.default_rng(30).random((50,))-0.5)*100

# add a few outliers

x_outliers = 50 * np.random.default_rng(80).random((10,))

y_outliers = ((x_outliers-15)**2) + (np.random.default_rng(500).random((10,))-0.5)*750

# concatenate the two groups of points together

x_points   = np.concatenate((x_parabola, x_outliers), axis=0)

y_points   = np.concatenate((y_parabola, y_outliers), axis=0)



# Now the plots and fits

###########################################################

# The 'underfitting' plot

###########################################################

fig = plt.figure(figsize=(7, 8))

ax = fig.add_subplot(3, 1, 1)

fig_1 = plt.scatter(x=x_points, y=y_points)

plt.axis('off')

x = np.linspace(0,50,400)

#ax.set_title ("underfitting", fontsize=18)

ax.text(10,450,'underfitting', fontsize=18)

fit = (np.polyfit(x_points,y_points, 1 ))

m = fit[0]

c = fit[1]

underfit = (m*x + c)

fig_1 = plt.plot(x, underfit,color='orange',linewidth=3 )

ax.set(xlim=(0, 45), ylim=(-200, 850))



###########################################################

# The 'good' plot

###########################################################

ax = fig.add_subplot(3, 1, 2)

#ax.set_title ("a good fit", fontsize=18)

ax.text(15,350,'a good fit', fontsize=18)

fit = (np.polyfit(x_points,y_points, 2 ))

a = fit[0]

c = fit[2]

m = fit[1]

goodfit = (a*x**2 + m*x + c)

fig_1 = plt.plot(x,goodfit,color='orange',linewidth=3)

fig_1 = plt.scatter(x=x_points, y=y_points)

ax.set(xlim=(0, 45), ylim=(-200, 850))

plt.axis('off')



###########################################################

# The 'overfitting' plot

###########################################################

ax = fig.add_subplot(3, 1, 3)

# overfit

#ax.set_title ("overfitting", fontsize=18)

ax.text(15,350,'overfitting', fontsize=18)

fit = (np.polyfit(x_points,y_points, 50 ))

overfit = np.poly1d(fit)

fig_1 = plt.plot(x,overfit(x),color='orange',linewidth=3)

fig_1 = plt.scatter(x=x_points, y=y_points)

ax.set(xlim=(0, 45), ylim=(-200, 850))

plt.axis('off')

plt.show()
# Make the dataset:

n_points = 75

mu, sigma = 0, 0.4

# The 'zeros' class (in blue)

zeros = np.zeros((n_points), dtype=int)

np.random.seed(1)

x_zeros = 0.7 + (np.random.normal(mu, sigma, n_points))*0.6

np.random.seed(220)

y_zeros = 0.3 + (np.random.normal(mu, sigma, n_points))*0.6

# The 'ones' class (in orange)

ones = np.ones((n_points), dtype=int)

np.random.seed(500)

x_ones = 0.3 + (np.random.normal(mu, sigma, n_points))*0.6

np.random.seed(5000)

y_ones = 0.7 + (np.random.normal(mu, sigma, n_points))*0.6

# Make a dataset dataframe

df_zeros = pd.DataFrame({'x': x_zeros,'y': y_zeros,'class' : zeros})

df_ones = pd.DataFrame({'x': x_ones,'y': y_ones,'class' : ones})

df = pd.concat([df_zeros,df_ones],ignore_index=True,axis=0)

# create the training data

X_train = df[['x','y']]

y_train = df[['class']]



# now for the classification:

from sklearn.tree import DecisionTreeClassifier

# underfit

underfit = DecisionTreeClassifier(max_depth=1,random_state=4)

underfit.fit(X_train, y_train)

# a good fit

goodfit = DecisionTreeClassifier(max_depth=2,random_state=4)

goodfit.fit(X_train, y_train)

# overfit

overfit = DecisionTreeClassifier(max_depth=5,random_state=4)

overfit.fit(X_train, y_train)



# produce a map of the predictions for a grid of points

xx, yy = np.meshgrid(np.arange(-0.3, 1.3, 0.001),np.arange(-0.3, 1.3, 0.001))

Z_underfit = underfit.predict(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape)

Z_goodfit  = goodfit.predict(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape)

Z_overfit  = overfit.predict(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape)



# and now the plots:

fig = plt.figure(figsize=(15, 4))

###########################################################

# The 'underfitting' plot

###########################################################

ax = fig.add_subplot(1, 3, 1)

ax.text(-0.25,-0.2,'underfitting', fontsize=18)

plt.scatter(x=x_zeros, y=y_zeros)

plt.scatter(x=x_ones, y=y_ones)

# we shall use the 'plasma' cmap as the two extremes of plasma are blue and orange

plt.contourf(xx, yy, Z_underfit, alpha=0.4, cmap='plasma')

plt.axis('off')



###########################################################

# The 'good' plot

###########################################################

ax = fig.add_subplot(1, 3, 2)

ax.text(-0.25,-0.2,'a good fit', fontsize=18)

plt.scatter(x=x_zeros, y=y_zeros)

plt.scatter(x=x_ones, y=y_ones)

plt.contourf(xx, yy, Z_goodfit, alpha=0.4, cmap='plasma')

plt.axis('off')



###########################################################

# The 'overfitting' plot

###########################################################

ax = fig.add_subplot(1, 3, 3)

ax.text(-0.25,-0.2,'overfitting', fontsize=18)

plt.scatter(x=x_zeros, y=y_zeros)

plt.scatter(x=x_ones, y=y_ones)

plt.contourf(xx, yy, Z_overfit, alpha=0.4, cmap='plasma')

plt.axis('off')



plt.show();
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score



#===========================================================================

# read in the data

#===========================================================================

train_data = pd.read_csv('../input/titanic/train.csv')

test_data  = pd.read_csv('../input/titanic/test.csv')

solution   = pd.read_csv('../input/submission-solution/submission_solution.csv')



#===========================================================================

# select some features

#===========================================================================

features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]



#===========================================================================

# for the features that are categorical we use pd.get_dummies:

# "Convert categorical variable into dummy/indicator variables."

#===========================================================================

X_train       = pd.get_dummies(train_data[features])

y_train       = train_data["Survived"]

final_X_test  = pd.get_dummies(test_data[features])



#===========================================================================

# perform the classification and the fit

#===========================================================================

classifier = DecisionTreeClassifier(random_state=4)

classifier.fit(X_train, y_train)



#===========================================================================

# use the model to predict 'Survived' for the test data

#===========================================================================

predictions = classifier.predict(final_X_test)



K_splits = 11

#===========================================================================

# calculate the scores

#===========================================================================

print(colored("The mean accuracy score of the train data is %.5f" % classifier.score(X_train, y_train), color='blue'))

CV_scores = cross_val_score(classifier, X_train, y_train, cv=K_splits)

print("The individual cross-validation scores are: \n",CV_scores)

print("The minimum cross-validation score is %.3f" % min(CV_scores))

print("The maximum cross-validation score is %.3f" % max(CV_scores))

print(colored("The mean  cross-validation   score is %.5f ± %0.2f" % (CV_scores.mean(), CV_scores.std() * 2), color='yellow'))

print(colored("The test (i.e. leaderboard)  score is %.5f" % accuracy_score(solution['Survived'],predictions), color='red'))
from yellowbrick.model_selection import CVScores

from sklearn.model_selection import StratifiedKFold

# Create a cross-validation strategy

cv = StratifiedKFold(n_splits=K_splits)

visualizer = CVScores(classifier, cv=cv, scoring='f1_weighted',size=(1200, 400))

visualizer.fit(X_train, y_train)

visualizer.show();
classifier = DecisionTreeClassifier(max_depth=1,max_features=1,random_state=4)
classifier.fit(X_train, y_train)

#===========================================================================

# use the model to predict 'Survived' for the test data

#===========================================================================

predictions = classifier.predict(final_X_test)

#===========================================================================

# calculate the scores

#===========================================================================

print(colored("The mean accuracy score of the train data is %.5f" % classifier.score(X_train, y_train), color='blue'))

CV_scores = cross_val_score(classifier, X_train, y_train, cv=K_splits)

print("The individual cross-validation scores are: \n",CV_scores)

print("The minimum cross-validation score is %.3f" % min(CV_scores))

print("The maximum cross-validation score is %.3f" % max(CV_scores))

print(colored("The mean  cross-validation   score is %.5f ± %0.2f" % (CV_scores.mean(), CV_scores.std() * 2), color='yellow'))

print(colored("The test (i.e. leaderboard)  score is %.5f" % accuracy_score(solution['Survived'],predictions), color='green'))

cv = StratifiedKFold(n_splits=K_splits)

visualizer = CVScores(classifier, cv=cv, scoring='f1_weighted',size=(1200, 400))

visualizer.fit(X_train, y_train)

visualizer.show();
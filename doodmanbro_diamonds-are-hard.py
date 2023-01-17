import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import math



from sklearn.linear_model import Ridge, LinearRegression, Lasso

import sklearn.ensemble as skens

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, KFold

import sklearn.metrics as metrics



%matplotlib inline
data = pd.read_csv('../input/diamonds.csv',index_col=0)

data.head()
data.describe()
# Pasted from Content Page:

# price price in US dollars ($326--$18,823)

# carat weight of the diamond (0.2--5.01)

# cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)

# color diamond colour, from J (worst) to D (best)

# clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

# x length in mm (0--10.74)

# y width in mm (0--58.9)

# z depth in mm (0--31.8)

# depth total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)

# table width of top of diamond relative to widest point (43--95)
# x, y, and z should not have zeroes. 

# e.g. --> if you go to Jared's and pick out a diamond for your fiance that's 0 mm wide,

# you're braver than me.



d = data.shape[0]

print('# Rows in the data before: {}'.format(d))

data = data[(data.x > 0) & ((data.y > 0) & (data.z > 0))]



print('# Rows after: {}'.format(data.shape[0]))

print('--------\nDifference of: {}'.format(d - data.shape[0]))



# Looks like a 20 instances of a "0" in one of x,y,z were removed

# only 20 out of ~54k, so I'm not gonna worry about possibly correcting/imputing

# these although that is a possibility
data['log_price'] = np.log(data.price)



plt.figure(figsize=(10,5))

plt.subplot(1,2,1);

sns.distplot(data.price);

plt.subplot(1,2,2);

sns.distplot(data.log_price);



data.drop('price', axis=1, inplace=True)



# 'price' is skewed. Makes sense. Pricier / quality diamonds are rarer for a reason.

# Perform a log transformation on the skewed 'price' data. Results: not normal,

# but bimodal and a big improvement.

# Information on log transforming technique: 

# https://stats.stackexchange.com/questions/107610/what-is-the-reason-the-log-transformation-is-used-with-right-skewed-distribution
### DATA VIZ ###



# plot the c's



c = ['cut','clarity','color']



plt.figure(figsize=(17,5))

for i in range(len(c)):

    plt.subplot(1,3,i+1)

    sns.countplot(data[c[i]], palette='Set2');

    plt.title('Value Counts of {}'.format(c[i]))
# good predictors?



p = ['carat','table','depth']

sns.pairplot(x_vars=p, y_vars=['log_price'], data=data, size=4.0);



# hmm in general heavy diamonds (high carat) are pricier!



# The carat is a unit of mass equal to 200 mg and is 

# used for measuring gemstones and pearls. - Wikipedia
# 'carat' seems to be a really good predicator of price

# Let's see what else we can found out about diamonds ...



sns.pairplot(x_vars=['carat','x','depth'], y_vars=['log_price'], data=data, hue='color', size=4.5);
sns.pairplot(x_vars=['carat','x','depth'], y_vars=['log_price'], data=data, hue='cut', size=4.5);
sns.pairplot(x_vars=['carat','x','depth'], y_vars=['log_price'], data=data, hue='clarity', size=4.5);



# Kind of hard to tell, but for the majority of diamonds, it looks like bigger (larger carat and x)

# do not mean necessarily mean the diamond is of the a better class of 'clarity','color', or 'cut'
# What about these guys?



xyz = ['x','y','z']

sns.pairplot(x_vars=xyz, y_vars=['log_price'], data=data, size=4.0);



# x and log_price seem to have a positive relationship

# Definitely a few outliers messing up our view in y and z
# lists from before are concatenated

numerics = xyz + p



# Normalize and plot

plt.figure(figsize=(18,14))

for col in numerics:

    mean = np.mean(data[col])

    std = np.std(data[col])

    data[col] = (data[col] - mean) / std



plt.subplot(2,1,1);

ax = sns.violinplot(data=data[numerics]);

ax.set_title('Violinplots of diamond data');



plt.subplot(2,1,2);

ax = sns.boxplot(data=data[numerics]);

ax.set_title('Boxplots of diamond data');



# Outliers strike again!
# map the c's ...

# Clarity:  (worst)I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF(best)

# Color: D (best) <---> J (worst)

# Cut: Fair (worst) - Good - Very Good - Premium - Ideal (best)



data.color = data.color.map({'J':1,'I':2,'H':3,'G':4,'F':5,'E':6,'D':7})

data.clarity = data.clarity.map({'I1':1, 'SI2':2, 'SI1':3, 'VS2':4, 'VS1':5, 'VVS2':6, 'VVS1':7, 'IF':8})

data.cut = data.cut.map({'Fair':1,'Good':2,'Very Good':3,'Premium':4,'Ideal':5})



# *** Good to note if you're new to data science: ***

# An order relationship must exist to encode this way.

# -- For example within the 'cut' attribute a value of 'Fair' is less (not equal to / not equally as valuable as) than

# a value of 'Ideal', as opposed to an attribute like 'Male or Female', which would be OneHotEncoded 

# (into two new binary columns) because both are equally as meaningful (Male is not greater than Female, & vice versa)

# Thus, an order relationship exists, and so 'Fair' --> 1 is less than 'Ideal' --> 5 

# Our regression model can pick up on this numerical relationship.
data.head()
plt.figure(figsize=(10,8))

sns.heatmap(data.corr(),annot=True,linewidths=0.5);
sns.pairplot(data);
# From our heatmap and pairplots: carat and x,y,z are highly correlated, but

# should we address what looks like outliers in y and z?



sns.pairplot(x_vars=xyz, y_vars='carat', data=data, size=4.0);
plt.figure(figsize=(18,5))



# outlier discussion:

# Are outliers EVIL? 

# Should you remove them??

# Let's look at 'y' and 'z', which seem to have a few questionably large points.



plt.subplot(1,4,1);

# y and z should/could look like x

ax = sns.regplot(x='x', y='carat', data=data, ci=0);

ax.set_title('Carat and X');



# what does y look like? Is the outlier affecting the regression line?

# Let's plot it again.

plt.subplot(1,4,2);

ax = sns.regplot(x='y', y='carat', data=data, ci=0);

ax.set_title('Carat and Y WITH outliers');

# Hmm maybe? It's possible those two points way down on the x axis are dragging the line down.



# Let's remove outliers and see how it looks:

data['y_test'] = data[data.y < 20].y # removing the two points with large values

plt.subplot(1,4,3);

ax = sns.regplot(x='y_test', y='carat', data=data, ci=0);

ax.set_title('Carat and Y WITHOUT outliers');

data.drop('y_test',axis=1,inplace=True)

# Looks pretty much like x. Nice!





# BUT let's go back and zoom in on the Carat and Y w/ outliers plot:

plt.subplot(1,4,4);

ax = sns.regplot(x='y', y='carat', data=data, ci=0);

ax.set_title('Carat and Y WITH outliers ZOOMED IN');

ax.set_ylim([-2,10]); # control the zoom of the plot.

ax.set_xlim([-2,5]);

# Hmm looks like x 



# Are these points influential?

# No, the third and fourth plots look identical and 

# the regression line still pretty much crosses through point (4,4) in both.

# Conclusion: outliers are real and not necessarily your enemy. 

# Huge (outlier-ish) diamonds like this actually exist and

# actually aren't the most important factor in determining price

# Source: http://www.jewelrywise.com/engagement-wedding/article/does-the-size-of-the-diamond-matter
# Even though it was fun to play around with them, we're going to drop 'x','y','z'

# They're all heavily correlated with 'carat' and will negatively affect a linear model.



data.drop(xyz, axis=1, inplace=True)
# Final look at our data

data.head()
# missing values?

missing = pd.DataFrame(data.isnull().sum(), columns=['total'])

missing



# nope.
### Model Building ###



def inv_log(preds):

    # apply inverse log function

    transformed_preds = []

    for val in preds:

        transformed_preds.append(np.round(math.exp(val), 2))

    return np.array(transformed_preds)

    



X = data.iloc[:,:-1]

y = data.log_price



X_train, X_50, y_train, y_50 = train_test_split(X,y, test_size=0.5, random_state=2)



print('Data split 50/50...\nShape of training: {}\nShape of Other: {}\n---'.format(X_train.shape, X_50.shape))



X_valid, X_test, y_valid, y_test = train_test_split(X_50, y_50, test_size=0.5, random_state=2)



print("Further split 'Other' into 50/50 validation and test sets...\nShape of Validation Set: {}\nShape of Test Set: {}".format(X_valid.shape, X_test.shape))



# Now we have a 50/25/25 split on our data
# Try some models



lr = LinearRegression()

rid = Ridge()

rf = skens.RandomForestRegressor()

gb = skens.GradientBoostingRegressor()



classifiers = [lr, rid, rf, gb]



kf = KFold(n_splits=5, shuffle=True, random_state=11)



results = []

names = []

for clf in classifiers:

    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kf, n_jobs=5)

    results.append(scores)

    name = str(clf.__class__).strip("'>").split('.')[-1]

    names.append(name)

    print(name + ':', scores)

    print('Average R-Squared Score:', np.mean(scores),'\n----')



# ensembles yield higher scores, but take a little longer to execute.
plt.figure(figsize=(13,6))

sns.boxplot(x=results, y=names);
rid = Ridge().fit(X_train, y_train)

y_pred_train = rid.predict(X_train)

y_pred = rid.predict(X_valid)

print('R Squared:\ntraining -- {}\nvalidation -- {}'.format(metrics.r2_score(y_train, y_pred_train), metrics.r2_score(y_valid, y_pred)))

error = inv_log(y_valid) - inv_log(y_pred)

print('Average Price Error: {}'.format(error.mean()))
# visualized

sns.regplot(x=y_pred, y=y_valid, marker='x',line_kws={'color':'red'});
# RF



rf = skens.RandomForestRegressor(n_estimators=15).fit(X_train, y_train)

y_pred_train = rf.predict(X_train)

y_pred = rf.predict(X_valid)

print('R Squared:\ntraining -- {}\nvalidation -- {}'.format(metrics.r2_score(y_train, y_pred_train), metrics.r2_score(y_valid, y_pred)))

error = inv_log(y_valid) - inv_log(y_pred)

print('Average Price Error: {}'.format(error.mean()))
# As expected: tighter than the linear regression model

sns.regplot(x=y_pred, y=y_valid, marker='x',line_kws={'color':'red'});
feats = pd.DataFrame(rf.feature_importances_, columns=['Importance'],

             index=X_train.columns).sort_values('Importance', ascending=False)

# feats.plot(kind='barh')

feats

# Carats are key!
# GB



gb = skens.GradientBoostingRegressor().fit(X_train, y_train)

y_pred_train = gb.predict(X_train)

y_pred = gb.predict(X_valid)

print('R Squared:\ntraining -- {}\nvalidation -- {}'.format(metrics.r2_score(y_train, y_pred_train), metrics.r2_score(y_valid, y_pred)))

error = inv_log(y_valid) - inv_log(y_pred)

print('Average Price Error: {}'.format(error.mean()))
sns.regplot(x=y_pred, y=y_valid, marker='x',line_kws={'color':'red'});
# more to come (maybe) with GridSearchCV and parameters
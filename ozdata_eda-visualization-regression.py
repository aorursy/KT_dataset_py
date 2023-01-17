import numpy as np 

import pandas as pd 

from sklearn.model_selection import train_test_split

import warnings

# suppress warnings later



import matplotlib.pyplot as plt

from matplotlib import gridspec

import matplotlib

import seaborn as sns

import missingno

from collections import Counter

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/insurance/insurance.csv')
df.head()
target = df['charges']

df = df.drop('charges', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size = 0.2, random_state = 42)
matplotlib.style.use('seaborn')

figure, ax = plt.subplots(nrows=3, ncols=2, figsize = (8,8))



plt.subplot(3,2,1)

plt.hist(X_train.age, bins = 20)

plt.title('Age distribution')

plt.subplot(3,2,2)

plt.bar(list(Counter(X_train.sex).keys()), height = list(Counter(X_train.sex).values()))

plt.title('Distribution by gender')

plt.subplot(3,2,3)

plt.bar(list(Counter(X_train.children).keys()), height = list(Counter(X_train.children).values()))

plt.title('Number of children')

plt.subplot(3,2,4)

plt.hist(X_train.bmi, bins = 15)

plt.title('BMI distribution')

plt.subplot(3,2,5)

plt.bar(list(Counter(X_train.smoker).keys()), height = list(Counter(X_train.smoker).values()))

plt.title('Smoker distribution')

plt.subplot(3,2,6)

plt.bar(list(Counter(X_train.region).keys()), height = list(Counter(X_train.region).values()))

plt.title('Region Distribution')

figure.tight_layout()

def cat_transform(df):

    for att in ['sex', 'smoker','region']:

        # transform data type to category

        df[att] = df[att].astype('category')

        # use cat.codes for encoding into numeric

        df[att] = df[att].cat.codes

    return df
X_train = cat_transform(X_train)

X_test = cat_transform(X_test)
# plotting correlation matrix over all attributes including the target

all_att = X_train.join(y_train)

all_att.head()

correlations = all_att.corr()

mask = np.triu(np.ones_like(correlations, dtype = np.bool))



f, ax = plt.subplots(figsize = (7,7))

cmap = sns.diverging_palette(220,10, as_cmap = True)



sns.heatmap(correlations, mask = mask, cmap = cmap, 

            vmax= .3, center=0, square = True)
f, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,5))

plt.subplot(121)

sns.boxplot(x = "smoker", y = 'charges', data = all_att, palette = 'Set3')

plt.subplot(122)

sns.violinplot(x = "smoker", y = 'charges', data = all_att, palette = 'Set3')

plt.suptitle('Charges distribution wrt Smoker status')
mean = X_train.bmi.mean()

median = X_train.bmi.median()

plt.axvline(mean, color='r', linestyle='-')

plt.axvline(median, color='g', linestyle='--')

sns.distplot(X_train.bmi)

plt.legend({'Mean':mean,'Median':median})
def bmi_split(df):

    df['underweight'] = df['bmi'] < 18.5

    df['healthyweight'] = (df['bmi'] >= 18.5) & (df['bmi'] < 25)

    df['overweight'] = (df['bmi'] >= 25) & (df['bmi'] < 30)

    df['obese'] = (df['bmi'] >= 30) & (df['bmi'] < 40)

    df['severelyobese'] = (df['bmi'] >= 40)

    return df
X_train = bmi_split(X_train)

X_test = bmi_split(X_test)
# add new columns to all_att df

all_att = X_train.join(y_train)



f, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (15,5))

plt.subplot(231)

sns.scatterplot(x = 'age', y = 'charges', data = all_att, legend = None)

plt.title('All values')

bmi_types = ['underweight', 'healthyweight', 'overweight', 'obese','severelyobese']

for i,htype in enumerate(bmi_types):

    plt.subplot(2,3,i+2)

    sns.scatterplot(x = 'age', y = 'charges', data = all_att, 

                hue = htype, size = htype,

                size_order = [True, False], legend = None)

    plt.title( '{} highlighted'.format(htype))

plt.tight_layout()

plt.show()
colors = ['Reds', 'Blues', 'Purples', 'Greens']

line_color = ['b', 'g', 'y', 'r']

figure, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (10,10), sharex = True)

figure.suptitle('Bi-variate distribution of Charges and BMI across regions', fontsize = 20)

for i in range(4):

    plt.subplot(2,2,i+1)

    median_bmi = all_att[all_att['region'] == i].bmi.median()

    mean_bmi = all_att[all_att['region'] == i].bmi.mean()

    median_charge = all_att[all_att['region'] == i].charges.median()

    mean_charge = all_att[all_att['region'] == i].charges.mean()

    plt.axvline(median_bmi, color=line_color[i], linestyle='--')

    plt.axvline(mean_bmi, color=line_color[i], linestyle='-')

    plt.axhline(median_charge, color=line_color[i], linestyle='--')

    plt.axhline(mean_charge, color=line_color[i], linestyle='-')

    sns.kdeplot(all_att[all_att['region'] == i].bmi, all_att[all_att.region == i].charges,

                cmap = colors[i], shade = True)

    plt.xlim(10,50)

    plt.ylim(-5000, 60000)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,7))

plt.subplot(121)

sns.scatterplot(x = 'age', y = 'charges', data = all_att, hue = 'children', size = 'children', legend = 'full')

plt.subplot(122)

sns.stripplot(x = "children", y = 'charges', data = all_att, palette = sns.cubehelix_palette())

plt.suptitle('Highlighting the number of children covered by insurance')
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error, make_scorer
y_train = np.log1p(y_train)

y_test = np.log1p(y_test)
sc = StandardScaler()

X_train.loc[:, ['age', 'bmi']] = sc.fit_transform(X_train.loc[:, ['age', 'bmi']])

X_test.loc[:, ['age', 'bmi']] = sc.fit_transform(X_test.loc[:, ['age', 'bmi']])
scorer = make_scorer(mean_squared_error, greater_is_better = False)



def rmse_cv_train(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))

    return(rmse)



def rmse_cv_test(model):

    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))

    return(rmse)
lr = LinearRegression()

lr.fit(X_train, y_train)



print("RMSE on Training set :", rmse_cv_train(lr).mean())

print("RMSE on Test set :", rmse_cv_test(lr).mean())

y_train_pred = lr.predict(X_train)

y_test_pred = lr.predict(X_test)
ridge = RidgeCV(alphas = [0.001, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])

ridge.fit(X_train, y_train)

alpha = ridge.alpha_

print("Best alpha :", alpha)



# Selecting more values, concentrated around the best value of alpha 

ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 

                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,

                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4, alpha * 1.5, alpha * 1.55], 

                cv = 10)

ridge.fit(X_train, y_train)

alpha = ridge.alpha_

print("Best alpha :", alpha)
print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())

print("Ridge RMSE on Test set :", rmse_cv_test(ridge).mean())
lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 

                          0.3, 0.6, 1], 

                max_iter = 50000, cv = 10)

lasso.fit(X_train, y_train)

alpha = lasso.alpha_

print("Best alpha :", alpha)



# Selecting more values, concentrated around the best value of alpha 

lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 

                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 

                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 

                          alpha * 1.4, alpha * 1.5, alpha * 1.55], 

                max_iter = 50000, cv = 10)

lasso.fit(X_train, y_train)

alpha = lasso.alpha_

print("Best alpha :", alpha)



print("Lasso RMSE on Training set :", rmse_cv_train(lasso).mean())

print("Lasso RMSE on Test set :", rmse_cv_test(lasso).mean())
from sklearn.tree import DecisionTreeRegressor

curr_err = np.inf

best_depth = 0

for i in range(10):

    dtr = DecisionTreeRegressor(max_depth=i+1)

    dtr.fit(X_train, y_train)

    if rmse_cv_train(dtr).mean() < curr_err:

        curr_err = rmse_cv_train(dtr).mean()

        best_depth = i 

print('The best maximum depth is :', best_depth)

dtr = DecisionTreeRegressor(max_depth=best_depth)

dtr.fit(X_train, y_train)

print("Decision Tree Regressor RMSE on Test set :", rmse_cv_test(dtr).mean())
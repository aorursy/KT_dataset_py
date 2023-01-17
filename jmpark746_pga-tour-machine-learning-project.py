# importing packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Importing the data 

df = pd.read_csv('../input/pgaTourData.csv')



# Examining the first 5 data

print(df.head())
df.info()
df.shape
# Replace NaN with 0 in Top 10 

df['Top 10'].fillna(0, inplace=True)

df['Top 10'] = df['Top 10'].astype(int)



# Replace NaN with 0 in # of wins

df['Wins'].fillna(0, inplace=True)

df['Wins'] = df['Wins'].astype(int)



# Drop NaN values 

df.dropna(axis = 0, inplace=True)
# Change Rounds to int

df['Rounds'] = df['Rounds'].astype(int)



# Change Points to int 

df['Points'] = df['Points'].apply(lambda x: x.replace(',',''))

df['Points'] = df['Points'].astype(int)



# Remove the $ and commas in money 

df['Money'] = df['Money'].apply(lambda x: x.replace('$',''))

df['Money'] = df['Money'].apply(lambda x: x.replace(',',''))

df['Money'] = df['Money'].astype(float)
df.info()
df.head()
df.describe()
# Looking at the distribution of data

f, ax = plt.subplots(nrows = 6, ncols = 3, figsize=(20,20))

distribution = df.loc[:,df.columns!='Player Name'].columns

rows = 0

cols = 0

for i, column in enumerate(distribution):

    p = sns.distplot(df[column], ax=ax[rows][cols])

    cols += 1

    if cols == 3:

        cols = 0

        rows += 1

# Looking at the number of players with Wins for each year 

win = df.groupby('Year')['Wins'].value_counts()

win = win.unstack()

win.fillna(0, inplace=True)



# Converting win into ints

win = win.astype(int)



print(win)
# Looking at the percentage of players without a win in that year 

players = win.apply(lambda x: np.sum(x), axis=1)

percent_no_win = win[0]/players

percent_no_win = percent_no_win*100

print(percent_no_win)
# Plotting percentage of players without a win each year 

fig, ax = plt.subplots()

bar_width = 0.8

opacity = 0.7 

index = np.arange(2010, 2019)



plt.bar(index, percent_no_win, bar_width, alpha = opacity)

plt.xticks(index)

plt.xlabel('Year')

plt.ylabel('%')

plt.title('Percentage of Players without a Win')
# Plotting the number of wins on a bar chart 

fig, ax = plt.subplots()

index = np.arange(2010, 2019)

bar_width = 0.2

opacity = 0.7 



def plot_bar(index, win, labels):

    plt.bar(index, win, bar_width, alpha=opacity, label=labels)



# Plotting the bars

rects = plot_bar(index, win[0], labels = '0 Wins')

rects1 = plot_bar(index + bar_width, win[1], labels = '1 Wins')

rects2 = plot_bar(index + bar_width*2, win[2], labels = '2 Wins')

rects3 = plot_bar(index + bar_width*3, win[3], labels = '3 Wins')

rects4 = plot_bar(index + bar_width*4, win[4], labels = '4 Wins')

rects5 = plot_bar(index + bar_width*5, win[5], labels = '5 Wins')



plt.xticks(index + bar_width, index)

plt.xlabel('Year')

plt.ylabel('Number of Wins')

plt.title('Distribution of Wins each Year')

plt.legend()
# Percentage of people who did not place in the top 10 each year

top10 = df.groupby('Year')['Top 10'].value_counts()

top10 = top10.unstack()

top10.fillna(0, inplace=True)

players = top10.apply(lambda x: np.sum(x), axis=1)



no_top10 = top10[0]/players * 100

print(no_top10)
# Who are some of the longest hitters 

distance = df[['Year','Player Name','Avg Distance']].copy()

distance.sort_values(by='Avg Distance', inplace=True, ascending=False)

print(distance.head())
# Who made the most money

money_ranking = df[['Year','Player Name','Money']].copy()

money_ranking.sort_values(by='Money', inplace=True, ascending=False)

print(money_ranking.head())
# Who made the most money each year

money_rank = money_ranking.groupby('Year')['Money'].max()

money_rank = pd.DataFrame(money_rank)

print(money_rank.iloc[0,0])



indexs = np.arange(2010, 2019)

names = []

for i in range(money_rank.shape[0]):

    temp = df.loc[df['Money'] == money_rank.iloc[i,0],'Player Name']

    names.append(str(temp.values[0]))



money_rank['Player Name'] = names

print(money_rank)
# Looking at the changes in statistics over time 

f, ax = plt.subplots(nrows = 5, ncols = 3, figsize=(35,65))

distribution = df.loc[:,(df.columns!='Player Name') & (df.columns!='Wins')].columns

distribution = distribution[distribution != 'Year']



print(distribution)

rows = 0

cols = 0

for i, column in enumerate(distribution):

    p = sns.boxplot(x = 'Year', y = column, data=df, ax=ax[rows][cols], showfliers=False)

    p.set_ylabel(column,fontsize=20)

    p.set_xlabel('Year',fontsize=20)

    cols += 1

    if cols == 3:

        cols = 0

        rows += 1

# Defining the players that had a win or more in each year 

champion = df.loc[df['Wins'] >= 1, :]

print(champion.head())
f, ax = plt.subplots(nrows = 8, ncols = 2, figsize=(35,65))

distribution = df.loc[:,df.columns!='Player Name'].columns

distribution = distribution[distribution != 'Year']



rows = 0

cols = 0

lower_better = ['Average Putts', 'Average Score']

for i, column in enumerate(distribution):

    avg = df.groupby('Year')[column].mean()

    best = champion.groupby('Year')[column].mean()

    ax[rows,cols].plot(avg, 'o-',)

    ax[rows,cols].plot(best, 'o-',)

    ax[rows,cols].set_title(column, fontsize = 20)

    

    cols += 1

    if cols == 2:

        cols = 0

        rows += 1
# Plot the correlation matrix between variables 

corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            cmap='coolwarm')

df.corr()['Wins']
# Importing the Machine Learning modules

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import RFE

from sklearn.metrics import classification_report

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVC  

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler





# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)



if __name__ == '__main__':

    with warnings.catch_warnings():

        warnings.simplefilter('ignore', category=ImportWarning)
# Adding the Winner column to determine if the player won that year or not 

df['Winner'] = df['Wins'].apply(lambda x: 1 if x>0 else 0)



# New DataFrame 

ml_df = df.copy()



# Y value for machine learning is the Winner column

target = df['Winner']



# Removing the columns Player Name, Wins, and Winner from the dataframe

ml_df.drop(['Player Name','Wins','Winner'], axis=1, inplace=True)

print(ml_df.head())
per_no_win = target.value_counts()[0] / (target.value_counts()[0] + target.value_counts()[1])

per_no_win = per_no_win.round(4)*100

print(str(per_no_win)+str('%'))
# Function for the logisitic regression 

def log_reg(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                   random_state = 10)

    clf = LogisticRegression().fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print('Accuracy of Logistic regression classifier on training set: {:.2f}'

         .format(clf.score(X_train, y_train)))

    print('Accuracy of Logistic regression classifier on test set: {:.2f}'

         .format(clf.score(X_test, y_test)))

    cf_mat = confusion_matrix(y_test, y_pred)

    confusion = pd.DataFrame(data = cf_mat)

    print(confusion)

    

    print(classification_report(y_test, y_pred))

    

    # Returning the 5 important features 

    rfe = RFE(clf, 5)

    rfe = rfe.fit(X, y)

    print('Feature Importance')

    print(X.columns[rfe.ranking_ == 1].values)

    

    print('ROC AUC Score: {:.2f}'.format(roc_auc_score(y_test, y_pred)))
log_reg(ml_df, target)
# Adding Domain Features 

ml_d = ml_df.copy()

# Top 10 / Money might give us a better understanding on how well they placed in the top 10

ml_d['Top10perMoney'] = ml_d['Top 10'] / ml_d['Money']



# Avg Distance / Fairway Percentage to give us a ratio that determines how accurate and far a player hits 

ml_d['DistanceperFairway'] = ml_d['Avg Distance'] / ml_d['Fairway Percentage']



# Money / Rounds to see on average how much money they would make playing a round of golf 

ml_d['MoneyperRound'] = ml_d['Money'] / ml_d['Rounds']
log_reg(ml_d, target)
# Adding Polynomial Features to the ml_df 

mldf2 = ml_df.copy()

poly = PolynomialFeatures(2)

poly = poly.fit(mldf2)

poly_feature = poly.transform(mldf2)

print(poly_feature.shape)



# Creating a DataFrame with the polynomial features 

poly_feature = pd.DataFrame(poly_feature, columns = poly.get_feature_names(ml_df.columns))

print(poly_feature.head())
log_reg(poly_feature, target)
def svc_class(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                   random_state = 10)

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)



    

    svclassifier = SVC(kernel='rbf', C=10000)  

    svclassifier.fit(X_train_scaled, y_train) 

    y_pred = svclassifier.predict(X_test_scaled) 

    print('Accuracy of SVM on training set: {:.2f}'

         .format(svclassifier.score(X_train_scaled, y_train)))

    print('Accuracy of SVM classifier on test set: {:.2f}'

         .format(svclassifier.score(X_test_scaled, y_test)))



    

    print('ROC AUC Score: {:.2f}'.format(roc_auc_score(y_test, y_pred)))
svc_class(ml_df, target)
svc_class(ml_d, target)
svc_class(poly_feature, target)
def random_forest(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                   random_state = 10)

    clf = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print('Accuracy of Random Forest classifier on training set: {:.2f}'

         .format(clf.score(X_train, y_train)))

    print('Accuracy of Random Forest classifier on test set: {:.2f}'

         .format(clf.score(X_test, y_test)))

    

    cf_mat = confusion_matrix(y_test, y_pred)

    confusion = pd.DataFrame(data = cf_mat)

    print(confusion)

    

    print(classification_report(y_test, y_pred))

    

    # Returning the 5 important features 

    rfe = RFE(clf, 5)

    rfe = rfe.fit(X, y)

    print('Feature Importance')

    print(X.columns[rfe.ranking_ == 1].values)

    

    print('ROC AUC Score: {:.2f}'.format(roc_auc_score(y_test, y_pred)))
random_forest(ml_df, target)
random_forest(ml_d, target)
random_forest(poly_feature, target)
# New DataFrame 

earning_df = df.copy()



# Y value for machine learning is the Money column

target = earning_df['Money']



# Removing the columns Player Name, Wins, Winner, Points, Top 10, and Money from the dataframe

earning_df.drop(['Player Name','Wins','Winner','Points','Top 10','Money'], axis=1, inplace=True)



print(earning_df.head())
# Importing the Machine Learning modules

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import RFE

from sklearn.metrics import classification_report

from sklearn.preprocessing import PolynomialFeatures



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures

def linear_reg(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 10)

    clf = LinearRegression().fit(X_train, y_train)

    y_pred = clf.predict(X_test)



    print('R-Squared on training set: {:.3f}'

          .format(clf.score(X_train, y_train)))

    print('R-Squared on test set {:.3f}'

          .format(clf.score(X_test, y_test)))

    

    print('linear model coeff (w):\n{}'

         .format(clf.coef_))

    print('linear model intercept (b): {:.3f}'

         .format(clf.intercept_))

linear_reg(earning_df, target)
# Creating a Polynomial Feature to improve R-Squared

poly = PolynomialFeatures(2)

poly = poly.fit(earning_df)

poly_earning = poly.transform(earning_df)

print(poly_feature.shape)



# Creating a DataFrame with the polynomial features 

poly_earning = pd.DataFrame(poly_feature, columns = poly.get_feature_names(earning_df.columns))
linear_reg(poly_earning, target)
# Adding a regularization penalty (Ridge)

def linear_reg_ridge(X, y, al):

    X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                   random_state = 10)

    clf = Ridge(alpha = al).fit(X_train, y_train)



    print('(poly deg 2 + ridge) R-squared score (training): {:.3f}'

         .format(clf.score(X_train, y_train)))

    print('(poly deg 2 + ridge) R-squared score (test): {:.3f}'

         .format(clf.score(X_test, y_test)))

    

    print('(poly deg 2 + ridge) linear model coeff (w):\n{}'

         .format(clf.coef_))

    print('(poly deg 2 + ridge) linear model intercept (b): {:.3f}'

         .format(clf.intercept_))
linear_reg_ridge(poly_earning, target, al = 1)
linear_reg_ridge(poly_earning, target, al = 100)
from sklearn.model_selection import cross_val_score



def cross_val(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 10)

    clf = Ridge().fit(X_train, y_train)

    scores = cross_val_score(clf, X, y, cv=5)

    

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print(scores)
cross_val(poly_earning, target)
# Using the Linear Regression to predict Tiger Wood's Earnings based on the Model

def find_earning(X,y,name,year):

    X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                   random_state = 10)

    clf = Ridge().fit(X_train, y_train)

    y_pred = clf.predict(X)

    y_pred = pd.Series(y_pred)



    pred_data = pd.concat([X, y_pred], axis=1)

    pred_name = pd.concat([pred_data, df['Player Name']], axis=1)



    return pred_name.loc[(pred_name['Player Name']==name) & (pred_name['Year']==year), 0]

print('Tiger Woods\' Predicted Earning: ' + 

      str(find_earning(X = poly_earning, y = target, name = 'Tiger Woods', year = 2013).values[0]))



# Tiger Wood's actual earnings in 2018 

tw13 = df.loc[(df['Player Name']=='Tiger Woods') & (df['Year']==2013), 'Money']

print('Tiger Woods\' Actual Earning: ' + str(tw13.values[0]))

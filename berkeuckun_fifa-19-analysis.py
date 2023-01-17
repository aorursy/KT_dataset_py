# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/fifa19/data.csv')
df.head(7)
df.columns
del df['Unnamed: 0']

del df['ID']

del df['Photo']

del df['Flag']

del df['Club Logo']
df.shape
df.describe()
df.head(7)
df[df['Club'] == 'FC Barcelona'].groupby(['Name', 'Overall','Potential'], as_index = False).mean().sort_values(by = 'Composure', ascending = False)
df['Jersey Number'] = df['Jersey Number'].values.astype(int)
dfcorr = df[['Age', 'Overall', 'Potential', 'Finishing', 'ShotPower', 'LongShots', 'Penalties', 'FKAccuracy']].corr()
cmap = sns.color_palette('viridis')

f = sns.heatmap(dfcorr, annot = True, cmap = cmap)

f.set_title('Correlation of Shooting Attributes of Football Players', fontsize = 12)

plt.show()
sns.pairplot(df[['Age', 'Overall', 'Potential', 'Finishing', 'ShotPower', 'LongShots', 'Penalties', 'FKAccuracy']]);
countries = df['Nationality'].unique()

top_countries = []

for i in countries:

    mean = df[df['Nationality'] == i]['Overall'][0:100].mean()

    if (mean > 75) & (len(df[df['Nationality'] == i]) > 20):

        top_countries.append(i)

        print('Average of Overall', i, 'is', df[df['Nationality'] == i]['Overall'][0:100].mean())

print('Top countries are: ', top_countries)
sns.distplot(df[df['Nationality'] == 'Brazil']['Potential']);
sns.countplot(df[df['Nationality'] == 'Portugal']['Overall'], palette='ocean_r');
columns = ['Finishing', 'Positioning', 'ShotPower', 'LongShots', 'Volleys', 'Penalties']

for i in columns:

    mean = df[i].mean()

    df[i].replace(np.nan, mean, inplace = True)
for i in columns:

    for j in columns:

        if i != j:

            corrv = np.corrcoef(df[i], df[j])[1,0]

            print('Correlation between {} and {} is equal to:'.format(i, j), corrv)

        elif i == j:

            print('-'*70)
att_corr = df[columns].corr()

sns.heatmap(att_corr, annot = True)

plt.show()
X = df.loc[:,['Positioning', 'ShotPower', 'LongShots', 'Volleys', 'Penalties']].values

y = df.loc[:, 'Finishing'].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 8, test_size = 0.2)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test ,y_pred)
sns.scatterplot(y_pred, y_test)

plt.plot(range(100), range(100), color = 'red')

plt.show()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.linear_model import SGDRegressor

regressor = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)

regressor.fit(X_train, y_train.ravel())
y_pred = regressor.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
regressor.coef_, regressor.intercept_
mean_squared_error(y_test, y_pred)
sns.scatterplot(y_pred, y_test)

plt.plot(range(100), range(100), color = 'red')

plt.show()
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha = 1, solver = 'cholesky')

ridge_reg.fit(X_train, y_train)

y_pred = ridge_reg.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
sns.scatterplot(y_pred, y_test)

plt.plot(range(100), range(100), color = 'red')

plt.show()
mean_squared_error(y_test, y_pred)
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha = 0.1)

lasso_reg.fit(X_train, y_train)

y_pred = lasso_reg.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
sns.scatterplot(y_pred, y_test)

plt.plot(range(100), range(100), color = 'red')

plt.show()
mean_squared_error(y_test, y_pred)
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha = 0.1, l1_ratio = 0.5)

elastic_net.fit(X_train, y_train)

y_pred = elastic_net.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
sns.scatterplot(y_pred, y_test)

plt.plot(range(80), range(80), color = '#d13495')

plt.show()
mean_squared_error(y_test, y_pred)
# Pace stats calculation

df['Pace'] = round(df['SprintSpeed'] * 0.55 + df['Acceleration'] * 0.45)

df['Pace'] = df['Pace'].replace(np.nan, df['Pace'].mean())

df['Pace'] = df['Pace'].astype(int)



# Shooting stats calculation

df['Shooting'] = round(df['Finishing'] * 0.45 + df['LongShots'] * 0.2 + df['ShotPower'] * 0.2 + df['Positioning'] * 0.05 + df['Penalties'] * 0.05 + df['Volleys'] * 0.05)

df['Shooting'] = df['Shooting'].replace(np.nan, df['Shooting'].mean())

df['Shooting'] = df['Shooting'].astype(int)



# Passing stats calculation

df['Passing'] = round(df['ShortPassing'] * 0.35 + df['Vision'] * 0.2 + df['Crossing'] * 0.2 + df['LongPassing'] * 0.15 + df['Curve'] * 0.05 + df['FKAccuracy'] * 0.05) 

df['Passing'] = df['Passing'].replace(np.nan, df['Passing'].mean())

df['Passing'] = df['Passing'].astype(int)



# Dribbling stats calculation

df['Dribbling_Ovr'] = round(df['Agility'] * 0.1 + df['Balance'] * 0.05 + df['BallControl'] * 0.35 + df['Dribbling'] * 0.5)

df['Dribbling_Ovr'] = df['Dribbling_Ovr'].replace(np.nan, df['Dribbling_Ovr'].mean())

df['Dribbling_Ovr'] = df['Dribbling_Ovr'].astype(int)



# Defending stats calculation

df['Defending'] = round(df['HeadingAccuracy'] * 0.1 + df['Interceptions'] * 0.2 + df['Marking'] * 0.3 + df['SlidingTackle'] * 0.1 + df['StandingTackle'] * 0.3)

df['Defending'] = df['Defending'].replace(np.nan, df['Defending'].mean())

df['Defending'] = df['Defending'].astype(int)



# Physical stats calculation

df['Physical'] = round(df['Aggression'] * 0.2 + df['Jumping'] * 0.05 + df['Stamina'] * 0.25 + df['Strength'] * 0.5) 

df['Physical'] = df['Physical'].replace(np.nan, df['Physical'].mean())

df['Physical'] = df['Physical'].astype(int)
main_stats = ['Pace', 'Shooting', 'Passing', 'Dribbling_Ovr', 'Defending', 'Physical']

for i in main_stats:

    for j in top_countries[:-1]:

        for k in range(2):

            player_position = np.array(df[df['Nationality'] == j]['Position'])[k]

            if player_position != 'GK':

                player_name = list(df[df['Nationality'] == j]['Name'])[k]

                stats = {i: np.array(df[df['Nationality'] == j][i])[k]}

                print(player_name, '\'s', i.lower() ,'stat is', stats.get(i))
sns.jointplot(x = df['Overall'], y = df['Age'], data = df, kind = 'kde');
def compare_players(player_1, player_2):

    player_name_1 = df[df['Name'] == player_1]

    player_name_2 = df[df['Name'] == player_2]

    player_1_stats = np.array(player_name_1[['Pace', 'Shooting', 'Passing', 'Dribbling_Ovr', 'Defending', 'Physical']])

    player_2_stats = np.array(player_name_2[['Pace', 'Shooting', 'Passing', 'Dribbling_Ovr', 'Defending', 'Physical']])

    difference = []

    for i in range(0, 6):

        diff = player_1_stats[0] - player_2_stats[0]

        difference.append(diff)

    difference = np.array(difference)

    result = np.concatenate((player_1_stats.reshape(6, 1), player_2_stats.reshape(6, 1), difference[0].reshape(6, 1)),1)

    print(player_1, player_2, '\n',result)
compare_players('L. Messi', 'Cristiano Ronaldo')
gold_band = df[df['Overall'] >= 75]

silver_band = df[(df['Overall'] < 75) & (df['Overall'] > 64)]

bronz_band = df[df['Overall'] <= 64]
df['Version'] = None
df.loc[df['Overall'] >= 75, 'Version'] = 3
df.loc[(df['Overall'] < 75) & (df['Overall'] > 64), 'Version'] = 2
df.loc[df['Overall'] <= 64, 'Version'] = 1
df['Card Type'] = None
df.loc[df['Overall'] >= 75, 'Card Type'] = 'Gold'
df.loc[(df['Overall'] < 75) & (df['Overall'] > 64), 'Card Type'] = 'Silver'
df.loc[df['Overall'] <= 64, 'Card Type'] = 'Bronze'
grid = sns.FacetGrid(df, col = 'Card Type', row = 'Preferred Foot')

grid.map(sns.distplot, 'Shooting', bins = 20, color = '#0f8da3')

grid.add_legend();
stats = ['Overall', 'Potential', 'Pace', 'Shooting', 'Passing', 'Dribbling_Ovr', 'Defending', 'Physical', 'Version']
df['Version'] = df['Version'].astype(int)
main_stats_corr = df[stats].corr()

cmap = sns.color_palette("mako")

sns.heatmap(main_stats_corr, annot = True, cmap=cmap,);
plt.figure(figsize = (16,9))

sns.set_style('darkgrid')

sns.set_palette('viridis')

plt.title('Overall Distribution of Card Types')

sns.boxplot(x = 'Card Type', y = 'Overall', data = df);
df.tail()
for i in main_stats:

    df['{} Band'.format(i)] = pd.cut(df[i], 10)
for i in main_stats:

    band_interval = df['{} Band'.format(i)].unique().sort_values(ascending = True)

    for j in range(len(band_interval)):

        df['{} Band'.format(i)] = df['{} Band'.format(i)].replace(band_interval[j], j + 1)
df.iloc[:, -8:]
X = df.loc[:, ['Pace Band', 'Shooting Band', 'Passing Band', 'Dribbling_Ovr Band', 'Defending Band', 'Physical Band']].values

y = df.loc[:, 'Version'].values.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 7, test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
acc_knn = knn.score(X_train, y_train) * 100

print(acc_knn)
from sklearn.ensemble import RandomForestClassifier

randf = RandomForestClassifier()

randf.fit(X_train, y_train)
y_pred = randf.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
acc_randf = randf.score(X_train, y_train) * 100

print(acc_randf)
from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
acc_linear_svc = svc.score(X_train, y_train) * 100

print(acc_linear_svc)
from sklearn.linear_model import Perceptron

perceptron = Perceptron()

perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
acc_perceptron = perceptron.score(X_train, y_train) * 100

print(acc_perceptron)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
acc_gnb = gnb.score(X_train, y_train) * 100

print(acc_gnb)
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
acc_dtree = dtree.score(X_train, y_train) * 100

print(acc_dtree)
models = pd.DataFrame({'Model' : ['K-Nearest Neighbors', 'Random Forest', 'Linear SVC', 'Gaussian NB', 'Perceptron' ,'Decision Tree']

                      ,'Score' : [acc_knn, acc_randf, acc_linear_svc, acc_gnb, acc_perceptron, acc_dtree]}).sort_values(by = 'Score', ascending=False)
models
def player_version_finder(Pace, Shooting, Passing, Dribbling_Ovr, Defending, Physical, classification = dtree):

    main_stats = ['Pace', 'Shooting', 'Passing', 'Dribbling_Ovr', 'Defending', 'Physical']

    variables = [Pace, Shooting, Passing, Dribbling_Ovr, Defending, Physical]

    band_df = pd.DataFrame()

    index = []

    for i, j in zip(variables, main_stats):

        band_df['{} Band'.format(j)] = pd.cut(df['{}'.format(j)], 10).unique().sort_values(ascending = True)

        for k in range(len(band_df['{} Band'.format(j)])):

            if i in band_df['{} Band'.format(j)].unique().sort_values(ascending = True)[k]:

                index.append(k + 1)

    converted_inputs = np.array(index).reshape(1, -1)

    prediction = classification.predict(converted_inputs)

    if prediction == 1:

        print('According to the {} model this has to be a Bronze player'.format(classification))

    elif prediction == 2:

        print('According to the {} model this has to be a Silver player'.format(classification))

    elif prediction == 3:

        print('According to the {} model this has to be a Gold player'.format(classification))
player_version_finder(79,60,75,72,40,67, randf)
rng = np.random.default_rng()

all_players = []

classifications = [randf, dtree, knn]

for j in classifications:

    print('-' * 100)

    for i in range(10):

        all_players.append(rng.integers(30, 89, size=6))

        print(all_players[i])

        player_version_finder(all_players[i][0], all_players[i][1], all_players[i][2], all_players[i][3], all_players[i][4], all_players[i][5], j)
%config IPCompleter.greedy=True

%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

sns.set()
folder_path = '/kaggle/input/formula-1-race-data-19502017/'

lapTimes = pd.read_csv(folder_path + 'lapTimes.csv', encoding='latin-1')

races = pd.read_csv(folder_path + 'races.csv', encoding='latin-1')

drivers = pd.read_csv(folder_path + 'drivers.csv', encoding='latin-1')

results = pd.read_csv(folder_path + 'results.csv', encoding='latin-1')

circuits = pd.read_csv(folder_path + 'circuits.csv', encoding='latin-1')

status = pd.read_csv(folder_path + 'status.csv', encoding='latin-1')
folder_path_2 = '/kaggle/input/formula1addonscores/'

top_100 = pd.read_csv(folder_path_2 + 'top_100.csv', encoding='latin-1', index_col = False)

top_100 = top_100.drop(['Unnamed: 0'], axis=1)

scores = pd.read_excel(folder_path_2 + 'score_season_2014_2015_2016.xlsx', header=0)

print(top_100.columns)

print(scores.columns)

print(top_100.head())

print(scores.head())
fig = plt.figure(figsize=(27,9))

plt.subplot(1,3,1)

plt.title('Scatter plot of race scores vs year')

_ = sns.scatterplot(x='year', y='points', data=top_100)

plt.xticks(rotation=45)

plt.subplot(1,3,2)

plt.title('Box plot of the race score distribution per year')

_ = sns.boxplot(x='year', y='points', data=top_100)

plt.xticks(rotation='45')

plt.subplot(1,3,3)

plt.title('The amount of races our score dataset contains per year')

_ = sns.countplot(x='year', data=top_100)

plt.xticks(rotation='45')

plt.show()
df_scores = pd.concat([top_100, scores])

df_scores = df_scores.drop_duplicates(subset='raceId')

print(df_scores.head())
df_scores.sort_values('points', ascending=False).head(10).name.value_counts()
df_scores.sort_values('points', ascending=False).tail(10).name.value_counts()
print(df_scores.groupby('name').points.mean().sort_values(ascending=False))
print(results[results['raceId'] == 973][['resultId', 'raceId', 'driverId','grid', 'position', 'statusId']])
print(status[status['statusId'].isin([131, 4, 130])])
df = pd.merge(left=df_scores, right=races, on='raceId', how='left')

df = df.dropna(axis=0)

dropIndex = df[df['raceId'] > 988].index

df.drop(dropIndex, inplace=True)

print(df.columns)

print(len(df))
print(status[status['statusId'].isin([1, 11, 12, 13, 14, 15, 16, 17, 18, 19])])
def count_dnf(_results, _raceid):

    # Count the number of "did not finished" per race

    # for each result, there is a statusId and a collection of these signify that the driver

    # finished,

    dnf_count = 0

    finished_status = [1, 11, 12, 13, 14, 15, 16, 17, 18, 19]   # theses statusId are given when a driver finishes



    for iter_status in _results[_results['raceId'] == _raceid]['statusId']:

        if iter_status not in finished_status:

            dnf_count += 1

        else:

            pass



    return dnf_count
df['dnf'] = df.raceId.apply(lambda x: count_dnf(results, x))
fig = plt.figure(figsize=(18,9))

_ = sns.scatterplot(x='points', y='dnf', data=df)

pf = np.polyfit(x=df.points, y=df.dnf, deg=1)

_ = plt.plot(np.linspace(4, 10, len(df)), pf[0]*np.linspace(4, 10, len(df)) + pf[1], 'r')

plt.title('y = kx + m where k=' + str(round(pf[0],1)) + " and m=" + str(round(pf[1],1)))

plt.show()
dnf_points_p = df[['points','dnf']].corr()

print(dnf_points_p)
fig = plt.figure(figsize=(18,18))

plt.subplot(2,1,1)

_ = sns.lineplot(x='lap', y='position', hue='driverId', data=lapTimes[lapTimes['raceId'] == 973], palette="ch:2.5,.25")

plt.title('Driver standings during Barcelona 2017 Grand Prix')

plt.subplot(2,1,2)

_ = sns.lineplot(x='lap', y='position', hue='driverId', data=lapTimes[lapTimes['raceId'] == 864])

plt.title('Driver standings druing Barcelona 2012 Grand Prix')
def count_overtakings(laptimes, raceid):

    # Number of overtakings

    # The theory here is that when one driver changes his or her position between two adjacent laps, then an overtaking

    # has occurred. Counting the number of occurences this way, and then divide by 2 will give us the number of

    # overtakings since 1 overtaking includes one driver advancing one position, while the other loses one.



    competing_drivers = []

    for driver in laptimes[laptimes.raceId == raceid].driverId:

        if driver not in competing_drivers:

            competing_drivers.append(driver)



    previous_position = 0

    overtakings = 0

    for driver in competing_drivers:

        for lapPosition in laptimes[(laptimes.raceId == raceid) & (laptimes.driverId == driver)].position:

            if lapPosition != previous_position:

                previous_position = lapPosition

                overtakings += 1



    return int(overtakings/2)

df['overtakings'] = df.raceId.apply(lambda x: count_overtakings(lapTimes, x))
fig = plt.figure(figsize=(18,9))

_ = sns.scatterplot(x='points', y='overtakings', data=df)

pf = np.polyfit(x=df.points, y=df.overtakings, deg=1)

_ = plt.plot(np.linspace(4, 10, len(df)), pf[0]*np.linspace(4, 10, len(df)) + pf[1], 'r')

plt.title('y = kx + m where k=' + str(round(pf[0],1)) + " and m=" + str(round(pf[1],1)))
dnf_overtakings_p = df[['points','overtakings']].corr()

print(dnf_overtakings_p)
def get_top_5_battle(raceid, results, laptimes):

    # Focus in a race is usually on the drivers in the top, so a measurement of how their "battle" is taking shape

    # throughout the race could be interesting to measure.

    # We will approach this by looking at the variance in positions for the drivers who end up in top 5

    f_top_5 = results[(results.raceId == raceid) & (results.position < 6)].sort_values(['position'], ascending=True)

    f_top_5_var = []



    for f_driver in f_top_5.driverId:

        f_t5_var = np.var(laptimes[(laptimes.driverId == f_driver) & (laptimes.raceId == raceid)].position)

        f_top_5_var.append(f_t5_var)



    f_top5score = 0

    for f_itervar in f_top_5_var:

        f_top5score = f_top5score + f_itervar



    return f_top5score
df['top5_battle'] = df.raceId.apply(lambda x: get_top_5_battle(x, results, lapTimes))
fig = plt.figure(figsize=(18,9))

_ = sns.scatterplot(x='points', y='top5_battle', data=df)

pf = np.polyfit(x=df.points, y=df.top5_battle, deg=1)

_ = plt.plot(np.linspace(4, 10, len(df)), pf[0]*np.linspace(4, 10, len(df)) + pf[1], 'r')

plt.title('y = kx + m where k=' + str(round(pf[0],1)) + " and m=" + str(round(pf[1],1)))
dnf_top5_p = df[['points','top5_battle']].corr()

print(dnf_top5_p)
def get_rank_vs_position(raceid, results):

    # How are the drivers rank affecting the overall satisfaction score of a race?

    # We want to test and see if the rank of the drivers in top 5 affects how good a race is, in layman's terms:

    # If a low-ranked driver finished top 5, is it more worth than if a top ranked driver wins the rays?

    f_top_5 = results[(results.raceId == raceid) & (results.position < 6)].sort_values(['position'], ascending=True)



    rvp_score = 0

    for position, rank in zip(f_top_5['position'], f_top_5['rank']):

        rvp_score += abs(position - rank)



    return rvp_score
df['rvp'] = df.raceId.apply(lambda x: get_rank_vs_position(x, results))
fig = plt.figure(figsize=(18,9))

_ = sns.scatterplot(x='points', y='rvp', data=df)

pf = np.polyfit(x=df.points, y=df.rvp, deg=1)

_ = plt.plot(np.linspace(4, 10, len(df)), pf[0]*np.linspace(4, 10, len(df)) + pf[1], 'r')

plt.title('y = kx + m where k=' + str(round(pf[0],1)) + " and m=" + str(round(pf[1],1)))
dnf_rvp_p = df[['points','rvp']].corr()

print(dnf_rvp_p)
from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.metrics import mean_squared_error as MSE

import optuna # Library we use for hyperparameter tuning
X = df[['raceId','dnf','overtakings','top5_battle','rvp']]

y = df['points']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Hyperparameter tuning

def objective(trial):

    dtrain = xgb.DMatrix(X_train, label=y_train)

    dtest = xgb.DMatrix(X_test, label=y_test)

    

    op_params = {

        'gamma': trial.suggest_uniform('gamma', 0.1, 1),

        'learning_rate': trial.suggest_uniform('learning_rate', 0.1, 0.9),

        'max_depth': trial.suggest_int('max_depth', 1, 5),

        'n_estimators': trial.suggest_int('n_estimators', 1000, 10000),

        'col_sample_by_tree': trial.suggest_uniform('col_sample_by_tree', 0.1, 0.5),

        'booster': 'gbtree'

    }

    

    op_model = xgb.train(op_params, dtrain)

    op_preds = op_model.predict(dtest)

    

    return MSE(op_preds, y_test)



study = optuna.create_study()

study.optimize(objective, n_trials=30)
params = study.best_params

params['booster'] = 'gbtree'

dtrain = xgb.DMatrix(X_train, label=y_train)

model = xgb.train(params, dtrain)
untuned_model = xgb.XGBRegressor(seed=42)

untuned_model.fit(X_train, y_train)

untuned_predictions = untuned_model.predict(X_test)
dtest = xgb.DMatrix(X_test, label=y_test)

predictions = model.predict(dtest)
acc = MSE(predictions, y_test)

print("Tuned model MSE score: " + str(acc))

avg_sc = np.ones((len(predictions), 1))*df.points.mean()

baseline = MSE(avg_sc, y_test)

print("Baseline score: " + str(baseline))

untuned_acc = MSE(untuned_predictions, y_test)

print("Untuned model score: " + str(untuned_acc))
pred_compare = pd.DataFrame({

    'real': y_test.values,

    'predictions': predictions

})

pred_compare = pred_compare.sort_values('real')
x_sup = np.linspace(0,len(predictions), len(predictions))

fig = plt.figure(figsize=(18,9))

_ = plt.plot(x_sup, pred_compare['real'], 'bo', x_sup, pred_compare['predictions'], 'rx')

plt.ylim(4,10)

plt.legend(['Tuned Model Predictions', 'Real Scores'])
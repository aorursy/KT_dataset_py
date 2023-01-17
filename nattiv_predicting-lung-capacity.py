# eda

import numpy as np 

import pandas as pd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt



# modeling, model tuning, and model scoring

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import explained_variance_score

from yellowbrick.regressor import ResidualsPlot

import scipy.stats as stats

from statsmodels.stats.stattools import durbin_watson

from statsmodels.stats.diagnostic import het_white
df = pd.read_csv('/kaggle/input/lung-capacity-smoker-and-non-smoker/Lung-Capacity-Smoker.csv')
df.head()
df.info()
df.describe()
# rename columns so they're easier to work with

df.columns = ['lung_capacity', 'age', 'height', 'smoke', 'gender', 'caesarean']
# looking at correlation on current data

df.corr()
# create categories from quantitative variables

df['age_bin'] = pd.qcut(df.age, 5, labels=[0, 1, 2, 3, 4])

df['height_bin'] = pd.cut(df.height, 5, labels=[0, 1, 2, 3, 4])
for var in ['age', 'height', 'lung_capacity']:

    _ = sns.boxplot(x=var, data=df)

    plt.show()
_ = sns.boxplot(x='lung_capacity', y='gender', data=df)

plt.show()
_ = plt.plot()

plt.scatter(df.age, df.lung_capacity)

plt.xlabel("Age")

plt.ylabel("Lung Capacity")

plt.show()
_ = plt.plot()

plt.scatter(df.height, df.lung_capacity)

plt.xlabel("Height")

plt.ylabel("Lung Capacity")

plt.show()


# one-hot encode categories for quantitative variables

df = df.join(pd.get_dummies(df.age_bin, prefix='age_bin'))

df = df.join(pd.get_dummies(df.height_bin, prefix='height_bin'))

df = df.join(pd.get_dummies(df.gender, prefix='gender'))



# make boolean vars

df.smoke = df.smoke.apply(lambda x: True if x == 'yes' else False)

df.caesarean = df.caesarean.apply(lambda x: True if x == 'yes' else False)
# because i plan to make multiple models, i create a dictionary to store their data

results = {}
results[0] = {'description': 'one-hot encoding, feature elimination'}



final_df = df.drop(columns=['age', 'height', 'gender', 'gender_female', 'gender_male', 'smoke', 'caesarean'])



# train-test split

x_train, x_test, y_train, y_test = train_test_split(

    final_df.iloc[:, 1:], final_df.iloc[:, 0], test_size=.1, random_state=42)



results[0]['x_test'] = x_test

results[0]['y_test'] = y_test

results[0]['x_train'] = x_train

results[0]['y_train'] = y_train

results[0]["model"] = LinearRegression()

results[0]["model"].fit(results[0]["x_train"], results[0]["y_train"])



results[0]["predictions"] = results[0]["model"].predict(results[0]["x_test"])

results[0]["residuals"] = [true - predicted for true, predicted in zip(results[0]["y_test"].values, results[0]["predictions"].reshape(-1))]



results[0]["r2_train"] = results[0]['model'].score(results[0]['x_train'], results[0]['y_train'])

results[0]["r2_test"] = results[0]['model'].score(results[0]['x_test'], results[0]['y_test'])
results[1] = {"description": "one-hot encoding, all features"}



final_df = df.drop(columns=['age', 'height', 'gender'])



# train-test split

x_train, x_test, y_train, y_test = train_test_split(

    final_df.iloc[:, 1:], final_df.iloc[:, 0], test_size=.1, random_state=42)



results[1]['x_test'] = x_test

results[1]['y_test'] = y_test

results[1]['x_train'] = x_train

results[1]['y_train'] = y_train
results[1]["model"] = LinearRegression()

results[1]["model"].fit(results[1]["x_train"], results[1]["y_train"])



results[1]["predictions"] = results[1]["model"].predict(results[1]["x_test"])

results[1]["residuals"] = [true - predicted for true, predicted in zip(results[1]["y_test"].values, results[1]["predictions"].reshape(-1))]

results[1]["r2_train"] = results[1]['model'].score(results[1]['x_train'], results[1]['y_train'])

results[1]["r2_test"] = results[1]['model'].score(results[1]['x_test'], results[1]['y_test'])
results[2] = {"description": "raw variables, feature_elimination"}



# train-test split

x_train, x_test, y_train, y_test = train_test_split(

    df.iloc[:, 1:3], df.iloc[:, 0], test_size=.1, random_state=42)



results[2]['x_test'] = x_test

results[2]['y_test'] = y_test

results[2]['x_train'] = x_train

results[2]['y_train'] = y_train
results[2]["model"] = LinearRegression()

results[2]["model"].fit(results[2]["x_train"], results[2]["y_train"])



results[2]["predictions"] = results[2]["model"].predict(results[2]["x_test"])

results[2]["residuals"] = [true - predicted for true, predicted in zip(results[2]["y_test"].values, results[2]["predictions"].reshape(-1))]



results[2]["r2_train"] = results[2]['model'].score(results[2]['x_train'], results[2]['y_train'])

results[2]["r2_test"] = results[2]['model'].score(results[2]['x_test'], results[2]['y_test'])
df.columns
# instatiate results

results[3] = {"description": "raw variables, all features"}



final_df = df[['lung_capacity', 'age', 'height', 'smoke', 'caesarean', 'gender_male', 'gender_female']]



# train-test split

x_train, x_test, y_train, y_test = train_test_split(

    final_df.iloc[:, 1:], final_df.iloc[:, 0], test_size=.1, random_state=42)



results[3]['x_test'] = x_test

results[3]['y_test'] = y_test

results[3]['x_train'] = x_train

results[3]['y_train'] = y_train



# instatiate model

results[3]["model"] = LinearRegression()

            

# fit model              

results[3]["model"].fit(results[3]["x_train"], results[3]["y_train"])



# predict and get residuals              

results[3]["predictions"] = results[3]["model"].predict(results[3]["x_test"])

results[3]["residuals"] = [true - predicted for true, predicted in zip(results[3]["y_test"].values, results[3]["predictions"].reshape(-1))]



# score 

results[3]["r2_train"] = results[3]['model'].score(results[3]['x_train'], results[3]['y_train'])

results[3]["r2_test"] = results[3]['model'].score(results[3]['x_test'], results[3]['y_test'])
for key in results.keys():

    print(results[key]["description"], results[key]["r2_test"])
print(f"r^2 = {round(results[3]['r2_train'],4)} on the train set")

print(f"r^2 = {round(results[3]['r2_test'], 4)} on the test set")



print(f"mean of residuals is {round(np.mean(results[3]['residuals']), 4)}")

print(f"result of durbin-watson test is {durbin_watson(results[3]['residuals'])}")





lm_stat, lm_p, f_stat, f_p = het_white(results[3]['residuals'], results[3]['x_test'])

print(f"result of white test is {f_p}")
_ = plt.plot()

sns.residplot(results[3]["predictions"], results[3]["y_test"].values, lowess=True, line_kws={'color': 'red', 'lw': 1, 'alpha': 1})

plt.show()
plt.figure(figsize=(7,7))

stats.probplot(results[3]["residuals"], dist="norm", plot=plt)

plt.show()
for i, f in enumerate(results[3]["x_train"].columns): 

    coef = results[3]["model"].coef_[i]

    std = results[3]["x_train"][f].std()

    print(f"{f} {coef*std}")
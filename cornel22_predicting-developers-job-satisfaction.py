import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from scipy.stats import pearsonr

import seaborn as sns

%matplotlib inline
df_orig = pd.read_csv('/kaggle/input/so-survey-2017/survey_results_public.csv')

df_schema = pd.read_csv('/kaggle/input/so-survey-2017/survey_results_schema.csv')

df_orig.shape
df_orig.head()
df = df_orig





# What was the question Stack Overflow asked?



print("Question: " + df_schema[df_schema['Column'] == "JobSeekingStatus"]['Question'].tolist()[0])



# What are the possible answers for JobSeekingStatus?



print("Answers:")

print(df['JobSeekingStatus'].unique())
# Reduce Dataframe to professional, full-time developers



df = df.loc[df['EmploymentStatus'] == 'Employed full-time']

df = df.loc[df['Professional'] == 'Professional developer']

df = df.drop('Respondent', axis=1)
df.shape
# Delete rows without a JobSeekingStatus

df = df.dropna(subset=['JobSeekingStatus'], axis=0)



# Delete columns with only NaNs

df = df.dropna(how='all', axis=1)



df.shape
# Create two categories of developers: those who are not interested in a new job (1) and those who are (0)

X = df.drop('JobSeekingStatus', axis=1)

y = pd.get_dummies(df['JobSeekingStatus'], prefix="JobSearch")

y = y['JobSearch_I am not interested in new job opportunities']
# Fill the NaNs in numerical columns with the mean



num_cols = X.select_dtypes(include=['float','int']).columns



for col in num_cols:

    X[col].fillna(X[col].mean(), inplace=True)



# Create dummy columns for categorical columns (takes a while...)



cat_cols = X.select_dtypes(include=['object']).columns



for col in cat_cols:

    X = pd.concat([X.drop(col, axis=1), pd.get_dummies(X[col], prefix=col, drop_first=True)], axis=1)
# Do the prediction

    

# Step 1: test train sample



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



# Step 2: create and train a classifier (may take a while, too...)



clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train)



# Step 3: How well does it predict?



y_pred = clf.predict(X_test)

print("Classification report:")

print(classification_report(y_test, y_pred))
# What are the 10 most important features?



importances = clf.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(10):

    print("%d. %s (%f)" % (f + 1, X_train.columns[indices[f]-1], importances[indices[f]]))



# Plot the feature importances

plt.figure()

plt.title("Feature importances")

plt.bar(range(10), importances[indices[:10]], color="r", yerr=std[indices[:10]], align="center")

plt.xticks(range(10), X_train.columns[indices[:10]-1], rotation='vertical')

plt.show()
df = df_orig
# Again, reduce Dataframe to professional, full-time developers



df = df.loc[df['EmploymentStatus'] == 'Employed full-time']

df = df.loc[df['Professional'] == 'Professional developer']

df = df.drop(['Respondent','JobSeekingStatus','ExpectedSalary'], axis=1)
# What are possible values for job satisfaction?



df['JobSatisfaction'].unique()
# Delete rows with no value for job satisfaction or salary



df = df.dropna(subset=['JobSatisfaction','Salary'], axis=0)
# Are both features linear correlated?



corr, _ = pearsonr(df['Salary'], df['JobSatisfaction'])

print('Pearsons correlation: %.3f' % corr)
## Plot a correlation matrix for a deeper look

## Source: https://seaborn.pydata.org/examples/many_pairwise_correlations.html



# Compute the correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Calculate average job satisfaction and salary for major countries



df = df.dropna(subset=['JobSatisfaction','Salary'], axis=0)

major_countries = df['Country'].value_counts()[:15].keys()



sal_mean = []

sat_mean = []



for i in range(len(major_countries)):

    sat_mean.append(df.loc[df['Country'] == major_countries[i]]['JobSatisfaction'].mean())

    sal_mean.append(df.loc[df['Country'] == major_countries[i]]['Salary'].mean())
# Compare these values by a scatter plot



plt.title("Job Satisfaction and Salary for 15 Countries")

plt.xlabel("Avg. Salary")

plt.ylabel("Avg. Job Satisfaction")

plt.scatter(sal_mean, sat_mean)

plt.show()
# Are they linear correlated?



corr, _ = pearsonr(sal_mean, sat_mean)

print('Pearsons correlation: %.3f' % corr)
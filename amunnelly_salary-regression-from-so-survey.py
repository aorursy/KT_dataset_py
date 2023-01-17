import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import datetime

%matplotlib inline
df = pd.read_csv('../input/survey_results_public.csv', low_memory=False)
df.info()
df.index = df['Respondent']
del(df['Respondent'])
df.info()
numerical = []
text = []
for c in df.columns:
    if df[c].dtype == 'float64':
        numerical.append(c)
    elif df[c].dtype == 'int64':
        numerical.append(c)
    else:
        text.append(c)
shorter_columns = ['ConvertedSalary',
                    'Hobby',
                     'OpenSource',
                     'Country',
                     'Employment',
                     'FormalEducation',
                     'UndergradMajor',
                     'CompanySize',
                     'DevType',
                     'YearsCoding',
                     'YearsCodingProf',
                     'DatabaseWorkedWith',
                     'PlatformWorkedWith',
                     'FrameworkWorkedWith',
                     'OperatingSystem',
                     'Age']

df = df[shorter_columns]
df.info()
df = df[df.ConvertedSalary > 0]
df.info()
df.Country.value_counts()
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map
country_map = shorten_categories(df.Country.value_counts(), 400)
df['Country_Shorter'] = df.Country.map(country_map)
df.Country_Shorter.value_counts()/df.shape[0]
education_dict = {"Bachelor’s degree (BA, BS, B.Eng., etc.)": "Batchelor's",
                    "Some college/university study without earning a degree": "Some college",
                    "Master’s degree (MA, MS, M.Eng., MBA, etc.)": "Masters",
                    "Associate degree": "Associate Degree",
                    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "High School",
                    "Professional degree (JD, MD, etc.)": "Professional",
                    "Other doctoral degree (Ph.D, Ed.D., etc.)": "Doctoral",
                    "nan": "nan",
                    "Primary/elementary school": "Elementary",
                    "I never completed any formal education": "None"}
df['Education'] = df.FormalEducation.map(education_dict)
fig, ax = plt.subplots(1,1, figsize=(12, 7))
df.boxplot('ConvertedSalary', 'Education', ax=ax)
plt.suptitle('Salary v Formal Education')
plt.title('')
plt.ylabel('Salary ($)')
plt.xticks(rotation=90);
fig, ax = plt.subplots(1,1, figsize=(12, 7))
df.boxplot('ConvertedSalary', 'YearsCodingProf', ax=ax)
plt.suptitle('Salary (US$) v Years Coding Professionally')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90);
fig, ax = plt.subplots(1,1, figsize=(12, 7))
df[df.ConvertedSalary <=250000].boxplot('ConvertedSalary', 'YearsCodingProf', ax=ax)
plt.suptitle('Salary (US$) v Years Coding Professionally, outliers removed')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90);
fig, ax = plt.subplots(1,1, figsize=(12, 7))
df.boxplot('ConvertedSalary', 'Country_Shorter', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90);
fig, ax = plt.subplots(1,1, figsize=(12, 7))
df[df.ConvertedSalary <= 250000].boxplot('ConvertedSalary', 'Country_Shorter', ax=ax)
plt.suptitle('Salary (US$) v Country, Outliers Removed')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90);
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
df2 = df.copy()
df2 = df2[(df2.ConvertedSalary <= 250000) & (df2.Country_Shorter != 'Other') & (df2.Country_Shorter != 'United States')]
del(df2['Country_Shorter'])
df2.info()
labels = df2['ConvertedSalary']
features = df2.drop('ConvertedSalary', axis=1)
dummies = pd.get_dummies(features)
features = dummies
features.shape
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
for i in [train_features, test_features, train_labels, test_labels]:
    print(len(i), type(i))
param_grid = [{'kernel':('linear', 'rbf'), 'C':[1, 10]}]
regressor = SVR()
gridsearch = GridSearchCV(regressor, param_grid, scoring='neg_mean_squared_error')
from scipy.sparse import csr_matrix
train_regressor_ready_data = csr_matrix(train_features.values)
gridsearch.fit(train_regressor_ready_data, train_labels.values)
regressor = gridsearch.best_estimator_
train_predictions = regressor.predict(train_regressor_ready_data)
rootMeanSquaredError_train = np.sqrt(mean_squared_error(train_labels, train_predictions))
print("${:,.02f}".format(rootMeanSquaredError_train))
test_regressor_ready_data = csr_matrix(test_features.values)
test_predictions = regressor.predict(test_regressor_ready_data)
rootMeanSquaredError_test = np.sqrt(mean_squared_error(test_labels, test_predictions))
print("${:,.02f}".format(rootMeanSquaredError_test))
levelOfFit = abs(rootMeanSquaredError_train-rootMeanSquaredError_test)/rootMeanSquaredError_train*100.0
print("There is a {:.02f}% difference between the root mean squared errors of the train set and the test set.".format(levelOfFit))
df2.Country.value_counts()
plotting_df = pd.DataFrame(train_labels)
plotting_df['PredictedSalary'] = train_predictions
plotting_df['Country'] = train_features.index.map(df.Country)
plotting_df['Education'] = train_features.index.map(df.Education)
plotting_df['Experience'] = train_features.index.map(df.YearsCodingProf)
plotting_df.head()
len(plotting_df.Country.unique())
byCountry = plotting_df.groupby('Country')
colors = [plt.get_cmap('inferno')(1. * i/255) for i in range(0, 255, 15)]
countries = plotting_df.Country.unique().tolist()
fig, ax = plt.subplots(1, 1, figsize=(14, 14))
i = 0
for a, b in byCountry:
    plt.scatter(b.ConvertedSalary, b.PredictedSalary, color=colors[i], label=a, alpha=0.5)
    i +=1
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.legend()
plt.title('Predicted v Actual Salary');
fig, ax = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(12, 12))
counter = 0
for i in range(4):
    for j in range(4):
        temp = byCountry.get_group(countries[counter])
        ax[i][j].scatter(temp['ConvertedSalary'], temp['PredictedSalary'], color = colors[counter])
        ax[i][j].set_title(countries[counter])
        counter += 1
plt.tight_layout()
import scipy.stats as stats
def create_correlations_table(groupedDf, sample_size_cutoff=0):
    holder = []
    for a, b in groupedDf:
        if len(b) > sample_size_cutoff:
            if type(a) == str:
                category = a
            else:
                category = str(a)
            temp = stats.pearsonr(b['ConvertedSalary'], b['PredictedSalary'])
            RMSE = np.sqrt(mean_squared_error(b.ConvertedSalary, b.PredictedSalary))
            holder.append([a, len(b), temp[0], temp[1], RMSE])
        else:
            continue

    correlations = pd.DataFrame(holder, columns = ['Country', 'Sample Size', 'Pearson R', 'Probability', 'RMSE'])
    correlations.sort_values('RMSE', inplace=True)
    return correlations
country_correlations = create_correlations_table(byCountry)
country_correlations
byExperience = plotting_df.groupby('Experience')
years = plotting_df.Experience.dropna().unique().tolist()
colors = [plt.get_cmap('inferno')(1. * i/255) for i in range(0, 255, 21)]
fig, ax = plt.subplots(1, 1, figsize=(14, 14))
i = 0
for a, b in byExperience:
    plt.scatter(b.ConvertedSalary, b.PredictedSalary, color=colors[i], label=a, alpha=0.5)
    i +=1
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.legend()
plt.title('Predicted v Actual Salary');
fig, ax = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(12, 12))
counter = 0
for i in range(4):
    for j in range(3):
        if counter < len(years):
            temp = byExperience.get_group(years[counter])
            ax[i][j].scatter(temp['ConvertedSalary'], temp['PredictedSalary'], color = colors[counter])
            ax[i][j].set_title(years[counter])
            counter += 1
        else:
            pass
plt.tight_layout()
experience_correlations = create_correlations_table(byExperience)
experience_correlations
byCountryEx = plotting_df.groupby(['Country', 'Experience'])
country_experience_df = create_correlations_table(byCountryEx, 100)
country_experience_df

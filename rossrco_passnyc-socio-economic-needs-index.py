#numeric
import numpy as np
import pandas as pd
import scipy
from collections import defaultdict

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
import folium

from IPython.display import display

plt.style.use('bmh')
%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['axes.titlepad'] = 25
sns.set_color_codes('pastel')

#Pandas warnings
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 100)

#system
import os
import gc
import datetime
import re
#print(os.listdir('../input'))

#Machine learning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize, RobustScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn import tree
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_squared_log_error, explained_variance_score, mean_absolute_error
from sklearn.dummy import DummyRegressor
#import the dataset
shsat_res = pd.read_csv('../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv')

#rename columns to short, query-friendly names
shsat_res.rename(columns = {'School name' : 'school',
                            'Year of SHST' : 'year',
                            'Grade level' : 'grade',
                            'Enrollment on 10/31' : 'enrollment',
                            'Number of students who registered for the SHSAT' : 'num_registered',
                            'Number of students who took the SHSAT' : 'num_took'},
                 inplace = True)

#convert school name to lower caps string (used when joining other data sources)
shsat_res.school = shsat_res.school.str.lower()

#derive additional features
shsat_res['took_pct_registered'] = shsat_res['num_took'] / shsat_res['num_registered']
shsat_res['took_pct_registered'] = shsat_res['took_pct_registered'].fillna(0)

shsat_res['registered_pct_enrolled'] = shsat_res['num_registered'] / shsat_res['enrollment']
shsat_res['registered_pct_enrolled'] = shsat_res['registered_pct_enrolled'].fillna(0)
#plt.title('Number Of Students Who Registered / Took The SHSAT')
sns.jointplot(x = 'num_took', y = 'num_registered', size = 9, data = shsat_res)
sns.lmplot(x = 'num_took', y = 'num_registered', col = 'year', hue = 'grade', data = shsat_res)
fig = plt.figure(figsize = (20, 8))

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('Number Of Students Who Took The SHSAT by Year and Grade')
sns.boxplot(x = 'year', y = 'num_took', hue = 'grade', data = shsat_res)
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('Number Of Students Who Registered For The SHSAT by Year and Grade')
sns.boxplot(x = 'year', y = 'num_registered', hue = 'grade', data = shsat_res)
shsat_res['took_pct_registered_label'] = pd.cut(x = shsat_res.took_pct_registered, bins = 3, labels = ['low', 'medium', 'high'])
sns.lmplot(x = 'num_took', y = 'num_registered', col = 'took_pct_registered_label', hue = 'took_pct_registered_label', data = shsat_res)
#sns.lmplot(x = 'num_took', y = 'num_registered', col = 'year', hue = 'took_pct_registered_label', data = shsat_res)
#import the data
school_explorer = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv')

#rename 1 column with inconsistent capitalization
school_explorer.rename(columns = {'Grade 3 Math - All Students tested' : 'Grade 3 Math - All Students Tested'}, inplace = True)

#rename school column (to match the name of the corresponding column in the SHSAT results dataset)
school_explorer.rename(columns = {'School Name' : 'school'}, inplace = True)

#convert school name to lower caps string (used when joining other data sources)
school_explorer['school'] = school_explorer['school'].str.lower()

#convert the community school string to flag
school_explorer['Community School?'] = school_explorer['Community School?'].map({'Yes' : 1, 'No' : 0})

#convert the income estimate to numeric
school_explorer['School Income Estimate'] = school_explorer['School Income Estimate'].str.replace(',', '')
school_explorer['School Income Estimate'] = school_explorer['School Income Estimate'].str.replace('$', '')
school_explorer['School Income Estimate'] = pd.to_numeric(school_explorer['School Income Estimate'])

#convert percentage columns to numeric format
prc = re.compile(r'%', re.I)
prc_columns = pd.Series(school_explorer.loc[0].apply(lambda x: True if prc.search(str(x)) else False))
prc_columns = prc_columns[prc_columns == True].index.tolist()

for c in prc_columns:
    school_explorer[c] = school_explorer[c].str.replace('%', '')
    school_explorer[c] = pd.to_numeric(school_explorer[c])
    school_explorer[c] = school_explorer[c] / 100

#derive charter school flag
charter_school = re.compile(r'charter school' , re.I)
school_explorer['charter_school'] = np.vectorize(lambda x : 0 if charter_school.search(x) is None else 1)(school_explorer['school'].values)

#derive diversity percentage
school_explorer['diversity_prc'] = 1 - school_explorer['Percent White']

#discard grade columns
grade_cols = ['Grade 8 ELA 4s - American Indian or Alaska Native',
              'Grade 8 ELA 4s - Black or African American',
              'Grade 8 ELA 4s - Hispanic or Latino',
              'Grade 8 ELA 4s - Asian or Pacific Islander',
              'Grade 8 ELA 4s - White',
              'Grade 8 ELA 4s - Multiracial',
              'Grade 8 ELA 4s - Limited English Proficient',
              'Grade 8 ELA 4s - Economically Disadvantaged',
              'Grade 8 ELA 4s - All Students',
              'Grade 8 ELA - All Students Tested']

all_grade_cols = []
for g in range(3, 9):
    for c in grade_cols:
        for s in ['ELA', 'Math']:
            all_grade_cols.append(c.replace('8', '%s' % g).replace('ELA', '%s' % s))

school_explorer.drop(all_grade_cols, axis = 1, inplace = True)

#discrad other irrelevant columns
school_explorer.drop(['Adjusted Grade', 'Other Location Code in LCGMS', 'Grades',
                      'Grade Low', 'Grade High', 'Percent Black / Hispanic', 'Percent Asian',
                      'Percent Black', 'Percent Hispanic', 'Rigorous Instruction Rating',
                      'Collaborative Teachers Rating', 'Supportive Environment Rating',
                      'Effective School Leadership Rating',
                      'Strong Family-Community Ties Rating', 'Trust Rating',
                      'Student Achievement Rating'], axis = 1, inplace = True)
shsat_res = shsat_res.merge(school_explorer, how = 'left', left_on = 'school', right_on = 'school')
pd.pivot_table(data = shsat_res[(shsat_res.year == 2016) & (shsat_res.grade == 8)],
               index = 'took_pct_registered_label',
               values = ['Average ELA Proficiency', 'Average Math Proficiency'],
               aggfunc = ['mean', 'median']).style.format('{:.2}')
fig = plt.figure(figsize = (20, 18))

to_plot = shsat_res[(shsat_res.year == 2016) & (shsat_res.grade == 8)]

ax1 = fig.add_subplot(2, 2, 1)
sns.boxplot(x = 'took_pct_registered_label', y = 'Average Math Proficiency', data = to_plot)
ax2 = fig.add_subplot(2, 2, 2)
sns.boxplot(x = 'took_pct_registered_label', y = 'Average ELA Proficiency', data = to_plot)
ax3 = fig.add_subplot(2, 2, 3)
sns.violinplot(x = 'took_pct_registered_label', y = 'Average Math Proficiency', data = to_plot)
ax4 = fig.add_subplot(2, 2, 4)
sns.violinplot(x = 'took_pct_registered_label', y = 'Average ELA Proficiency', data = to_plot)
school_explorer_ind = ['Percent of Students Chronically Absent', 'Rigorous Instruction %',
                       'Economic Need Index', 'Student Attendance Rate', 'diversity_prc',
                       'Community School?', 'charter_school', 'Trust %',
                       'Collaborative Teachers %', 'Supportive Environment %',
                       'Effective School Leadership %', 'Strong Family-Community Ties %',
                       'School Income Estimate']

target_ind = ['took_pct_registered', 'Average Math Proficiency', 'Average ELA Proficiency']
to_plot = shsat_res[(shsat_res.year == 2016) & (shsat_res.grade == 8)]

shsat_res_corr = to_plot[target_ind + school_explorer_ind].corr()

box_style = dict(boxstyle = 'round', fc = (1.0, 0.7, 0.7), ec = 'none')
arrowprp = dict(arrowstyle = 'wedge,tail_width=1.',
                fc = (1.0, 0.7, 0.7),
                ec = 'none',
                patchA = None,
                patchB = None,
                relpos = (0.2, 0.5))

fig = plt.figure(figsize = (16, 26))
plt.subplots_adjust(hspace = 0.6)

for c, i in zip(target_ind, range(1, len(target_ind) + 1)):
    ax = fig.add_subplot(3, 1, i)
    to_plot = shsat_res_corr.loc[c, :].drop(target_ind)
    x_ticks = range(len(to_plot.index))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(to_plot.index, rotation = 45)
    ax.set_title('Correlation Between %s And The Socio-Academic Features In The School Explorer Dataset' % c)
    if c == 'took_pct_registered':
        ax.annotate(s = '''Diversity Prc is less influential
        for Took Pct Registered''',
              xy = (6, 0.3),
              xycoords = 'data',
              xytext = (55, 0),
              textcoords = 'offset points',
              size = 20,
              va = 'center',
              bbox = box_style,
              arrowprops = arrowprp)
        
        ax.annotate(s = '''Community School is more influential
        for Took Pct Registered''',
              xy = (5, -0.3),
              xycoords = 'data',
              xytext = (55, 0),
              textcoords = 'offset points',
              size = 20,
              va = 'center',
              bbox = box_style,
              arrowprops = arrowprp)
    
    ax.bar(x_ticks, to_plot)
#fig.autofmt_xdate()
fig = plt.figure(figsize = (30, 12))
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('Correlation Between The Features Of The D5 Dataset')
sns.heatmap(shsat_res_corr, annot = True, cmap = 'Spectral')
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('Correlation Between The Features Of The School Explorer Dataset')

target_ind.pop(target_ind.index('took_pct_registered'))
school_explorer_corr = school_explorer[target_ind + school_explorer_ind].corr()

sns.heatmap(school_explorer_corr, annot = True, cmap = 'Spectral')
pd.pivot_table(data = school_explorer, index = 'Community School?', columns = 'charter_school', values = 'school', aggfunc = 'count')
ny_crimes = pd.read_csv('../input/crimes-new-york-city/NYPD_Complaint_Data_Historic.csv', index_col = 0)
ny_precinct = pd.read_csv('../input/nyc-precinct-zip-codes/precinct_zip.csv')

#dates cannot be parsed because of inconsistent formatting
#we will derive a function that parses the dates where the format is month/day/year
#the rest of the rows will return not-any-date (nad)
def conv_date(date):
    try:
        return datetime.datetime.strptime(date, '%m/%d/%Y').date()
    except ValueError:
        return 'nad'
    except TypeError:
        return 'nad'

ny_crimes['date'] = ny_crimes.CMPLNT_FR_DT.apply(conv_date)

ny_crimes.drop(['CMPLNT_FR_TM', 'CMPLNT_TO_DT', 'CMPLNT_TO_TM',
                'RPT_DT', 'PD_CD', 'PD_DESC', 'LOC_OF_OCCUR_DESC',
                'PREM_TYP_DESC', 'PARKS_NM', 'HADEVELOPT',
                'X_COORD_CD', 'Y_COORD_CD', 'CMPLNT_FR_DT'],
               axis = 1,
               inplace = True)

ny_crimes = ny_crimes[ny_crimes.date != 'nad']

ny_crimes = ny_crimes.merge(ny_precinct, how = 'left', left_on = 'ADDR_PCT_CD', right_on = 'precinct')
ny_crimes.head()
m = folium.Map(location = [40.75, -73.85], tiles = 'cartodbpositron', zoom_start = 11.25)

to_plot = ny_crimes[(ny_crimes.date >= datetime.date(2015, 1, 1)) &
                    (ny_crimes.date <= datetime.date(2015, 12, 31)) &
                    (ny_crimes.Latitude.notna())].sample(500)

for i in range(0, len(to_plot)):
    if to_plot.iloc[i].LAW_CAT_CD == 'VIOLATION':
        c = 'blue'
    elif to_plot.iloc[i].LAW_CAT_CD == 'MISDEMEANOR':
        c = 'orange'
    else:
        c = 'red'
    
    folium.Circle(location = [to_plot.iloc[i].Latitude ,
                              to_plot.iloc[i].Longitude],
                  radius = 150,
                  color = c,
                  fill = True,
                  stroke = True,
                  fillOpacity = 0.2
   ).add_to(m)

mapWidth, mapHeight = (400, 500)
m
ny_crimes_by_zip = pd.pivot_table(data = ny_crimes[(ny_crimes.date >= datetime.date(2015, 1, 1)) & (ny_crimes.date <= datetime.date(2015, 12, 31))],
                                  index = 'zip',
                                  columns = 'LAW_CAT_CD',
                                  values = 'date',
                                  aggfunc = 'count',
                                  fill_value = 0)

ny_crimes_by_zip.head()
ny_crimes_by_zip.describe().style.format('{:.2}')
to_plot = ny_crimes_by_zip.describe().loc['std'] / (ny_crimes_by_zip.describe().loc['max'] - ny_crimes_by_zip.describe().loc['min'])
for c in to_plot.index:
    print('The standard deviation as percentage of the range for {} is: {:.2%}'.format(c, to_plot[c]))
print('''The number of unique ZIP codes in the NY Crimes dataset is: {}. \n
The number of unique ZIP codes in the school explorer dataset is: {}'''.format(len(ny_crimes_by_zip.index.unique()),
                                                                               len(school_explorer.Zip.unique())))
school_explorer = school_explorer.merge(ny_crimes_by_zip, how = 'left', left_on = 'Zip', right_index = True)

imp = SimpleImputer(strategy = 'median', copy = False)

columns_to_impute = ny_crimes_by_zip.columns.tolist() + ['School Income Estimate']

imputed_columns = pd.DataFrame(imp.fit_transform(school_explorer[columns_to_impute]), columns = columns_to_impute)
school_explorer.drop(columns_to_impute, axis = 1, inplace = True)
school_explorer = school_explorer.join(imputed_columns)
for c in columns_to_impute:
    school_explorer['pct_%s' % c] = school_explorer[c] / school_explorer[c].sum()
crime_ind = ['pct_FELONY']
school_explorer_corr = school_explorer[target_ind + crime_ind + school_explorer_ind].corr()
to_plot = school_explorer_corr.loc['pct_FELONY', :].sort_values(ascending = False).drop(crime_ind)#.transpose().iloc[:, 1:]

#fig = plt.figure(figsize = (14, 6))
x_ticks = range(len(to_plot.index))
plt.xticks(x_ticks, to_plot.index, rotation = 90)
plt.title('Correlation Between Percentage of Felonies Committed In a Certain ZIP Code \nAnd The Socio-Academic Features In The School Explorer Dataset')
plt.bar(x_ticks, to_plot)
us_household_income = pd.read_csv('../input/us-household-income-stats-geo-locations/kaggle_income.csv', encoding = 'ISO-8859-1')
us_household_income.rename(columns = {'Mean' : 'income_mean', 'Median' : 'income_median', 'Stdev' : 'income_stdev'}, inplace = True)
print('The number of ZIP codes from the school explorer dataset, not present in the US Household Income dataset is: {}'\
      .format(len(school_explorer[~(school_explorer.Zip.isin(us_household_income.Zip_Code))])))
us_household_income[us_household_income.State_ab == 'NY'].head()
us_household_income_med = pd.pivot_table(data = us_household_income[(us_household_income.State_ab == 'NY') & (us_household_income.income_mean != 0)],
                                         index = 'Zip_Code',
                                         values = ['income_mean', 'income_median', 'income_stdev'],
                                         aggfunc = 'mean')
columns_to_impute = ['income_mean', 'income_median', 'income_stdev']

school_explorer = school_explorer.merge(us_household_income_med, how = 'left', left_on = 'Zip', right_index = True)

imputed_columns = pd.DataFrame(imp.fit_transform(school_explorer[columns_to_impute]), columns = columns_to_impute)

school_explorer.drop(columns_to_impute, axis = 1, inplace = True)
school_explorer = school_explorer.join(imputed_columns)
income_ind = ['income_mean', 'income_median', 'income_stdev']
school_explorer_corr = school_explorer[target_ind + income_ind + crime_ind + school_explorer_ind].corr()

fig = plt.figure(figsize = (22, 9))
plt.subplots_adjust(wspace = 0.75)

for c, i in zip(income_ind, range(1, len(income_ind) + 1)):
    ax = fig.add_subplot(1, 3, i)
    to_plot = school_explorer_corr.loc[c, :].sort_values(ascending = False).drop(income_ind)
    x_ticks = range(len(to_plot.index))
    ax.set_yticks(x_ticks)
    ax.set_yticklabels(to_plot.index, rotation = 0)
    ax.set_title('Correlation Between %s \n And The Socio-Academic Features \n In The School Explorer Dataset' % c)
    ax.barh(x_ticks, to_plot)
ny_housing = pd.read_csv('../input/housing-new-york-units/housing-new-york-units-by-building.csv', parse_dates = ['Project Start Date', 'Project Completion Date'])
ny_housing[(ny_housing.Postcode.notna())].head()
ny_housing_by_zip_unit_type = pd.pivot_table(data = ny_housing[(ny_housing.Postcode.notna())],
                                             index = 'Postcode',
                                             values = ['Very Low Income Units', 'Low Income Units',
                                                       'Counted Rental Units', 'Counted Homeownership Units'],
                                             aggfunc = 'sum',
                                             fill_value = 0)

ny_housing_by_zip_constr_type = pd.pivot_table(data = ny_housing[ny_housing.Postcode.notna()],
                                               index = 'Postcode',
                                               columns = 'Reporting Construction Type',
                                               values = 'Total Units',
                                               aggfunc = 'sum',
                                               fill_value = 0)
ny_housing_by_zip_constr_type['total_units'] = ny_housing_by_zip_constr_type['New Construction'] + ny_housing_by_zip_constr_type['Preservation']

school_explorer = school_explorer.merge(ny_housing_by_zip_unit_type, how = 'left', left_on = 'Zip', right_index = True)
school_explorer = school_explorer.merge(ny_housing_by_zip_constr_type, how = 'left', left_on = 'Zip', right_index = True)
columns_to_impute = ['Counted Homeownership Units', 'Counted Rental Units',
                     'Low Income Units', 'Very Low Income Units',
                     'New Construction', 'Preservation', 'total_units']

imputed_columns = pd.DataFrame(imp.fit_transform(school_explorer[columns_to_impute]), columns = columns_to_impute)

school_explorer.drop(columns_to_impute, axis = 1, inplace = True)
school_explorer = school_explorer.join(imputed_columns)
school_explorer['pct_new_construction'] = school_explorer['New Construction'] / school_explorer['total_units']
school_explorer['pct_rental_units'] = school_explorer['Counted Rental Units'] / school_explorer['total_units']
school_explorer['pct_low_income_units'] = school_explorer['Low Income Units'] / school_explorer['total_units']
school_explorer['pct_very_low_income_units'] = school_explorer['Very Low Income Units'] / school_explorer['total_units']
housing_ind = ['pct_new_construction', 'pct_rental_units', 'pct_low_income_units', 'pct_very_low_income_units']
school_explorer_corr = school_explorer[target_ind +
                                       housing_ind +
                                       income_ind +
                                       crime_ind +
                                       school_explorer_ind].corr()

fig = plt.figure(figsize = (24, 8))
plt.subplots_adjust(wspace = 0.9)

for c, i in zip(housing_ind, range(1, len(housing_ind) + 1)):
    ax = fig.add_subplot(1, 4, i)
    to_plot = school_explorer_corr.loc[c, :].sort_values(ascending = False).drop(housing_ind)
    x_ticks = range(len(to_plot.index))
    ax.set_yticks(x_ticks)
    ax.set_yticklabels(to_plot.index, rotation = 0)
    ax.set_title('Correlation Between %s \n And The Socio-Academic Features \n In The School Explorer Dataset' % c)
    ax.barh(x_ticks, to_plot)

#for f in housing_features:
#    display(school_explorer_corr.sort_values(by = f, ascending = False)[[f]].transpose().iloc[:, 1:])
highschool_dir = pd.read_csv('../input/nyc-high-school-directory/2016-doe-high-school-directory.csv')

#rename school column (to match the name of the corresponding column in the school explorer results dataset)
highschool_dir.rename(columns = {'school_name' : 'school'}, inplace = True)

#convert school name to lower caps string (used when joining other data sources)
highschool_dir['school'] = highschool_dir['school'].str.lower()
highschool_dir.head()
#convert the shared space string to flag
highschool_dir['shared_space'] = highschool_dir['shared_space'].map({'Yes' : 1, 'No' : 0})

#derive number of transport methods for every school
highschool_dir['num_bus'] = highschool_dir['bus'].fillna('').apply(lambda x : len(x.split(',')))
highschool_dir['num_subway'] = highschool_dir['subway'].fillna('').apply(lambda x : len(x.split(';')))
school_explorer = school_explorer.merge(highschool_dir[['school', 'borough','shared_space', 'num_bus', 'num_subway']], how = 'left', on = 'school')
columns_to_impute = ['shared_space', 'num_bus', 'num_subway']

imputed_columns = pd.DataFrame(imp.fit_transform(school_explorer[columns_to_impute]), columns = columns_to_impute)

school_explorer.drop(columns_to_impute, axis = 1, inplace = True)
school_explorer = school_explorer.join(imputed_columns)
school_dir_ind = ['shared_space', 'num_bus', 'num_subway']
school_explorer_corr = school_explorer[target_ind +
                                       school_dir_ind +
                                       housing_ind +
                                       income_ind +
                                       crime_ind +
                                       school_explorer_ind].corr()

fig = plt.figure(figsize = (24, 10))
plt.subplots_adjust(wspace = 0.9)

for c, i in zip(school_dir_ind, range(1, len(school_dir_ind) + 1)):
    ax = fig.add_subplot(1, 4, i)
    to_plot = school_explorer_corr.loc[c, :].sort_values(ascending = False).drop(school_dir_ind)
    x_ticks = range(len(to_plot.index))
    ax.set_yticks(x_ticks)
    ax.set_yticklabels(to_plot.index, rotation = 0)
    ax.set_title('Correlation Between %s \n And The Socio-Academic Features \n In The School Explorer Dataset' % c)
    ax.barh(x_ticks, to_plot)


#for f in school_dir_features:
#    display(school_explorer_corr.sort_values(by = f, ascending = False)[[f]].transpose().iloc[:, 1:])
plt.title('Distribution Of The Average Math Proficiency')
sns.distplot(school_explorer[school_explorer['Average Math Proficiency'].notna()]['Average Math Proficiency'])
pred_cols = school_dir_ind + housing_ind + income_ind + crime_ind + school_explorer_ind

target_col1 = 'Average Math Proficiency'
target_col2 = 'Average ELA Proficiency'
scaler = MinMaxScaler((0, 1))

school_explorer = school_explorer[school_explorer[target_col1].notna()]

final_data = school_explorer[[target_col1] + pred_cols]

final_data = pd.DataFrame(scaler.fit_transform(final_data), columns = final_data.columns)
final_data = pd.DataFrame(normalize(final_data), columns = final_data.columns)

final_data.head()
final_data.describe()
models = {'ada_boost' : AdaBoostRegressor(),
          'linear_regression' : LinearRegression(),
          'lgbm' : LGBMRegressor(),
          'sgd_regressor' : SGDRegressor(),
          'decision_tree' : tree.DecisionTreeRegressor(),
          'random_forest' : RandomForestRegressor(),
          'dummy_0' : DummyRegressor(strategy = 'constant', constant = 0.),
          'dummy_0.5' : DummyRegressor(strategy = 'constant', constant = 0.5),
          'dummy_1' : DummyRegressor(strategy = 'constant', constant = 1.)}
metrics = {'mean_squared_error' : mean_squared_error,
           'r2_score' : r2_score,
           'mean_abs_error' : mean_absolute_error,
           'explained_variance_score' : explained_variance_score}
X_train, X_test, y_train, y_test = train_test_split(final_data[pred_cols],
                                                    final_data[target_col1],
                                                    test_size = 0.25,
                                                    random_state = 42)
n_folds = 3

scoring = ['neg_mean_absolute_error',
           'neg_mean_squared_error',
           'neg_median_absolute_error',
           'r2',
           'completeness_score',
           'explained_variance']

model_eval = defaultdict(list)
score_vis = defaultdict(list)

for m_name, reg in models.items():
    scores = cross_validate(reg, X_train, y_train, scoring = scoring, cv = n_folds, return_train_score = True)
    
    for k, v in scores.items():
        model_eval[k].append(v.mean())
        score_vis[k].extend(v)
    
    score_vis['model'].extend([m_name] * 3)

model_eval_res = pd.DataFrame(model_eval, index = models.keys()).transpose()
score_vis_res = pd.DataFrame(score_vis)

model_eval_res.style.format('{:.3f}')
fig = plt.figure(figsize = (20, 8))

to_plot = score_vis_res[~score_vis_res.model.isin(['dummy_0', 'dummy_0.5', 'dummy_1'])]

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('R2 Score For All Non-Dummy Models')
sns.boxplot(y = 'test_r2',
            x = 'model',
            data = to_plot,
            color = 'b')

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('Fit Time For All Models')
sns.boxplot(y = 'fit_time',
            x = 'model',
            data = to_plot,
            color = 'b')
reg = LinearRegression()

if isinstance(reg, SGDRegressor):
    hyperparam = {'average' : [True, False],
                  'alpha' : np.geomspace(1e-4, 1e2, num = 7)}
if isinstance(reg, AdaBoostRegressor):
    hyperparam = {'learning_rate' : np.geomspace(1e-4, 1e2, num = 7)}
if isinstance(reg, LinearRegression):
    hyperparam = {'normalize' : [True, False]}
if isinstance(reg, (RandomForestRegressor, LGBMRegressor)):
    hyperparam = {'max_depth' : [10, 50, 100, 500, 1000]}

final_reg = GridSearchCV(reg,
                         hyperparam,
                         scoring = scoring,
                         refit = 'r2')

final_reg.fit(X_train, y_train)
best_model = final_reg.best_estimator_
final_reg.best_params_
final_eval = defaultdict(list)

for metric, scorer in metrics.items():
    for set_type, set_data in zip(('train', 'test'), ([y_train, X_train], [y_test, X_test])):
        final_eval['metric'].append(metric + '_' + set_type) 
        final_eval['value'].append(scorer(set_data[0], best_model.predict(set_data[1])))

final_eval_res = pd.DataFrame(final_eval)
final_eval_res
#n_best = 15

if isinstance(best_model, (SGDRegressor, LinearRegression)):
    feature_importances = pd.Series(best_model.coef_, index = X_train.columns)
else:
    feature_importances = pd.Series(best_model.feature_importances_, index = X_train.columns)

feature_importances.sort_values(ascending = False)#[:n_best]
to_plot = feature_importances.sort_values(ascending = False)#[:n_best]

if isinstance(best_model, (SGDRegressor, LinearRegression)):
    plt.title('Feature Coefficients')
    plt.xlabel('feature coefficient')
else:
    plt.title('Most Important Features')
    plt.xlabel('feature importance')

y_ticks = range(len(to_plot))
plt.yticks(y_ticks, to_plot.index)
plt.barh(y_ticks, to_plot)
school_explorer['scioecon_need'] = 1 - best_model.predict(final_data[pred_cols])
school_explorer['scioecon_need_label'] = pd.cut(x = school_explorer.scioecon_need,
                                                bins = 5,
                                                labels = ['blue', 'green', 'yellow', 'orange', 'red'])

#[-1, 0.40, 0.55, 0.7, 0.85, 1]
#[-1, 0.20, 0.55, 0.8, 0.9, 1]
columns_to_plot = ['school', 'scioecon_need_label']
to_plot = pd.DataFrame(columns = columns_to_plot)
for l in ['red', 'orange', 'yellow', 'green', 'blue']:
    to_plot = to_plot.append(school_explorer[school_explorer['scioecon_need_label'] == l][columns_to_plot].head(3))
to_plot
m = folium.Map(location = [40.75, -73.85], tiles = 'cartodbpositron', zoom_start = 11.25)

to_plot = school_explorer

for i in range(0, len(to_plot)):
    c = to_plot.iloc[i].scioecon_need_label
    
    folium.Circle(location = [to_plot.iloc[i].Latitude ,
                              to_plot.iloc[i].Longitude],
                  radius = 150,
                  color = c,
                  fill = True,
                  stroke = True,
                  fillOpacity = 0.2
   ).add_to(m)

mapWidth, mapHeight = (400, 500)
m
shsat_res = shsat_res.merge(school_explorer[['school', 'scioecon_need', 'scioecon_need_label']], how = 'left', left_on = 'school', right_on = 'school')
to_plot = shsat_res[(shsat_res.year == 2016) & (shsat_res.grade == 8)]

plt.title('Economic Need Index Distribution Per Take As Percentage Of Registered Label')
sns.boxplot(y = 'scioecon_need', x = 'took_pct_registered_label', data = to_plot, color = 'b')
#export the results
school_explorer.to_csv('school_explorer_seni_labels.csv')
fig = plt.figure(figsize = (20, 22))

cols_to_plot = ['diversity_prc', 'Economic Need Index', 'income_mean', 'pct_low_income_units', 'pct_FELONY', 'income_stdev']

for c, i in zip(cols_to_plot, range(1, len(cols_to_plot) + 1) ):
    ax = fig.add_subplot(3, 2, i)
    ax.set_title('Distribution Of %s By Socio-Economic Label' % c)
    sns.boxplot(y = c, x = 'scioecon_need_label', data = school_explorer, color = 'b')
fig = plt.figure(figsize = (20, 18))

cols_to_plot = ['Percent of Students Chronically Absent', 'Supportive Environment %',
                'Rigorous Instruction %', 'Strong Family-Community Ties %']

for c, i in zip(cols_to_plot, range(1, len(cols_to_plot) + 1) ):
    ax = fig.add_subplot(2, 2, i)
    ax.set_title('Distribution Of %s By Socio-Economic Label' % c)
    sns.boxplot(y = c, x = 'scioecon_need_label', data = school_explorer, color = 'b')

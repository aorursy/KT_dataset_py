###### This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import cufflinks as cf
import scipy


import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import glob

# Any results you write to the current directory are saved as output.
import plotly
plotly.__version__
print(os.listdir("../input/estimate-of-median-household-income-group-series"))
hhi_files = glob.glob("../input/estimate-of-median-household-income-group-series/*.csv")
hhi_data = {os.path.basename(fname)[40:-4]:pd.read_csv(fname) for fname in hhi_files}
hhi_data.keys()
hhi_data['fairfax-county-va'].tail()
counties = list(hhi_data.keys())
hhi_2016 = pd.DataFrame(dict(
    county=counties,
    state=[c[-2:] for c in counties],
    value=[float(hhi_data[c].value.iloc[-1]) for c in counties]
)).sort_values("state")
hhi_2016
hhi_2016.state.unique()
region_look_up = {
    'ca': 'California',
    'co': 'Western',
    'fl': "Southern",
    'ga': "Southern",
    'il': "Midwestern",
    'md': "Northeastern",
    'mo': "Midwestern", 
    'nj': 'Northeastern',
    'ny': "Northeastern",
    'pa': "Northeastern",
    'tx': "Southern",
    'va': "Southern",
    'wa': "Western",
    'wi': "Midwestern"}
hhi_2016 = hhi_2016.assign(Region=[region_look_up[s] for s in hhi_2016.state])
hhi_2016

regional_median_income = hhi_2016.groupby("Region").mean()
regional_median_income

raw_region = pd.read_csv("../input/college-salaries/salaries-by-region.csv")
raw_college_type = pd.read_csv("../input/college-salaries/salaries-by-college-type.csv")
raw_degrees = pd.read_csv("../input/college-salaries/degrees-that-pay-back.csv")

print("Region:", raw_region.shape, raw_region.columns)
print("college_type:", raw_college_type.shape, raw_college_type.columns)
print("degrees:", raw_degrees.shape, raw_degrees.columns)
raw_degrees.head(3)
init_notebook_mode(connected=True)
cf.go_offline() #cufflinks links dataframes to plotly
iplot([{"x": [1, 2, 3], "y": [3, 1, 6]}])
iplot([go.Scatter(x=[1, 2, 3], y=[3, 1, 6])])
x = np.random.randn(2000)
y = np.random.randn(2000)
iplot([go.Histogram2dContour(x=x, y=y, contours=dict(coloring='heatmap')),
       go.Scatter(x=x, y=y, mode='markers', marker=dict(color='white', size=3, opacity=0.3))], show_link=False)
dir(go)
raw_degrees.head()
cols = [c for c in raw_degrees.columns if "Salary" in c and not "change" in c]
cols
degrees = raw_degrees
degrees[cols] = degrees[cols].replace({'\$': '',",": ''}, regex=True).astype(float) # stripped the characters and 
#converted to numerical value "float"
degrees.head()
raw_region.head()
cols = [c for c in raw_region.columns if "Salary" in c ]
cols
region = raw_region
region[cols] = region[cols].replace({'\$': '',",": ''}, regex=True).astype(float) # stripped the characters and 
#converted to numerical value "float"
region.head()
raw_college_type.head()
cols = [c for c in raw_college_type.columns if "Salary" in c ]
cols
college_type = raw_college_type
college_type[cols] = college_type[cols].replace({'\$': '',",": ''}, regex=True).astype(float) # stripped the characters and 
#converted to numerical value "float"
college_type.head()
data = college_type['Starting Median Salary']
iplot([go.Histogram(x=data)])
college_type["School Type"].value_counts()
college_type[college_type["School Type"]=="Ivy League"]
college_type[college_type["Starting Median Salary"]> 65e+3]
college_type.groupby("School Type").median().iplot(kind="bar")
region.groupby("Region").median().iplot(kind="bar")
columns = ["Undergraduate Major", "Starting Median Salary", "Mid-Career Median Salary"]
degrees[columns].sort_values("Mid-Career Median Salary").set_index("Undergraduate Major").iplot(
    kind='barh', subplots=False, bargap=.1, bargroupgap=.5,
    dimensions=(800, 1200), margin=dict(l=250, r=20)
)
college_type.columns
college_type_data = pd.DataFrame(dict(
        school_name=college_type['School Name'],
        school_type=college_type['School Type'],
        starting_salary=college_type['Starting Median Salary']))

print(college_type_data.shape)
college_type_data.replace([np.inf, -np.inf], np.nan).dropna().shape
college_type_data.head()

college_type_lm = ols('starting_salary ~ school_name+school_type', data=college_type_data).fit() #linear model
table = sm.stats.anova_lm(college_type_lm, typ=2) # Type 2 ANOVA DataFrame

print(table)
region.columns
#[regional_median_income.loc(r) for r in region.Region]
regional_median_income.loc['California'] 
college_region_data = pd.DataFrame(dict(
        school_name=region['School Name'],
        school_region=region['Region'],
        starting_salary=region['Starting Median Salary'],
        median_hh_income=[float(regional_median_income.loc[r]) for r in region.Region]))

print(college_region_data.shape)
print(college_region_data.replace([np.inf, -np.inf], np.nan).dropna().shape)
college_region_data.head()
college_region_lm = ols('starting_salary ~ median_hh_income', data=college_region_data).fit() #linear model
table = sm.stats.anova_lm(college_region_lm, typ=2) # Type 2 ANOVA DataFrame

print(table)
college_region_lm = ols('starting_salary ~ median_hh_income', data=college_region_data).fit() #linear model
college_region_school = ols('starting_salary ~ median_hh_income+school_name', data=college_region_data).fit()
college_region_school.compare_f_test(college_region_lm)
# (F-Statistic, p-value, increase in degrees of freedom)
college_region_plusfit = college_region_data.assign(resid=college_region_lm.resid)
college_region_plusfit.head()
college_region_plusfit_lm = ols('starting_salary ~ resid+school_region', data=college_region_plusfit).fit() #linear model
table = sm.stats.anova_lm(college_region_plusfit_lm, typ=2) # Type 2 ANOVA DataFrame

print(table)
#college_region_plusfit_lm.summary()
college_region_plusfit_lm.summary()
college_region_anova_lm = ols('starting_salary ~ school_region', data=college_region_plusfit).fit() #linear model
#Is salary different across the regions?
table = sm.stats.anova_lm(college_region_anova_lm, typ=2) # Type 2 ANOVA DataFrame

print(table)
college_region_anova_lm.summary()
hh_income_anova_lm = ols('median_hh_income ~ school_region', data=college_region_plusfit).fit() #linear model
#Is standard of living different across the regions?
table = sm.stats.anova_lm(hh_income_anova_lm, typ=2) # Type 2 ANOVA DataFrame

print(table)
hh_income_anova_lm.summary()
college_region_plusfit.head()
college_region_plusfit[['school_region','starting_salary','median_hh_income']].groupby("school_region").median().iplot(kind="bar")
college_salary_data = pd.DataFrame(dict(
        school_name=region['School Name'],
        school_region=region['Region'],
        starting_salary=region['Starting Median Salary'],
        mid_career_salary=region['Mid-Career Median Salary'],
        median_hh_income=[float(regional_median_income.loc[r]) for r in region.Region]))
college_salary_data.groupby("school_region").median().iplot(kind="bar")
binned_college_data = college_salary_data.groupby("school_region").median() #bin data by school region
scipy.stats.ks_2samp(binned_college_data.starting_salary,binned_college_data.median_hh_income) #KS test

print(normalized_college_salary_data.starting_salary-normalized_college_salary_data.median_hh_income)
normalized_college_salary_data = pd.DataFrame(dict(
    starting_salary=binned_college_data.starting_salary/binned_college_data.starting_salary.median(),
    median_hh_income= binned_college_data.median_hh_income/binned_college_data.median_hh_income.median(),
    mid_career_salary=binned_college_data.mid_career_salary/binned_college_data.mid_career_salary.median(),
)

)
print("Starting Salary Compared Against Median HH Income")

print(scipy.stats.ks_2samp(
   normalized_college_salary_data.starting_salary,
   normalized_college_salary_data.median_hh_income
)) #KS test



print("Starting Salary Compared Against Mid-Career Salary")

print(scipy.stats.ks_2samp(
   normalized_college_salary_data.starting_salary,
   normalized_college_salary_data.mid_career_salary
)) #KS test

print("Mid-Career Salary Compared Against Median HH Income")
print(scipy.stats.ks_2samp(
   normalized_college_salary_data.mid_career_salary,
   normalized_college_salary_data.median_hh_income
)) #KS test

#normalized_college_salary_data.groupby("school_region").median().iplot(kind="bar")
normalized_college_salary_data.iplot(kind="bar")
binned_college_data.head()
normalized_college_salary_data = pd.DataFrame(dict(
        school_name=region['School Name'],
        school_region=region['Region'],
        starting_salary=region['Starting Median Salary'],
        mid_career_salary=region['Mid-Career Median Salary'],
        median_hh_income=[float(regional_median_income.loc[r]) for r in region.Region]))

normalized_college_salary_data = normalized_college_salary_data.assign(
    starting_salary=normalized_college_salary_data.starting_salary/normalized_college_salary_data.starting_salary.median(),
    mid_career_salary=normalized_college_salary_data.mid_career_salary/normalized_college_salary_data.mid_career_salary.median(),
    median_hh_income=normalized_college_salary_data.median_hh_income/normalized_college_salary_data.median_hh_income.median()
)

print("Starting Salary Compared Against Median HH Income")

print(scipy.stats.ks_2samp(
   normalized_college_salary_data.starting_salary,
   normalized_college_salary_data.median_hh_income
)) #KS test



print("Starting Salary Compared Against Mid-Career Salary")

print(scipy.stats.ks_2samp(
   normalized_college_salary_data.starting_salary,
   normalized_college_salary_data.mid_career_salary
)) #KS test

print("Mid-Career Salary Compared Against Median HH Income")
print(scipy.stats.ks_2samp(
   normalized_college_salary_data.mid_career_salary,
   normalized_college_salary_data.median_hh_income
)) #KS test
normalized_college_salary_data.groupby("school_region").median().iplot(kind="bar")
[g for region,g in college_salary_data.groupby("school_region").mid_career_salary]
data = [
    go.Histogram(
        x=g,
        name=region,
        opacity=0.75,
        histnorm='percent',
    )
    for region,g in college_salary_data.groupby("school_region").mid_career_salary
    
]

layout = go.Layout(barmode='overlay')
fig = go.Figure(data=data, layout=layout)

iplot(fig)# filename='overlaid histogram')
#iplot([go.Histogram(x=data)])
data = [
    go.Histogram(
        x=g,
        name=region,
        opacity=0.75,
        histnorm='percent',
    )
    for region,g in college_salary_data.groupby("school_region").starting_salary
    
]

layout = go.Layout(barmode='overlay')
fig = go.Figure(data=data, layout=layout)

iplot(fig)# filename='overlaid histogram')
#iplot([go.Histogram(x=data)])

data = [
    go.Histogram(
        x=g,
        name=school_type,
        opacity=0.75,
        histnorm='percent',
    )
    for school_type,g in college_type.groupby("School Type")['Starting Median Salary']
    
]

layout = go.Layout(barmode='overlay')
fig = go.Figure(data=data, layout=layout)

iplot(fig)

data = [
    go.Histogram(
        x=g,
        name=school_type,
        opacity=0.75,
        histnorm='percent',
    )
    for school_type,g in college_type.groupby("School Type")['Mid-Career Median Salary']
    
]

layout = go.Layout(barmode='overlay')
fig = go.Figure(data=data, layout=layout)

iplot(fig)
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=25, random_state=0)
degrees.head()
region.head()
college_type.head()
college_salary_data.head()
merged_data_set = college_salary_data.merge(college_type[['School Name','School Type']],left_on='school_name',right_on='School Name')
merged_data_set.head()
x = merged_data_set[['school_region','School Type','median_hh_income']]
#y = 0+(merged_data_set.starting_salary > merged_data_set.median_hh_income.median())
# Starting Salary $50000
#y = 0+(merged_data_set.starting_salary > 50000)


# Starting Salary $60000
# y = 0+(merged_data_set.starting_salary > 60000)


#Starting Salary $70000
# y = 0+(merged_data_set.starting_salary > 70000)

#Starting Salary $75000
#y = 0+(merged_data_set.starting_salary > 75000)

#Starting Salary $80000
# y = 0+(merged_data_set.starting_salary > 80000)
# Mid-Career Salary $80000
#y = 0+(merged_data_set.mid_career_salary > 80000)


# Mid-Career Salary $90000
# y = 0+(merged_data_set.mid_career_salary > 90000)


# Mid-Career Salary $100000
# y = 0+(merged_data_set.mid_career_salary > 100000)

# Mid-Career Salary $110000
# y = 0+(merged_data_set.mid_career_salary > 110000)

#Mid-Career Salary $120000
# y = 0+(merged_data_set.mid_career_salary > 120000)

#Mid-Career Salary $130000
y = 0+(merged_data_set.mid_career_salary > 130000)
x_dummies = pd.get_dummies(x)
x_dummies.head()
model.fit(x_dummies,y)
import numpy as np
import matplotlib.pyplot as plt
importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)#[::-1]
print("Feature ranking:")

for f in range(x_dummies.shape[1]):
    # print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print("%d. %s (%f)" % (f + 1, x_dummies.columns[indices[f]], importances[indices[f]]))
plt.figure(figsize=(10,20))
plt.title("Feature importances")
plt.barh(range(x_dummies.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
x_labels = [x_dummies.columns[indices[f]] for f in range(x_dummies.shape[1])]
plt.yticks(range(x_dummies.shape[1]), x_labels)
plt.ylim([-1, x_dummies.shape[1]])
plt.show()


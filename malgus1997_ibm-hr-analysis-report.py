import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import statistics

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff



%matplotlib inline

data = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.shape
data.head()
data.columns
for i in data.columns:

    x = data[i]

    if len(set(x)) == 1:    

        data.drop(i, axis=1, inplace=True)

len(data.columns)
data.dtypes
data.isnull().sum().sum()
data.describe(percentiles = [.01,.1,.95,.99])
data.Attrition.value_counts()
labels = ['Leave','Stay']

values = [data.Attrition.value_counts().Yes, data.Attrition.value_counts().No]

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
Stay=data[data['Attrition'] == "No"]

Left=data[data['Attrition'] == "Yes"]
data["Age"].describe()
sns.distplot(data['Age']);
data.Age.groupby(data.Attrition).mean()
sns.distplot(Stay.Age.sample(200), hist = True, kde = True)

sns.distplot(Left.Age.sample(200), hist = True, kde = True)
sns.boxplot(x="Attrition", y="Age", data=data)
stats.ttest_ind(Stay.Age,Left.Age)
data.HourlyRate.describe()
data.HourlyRate.groupby(data.Attrition).mean()
sns.distplot(Stay.HourlyRate, hist = True, kde = True)

sns.distplot(Left.HourlyRate, hist = True, kde = True)
stats.ttest_ind(Stay.HourlyRate,Left.HourlyRate)
data.DistanceFromHome.describe()
data.DistanceFromHome.groupby(data.Attrition).mean()
sns.distplot(data['DistanceFromHome'])
# Test 

s1 = pd.Series([])

for i in range(100):

    random_distance_subset = data.DistanceFromHome.sample(n=100)

    s1=pd.concat([s1, random_distance_subset])

sns.distplot(s1)
sns.catplot(x="Attrition", y="DistanceFromHome",  kind="box", data=data);
sns.distplot(Stay.DistanceFromHome, hist = True, kde = True)

sns.distplot(Left.DistanceFromHome, hist = True, kde = True)
d1 = np.log(Stay.DistanceFromHome)

d2 = np.log(Left.DistanceFromHome)
stats.ttest_ind(d1,d2)
numeric_feature = ['Age','DailyRate','DistanceFromHome','EmployeeNumber','HourlyRate','MonthlyIncome','MonthlyRate','NumCompaniesWorked','PercentSalaryHike','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']

t_val = []

p_val = []

key_factor = []

for i in numeric_feature:

    t, p = stats.ttest_ind(Stay[i],Left[i])

    t_val.append(t)

    p_val.append(p)

    key_fact = 'No'

    if(p < 0.05):

        key_fact = 'Yes'

    key_factor.append(key_fact)

d = {'name': numeric_feature, 't_val': t_val, 'p_val': p_val, 'Is_keyfactor': key_factor}

df = pd.DataFrame(data=d)

df.sort_values(by=['Is_keyfactor'],ascending=False)
Att_JbS = pd.crosstab(data['Attrition'],data['JobSatisfaction'])

Att_JbS
x=['Leave', 'Stay']

fig = go.Figure(go.Bar(x=x, y=[18,27.8], name='Very High'))

fig.add_trace(go.Bar(x=x, y=[19,19.4], name='High'))

fig.add_trace(go.Bar(x=x, y=[30,30.8], name='Medium'))

fig.add_trace(go.Bar(x=x, y=[33,22], name='Low'))

fig.update_layout(barmode='stack')

fig.show()
chi2, p, dof, ex = stats.chi2_contingency(Att_JbS)

chi2, p
JL_Att = pd.crosstab(data['JobLevel'],data['Attrition'])

JL_Att
JL_Att['Yes1'] = (JL_Att['Yes']/ JL_Att['Yes'].sum())*100

JL_Att['No1'] = (JL_Att['No']/ JL_Att['No'].sum())*100

JL_Att
x=['Stay', 'Leave']

fig = go.Figure(go.Bar(x=x, y=[JL_Att['No1'][1],JL_Att['Yes1'][1]], name=1))

fig.add_trace(go.Bar(x=x, y=[JL_Att['No1'][2],JL_Att['Yes1'][2]], name=2))

fig.add_trace(go.Bar(x=x, y=[JL_Att['No1'][3],JL_Att['Yes1'][3]], name=3))

fig.add_trace(go.Bar(x=x, y=[JL_Att['No1'][4],JL_Att['Yes1'][4]], name=4))

fig.add_trace(go.Bar(x=x, y=[JL_Att['No1'][5],JL_Att['Yes1'][5]], name=5))



fig.update_layout(barmode='stack')

fig.show()
chi2, p, dof, ex = stats.chi2_contingency(pd.crosstab(data['Attrition'],data['JobLevel']))

chi2, p
categorical_feature = ['Attrition','BusinessTravel','Department','Education','EducationField','EnvironmentSatisfaction','Gender','JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus','OverTime','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','WorkLifeBalance']

chi2_val = []

p2_val = []

key_factor2 = []

for i in categorical_feature:

    chi2, p, dof, ex = stats.chi2_contingency(pd.crosstab(data[i],data['Attrition']))

    chi2_val.append(chi2)

    p2_val.append(p)

    key_fact2 = 'No'

    if(p < 0.05):

        key_fact2 = 'Yes'

    key_factor2.append(key_fact2)
import plotly.graph_objects as go



fig = go.Figure(data=[go.Table(

    header=dict(values=['Name', 'Chi2_value','p_val','Is_keyfactor'],

                line_color='darkslategray',

                fill_color='lightskyblue',

                align='left'),

    cells=dict(values=[categorical_feature, # 1st column

                       chi2_val,

                       p2_val,

                       key_factor2], # 2nd column

               line_color='darkslategray',

               fill_color='lightcyan',

               align='left'))

])



fig.update_layout(width=1000, height=600)

fig.show()
data.JobSatisfaction.value_counts()
labels = ['1','2','3','4']



values = [data.JobSatisfaction.value_counts()[1], data.JobSatisfaction.value_counts()[2],data.JobSatisfaction.value_counts()[3],data.JobSatisfaction.value_counts()[4]]

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
best_satis = data[data['JobSatisfaction'] == 4]

good_satis = data[data['JobSatisfaction'] == 3]

medium_satis= data[data['JobSatisfaction'] == 2]

bad_satis = data[data['JobSatisfaction'] == 1]
data.DistanceFromHome.groupby(data['JobSatisfaction']).mean()
sns.distplot(best_satis.DistanceFromHome, hist = True, kde = True)

sns.distplot(good_satis.DistanceFromHome, hist = True, kde = True)

sns.distplot(medium_satis.DistanceFromHome, hist = True, kde = True)

sns.distplot(bad_satis.DistanceFromHome, hist = True, kde = True)
stats.f_oneway(best_satis.DistanceFromHome, good_satis.DistanceFromHome, medium_satis.DistanceFromHome, bad_satis.DistanceFromHome)
data.DailyRate.groupby(data['JobSatisfaction']).mean()
sns.distplot(best_satis.DailyRate, hist = True, kde = True)

sns.distplot(good_satis.DailyRate, hist = True, kde = True)

sns.distplot(medium_satis.DailyRate, hist = True, kde = True)

sns.distplot(bad_satis.DailyRate, hist = True, kde = True)
stats.f_oneway(best_satis.DailyRate, good_satis.DailyRate, medium_satis.DailyRate, bad_satis.DailyRate)
f_val = []

p_anova = []

anova_key = []

for i in numeric_feature:

    f, p = stats.f_oneway(best_satis[i], good_satis[i], medium_satis[i], bad_satis[i])

    f_val.append(f)

    p_anova.append(p)

    key_fact = 'No'

    if(p < 0.05):

        key_fact = 'Yes'

    anova_key.append(key_fact)

d_anova = {'name': numeric_feature, 'f_val': f_val, 'p_val': p_anova, 'Is_keyfactor': anova_key}

df_anova = pd.DataFrame(data=d_anova)

df_anova
JL_JS = pd.crosstab(data['JobLevel'],data['JobSatisfaction'])

JL_JS
JL_JS['N1'] = (JL_JS[1]/ JL_JS[1].sum())*100

JL_JS['N2'] = (JL_JS[2]/ JL_JS[2].sum())*100

JL_JS['N3'] = (JL_JS[3]/ JL_JS[3].sum())*100

JL_JS['N4'] = (JL_JS[4]/ JL_JS[4].sum())*100
x=['Low','Medium','High','Very High']

fig = go.Figure(go.Bar(x=x, y=[JL_JS['N1'][1],JL_JS['N2'][1],JL_JS['N3'][1],JL_JS['N4'][1]], name=1))

fig.add_trace(go.Bar(x=x, y=[JL_JS['N1'][2],JL_JS['N2'][2],JL_JS['N3'][2],JL_JS['N4'][2]], name=2))

fig.add_trace(go.Bar(x=x, y=[JL_JS['N1'][3],JL_JS['N2'][3],JL_JS['N3'][3],JL_JS['N4'][3]], name=3))

fig.add_trace(go.Bar(x=x, y=[JL_JS['N1'][4],JL_JS['N2'][4],JL_JS['N3'][4],JL_JS['N4'][4]], name=4))

fig.add_trace(go.Bar(x=x, y=[JL_JS['N1'][5],JL_JS['N2'][5],JL_JS['N3'][5],JL_JS['N4'][5]], name=5))



fig.update_layout(barmode='stack')

fig.show()
chi2, p, dof, ex = stats.chi2_contingency(pd.crosstab(data['JobLevel'],data['JobSatisfaction']))

chi2, p
BT_JS = pd.crosstab(data['BusinessTravel'],data['JobSatisfaction'])

BT_JS
BT_JS['N1'] = (BT_JS[1]/ BT_JS[1].sum())*100

BT_JS['N2'] = (BT_JS[2]/ BT_JS[2].sum())*100

BT_JS['N3'] = (BT_JS[3]/ BT_JS[3].sum())*100

BT_JS['N4'] = (BT_JS[4]/ BT_JS[4].sum())*100

BT_JS

# BT_JS['N1'][0]
x=['Low','Medium','High','Very High']

fig = go.Figure(go.Bar(x=x, y=[BT_JS['N1'][0],BT_JS['N2'][0],BT_JS['N3'][0],BT_JS['N4'][0]], name='Non-Travel'))

fig.add_trace(go.Bar(x=x, y=[BT_JS['N1'][1],BT_JS['N2'][1],BT_JS['N3'][1],BT_JS['N4'][1]], name='Travel_frequently'))

fig.add_trace(go.Bar(x=x, y=[BT_JS['N1'][2],BT_JS['N2'][2],BT_JS['N3'][2],BT_JS['N4'][2]], name='Travel_Rarely'))



fig.update_layout(barmode='stack')

fig.show()
chi2, p, dof, ex = stats.chi2_contingency(pd.crosstab(data['BusinessTravel'],data['JobSatisfaction']))

chi2, p
chi2_val_js = []

p2_val_js = []

key_factor2_js = []

for i in categorical_feature:

    chi2, p, dof, ex = stats.chi2_contingency(pd.crosstab(data[i],data['JobSatisfaction']))

    chi2_val_js.append(chi2)

    p2_val_js.append(p)

    key_fact2_js = 'No'

    if(p < 0.05):

        key_fact2_js = 'Yes'

    key_factor2_js.append(key_fact2_js)

fig = go.Figure(data=[go.Table(

    header=dict(values=['Name', 'Chi2_value','p_val','Is_keyfactor'],

                line_color='darkslategray',

                fill_color='lightskyblue',

                align='left'),

    cells=dict(values=[categorical_feature, # 1st column

                       chi2_val_js,

                       p2_val_js,

                       key_factor2_js], # 2nd column

               line_color='darkslategray',

               fill_color='lightcyan',

               align='left'))

])



fig.update_layout(width=1000, height=600)

fig.show()
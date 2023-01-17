# load our necessary libary
import pandas as pd
from pandas import Series
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
%matplotlib inline
df_survey_results = pd.read_csv('../input/survey_results_public.csv', low_memory=False)
plt.figure(figsize=(10,8))

df_survey_results.WakeTime.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15))
plt.savefig('sta.png')
plt.show()
# plt.figure(figsize=(15, 8))
f,ax=plt.subplots(1,2,figsize=(25,10))

sns.countplot(x="WakeTime", hue="JobSatisfaction", data=df_survey_results, ax=ax[0])
sns.countplot(x='WakeTime', hue='CareerSatisfaction', data=df_survey_results, ax=ax[1])
ax[0].set_title('Effect of wake up time in job satisfaction', fontsize=18)
ax[1].set_title('Effect of wake up time in career satisfaction', fontsize=18)
ax[0].tick_params(axis='x', labelsize=18,rotation = 90)
ax[1].tick_params(axis='x', labelsize=18, rotation=90)
plt.setp(ax[0].get_legend().get_texts(), fontsize='18') # for legend text
plt.setp(ax[0].get_legend().get_title(), fontsize='20') # for legend title
plt.setp(ax[1].get_legend().get_texts(), fontsize='18') # for legend text
plt.setp(ax[1].get_legend().get_title(), fontsize='20') # for legend title
plt.show()
f,ax=plt.subplots(1,2,figsize=(25,10))

df_survey_results.Exercise.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax = ax[0])
df_survey_results.SkipMeals.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax = ax[1])
ax[0].tick_params(axis='y', labelsize=18)
ax[1].tick_params(axis='y', labelsize=18)
ax[0].set_title('How developers exercise in a week', fontsize=18)
ax[1].set_title('How developers skip their meals in a week', fontsize=18)
plt.show()
f,ax=plt.subplots(1,2,figsize=(25,10))

df_survey_results.HoursComputer.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax = ax[0])
df_survey_results.HoursOutside.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax = ax[1])
ax[0].tick_params(axis='y', labelsize=18)
ax[1].tick_params(axis='y', labelsize=18)
ax[0].set_title('Time spent on computer in a day', fontsize=18)
ax[1].set_title('Time spent outside in a day', fontsize=18)
plt.show()
plt.figure(figsize=(15,10))

df_survey_results.ErgonomicDevices.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15))
plt.title('Ergonomic devices use by developers', fontsize=18)
plt.yticks(fontsize=18)
plt.show()

f,ax=plt.subplots(2,4,figsize=(25,25))

from pandas import Series

s = df_survey_results['LanguageWorkedWith'].str.split(';').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'Language'
df_language = df_survey_results.join(s)
df_language.Language.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][0])



s = df_survey_results['DatabaseWorkedWith'].str.split(';').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'Database'
df_database = df_survey_results.join(s)
df_database.Database.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][1])

s = df_survey_results['PlatformWorkedWith'].str.split(';').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'Platform'
df_platform = df_survey_results.join(s)
df_platform.Platform.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][2])

s = df_survey_results['FrameworkWorkedWith'].str.split(';').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'Framework'
df_framework = df_survey_results.join(s)
df_framework.Framework.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][3])


s = df_survey_results['IDE'].str.split(';').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'ide'
df_ide = df_survey_results.join(s)
df_ide.ide.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][0])

s = df_survey_results['Methodology'].str.split(';').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'Working_Methodology'
df_methodology= df_survey_results.join(s)
df_methodology.Working_Methodology.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][1])

s = df_survey_results['VersionControl'].str.split(';').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'Version Control'
df_version= df_survey_results.join(s)
df_version['Version Control'].value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][2])


df_survey_results.OperatingSystem.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][3])






plt.subplots_adjust(wspace=0.8)
ax[0][0].set_title('Top programming languages used by our developers')
ax[0][1].set_title('Top Database solutions used by our developers')
ax[0][2].set_title('Top platforms used by our developers')
ax[0][3].set_title('Top framework used by our developers')
ax[1][0].set_title('Top IDE used by our developers')
ax[1][1].set_title('Top methodology used by our developers')
ax[1][2].set_title('Top Version control used by our developers')
ax[1][3].set_title('Top operating used by our developers')
plt.show()
MULTIPLE_CHOICE = [
    'DatabaseWorkedWith','PlatformWorkedWith',
    'PlatformDesireNextYear','Methodology','VersionControl','LanguageWorkedWith']
temp_df = df_survey_results[MULTIPLE_CHOICE]
# Go through all object columns
for c in MULTIPLE_CHOICE:
    
    # Check if there are multiple entries in this column
    temp = temp_df[c].str.split(';', expand=True)

    # Get all the possible values in this column
    new_columns = pd.unique(temp.values.ravel())
    for new_c in new_columns:
        if new_c and new_c is not np.nan:
            
            # Create new column for each unique column
            idx = temp_df[c].str.contains(new_c, regex=False).fillna(False)
            temp_df.loc[idx, f"{c}_{new_c}"] = 1

    # Info to the user
    print(f">> Multiple entries in {c}. Added {len(new_columns)} one-hot-encoding columns")

    # Drop the original column
    temp_df.drop(c, axis=1, inplace=True)
        
# For all the remaining categorical columns, create dummy columns
temp_df = pd.get_dummies(temp_df)
temp_df = temp_df.fillna(0)

use_features = [x for x in temp_df.columns if x.find('LanguageWorkedWith_') != -1]

def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    
    df = df[use_features]
    df.rename(columns=lambda x: x.split('_')[1], inplace=True)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    
plot_corr(temp_df)
plt.figure(figsize=(15,13))
df3 = df_survey_results.dropna(subset=['Student', 'HopeFiveYears'])
sns.heatmap(pd.crosstab(df3.HopeFiveYears, df3.Student))
plt.title('Plan for five years of our future developers')
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.ylabel('Hope for five years',fontsize=18)
plt.xlabel('Student',fontsize=18)
plt.show()
f,ax=plt.subplots(3,1,figsize=(10,25))

sns.heatmap(pd.crosstab(df_survey_results.FormalEducation, df_survey_results.Student), ax=ax[0])
sns.heatmap(pd.crosstab(df_survey_results.UndergradMajor, df_survey_results.Student), ax=ax[1])
sns.heatmap(pd.crosstab(df_survey_results.JobSearchStatus, df_survey_results.Student), ax=ax[2])
plt.subplots_adjust(wspace=0.8)
plt.show()
f,ax=plt.subplots(2,4,figsize=(25,25))
df_survey_results.StackOverflowRecommend.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][0])
df_survey_results.StackOverflowVisit.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][1])
df_survey_results.StackOverflowHasAccount.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][2])
df_survey_results.StackOverflowParticipate.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][3])
df_survey_results.StackOverflowJobs.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][0])
df_survey_results.StackOverflowDevStory.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][1])
df_survey_results.StackOverflowJobsRecommend.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][2])
df_survey_results.StackOverflowConsiderMember.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][3])




plt.subplots_adjust(wspace=0.8)
ax[0][0].set_title('Developers recommend Stackoverflow')
ax[0][1].set_title('Developers visit Stackoverflow')
ax[0][2].set_title('Developers have account on Stackoverflow')
ax[0][3].set_title('Developers participate in Stackoverflow')
ax[1][0].set_title('Developers using Stackoverflow job board')
ax[1][1].set_title('Developers using Stackoverflow Developer Story feature')
ax[1][2].set_title('Developers recommend Stackoverflow Jobs')
ax[1][3].set_title('Developers who are Stackoverflow members')
plt.show()
f,ax=plt.subplots(3,2,figsize=(25,30))

sns.heatmap(pd.crosstab(df_survey_results.StackOverflowDevStory, df_survey_results.StackOverflowParticipate), ax=ax[0][0])
sns.heatmap(pd.crosstab(df_survey_results.StackOverflowVisit, df_survey_results.JobSearchStatus), ax = ax[0][1])
sns.heatmap(pd.crosstab(df_survey_results.StackOverflowVisit, df_survey_results.Employment), ax= ax[1][0])
sns.heatmap(pd.crosstab(df_survey_results.StackOverflowVisit, df_survey_results.OpenSource), ax= ax[1][1])
sns.heatmap(pd.crosstab(df_survey_results.StackOverflowVisit, df_survey_results.Age), ax= ax[2][0])
sns.heatmap(pd.crosstab(df_survey_results.StackOverflowVisit, df_survey_results.AdBlocker), ax= ax[2][1])

ax[0][0].tick_params(axis='y', labelsize=18)
ax[0][0].tick_params(axis='x', labelsize=15)
ax[0][1].tick_params(axis='y', labelsize=18)
ax[0][1].tick_params(axis='x', labelsize=15)
ax[1][0].tick_params(axis='y', labelsize=18)
ax[1][0].tick_params(axis='x', labelsize=15)
ax[1][1].tick_params(axis='y', labelsize=18)
ax[1][1].tick_params(axis='x', labelsize=18)
ax[2][0].tick_params(axis='y', labelsize=18)
ax[2][0].tick_params(axis='x', labelsize=18)
ax[2][1].tick_params(axis='y', labelsize=18)
ax[2][1].tick_params(axis='x', labelsize=18)

plt.subplots_adjust(wspace=0.8, hspace=.99)
plt.show()

f,ax=plt.subplots(2,2,figsize=(25,25))
df_survey_results.EthicsChoice.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][0])
df_survey_results.EthicsReport.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[0][1])
df_survey_results.EthicsResponsible.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][0])
df_survey_results.EthicalImplications.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno_r',15), ax=ax[1][1])



plt.subplots_adjust(wspace=0.8)
ax[0][0].set_title('Response for writing code for a project that is unethical')
ax[0][1].set_title('Do developers want to report for the unethical project')
ax[1][0].set_title('Who is reponsible for code that is unethical')
ax[1][1].set_title('Do developers think about ethical purpose of their code')
plt.show()
# df_survey_results_without_nan_salry = df_survey_results.dropna(subset=['ConvertedSalary'])
# sns.distplot(df_survey_results_without_nan_salry.ConvertedSalary)

data_dem = df_survey_results[(df_survey_results['ConvertedSalary']>5000) & (df_survey_results['ConvertedSalary']<1000000)]

plt.subplots(figsize=(15,8))
sns.distplot(data_dem['ConvertedSalary'])
plt.title('Income histograms and fitted distribtion',size=15)
plt.show();
print('The median salary of developers: {} USD'.format(data_dem['ConvertedSalary'].median()
))
print('The mean salary of developers: {:0.2f} USD'.format(data_dem['ConvertedSalary'].mean()
))
plt.figure(figsize=(15,8))
sns.violinplot(x='ConvertedSalary', data=data_dem)
plt.title("Salary distribution of our developers", fontsize=16)
plt.xlabel("Annual Salary", fontsize=16)
plt.show();
temp=data_dem[data_dem.Gender.isin(['Male','Female'])]
plt.figure(figsize=(10,8))
sns.violinplot( y='ConvertedSalary', x='Gender',data=temp)
plt.title("Salary distribution Vs Gender", fontsize=16)
plt.ylabel("Annual Salary", fontsize=16)
plt.xlabel("Gender", fontsize=16)
plt.show();
resp_coun=df_survey_results['Country'].value_counts()[:15].to_frame()

f,ax=plt.subplots(1,1,figsize=(18,8))
max_coun=df_survey_results.groupby('Country')['ConvertedSalary'].median().to_frame()
max_coun=max_coun[max_coun.index.isin(resp_coun.index)]
max_coun.sort_values(by='ConvertedSalary',ascending=True).plot.barh(width=0.8,ax=ax,color=sns.color_palette('RdYlGn'))
ax.axvline(df_survey_results['ConvertedSalary'].median(),linestyle='dashed')
ax.set_title('Compensation of Top 15 Respondent Countries')
ax.set_xlabel('')
ax.set_ylabel('')
plt.subplots_adjust(wspace=0.8)
plt.show()

rates_ppp={'Countries':['United States','India','United Kingdom','Germany','France','Brazil','Canada','Spain','Australia','Russian Federation','Italy',"People 's Republic of China",'Netherlands', 'Sweden', 'Poland', 'Ukraine'],
           'Currency':['USD','INR','GBP','EUR','EUR','BRL','CAD','EUR','AUD','RUB','EUR','CNY','EUR', 'SEK', 'PLN', 'UAH'],
           'PPP':[1.00,17.7,0.7,0.78,0.81,2.05,1.21,0.66,1.46,25.13,0.74,3.51,0.8, 9.125, 1.782, 8.56],
          'exchange_rate': [1,67.56, 0.75, 0.85, 0.85, 3.74, 1.30, 0.85, 1.32, 62.37, 0.85, 6.41, 0.85, 8.72, 3.64, 26.16]}

rates_ppp = pd.DataFrame(data=rates_ppp)
rates_ppp
rates_ppp['PPP/MER']=rates_ppp['PPP']*rates_ppp['exchange_rate']

#keep the PPP/MER rates plus the 'Countries' column that will be used for the merge
rates_ppp
pd.set_option('display.float_format', lambda x: '%.3f' % x)

temp = df_survey_results.loc[df_survey_results['Country'].isin(rates_ppp.Countries)]
temp = temp.dropna(subset=['ConvertedSalary'])
temp = temp[(temp['ConvertedSalary']>500) & (temp['ConvertedSalary']<1000000)]
temp = temp.merge(rates_ppp,left_on='Country',right_on='Countries',how='left')[['Country', 'ConvertedSalary','PPP/MER', 'exchange_rate']]
temp['AdjustedSalary']=temp['ConvertedSalary']*temp['exchange_rate']/temp['PPP/MER']


d_salary = {}
for country in temp['Country'].value_counts().index :
    d_salary[country]=temp[temp['Country']==country]['AdjustedSalary'].median()
    
median_wages = pd.DataFrame.from_dict(data=d_salary, orient='index').round(2)
median_wages.sort_values(by=list(median_wages),axis=0, ascending=True, inplace=True)
ax = median_wages.plot(kind='barh',figsize=(15,8),width=0.7,align='center')
ax.legend_.remove()
ax.set_title("Adjusted incomes over the world",fontsize=16)
ax.set_xlabel("Amount", fontsize=14)
ax.set_ylabel("Country", fontsize=14)
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
    tick.set_fontsize(10)
plt.tight_layout()
plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))



max_coun.sort_values(by='ConvertedSalary',ascending=True).plot.barh(width=0.8,color=sns.color_palette('RdYlGn'), ax=ax[0])
ax[0].axvline(df_survey_results['ConvertedSalary'].median(),linestyle='dashed')
ax[0].set_title('Compensation of Top 15 Respondent Countries')
ax[0].set_xlabel('')
ax[0].set_ylabel('')


ax[1] = median_wages.plot(kind='barh',figsize=(15,8),width=0.7,align='center', ax=ax[1])
ax[1].legend_.remove()
ax[1].set_title("Adjusted incomes over the world",fontsize=16)
ax[1].set_xlabel("Amount", fontsize=14)
ax[1].set_ylabel("Country", fontsize=14)
for tick in ax[1].get_xticklabels():
    tick.set_rotation(0)
    tick.set_fontsize(10)
plt.tight_layout()
plt.show()
df_survey_predict = df_survey_results.copy()
df_survey_predict = df_survey_predict[['FormalEducation', 'YearsCodingProf', 'Age', 'Gender','ConvertedSalary']]
df_survey_predict = df_survey_predict.dropna()
df_survey_predict = df_survey_predict[df_survey_predict.Gender.isin(['Male','Female'])]
df_survey_predict = df_survey_predict[(df_survey_predict['ConvertedSalary']>100) & (df_survey_predict['ConvertedSalary']<1000000)]
df_survey_predict['YearsCodingProf'] = df_survey_predict['YearsCodingProf'].astype(str).str.replace(' years','').str.replace(' or more', '').str.split('-', expand=True).astype(float).mean(axis=1)
df_survey_predict['Age'] = df_survey_predict['Age'].astype(str).str.replace(' years old','').str.replace('Under ', '').str.replace(' years or older', '').str.split('-', expand=True).astype(float).mean(axis=1)
df_survey_predict['FormalEducation'] = df_survey_predict['FormalEducation'].str.replace(r"\(.*\)","")
df_survey_predict['FormalEducation'] = df_survey_predict['FormalEducation'].str.replace('Some college/university study without earning a degree','Some college W.E.D')
df_survey_predict['FormalEducation'] = df_survey_predict['FormalEducation'].str.replace('I never completed any formal education','Never completed')

df_survey_predict.head()
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols("ConvertedSalary ~ Age + Gender + FormalEducation + YearsCodingProf", data=df_survey_predict).fit()
model.summary()
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model, "YearsCodingProf", fig=fig)
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model, "Age", fig=fig)
fig, ax = plt.subplots(figsize=(12,8))
temp = df_survey_predict.groupby('FormalEducation')[['ConvertedSalary', 'YearsCodingProf', 'Age']].mean()
fig = sm.graphics.plot_partregress("ConvertedSalary", "Age", ["YearsCodingProf"],  ax=ax, data=temp)
from IPython.display import Image

Image("../input/databaseschema/DataBase_schema.png")
import os

print(os.listdir('../input/open-university-learning-analytics-dataset'))
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import os

import seaborn as sns





assessments = pd.read_csv('../input/open-university-learning-analytics-dataset/assessments.csv')

student_assessment = pd.read_csv('../input/open-university-learning-analytics-dataset/studentAssessment.csv')

student_info = pd.read_csv('../input/open-university-learning-analytics-dataset/studentInfo.csv')

vle_activity = pd.read_csv('../input/open-university-learning-analytics-dataset/studentVle.csv')



#we only need code_module so that we can perform a join with studenAssessment and then studentInfo tables

assessments.drop(['code_presentation','assessment_type','date','weight'], axis = 1, inplace = True)

#assessments.code_module.value_counts()
#merge individual assessments data with assessment data, so that we know which module each assessment belongs to

comb_assess = pd.merge(student_assessment,assessments,on='id_assessment')

comb_assess.drop(['is_banked','date_submitted'],axis = 1,inplace=True)

comb_assess.dtypes
#clean some of the data - we don't want score as a string, we want it as an integer. So let's remove the ?

#comb_assess.drop(comb_assess[comb_assess.score == '?'].index, inplace = True);

#comb_assess.score = comb_assess.score.astype(int);

#comb_assess.dtypes
#how much data do we have?

comb_assess.shape
#group by student and then subgroup by module, so that we retain ability to evaluate different module predictability

grouped = comb_assess.groupby(['id_student','code_module']).mean()

grouped.sort_values('id_student')

#we can't keep id_assessment, because we have just grouped by module. We have to group by module, as our vle interaction data is 

#only grouped by module

grouped.drop(['id_assessment'],axis=1,inplace = True)

grouped.head()
student_info.shape
#put it all together

student_all_info = pd.merge(student_info,grouped,on='id_student')

#just getting a feel for it - how many modules is each student enrolled in?

fig1 = student_all_info.groupby(['id_student']).code_module.count().sort_values().hist()

fig1.set_title('Number of modules by student')

fig1.set_xlabel('Number of modules')

fig1.set_ylabel('Number of students')
#total number of clicks per student by module

vle_grouped = vle_activity.groupby(['id_student','code_module']).sum()

#we have to drop the columns below as we have grouped by student and subgrouped by module, so they are meaningless

vle_grouped.drop(['id_site','date'],axis=1,inplace=True)

vle_grouped.head()

student_all_info.shape
#left join as we want to keep student info where no clicks were made

df = pd.merge(student_all_info,vle_grouped,on = ['id_student','code_module'],how='left')
df.shape
#remove rows where there are null values for sum_click. There are only 201 so it won't have a huge impact

df.dropna(inplace=True)
#save the new table

df.to_csv('joinedData.csv',index=False)
df.dtypes
df.code_module = pd.Categorical(df.code_module)

df.code_presentation = pd.Categorical(df.code_presentation)

df.gender = pd.Categorical(df.gender)

df.region = pd.Categorical(df.region)

df.highest_education = pd.Categorical(df.highest_education)

df.imd_band = pd.Categorical(df.imd_band)

df.age_band = pd.Categorical(df.age_band)

df.disability = pd.Categorical(df.disability)

df.final_result = pd.Categorical(df.final_result)
df.dtypes
df.head()
import plotly.express as px

data = df

fig = px.box(data, x="code_module", y="score",title='Student average scores by Module')



fig.show()
data = df

fig = px.box(data, x="region", y="score",title='Student average scores by Region')



fig.show()
highest_ed = df.highest_education.value_counts()

f, ax = plt.subplots(figsize=(18,5))

ax.bar(highest_ed.index,highest_ed)

ax.set_ylabel('number of students')

df.highest_education.cat.categories
for_bar = df.pivot_table(index = 'highest_education', columns='gender', values = 'score')

for_bar.plot(kind='bar')
df.head()
interaction_by_module = df.sum_click

fig2, ax2 = plt.subplots(figsize=(5,5));

ax2.hist(interaction_by_module,bins=50);

ax2.set_xlabel('Number of Clicks by module');

ax2.set_title('Number of clicks by module for each student');

ax2.set_ylabel('Number of occurences');

interaction_by_module = df.sum_click

fig2, ax2 = plt.subplots(figsize=(5,5));

ax2.boxplot(interaction_by_module);

ax2.set_xlabel('Number of Clicks by module');

ax2.set_title('Number of clicks by module for each student');

ax2.set_ylabel('Number of occurences');
plt.scatter((df.sum_click),(df.score))
plt.scatter((df.studied_credits),(df.score))
bins = [0,50,100,150,200,250,300,350,400,450,500,550,600]

df['studied_credits'] = pd.cut(df['studied_credits'], bins=bins)
df2 = df.groupby(['gender','code_module']).score.mean()
df3 = df.groupby(['gender','code_module']).sum_click.mean()

codes =  df2.index.get_level_values(1)

codes

sns.scatterplot(df3,df3.index.get_level_values(1), hue = df2.index.get_level_values(0), legend='full');
import seaborn as sns

codes =  df2.index.get_level_values(1)

codes

sns.scatterplot(df2,df2.index.get_level_values(1), hue = df2.index.get_level_values(0), legend='full');
df.groupby('final_result').sum_click.mean().sort_values().plot(kind='bar',)
data = df

fig = px.box(data, x="final_result", y="sum_click",title='Student final score by the total clicks on the vle')



fig.show()
df.groupby('final_result').score.mean().sort_values().plot(kind='bar',)
plt.scatter((df.sum_click[df.sum_click < 700]),(df.score[df.sum_click < 700]))
#Let's check the datatypes

df.dtypes
df.isnull().sum()

df.dropna(inplace=True)
df.describe()
df_target = df.score

df.drop(['score'],axis=1,inplace=True)

df.drop(['id_student'],axis=1,inplace=True)
df.head()

df.dtypes
#Check multicollinearity

corr = df.corr()

corr
df.hist()
df.num_of_prev_attempts = (df.num_of_prev_attempts - df.num_of_prev_attempts.mean())/df.num_of_prev_attempts.std()

df.sum_click = (df.sum_click - df.sum_click.mean())/df.sum_click.std()

df_target = (df_target - df_target.mean())/df_target.std()
df.hist();
df_target.hist();
#test skew and kurtosis

print("Kurtosis",df.kurtosis(axis=0))

print("Skew",df.skew(axis=0))

print("Target Kurtosis",df_target.kurtosis(axis=0))

print("Target Skew",df_target.skew(axis=0))

df.num_of_prev_attempts.value_counts()
df_trans = df

df_trans.head()

#df_trans['num_of_prev_attempts'] = np.log(df.num_of_prev_attempts)

#df_trans['studied_credits'] = np.log(df.studied_credits)

#df_trans['sum_click'] = np.log(df.sum_click)

#df_trans_target = np.log(df_target)
df_target.hist()

df_trans.hist()
df_trans = pd.get_dummies(df_trans)

df_trans.shape

df_trans.dtypes

for i in df_trans.columns[2:]:

    df_trans[i] = df_trans[i].astype('category')
df_trans.dtypes

df_trans['score']=df_target

plt.scatter(df_trans['sum_click'],df_target)

df_trans.sum_click.isnull().sum()

df_trans.shape
#replace spaces in strings with _ for modelling purposes

df_trans.columns = df_trans.columns.str.replace(' ', '_')

df_trans.columns = df_trans.columns.str.replace('-', '_')

df_trans.columns = df_trans.columns.str.replace('%', '')

df_trans.columns = df_trans.columns.str.replace('?', '')

df_trans.columns = df_trans.columns.str.replace('<', '')

df_trans.columns = df_trans.columns.str.replace('=', '')

df_trans.columns = df_trans.columns.str.replace(']', ')')

df_trans.columns = df_trans.columns.str.replace('(', '')

df_trans.columns = df_trans.columns.str.replace(')', '')

df_trans.columns = df_trans.columns.str.replace(',', '')



df_trans.head()
import statsmodels.formula.api as smf



dfcol = ['num_of_prev_attempts','sum_click']

result = []

for count, i in enumerate(dfcol):

    formula = 'score ~' + ' ' + i

    model = smf.ols(formula, data = df_trans).fit()

    #print(model.params[0],model.params[1],model.pvalues[1])

    result.append([i, model.rsquared, model.params[0],model.params[1],model.pvalues[1]])

result
df_trans.columns
cols_module= df_trans.columns[2:9]

cols_pres= df_trans.columns[9:13]

cols_gender = df_trans.columns[13:15]

cols_region = df_trans.columns[13:28]

cols_ed = df_trans.columns[28:33]

cols_imd = df_trans.columns[33:44]

cols_age = df_trans.columns[44:47]

cols_cred = df_trans.columns[47:59]

cols_dis = df_trans.columns[59:61]

cols_result = df_trans.columns[61:65]



print(cols_result)

cols = [cols_module, cols_pres , cols_gender, cols_region,cols_ed,cols_imd,cols_age,cols_cred,cols_dis,cols_result]

for col in cols:

    sum_cols = "+".join(col)

    form = "score ~" + sum_cols

    model = smf.ols(formula= form, data= df_trans).fit()

    #model = smf.ols(formula, data = df).fit()

    print(model.summary())
df_final = df_trans.drop(["num_of_prev_attempts","code_module_AAA","code_presentation_2013B","gender_F","region_East_Anglian_Region","highest_education_No_Formal_quals","imd_band_0_10","studied_credits_550_600","disability_Y","final_result_Fail"], axis=1)

y = df_final[["score"]]

X = df_final.drop(["score"], axis=1)
df_final.shape
import statsmodels.formula.api as smf

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression







r_list = []

adj_r_list = []

list_n = list(range(1,56,2))

for n in list_n: 

    linreg = LinearRegression()

    select_n = RFE(linreg, n_features_to_select = n)

    select_n = select_n.fit(X, np.ravel(y))

    selected_columns = X.columns[select_n.support_ ]

    linreg.fit(X[selected_columns],y)

    yhat = linreg.predict(X[selected_columns])

    SS_Residual = np.sum((y-yhat)**2)

    SS_Total = np.sum((y-np.mean(y))**2)

    r_squared = 1 - (float(SS_Residual))/SS_Total

    print(r_squared)

    adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)

    print(adjusted_r_squared)

r_list.append(r_squared)

adj_r_list.append(adjusted_r_squared)
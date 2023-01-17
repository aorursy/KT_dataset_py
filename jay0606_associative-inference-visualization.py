import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from scipy import stats



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/associativeinference/Associative Inference.csv")
data.head(5)
data.columns
age_group = data[["AgeGroup", "AB1-BC1"]]

age_group.head()



grouped = age_group.groupby('AgeGroup', as_index=False)

summary = grouped.mean()

summary



temp = grouped.describe()["AB1-BC1"][["count", 'std']]

temp



summary["std_err"] = temp["std"] / np.sqrt(temp["count"])

summary = summary.reindex([1,0,3,2])



fig = px.bar(summary, x="AgeGroup", y="AB1-BC1", error_y="std_err", width=500, title="AB1-BC1")

fig.show()
age_group = data[["AgeGroup", 'AB_acc_1', 'AB_acc_2', 'AB_acc_3', 'AB_acc_4', 

                  'BC_acc_1', 'BC_acc_2', 'BC_acc_3', 'BC_acc_4']]

age_group.head()



grouped = age_group.groupby('AgeGroup', as_index=False)

summary = grouped.mean()

summary



col_names = ['AB_acc_1', 'AB_acc_2', 'AB_acc_3', 'AB_acc_4', 'BC_acc_1', 'BC_acc_2', 'BC_acc_3', 'BC_acc_4']





# add standard error for each accuracy

for name in col_names:

    temp = grouped.describe()[name][["count", 'std']]

    summary["se_" + name[:2] + name[-1]] = temp["std"] / np.sqrt(temp["count"])

    

# get AB, BC values from AgeGroup

child_AB = summary[summary["AgeGroup"] == "child"][['AB_acc_1', 'AB_acc_2', 'AB_acc_3', 'AB_acc_4']].values[0]

child_BC = summary[summary["AgeGroup"] == "child"][['BC_acc_1', 'BC_acc_2', 'BC_acc_3', 'BC_acc_4']].values[0]



ado_AB = summary[summary["AgeGroup"] == "adolescent"][['AB_acc_1', 'AB_acc_2', 'AB_acc_3', 'AB_acc_4']].values[0]

ado_BC = summary[summary["AgeGroup"] == "adolescent"][['BC_acc_1', 'BC_acc_2', 'BC_acc_3', 'BC_acc_4']].values[0]



y_adl_AB = summary[summary["AgeGroup"] == "younger adult"][['AB_acc_1', 'AB_acc_2', 'AB_acc_3', 'AB_acc_4']].values[0]

y_adl_BC = summary[summary["AgeGroup"] == "younger adult"][['BC_acc_1', 'BC_acc_2', 'BC_acc_3', 'BC_acc_4']].values[0]



o_adl_AB = summary[summary["AgeGroup"] == "older adult"][['AB_acc_1', 'AB_acc_2', 'AB_acc_3', 'AB_acc_4']].values[0]

o_adl_BC = summary[summary["AgeGroup"] == "older adult"][['BC_acc_1', 'BC_acc_2', 'BC_acc_3', 'BC_acc_4']].values[0]



# get standard error for AB, BC values from AgeGroup

child_AB_se = summary[summary["AgeGroup"] == "child"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

child_BC_se = summary[summary["AgeGroup"] == "child"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



ado_AB_se = summary[summary["AgeGroup"] == "adolescent"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

ado_BC_se = summary[summary["AgeGroup"] == "adolescent"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



y_adl_AB_se = summary[summary["AgeGroup"] == "younger adult"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

y_adl_BC_se = summary[summary["AgeGroup"] == "younger adult"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



o_adl_AB_se = summary[summary["AgeGroup"] == "older adult"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

o_adl_BC_se = summary[summary["AgeGroup"] == "older adult"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]





# make 2x2 subplots

fig = make_subplots(rows=2, cols=2,

                    subplot_titles=("Children", "Adolescent", "Younger Adults", "Older Adults"))



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=child_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=child_AB_se)),  row=1, col=1)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=child_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=child_BC_se)),row=1, col=1)   



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=ado_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=ado_AB_se)),row=1, col=2)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=ado_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=ado_BC_se)),row=1, col=2)  



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=y_adl_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=y_adl_AB_se)),row=2, col=1)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=y_adl_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=y_adl_BC_se)),row=2, col=1)  



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=o_adl_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=o_adl_AB_se)),row=2, col=2)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=o_adl_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=o_adl_BC_se)),row=2, col=2)  



fig.update_layout(width=850, height=600)

fig.show()
age_group = data[["AgeGroup", 'AB_rt_1', 'AB_rt_2', 'AB_rt_3', 'AB_rt_4', 

                  'BC_rt_1', 'BC_rt_2', 'BC_rt_3', 'BC_rt_4']]

age_group.head()



grouped = age_group.groupby('AgeGroup', as_index=False)

summary = grouped.mean()

summary



col_names = ['AB_rt_1', 'AB_rt_2', 'AB_rt_3', 'AB_rt_4', 'BC_rt_1', 'BC_rt_2', 'BC_rt_3', 'BC_rt_4']





# add standard error for each accuracy

for name in col_names:

    temp = grouped.describe()[name][["count", 'std']]

    summary["se_" + name[:2] + name[-1]] = temp["std"] / np.sqrt(temp["count"])

    

# get AB, BC values from AgeGroup

child_AB = summary[summary["AgeGroup"] == "child"][['AB_rt_1', 'AB_rt_2', 'AB_rt_3', 'AB_rt_4']].values[0]

child_BC = summary[summary["AgeGroup"] == "child"][['BC_rt_1', 'BC_rt_2', 'BC_rt_3', 'BC_rt_4']].values[0]



ado_AB = summary[summary["AgeGroup"] == "adolescent"][['AB_rt_1', 'AB_rt_2', 'AB_rt_3', 'AB_rt_4']].values[0]

ado_BC = summary[summary["AgeGroup"] == "adolescent"][['BC_rt_1', 'BC_rt_2', 'BC_rt_3', 'BC_rt_4']].values[0]



y_adl_AB = summary[summary["AgeGroup"] == "younger adult"][['AB_rt_1', 'AB_rt_2', 'AB_rt_3', 'AB_rt_4']].values[0]

y_adl_BC = summary[summary["AgeGroup"] == "younger adult"][['BC_rt_1', 'BC_rt_2', 'BC_rt_3', 'BC_rt_4']].values[0]



o_adl_AB = summary[summary["AgeGroup"] == "older adult"][['AB_rt_1', 'AB_rt_2', 'AB_rt_3', 'AB_rt_4']].values[0]

o_adl_BC = summary[summary["AgeGroup"] == "older adult"][['BC_rt_1', 'BC_rt_2', 'BC_rt_3', 'BC_rt_4']].values[0]



# get standard error for AB, BC values from AgeGroup

child_AB_se = summary[summary["AgeGroup"] == "child"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

child_BC_se = summary[summary["AgeGroup"] == "child"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



ado_AB_se = summary[summary["AgeGroup"] == "adolescent"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

ado_BC_se = summary[summary["AgeGroup"] == "adolescent"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



y_adl_AB_se = summary[summary["AgeGroup"] == "younger adult"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

y_adl_BC_se = summary[summary["AgeGroup"] == "younger adult"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



o_adl_AB_se = summary[summary["AgeGroup"] == "older adult"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

o_adl_BC_se = summary[summary["AgeGroup"] == "older adult"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]





# make 2x2 subplots

fig = make_subplots(rows=2, cols=2,

                    subplot_titles=("Children", "Adolescent", "Younger Adults", "Older Adults"))



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=child_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=child_AB_se)),  row=1, col=1)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=child_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=child_BC_se)),row=1, col=1)   



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=ado_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=ado_AB_se)),row=1, col=2)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=ado_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=ado_BC_se)),row=1, col=2)  



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=y_adl_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=y_adl_AB_se)),row=2, col=1)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=y_adl_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=y_adl_BC_se)),row=2, col=1)  



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=o_adl_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=o_adl_AB_se)),row=2, col=2)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=o_adl_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=o_adl_BC_se)),row=2, col=2)  



fig.update_layout(width=850, height=600)

fig.show()
# create a copy of data

data_new = data.copy()

data_new["AB_acc_4_T"] = data["AB_acc_4"] == 1

data_new["AB_acc_4_T"] = data_new["AB_acc_4_T"].apply(lambda x: "correct" if x else "incorrect")
data_new["AB_acc_4_T"].value_counts()
data_new[data_new["AB_acc_4_T"] == "correct"][["BC_acc_4"]].describe()
data_new[data_new["AB_acc_4_T"] == "incorrect"][["BC_acc_4"]].describe()
fig = px.histogram(data_new, x="BC_acc_4", color="AB_acc_4_T", marginal="rug", barmode="overlay", 

                   width=700, title="BC_acc_4 distribution between AB_acc_4 = 1 or not 1")

fig.show()
data_new = data.copy()

data_new["BC1-AC"] = data_new['BC_acc_1'] - data_new['AC_acc']



age_group = data_new[["AgeGroup", "BC1-AC"]]

age_group.head()



grouped = age_group.groupby('AgeGroup', as_index=False)

summary = grouped.mean()



temp = grouped.describe()["BC1-AC"][["count", 'std']]

temp



summary["std_err"] = temp["std"] / np.sqrt(temp["count"])

summary = summary.reindex([1,0,3,2])

print(summary)



fig = px.bar(summary, x="AgeGroup", y="BC1-AC", error_y="std_err", width=500, title="(BC rep1 - AC rep1) x Age")

fig.show()
def age_ac_plot(x, y, gender):

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

    line = slope * x + intercept



    trace1 = go.Scatter(

                      x=x,

                      y=y,

                      mode='markers',

                      name=gender

                      )

    trace2 = go.Scatter(

                      x=x,

                      y=line,

                      mode='lines',

                      name='regression line',

                      )

    

    traces = [trace1, trace2]

    fig = go.Figure(data=traces)

    fig.update_layout(width=700)

    fig.update_layout(title_text=f'R^2 = {r_value*r_value:.3f}, p={p_value:.4f}', 

                      xaxis_title="Age",

                      yaxis_title="AC conditional accuracy")

    fig.show();
older_data = data[data["Age"] >= 60 ]



older_m_data = older_data[older_data["Sex"] == "M"]

older_f_data = older_data[older_data["Sex"] == "F"]



young_data = data[data["Age"] < 60 ]



young_m_data = young_data[young_data["Sex"] == "M"]

young_f_data = young_data[young_data["Sex"] == "F"]
fig = px.scatter(older_data, x="Age", y="AC_acc", width=600, color="Sex")

fig.update_layout(title_text="AC_acc of older people")

fig.show()
age_ac_plot(older_data["Age"], older_data["AC_acc"], "Male + Female")
age_ac_plot(older_m_data["Age"], older_m_data["AC_acc"], "Male")
age_ac_plot(older_f_data["Age"], older_f_data["AC_acc"], "Female")
fig = px.scatter(young_data, x="Age", y="AC_acc", width=600, color="Sex")

fig.update_layout(title_text="AC_acc of young people") 

fig.show()
age_ac_plot(young_data["Age"], young_data["AC_acc"], "Male + Female")
age_ac_plot(young_m_data["Age"], young_m_data["AC_acc"], "Male")
age_ac_plot(young_f_data["Age"], young_f_data["AC_acc"], "Female")
data.columns
data_new = data.copy()

data_new["is_ac_old_rt"] = data["Age"] >= 60



fig = px.histogram(data_new, x="AC_rt", color="is_ac_old_rt", marginal="rug", barmode="overlay", 

                   width=700, title="AC reaction time between young and old")

fig.show()
#Create a boxplot

data_new.boxplot("AC_rt", by="is_ac_old_rt", figsize=(10, 6));
# Anova

import statsmodels.api as sm

from statsmodels.formula.api import ols



mod = ols('AC_rt ~ is_ac_old_rt', data=data_new).fit()

                

aov_table = sm.stats.anova_lm(mod, typ=2)

aov_table
files = ["/kaggle/input/abbcac-rt/AB_final_rt.csv",

         "/kaggle/input/abbcac-rt/BC_final_rt.csv",

         "/kaggle/input/abbcac-rt/AC_rt.csv"]
def rt_by_acc(file):

    

    AB_final = pd.read_csv(file)

    AB_final["RT_z"] = 0



    for idx in AB_final['sub'].unique():

        sub_rt = AB_final[AB_final["sub"] == idx]["RT"]

        sub_mean = np.mean(AB_final[AB_final["sub"] == idx]["RT"])

        sub_std = np.std(AB_final[AB_final["sub"] == idx]["RT"])

        sub_zs = (sub_rt - sub_mean) / sub_std

        AB_final.loc[AB_final["sub"] == idx, "RT_z"] = sub_zs



    AB_final_r = AB_final[AB_final["isCorrect"] == 1]

    AB_final_w = AB_final[AB_final["isCorrect"] == 0]





    fig = make_subplots(rows=1, cols=2)



    trace1 = go.Histogram(x=AB_final_r["RT_z"], name="correct")

    trace2 = go.Histogram(x=AB_final_w["RT_z"], nbinsx=10, name="incorrect")



    fig.append_trace(trace1, 1, 1)

    fig.append_trace(trace2, 1, 2)





    fig.update_layout(title_text=f'{file[24:26]}_final reaction time by accuracy', 

                      xaxis_title="Z-score")



    fig.show()

    

    print("Average reaction time if answer correctly:", np.round(np.mean(AB_final_r["RT_z"]), 4))

    print("Average reaction time if answer incorrectly:", np.round(np.mean(AB_final_w["RT_z"]), 4))



    mod = ols('RT_z ~ isCorrect', data=AB_final).fit()                

    aov_table = sm.stats.anova_lm(mod, typ=2)

    print(aov_table)
rt_by_acc(files[0])
rt_by_acc(files[1])
rt_by_acc(files[2])
def rt_by_age(file):

    

    AB_final = pd.read_csv(file)

    

    sub_rt = AB_final["RT"]

    sub_mean = np.mean(AB_final["RT"])

    sub_std = np.std(AB_final["RT"])

    sub_zs = (sub_rt- sub_mean) / sub_std

    AB_final["RT_z"] = sub_zs



    

    AB_final_child = AB_final[AB_final["ageGroup"] == "child"]

    AB_final_young = AB_final[AB_final["ageGroup"] == 'young adult']

    AB_final_ado = AB_final[AB_final["ageGroup"] == 'adolescent']



    

    fig = make_subplots(rows=1, cols=3)



    trace1 = go.Histogram(x=AB_final_child["RT_z"], name="child")

    trace2 = go.Histogram(x=AB_final_young["RT_z"], name="young adult")

    trace3 = go.Histogram(x=AB_final_ado["RT_z"], name="adolescent")





    fig.append_trace(trace1, 1, 1)

    fig.append_trace(trace2, 1, 2)

    fig.append_trace(trace3, 1, 3)





    fig.update_layout(title_text=f'{file[24:26]}_final reaction time by age', 

                      xaxis_title="Z-score")



    fig.show()

    

    print("Average reaction time of child:", np.round(np.mean(AB_final_child["RT_z"]), 4))

    print("Average reaction time of young adults:", np.round(np.mean(AB_final_young["RT_z"]), 4))

    print("Average reaction time of adolescent:", np.round(np.mean(AB_final_ado["RT_z"]), 4))



    mod = ols('RT_z ~ ageGroup', data=AB_final).fit()                

    aov_table = sm.stats.anova_lm(mod, typ=2)

    print(aov_table)
rt_by_age(files[0])
rt_by_age(files[1])
rt_by_age(files[2])
df = pd.read_csv("/kaggle/input/summer-data/trailbytrail.csv")
df["ageGroup3"].unique()
df = pd.read_csv("/kaggle/input/summer-data/trailbytrail.csv")

age_group = df[["ageGroup3", 'AB1_acc', 'AB2_acc', 'AB3_acc', 'AB4_acc', 

                'BC1_acc', 'BC2_acc', 'BC3_acc', 'BC4_acc', "AC_acc"]]

age_group.head()



grouped = age_group.groupby('ageGroup3', as_index=False)

summary = grouped.mean()

summary



col_names = ['AB1_acc', 'AB2_acc', 'AB3_acc', 'AB4_acc', 'BC1_acc', 'BC2_acc', 'BC3_acc', 'BC4_acc', "AC_acc"]





# add standard error for each accuracy

for name in col_names:

    temp = grouped.describe()[name][["count", 'std']]

    summary["se_" + name[:3]] = temp["std"] / np.sqrt(temp["count"])   





# get AB, BC values from AgeGroup

yc_AB = summary[summary["ageGroup3"] == "younger children"][['AB1_acc', 'AB2_acc', 'AB3_acc', 'AB4_acc']].values[0]

yc_BC = summary[summary["ageGroup3"] == "younger children"][['BC1_acc', 'BC2_acc', 'BC3_acc', 'BC4_acc']].values[0]



oc_AB = summary[summary["ageGroup3"] == "older children"][['AB1_acc', 'AB2_acc', 'AB3_acc', 'AB4_acc']].values[0]

oc_BC = summary[summary["ageGroup3"] == "older children"][['BC1_acc', 'BC2_acc', 'BC3_acc', 'BC4_acc']].values[0]



ya_AB = summary[summary["ageGroup3"] == "young adult"][['AB1_acc', 'AB2_acc', 'AB3_acc', 'AB4_acc']].values[0]

ya_BC = summary[summary["ageGroup3"] == "young adult"][['BC1_acc', 'BC2_acc', 'BC3_acc', 'BC4_acc']].values[0]



yado_AB = summary[summary["ageGroup3"] == "younger adolescent"][['AB1_acc', 'AB2_acc', 'AB3_acc', 'AB4_acc']].values[0]

yado_BC = summary[summary["ageGroup3"] == "younger adolescent"][['BC1_acc', 'BC2_acc', 'BC3_acc', 'BC4_acc']].values[0]



oa_AB = summary[summary["ageGroup3"] == "older adolescent"][['AB1_acc', 'AB2_acc', 'AB3_acc', 'AB4_acc']].values[0]

oa_BC = summary[summary["ageGroup3"] == "older adolescent"][['BC1_acc', 'BC2_acc', 'BC3_acc', 'BC4_acc']].values[0]



oadu_AB = summary[summary["ageGroup3"] == "older adult"][['AB1_acc', 'AB2_acc', 'AB3_acc', 'AB4_acc']].values[0]

oadu_BC = summary[summary["ageGroup3"] == "older adult"][['BC1_acc', 'BC2_acc', 'BC3_acc', 'BC4_acc']].values[0]



# get standard error for AB, BC values from AgeGroup

yc_AB_se = summary[summary["ageGroup3"] == "younger children"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

yc_BC_se = summary[summary["ageGroup3"] == "younger children"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



oc_AB_se = summary[summary["ageGroup3"] == "older children"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

oc_BC_se = summary[summary["ageGroup3"] == "older children"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



ya_AB_se = summary[summary["ageGroup3"] == "young adult"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

ya_BC_se = summary[summary["ageGroup3"] == "young adult"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



yado_AB_se = summary[summary["ageGroup3"] == "younger adolescent"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

yado_BC_se = summary[summary["ageGroup3"] == "younger adolescent"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



oa_AB_se = summary[summary["ageGroup3"] == "older adolescent"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

oa_BC_se = summary[summary["ageGroup3"] == "older adolescent"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



oadu_AB_se = summary[summary["ageGroup3"] == "older adult"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

oadu_BC_se = summary[summary["ageGroup3"] == "older adult"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



# make 2x2 subplots

fig = make_subplots(rows=3, cols=2,

                    subplot_titles=('younger children', 'older children', 'younger adolescent', 

                                    'older adolescent', 'young adult', 'older adult'))



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=yc_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=yc_AB_se)),  row=1, col=1)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=yc_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=yc_BC_se)),row=1, col=1)



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=oc_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=oc_AB_se)),  row=1, col=2)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=oc_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=oc_BC_se)),row=1, col=2)  



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=yado_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=yado_AB_se)),  row=2, col=1)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=yado_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=yado_BC_se)),row=2, col=1)  



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=oa_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=oa_AB_se)),  row=2, col=2)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=oa_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=oa_BC_se)),row=2, col=2)  



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=ya_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=ya_AB_se)),  row=3, col=1)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=ya_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=ya_BC_se)),row=3, col=1)  



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=oadu_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=oadu_AB_se)),  row=3, col=2)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=oadu_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=oadu_BC_se)),row=3, col=2)  



fig.update_layout(width=1000, height=1000, title_text='Accuracy comparison among groups with error bar')
# get AC values from AgeGroup

# get AB, BC values from AgeGroup

yc_AB = summary[summary["ageGroup3"] == "younger children"][['AC_acc']].values[0][0]

yc_AB_se = summary[summary["ageGroup3"] == "younger children"][['se_AC_']].values[0][0]



oc_AB = summary[summary["ageGroup3"] == "older children"][['AC_acc']].values[0][0]

oc_AB_se = summary[summary["ageGroup3"] == "older children"][['se_AC_']].values[0][0]



ya_AB = summary[summary["ageGroup3"] == "young adult"][['AC_acc']].values[0][0]

ya_AB_se = summary[summary["ageGroup3"] == "young adult"][['se_AC_']].values[0][0]



yado_AB = summary[summary["ageGroup3"] == "younger adolescent"][['AC_acc']].values[0][0]

yado_AB_se = summary[summary["ageGroup3"] == "younger adolescent"][['se_AC_']].values[0][0]



oa_AB = summary[summary["ageGroup3"] == "older adolescent"][['AC_acc']].values[0][0]

oa_AB_se = summary[summary["ageGroup3"] == "older adolescent"][['se_AC_']].values[0][0]



oadu_AB = summary[summary["ageGroup3"] == "older adult"][['AC_acc']].values[0][0]

oadu_AB_se = summary[summary["ageGroup3"] == "older adult"][['se_AC_']].values[0][0]



AC_df = pd.DataFrame({"category": ["younger children", "older children", "younger adolescent",

                                   "older adolescent", "young adult","older adult"],

                      "value": [yc_AB, oc_AB, yado_AB, oa_AB,  ya_AB, oadu_AB],

                      "se": [yc_AB_se, oc_AB_se, yado_AB_se, oa_AB_se, ya_AB_se, oadu_AB_se] 

                     })

AC_df
fig = go.Figure()

fig.add_trace(go.Bar(

    x=AC_df['category'], y=AC_df["value"],

    error_y=dict(type='data', array=AC_df["se"])

))

fig.update_layout(title_text='AC Accuracy comparison among groups with error bar', 

                  xaxis_title="category",

                  yaxis_title="AC Accuracy")
summary
# Question remained: How to group by individual subjects
df.columns
df = pd.read_csv("/kaggle/input/summer-data/trailbytrail.csv")

age_group = df[["ageGroup3", 'AB1_RT', 'AB2_RT', 'AB3_RT', 'AB4_RT', 

                'BC1_RT', 'BC2_RT', 'BC3_RT', 'BC4_RT', "AC_RT"]]

age_group.head()



grouped = age_group.groupby('ageGroup3', as_index=False)

summary = grouped.mean()

summary



col_names = ['AB1_RT', 'AB2_RT', 'AB3_RT', 'AB4_RT', 'BC1_RT', 'BC2_RT', 'BC3_RT', 'BC4_RT', "AC_RT"]





# add standard error for each accuracy

for name in col_names:

    temp = grouped.describe()[name][["count", 'std']]

    summary["se_" + name[:3]] = temp["std"] / np.sqrt(temp["count"])   





# get AB, BC values from AgeGroup

yc_AB = summary[summary["ageGroup3"] == "younger children"][['AB1_RT', 'AB2_RT', 'AB3_RT', 'AB4_RT']].values[0]

yc_BC = summary[summary["ageGroup3"] == "younger children"][['BC1_RT', 'BC2_RT', 'BC3_RT', 'BC4_RT']].values[0]



oc_AB = summary[summary["ageGroup3"] == "older children"][['AB1_RT', 'AB2_RT', 'AB3_RT', 'AB4_RT']].values[0]

oc_BC = summary[summary["ageGroup3"] == "older children"][['BC1_RT', 'BC2_RT', 'BC3_RT', 'BC4_RT']].values[0]



ya_AB = summary[summary["ageGroup3"] == "young adult"][['AB1_RT', 'AB2_RT', 'AB3_RT', 'AB4_RT']].values[0]

ya_BC = summary[summary["ageGroup3"] == "young adult"][['BC1_RT', 'BC2_RT', 'BC3_RT', 'BC4_RT']].values[0]



yado_AB = summary[summary["ageGroup3"] == "younger adolescent"][['AB1_RT', 'AB2_RT', 'AB3_RT', 'AB4_RT']].values[0]

yado_BC = summary[summary["ageGroup3"] == "younger adolescent"][['BC1_RT', 'BC2_RT', 'BC3_RT', 'BC4_RT']].values[0]



oa_AB = summary[summary["ageGroup3"] == "older adolescent"][['AB1_RT', 'AB2_RT', 'AB3_RT', 'AB4_RT']].values[0]

oa_BC = summary[summary["ageGroup3"] == "older adolescent"][['BC1_RT', 'BC2_RT', 'BC3_RT', 'BC4_RT']].values[0]



oadu_AB = summary[summary["ageGroup3"] == "older adult"][['AB1_RT', 'AB2_RT', 'AB3_RT', 'AB4_RT']].values[0]

oadu_BC = summary[summary["ageGroup3"] == "older adult"][['BC1_RT', 'BC2_RT', 'BC3_RT', 'BC4_RT']].values[0]



# get standard error for AB, BC values from AgeGroup

yc_AB_se = summary[summary["ageGroup3"] == "younger children"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

yc_BC_se = summary[summary["ageGroup3"] == "younger children"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



oc_AB_se = summary[summary["ageGroup3"] == "older children"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

oc_BC_se = summary[summary["ageGroup3"] == "older children"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



ya_AB_se = summary[summary["ageGroup3"] == "young adult"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

ya_BC_se = summary[summary["ageGroup3"] == "young adult"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



yado_AB_se = summary[summary["ageGroup3"] == "younger adolescent"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

yado_BC_se = summary[summary["ageGroup3"] == "younger adolescent"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



oa_AB_se = summary[summary["ageGroup3"] == "older adolescent"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

oa_BC_se = summary[summary["ageGroup3"] == "older adolescent"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



oadu_AB_se = summary[summary["ageGroup3"] == "older adult"][['se_AB1', 'se_AB2', 'se_AB3', 'se_AB4']].values[0]

oadu_BC_se = summary[summary["ageGroup3"] == "older adult"][['se_BC1', 'se_BC2', 'se_BC3', 'se_BC4']].values[0]



# make 2x2 subplots

fig = make_subplots(rows=3, cols=2,

                    subplot_titles=('younger children', 'older children', 'younger adolescent', 

                                    'older adolescent', 'young adult', 'older adult'))



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=yc_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=yc_AB_se)),  row=1, col=1)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=yc_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=yc_BC_se)),row=1, col=1)



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=oc_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=oc_AB_se)),  row=1, col=2)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=oc_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=oc_BC_se)),row=1, col=2)  



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=yado_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=yado_AB_se)),  row=2, col=1)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=yado_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=yado_BC_se)),row=2, col=1)  



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=oa_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=oa_AB_se)),  row=2, col=2)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=oa_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=oa_BC_se)),row=2, col=2)  



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=ya_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=ya_AB_se)),  row=3, col=1)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=ya_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=ya_BC_se)),row=3, col=1)  



fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=oadu_AB, name="AB",

                         line=dict(color='firebrick', width=2),

                         error_y=dict(array=oadu_AB_se)),  row=3, col=2)

fig.add_trace(go.Scatter(x=['1', '2', '3', '4'], y=oadu_BC, name="BC",

                         line=dict(color='royalblue', width=2),

                         error_y=dict(array=oadu_BC_se)),row=3, col=2)  



fig.update_layout(width=1000, height=1000, title_text='Reaction time comparison among groups with error bar')
# get AC values from AgeGroup

# get AB, BC values from AgeGroup

yc_AB = summary[summary["ageGroup3"] == "younger children"][['AC_RT']].values[0][0]

yc_AB_se = summary[summary["ageGroup3"] == "younger children"][['se_AC_']].values[0][0]



oc_AB = summary[summary["ageGroup3"] == "older children"][['AC_RT']].values[0][0]

oc_AB_se = summary[summary["ageGroup3"] == "older children"][['se_AC_']].values[0][0]



ya_AB = summary[summary["ageGroup3"] == "young adult"][['AC_RT']].values[0][0]

ya_AB_se = summary[summary["ageGroup3"] == "young adult"][['se_AC_']].values[0][0]



yado_AB = summary[summary["ageGroup3"] == "younger adolescent"][['AC_RT']].values[0][0]

yado_AB_se = summary[summary["ageGroup3"] == "younger adolescent"][['se_AC_']].values[0][0]



oa_AB = summary[summary["ageGroup3"] == "older adolescent"][['AC_RT']].values[0][0]

oa_AB_se = summary[summary["ageGroup3"] == "older adolescent"][['se_AC_']].values[0][0]



oadu_AB = summary[summary["ageGroup3"] == "older adult"][['AC_RT']].values[0][0]

oadu_AB_se = summary[summary["ageGroup3"] == "older adult"][['se_AC_']].values[0][0]



AC_df = pd.DataFrame({"category": ["younger children", "older children", "younger adolescent",

                                   "older adolescent", "young adult","older adult"],

                      "value": [yc_AB, oc_AB, yado_AB, oa_AB,  ya_AB, oadu_AB],

                      "se": [yc_AB_se, oc_AB_se, yado_AB_se, oa_AB_se, ya_AB_se, oadu_AB_se] 

                     })

AC_df
fig = go.Figure()

fig.add_trace(go.Bar(

    x=AC_df['category'], y=AC_df["value"],

    error_y=dict(type='data', array=AC_df["se"])

))

fig.update_layout(title_text='AC Reaction Time comparison among groups with error bar', 

                  xaxis_title="category",

                  yaxis_title="AC Accuracy")
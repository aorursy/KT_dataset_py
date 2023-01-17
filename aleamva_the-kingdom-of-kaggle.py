import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
%matplotlib inline

#Loading answers for multiple choice questions
multiple = pd.read_csv('../input/multipleChoiceResponses.csv', low_memory=False)

#Plotting question 24#
colors1 = 'deeppink','mediumvioletred','blueviolet','steelblue','turquoise','chartreuse','yellow','darkorange','crimson','lightcoral'
coding = multiple.Q24.value_counts()
plot1 = coding.plot.barh(figsize = [8,8], fontsize = 12, color = colors1, title = 'Years of Experience of Wisezards and Neophytes')

plot1.set_xlabel("Kaggle's people")
plot1.set_ylabel("Years writing code")


qs = multiple[["Q24","Q1","Q8","Q2","Q17"]].iloc[1:] 


#Combining question 1 with main question 24#

q_F = qs.query("Q1 == 'Female'")
q_M = qs.query("Q1 == 'Male'")

nms= ['Male','Female']
FM_24 = pd.DataFrame([q_M.Q24.value_counts(),q_F.Q24.value_counts()],index=nms)

#Ploting#
my_colors = 'lightseagreen','salmon'
gender = FM_24.T.plot(kind = "barh", color=my_colors, figsize=[8,8], fontsize=12,title = "Women influency")

gender.set_xlabel("Kaggle's people")
gender.set_ylabel("Years writing code")

#For compare age with only a range of experience's years#

r_0 = qs.query("Q24 == '< 1 year'")

rn_M = r_0.query("Q1 == 'Male'")
rn_F = r_0.query("Q1 == 'Female'")

name = ['Male','Female']
rn_1 = pd.DataFrame([rn_M.Q2.value_counts(),rn_F.Q2.value_counts()], index = name)

rnges = ["'< 1 year'","'1-2 years'","'3-5 years'","'5-10 years'","'10-20 years'","'20-30 years'","'30-40 years'",
         "'40+ years'", "'I have never written code but I want to learn'","'I have never written code and I do not want to learn'"]
rns_xp = []
for i in range(10):
    a = 'Q24 == '
    r0_1 = qs.query(a + rnges[i])
    rn_F = r0_1.query("Q1 == 'Female'")
    rn_M = r0_1.query("Q1 == 'Male'")
    nms= ['Male','Female']
    rn_1 = pd.DataFrame([rn_M.Q2.value_counts(),rn_F.Q2.value_counts()],index=nms).T
    rns_xp.append(rn_1)

#Plotting graph of Age for 1-2 years of experience#
my_colors1 = 'rebeccapurple','seagreen'
exp_1 = rns_xp[1].plot(kind = "barh", figsize=[10,10], color = my_colors1, fontsize=12, title = 'Age of Neophytes from 1 to 2 years of experience')
exp_1.set_xlabel("Kaggle's people")
exp_1.set_ylabel("Age ranges")
#Plotting graph of Age for 30-40 years of experience#
my_colors1 = 'mediumpurple','lightseagreen'
exp_1 = rns_xp[7].plot(kind = "barh", figsize=[10,10], color = my_colors1, fontsize=12, title = 'Age of Wisezards from 30 to 40 years of experience')
exp_1.set_xlabel("Kaggle's people")
exp_1.set_ylabel("Age ranges")
#For compare programming language with only a range of experience's years#
rnges = ["'< 1 year'","'1-2 years'","'3-5 years'","'5-10 years'","'10-20 years'","'20-30 years'","'30-40 years'",
         "'40+ years'", "'I have never written code but I want to learn'","'I have never written code and I do not want to learn'"]
rns_lang = []
for i in range(10):
    a = 'Q24 == '
    r0_1 = qs.query(a + rnges[i])
    rn_F = r0_1.query("Q1 == 'Female'")
    rn_M = r0_1.query("Q1 == 'Male'")
    nms= ['Male','Female']
    rn_1 = pd.DataFrame([rn_M.Q17.value_counts(),rn_F.Q17.value_counts()],index=nms).T
    rns_lang.append(rn_1)
    
#Plotting graph of programming language for 1-2 years of experience#    
my_colors2 = 'limegreen','deeppink'
lang = rns_lang[1].plot(kind = "barh", figsize=[10,10], color=my_colors2, fontsize=12, title = 'Programming language used for Neophytes from 1 to 2 years of experience')
lang.set_xlabel("Kaggle's people")
lang.set_ylabel("Programming language")
#Plotting graph of programming language for 30-40 years of experience# 
my_colors2 = 'darkslategrey','crimson'
lang1 = rns_lang[7].plot(kind = "barh", figsize=[10,10], color=my_colors2, fontsize=12, title = 'Programming language used for Wisezards from 30 to 40 years of experience')
lang1.set_xlabel("Kaggle's people")
lang1.set_ylabel("Programming language")
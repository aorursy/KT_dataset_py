import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 13,8
sns.set_style("darkgrid")
import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")
#loading dataset
main_survey = pd.read_csv("../input/multipleChoiceResponses.csv")
text_survey = pd.read_csv("../input/freeFormResponses.csv")
main_survey.head()
text_survey.head()
# lets start with question number 1 :Gender analysis
plt.figure(figsize=(13,8))
a = sns.countplot((main_survey.Q1[1:]),palette="Set1", order = main_survey.Q1[1:].value_counts().index)
plt.title("Gender analysis")
plt.xlabel("gender")
plt.ylabel("respondant")
for p in a.patches:
    plt.annotate('{:d}'.format(p.get_height()),(p.get_x(), p.get_height()) )
plt.tight_layout()

main_survey.columns
plt.figure(figsize=(13,8))
sns.countplot(main_survey["Q2"][1:],hue = main_survey.Q1[1:],palette="rainbow",order = main_survey["Q2"][1:].value_counts().index )
plt.title("gender and Age comparison")
plt.xlabel("Age bins")
plt.ylabel("respondant")
plt.show()
main_survey.Q3.value_counts()
plt.figure(figsize = (10,60))
sns.countplot(y = main_survey.Q3[1:],hue = main_survey.Q1[1:],palette="rainbow",order = main_survey.Q3[1:].value_counts().index )
plt.title("country and age comparison")
plt.ylabel("country")
plt.xlabel("respondant")
plt.show()
df = pd.crosstab(index = main_survey.Q3[1:],columns=main_survey.Q1[1:],colnames=["Gender"],rownames=["Country"])
df.drop(columns=['Prefer not to say','Prefer to self-describe'],axis = 1,inplace = True)
df["percent_difference"] = (np.abs((df["Female"]-df["Male"]))/(df["Female"]+df["Male"]))*100
df
#checking in which country the percent difference between men and women respondants is less.
print("country with less than 50 % difference ",df[df["percent_difference"]<=50].index)
print("*"*20)
print("country with less than 70 % difference ",df[df["percent_difference"]<=70].index)
df2 = pd.crosstab(index = main_survey.Q3[1:],columns=main_survey.Q2[1:],colnames=["age_group"],rownames=["Country"])
plt.figure(figsize=(20,20))
sns.heatmap(data = df2,cmap = "rainbow",linewidths=1,annot=True,fmt = 'd')
plt.figure(figsize=(13,8))
sns.countplot(y = main_survey.Q4[1:],palette="rainbow")
plt.title("level of Degree")
plt.ylabel("Degree")
plt.xlabel("respondant")
plt.show()
print(main_survey.Q4[1:].value_counts())
# studies compared with age group / gender
df3 = pd.crosstab(index = main_survey.Q4[1:],columns=main_survey.Q2[1:])
df3
df_m = main_survey.copy(deep = True)
df_m[(df_m["Q2"]== "18-21") & (df_m["Q4"]== "Doctoral degree")]
#Undergrad Major
plt.figure(figsize=(13,8))
sns.countplot(y = df_m.Q5[1:],palette="rainbow")
plt.title("undergraduate major")
plt.ylabel("Majors")
plt.xlabel("respondant")
plt.show()
#current/latest Job roles:
plt.figure(figsize=(13,8))
sns.countplot(y = df_m["Q6"][1:])
plt.title("CURRENT/ LATEST JOB ROLES")
plt.ylabel("Job Roles")
plt.xlabel("respondant")
plt.show()
#checking degree vs job role
plt.figure(figsize=(13,8))
df4 = pd.crosstab(index = df_m.Q6[1:],columns=df_m.Q4[1:])
sns.heatmap(data = df4,cmap = "rainbow",annot= True,linewidths=2)
plt.title("degree vs job role")
plt.ylabel("Job Roles")
plt.xlabel("Degrees")
plt.show()
#job role vs major
plt.figure(figsize=(13,8))
df5 = pd.crosstab(index = df_m.Q6[1:],columns=df_m.Q5[1:])
sns.heatmap(data = df5,cmap = "rainbow",annot= True,linewidths=2)
#the below graph shows that respondants with CS major currently have quite a amount of jobs 
#in data science and software engg field.
#'In what industry is your current employer/contract 
#(or your most recent employer if retired)? - Selected Choice
sns.countplot(y = df_m["Q7"][1:])
plt.title("industry is your current employer")
plt.ylabel("industry")
plt.xlabel("respondant")
plt.show()
plt.figure(figsize=(13,8))
sns.countplot(x = df_m["Q8"][1:],order=df_m["Q8"][1:].value_counts().index)
plt.title("year of Experience")
plt.xlabel("no. of years of experience")
plt.ylabel("respondant")
plt.show()
pd.crosstab(index = df_m.Q6[1:],columns=df_m.Q8[1:])
plt.figure(figsize=(13,8))
sns.countplot(y = df_m.Q9[1:],order = df_m.Q9[1:].value_counts().index)
plt.title("Earning per year")
plt.ylabel("Earnings per year")
plt.xlabel("respondant")
plt.show()
df_m.Q10.value_counts()
dfq11 = df_m[['Q11_Part_1', 'Q11_Part_2', 'Q11_Part_3','Q11_Part_4', 'Q11_Part_5', 'Q11_Part_6', 'Q11_Part_7']]
#dfq12 = df_m[['Q12_Part_1_TEXT', 'Q12_Part_2_TEXT', 'Q12_Part_3_TEXT','Q12_Part_4_TEXT', 'Q12_Part_5_TEXT', 'Q12_OTHER_TEXT']]
dfq13 = df_m[['Q13_Part_1','Q13_Part_2', 'Q13_Part_3', 'Q13_Part_4', 'Q13_Part_5', 'Q13_Part_6','Q13_Part_7', 'Q13_Part_8', 'Q13_Part_9', 'Q13_Part_10', 'Q13_Part_11','Q13_Part_12', 'Q13_Part_13', 'Q13_Part_14', 'Q13_Part_15']]
dfq14 = df_m[['Q14_Part_1', 'Q14_Part_2', 'Q14_Part_3', 'Q14_Part_4', 'Q14_Part_5','Q14_Part_6', 'Q14_Part_7', 'Q14_Part_8', 'Q14_Part_9', 'Q14_Part_10','Q14_Part_11']]
dfq15 = df_m[['Q15_Part_1', 'Q15_Part_2', 'Q15_Part_3', 'Q15_Part_4', 'Q15_Part_5','Q15_Part_6', 'Q15_Part_7']]
dfq16 = df_m[['Q16_Part_1', 'Q16_Part_2', 'Q16_Part_3', 'Q16_Part_4', 'Q16_Part_5','Q16_Part_6', 'Q16_Part_7', 'Q16_Part_8', 'Q16_Part_9', 'Q16_Part_10','Q16_Part_11', 'Q16_Part_12', 'Q16_Part_13', 'Q16_Part_14','Q16_Part_15', 'Q16_Part_16', 'Q16_Part_17', 'Q16_Part_18']]
dfq19 = df_m[['Q19_Part_1', 'Q19_Part_2','Q19_Part_3', 'Q19_Part_4', 'Q19_Part_5', 'Q19_Part_6', 'Q19_Part_7','Q19_Part_8', 'Q19_Part_9', 'Q19_Part_10', 'Q19_Part_11', 'Q19_Part_12','Q19_Part_13', 'Q19_Part_14', 'Q19_Part_15', 'Q19_Part_16','Q19_Part_17', 'Q19_Part_18', 'Q19_Part_19']]
dfq21 = df_m[['Q21_Part_1','Q21_Part_2', 'Q21_Part_3', 'Q21_Part_4', 'Q21_Part_5', 'Q21_Part_6','Q21_Part_7', 'Q21_Part_8', 'Q21_Part_9', 'Q21_Part_10', 'Q21_Part_11','Q21_Part_12', 'Q21_Part_13']]
dfq27 = df_m[['Q27_Part_1', 'Q27_Part_2', 'Q27_Part_3', 'Q27_Part_4', 'Q27_Part_5','Q27_Part_6', 'Q27_Part_7', 'Q27_Part_8', 'Q27_Part_9', 'Q27_Part_10','Q27_Part_11', 'Q27_Part_12', 'Q27_Part_13', 'Q27_Part_14','Q27_Part_15', 'Q27_Part_16', 'Q27_Part_17', 'Q27_Part_18','Q27_Part_19', 'Q27_Part_20']]
dfq28 = df_m[['Q28_Part_1', 'Q28_Part_2', 'Q28_Part_3','Q28_Part_4', 'Q28_Part_5', 'Q28_Part_6', 'Q28_Part_7', 'Q28_Part_8','Q28_Part_9', 'Q28_Part_10', 'Q28_Part_11', 'Q28_Part_12','Q28_Part_13', 'Q28_Part_14', 'Q28_Part_15', 'Q28_Part_16','Q28_Part_17', 'Q28_Part_18', 'Q28_Part_19', 'Q28_Part_20','Q28_Part_21', 'Q28_Part_22', 'Q28_Part_23', 'Q28_Part_24','Q28_Part_25', 'Q28_Part_26', 'Q28_Part_27', 'Q28_Part_28','Q28_Part_29', 'Q28_Part_30', 'Q28_Part_31', 'Q28_Part_32','Q28_Part_33', 'Q28_Part_34', 'Q28_Part_35', 'Q28_Part_36','Q28_Part_37', 'Q28_Part_38', 'Q28_Part_39', 'Q28_Part_40','Q28_Part_41', 'Q28_Part_42', 'Q28_Part_43']]
dfq29 = df_m[['Q29_Part_1', 'Q29_Part_2', 'Q29_Part_3','Q29_Part_4', 'Q29_Part_5', 'Q29_Part_6', 'Q29_Part_7', 'Q29_Part_8','Q29_Part_9', 'Q29_Part_10', 'Q29_Part_11', 'Q29_Part_12','Q29_Part_13', 'Q29_Part_14', 'Q29_Part_15', 'Q29_Part_16','Q29_Part_17', 'Q29_Part_18', 'Q29_Part_19', 'Q29_Part_20','Q29_Part_21', 'Q29_Part_22', 'Q29_Part_23', 'Q29_Part_24','Q29_Part_25', 'Q29_Part_26', 'Q29_Part_27', 'Q29_Part_28']]
dfq30 = df_m[['Q30_Part_1', 'Q30_Part_2', 'Q30_Part_3','Q30_Part_4', 'Q30_Part_5', 'Q30_Part_6', 'Q30_Part_7', 'Q30_Part_8','Q30_Part_9', 'Q30_Part_10', 'Q30_Part_11', 'Q30_Part_12','Q30_Part_13', 'Q30_Part_14', 'Q30_Part_15', 'Q30_Part_16','Q30_Part_17', 'Q30_Part_18', 'Q30_Part_19', 'Q30_Part_20','Q30_Part_21', 'Q30_Part_22', 'Q30_Part_23', 'Q30_Part_24','Q30_Part_25']]
dfq31 = df_m[['Q31_Part_1', 'Q31_Part_2', 'Q31_Part_3', 'Q31_Part_4', 'Q31_Part_5','Q31_Part_6', 'Q31_Part_7', 'Q31_Part_8', 'Q31_Part_9', 'Q31_Part_10','Q31_Part_11', 'Q31_Part_12', 'Q31_OTHER_TEXT', 'Q32', 'Q32_OTHER','Q33_Part_1', 'Q33_Part_2', 'Q33_Part_3', 'Q33_Part_4', 'Q33_Part_5','Q33_Part_6', 'Q33_Part_7', 'Q33_Part_8', 'Q33_Part_9', 'Q33_Part_10','Q33_Part_11']]
dfq33 = df_m[['Q33_Part_1', 'Q33_Part_2', 'Q33_Part_3', 'Q33_Part_4', 'Q33_Part_5','Q33_Part_6', 'Q33_Part_7', 'Q33_Part_8', 'Q33_Part_9', 'Q33_Part_10','Q33_Part_11']]
dfq34 = df_m[['Q34_Part_1', 'Q34_Part_2','Q34_Part_3', 'Q34_Part_4', 'Q34_Part_5', 'Q34_Part_6']]
dfq35 = df_m[['Q35_Part_1', 'Q35_Part_2', 'Q35_Part_3','Q35_Part_4', 'Q35_Part_5', 'Q35_Part_6']]
dfq36 = df_m[['Q36_Part_1', 'Q36_Part_2', 'Q36_Part_3','Q36_Part_4', 'Q36_Part_5', 'Q36_Part_6', 'Q36_Part_7', 'Q36_Part_8','Q36_Part_9', 'Q36_Part_10', 'Q36_Part_11', 'Q36_Part_12','Q36_Part_13']]
dfq38 = df_m[['Q38_Part_1','Q38_Part_2', 'Q38_Part_3', 'Q38_Part_4', 'Q38_Part_5', 'Q38_Part_6','Q38_Part_7', 'Q38_Part_8', 'Q38_Part_9', 'Q38_Part_10', 'Q38_Part_11','Q38_Part_12', 'Q38_Part_13', 'Q38_Part_14', 'Q38_Part_15','Q38_Part_16', 'Q38_Part_17', 'Q38_Part_18', 'Q38_Part_19','Q38_Part_20', 'Q38_Part_21', 'Q38_Part_22']]
dfq39 = df_m[['Q39_Part_1', 'Q39_Part_2']]
dfq41 = df_m[['Q41_Part_1', 'Q41_Part_2','Q41_Part_3']]
dfq42 = df_m[['Q42_Part_1', 'Q42_Part_2', 'Q42_Part_3', 'Q42_Part_4','Q42_Part_5']]
dfq44 = df_m[['Q44_Part_1', 'Q44_Part_2','Q44_Part_3', 'Q44_Part_4', 'Q44_Part_5', 'Q44_Part_6']]
dfq45 = df_m[['Q45_Part_1', 'Q45_Part_2', 'Q45_Part_3', 'Q45_Part_4','Q45_Part_5', 'Q45_Part_6']]
dfq47 = df_m[['Q47_Part_1', 'Q47_Part_2','Q47_Part_3', 'Q47_Part_4', 'Q47_Part_5', 'Q47_Part_6', 'Q47_Part_7','Q47_Part_8', 'Q47_Part_9', 'Q47_Part_10', 'Q47_Part_11', 'Q47_Part_12','Q47_Part_13', 'Q47_Part_14', 'Q47_Part_15', 'Q47_Part_16']]
dfq49 = df_m[['Q49_Part_1', 'Q49_Part_2', 'Q49_Part_3', 'Q49_Part_4', 'Q49_Part_5','Q49_Part_6', 'Q49_Part_7', 'Q49_Part_8', 'Q49_Part_9', 'Q49_Part_10','Q49_Part_11', 'Q49_Part_12']]
dfq50 = df_m[['Q50_Part_1','Q50_Part_2', 'Q50_Part_3', 'Q50_Part_4', 'Q50_Part_5', 'Q50_Part_6','Q50_Part_7', 'Q50_Part_8']]
#12,17,18,22,23,24,25,26,32,37,40,44,46,48
def combine_analysis(dfq):
    collective_values = pd.Series()
    st = pd.Series()
    for i in dfq:
        st = dfq[i][0]
        b = st.split("-")[0]
        dfq[i].dropna(inplace = True)
        collective_values = collective_values.append(dfq[i][1:])
    return(collective_values,b)
def qplot(a,b):
    plt.figure(figsize=(10,10))
    sns.countplot(y = a , order = a.value_counts().index)
    plt.title(b)
    plt.xlabel("respondants")
    plt.plot()
def parts_analysis(dfq):
    a,b = combine_analysis(dfq)
    qplot(a,b)
def non_mcq(dfm):
    sns.countplot(y = df_m[dfm][1:])
    plt.xlabel("respondants")
    plt.title(df_m[dfm][0])
#question Number 11:
parts_analysis(dfq11)
# question number 12: 
non_mcq("Q12_MULTIPLE_CHOICE")
#question number 13:
plt.figure(figsize=(13,8))
parts_analysis(dfq13)
#question number : 14
plt.figure(figsize=(13,8))
parts_analysis(dfq14)
s = text_survey.Q14_OTHER_TEXT.str.lower()
s.dropna().value_counts().head(5)
#question number : 15
plt.figure(figsize=(13,8))
parts_analysis(dfq15)
#question number : 16
plt.figure(figsize=(13,8))
parts_analysis(dfq16)
#question number : 17
plt.figure(figsize=(13,8))
non_mcq("Q17")
#question number : 18
plt.figure(figsize=(13,8))
non_mcq("Q18")
#question number : 19
plt.figure(figsize=(13,8))
parts_analysis(dfq19)
#question number : 20
plt.figure(figsize=(13,8))
non_mcq("Q20")
#question number : 21
plt.figure(figsize=(13,8))
parts_analysis(dfq21)
#question number : 22
plt.figure(figsize=(13,8))
non_mcq("Q22")
#question number : 23
plt.figure(figsize=(13,8))
non_mcq("Q23")
#comparing it with jobs post
plt.figure(figsize=(13,8))
t = pd.crosstab(index = df_m.Q6[1:],columns=df_m.Q23[1:])
sns.heatmap(data = t, annot= True,cmap = "rainbow",linewidths=1)
#question number : 24
plt.figure(figsize=(13,8))
non_mcq("Q24")
#question number : 25
plt.figure(figsize=(13,8))
non_mcq("Q25")
#question number : 26
plt.figure(figsize=(13,8))
non_mcq("Q26")
#question number : 27
plt.figure(figsize=(13,8))
parts_analysis(dfq27)
#question number : 28
plt.figure(figsize=(13,8))
parts_analysis(dfq28)
text_survey.Q28_OTHER_TEXT.str.lower().dropna().value_counts().head()
#question number : 29
plt.figure(figsize=(13,8))
parts_analysis(dfq29)
#question number : 30
plt.figure(figsize=(13,8))
parts_analysis(dfq30)
text_survey.Q30_OTHER_TEXT.dropna().str.lower().value_counts().head(5)
#question number : 32
plt.figure(figsize=(13,8))
#plt.figure(figsize = (20,20))
non_mcq("Q32")
#question number : 33
plt.figure(figsize = (13,8))
#plt.figure(figsize = (20,20))
parts_analysis(dfq33)
#question number : 36
#plt.figure(figsize = (20,20))
parts_analysis(dfq36)
#question number : 38
parts_analysis(dfq38)
#question number : 39
parts_analysis(dfq39)
#question number : 40
plt.figure(figsize = (13,8))
non_mcq("Q40")
#question number : 46
plt.figure(figsize = (13,8))
non_mcq("Q46")
#question number : 48
plt.figure(figsize = (13,8))
#plt.figure(figsize=(20,60))
non_mcq("Q48")
#question number : 49
#plt.figure(figsize=(20,60))
parts_analysis(dfq49)
#question number : 50
#plt.figure(figsize=(20,60))
parts_analysis(dfq50)
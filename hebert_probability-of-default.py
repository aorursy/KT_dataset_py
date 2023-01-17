import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

import matplotlib as mpl

import sklearn

from sklearn.linear_model import LogisticRegression

mpl.style.use(["ggplot"])
CreditRisk_training = pd.read_csv("../input/GiveMeSomeCredit/cs-training.csv")
print("the shape of the dataset is: {}".format(CreditRisk_training.shape))

CreditRisk_training.head()
df = CreditRisk_training.copy()

df.rename(columns={"Unnamed: 0":"Id"},inplace=True)

df.dtypes
df.describe().T
y = df.isnull().sum()

(fig,ax) = plt.subplots(figsize=(10,8))





pd.DataFrame(y).reset_index().sort_values(0).plot(ax=ax,kind="barh",y=0,x="index",label="number of missing values")

plt.ylabel("Variables")

plt.title("Missing values on variables")

plt.show()
log_income=np.log(df.MonthlyIncome+0.0005)

df.insert(7,"log_income",log_income)
(fig,(ax1,ax2)) = plt.subplots(1,2,figsize=(15,8))

fig.suptitle("Income distribution with and without transformation",fontsize=16)

ax2.set_title("Natural Logarithm transformation")

ax1.set_title("No transformation")







df.MonthlyIncome.plot(ax=ax1,kind="hist",x="MonthlyIncome",bins=50)

df.log_income.plot(ax=ax2,kind="hist",x="log_income",bins=50)



plt.show()
df.MonthlyIncome.isnull().sum()/df.shape[0]
(fig,ax)=plt.subplots(figsize=(13,8))

ax.set_yscale('log')

df.loc[:,["SeriousDlqin2yrs","MonthlyIncome"]].boxplot(ax=ax,by="SeriousDlqin2yrs")

plt.show()
df[df.MonthlyIncome.isnull()].loc[:,"SeriousDlqin2yrs"].mean()
df.loc[:,"SeriousDlqin2yrs"].mean()
Income_cat = pd.qcut(df.log_income,q=30)

df.insert(7,"cat_income",Income_cat)
def summary_woe_func(X,y,df):

    #df[X]=df[X].astype(type_)

    df_missing = df[df[X].isnull()]

    

    summary_woe = df.groupby(X).agg({y:["count","sum"]})

    summary_woe.columns =["Count","Event"]

    summary_woe["perc"]=summary_woe.Count/summary_woe.sum().Count

    summary_woe["Non_event"]=summary_woe.Count-summary_woe.Event

    summary_woe["odd_i"] = summary_woe.Event/summary_woe.Non_event

    

    overall_event = summary_woe.sum().Event

    overall_non_event = summary_woe.sum().Non_event

    overall_odd = overall_event/overall_non_event

    

    data_set=pd.DataFrame()

    

    if df_missing.shape[0]!=0:

        data_set = df_missing.agg({y:["count","sum"]}).T

        data_set["Non_event"]=data_set["count"]-data_set["sum"]

        data_set["odd_i"] = data_set["sum"]/data_set["Non_event"]

        data_set["woe"]=np.log(data_set.odd_i/overall_odd)

   

    summary_woe["woe"]=np.log(summary_woe.odd_i/overall_odd)

    IV = ((summary_woe.Event/overall_event - summary_woe.Non_event/overall_non_event)*summary_woe.woe).sum()

    

    return({"summary_woe":summary_woe,"IV":IV,"missing":data_set})
def representation_woe(summary_tabl,data_missing):

    #Build the figure

    fig = plt.figure(figsize=(12,8))

    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)



    #construct the axes

    main_ax = fig.add_subplot(grid[1:,:])

    volume_ax = fig.add_subplot(grid[0,:], sharex=main_ax)



    #draw the graphs on the respective axis

    summary_tabl.plot(ax=volume_ax,kind="bar",y="perc",color="red",label="Size of each Bucket",alpha=0.5)

    if data_missing.shape[0]!=0:

        main_ax.axhline(data_missing["woe"][0],color="green",label="WoE of the observations with missing values")



    summary_tabl.plot(ax=main_ax,kind="bar",y="woe",label="Weight of evidence",color="blue")

    main_ax.legend()



    plt.show()

    return None
woe_Income = summary_woe_func("cat_income","SeriousDlqin2yrs",df)

representation_woe(woe_Income["summary_woe"],woe_Income['missing'])
group_1=['(6.758, 7.378]','(7.378, 7.603]','(7.603, 7.79]','(7.79, 7.901]','(7.901, 8.006]','(8.006, 8.089]']

group_2=['(8.089, 8.161]','(8.161, 8.243]','(8.243, 8.294]','(8.294, 8.359]','(8.359, 8.422]','(8.422, 8.492]']

group_3=['(8.492, 8.526]','(8.526, 8.594]','(8.594, 8.648]','(8.648, 8.7]','(8.7, 8.748]','(8.748, 8.807]']

group_4=['(8.987, 9.048]','(9.048, 9.114]','(9.114, 9.201]']

group_5 = ['(9.201, 9.259]','(9.259, 9.364]','(9.364, 9.489]','(9.489, 9.716]']

group_6 = ['(8.807, 8.865]','(8.865, 8.923]']

group_7 = ['(8.923, 8.987]','nan']



cat_income_final = [pd.Interval(left=6.758,right=8.089,closed="right") if str(x) in group_1 else x for x in df["cat_income"] ]

cat_income_final = [pd.Interval(left=8.089,right=8.492,closed="right") if str(x) in group_2 else x for x in cat_income_final ]

cat_income_final = [pd.Interval(left=8.492,right=8.807,closed="right") if str(x) in group_3 else x for x in cat_income_final ]

cat_income_final = [pd.Interval(left=8.987,right=9.201,closed="right") if str(x) in group_4 else x for x in cat_income_final ]

cat_income_final = [pd.Interval(left=9.201,right=9.716,closed="right") if str(x) in group_5 else x for x in cat_income_final ]

cat_income_final = [pd.Interval(left=8.807,right=8.923,closed="right") if str(x) in group_6 else x for x in cat_income_final ]

cat_income_final = [pd.Interval(left=8.923,right=8.987,closed="right") if str(x) in group_7 else x for x in cat_income_final ]



df["cat_income_final"]=cat_income_final

woe_Income = summary_woe_func("cat_income_final","SeriousDlqin2yrs",df)

representation_woe(woe_Income["summary_woe"],woe_Income['missing'])
woe = summary_woe_func("NumberOfDependents","SeriousDlqin2yrs",df)

representation_woe(woe["summary_woe"],woe['missing'])
group_1 = [1.0,2.0]

group_2 = ['3.0','4.0','5.0','6.0','7.0','8.0','9.0','10.0','13.0','20.0']

group_3 = ['0.0','nan']



dep_final = ['itermed' if x in group_1 else str(x) for x in df["NumberOfDependents"] ]

dep_final = ['high' if x in group_2 else x for x in dep_final ]

dep_final = ['low' if x in group_3 else x for x in dep_final ]



df["dep_final"]=dep_final

woe = summary_woe_func("dep_final","SeriousDlqin2yrs",df)

representation_woe(woe["summary_woe"],woe['missing'])
(fig,ax)=plt.subplots(figsize=(13,8))

ax.set_yscale('log')

df.loc[:,["SeriousDlqin2yrs","RevolvingUtilizationOfUnsecuredLines"]].boxplot(ax=ax,by="SeriousDlqin2yrs")

plt.show()
log_Revolving = np.log(df.RevolvingUtilizationOfUnsecuredLines+0.00005)

df.insert(7,"log_Revolving",log_Revolving)
df.log_Revolving.plot.hist(bins=20,figsize=(10,7))

plt.show()
Revolving_cat = pd.qcut(df.log_Revolving,q=10)
df.insert(7,"Revolving_cat",Revolving_cat)

woe = summary_woe_func("Revolving_cat","SeriousDlqin2yrs",df)

representation_woe(woe["summary_woe"],woe['missing'])
df["revolving_final"]=df["Revolving_cat"]
(fig,ax)=plt.subplots(figsize=(13,8))

#ax.set_yscale('log')

df.loc[:,["SeriousDlqin2yrs","age"]].boxplot(ax=ax,by="SeriousDlqin2yrs")

plt.show()
age_cat = pd.qcut(df.age,q=10)

df.insert(7,"age_cat_final",age_cat)

woe_age = summary_woe_func("age_cat_final","SeriousDlqin2yrs",df)

representation_woe(woe_age["summary_woe"],woe_age['missing'])
(fig,ax)=plt.subplots(figsize=(13,8))



df.loc[:,["SeriousDlqin2yrs","NumberOfTime30-59DaysPastDueNotWorse"]].boxplot(ax=ax,by="SeriousDlqin2yrs")

plt.show()

df.loc[:,["SeriousDlqin2yrs","NumberOfTime30-59DaysPastDueNotWorse"]].groupby("SeriousDlqin2yrs").describe()
test_cut = np.linspace(-0.001,100,5)

worse_cat = pd.cut(df["NumberOfTime30-59DaysPastDueNotWorse"],test_cut)

df.insert(7,"worse_cat_final",worse_cat)

woe = summary_woe_func("worse_cat_final","SeriousDlqin2yrs",df)

representation_woe(woe["summary_woe"],woe['missing'])
df.worse_cat_final.value_counts()
(fig,ax)=plt.subplots(figsize=(13,8))

ax.set_yscale('log')

df.loc[:,["SeriousDlqin2yrs","DebtRatio"]].boxplot(ax=ax,by="SeriousDlqin2yrs")

plt.show()
log_DebtRatio = np.log(df.DebtRatio+0.00005)

df.insert(7,"log_DebtRatio",log_DebtRatio)
DebtRatio_cat = pd.qcut(df.log_DebtRatio,q=10)

df.sort_values("log_DebtRatio",inplace=True)

df.insert(7,"DebtRatio_cat_final",DebtRatio_cat)

woe = summary_woe_func("DebtRatio_cat_final","SeriousDlqin2yrs",df)

representation_woe(woe["summary_woe"],woe['missing'])
(fig,ax)=plt.subplots(figsize=(12,7))

ax.set_title("Number of open credit lines")

df.NumberOfOpenCreditLinesAndLoans.plot(kind="hist",bins=50)

plt.show()
df.NumberOfOpenCreditLinesAndLoans.describe()
woe = summary_woe_func("NumberOfOpenCreditLinesAndLoans","SeriousDlqin2yrs",df)

representation_woe(woe["summary_woe"],woe['missing'])
cat_number = pd.qcut(df.NumberOfOpenCreditLinesAndLoans,q=10)

df.insert(7,"NumberOfOpenLines_Cat",cat_number)

woe = summary_woe_func("NumberOfOpenLines_Cat","SeriousDlqin2yrs",df)

representation_woe(woe["summary_woe"],woe['missing'])
group_1=['(3.0, 4.0]','(4.0, 5.0]']

group_2=['(9.0, 10.0]','(10.0, 12.0]']



NumberOfOpenLines_Cat_final = [pd.Interval(left=3.0,right=5.0,closed="right") if str(x) in group_1 else x for x in df["NumberOfOpenLines_Cat"] ]

NumberOfOpenLines_Cat_final = [pd.Interval(left=9.0,right=12.0,closed="right") if str(x) in group_2 else x for x in NumberOfOpenLines_Cat_final ]



df["NumberOfOpenLines_Cat_final"]=NumberOfOpenLines_Cat_final

woe_Income = summary_woe_func("NumberOfOpenLines_Cat_final","SeriousDlqin2yrs",df)

representation_woe(woe_Income["summary_woe"],woe_Income['missing'])
woe = summary_woe_func("NumberOfTimes90DaysLate","SeriousDlqin2yrs",df)

representation_woe(woe["summary_woe"],woe['missing'])
cat_numberTimes90Days = ["NeverLate" if x==0 else "OnceMoreLate" for x in df.NumberOfTimes90DaysLate]



df["cat_numberTimes90Days_final"]=cat_numberTimes90Days

woe = summary_woe_func("cat_numberTimes90Days_final","SeriousDlqin2yrs",df)

representation_woe(woe["summary_woe"],woe['missing'])
woe = summary_woe_func("NumberRealEstateLoansOrLines","SeriousDlqin2yrs",df)

representation_woe(woe["summary_woe"],woe['missing'])
catRealEstateLoans = ["2" if x>1 else str(x) for x in df.NumberRealEstateLoansOrLines]



df["catRealEstateLoans_final"]=catRealEstateLoans

woe = summary_woe_func("catRealEstateLoans_final","SeriousDlqin2yrs",df)

representation_woe(woe["summary_woe"],woe['missing'])
woe = summary_woe_func("NumberOfTime60-89DaysPastDueNotWorse","SeriousDlqin2yrs",df)

representation_woe(woe["summary_woe"],woe['missing'])
cat_NumberOfTime60_89 = ["0" if x==0 else "1" for x in df["NumberOfTime60-89DaysPastDueNotWorse"]]



df["cat_NumberOfTime60_89_final"]=cat_NumberOfTime60_89

woe = summary_woe_func("cat_NumberOfTime60_89_final","SeriousDlqin2yrs",df)

representation_woe(woe["summary_woe"],woe['missing'])
df.columns.values
liste_variable = [x for x in df.columns.values if bool(re.findall("final",x))]

IV_recap = pd.DataFrame({"Variable":liste_variable,"IV":[summary_woe_func(x,"SeriousDlqin2yrs",df)["IV"] for x in liste_variable]}).sort_values("IV",ascending=False)

IV_recap
features = pd.get_dummies(df.loc[:,liste_variable])

features
LR = LogisticRegression()

LR.fit(features,df.SeriousDlqin2yrs)
proba = LR.predict_proba(features)[:,1]

fpr, tpr, thresholds = sklearn.metrics.roc_curve(df.SeriousDlqin2yrs,proba)
plt.plot(fpr,tpr)

plt.show()
LR.score(features,df.SeriousDlqin2yrs)
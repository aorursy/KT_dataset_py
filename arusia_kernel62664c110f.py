import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import seaborn as sns
import matplotlib.pyplot as plt
print(os.listdir('../input'))
%matplotlib inline
#reading data frame
df_train = pd.read_csv("../input/cs-training.csv")
df_test = pd.read_csv("../input/cs-test.csv")
df_train=df_train.append(df_test); #Appending test data too!
print("Number of columns = ",len(df_train.columns))
print("Number of rows = ",len(df_train))
df_train.head(10)
#Histogram plot of age
fig, axs = plt.subplots(3,3)
fig.set_size_inches(18.5, 12)
sns.distplot(df_train["age"],ax=axs[0,0]);
sns.distplot(df_train["RevolvingUtilizationOfUnsecuredLines"],ax=axs[0,1]);
sns.distplot(df_train["DebtRatio"],ax=axs[0,2]);

g=sns.distplot(df_train["MonthlyIncome"].dropna(),ax=axs[1,0]);
g.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

sns.distplot(df_train["NumberOfOpenCreditLinesAndLoans"].dropna().astype('int'),ax=axs[1,1]);
sns.distplot(df_train["SeriousDlqin2yrs"].dropna().astype('int'),ax=axs[1,2],kde=False);

sns.distplot(df_train["NumberOfTime30-59DaysPastDueNotWorse"].dropna(),ax=axs[2,0]);
sns.distplot((df_train["NumberOfDependents"].dropna()).astype('int'),ax=axs[2,1]);
sns.distplot(df_train["NumberOfTimes90DaysLate"].dropna().astype('int'),ax=axs[2,2]);


fig, axs = plt.subplots(1,2)
fig.set_size_inches(12,4)
sns.distplot(df_train["NumberRealEstateLoansOrLines"].dropna().astype('int'),ax=axs[0]);
sns.distplot(df_train["NumberOfTime60-89DaysPastDueNotWorse"].dropna().astype('int'),ax=axs[1]);

for column in df_train.columns:
    print("Kurtosis  ",column, ": %f" % df_train[column].kurt())
#Correlation heat map
corr = df_train.corr(method="spearman")
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, square=True, annot=True);
fig,axs=plt.subplots(1,3)
fig.set_size_inches(20,4)
axs[0].scatter(df_train["NumberRealEstateLoansOrLines"],df_train["NumberOfOpenCreditLinesAndLoans"]);
axs[0].set_xlabel("NumberRealEstateLoansOrLines")
axs[0].set_ylabel("NumberOfOpenCreditLinesAndLoans")

axs[1].scatter(df_train["DebtRatio"],df_train["NumberRealEstateLoansOrLines"]);
axs[1].set_xlabel("DebtRatio")
axs[1].set_ylabel("NumberRealEstateLoansOrLines")

augdata=(df_train[["MonthlyIncome","NumberRealEstateLoansOrLines"]]).dropna()
axs[2].scatter(augdata["MonthlyIncome"],augdata["NumberRealEstateLoansOrLines"]);
axs[2].set_xlabel("MonthlyIncome")
axs[2].set_ylabel("NumberRealEstateLoansOrLines")
for tick in axs[2].get_xticklabels():
    tick.set_rotation(90)

plt.show()
fig,axs=plt.subplots(1,2)
fig.set_size_inches(11,4)

axs[0].scatter(df_train["DebtRatio"],df_train["NumberRealEstateLoansOrLines"]);
axs[0].set_xlim([100,40000])
axs[0].set_ylim([-1,10])

axs[1].scatter(df_train["MonthlyIncome"],df_train["NumberRealEstateLoansOrLines"]);
axs[1].set_xlim([0,100000])
axs[1].set_ylim([-1,50]);
#Pair wise plot 
sns.set()
important_columns = ['MonthlyIncome', 'DebtRatio', 'age', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate']
figs=sns.pairplot(df_train[important_columns].dropna(), size = 2.5, aspect=1.5)
for i in range(5):
    for j in range(2):
        for ticks in figs.axes[i,j].get_xticklabels():
            ticks.set_rotation(90)
plt.show();

#Monthly income statistics
df_train["MonthlyIncome"].describe()
#Monthly income univariate plot
fig, axs = plt.subplots(1,2)
fig.set_size_inches(12, 4)
#Zoomed out plot
g=sns.distplot(df_train["MonthlyIncome"].dropna(),ax=axs[0]);
g.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#Zoomed in plot
sns.distplot(df_train["MonthlyIncome"].loc[df_train["MonthlyIncome"]<0.5e5].dropna(),ax=axs[1]);
df_train_mod=df_train.loc[df_train["MonthlyIncome"]!=0.0].dropna().copy()
df_train_mod["MonthlyIncome"]=np.log10(df_train_mod["MonthlyIncome"])
g=sns.distplot(df_train_mod["MonthlyIncome"]);
g.axes.set_xlabel("Log10 of MonthlyIncome");
#Correlation heat map
corr = df_train_mod.corr(method="spearman")
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, square=True, annot=True);
df_train_mod2 = df_train_mod.loc[df_train_mod["DebtRatio"]!=0.0].copy()
df_train_mod2["DebtRatio"] = np.log10(df_train_mod2["DebtRatio"])
df_train_mod3 = df_train_mod2.loc[df_train_mod2["NumberRealEstateLoansOrLines"]!=0.0].copy()
df_train_mod3["NumberRealEstateLoansOrLines"] = np.log10(df_train_mod3["NumberRealEstateLoansOrLines"])
#Correlation heat map
corr = df_train_mod3.corr(method="spearman")
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, square=True, annot=True);
plt.scatter(df_train_mod3['MonthlyIncome'],df_train_mod3['DebtRatio']);
plt.xlabel('log10 of MonthlyIncome')
plt.ylabel('log10 of DebtRatio');
fig,axs=plt.subplots(1,2)
fig.set_size_inches(16, 4)
axs[0].scatter(df_train['SeriousDlqin2yrs'],df_train['MonthlyIncome'])
axs[0].set_ylabel('MonthlyIncome')
axs[0].set_xlabel('SeriousDlqin2yrs')
axs[1].scatter(df_train['RevolvingUtilizationOfUnsecuredLines'],df_train['MonthlyIncome']);
axs[1].set_ylabel('MonthlyIncome')
axs[1].set_xlabel('RevolvingUtilizationOfUnsecuredLines');



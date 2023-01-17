import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
## understanding data
mh = pd.read_csv('../input/mental-health-in-tech-2016/mental-heath-in-tech-2016_20161114.csv')
print(mh.shape)

#print(mh.columns[:5] # uncomment to print

## check data type
#print(mh.info()) # uncomment to print

## check numeric variables
#print(mh.describe()) # uncomment to print

## check missing values
#print(mh.isnull().sum()) # uncomment to print

#print(mh.head(2))# uncomment to print
## check missing pattern
print(mh.isnull().sum())
sns.distplot(mh.isnull().sum()) 

## make a copy of the original data
df = mh.copy()

########################## Columns with Missing values > 300 #####################################################################
## columns with more than 300 missing values --  behaviors behind the missing pattern
## Possibly there is a reason why the pariticipants tend to "refuse" to answer the questions -- worth further investigation

col_m = df.isnull().sum() > 300 
cols_m= col_m[col_m==True]
#print(len(cols_m))
df_missing = df[cols_m.index]
#print(df_missing.shape)
df_missing.columns
## select ONLY columns that have missing values <300 out of 1433 (missing less than 21% ) for further analysis
col = df.isnull().sum() < 300 
cols = col[col==True]
print(len(cols))
df1 = df[cols.index]
print(df1.shape)
# df1.columns
## check demographic information
fig,axs = plt.subplots(1,2)
fig1 =sns.distplot(df1["What is your age?"],ax = axs[0])

## remove outliers
df1["age"] = df1["What is your age?"]
df2 = df1[(18 < df1["age"])  & (df1["age"] < 65)]
fig2 = sns.distplot(df2["What is your age?"], ax = axs[1])
fig2.set_title("With outliers removed")
 ### 4.3 clean up messy data and recode
 #print(df1["What is your gender?"].unique())
mylist = ["Male","Man","man","male","MALE","female","Female", "F","M","f","m","woman","Woman"]

gender= df2["What is your gender?"].apply(lambda x: (x in mylist)) ## Return True/False
df2["What is your gender?"] = df2["What is your gender?"][gender]  ## Only retain True 

print(df1.shape)
#df1["What is your gender?"].unique()

## Recoding
mapping =({"male":"Male",
           "Man":"Male",
           "man":"Male",
           "m":"Male",
           "M":"Male",
           "MALE":"Male",
           "Male":"Male",
           "woman":"Female",
           "Woman":"Female",
           "Female":"Female",
           "female":"Female",
           "f":"Female",
           "F":"Female"})

df2['gender'] = df2["What is your gender?"].map(mapping)
print(df2['gender'].value_counts())
## Group items into sub groups for further analysis

## demographic
print("Demographic info")
print(df2.columns[39:43])
print()

## status
print("status")
print(df2.columns[[0,32,33,34,35,36,37,38]])
print()

##info about company
print("info about company")
print(df2.columns[[1,2]])
print()

## related to previous employers
print("Items related to previous employers")
print(df2.columns[15:25])
print()

## health insurance coverage
print("Mental health coverage&resources")
print(df2.columns[[3,4,5]])
print()

## Mental health conseuqences
print("Mental health consequences")
print(df2.columns[[6,7,8,9,10,11,28,29]])
print()

## Past experiences
print()
print("Past experiences")
print(df2.columns[[13,31]])

## Willing to talk
print()
print("Willing to talk about")
print(df2.columns[[26,27,30]])
print(df2.shape)
print()
print(df2.iloc[:,2].value_counts(dropna = False))
print()
df_f = df2[df2.iloc[:,2] == 1] ## include ONLY Tech company workers for further analyisis ==> N =878
print(df_f.shape)
fig,axs = plt.subplots(1,3, figsize=(18,4))
#fig.suptitle("mentall health by gender")

## Mental health
mhc_labels = ["Yes","No","Maybe"]
mhc_size = df_f.iloc[:,34].value_counts()
explode = (0,0.1,0)
axs[0].pie(mhc_size, labels = mhc_labels,explode = explode,shadow = True, startangle = 90,autopct='%1.1f%%')
axs[0].set_title(df1.columns[34])

## mhc by gender
table = df_f.pivot_table(index = df_f.columns[34], columns = "gender" ,values = df_f.columns[0], aggfunc = len)
table_per = 100 * table/table.sum()

print(table)
print(table_per)
vals = np.array(table)

cmap = plt.get_cmap("Accent")
outer_colors = cmap(np.arange(3)*4)
inner_colors = cmap([1,2,5,6])

outer_label = ["Yes","No","Maybe"]  ## Be careful with the order of labeling

axs[1].pie(vals.sum(axis=1), radius=1, colors=outer_colors,labels = outer_label,
       wedgeprops=dict(width=0.3, edgecolor='w'),autopct='%1.1f%%')

axs[1].pie(vals.flatten(), radius=1-0.3, colors=inner_colors,
       wedgeprops=dict(width=0.3, edgecolor='w'),autopct='%1.1f%%')   ## can't do inner label, needs to be improved, otherwise 
axs[1].set_title("Mental health by Gender")
                                                                      ## the nested pie chart is limited in usefulness

table_per.plot(kind="bar",ax = axs[2]) ## a bar plot conveys information more clearly in this case
axs[2].set_title("Mental health by Gender")
## Chi-square test to statistical validate
from scipy.stats import chi2_contingency
chi2, p, dof, ex = chi2_contingency(table, correction=False)
print("chi-square: ", chi2)
print("p value: ", p)
print("degrees of freedom: ", dof)
print("expected", ex)
ax = sns.countplot(df1.iloc[:,3])

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,'{:1.2f}'.format(100*height/float(len(df_f))),
            ha="center") 
ax = sns.countplot(df_f.iloc[:,6])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,'{:1.2f}'.format(100*height/float(len(df_f))),
            ha="center")
ax = sns.countplot(df_f.iloc[:,6])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,'{:1.2f}'.format(100*height/float(len(df_f))),
            ha="center")
##  Descriptive 

plt.figure(figsize=(16,6))
df_f1 = df_f.iloc[:,8:12]
ax = sns.countplot(hue="variable", x="value",  data=pd.melt(df_f1))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,'{:1.2f}'.format(100*height/float(len(df_f))),
            ha="center") 
#ax.legend("upper center")
#warnings.simplefilter("ignore")
#df_f2 = df_f.iloc[:,28:30]
#sns.countplot(hue="variable", x="value", data=pd.melt(df_f2),ax = axs[1])


##"https://stackoverflow.com/questions/46223224/matplotlib-plot-countplot-for-two-or-more-column-on-single-plot"
## Seaborn usually works best with long form datasets. 
##I.e. instead of 3 columns with different options for each attribute you would have two columns, 
##one for the options and one for the attributes. This can easily be created via pd.melt. 
##Then the hue value can be used on the "options" column:
plt.figure(figsize=(18,8))
df_f1 = df_f.iloc[:,37:39]
ax = sns.countplot(hue="variable", x="value", 
                   order =["Not applicable to me","Often","Sometimes","Rarely","Never"],  data=pd.melt(df_f1))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,'{:1.2f}'.format(100*height/float(len(df_f))),
            ha="center") 
plt.figure(figsize=(12,4))
df_f1 = df_f.iloc[:,26:28]
ax = sns.countplot(hue="variable", x="value", data=pd.melt(df_f1))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,'{:1.2f}'.format(100*height/float(len(df_f))),
            ha="center") 
## Q5: Is gender related to the willingness to tell if have mental health conditions?

pivot_table = df_f.pivot_table(index = df_f.columns[27], columns = "gender", values = df_f.columns[0], aggfunc = len)
#pivot_table.plot(kind="bar")

pivot_table_per = 100 * pivot_table/pivot_table.sum()
print(pivot_table_per)
pivot_table_per.plot(kind="bar")

chi2, p, dof, ex = chi2_contingency(pivot_table, correction=False)
print("chi-square: ", chi2)
print("p value: ", p)
print("degrees of freedom: ", dof)
print("expected", ex)
pivot_table = df_f.pivot_table(index = df_f.columns[27], columns = "gender", values = "age", aggfunc = np.mean)
pivot_table.plot(kind="bar")
plt.title("by age and gender")

import statsmodels.api as sm
from statsmodels.formula.api import ols
# Ordinary Least Squares (OLS) model

df_f["bring_up"] = df_f[df_f.columns[27]]
model = ols('age ~ C(gender) + C(bring_up)', data=df_f).fit()
anova_table = sm.stats.anova_lm(model, typ=3)
anova_table
pivot_table = df_f.pivot_table(index = df_f.columns[3], columns = df_f.columns[27], values = df_f.columns[0], aggfunc = len)
#pivot_table.plot(kind="bar")

pivot_table_per = 100 * pivot_table/pivot_table.sum()
#print(pivot_table_per)
pivot_table_per.plot(kind="bar")

chi2, p, dof, ex = chi2_contingency(pivot_table, correction=False)
print("chi-square: ", chi2)
print("p value: ", p)
print("degrees of freedom: ", dof)
print("expected", ex)
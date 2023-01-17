import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.decomposition import PCA
import seaborn as sns
sns.set(style="darkgrid")
pd.options.display.width = 100
pd.options.display.precision = 4
pd.options.display.max_columns = 10
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/train.csv')
# Tabulate single variables with proportions
def tabulate(var):
    target = pd.crosstab(var, columns='count')
    target['prop'] = target / target.sum()
    print(target)
# Crosstabulate two variables with proportions
def crosstab(var1, var2):
    cross = pd.crosstab(index=var1, columns=var2, margins=True)
    prop_cross = cross / cross.loc["All", "All"]
    cross_concat = pd.concat([cross, prop_cross], axis=1)
    print(cross_concat)
# Get mean of a variable grouped by another variable
def get_mean_by_group(varlist, group):
    mean_df = pd.DataFrame()
    for var in varlist:
        mean_df[var.name] = var.groupby(group).mean()
    print(mean_df)
# Create a correlation heatmap
def corr_heatmap(df):
    corr = df.corr()
    fig, (ax) = plt.subplots(1, 1, figsize=(14,8))
    hm = sns.heatmap(corr, ax=ax, cmap="coolwarm",annot=True, fmt='.4f', linewidths=.05)
    fig.subplots_adjust(top=0.95, left=0.2)
    fig.suptitle('Correlation Heatmap', fontsize=14, fontweight='bold')
def cleandatasetyesno(data):
    varlists=['dependency','edjefe','edjefa']
    for varlist in varlists: 
        data[varlist].replace(('yes', 'no'), (1, 0), inplace=True)
        data[varlist]=pd.to_numeric(data[varlist])
    return data
# Graph for three variables
def three_var_graph(varlist, titles, asset):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(12,4), sharex=True, sharey=True)
    f = sns.catplot("Target", varlist[0], data=df, ci=None, ax=ax1, kind="bar")
    plt.close(f.fig)
    ax1.set_title(titles[0])
    ax1.set_ylim([0,1])
    ax1.set_ylabel("")
    g = sns.catplot("Target", varlist[1], data=df, ci=None, ax=ax2, kind="bar")
    plt.close(g.fig)
    ax2.set_title(titles[1])
    ax2.set_ylabel("")
    h = sns.catplot("Target", varlist[2], data=df, ci=None, ax=ax3, kind="bar")
    plt.close(h.fig)
    ax3.set_title(titles[2])
    ax3.set_ylabel("")
    plt.suptitle(asset, y=1)
# Graph for four variables
def four_var_graph(varlist, titles, condition):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10,8))
    c = sns.catplot("Target", varlist[0], data=df, ci=None, ax=ax1, kind="bar")
    plt.close(c.fig)
    ax1.set_ylim([0,1])
    ax1.set_ylabel("")
    ax1.set_title(titles[0])
    g = sns.catplot("Target", varlist[1], data=df, ci=None, ax=ax2, kind="bar")
    plt.close(g.fig)
    ax2.set_ylabel("")
    ax2.set_title(titles[1])
    h = sns.catplot("Target", varlist[2], data=df, ci=None, ax=ax3, kind="bar")
    plt.close(h.fig)
    ax3.set_ylabel("")
    ax3.set_title(titles[2])
    i = sns.catplot("Target", varlist[3], data=df, ci=None, ax=ax4, kind="bar")
    plt.close(i.fig)
    ax4.set_ylabel("")
    ax4.set_title(titles[3])
    plt.suptitle(condition,y=0.93)
# Graph for five variables
def five_var_graph(varlist, titles, condition):
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey=True, figsize=(12,8))
    c = sns.catplot("Target", varlist[0], data=df, ci=None, ax=ax1, kind="bar")
    plt.close(c.fig)
    ax1.set_ylim([0,1])
    ax1.set_ylabel("")
    ax1.set_title(titles[0])
    g = sns.catplot("Target", varlist[1], data=df, ci=None, ax=ax2, kind="bar")
    plt.close(g.fig)
    ax2.set_ylabel("")
    ax2.set_title(titles[1])
    h = sns.catplot("Target", varlist[2], data=df, ci=None, ax=ax3, kind="bar")
    plt.close(h.fig)
    ax3.set_ylabel("")
    ax3.set_title(titles[2])
    i = sns.catplot("Target", varlist[3], data=df, ci=None, ax=ax4, kind="bar")
    plt.close(i.fig)
    ax4.set_ylabel("")
    ax4.set_title(titles[3])
    j = sns.catplot("Target", varlist[4], data=df, ci=None, ax=ax5, kind="bar")
    plt.close(j.fig)
    ax5.set_ylabel("")
    ax5.set_title(titles[4])
    ax5.set_ylabel("")
    ax6.axis('off')
    plt.suptitle(condition,y=0.97)
# Graph for six variables
def six_var_graph(varlist, titles, condition):
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12,8))
    c = sns.catplot("Target", varlist[0], data=df, ci=None, ax=ax1, kind="bar")
    plt.close(c.fig)
    ax1.set_ylim([0,1])
    ax1.set_ylabel("")
    ax1.set_title(titles[0])
    g = sns.catplot("Target", varlist[1], data=df, ci=None, ax=ax2, kind="bar")
    plt.close(g.fig)
    ax2.set_ylabel("")
    ax2.set_title(titles[1])
    h = sns.catplot("Target", varlist[2], data=df, ci=None, ax=ax3, kind="bar")
    plt.close(h.fig)
    ax3.set_ylabel("")
    ax3.set_title(titles[2])
    i = sns.catplot("Target", varlist[3], data=df, ci=None, ax=ax4, kind="bar")
    plt.close(i.fig)
    ax4.set_ylabel("")
    ax4.set_title(titles[3])
    j = sns.catplot("Target", varlist[4], data=df, ci=None, ax=ax5, kind="bar")
    plt.close(j.fig)
    ax5.set_ylabel("")
    ax5.set_title(titles[4])
    k = sns.catplot("Target", varlist[5], data=df, ci=None, ax=ax6, kind="bar")
    plt.close(k.fig)
    ax6.set_ylabel("")
    ax6.set_title(titles[5])
    plt.suptitle(condition,y=0.95)
# Number of individual records in dataset
len(df)
# Number of households
len(df['idhogar'].unique())
# Average household size
round(len(df) / len(df['idhogar'].unique()), 2)
df.head()
df_slim=df.drop(['Id','idhogar','Target'], axis=1)
dfType=df_slim.dtypes
dfType=pd.DataFrame(dfType.value_counts())
dfType=dfType.reset_index()
dfType.columns=["VarType","value"]
dfType['percent']=(dfType['value']/len(df_slim.columns))*100

dfType['Variables']=""
vartypes=["object","float64", "int64"]
for vartype in vartypes:
    ObjectVarNam=df_slim.columns[df_slim.dtypes==vartype].tolist()
    strObjectVarNam=', '.join('"{0}"'.format(w) for w in ObjectVarNam)
    dfType.loc[(dfType['VarType']==vartype), "Variables"]= strObjectVarNam

dfType
# Number of household heads
tabulate(df['parentesco1'])
missing_hh = df['parentesco1'].groupby(df['idhogar']).sum()
missing_hh[missing_hh==0]
# Creating single variable for area (only for exploratory analysis purposes)
df["area"] = "Urban"
df.loc[df["area2"]==1, "area"] = "Rural"
tabulate(df["area"])
# Creating single variable for region (only for exploratory analysis purposes)
df["region"] = "Central"
df.loc[df["lugar2"]==1, "region"] = "Chorotega"
df.loc[df["lugar3"]==1, "region"] = "Pacifico Central"
df.loc[df["lugar4"]==1, "region"] = "Brunca"
df.loc[df["lugar5"]==1, "region"] = "Huetar Atlantica"
df.loc[df["lugar6"]==1, "region"] = "Huetar Norte"
tabulate(df["region"])
missing_hh_list = ['03c6bdf85', '09b195e7a', '1367ab31d', '1bc617b23', '374ca5a19', '61c10e099',
                   '6b1b2405f', '896fe6d3e', 'a0812ef17', 'ad687ad89', 'b1f4d89d7', 'bfd5067c2',
                   'c0c8a5013', 'd363d9183', 'f2bfa75c4']
tabulate(df.area[df.idhogar.isin(missing_hh_list)])
tabulate(df.region[df.idhogar.isin(missing_hh_list)])
missing = pd.DataFrame(df.isnull().sum())
missing['Proportion'] = missing/len(df)
missing.columns=["Missing", "Proportion"]
missing[missing["Missing"] > 0]
# Number of households for which monthly rent payment is missing
len(df.idhogar[df['v2a1'].isnull()].unique())
sns.distplot(df.v2a1[df['v2a1'].isnull()==False])
# Number of individuals who rent
tabulate(df.tipovivi3)
# v2a1 is missing for individuals who don't rent
tabulate(df.tipovivi3[df['v2a1'].isnull()])
# Number of individuals who own and are paying installments
tabulate(df.tipovivi2)
tabulate(df.tipovivi2[df['v2a1'].isnull()])
# Number of people who own a fully paid house
tabulate(df.tipovivi1)
tabulate(df.tipovivi1[df['v2a1'].isnull()])
# Households for which number of tablets per household is missing
len(df.idhogar[df['v18q1'].isnull()].unique())
tabulate(df.v18q)
tabulate(df.v18q1)
tabulate(df.v18q[df['v18q1'].isnull()])
df.loc[df['v18q1'].isnull(), 'v18q1'] = 0
tabulate(df.rez_esc)
pd.crosstab(df.rez_esc, df.age)
tabulate(df.age[df['rez_esc'].isnull() & (df.age < 18)])

df.idhogar[df.meaneduc.isnull()]

df.hhsize[df.meaneduc.isnull()]
df.escolari[df.meaneduc.isnull()]
df.age[df.meaneduc.isnull()]
temp_df = df['escolari'][df['age']>=18].groupby(df['idhogar']).mean().to_frame()
temp_df.rename(columns={'escolari': 'meaneduc_new'}, inplace=True)
temp_df['meaneduc_new'].describe()
df = df.merge(temp_df, left_on="idhogar", right_index=True, how="outer")
df[["idhogar", "age", "escolari", "meaneduc_new", "meaneduc"]].head(10)
df['meaneduc_new'].isnull().sum()
df[["idhogar", "age", "meaneduc", "escolari"]][df['meaneduc_new'].isnull()]
df["meaneduc_new"].fillna(8.6161, inplace=True)
df["meaneduc_new"].isnull().sum()
df['meaneduc_new'].describe()
sns.distplot(df.meaneduc_new)
tabulate(df.Target)
# Creating single variable for sex
df["sex"] = "Male"
df.loc[df["female"]==1, "sex"] = "Female"
tabulate(df["sex"])
crosstab(df.Target, df.area)
get_mean_by_group([df.instlevel1, df.instlevel2, df.instlevel3, df.instlevel4,
                   df.instlevel5, df.instlevel6, df.instlevel7, df.instlevel8,
                   df.instlevel9], df.Target)
get_mean_by_group([df.v18q, df.refrig, df.computer, df.television, df.mobilephone, df.qmobilephone], df.Target)
get_mean_by_group([df.epared1, df.epared2, df.epared3, df.etecho1, df.etecho2, df.etecho3, df.eviv1, df.eviv2, df.eviv3], df.Target)
# Water
get_mean_by_group([df.abastaguadentro, df.abastaguafuera, df.abastaguano], df.Target)
# Toilet
get_mean_by_group([df.sanitario1, df.sanitario2, df.sanitario3, df.sanitario5, df.sanitario6], df.Target)
# Rubbish disposal
get_mean_by_group([df.elimbasu1, df.elimbasu2, df.elimbasu3, df.elimbasu4, df.elimbasu5, df.elimbasu6], df.Target)
# Cooking fuel
get_mean_by_group([df.energcocinar1, df.energcocinar2, df.energcocinar3, df.energcocinar4], df.Target)
# Electricity
get_mean_by_group([df.public, df.planpri, df.noelec, df.coopele], df.Target)
tabulate(df.dependency)
tabulate(df.SQBdependency[df.dependency=="yes"])
tabulate(df.SQBdependency[df.dependency=="no"])
tabulate(df.edjefe)
tabulate(df.SQBedjefe[df.edjefe=="yes"])
tabulate(df.SQBedjefe[df.edjefe=="no"])
tabulate(df.edjefa)
# Replacing yes/no values with 1/0 using helper function
df = cleandatasetyesno(df)
# Categorical variable for dependency
df['depend_cat'] = 1
df.loc[df['dependency']==1, 'depend_cat'] = 2
df.loc[df['dependency'] > 1, 'depend_cat'] = 3
tabulate(df.depend_cat)
get_mean_by_group([df.Target], df.depend_cat)
tabulate(df["edjefe"])
df.loc[(df["parentesco1"]==1) & (df["female"]==1), "edjefe"] = 99 # 99 is coded if the household head is a woman or individual not household head
df.loc[df["parentesco1"]==0, "edjefe"] = 99
tabulate(df["edjefe"])
df.loc[(df["parentesco1"]==1) & (df["male"]==1), "edjefa"] = 99 # 99 is coded if the household head is a man or individual not household head
df.loc[df["parentesco1"]==0, "edjefa"] = 99
tabulate(df["edjefa"])
print("Number of male household heads: {}".format(len(df.parentesco1[(df["parentesco1"]==1) & (df["male"]==1)])))
print("Number of female household heads: {}".format(len(df.parentesco1[(df["parentesco1"]==1) & (df["female"]==1)])))
# Creating categorical variables for household head female education
df["fhh_educ_cat1"] = 0
df.loc[df.edjefa < 6, "fhh_educ_cat1"] = 1
df["fhh_educ_cat2"] = 0
df.loc[(df.edjefa >= 6) & (df.edjefa < 12), "fhh_educ_cat2"] = 1
df["fhh_educ_cat3"] = 0
df.loc[(df.edjefa >= 12) & (df.edjefa < 99), "fhh_educ_cat3"] = 1
# One variable with categories for female household head education
df["fhh_educ"] = 0
df.loc[df.fhh_educ_cat1==1, "fhh_educ"] = 1
df.loc[df.fhh_educ_cat2==1, "fhh_educ"] = 2
df.loc[df.fhh_educ_cat3==1, "fhh_educ"] = 3
tabulate(df.fhh_educ)
# Creating categorical variables for household head male education
df["mhh_educ_cat1"] = 0
df.loc[df.edjefe < 6, "mhh_educ_cat1"] = 1
df["mhh_educ_cat2"] = 0
df.loc[(df.edjefe >= 6) & (df.edjefe < 12), "mhh_educ_cat2"] = 1
df["mhh_educ_cat3"] = 0
df.loc[(df.edjefe >= 12) & (df.edjefe < 99), "mhh_educ_cat3"] = 1
# One variable with categories for male household head education
df["mhh_educ"] = 0
df.loc[df.mhh_educ_cat1==1, "mhh_educ"] = 1
df.loc[df.mhh_educ_cat2==1, "mhh_educ"] = 2
df.loc[df.mhh_educ_cat3==1, "mhh_educ"] = 3
tabulate(df.mhh_educ)
# Creating feature cross for male and female household heads
df["male_hh"] = df["parentesco1"] * df["male"]
df["female_hh"] = df["parentesco1"] * df["female"]
corr_heatmap(df[["Target", "epared1", "epared2", "epared3", "etecho1", "etecho2", "etecho3", "eviv1", "eviv2", "eviv3"]])
# water and electricity
corr_heatmap(df[["Target", "abastaguadentro", "abastaguafuera", "abastaguano", "public", "planpri", "noelec", "coopele"]])
# Water and toilet
corr_heatmap(df[["Target", "abastaguadentro", "abastaguafuera", "abastaguano", 
               "sanitario1", "sanitario2", "sanitario3", "sanitario5", "sanitario6"]])
# Energy and toilet
corr_heatmap(df[["Target", "energcocinar1", "energcocinar2", "energcocinar3", "energcocinar4",
                 "sanitario1", "sanitario2", "sanitario3", "sanitario5", "sanitario6"]])
# Rubbish disposal and toilet
corr_heatmap(df[["Target", "elimbasu1", "elimbasu2", "elimbasu3", "elimbasu4", "elimbasu6",
    "sanitario1", "sanitario2", "sanitario3", "sanitario5", "sanitario6"]])
# why I didn't include elimbasu5 in the corr_heatmap
tabulate(df['elimbasu5'])
corr_heatmap(df[["Target", "overcrowding", "tipovivi1", "tipovivi2", "tipovivi3",
                 "tipovivi4", "tipovivi5", "computer", "television", "mobilephone", "qmobilephone"]])
corr_heatmap(df[["Target", "v18q", "v18q1", "computer", "television", "mobilephone", "qmobilephone"]])
sns.countplot("Target", data=df)
ax = sns.barplot(x="Target", y="Target", data=df, estimator=lambda x: len(x) / len(df) * 100)
ax.set(xlabel='Target', ylabel='Percent')
# Average number of rooms per household by region and target
sns.catplot("Target", "rooms", kind="box", col="region", col_wrap=3, height=4, data=df)
# Average years of education by region and target
sns.catplot("Target", "escolari", kind="box", col="region", col_wrap=3, height=4, data=df)
# Dependency by region and target
sns.catplot("Target", "dependency", kind="bar", col="region", col_wrap=3, height=4, ci=None, data=df)
# Overcrowding by region and target
sns.catplot("Target", "overcrowding", kind="bar", col="region", col_wrap=3, height=4, ci=None, data=df)
# Average household size by region and target
sns.catplot("Target", "hogar_total", kind="bar", col="region", col_wrap=3, height=4, ci=None, data=df)
sns.set(style="white")
df["v2a1_2"] = np.log(df["v2a1"])
plt.subplots(figsize=(10,6))
sns.scatterplot("escolari", "v2a1_2", hue="area", alpha=0.7, data=df)
plt.ylabel("Monthly rent in logs")
ax = sns.catplot("Target", "v2a1_2", kind="box", data=df.query("v2a1 < 300000"), height=6, aspect=1.2, col="area")
ax.set(ylabel="Monthly rent in logs")
ax = sns.catplot("Target", "v2a1_2", kind="box", data=df.query("v2a1 < 300000"), height=4, aspect=1.2, col="region", col_wrap=3)
ax.set(ylabel="Monthly rent in logs")
three_var_graph(["etecho1", "etecho2", "etecho3"], ["Bad", "Regular", "Good"], "Roof")
three_var_graph(["epared1", "epared2", "epared3"], ["Bad", "Regular", "Good"], "Walls")
three_var_graph(["eviv1", "eviv2", "eviv3"], ["Bad", "Regular", "Good"], "Floor")
tabulate(df["computer"])
# Total number of computers by household
temp_df = df["computer"].groupby(df["idhogar"]).sum().to_frame()
df = df.merge(temp_df, left_on="idhogar", right_index=True, how="outer")
tabulate(df["computer_y"])
# Total number of televisions by household
temp_df = df["television"].groupby(df["idhogar"]).sum().to_frame()
df = df.merge(temp_df, left_on="idhogar", right_index=True, how="outer")
tabulate(df["television_y"])
three_var_graph(["computer_y", "television_y", "v18q1"], ["Computer", "Television", "Tablets"], "Average assets per household")
five_var_graph(["tipovivi1", "tipovivi2", "tipovivi3", "tipovivi4", "tipovivi5"], 
             ["Owned", "Installments", "Rented", "Precarious", "Other"],
             "Housing")
five_var_graph(["sanitario1", "sanitario2", "sanitario3", "sanitario5", "sanitario6"], 
             ["No toilet", "Sewer", "Septic tank", "Black hole or latrine", "Other"],
             "Toilet")
four_var_graph(["energcocinar1", "energcocinar2", "energcocinar3", "energcocinar4"],
               ["None", "Electricity", "Gas", "Wood or charcoal"],
                "Cooking fuel")
sns.catplot("Target", "qmobilephone", kind="bar", ci=None, data=df)
four_var_graph(["public", "planpri", "noelec", "coopele"],
               ["Public", "Private", "None", "Cooperative"],
               "Electricity provider")
six_var_graph(["elimbasu1", "elimbasu2", "elimbasu3", "elimbasu4", "elimbasu5", "elimbasu6"],
              ["Tanker truck", "Buried", "Burned", "Thrown away", "In river", "Other"],
              "Rubbish disposal")
f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(16,8))
c = sns.catplot("Target", "paredblolad", data=df, ci=None, ax=ax1, kind="bar")
plt.close(c.fig)
ax1.set_ylim([0,1])
ax1.set_ylabel("")
ax1.set_title("Block or brick")
g = sns.catplot("Target", "paredzocalo", data=df, ci=None, ax=ax2, kind="bar")
plt.close(g.fig)
ax2.set_ylabel("")
ax2.set_title("Socket")
h = sns.catplot("Target", "paredpreb", data=df, ci=None, ax=ax3, kind="bar")
plt.close(h.fig)
ax3.set_ylabel("")
ax3.set_title("Cement")
i = sns.catplot("Target", "pareddes", data=df, ci=None, ax=ax4, kind="bar")
plt.close(i.fig)
ax4.set_ylabel("")
ax4.set_title("Waste material")
j = sns.catplot("Target", "paredmad", data=df, ci=None, ax=ax5, kind="bar")
plt.close(j.fig)
ax5.set_ylabel("")
ax5.set_title("Wood")
k = sns.catplot("Target", "paredzinc", data=df, ci=None, ax=ax6, kind="bar")
plt.close(k.fig)
ax6.set_ylabel("")
ax6.set_title("Zinc")
l = sns.catplot("Target", "paredfibras", data=df, ci=None, ax=ax7, kind="bar")
plt.close(l.fig)
ax7.set_ylabel("")
ax7.set_title("Natural fibers")
m = sns.catplot("Target", "paredother", data=df, ci=None, ax=ax8, kind="bar")
plt.close(m.fig)
ax8.set_ylabel("")
ax8.set_title("Other")
plt.suptitle("Wall material",y=0.95)
six_var_graph(["pisomoscer", "pisocemento", "pisoother", "pisonatur", "pisonotiene", "pisomadera"],
              ["Mosaic or ceramic", "Cement", "Other", "Natural material", "No floor", "Wood"],
              "Floor material")
four_var_graph(["techozinc", "techoentrepiso", "techocane", "techootro"],
               ["Metal foil or zinc", "Fiber cement", "Natural fibers", "Other"],
               "Roof material")
sns.catplot("Target", "cielorazo", kind="bar", ci=None, data=df)
plt.ylim([0,1])
plt.ylabel("Have a ceiling")
sns.catplot("Target", "dis", kind="bar", ci=None, data=df)
plt.ylim([0,1])
plt.ylabel("Disabled")
f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(16,8))
c = sns.catplot("Target", "estadocivil1", data=df, ci=None, ax=ax1, kind="bar")
plt.close(c.fig)
ax1.set_ylim([0,1])
ax1.set_ylabel("")
ax1.set_title("Less than 10 years")
g = sns.catplot("Target", "estadocivil2", data=df, ci=None, ax=ax2, kind="bar")
plt.close(g.fig)
ax2.set_ylabel("")
ax2.set_title("Free or coupled union")
h = sns.catplot("Target", "estadocivil3", data=df, ci=None, ax=ax3, kind="bar")
plt.close(h.fig)
ax3.set_ylabel("")
ax3.set_title("Married")
i = sns.catplot("Target", "estadocivil4", data=df, ci=None, ax=ax4, kind="bar")
plt.close(i.fig)
ax4.set_ylabel("")
ax4.set_title("Divorced")
j = sns.catplot("Target", "estadocivil5", data=df, ci=None, ax=ax5, kind="bar")
plt.close(j.fig)
ax5.set_ylabel("")
ax5.set_title("Separated")
k = sns.catplot("Target", "estadocivil6", data=df, ci=None, ax=ax6, kind="bar")
plt.close(k.fig)
ax6.set_ylabel("")
ax6.set_title("Widower")
l = sns.catplot("Target", "estadocivil7", data=df, ci=None, ax=ax7, kind="bar")
plt.close(l.fig)
ax7.set_ylabel("")
ax7.set_title("Single")
ax8.axis("off")
plt.suptitle("Civil status",y=0.95)
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10,8))
c = sns.catplot("Target", "hogar_nin", data=df, ci=None, ax=ax1, kind="bar")
plt.close(c.fig)
ax1.set_ylim([0,5])
ax1.set_ylabel("")
ax1.set_title("0-19 years")
g = sns.catplot("Target", "hogar_adul", data=df, ci=None, ax=ax2, kind="bar")
plt.close(g.fig)
ax2.set_ylabel("")
ax2.set_title("Adults")
h = sns.catplot("Target", "hogar_mayor", data=df, ci=None, ax=ax3, kind="bar")
plt.close(h.fig)
ax3.set_ylabel("")
ax3.set_title("65+ years")
i = sns.catplot("Target", "hogar_total", data=df, ci=None, ax=ax4, kind="bar")
plt.close(i.fig)
ax4.set_ylabel("")
ax4.set_title("Total")
plt.suptitle("Household members",y=0.93)
df_materials = df[["paredblolad", "paredzocalo", "paredpreb", "pareddes", "paredmad", "paredzinc",
                   "paredfibras", "paredother", "pisomoscer", "pisocemento", "pisoother", "pisonatur",
                   "pisonotiene", "pisomadera", "techozinc", "techoentrepiso", "techoentrepiso", "techocane",
                   "techootro"]]
pca_materials = PCA().fit(df_materials)
print("Explained variance by component: %s" % pca_materials.explained_variance_ratio_)
pca_materials = pca_materials.transform(df_materials)
pca_mat_df = pd.DataFrame(data=pca_materials)
pca_mat_df[pca_mat_df.columns[0:6]].head(10)
pca_mat_df = pca_mat_df[pca_mat_df.columns[0:6]]
pca_mat_df.columns = ['pca_mat_1', 'pca_mat_2', 'pca_mat_3', 'pca_mat_4', 'pca_mat_5', 'pca_mat_6']
pca_mat_df.head()
# No correlation
corr_heatmap(pca_mat_df)
len(pca_mat_df)
pca_mat_df.isnull().sum()
df_living = df[["v14a", "cielorazo", "abastaguadentro", "abastaguafuera", "abastaguano", "public", "planpri",
                "noelec", "coopele", "sanitario1", "sanitario2", "sanitario3", "sanitario5", "sanitario6",
                "energcocinar1", "energcocinar2", "energcocinar3", "energcocinar4", "elimbasu1", "elimbasu2",
                "elimbasu3", "elimbasu4", "elimbasu5", "elimbasu6", "epared1", "epared2", "epared3",
                "etecho1", "etecho2", "etecho3", "eviv1", "eviv2", "eviv3"]]
pca_living = PCA().fit(df_living)
print("Explained variance by component: %s" % pca_living.explained_variance_ratio_)
pca_living = pca_living.transform(df_living)
pca_liv_df = pd.DataFrame(data=pca_living)
pca_liv_df[pca_liv_df.columns[0:10]].head(10)
pca_liv_df = pca_liv_df[pca_liv_df.columns[0:10]]
pca_liv_df.columns = ['pca_liv_1', 'pca_liv_2', 'pca_liv_3', 'pca_liv_4', 'pca_liv_5', 'pca_liv_6',
                      'pca_liv_7', 'pca_liv_8', 'pca_liv_9', 'pca_liv_10']
pca_liv_df.head()
# No correlation
corr_heatmap(pca_liv_df)
len(pca_liv_df)
pca_liv_df.isnull().sum()
pca_mat_liv_df = pd.concat([pca_mat_df, pca_liv_df], axis=1)
pca_mat_liv_df.head()
df_assets = df[['rooms', 'bedrooms', 'v18q1', 'computer_y', 'television_y', 'qmobilephone', 'refrig']]
pca_assets = PCA().fit(df_assets)
print("Explained variance by component: %s" % pca_assets.explained_variance_ratio_)
pca_assets = pca_assets.transform(df_assets)
pca_assets_df = pd.DataFrame(data=pca_assets)
pca_assets_df[pca_assets_df.columns[0:4]].head(10)
pca_assets_df = pca_assets_df[pca_assets_df.columns[0:4]]
pca_assets_df.columns = ['pca_assets_1', 'pca_assets_2', 'pca_assets_3', 'pca_assets_4']
pca_assets_df.head()
# No correlation
corr_heatmap(pca_assets_df)
len(pca_assets_df)
pca_assets_df.isnull().sum()
pca_df = pd.concat([pca_mat_liv_df, pca_assets_df], axis=1)
pca_df.head()
features = df[["hacdor", "hacapo", "r4h1", "r4h2", "r4m1", "r4m2", "tamhog", "escolari", "dis", "male", "female",
               "hogar_nin", "hogar_adul", "hogar_mayor", "instlevel1", "instlevel2", "instlevel3", "instlevel4",
               "instlevel5", "instlevel6", "instlevel7", "instlevel8", "instlevel9", "overcrowding", "tipovivi1",
               "tipovivi2", "tipovivi3", "tipovivi4", "tipovivi5", "lugar1", "lugar2", "lugar3", "lugar4", 
               "lugar5", "lugar6", "area1", "area2", "SQBage", "meaneduc_new", "depend_cat", "fhh_educ", "mhh_educ",
               "male_hh", "female_hh", "Target"]]
features = pd.concat([features, pca_df], axis=1)
features.head()
len(features)
missing = pd.DataFrame(features.isnull().sum())
missing['Proportion'] = missing/len(features)
missing.columns=["Missing", "Proportion"]
missing[missing["Missing"] > 0]
df_test = pd.read_csv('../input/test.csv')
# Sample size (number of individuals)
len(df_test)
# Number of households
len(df_test['idhogar'].unique())
# Average household size
round(len(df_test) / len(df_test['idhogar'].unique()), 2)
tabulate(df_test["parentesco1"])
missing = pd.DataFrame(df_test.isnull().sum())
missing['Proportion'] = missing/len(df_test)
missing.columns=["Missing", "Proportion"]
missing[missing["Missing"] > 0]
df_test.head()
# Households for which number of tablets per household is missing
len(df_test.idhogar[df_test['v18q1'].isnull()].unique())
tabulate(df_test.v18q)
tabulate(df_test.v18q1)
tabulate(df_test.v18q[df_test['v18q1'].isnull()])
df_test.loc[df_test['v18q1'].isnull(), 'v18q1'] = 0
tabulate(df_test.v18q1)
df_test[["idhogar", "age", "female"]][df_test.meaneduc.isnull()]
temp_df = df_test['escolari'][df_test['age']>=18].groupby(df_test['idhogar']).mean().to_frame()
temp_df.rename(columns={'escolari': 'meaneduc_new'}, inplace=True)
df_test = df_test.merge(temp_df, left_on="idhogar", right_index=True, how="outer")
df_test[["idhogar", "age", "escolari", "meaneduc_new", "meaneduc"]].head(10)
df_test['meaneduc_new'].isnull().sum()
df_test["meaneduc_new"].describe()
df_test["meaneduc_new"].fillna(8.5675, inplace=True)
df_test["meaneduc_new"].isnull().sum()
sns.distplot(df_test.meaneduc_new)
df_test = cleandatasetyesno(df_test)
# Categorical variable for dependency
df_test['depend_cat'] = 1
df_test.loc[df_test['dependency']==1, 'depend_cat'] = 2
df_test.loc[df_test['dependency'] > 1, 'depend_cat'] = 3
tabulate(df_test.depend_cat)
df_test.loc[(df_test["parentesco1"]==1) & (df_test["male"]==1), "edjefa"] = 99
df_test.loc[df_test["parentesco1"]==0, "edjefa"] = 99
tabulate(df_test["edjefa"])
df_test.loc[(df_test["parentesco1"]==1) & (df_test["female"]==1), "edjefe"] = 99
df_test.loc[df_test["parentesco1"]==0, "edjefe"] = 99
tabulate(df_test["edjefe"])
print("Number of male household heads: {}".format(len(df_test.parentesco1[(df_test["parentesco1"]==1) & (df_test["male"]==1)])))
print("Number of female household heads: {}".format(len(df_test.parentesco1[(df_test["parentesco1"]==1) & (df_test["female"]==1)])))
# Creating categorical variables for household head female education
df_test["fhh_educ_cat1"] = 0
df_test.loc[df_test.edjefa < 6, "fhh_educ_cat1"] = 1
df_test["fhh_educ_cat2"] = 0
df_test.loc[(df_test.edjefa >= 6) & (df_test.edjefa < 12), "fhh_educ_cat2"] = 1
df_test["fhh_educ_cat3"] = 0
df_test.loc[(df_test.edjefa >= 12) & (df_test.edjefa < 99), "fhh_educ_cat3"] = 1
df_test["fhh_educ"] = 0
df_test.loc[df_test.fhh_educ_cat1==1, "fhh_educ"] = 1
df_test.loc[df_test.fhh_educ_cat2==1, "fhh_educ"] = 2
df_test.loc[df_test.fhh_educ_cat3==1, "fhh_educ"] = 3
tabulate(df_test.fhh_educ)
# Creating categorical variables for household head male education
df_test["mhh_educ_cat1"] = 0
df_test.loc[df_test.edjefe < 6, "mhh_educ_cat1"] = 1
df_test["mhh_educ_cat2"] = 0
df_test.loc[(df_test.edjefe >= 6) & (df_test.edjefe < 12), "mhh_educ_cat2"] = 1
df_test["mhh_educ_cat3"] = 0
df_test.loc[(df_test.edjefe >= 12) & (df_test.edjefe < 99), "mhh_educ_cat3"] = 1
df_test["mhh_educ"] = 0
df_test.loc[df_test.mhh_educ_cat1==1, "mhh_educ"] = 1
df_test.loc[df_test.mhh_educ_cat2==1, "mhh_educ"] = 2
df_test.loc[df_test.mhh_educ_cat3==1, "mhh_educ"] = 3
tabulate(df_test.mhh_educ)
tabulate(df_test["computer"])
temp_df = df_test["computer"].groupby(df_test["idhogar"]).sum().to_frame()
df_test = df_test.merge(temp_df, left_on="idhogar", right_index=True, how="outer")
tabulate(df_test["computer_y"])
temp_df = df_test["television"].groupby(df_test["idhogar"]).sum().to_frame()
df_test = df_test.merge(temp_df, left_on="idhogar", right_index=True, how="outer")
tabulate(df_test["television_y"])
df_test["male_hh"] = df_test["parentesco1"] * df_test["male"]
df_test["female_hh"] = df_test["parentesco1"] * df_test["female"]
df_materials = df_test[["paredblolad", "paredzocalo", "paredpreb", "pareddes", "paredmad", "paredzinc",
                   "paredfibras", "paredother", "pisomoscer", "pisocemento", "pisoother", "pisonatur",
                   "pisonotiene", "pisomadera", "techozinc", "techoentrepiso", "techoentrepiso", "techocane",
                   "techootro"]]
pca_materials = PCA().fit(df_materials)
print("Explained variance by component: %s" % pca_materials.explained_variance_ratio_)
pca_materials = pca_materials.transform(df_materials)
pca_mat_df = pd.DataFrame(data=pca_materials)
pca_mat_df[pca_mat_df.columns[0:6]].head(10)
pca_mat_df = pca_mat_df[pca_mat_df.columns[0:6]]
pca_mat_df.columns = ['pca_mat_1', 'pca_mat_2', 'pca_mat_3', 'pca_mat_4', 'pca_mat_5', 'pca_mat_6']
pca_mat_df.head()
len(pca_mat_df)
pca_mat_df.isnull().sum()
df_living = df_test[["v14a", "cielorazo", "abastaguadentro", "abastaguafuera", "abastaguano", "public", "planpri",
                "noelec", "coopele", "sanitario1", "sanitario2", "sanitario3", "sanitario5", "sanitario6",
                "energcocinar1", "energcocinar2", "energcocinar3", "energcocinar4", "elimbasu1", "elimbasu2",
                "elimbasu3", "elimbasu4", "elimbasu5", "elimbasu6", "epared1", "epared2", "epared3",
                "etecho1", "etecho2", "etecho3", "eviv1", "eviv2", "eviv3"]]
pca_living = PCA().fit(df_living)
print("Explained variance by component: %s" % pca_living.explained_variance_ratio_)
pca_living = pca_living.transform(df_living)
pca_liv_df = pd.DataFrame(data=pca_living)
pca_liv_df[pca_liv_df.columns[0:10]].head(10)
pca_liv_df = pca_liv_df[pca_liv_df.columns[0:10]]
pca_liv_df.columns = ['pca_liv_1', 'pca_liv_2', 'pca_liv_3', 'pca_liv_4', 'pca_liv_5', 'pca_liv_6',
                      'pca_liv_7', 'pca_liv_8', 'pca_liv_9', 'pca_liv_10']
pca_liv_df.head()
len(pca_liv_df)
pca_liv_df.isnull().sum()
pca_mat_liv_df = pd.concat([pca_mat_df, pca_liv_df], axis=1)
pca_mat_liv_df.head()
df_assets = df_test[['rooms', 'bedrooms', 'v18q1', 'computer_y', 'television_y', 'qmobilephone', 'refrig']]
pca_assets = PCA().fit(df_assets)
print("Explained variance by component: %s" % pca_assets.explained_variance_ratio_)
pca_assets = pca_assets.transform(df_assets)
pca_assets_df = pd.DataFrame(data=pca_assets)
pca_assets_df[pca_assets_df.columns[0:4]].head(10)
pca_assets_df = pca_assets_df[pca_assets_df.columns[0:4]]
pca_assets_df.columns = ['pca_assets_1', 'pca_assets_2', 'pca_assets_3', 'pca_assets_4']
pca_assets_df.head()
len(pca_assets_df)
pca_assets_df.isnull().sum()
pca_df = pd.concat([pca_mat_liv_df, pca_assets_df], axis=1)
pca_df.head()
test = df_test[["Id","hacdor", "hacapo", "r4h1", "r4h2", "r4m1", "r4m2", "tamhog", "escolari", "dis", "male", "female",
               "hogar_nin", "hogar_adul", "hogar_mayor", "instlevel1", "instlevel2", "instlevel3", "instlevel4",
               "instlevel5", "instlevel6", "instlevel7", "instlevel8", "instlevel9", "overcrowding", "tipovivi1",
               "tipovivi2", "tipovivi3", "tipovivi4", "tipovivi5", "lugar1", "lugar2", "lugar3", "lugar4", 
               "lugar5", "lugar6", "area1", "area2", "SQBage", "meaneduc_new", "depend_cat", "fhh_educ", "mhh_educ",
               "male_hh", "female_hh"]]
test = pd.concat([test, pca_df], axis=1)
test.head()
test["Target"] = np.nan
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
ffeatures = pd.concat([features, test[test.columns[1:]]], axis=0, sort=False)
ffeatures["Target"] = ffeatures["Target"] - 1
tabulate(ffeatures["Target"])
#data_balance=ffeatures.drop(ffeatures.query('Target == 3').sample(frac=.75).index)
data_balance=ffeatures
train_labels = np.array(list(data_balance[data_balance['Target'].notnull()]['Target'].astype(np.uint8)))
train_set = data_balance[data_balance['Target'].notnull()]
test_set = data_balance[data_balance['Target'].isnull()]
pipeline = Pipeline([('imputer', Imputer(strategy = 'median')),('scaler', MinMaxScaler())])

X_train=train_set.drop(['Target'], axis=1)
y_train=train_set['Target']

X_test=test_set.drop(['Target'], axis=1)
y_test=test_set['Target']
# Fit and transform training data
X_train = pipeline.fit_transform(X_train)
#X_test = pipeline.transform(X_test)
X_test = pipeline.fit_transform(X_test)
from fastai.structured import *
from fastai.column_data import *
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
train_df = X_train
test_df = X_test
X_train, X_test, y_train, y_test = train_test_split(train_df,y_train,test_size=.3, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = dict()
params['max_depth'] = 70
params['min_child_weight'] = 1
params['num_class'] = 4
params['subsample'] = 1.0
params['colsample_bytree'] = 1.0 
params['eta'] = .2  
params['silent']=1
num_boost_round = 1000
cv_results = xgb.cv( params, dtrain, num_boost_round=num_boost_round, seed=42, nfold=5, metrics={'merror'}, early_stopping_rounds=100 )
cv_results['test-merror-mean'].min()
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=100
)
num_boost_round = model.best_iteration + 1

best_model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")]
)
accuracy_score(best_model.predict(dtest).astype(np.uint8), y_test)
best_model.save_model("XGB_tuned.model")
train_labels = np.array(list(data_balance[data_balance['Target'].notnull()]['Target'].astype(np.uint8)))
train_set = data_balance[data_balance['Target'].notnull()]
test_set = data_balance[data_balance['Target'].isnull()]
pipeline = Pipeline([('imputer', Imputer(strategy = 'median')),('scaler', MinMaxScaler())])

X_train=train_set.drop(['Target'], axis=1)
y_train=train_set['Target']

X_test=test_set.drop(['Target'], axis=1)
y_test=test_set['Target']
tabulate(y_train)
# Fit and transform training data
X_train = pipeline.fit_transform(X_train)
#X_test = pipeline.transform(X_test)
X_test = pipeline.fit_transform(X_test)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
num_boost_round = 1000
cv_results = xgb.cv( params, dtrain, num_boost_round=num_boost_round, seed=42, nfold=5, metrics={'merror'}, early_stopping_rounds=100 )
cv_results['test-merror-mean'].min()
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round
)
model.save_model("XGB_tuned.model")
loaded_model = xgb.Booster()
loaded_model.load_model("XGB_tuned.model")
xgb_pred=loaded_model.predict(dtest)
len(xgb_pred)
xgb_pred
test['Target'] = np.array(xgb_pred).astype(int) + 1
ad_submit = pd.concat([test[test.columns[0]], test['Target'] ], axis=1)
ad_submit.head()
ad_submit.to_csv("ad_submit_5.csv", index=False)
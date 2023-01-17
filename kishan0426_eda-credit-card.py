import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import plotly.express as px
app = pd.read_csv("application_data.csv")
app.head()
app.info()
app.describe()
app.shape
#Set display settings
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)
(app.isnull().sum()*100/len(app)).sort_values(ascending=False)
missing= app.isnull().sum()*100/len(app)
rem=missing[missing>47].sort_values(ascending=False)
NA_perc_45=app.isnull().sum()
dropNA=NA_perc_45[NA_perc_45.values > (45*len(app)/100)]
plt.figure(figsize=(20,4))
ax = dropNA.plot(kind='bar',color='blue')
plt.title('Columns having null values > 45%')
plt.show()
app.drop(list(rem.index.values),axis=1,inplace=True)
# plotly representation
plt.figure(figsize=[20,10])
plt.suptitle("Credit Bureau Enquires", fontsize=20)
plt.subplot(2,3,1)
ax1 = app.AMT_REQ_CREDIT_BUREAU_YEAR.value_counts().plot.bar()
plt.title("AMT_REQ_CREDIT_BUREAU_YEAR" )
plt.subplot(2,3,2)
ax1 = app.AMT_REQ_CREDIT_BUREAU_QRT.value_counts().plot.bar()
plt.title("AMT_REQ_CREDIT_BUREAU_QRT")
plt.subplot(2,3,3)
ax1 = app.AMT_REQ_CREDIT_BUREAU_MON.value_counts().plot.bar()
plt.title("AMT_REQ_CREDIT_BUREAU_MON" )
plt.subplot(2,3,4)
ax1 = app.AMT_REQ_CREDIT_BUREAU_WEEK.value_counts().plot.bar()
plt.title("AMT_REQ_CREDIT_BUREAU_WEEK" )
plt.subplot(2,3,5)
ax1 = app.AMT_REQ_CREDIT_BUREAU_DAY.value_counts().plot.bar()
plt.title("AMT_REQ_CREDIT_BUREAU_DAY")
plt.subplot(2,3,6)
ax1 = app.AMT_REQ_CREDIT_BUREAU_HOUR.value_counts().plot.bar()
plt.title("AMT_REQ_CREDIT_BUREAU_HOUR")
plt.show()
app["AMT_REQ_CREDIT_BUREAU"] = app[["AMT_REQ_CREDIT_BUREAU_HOUR","AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_WEEK",
                                    "AMT_REQ_CREDIT_BUREAU_MON","AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_YEAR"]].sum(axis=1)
app[app.CNT_FAM_MEMBERS.isnull()]
app.CNT_FAM_MEMBERS = app.CNT_FAM_MEMBERS.replace(np.nan,1)
#Fix data type
app.CNT_FAM_MEMBERS=app.CNT_FAM_MEMBERS.astype('int')
# the above columns are having missing value in same set of clients.
app[(app.DEF_30_CNT_SOCIAL_CIRCLE.isnull()) & (app.OBS_30_CNT_SOCIAL_CIRCLE.isnull()) 
    & (app.DEF_60_CNT_SOCIAL_CIRCLE.isnull()) & (app.OBS_60_CNT_SOCIAL_CIRCLE.isnull() & app.TARGET==1)].shape
plt.figure(figsize=[24,15])
plt.suptitle("Contact Details", fontsize=22)
plt.subplot(2,3,1)
ax = sns.countplot(x="FLAG_MOBIL", hue="FLAG_MOBIL", data=app)
plt.title("Mobile contact provided(1)")
plt.subplot(2,3,2)
ax = sns.countplot(x="FLAG_EMP_PHONE", hue="FLAG_EMP_PHONE", data=app)
plt.title("EMP contact provided(1)")
plt.subplot(2,3,3)
ax = sns.countplot(x="FLAG_WORK_PHONE", hue="FLAG_WORK_PHONE", data=app)
plt.title("Work contact provided(1)")
plt.subplot(2,3,4)
ax = sns.countplot(x="FLAG_CONT_MOBILE", hue="FLAG_CONT_MOBILE", data=app)
plt.title("Contact over Mobile(1)")
plt.subplot(2,3,5)
ax = sns.countplot(x="FLAG_PHONE", hue="FLAG_PHONE", data=app)
plt.title("Contact over phone(1)")
plt.subplot(2,3,6)
ax = sns.countplot(x="FLAG_EMAIL", hue="FLAG_EMAIL", data=app)
plt.title("Contact over email(1)")
plt.show()
app["FLAG_DOCUMENT"]=app[['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
       'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
       'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
       'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
       'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
       'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
       'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']].sum(axis=1)
app.FLAG_DOCUMENT= app.FLAG_DOCUMENT.replace(1, "YES")
app.FLAG_DOCUMENT= app.FLAG_DOCUMENT.replace(2, "YES")
app.FLAG_DOCUMENT= app.FLAG_DOCUMENT.replace(3, "YES")
app.FLAG_DOCUMENT= app.FLAG_DOCUMENT.replace(4, "YES")
app.FLAG_DOCUMENT= app.FLAG_DOCUMENT.replace(0, "NO")
# app.FLAG_DOCUMENT.value_counts()
# app.FLAG_DOCUMENT.isnull().sum()
print(app.FLAG_DOCUMENT.value_counts(normalize=True)*100)
app.drop(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
       'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
       'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
       'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
       'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
       'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
       'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'], axis=1,inplace=True)
app.FLAG_DOCUMENT.value_counts().plot.barh()
plt.show()
#app.info(verbose=True)
#app.nunique().sort_values()
# Whether Clients have car, realty assets
condition=[((app.FLAG_OWN_CAR=="Y") & (app.FLAG_OWN_REALTY=="Y")),
           ((app.FLAG_OWN_CAR=="Y") & (app.FLAG_OWN_REALTY=="N")),
           ((app.FLAG_OWN_CAR=="N") & (app.FLAG_OWN_REALTY=="Y")),
           ((app.FLAG_OWN_CAR=="N") & (app.FLAG_OWN_REALTY=="N"))]
values=["Both","Car","Realty","No"]
app['OWN_CAR_REALTY'] = np.select(condition, values)
colors = ['#2ca02c', '#d62728']
app.drop(["FLAG_OWN_CAR","FLAG_OWN_REALTY"], axis=1,inplace=True)
app.OWN_CAR_REALTY.value_counts().plot.bar(color=colors)
plt.show()
#Check Gender variable and mark the unknown
app.CODE_GENDER= app.CODE_GENDER.replace(["XNA"], np.NAN)
app.CODE_GENDER.value_counts()
#Updating the below catgorical data types
app.REG_REGION_NOT_LIVE_REGION= app.REG_REGION_NOT_LIVE_REGION.replace(1, "different")
app.REG_REGION_NOT_LIVE_REGION= app.REG_REGION_NOT_LIVE_REGION.replace(0, "same")
app.REG_REGION_NOT_WORK_REGION= app.REG_REGION_NOT_WORK_REGION.replace(1, "different")
app.REG_REGION_NOT_WORK_REGION= app.REG_REGION_NOT_WORK_REGION.replace(0, "same")
app.LIVE_REGION_NOT_WORK_REGION= app.LIVE_REGION_NOT_WORK_REGION.replace(1, "different")
app.LIVE_REGION_NOT_WORK_REGION= app.LIVE_REGION_NOT_WORK_REGION.replace(0, "same")
app.REG_CITY_NOT_LIVE_CITY= app.REG_CITY_NOT_LIVE_CITY.replace(1, "different")
app.REG_CITY_NOT_LIVE_CITY= app.REG_CITY_NOT_LIVE_CITY.replace(0, "same")
app.REG_CITY_NOT_WORK_CITY= app.REG_CITY_NOT_WORK_CITY.replace(1, "different")
app.REG_CITY_NOT_WORK_CITY= app.REG_CITY_NOT_WORK_CITY.replace(0, "asame")
app.LIVE_CITY_NOT_WORK_CITY= app.LIVE_CITY_NOT_WORK_CITY.replace(1, "different")
app.LIVE_CITY_NOT_WORK_CITY= app.LIVE_CITY_NOT_WORK_CITY.replace(0, "same")
#coercing the NA values
app['OBS_30_CNT_SOCIAL_CIRCLE'] = app['OBS_30_CNT_SOCIAL_CIRCLE'].apply(pd.to_numeric,  errors='coerce')
app['DEF_30_CNT_SOCIAL_CIRCLE'] = app['DEF_30_CNT_SOCIAL_CIRCLE'].apply(pd.to_numeric,  errors='coerce')
app['OBS_60_CNT_SOCIAL_CIRCLE'] = app['OBS_60_CNT_SOCIAL_CIRCLE'].apply(pd.to_numeric,  errors='coerce')
app['DEF_60_CNT_SOCIAL_CIRCLE'] = app['DEF_60_CNT_SOCIAL_CIRCLE'].apply(pd.to_numeric,  errors='coerce')
app['CNT_FAM_MEMBERS'] = app['CNT_FAM_MEMBERS'].apply(pd.to_numeric,  errors='coerce')
app[app.ORGANIZATION_TYPE=="XNA"]["NAME_INCOME_TYPE"].value_counts()
# app.ORGANIZATION_TYPE.value_counts()

for index,row in app[app.ORGANIZATION_TYPE=="XNA"].iterrows():
    if row["NAME_INCOME_TYPE"]=="Pensioner":
        app.at[index,'ORGANIZATION_TYPE']="Pensioner"
    elif row["NAME_INCOME_TYPE"]=="Unemployed":
        app.at[index,'ORGANIZATION_TYPE']="Unemployed"
    elif row["NAME_INCOME_TYPE"]=="Cleaning staff":
        app.at[index,'ORGANIZATION_TYPE']="Cleaning"
        
app.ORGANIZATION_TYPE= app.ORGANIZATION_TYPE.replace(["Trade: type 7","Trade: type 3","Trade: type 2","Trade: type 6","Trade: type 1","Trade: type 4","Trade: type 5","Trade: type 8"], "Trade")
app.ORGANIZATION_TYPE= app.ORGANIZATION_TYPE.replace(["Business Entity Type 3","Business Entity Type 2","Business Entity Type 1"], "Business")
app.ORGANIZATION_TYPE= app.ORGANIZATION_TYPE.replace(["Industry: type 9","Industry: type 3","Industry: type 11","Industry: type 7","Industry: type 1","Industry: type 6","Industry: type 10","Industry: type 2","Industry: type 12","Industry: type 4","Industry: type 5","Industry: type 13","Industry: type 8"], "Industry")
app.ORGANIZATION_TYPE= app.ORGANIZATION_TYPE.replace(["Transport: type 4","Transport: type 2","Transport: type 3","Transport: type 1"], "Transport")
plt.figure(figsize=[15,5])
app.ORGANIZATION_TYPE.value_counts().plot.bar()
plt.show()
plt.figure(figsize=[15,5])
app.OCCUPATION_TYPE.value_counts().plot.bar()
plt.show()
app.NAME_INCOME_TYPE.value_counts()
app.NAME_INCOME_TYPE= app.NAME_INCOME_TYPE.replace(["Commercial associate","State servant","Businessman","Maternity leave"], "Working")
app.NAME_INCOME_TYPE.value_counts().plot.line()
plt.show()
app.NAME_TYPE_SUITE.value_counts()
app.NAME_TYPE_SUITE= app.NAME_TYPE_SUITE.replace(["Other_B","Other_A"], "Others")
app.NAME_TYPE_SUITE= app.NAME_TYPE_SUITE.replace(["Spouse, partner","Children"], "Family")
app.NAME_TYPE_SUITE.value_counts()
app.NAME_FAMILY_STATUS.value_counts()
app.NAME_FAMILY_STATUS= app.NAME_FAMILY_STATUS.replace(["Civil marriage"], "Married")
app.NAME_FAMILY_STATUS= app.NAME_FAMILY_STATUS.replace(["Single / not married"], "Single")
app.NAME_FAMILY_STATUS.value_counts()
app.NAME_EDUCATION_TYPE.value_counts().plot.barh()
plt.show()
app['AGE']=round(-(app.DAYS_BIRTH)/365)
app.drop(["DAYS_BIRTH"], axis=1,inplace=True)
app.AGE.describe()
cut_bins = [20,30,40,50,60,70]
cut_bin_labels=["20-30","30-40","40-50","50-60","60-70"]
app.AGE = pd.cut(app.AGE, bins=cut_bins, labels=cut_bin_labels)
app.AGE.value_counts()
# app.DAYS_EMPLOYED.value_counts().head()
app['WORK_EXPERIENCE']=round(abs(app.DAYS_EMPLOYED)/365,2)
app.drop(["DAYS_EMPLOYED"], axis=1,inplace=True)
app[app.WORK_EXPERIENCE>1000.00].shape
cut_bins = [0,10,20,30,40,50,1001]
cut_bin_labels=["0-10","10-20","20-30","30-40","40-50",">50"]
app.WORK_EXPERIENCE = pd.cut(app.WORK_EXPERIENCE, bins=cut_bins, labels=cut_bin_labels)
app['REGISTRATION']=round(abs(app.DAYS_REGISTRATION)/365,2)
app.drop(["DAYS_REGISTRATION"], axis=1,inplace=True)
app['ID_PUBLISH']=round(abs(app.DAYS_ID_PUBLISH)/365,2)
app.drop(["DAYS_ID_PUBLISH"], axis=1,inplace=True)
app['LAST_PHONE_CHANGE']=round(abs(app.DAYS_LAST_PHONE_CHANGE)/365,3)
app.drop(["DAYS_LAST_PHONE_CHANGE"], axis=1,inplace=True)
cut_bins = [0,10,20,30,40,50,60,70]
cut_bin_labels=["0-10","10-20","20-30","30-40","40-50","50-60","60-70"]
app.REGISTRATION = pd.cut(app.REGISTRATION, bins=cut_bins, labels=cut_bin_labels)
app.REGISTRATION.describe()
# app.CNT_CHILDREN.describe()
app.CNT_CHILDREN.quantile([0.6,0.8,0.97,0.99,0.999,0.9999,0.99999,1])
plt.figure(figsize=[7,4])
app.CNT_CHILDREN.value_counts().plot.box()
plt.show()
app.AMT_INCOME_TOTAL.describe()
# app.AMT_INCOME_TOTAL.describe()
plt.figure(figsize=[7,4])
app.AMT_INCOME_TOTAL.value_counts().plot.box()
plt.show()
# app.OBS_30_CNT_SOCIAL_CIRCLE.describe()
app.OBS_30_CNT_SOCIAL_CIRCLE.value_counts().plot.box()
plt.show()
app.OBS_30_CNT_SOCIAL_CIRCLE.quantile([0.1,0.2,0.3,0.4,0.5,0.8,0.9,0.999999,1])
# app.OBS_60_CNT_SOCIAL_CIRCLE.describe()
app.OBS_60_CNT_SOCIAL_CIRCLE.quantile([0.1,0.2,0.3,0.4,0.8,0.9,1])
fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(30,20))
plt.suptitle('Credit amount distribution VS Organization Type')
y = 0
for x in np.unique(app['ORGANIZATION_TYPE'].dropna()):
    sns.distplot(app.loc[app['ORGANIZATION_TYPE']==x, 'AMT_CREDIT'], ax=axes.flat[y])
    axes.flat[y].set_title(x)
    y += 1
    
plt.tight_layout()
plt.subplots_adjust(top=.90)
plt.show()
fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(30,20))
plt.suptitle('Income amount distribution VS Organization Type')
y = 0
for x in np.unique(app['ORGANIZATION_TYPE'].dropna()):
    sns.distplot(app.loc[app['ORGANIZATION_TYPE']==x, 'AMT_INCOME_TOTAL'], ax=axes.flat[y])
    axes.flat[y].set_title(x)
    y += 1
    
plt.tight_layout()
plt.subplots_adjust(top=.90)
plt.show()
app.info(verbose=True)
# app.info(verbose=True)
application_data=app[["SK_ID_CURR","TARGET","AGE","NAME_CONTRACT_TYPE","CODE_GENDER","OWN_CAR_REALTY","CNT_CHILDREN","CNT_FAM_MEMBERS",
     "WORK_EXPERIENCE","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE","NAME_TYPE_SUITE","NAME_INCOME_TYPE",
     "ORGANIZATION_TYPE","OCCUPATION_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
     "REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY","DEF_30_CNT_SOCIAL_CIRCLE","DEF_60_CNT_SOCIAL_CIRCLE",
     "OBS_30_CNT_SOCIAL_CIRCLE","OBS_60_CNT_SOCIAL_CIRCLE","AMT_REQ_CREDIT_BUREAU"]]
application_data.groupby(["NAME_CONTRACT_TYPE"]).TARGET.value_counts().plot.bar(color=colors)
plt.show()
tar1= application_data[application_data.TARGET==1]
tar0= application_data[application_data.TARGET==0]
corr=tar1.corr()
corr=corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))
Ctar1=corr.unstack().reset_index()
Ctar1.columns=['Var1','Var2','correlation']
Ctar1.dropna(subset=['correlation'], inplace=True)
Ctar1['correlation_abs']=Ctar1['correlation'].abs()
Ctar1.sort_values("correlation_abs",ascending=False,inplace=True)
corr=tar0.corr()
corr=corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))
Ctar0=corr.unstack().reset_index()
Ctar0.columns=['Var1','Var2','correlation']
Ctar0.dropna(subset=['correlation'], inplace=True)
Ctar0['correlation_abs']=Ctar0['correlation'].abs()
Ctar0.sort_values("correlation_abs",ascending=False,inplace=True)
plt.figure(figsize=[24,12])
plt.suptitle("Correlation of Application Data", fontsize=14)
plt.subplot(1,2,1)
e = pd.pivot_table(data=Ctar1, index="Var1", columns="Var2", values="correlation_abs")
ax1=sns.heatmap(e,annot=True,cmap="BuGn")
plt.title("TARGET 1" )

plt.subplot(1,2,2)
e = pd.pivot_table(data=Ctar0, index="Var1", columns="Var2", values="correlation_abs")
ax1=sns.heatmap(e,annot=True,cmap="BuGn")
plt.title("TARGET 0" )

bottom, top=ax1.get_ylim()
ax1.set_ylim(bottom+0.5, top-0.5)
plt.show()
print("TARGET 0 Data")
Ctar0.plot(kind='scatter',x="Var1",y="Var2",c="correlation_abs",colormap="Blues")
plt.show()
print("TARGET 1 Data")
Ctar1.plot(kind='scatter',x="Var1",y="Var2",c="correlation_abs",colormap="Blues")
plt.show()
#NAME_CONTRACT_TYPE
labels = 'Cash Loans', 'Revolving Loans'
size_tar0 = [tar0.NAME_CONTRACT_TYPE.value_counts()]
size_tar1 = [tar1.NAME_CONTRACT_TYPE.value_counts()]
explode = (0, 0.1)
fig, ax1 = plt.subplots()
ax1.pie(size_tar0,explode=explode,labels=labels, autopct='%1.2f%%')
plt.title("Payment on Time")
fig, ax1 = plt.subplots()
ax1.pie(size_tar1,explode=explode,labels=labels, autopct='%1.2f%%')
plt.title("Payment with Difficulties")
plt.show()
#CODE_GENDER
labels = 'Female', 'Male'
size_tar0 = [tar0.CODE_GENDER.value_counts()]
size_tar1 = [tar1.CODE_GENDER.value_counts()]
explode = (0, 0.1)
fig, ax1 = plt.subplots()
ax1.pie(size_tar0,explode=explode,labels=labels, autopct='%1.2f%%')
plt.title("Clients: TARGET 0")
plt.show()
fig, ax1 = plt.subplots()
ax1.pie(size_tar1,explode=explode,labels=labels, autopct='%1.2f%%')
plt.title("Clients: TARGET 1")
plt.show()
#OWN_CAR_REALTY
pd.concat([tar0.OWN_CAR_REALTY.value_counts().rename("Payment On Time"),
           tar1.OWN_CAR_REALTY.value_counts().rename("Payment with Difficulties")],axis=1).plot.bar()
plt.title("Clients owning Assets")
plt.show()
pd.concat([tar0.CNT_CHILDREN.value_counts().rename("Payment On Time"),
           tar1.CNT_CHILDREN.value_counts().rename("Payment with Difficulties")],axis=1).plot.bar(stacked=True)
plt.title("Clients Children Count")
plt.show()
# CNT_FAM_MEMBERS
plt.figure(figsize=[10,7])
sns.distplot(tar0.CNT_FAM_MEMBERS,hist=False, label="Payment On-Time",bins=40,color="Green")
sns.distplot(tar1.CNT_FAM_MEMBERS,hist=False,label="Payment difficuties",bins=40,color="Red")
plt.show()

tar0[["AMT_INCOME_TOTAL","AMT_ANNUITY","AMT_CREDIT","AMT_GOODS_PRICE"]].plot.box(vert=False)
plt.title("Payment On Time")
tar1[["AMT_INCOME_TOTAL","AMT_ANNUITY","AMT_CREDIT","AMT_GOODS_PRICE"]].plot.box(vert=False)
plt.title("Payment with Difficulties")
plt.show()
tar0[["AMT_ANNUITY","AMT_CREDIT","AMT_GOODS_PRICE"]].plot.box(vert=False)
plt.title("Payment On Time")
tar1[["AMT_ANNUITY","AMT_CREDIT","AMT_GOODS_PRICE"]].plot.box(vert=False)
plt.title("Payment with Difficulties")
plt.show()

# EMPLOYED_INYEARS
Target0 = tar0.WORK_EXPERIENCE.value_counts()
Target1 = tar1.WORK_EXPERIENCE.value_counts()
df_plot  = pd.DataFrame([Target0,Target1])
df_plot.index=['Pays On-Time','Payment Difficuties']
# Bar plot
df_plot.plot(kind='bar',colormap='Dark2', title='Clients Work-Experience in years');
plt.show()

# CREDIT_BUREAU ENQUIRIES: TOTAL
plt.figure(figsize=[10,7])
sns.distplot(tar0.AMT_REQ_CREDIT_BUREAU,hist=False, label="Payment On-Time",bins=40,color="Blue",rug=True)
sns.distplot(tar1.AMT_REQ_CREDIT_BUREAU,hist=False, label="Payment difficuties",bins=40,color="Red",rug=True)
plt.show()
#DAYS PAST DUE - DEFAULTED
plt.figure(figsize=[14,10])
plt.subplot(2,2,1)
tar0.DEF_30_CNT_SOCIAL_CIRCLE.value_counts().plot.bar()
plt.title("Defaulted_30Days_Target0")
plt.subplot(2,2,2)
tar1.DEF_30_CNT_SOCIAL_CIRCLE.value_counts().plot.bar()
plt.title("Defaulted_30Days_Target1")
plt.subplot(2,2,3)
tar0.DEF_60_CNT_SOCIAL_CIRCLE.value_counts().plot.bar()
plt.title("Defaulted_60Days_Target0")
plt.subplot(2,2,4)
tar1.DEF_60_CNT_SOCIAL_CIRCLE.value_counts().plot.bar()
plt.title("Defaulted_60Days_Target1")
plt.show()
# DAYS PAST DUE - OBSERVED
plt.figure(figsize=[14,10])
plt.subplot(2,2,1)
tar0.OBS_30_CNT_SOCIAL_CIRCLE.value_counts().plot.bar()
plt.title("Observed_30Days_Target0")
plt.subplot(2,2,2)
tar1.OBS_30_CNT_SOCIAL_CIRCLE.value_counts().plot.bar()
plt.title("Observed_30Days_Target1")
plt.subplot(2,2,3)
tar0.OBS_60_CNT_SOCIAL_CIRCLE.value_counts().plot.bar()
plt.title("Observed_60Days_Target0")
plt.subplot(2,2,4)
tar1.OBS_60_CNT_SOCIAL_CIRCLE.value_counts().plot.bar()
plt.title("Observed_60Days_Target1")
plt.show()
Target0 = tar0.AGE.value_counts()
Target1 = tar1.AGE.value_counts()
df_plot  = pd.DataFrame([Target0,Target1])
df_plot.index=['Pays On-Time','Payment Difficuties']
# Bar plot
df_plot.plot(kind='barh',colormap='Accent', title='Clients AGE Limit');
plt.show()
Target0 = tar0.NAME_EDUCATION_TYPE.value_counts()
Target1 = tar1.NAME_EDUCATION_TYPE.value_counts()
df_plot  = pd.DataFrame([Target0,Target1])
df_plot.index=['Pays On-Time','Payment Difficuties']
# Bar plot
df_plot.plot(kind='bar',colormap='Accent', title='Clients AGE Limit');
plt.show()
#AMT_INCOME_TOTAL and  AMT_CREDIT
sns.regplot(x=tar0.AMT_INCOME_TOTAL,y=tar0.AMT_CREDIT)
plt.show()

sns.regplot(x=tar1.AMT_INCOME_TOTAL,y=tar1.AMT_CREDIT)
plt.show()
#AMT_INCOME_TOTAL and  AMT_ANNUITY
print("TARGET 0 : Correlation between Total Income and Annuity Amount")
sns.set_style("white")
sns.set(context="notebook",style='darkgrid',palette='deep',font='sans-serif',font_scale=1,color_codes=True,rc=None)
sns.jointplot(tar0[tar0.AMT_INCOME_TOTAL<= tar0.AMT_INCOME_TOTAL.quantile(0.95)].AMT_INCOME_TOTAL,
              tar0[tar0.AMT_INCOME_TOTAL<= tar0.AMT_INCOME_TOTAL.quantile(0.95)].AMT_ANNUITY,kind="reg")
plt.show()
print("TARGET 1 : Correlation between Total Income and Annuity Amount")
sns.set_style("white")
sns.set(context="notebook",style='darkgrid',palette='deep',font='sans-serif',font_scale=1,color_codes=True,rc=None)
sns.jointplot(tar1[tar1.AMT_INCOME_TOTAL<= tar1.AMT_INCOME_TOTAL.quantile(0.95)].AMT_INCOME_TOTAL,
              tar1[tar1.AMT_INCOME_TOTAL<= tar1.AMT_INCOME_TOTAL.quantile(0.95)].AMT_ANNUITY,kind="reg")
plt.show()
#AMT_ANNUITY and AMT_CREDIT
print("TARGET 0 : Correlation between Credit and Annuity Amount")
sns.set_style("white")
sns.scatterplot(tar0.AMT_ANNUITY, tar0.AMT_CREDIT,markers=True )
plt.show()
print("TARGET 1 : Correlation between Credit and Annuity Amount")
sns.scatterplot(tar1.AMT_ANNUITY, tar1.AMT_CREDIT,markers=True )
plt.show()

#AMT_GOODS_PRICE and AMT_CREDIT
print("TARGET 0 : Correlation between Goods price and Annuity Amount")
sns.set_style("white")
sns.scatterplot(tar0.AMT_GOODS_PRICE, tar0.AMT_ANNUITY )
plt.show()
print("TARGET 1 : Correlation between Goods price and Annuity Amount")
sns.scatterplot(tar1.AMT_GOODS_PRICE, tar1.AMT_ANNUITY )
plt.show()

plt.figure(figsize=[14,7])
plt.suptitle("Bivariate: AGE vs CONTRACT TYPE", fontsize=20)
plt.subplot(1,2,1)
plt.title("Target 0")
sns.countplot(x=tar0.AGE, hue="NAME_CONTRACT_TYPE",data=tar0,palette="mako")  
plt.subplot(1,2,2)
plt.title("Target 1")
sns.countplot(x=tar1.AGE, hue="NAME_CONTRACT_TYPE",data=tar1,palette="mako")  
plt.show()
plt.figure(figsize=[20,8])
plt.suptitle("Bivariate: AGE vs WORK EXPERIENCE", fontsize=20)
plt.subplot(1,2,1)
plt.title("Target 0")
sns.countplot(x=tar0.WORK_EXPERIENCE, hue="AGE",data=tar0,palette="Dark2")
plt.subplot(1,2,2)
plt.title("Target 1")
sns.countplot(x=tar1.WORK_EXPERIENCE, hue="AGE",data=tar1,palette="Dark2")  
plt.show()
plt.figure(figsize=[14,5])
plt.suptitle("Bivariate: WORK EXPERIENCE vs INCOME TYPE", fontsize=20)
plt.subplot(1,2,1)
plt.title("Target 0")
chart=sns.countplot(tar0.NAME_INCOME_TYPE, hue="WORK_EXPERIENCE",data=tar0,palette="Dark2")
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize=12)
plt.subplot(1,2,2)
plt.title("Target 1")
chart=sns.countplot(x=tar1.NAME_INCOME_TYPE, hue="WORK_EXPERIENCE",data=tar1,palette="Dark2")  
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize=12)
plt.show()

plt.figure(figsize=[14,5])
plt.suptitle("Bivariate: WORK EXPERIENCE vs INCOME TYPE", fontsize=20)
plt.subplot(1,2,1)
plt.title("Target 0")
chart=sns.countplot(tar0.NAME_INCOME_TYPE, hue="NAME_CONTRACT_TYPE",data=tar0,palette="Dark2")
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize=12)
plt.subplot(1,2,2)
plt.title("Target 1")
chart=sns.countplot(x=tar1.NAME_INCOME_TYPE, hue="NAME_CONTRACT_TYPE",data=tar1,palette="Dark2")  
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize=12)
plt.show()

plt.figure(figsize=[24,10])
plt.suptitle("Bivariate: OCCUPATION_TYPE vs TOTAL INCOME", fontsize=20)
plt.subplot(1,2,1)
chart=tar0.groupby(["OCCUPATION_TYPE"])["AMT_INCOME_TOTAL"].median().plot.bar()
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize=16)
plt.title("Target 0",fontsize=20)
plt.subplot(1,2,2)
chart=tar1.groupby(["OCCUPATION_TYPE"])["AMT_INCOME_TOTAL"].median().plot.bar()
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize=16)
plt.title("Target 1",fontsize=20)
plt.show()

plt.figure(figsize=[24,10])
plt.subplot(1,2,1)
chart=tar0.groupby(["OCCUPATION_TYPE"])["AMT_CREDIT"].median().plot.bar()
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize=16)
plt.title("Target 0",fontsize=20)
plt.subplot(1,2,2)
chart=tar1.groupby(["OCCUPATION_TYPE"])["AMT_CREDIT"].median().plot.bar()
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize=16)
plt.title("Target 1",fontsize=20)
plt.suptitle("Bivariate: OCCUPATION_TYPE vs CREDIT", fontsize=20)

plt.show()
plt.figure(figsize=[20,7])
sns.lineplot(x=tar0.NAME_FAMILY_STATUS, y=tar0.AMT_ANNUITY,hue=tar0.NAME_CONTRACT_TYPE,data=tar0,palette="twilight_r")
sns.lineplot(x=tar1.NAME_FAMILY_STATUS, y=tar1.AMT_ANNUITY,hue=tar1.NAME_CONTRACT_TYPE,data=tar1,palette="rainbow")
plt.show()
plt.figure(figsize=[14,7])
plt.suptitle("60 Defaulted DPD for TARGET 1", fontsize=20)
plt.subplot(1,1,1)
chart=sns.lineplot(x=tar1.ORGANIZATION_TYPE, y=tar1.DEF_60_CNT_SOCIAL_CIRCLE,hue=tar1.AGE,data=tar1,palette="rainbow")
chart.set_xticklabels(labels=tar1.ORGANIZATION_TYPE, rotation=90, fontsize=16)
plt.show()


plt.figure(figsize=[14,7])
plt.suptitle("60 Defaulted DPD for TARGET 0", fontsize=20)
plt.subplot(1,1,1)
chart=sns.lineplot(x=tar0.ORGANIZATION_TYPE, y=tar0.DEF_60_CNT_SOCIAL_CIRCLE,hue=tar0.AGE,data=tar0,palette="rainbow")
chart.set_xticklabels(labels=tar0.ORGANIZATION_TYPE, rotation=90, fontsize=16)
plt.show()

prev = pd.read_csv("previous_application.csv")
prev.head()
prev.replace("XNA", np.nan,inplace=True)
prev.replace("XAP", np.nan,inplace=True)
(prev.isnull().sum()*100/len(prev)).sort_values(ascending=False).head(20)
df1=prev[["SK_ID_CURR","NAME_CONTRACT_STATUS","NAME_CONTRACT_TYPE","AMT_ANNUITY","AMT_APPLICATION",
"AMT_CREDIT","AMT_GOODS_PRICE","DAYS_DECISION","NAME_CLIENT_TYPE","CHANNEL_TYPE","SELLERPLACE_AREA",
"CNT_PAYMENT","NAME_YIELD_GROUP","PRODUCT_COMBINATION"]]
#dummy created for avoiding mess
df=application_data
df2=pd.merge(df,df1,how="left",on="SK_ID_CURR")
df.shape,df1.shape,df2.shape
# previous application contract status
df2.NAME_CONTRACT_STATUS.replace("Canceled", "Cancelled",inplace=True)
# df2.NAME_CONTRACT_STATUS.value_counts(normalize=True)
df2.NAME_CONTRACT_STATUS.value_counts().plot.bar()
plt.show()
df2.NAME_CONTRACT_TYPE_y.value_counts().plot.line()
plt.show()
df2.NAME_CLIENT_TYPE.value_counts().plot.bar()
plt.show()
### Binning

#df2.CNT_PAYMENT.quantile([0.1,0.25,0.5,0.75,0.9,0.95,1])
cut_bins = list(df2.CNT_PAYMENT.quantile([0.1,0.25,0.5,0.75,0.9,0.95,0.99,1]))
cut_bin_labels=["0-6","6-12","12-24","24-36","36-48","48-60","60-84"]
df2.CNT_PAYMENT = pd.cut(df2.CNT_PAYMENT, bins=cut_bins, labels=cut_bin_labels)
df2.CNT_PAYMENT.value_counts().plot.bar()
plt.show()
plt.figure(figsize=[20,10])
plt.suptitle("Previous Data based on Contract/Contract Status/Annuity", fontsize=20)
chart=sns.boxplot(x=df2.AMT_ANNUITY_y,y=df2.NAME_CONTRACT_TYPE_y, hue="NAME_CONTRACT_STATUS",data=df2,palette="CMRmap_r")
chart.set_yticklabels(chart.get_yticklabels(), fontsize=16)
plt.show()
plt.figure(figsize=[20,10])
sns.pairplot(df2[['AMT_ANNUITY_y', 'AMT_CREDIT_y', 'AMT_GOODS_PRICE_y']])
plt.show()
df2[["NAME_CONTRACT_STATUS","CNT_PAYMENT","NAME_CLIENT_TYPE"]]
plt.figure(figsize=[14,7])
plt.subplot(1,2,1)
res = pd.pivot_table(data=df2,index="NAME_CONTRACT_STATUS", columns="NAME_CLIENT_TYPE", values="AMT_ANNUITY_y", aggfunc=np.median)
ax1 = sns.heatmap(res)
bottom, top = ax1.get_ylim()
ax1.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Annuity Amount")

plt.subplot(1,2,2)
res1 = pd.pivot_table(data=df2,index="NAME_CONTRACT_STATUS", columns="NAME_CLIENT_TYPE", values="AMT_CREDIT_y", aggfunc=np.median)
ax2 = sns.heatmap(res1)
plt.title("Credit Amount")
bottom, top = ax2.get_ylim()
ax2.set_ylim(bottom + 0.5, top - 0.5)

plt.show()
plt.figure(figsize=[14,7])
plt.subplot(1,2,1)
res = pd.pivot_table(data=df2,index="CNT_PAYMENT", columns="NAME_CONTRACT_TYPE_y", values="AMT_CREDIT_y", aggfunc=np.median)
ax1 = sns.heatmap(res)
bottom, top = ax1.get_ylim()
ax1.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Annuity Amount")

plt.subplot(1,2,2)
res1 = pd.pivot_table(data=df2,index="CNT_PAYMENT", columns="NAME_CONTRACT_TYPE_y", values="AMT_CREDIT_y", aggfunc=np.median)
ax2 = sns.heatmap(res1)
plt.title("Credit Amount")
bottom, top = ax2.get_ylim()
ax2.set_ylim(bottom + 0.5, top - 0.5)

plt.show()
sns.catplot(x="AMT_CREDIT_x", y="NAME_TYPE_SUITE", data=df2,hue="NAME_CONTRACT_STATUS",palette="prism",kind="violin");
plt.show()
sns.catplot(x="AMT_CREDIT_x", y="NAME_TYPE_SUITE", data=df2,hue="NAME_CLIENT_TYPE",palette="prism",kind="violin");
plt.show()
sns.catplot(x="OWN_CAR_REALTY", y="AMT_ANNUITY_x", hue="NAME_INCOME_TYPE", kind="bar", data=df2)
plt.show()
sns.catplot(x="TARGET", y="DEF_30_CNT_SOCIAL_CIRCLE", kind="boxen",
            data=df2.sort_values("TARGET"))
sns.catplot(x="TARGET", y="DEF_60_CNT_SOCIAL_CIRCLE", kind="boxen",
            data=df2.sort_values("TARGET"))
plt.show()
sns.catplot(x="TARGET", y="AMT_REQ_CREDIT_BUREAU", kind="boxen",
            data=df2.sort_values("TARGET"))
plt.show()
#df2.AMT_INCOME_TOTAL.quantile(0.99)
sep=df2[((df2.TARGET==1) & (df2.AMT_INCOME_TOTAL<df2.AMT_INCOME_TOTAL.quantile(0.99)))]
fig = px.box(sep, x='NAME_INCOME_TYPE', y='AMT_INCOME_TOTAL', color='CNT_FAM_MEMBERS', notched=True)
fig.show()
fig = px.box(sep, x='NAME_INCOME_TYPE', y='AMT_INCOME_TOTAL', color='CNT_FAM_MEMBERS', notched=True)
fig.show()
fig = px.box(sep, x='NAME_INCOME_TYPE', y='AMT_GOODS_PRICE_x', color='CNT_FAM_MEMBERS', notched=True)
fig.show()

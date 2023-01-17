import numpy as np

import pandas as pd
# for getting the file path

import os

print(os.listdir('../input'))
#Data Visualization Libraries

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
# reading the dataset

df=pd.read_csv("/kaggle/input/online-shoppers-intention/online_shoppers_intention.csv")



# checking the shape of the data

df.shape
df.head()
df.info()
print('The basic distribution of the dataset', df.describe())
df['Administrative']=df['Administrative'].astype(object)

df['Informational']=df['Informational'].astype(object)

df['ProductRelated']=df['ProductRelated'].astype(object)

df['OperatingSystems']=df['OperatingSystems'].astype(object)

df['Browser']=df['Browser'].astype(object)

df['Region']=df['Region'].astype(object)

df['TrafficType']=df['TrafficType'].astype(object)

df['SpecialDay']=df['SpecialDay'].astype(object)

df.info()
# Checking the unique visitor types

df['VisitorType'].value_counts()
# Checking the unique Browsers

df['Browser'].value_counts()
plt.figure(figsize = (18,7))

sns.countplot(df['Administrative'], color = "crimson")

plt.show()
plt.figure(figsize = (18,7))

sns.distplot(df['Administrative_Duration'], color = "crimson")

plt.show()
plt.figure(figsize = (18,7))

sns.countplot(df['Informational'], color = "crimson")

plt.show()
plt.figure(figsize = (18,7))

sns.countplot(df['Informational_Duration'], color = "crimson")

plt.show()
plt.figure(figsize = (18,7))

sns.countplot(df['ProductRelated'], color = "crimson")

plt.show()
from sklearn.preprocessing import quantile_transform

import scipy.stats as stats

pro_duratn = quantile_transform(df[['ProductRelated_Duration']], output_distribution='normal',random_state=0, copy='warn').flatten()

inf_duration= quantile_transform(df[['Informational_Duration']], output_distribution='uniform',random_state=0, copy='warn').flatten()

adm_duration= quantile_transform(df[['Administrative_Duration']], output_distribution='normal',random_state=0, copy='warn').flatten()



#pro_duratn, _ = stats.boxcox(df[['ProductRelated_Duration']])

#inf_duration, _ = stats.boxcox(df[['Informational_Duration']])

#adm_duration, _ = stats.boxcox(df[['Administrative_Duration']])



#sns.distplot(pro_duratn, color = "crimson")

#plt.show()

#sns.distplot(inf_duration, color = "crimson")

#plt.show()

#sns.distplot(adm_duration, color = "crimson")

#plt.show()
plt.figure(figsize = (18,7))

sns.distplot(df['BounceRates'], color = "crimson")

plt.show()
plt.figure(figsize = (18,7))

sns.distplot(df['ExitRates'], color = "crimson")

plt.show()
df['SpecialDay'].value_counts()
plt.figure(figsize = (18,7))



sns.countplot(df['SpecialDay'], palette = 'pastel')

plt.show()
plt.figure(figsize = (18,7))

sns.countplot(df['OperatingSystems'], palette = 'pastel')

plt.show()
plt.figure(figsize = (18,7))

sns.countplot(df['Browser'], palette = 'pastel')

plt.show()
plt.figure(figsize = (18,7))

sns.countplot(df['Region'], palette = 'pastel')

plt.show()
plt.figure(figsize = (18,7))

sns.countplot(df['TrafficType'], palette = 'pastel')

plt.show()
print(df['Month'].value_counts())

print(df['VisitorType'].value_counts())
# Month

df['Month'].value_counts().plot(kind = "bar")



# By Sns

#sns.countplot(x = "Month", data = df)

plt.xticks(rotation = 90)

plt.show()
# VisitoType

df['VisitorType'].value_counts().plot(kind = "bar")



# By Sns

#sns.countplot(x = "Month", data = df)

plt.xticks(rotation = 90)

plt.show()


plt.figure(figsize = (10,7))



plt.subplot(1, 2, 1)

sns.countplot(df['Weekend'], palette = 'pastel')

plt.title('Buy or Not', fontsize = 20)

plt.xlabel('Revenue or not', fontsize = 15)

plt.ylabel('count', fontsize = 15)



# checking the Distribution of customers on Weekend

plt.subplot(1, 2, 2)

sns.countplot(df['Weekend'], palette = 'inferno')

plt.title('Purchase on Weekends', fontsize = 20)

plt.xlabel('Weekend or not', fontsize = 15)

plt.ylabel('count', fontsize = 15)



plt.show()
# Cat Vs Num

#plt.figure(figsize = (25,18))



sns.boxplot(df['Revenue'], df['Informational_Duration'], palette = 'rainbow')

plt.title('Info. duration vs Revenue', fontsize = 30)

plt.xlabel('Info. duration', fontsize = 15)

plt.ylabel('Revenue', fontsize = 15)

plt.show()
# Cat Vs Num

#plt.figure(figsize = (25,18))

sns.boxplot(df['Revenue'], df['Administrative_Duration'], palette = 'pastel')

plt.title('Admn. duration vs Revenue', fontsize = 30)

plt.xlabel('Admn. duration', fontsize = 15)

plt.ylabel('Revenue', fontsize = 15)

plt.show()
# Cat Vs Num

#plt.figure(figsize = (25,18))

sns.boxplot(df['Revenue'], df['ProductRelated_Duration'], palette = 'dark')

plt.title('Product Related duration vs Revenue', fontsize = 30)

plt.xlabel('Product Related duration', fontsize = 15)

plt.ylabel('Revenue', fontsize = 15)

plt.show()


# Cat Vs Num

#plt.figure(figsize = (25,18))

sns.boxplot(df['Revenue'], df['ExitRates'], palette = 'rainbow')

plt.title('ExitRates vs Revenue', fontsize = 30)

plt.xlabel('ExitRates', fontsize = 15)

plt.ylabel('Revenue', fontsize = 15)

plt.show()
# Cat Vs Num

# page values vs revenue

#plt.figure(figsize = (15,9))

sns.stripplot(df['Revenue'], df['PageValues'], palette = 'autumn')

plt.title('PageValues vs Revenue', fontsize = 30)

plt.xlabel('PageValues', fontsize = 15)

plt.ylabel('Revenue', fontsize = 15)

plt.show()
# Cat Vs Num

# bounce rates vs revenue



#plt.figure(figsize = (15,9))

sns.stripplot(df['Revenue'], df['BounceRates'], palette = 'magma')

plt.title('Bounce Rates vs Revenue', fontsize = 30)

plt.xlabel('Boune Rates', fontsize = 15)

plt.ylabel('Revenue', fontsize = 15)

plt.show()
# Cat Vs Cat



# weekend vs Revenue

data = pd.crosstab(df['Weekend'], df['Revenue'])

data.plot(kind = 'bar', stacked = True, color  = ['lightpink', 'yellow'])

plt.title('Weekend vs Revenue')

plt.show()
# Cat vs Cat

# Traffic Type vs Revenue

data = pd.crosstab(df['TrafficType'], df['Revenue'])

data.plot(kind = 'bar', stacked = True, figsize = (8, 5), color = ['orange', 'green'])

plt.title('Traffic Type vs Revenue')

plt.show()
# Cat vs Cat

# visitor type vs revenue

data = pd.crosstab(df['VisitorType'], df['Revenue'])

data.plot(kind = 'bar', stacked = True, figsize = (8, 5), color = ['lightgreen', 'green'])

plt.title('Visitor Type vs Revenue')

plt.show()
# Cat vs Cat

# Region vs revenue

data = pd.crosstab(df['Region'], df['Revenue'])

data.plot(kind = 'bar', stacked = True, figsize = (8, 5), color = ['lightblue', 'green'])

plt.title('Visitor Type vs Revenue')

plt.show()
df['OperatingSystems']=df['OperatingSystems'].astype(object)

df['Browser']=df['Browser'].astype(object)

df['Region']=df['Region'].astype(object)

df['TrafficType']=df['TrafficType'].astype(object)

df['SpecialDay']=df['SpecialDay'].astype(object)

df['Administrative']=df['Administrative'].astype(object)

df['Informational']=df['Informational'].astype(object)

df['ProductRelated']=df['ProductRelated'].astype(object)
# month vs pagevalues with respect to revenue

plt.figure(figsize = (25,20))

sns.boxplot(x = df['Month'], y = df['PageValues'], hue = df['Revenue'], palette = 'inferno')

plt.title('Month vs PageValues with respect to Revenue', fontsize = 30)

plt.show()
# month vs exitrates with respect to revenue

plt.figure(figsize = (25,20))

#plt.subplot(2, 2, 2)

sns.boxplot(x = df['Month'], y = df['ExitRates'], hue = df['Revenue'], palette = 'Reds')

plt.title('Month vs ExitRates with respect to Revenue', fontsize = 30)

plt.show()
# month vs bouncerates with respect to revenue

plt.figure(figsize = (25,20))

sns.boxplot(x = df['Month'], y = df['BounceRates'], hue = df['Revenue'], palette = 'Oranges')

plt.title('Month vs BounceRates with respect to Rev.', fontsize = 30)

plt.show()
# VisitorType vs Bouncerates with respect to revenue

plt.figure(figsize = (25,20))

sns.boxplot(x = df['VisitorType'], y = df['BounceRates'], hue = df['Revenue'], palette = 'Purples')

plt.title('Visitors vs BounceRates with respect to Rev.', fontsize = 30)

plt.show()
# visitor type vs exit rates w.r.t revenue

plt.figure(figsize = (25,20))

sns.boxplot(x = df['VisitorType'], y = df['ExitRates'], hue = df['Revenue'], palette = 'rainbow')

plt.title('VisitorType vs ExitRates wrt Rev.', fontsize = 30)

plt.show()
# visitor type vs exit rates w.r.t revenue

plt.figure(figsize = (25,20))

sns.boxplot(x = df['VisitorType'], y = df['PageValues'], hue = df['Revenue'], palette = 'gnuplot')

plt.title('VisitorType vs PageValues wrt Rev.', fontsize = 30)

plt.show()
# region vs pagevalues w.r.t. revenue

plt.figure(figsize = (25,20))

sns.boxplot(x = df['Region'], y = df['PageValues'], hue = df['Revenue'], palette = 'Greens')

plt.title('Region vs PageValues wrt Rev.', fontsize = 30)

plt.show()
#region vs exit rates w.r.t. revenue

plt.figure(figsize = (25,20))

sns.boxplot(x = df['Region'], y = df['ExitRates'], hue = df['Revenue'], palette = 'spring')

plt.title('Region vs Exit Rates w.r.t. Revenue', fontsize = 30)

plt.show()
df_w = df[['Weekend','Revenue']]

df_w.head()
df_w1 = pd.get_dummies(df_w)
df_w1.head()
df_w1.Weekend = df_w1.Weekend.map({False : 0, True : 1})
df_w1.Revenue = df_w1.Revenue.map({False : 0, True :1})
df_w1.head()
from scipy.stats import chi2_contingency

from scipy.stats import chi2
ct=pd.crosstab(df_w.Weekend, df_w.Revenue)

ct
nn=np.array(ct)

nn
stat, p, dof, expected = chi2_contingency(nn)

print('dof=%d' % dof)

print(expected)

# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print(' Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')

# interpret p-value

alpha = 1.0 - prob

print('significance=%.3f, p=%.3f' % (alpha, p))

if p <= alpha:

    print('Dependent (reject H0)')

else:

    print('Independent (fail)')
df.columns
df_vt = pd.crosstab(df.VisitorType, df.Revenue)

df_vt
vt = np.array(df_vt)

vt
stat, p, dof, expected = chi2_contingency(vt)

print('dof=%d' % dof)

print(expected)

# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')

# interpret p-value

alpha = 1.0 - prob

print('significance=%.3f, p=%.3f' % (alpha, p))

if p <= alpha:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
df_tt = pd.crosstab(df.TrafficType, df.Revenue)

df_tt.plot(kind = 'bar')
df_tt.TrafficType = df.TrafficType.replace(to_replace = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], value = 5)
df_tt = pd.crosstab(df_tt.TrafficType, df.Revenue)

df_tt.plot(kind = 'bar')
df.TrafficType.nunique()
tt = np.array(df_tt)

tt
stat, p, dof, expected = chi2_contingency(tt)

print('dof=%d' % dof)

print(expected)

# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')

# interpret p-value

alpha = 1.0 - prob

print('significance=%.3f, p=%.3f' % (alpha, p))

if p <= alpha:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
df_r = pd.crosstab(df.Region, df.Revenue)

df_r.plot(kind = 'bar')

plt.show()
df_r
df_r.iloc[5,:]
df_r.Region = df.Region.replace(to_replace = [5,6,7,8,9], value = 5)
print(df.Region.nunique())

print(df_r.Region.unique())
df_r.Region.unique()
df_r = pd.crosstab(df_r.Region, df.Revenue)

df_r.plot(kind = 'bar')

plt.show()
rg = np.array(df_r)

rg
stat, p, dof, expected = chi2_contingency(rg)

print('dof=%d' % dof)

print(expected)

# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')

# interpret p-value

alpha = 1.0 - prob

print('significance=%.3f, p=%.3f' % (alpha, p))

if p <= alpha:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
df_b = pd.crosstab(df.Browser, df.Revenue)

df_b.plot(kind = 'bar')

plt.show()
df_b.Browser = df.Browser.replace(to_replace = [3,4,5,6,7,8,9,10,11,12,13], value = 3)
print(df.Browser.nunique())

print(df_b.Browser.unique())
df_b = pd.crosstab(df_b.Browser, df.Revenue)

df_b.plot(kind = 'bar')
b = np.array(df_b)

b
stat, p, dof, expected = chi2_contingency(b)

print('dof=%d' % dof)

print(expected)

# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')

# interpret p-value

alpha = 1.0 - prob

print('significance=%.3f, p=%.3f' % (alpha, p))

if p <= alpha:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
df_os = pd.crosstab(df.OperatingSystems, df.Revenue)

df_os.plot(kind = 'bar')

plt.show()
df_os
df_os.OperatingSystems = df.OperatingSystems.replace(to_replace = [4,5,6,7,8], value =4)
df_os = pd.crosstab(df_os.OperatingSystems, df.Revenue)

df_os
os = np.array(df_os)

os
stat, p, dof, expected = chi2_contingency(os)

print('dof=%d' % dof)

print(expected)

# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')

# interpret p-value

alpha = 1.0 - prob

print('significance=%.3f, p=%.3f' % (alpha, p))

if p <= alpha:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
df_m = pd.crosstab(df.Month, df.Revenue)

df_m.plot(kind = 'bar')

plt.show()
df_m.Month = df.Month.replace(to_replace = ['Aug','Feb',' Jul','June','Oct', 'Sep'], value = 'Rest')
df_m = pd.crosstab(df_m.Month, df.Revenue)

df_m.plot(kind = 'bar')

plt.show()
m = np.array(df_m)

m
stat, p, dof, expected = chi2_contingency(m)

print('dof=%d' % dof)

print(expected)

# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')

# interpret p-value

alpha = 1.0 - prob

print('significance=%.3f, p=%.3f' % (alpha, p))

if p <= alpha:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
df_sd = pd.crosstab(df.SpecialDay, df.Revenue)

df_sd.plot(kind = 'bar')

plt.show()
df_sd.SpecialDay = df.SpecialDay.replace(to_replace = [0.2,0.4,0.6,0.8,1.0], value = 1.0)
df_sd = pd.crosstab(df_sd.SpecialDay, df.Revenue)

df_sd.plot(kind = 'bar')

plt.show()
sd = np.array(df_sd)

sd
stat, p, dof, expected = chi2_contingency(sd)

print('dof=%d' % dof)

print(expected)

# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')

# interpret p-value

alpha = 1.0 - prob

print('significance=%.3f, p=%.3f' % (alpha, p))

if p <= alpha:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
sns.pairplot(df,x_vars=['BounceRates','ExitRates'],y_vars=['BounceRates','ExitRates'],hue='Revenue',diag_kind='kde')

plt.show()
df.isnull().sum()[df.isnull().sum()>0]
# For Administrative_Duaration

q1_adm=np.quantile(df.Administrative_Duration,0.25)

q3_adm=np.quantile(df.Administrative_Duration,0.75)

iqr_adm=q3_adm-q1_adm

ll=q1_adm-(1.5*iqr_adm)

ul=q3_adm+(1.5*iqr_adm)

df_ad_out=df[(df.Administrative_Duration<ll) | (df.Administrative_Duration>ul)]

df_ad_out.shape
# for Informational_Duration

q1_inf=np.quantile(df.Informational_Duration,0.25)

q3_inf=np.quantile(df.Informational_Duration,0.75)

iqr_inf=q3_inf-q1_inf

ll=q1_inf-(1.5*iqr_inf)

ul=q3_inf+(1.5*iqr_inf)

df_inf_out=df[(df.Informational_Duration<ll) | (df.Informational_Duration>ul)]

df_inf_out.shape
# for Product Related Duaration

q1_pro=np.quantile(df.ProductRelated_Duration,0.25)

q3_pro=np.quantile(df.ProductRelated_Duration,0.75)

iqr_pro=q3_pro-q1_pro

ll=q1_pro-(1.5*iqr_pro)

ul=q3_pro+(1.5*iqr_pro)

df_pro_out=df[(df.ProductRelated_Duration<ll) | (df.ProductRelated_Duration>ul)]

df_pro_out.shape
# For Bounce Rate

q1_bou=np.quantile(df.BounceRates,0.25)

q3_bou=np.quantile(df.BounceRates,0.75)

iqr_bou=q3_bou-q1_bou

ll=q1_bou-(1.5*iqr_bou)

ul=q3_bou+(1.5*iqr_bou)

df_bou_out=df[(df.BounceRates<ll) | (df.BounceRates>ul)]

df_bou_out.shape
# for Exit Rate

q1_ex=np.quantile(df.ExitRates,0.25)

q3_ex=np.quantile(df.ExitRates,0.75)

iqr_ex=q3_ex-q1_ex

ll=q1_ex-(1.5*iqr_ex)

ul=q3_ex+(1.5*iqr_ex)

df_ex_out=df[(df.ExitRates<ll) | (df.ExitRates>ul)]

df_ex_out.shape
# for Page Values

q1_pg=np.quantile(df.PageValues,0.25)

q3_pg=np.quantile(df.PageValues,0.75)

iqr_pg=q3_pg-q1_pg

ll=q1_pg-(1.5*iqr_pg)

ul=q3_pg+(1.5*iqr_pg)

df_pg_out=df[(df.PageValues<ll) | (df.PageValues>ul)]

df_pg_out.shape
dff=pd.DataFrame()
dff['Administrative_Duration']=df.index.isin(df_ad_out.index)

dff['Informational_Duration']=df.index.isin(df_inf_out.index)

dff['ProductRelated_Duration']=df.index.isin(df_pro_out.index)

dff['BounceRates']=df.index.isin(df_bou_out.index)

dff['ExitRates']=df.index.isin(df_ex_out.index)

dff['PageValues']=df.index.isin(df_pg_out.index)
# Plotting heat map for Otliers
plt.figure(figsize=(20,15))

sns.heatmap(dff , yticklabels = False , cbar = False , cmap = 'viridis')

plt.show()
# Converting Booleans into 1's and 0's

bool_map={True:1,False:0}

df.Weekend.replace(bool_map,inplace=True)

df.Revenue.replace(bool_map,inplace=True)
df.head()
dff.head()
dff['multi'] = ['Y' if x >= 4 else 'N' for x in np.sum(dff.values == True, 1)]
dff['multi'] = ['Y' if x >= 4 else 'N' for x in np.sum(dff.values == True, 1)]
df_new=df[dff['multi']=='N']

df_new.shape
# Converting Booleans into 1's and 0's

bool_map={True:1,False:0}

df_new.Weekend.replace(bool_map,inplace=True)

df_new.Revenue.replace(bool_map,inplace=True)

df_new.head()
# Replacing the Outliers with NAN

df_new.loc[(dff['Administrative_Duration']==True),'Administrative_Duration']=np.NAN

df_new.loc[(dff['Informational_Duration']==True),'Informational_Duration']=np.NAN

df_new.loc[(dff['ProductRelated_Duration']==True),'ProductRelated_Duration']=np.NAN

df_new.loc[(dff['BounceRates']==True),'BounceRates']=np.NAN

df_new.loc[(dff['PageValues']==True),'PageValues']=np.NAN

#df_new=df_new.drop('ExitRates',axis=1)
df_new.isnull().sum()[df_new.isnull().sum()>0]
imp_col=df_new.isnull().sum()[df_new.isnull().sum()>0].index
# Creating dummy Variables

df_dum=pd.get_dummies(df_new)

df_dum.head()
!pip install impyute
from impyute.imputation.cs import mice
imputed_df=mice(df_dum.values)
imputed_df=pd.DataFrame(imputed_df,columns=df_dum.columns)
imputed_df.head()
X=imputed_df.drop(['Revenue','ExitRates'],axis=1)

Y=imputed_df.Revenue

#Y.value_counts(normalize=True)
from sklearn.metrics import f1_score,cohen_kappa_score,classification_report,confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
imputed_df['Revenue'].value_counts().plot(kind='bar')

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegressionCV

log=LogisticRegression(C=0.005994,penalty='l1',solver='liblinear')

log.fit(x_train,y_train)

print('Train score:',log.score(x_train,y_train))

print('Test score:',log.score(x_test,y_test))

#log.C_
log_pred=log.predict(x_test)

print('F1 Score:',f1_score(y_test,log_pred))

print('Kappa Score:',cohen_kappa_score(y_test,log_pred))

print('Classification report:\n',classification_report(y_test,log_pred))
from sklearn.metrics import roc_curve,auc

from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB
fpr,tpr,thresh=roc_curve(y_test,log_pred)

auc_log=auc(fpr,tpr)
dt=DecisionTreeClassifier(max_depth=6)

dt.fit(x_train,y_train)

print('Train score:',dt.score(x_train,y_train))

print('Test score:',dt.score(x_test,y_test))
dt_pred=dt.predict(x_test)

print('F1 Score:',f1_score(y_test,dt_pred))

print('Kappa Score:',cohen_kappa_score(y_test,dt_pred))

print('Classification report:\n',classification_report(y_test,dt_pred))
fpr_dt,tpr_dt,thresh=roc_curve(y_test,dt_pred)

auc_dt=auc(fpr,tpr)
rf_sm=RandomForestClassifier(max_depth=6)

rf_sm.fit(x_train,y_train)

print('Train score:',rf_sm.score(x_train,y_train))

print('Test score:',rf_sm.score(x_test,y_test))
rf_sm_pred=rf_sm.predict(x_test)

print('F1 Score:',f1_score(y_test,rf_sm_pred))

print('Kappa Score:',cohen_kappa_score(y_test,rf_sm_pred))

print('Classification report:\n',classification_report(y_test,rf_sm_pred))
fpr_rf,tpr_rf,thresh=roc_curve(y_test,rf_sm_pred)

auc_rf=auc(fpr,tpr)
plt.plot(fpr,tpr, label='LR(area = %0.2f)' % auc_log,color='red')

plt.plot(fpr_dt, tpr_dt, label='DT(area = %0.2f)' % auc_dt,color='green')

plt.plot(fpr_rf, tpr_rf, label='RF(area = %0.2f)' % auc_rf,color='blue')

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
gnb=GaussianNB()

gnb.fit(x_train,y_train)

print('Train score:',gnb.score(x_train,y_train))

print('Test score:',gnb.score(x_test,y_test))
nb_pred=gnb.predict(x_test)

fpr2,tpr2,thresh=roc_curve(y_test,nb_pred)

auc2=auc(fpr2,tpr2)
dt=DecisionTreeClassifier(max_depth=6)

dt.fit(x_train,y_train)

print('Train score:',dt.score(x_train,y_train))

print('Test score:',dt.score(x_test,y_test))
dt_pred=dt.predict(x_test)

print('F1 Score:',f1_score(y_test,dt_pred))

print('Kappa Score:',cohen_kappa_score(y_test,dt_pred))
fpr3,tpr3,thresh=roc_curve(y_test,dt_pred)

auc3=auc(fpr3,tpr3)
rf_sm=RandomForestClassifier(max_depth=6)

rf_sm.fit(x_train,y_train)

print('Train score:',rf_sm.score(x_train,y_train))

print('Test score:',rf_sm.score(x_test,y_test))
rf_pred_sm=rf_sm.predict(x_test)

print('F1 Score:',f1_score(y_test,rf_pred_sm))

print('Kappa Score:',cohen_kappa_score(y_test,rf_pred_sm))
fpr4,tpr4,thresh=roc_curve(y_test,rf_pred_sm)

auc4=auc(fpr4,tpr4)
!pip install -U imbalanced-learn
from imblearn.over_sampling import RandomOverSampler

smote=RandomOverSampler(random_state=42)

X_new,Y_new=smote.fit_sample(X,Y)

X_new=pd.DataFrame(X_new,columns=X.columns)

Y_new=pd.DataFrame(Y_new,columns=['Revenue'])

X_new.head()
Y_new['Revenue'].value_counts().plot(kind='bar')

plt.show()
x_train,x_test,y_train,y_test=train_test_split(X_new,Y_new,test_size=0.3,random_state=1)
log=LogisticRegression(penalty='l1',solver='liblinear')

log.fit(x_train,y_train)

print('Train score:',log.score(x_train,y_train))

print('Test score:',log.score(x_test,y_test))

log_pred=log.predict(x_test)

print('F1 Score:',f1_score(y_test,log_pred))

print('Kappa Score:',cohen_kappa_score(y_test,log_pred))

print('Classification report:\n',classification_report(y_test,log_pred))
fpr1,tpr1,thresh=roc_curve(y_test,log_pred)

auc1=auc(fpr1,tpr1)
gnb=GaussianNB()

gnb.fit(x_train,y_train)

print('Train score:',gnb.score(x_train,y_train))

print('Test score:',gnb.score(x_test,y_test))
nb_pred=gnb.predict(x_test)

fpr2,tpr2,thresh=roc_curve(y_test,nb_pred)

auc2=auc(fpr2,tpr2)
dt=DecisionTreeClassifier(max_depth=6)

dt.fit(x_train,y_train)

print('Train score:',dt.score(x_train,y_train))

print('Test score:',dt.score(x_test,y_test))
dt_pred=dt.predict(x_test)

print('F1 Score:',f1_score(y_test,dt_pred))

print('Kappa Score:',cohen_kappa_score(y_test,dt_pred))
fpr3,tpr3,thresh=roc_curve(y_test,dt_pred)

auc3=auc(fpr3,tpr3)
rf_sm=RandomForestClassifier(max_depth=6)

rf_sm.fit(x_train,y_train)

print('Train score:',rf_sm.score(x_train,y_train))

print('Test score:',rf_sm.score(x_test,y_test))
rf_pred_sm=rf_sm.predict(x_test)

print('F1 Score:',f1_score(y_test,rf_pred_sm))

print('Kappa Score:',cohen_kappa_score(y_test,rf_pred_sm))
fpr4,tpr4,thresh=roc_curve(y_test,rf_pred_sm)

auc4=auc(fpr4,tpr4)
imp=pd.DataFrame(rf_sm.feature_importances_, columns = ["Imp"], index =x_train.columns)
plt.figure(figsize=(20,8))

imp.sort_values('Imp',ascending=False).head(70).plot(kind='bar')

plt.xticks(rotation=80)

plt.show()
imp.sort_values('Imp',ascending=False).head()
imp.sort_values('Imp',ascending=False).head()
imp2=imp[imp["Imp"]>0.0005]

len(imp2['Imp'])
imp2.sort_values('Imp',ascending=False).index
xnew=X_new[imp2.index]

x_train,x_test,y_train,y_test=train_test_split(xnew,Y_new,test_size=0.3,random_state=1)
log=LogisticRegression(penalty='l1',solver='liblinear')

log.fit(x_train,y_train)

print('Train score:',log.score(x_train,y_train))

print('Test score:',log.score(x_test,y_test))
log_sm1_pred=log.predict(x_test)

print('F1 Score:',f1_score(y_test,log_sm1_pred))

print('Kappa Score:',cohen_kappa_score(y_test,log_sm1_pred))
dt=DecisionTreeClassifier(max_depth=6)

dt.fit(x_train,y_train)

print('Train score:',dt.score(x_train,y_train))

print('Test score:',dt.score(x_test,y_test))
dt_sm1_pred_sm=dt.predict(x_test)

print('F1 Score:',f1_score(y_test,dt_sm1_pred_sm))

print('Kappa Score:',cohen_kappa_score(y_test,dt_sm1_pred_sm))
rf_sm1=RandomForestClassifier(n_estimators=50,max_depth=16)

rf_sm1.fit(x_train,y_train)

print('Train score:',rf_sm1.score(x_train,y_train))

print('Test score:',rf_sm1.score(x_test,y_test))
rf_sm1_pred_sm=rf_sm1.predict(x_test)

print('F1 Score:',f1_score(y_test,rf_sm1_pred_sm))

print('Kappa Score:',cohen_kappa_score(y_test,rf_sm1_pred_sm))
fpr5,tpr5,thresh=roc_curve(y_test,rf_sm1_pred_sm)

auc5=auc(fpr5,tpr5)
gb=GradientBoostingClassifier(n_estimators=50,max_depth=5)

gb.fit(x_train,y_train)

print('Train score:',gb.score(x_train,y_train))

print('Test score:',gb.score(x_test,y_test))
gb_pred_sm=gb.predict(x_test)

print('F1 Score:',f1_score(y_test,gb_pred_sm))

print('Kappa Score:',cohen_kappa_score(y_test,gb_pred_sm))
fpr6,tpr6,thresh=roc_curve(y_test,gb_pred_sm)

auc6=auc(fpr6,tpr6)
plt.plot(fpr1,tpr1, label='LR(area = %0.2f)' % auc1,color='red')

plt.plot(fpr2, tpr2, label='NB(area = %0.2f)' % auc2,color='black')

plt.plot(fpr3, tpr3, label='DT(area = %0.2f)' % auc3,color='magenta')

plt.plot(fpr4, tpr4, label='RF(area = %0.2f)' % auc4,color='blue')

plt.plot(fpr6, tpr6, label='GB(area = %0.2f)' % auc6,color='pink')

plt.plot(fpr6, tpr6, label='RF with FS(area = %0.2f)' % auc5,color='green')





plt.plot([0, 1], [0, 1], 'k--',color='grey')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
y = df['Revenue']
from sklearn.model_selection import KFold,cross_val_score

from sklearn.metrics import f1_score

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import classification_report
models = []

models.append(('LR', LogisticRegression(penalty='l1',solver='liblinear')))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('RF', RandomForestClassifier()))

models.append(('GBM', GradientBoostingClassifier()))

results = []

names = []

scoring = 'accuracy'

for name, model in models:

    kfold = KFold(n_splits=10, random_state=12345)

    cv_results = cross_val_score(model, xnew, Y_new, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (std=%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
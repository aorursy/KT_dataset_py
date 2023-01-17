import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #visualition

import matplotlib.pyplot as plt

import missingno as msno

import scipy.stats as stats

import statsmodels.api as sm

import pylab 

import scipy

from scipy.stats import mannwhitneyu

from scipy.stats import chi2_contingency

from scipy.stats import kstest

from yellowbrick.cluster import KElbowVisualizer

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import sklearn.metrics as metrics





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
germanCreditData=pd.read_csv("/kaggle/input/german-credit-data-with-risk/german_credit_data.csv")
df=germanCreditData.copy()



df.head(10)
df.info()

print("\n")

print("shape: ",df.shape)

df.drop(df.columns[[0]],axis=1,inplace=True)
df.columns
oldColumn = df.columns



newColumn = ["age","sex","job","housing","savingAccounts","checkingAccount","creditAmount","duration","purpose","risk"]
#df.rename(columns={"Age":"age"})



for i in range(len(newColumn)):

    

    df.rename(columns={oldColumn[i]:newColumn[i]},inplace=True)

                

df
df.isnull().values.any()
df.isnull().sum()
msno.bar(df,color=sns.color_palette("deep"));
msno.heatmap(df);
msno.matrix(df,color=(0.5,0.3,0.2));
df.savingAccounts=df.savingAccounts.fillna(value="no account")

df.checkingAccount=df.checkingAccount.fillna(value="no account")

df
ekle = pd.DataFrame(

        {'housing': pd.Categorical(

              values =  df["housing"],

              categories=["free","rent","own"]),



         'savingAccounts': pd.Categorical(

             values = df["savingAccounts"],

             categories=["no account","little","moderate","rich","quite rich"]),



         'checkingAccount': pd.Categorical(

             values = df["checkingAccount"],

             categories=["no account","little","moderate","rich"])

        }

    )
df1 = df.copy()

ekle = ekle.apply(lambda x: x.cat.codes)

ekle.head()
del df1["savingAccounts"]

del df1["checkingAccount"]

del df1["housing"]

df1 = pd.concat([df1,ekle],axis=1)

df1.head()
df1=pd.get_dummies(df1, columns = ["sex"], prefix = ["sex"])

df1=pd.get_dummies(df1, columns = ["risk"], prefix = ["risk"])
del df1["sex_male"]

del df1["risk_bad"]

df1.rename(columns={"risk_good":"risk",

                  "sex_female":"sex"},inplace=True)
df.duration.plot(kind='hist',color='green',bins=20,figsize=(10,5))

plt.title("duration Variable Histogram Chart");
plt.subplot(2,1,1)

df.creditAmount.plot(kind='hist',color='pink',bins=50,figsize=(10,10))

plt.title("creditAmount Variable Histogram Chart");
stats.probplot(df.creditAmount, dist="norm", plot=pylab)

pylab.show()
stat, p = stats.kstest(df["creditAmount"], 'norm')

print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05

if p > alpha:

    print('Credit Amount is distributed normally(H0:fail to reject)')

else:

    print('Credit Amount is not distributed normally.(H0:reject)')
group1 = df1["creditAmount"][df1["risk"] == 1]

group2 = df1["creditAmount"][df1["risk"] == 0]

stat, p = scipy.stats.mannwhitneyu(group1,group2)

print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05

if p > alpha:

    print('it is not significant between Risk and Credit Amount(H0:fail to reject)')

else:

    print('it is significant between Risk and Credit Amount(H0:reject)')
sns.set(style="ticks", palette="pastel")

sns.boxplot(x="risk",y="creditAmount",

             palette=["m", "g"],

            data=df)

sns.despine(offset=10, trim=True)


sns.set(style="ticks", palette="pastel")

# Draw a nested boxplot to show bills by day and time

sns.boxplot(x="housing",y="creditAmount",

            hue="risk", palette=["m", "g"],

            data=df)

sns.despine(offset=10, trim=True)
sns.set(style="whitegrid", palette="pastel", color_codes=True)

sns.violinplot(x="job", y="creditAmount", hue="risk",

               split=True, inner="quarts",

               palette={"good": "G", "bad": "B"},

               data=df);

sns.despine(left=True);
df.purpose.value_counts()
sns.set(style="whitegrid")

sns.boxenplot(x="purpose", y="creditAmount",

              color="b",

              scale="linear", data=df);
sns.pairplot(df, height=3,

                 vars=["creditAmount","duration"],hue="risk");
sns.barplot(x='sex',y='creditAmount',hue='risk',data=df);
sns.boxplot(df.creditAmount);
Q1 = df1.creditAmount.quantile(0.25)

Q3 = df1.creditAmount.quantile(0.75)

IQR = Q3 - Q1
print("Q1:",Q1)

print("Q3:",Q3)

print("IQR:",IQR)
upper_value = Q3 + 1.5*IQR

lower_value = Q1 - 1.5*IQR
print("upper_value:",upper_value)

print("lower_value:",lower_value)
outlier_values = (df1.creditAmount < lower_value) | (df1.creditAmount > upper_value)
df1.creditAmount[outlier_values].value_counts().sum() 
upper_outlier = df1.creditAmount> upper_value

upper_outlier.sum()
df1.creditAmount[upper_outlier] = upper_value
sns.boxplot(df1.creditAmount);
df1.columns
print("Purpose : ",df.purpose.unique())

print("Sex : ",df.sex.unique())

print("Housing : ",df.housing.unique())

print("Saving accounts : ",df['savingAccounts'].unique())

print("Risk : ",df['risk'].unique())

print("Checking account : ",df['checkingAccount'].unique())
df1.age.unique
stat, p = stats.kstest(df["age"], 'norm')

print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05

if p > alpha:

    print('Age is distributed normally(H0:fail to reject)')

else:

    print('Age is not distributed normally.(H0:reject)')
group1 = df["age"][df1["risk"] == 1]

group2 = df["age"][df1["risk"] == 0]

stat, p = scipy.stats.mannwhitneyu(group1,group2)

print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05

if p > alpha:

    print('it is not significant between Risk and Age(H0:fail to reject)')

else:

    print('it is significant between Risk and Age(H0:reject)')
sns.swarmplot(x='risk',y='age',hue='sex',data=df1);
from sklearn.cluster import KMeans

columns = ['job', 'creditAmount', 'duration', 'purpose', 'housing',

       'savingAccounts', 'checkingAccount', 'sex', 'risk']

kumeleme = df1.drop(columns,axis=1)

kumeleme
kmeans = KMeans()

clust = KElbowVisualizer(kmeans, k = (2,20))

clust.fit(kumeleme)

clust.poof()
df1.head()
k_means = KMeans(n_clusters = 3).fit(kumeleme)

cluster = k_means.labels_

plt.scatter(df1.iloc[:,0], df.iloc[:,9], c = cluster, s = 60, cmap = "winter");
df1["age"] = cluster
df1.age.value_counts()
nl = "\n"

crosstab = pd.crosstab(df1['age'], df1['risk'])

chi2, p, dof, expected = stats.chi2_contingency(crosstab)

print(f"Chi-square= {chi2}{nl}p-value= {p}{nl}Degrees of freedom= {dof}")

alpha = 0.05

if p > alpha:

    print('it is not significant between Age and Risk(H0:fail to reject)')

else:

    print('it is significant between Age and Risk(H0:reject)')
sns.countplot(x="age",hue="risk",data=df1);
df.sex.value_counts()
sns.countplot(x="sex",hue="risk",data=df);
nl = "\n"

crosstab = pd.crosstab(df1['sex'], df1['risk'])

chi2, p, dof, expected = stats.chi2_contingency(crosstab)

print(f"Chi-square= {chi2}{nl}p-value= {p}{nl}Degrees of freedom= {dof}")

alpha = 0.05

if p > alpha:

    print('it is not significant between Sex and Risk(H0:fail to reject)')

else:

    print('it is significant between Sex and Risk(H0:reject)')
df.risk.value_counts()
df.housing.value_counts()
sns.countplot(x="housing",hue="risk",data=df1);
nl = "\n"

crosstab = pd.crosstab(df1['housing'], df1['risk'])

chi2, p, dof, expected = stats.chi2_contingency(crosstab)

print(f"Chi-square= {chi2}{nl}p-value= {p}{nl}Degrees of freedom= {dof}")

alpha = 0.05

if p > alpha:

    print('it is not significant between Housing and Risk(H0:fail to reject)')

else:

    print('it is significant between Housing and Risk(H0:reject)')
df.checkingAccount.value_counts()
sns.countplot(x="checkingAccount",hue="risk",data=df1);
nl = "\n"

crosstab = pd.crosstab(df1['checkingAccount'], df1['risk'])

chi2, p, dof, expected = stats.chi2_contingency(crosstab)

print(f"Chi-square= {chi2}{nl}p-value= {p}{nl}Degrees of freedom= {dof}")

alpha = 0.05

if p > alpha:

    print('it is not significant between Checking Account and Risk(H0:fail to reject)')

else:

    print('it is significant between Checking Account and Risk(H0:reject)')
df.savingAccounts.value_counts()
sns.countplot(x="savingAccounts",hue="risk",data=df1);
nl = "\n"

crosstab = pd.crosstab(df1['savingAccounts'], df1['risk'])

chi2, p, dof, expected = stats.chi2_contingency(crosstab)

print(f"Chi-square= {chi2}{nl}p-value= {p}{nl}Degrees of freedom= {dof}")

alpha = 0.05

if p > alpha:

    print('it is not significant between Saving Accounts and Risk(H0:fail to reject)')

else:

    print('it is significant between Saving Accounts and Risk(H0:reject)')
risk2=df.risk.value_counts()
sns.barplot( x=risk2.index,y=risk2.values,data=df);
sns.countplot(x="purpose",hue="risk",data=df);
nl = "\n"

crosstab = pd.crosstab(df1['purpose'], df1['risk'])

chi2, p, dof, expected = stats.chi2_contingency(crosstab)

print(f"Chi-square= {chi2}{nl}p-value= {p}{nl}Degrees of freedom= {dof}")

alpha = 0.05

if p > alpha:

    print('it is not significant between Purpose and Risk(H0:fail to reject)')

else:

    print('it is significant between Purpose and Risk(H0:reject)')
purpose_vs_Risk = pd.crosstab(index=df1["purpose"], 

                             columns=df1["risk"],

                             margins=True)



purpose_vs_Risk
df1.purpose[df1.purpose == "domestic appliances"] = "furniture/equipment"
nl = "\n"

crosstab = pd.crosstab(df1['purpose'], df1['risk'])

chi2, p, dof, expected = stats.chi2_contingency(crosstab)

print(f"Chi-square= {chi2}{nl}p-value= {p}{nl}Degrees of freedom= {dof}")

alpha = 0.05

if p > alpha:

    print('it is not significant between Purpose and Risk(H0:fail to reject)')

else:

    print('it is significant between Purpose and Risk(H0:reject)')
ekle1 = pd.DataFrame({'purpose': pd.Categorical(

             values = df1["purpose"],

             categories=["repairs","vacation/others","furniture/equipment"

                         ,"radio/TV","education","business","car"])

    }

)
df2 = df1.copy()

ekle1 = ekle1.apply(lambda x: x.cat.codes)

ekle1.head()
del df2["purpose"]

df2 = pd.concat([df2,ekle1],axis=1)

df2.head()
df1=pd.get_dummies(df1, columns = ["purpose"], prefix = ["p"])
del df1["p_repairs"]

df1.head()
nl = "\n"

crosstab = pd.crosstab(df1['job'], df1['risk'])

chi2, p, dof, expected = stats.chi2_contingency(crosstab)

print(f"Chi-square= {chi2}{nl}p-value= {p}{nl}Degrees of freedom= {dof}")

alpha = 0.05

if p > alpha:

    print('it is not significant between Job and Risk(H0:fail to reject)')

else:

    print('it is significant between Job and Risk(H0:reject)')
job_vs_Risk = pd.crosstab(index=df1["job"], 

                             columns=df1["risk"],

                             margins=True)



job_vs_Risk
df2.head()
y = df2["risk"]

X = df2.drop(["risk"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.30, 

                                                    random_state=982)
xgb_tuned1 = XGBClassifier(learning_rate= 0.01, 

                                max_depth= 7, 

                                n_estimators= 1000, 

                                subsample= 0.7).fit(X_train, y_train)

y_pred = xgb_tuned1.predict(X_test)

accuracy_score(y_test,y_pred)
metrics.confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

roc_auc = metrics.auc(fpr, tpr)

plt.title('Feature Characteristics')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
feature_imp = pd.Series(xgb_tuned1.feature_importances_,

                        index=X_train.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel("Feature Significance Scores")

plt.ylabel('Features')

plt.title("Significance Levels")

plt.show()
df1.head()
y = df1["risk"]

X = df1.drop(["risk"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.30, 

                                                    random_state=982)
xgb_tuned2 = XGBClassifier(learning_rate= 0.01, 

                                max_depth= 7, 

                                n_estimators= 1000, 

                                subsample= 0.7).fit(X_train, y_train)

y_pred = xgb_tuned2.predict(X_test)

accuracy_score(y_test,y_pred)
metrics.confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

roc_auc = metrics.auc(fpr, tpr)

plt.title('Feature Characteristics')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
feature_imp = pd.Series(xgb_tuned2.feature_importances_,

                        index=X_train.columns).sort_values(ascending=False)



sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel("Feature Significance Scores")

plt.ylabel('Features')

plt.title("Significance Levels")

plt.show()
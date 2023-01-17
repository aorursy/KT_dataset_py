# Importing Librarys

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV # to split the data

from sklearn.metrics import accuracy_score, classification_report, roc_curve #To evaluate our model

from sklearn.externals import joblib

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

from IPython.display import display, HTML

pd.set_option('display.max_columns', None)



# Algorithmns models to be compared

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier



import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/german-data/german.data",sep=" ",header=None)

headers=["Status of existing checking account","Duration in month","Credit history",\

         "Purpose","Credit amount","Savings account/bonds","Present employment since",\

         "Installment rate in percentage of disposable income","Personal status and sex",\

         "Other debtors / guarantors","Present residence since","Property","Age in years",\

        "Other installment plans","Housing","Number of existing credits at this bank",\

        "Job","Number of people being liable to provide maintenance for","Telephone","foreign worker","Risk"]

df.columns=headers
print(df.shape)

print (df.columns)
# To preview the data set

df.head(5)
# Looking unique values

print(df.nunique())
# Remove the missing values

df = df.dropna(how='any',axis=0)
Status_of_existing_checking_account={'A14':"no checking account",'A11':"<0 SGD", 'A12': "0 <= <200 SGD",'A13':">= 200 SGD "}

df["Status of existing checking account"]=df["Status of existing checking account"].map(Status_of_existing_checking_account)



Credit_history={"A34":"critical account","A33":"delay in paying off","A32":"existing credits paid back duly till now","A31":"all credits at this bank paid back duly","A30":"no credits taken"}

df["Credit history"]=df["Credit history"].map(Credit_history)



Purpose={"A40" : "car (new)", "A41" : "car (used)", "A42" : "furniture/equipment", "A43" :"radio/television" , "A44" : "domestic appliances", "A45" : "repairs", "A46" : "education", 'A47' : 'vacation','A48' : 'retraining','A49' : 'business','A410' : 'others'}

df["Purpose"]=df["Purpose"].map(Purpose)



Saving_account={"A65" : "no savings account","A61" :"<100 SGD","A62" : "100 <= <500 SGD","A63" :"500 <= < 1000 SGD", "A64" :">= 1000 SGD"}

df["Savings account/bonds"]=df["Savings account/bonds"].map(Saving_account)



Present_employment={'A75':">=7 years", 'A74':"4<= <7 years",  'A73':"1<= < 4 years", 'A72':"<1 years",'A71':"unemployed"}

df["Present employment since"]=df["Present employment since"].map(Present_employment)



Personal_status_and_sex={ 'A95':"female:single",'A94':"male:married/widowed",'A93':"male:single", 'A92':"female:divorced/separated/married", 'A91':"male:divorced/separated"}

df["Personal status and sex"]=df["Personal status and sex"].map(Personal_status_and_sex)



Other_debtors_guarantors={'A101':"none", 'A102':"co-applicant", 'A103':"guarantor"}

df["Other debtors / guarantors"]=df["Other debtors / guarantors"].map(Other_debtors_guarantors)



Property={'A121':"real estate", 'A122':"savings agreement/life insurance", 'A123':"car or other", 'A124':"unknown / no property"}

df["Property"]=df["Property"].map(Property)



Other_installment_plans={'A143':"none", 'A142':"store", 'A141':"bank"}

df["Other installment plans"]=df["Other installment plans"].map(Other_installment_plans)



Housing={'A153':"for free", 'A152':"own", 'A151':"rent"}

df["Housing"]=df["Housing"].map(Housing)



Job={'A174':"management/ highly qualified employee", 'A173':"skilled employee / official", 'A172':"unskilled - resident", 'A171':"unemployed/ unskilled  - non-resident"}

df["Job"]=df["Job"].map(Job)



Telephone={'A192':"yes", 'A191':"none"}

df["Telephone"]=df["Telephone"].map(Telephone)



foreign_worker={'A201':"yes", 'A202':"no"}

df["foreign worker"]=df["foreign worker"].map(foreign_worker)



risk={1:"Good Risk", 2:"Bad Risk"}

df["Risk"]=df["Risk"].map(risk)
df.head(5)
# Total number of good and bad risk

ax = sns.catplot(x='Risk', kind="count", palette="ch:.25", data=df)

ax.fig.subplots_adjust(top=0.9)

ax.fig.suptitle('Total number of Good and Bad risk', fontsize=16)
# Amount of loans granted according to purpose

n_credits = df.groupby("Purpose")["Personal status and sex"].count().rename("Count").reset_index()

n_credits.sort_values(by=["Count"], ascending=False, inplace=True)



plt.figure(figsize=(10,6))

bar = sns.barplot(x="Purpose",y="Count",data=n_credits)

bar.set_xticklabels(bar.get_xticklabels(), rotation=60)

plt.title("Amount of loans granted according to Purpose")

plt.ylabel("Number of granted credits")

plt.tight_layout()
# Amount of loans granted according to age group

n_credits = df.groupby("Age in years")["Personal status and sex"].count().rename("Count").reset_index()

n_credits.sort_values(by=["Count"], ascending=False, inplace=True)



plt.figure(figsize=(10,6))

bar = sns.barplot(x="Age in years",y="Count",data=n_credits)

bar.set_xticklabels(bar.get_xticklabels(), rotation=60)

plt.title("Amount of loans granted according to age group")

plt.ylabel("Number of granted credits")

plt.tight_layout()
# Credit Distribution based on sex



fig = plt.figure(figsize=(7,7))   # Veri kümesinde ki cinsiyet dağılımı

df['Personal status and sex'].value_counts().plot(kind = 'pie', autopct='%.1f%%')

plt.ylabel(" ", fontsize = 20)

plt.title("Credit Distribution by Personal status and sex")

print("")
def style_stats_specific_cell(x):



    color_thresh = 'background-color: red'

    

    df_color = pd.DataFrame('', index=x.index, columns=x.columns)

    rows_number=len(x.index)

    for r in range(0,rows_number):

        try:

            val=(x.iloc[r, 1])

            if val>0.05:

                df_color.iloc[r, 1]=color_thresh

        except:

            pass

    return df_color
column_names_cat_stats=["Status of existing checking account","Credit history","Purpose",\

"Savings account/bonds","Present employment since",\

"Installment rate in percentage of disposable income","Personal status and sex",\

"Other debtors / guarantors","Present residence since","Property",\

"Other installment plans","Housing","Number of existing credits at this bank",\

"Job","Number of people being liable to provide maintenance for","Telephone","foreign worker"]



statistical_significance=[]

for attr in column_names_cat_stats:

    data_count=pd.crosstab(df[attr],df["Risk"]).reset_index()

    obs=np.asarray(data_count[["Bad Risk","Good Risk"]])

    chi2, p, dof, expected = stats.chi2_contingency(obs)

    statistical_significance.append([attr,round(p,4)])

statistical_significance=pd.DataFrame(statistical_significance)

statistical_significance.columns=["Attribute","P-value"]

display(statistical_significance.style.apply(style_stats_specific_cell, axis=None))



print("\n")



statistical_significance=[]

column_names_cont_stats=["Credit amount","Age in years","Duration in month"]

good_risk_df = df[df["Risk"]=="Good Risk"]

bad_risk_df = df[df["Risk"]=="Bad Risk"]

for attr in column_names_cont_stats:

    statistic, p=stats.f_oneway(good_risk_df[attr].values,bad_risk_df[attr].values)

    statistical_significance.append([attr,round(p,4)])

statistical_significance=pd.DataFrame(statistical_significance)

statistical_significance.columns=["Attribute","P-value"]

display(statistical_significance.style.apply(style_stats_specific_cell, axis=None))
attr_significant=["Status of existing checking account","Credit history","Purpose",\

"Savings account/bonds","Present employment since",\

"Personal status and sex","Property","Other installment plans","Housing","foreign worker",\

"Credit amount","Age in years","Duration in month"]

target_variable=["Risk"]

df=df[attr_significant+target_variable]
col_cat_names=["Status of existing checking account","Credit history","Purpose",\

"Savings account/bonds","Present employment since",\

"Personal status and sex","Property","Other installment plans","Housing","foreign worker"]

for attr in col_cat_names:

    df = df.merge(pd.get_dummies(df[attr], prefix=attr), left_index=True, right_index=True)

    df.drop(attr,axis=1,inplace=True)
df.head(5)
risk={"Good Risk":1, "Bad Risk":0}

df["Risk"]=df["Risk"].map(risk)
X = df.drop('Risk', 1).values #independent variables

y = df["Risk"].values #target variables
plt.figure(figsize=(14,12))

sns.heatmap(df.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True,  linecolor='white', annot=True)

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
# to feed the random state

seed = 7

results = []

names = []

scoring = 'recall'



# prepare models

models = []

models.append(('LogisticRegression\t\t', LogisticRegression()))

models.append(('LinearDiscriminantAnalysis\t', LinearDiscriminantAnalysis()))

models.append(('KNeighborsClassifier\t\t', KNeighborsClassifier()))

models.append(('DecisionTreeClassifier\t\t', DecisionTreeClassifier()))

models.append(('GaussianNB\t\t\t', GaussianNB()))

models.append(('RandomForestClassifier\t\t', RandomForestClassifier()))

models.append(('SVC\t\t\t\t', SVC(gamma='auto')))

models.append(('GradientBoostingClassifier\t', GradientBoostingClassifier()))

models.append(('XGBClassifier\t\t\t', XGBClassifier()))

print("Accuracy_Score:")

print("--------------")

for name, model in models:

    model.fit(X_train,y_train)

    kfold = KFold(n_splits=10, random_state=seed)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    print(name,'= {0:.0f}%'.format(accuracy_score(y_test, model.predict(X_test)) * 100))
importance = model.feature_importances_

feature_indexes_by_importance = importance.argsort()



for index in feature_indexes_by_importance:

  print("{} - {:.2f}%".format(df.columns[index], (importance[index] * 100.0)))
model = XGBClassifier()

model.fit(X_train, y_train)

print('Accuracy_Score = {:.0f}%'.format(accuracy_score(y_test, model.predict(X_test)) * 100))

print('classification_report = ',classification_report(y_test, model.predict(X_test)))
param_grid = {"max_depth": [3, 5, 7, 10],

              "n_estimators":[10, 50, 250, 500, 1000],

              "max_features": [4, 7, 15, 20],

              "learning_rate": [0.1, 0.05, 0.001]}

model = XGBClassifier()

grid_search = GridSearchCV(model, param_grid=param_grid, n_jobs=4, verbose=100)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
model = XGBClassifier(learning_rate= 0.05, max_depth= 3, max_features= 4, n_estimators= 250)

model.fit(X_train, y_train)

print(model, '\nAccuracy_Score = {:.0f}%'.format(accuracy_score(y_test, model.predict(X_test)) * 100))

print('Classification_Report = ',classification_report(y_test, model.predict(X_test)))
joblib.dump(model, 'trained_credit_risk_model.pkl')
model = LogisticRegression()

model.fit(X_train, y_train)

print(model, '\nAccuracy_Score = {:.0f}%'.format(accuracy_score(y_test, model.predict(X_test)) * 100))

print('classification_report = ',classification_report(y_test, model.predict(X_test)))
dual=[True,False]

max_iter=[100,110,120,130,140]

C = [1.0,1.5,2.0,2.5]

param_grid = dict(dual=dual,max_iter=max_iter,C=C)

model = LogisticRegression()

grid_search = GridSearchCV(model, param_grid, n_jobs=4, verbose=100)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
model = LogisticRegression(C= 1.0, dual= False, max_iter= 100)

model.fit(X_train, y_train)

print(model, '\nAccuracy_Score = {:.0f}%'.format(accuracy_score(y_test, model.predict(X_test)) * 100))

print('classification_report = ',classification_report(y_test, model.predict(X_test)))
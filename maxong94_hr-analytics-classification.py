import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data_dict = "/kaggle/input/hr-analytics-case-study/data_dictionary.xlsx"

data_dict= pd.read_excel(data_dict)

employee_survey = "/kaggle/input/hr-analytics-case-study/employee_survey_data.csv"



satis_survey= pd.read_csv(employee_survey)

satis_survey.head(5)
data_dict
time_in = "/kaggle/input/hr-analytics-case-study/in_time.csv"



time_in = pd.read_csv(time_in)

time_in.head(5)
manager_rating = "/kaggle/input/hr-analytics-case-study/manager_survey_data.csv"

manager_rating= pd.read_csv(manager_rating)

manager_rating.head(5)
time_out = "/kaggle/input/hr-analytics-case-study/out_time.csv"



time_out= pd.read_csv(time_out)

time_out.head(5)
general = "/kaggle/input/hr-analytics-case-study/general_data.csv"



general = pd.read_csv(general)

general.head(5)
general.shape
ordinal = general[["Education", 'JobLevel','TrainingTimesLastYear']]
survey = satis_survey.merge(manager_rating, on = "EmployeeID")

survey = survey.astype("float64")

subset = survey.drop(["EmployeeID"],axis =1)

survey_results= pd.concat([subset,ordinal], axis =1)

survey_results

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")



b= imputer.fit_transform(survey_results)

column_names= list(survey_results.columns)

final_ordinal= pd.DataFrame(b, columns = column_names)



final_ordinal_df = pd.concat([final_ordinal, general["Attrition"]],axis = 1)

final_ordinal_df
#general.dtypes

quali = [i for i in general.columns if general[i].dtypes == "object"]

quanti = [i for i in general.columns if i not in quali]

quanti_final = list(set(quanti).difference({'StockOptionLevel','Education','JobLevel','TrainingTimesLastYear',"EmployeeID","EmployeeCount","StandardHours"}))

print("Number of quali features :", len(quali))

print("="* 50)

print(quali)

print("="* 50)

print("Number of quanti features :", len(quanti_final))

print("="* 50)

print(quanti_final)

print("="* 50)

print("Number of ordinal features: ",len(list(final_ordinal.columns)))

print("="* 50)

print(list(final_ordinal.columns))



#general.dtypes.index
df_quanti= general[quanti_final]

df_quanti["TotalWorkingYears"].value_counts(dropna = False)

df_quanti["TotalWorkingYears"] = df_quanti["TotalWorkingYears"].fillna(df_quanti["TotalWorkingYears"].mode()[0])

df_quanti["NumCompaniesWorked"] = df_quanti["NumCompaniesWorked"].fillna(1.0)

df_quanti["NumCompaniesWorked"].value_counts(dropna = False)
df_quanti.isnull().sum()
df_quanti_total = pd.concat([df_quanti,general[["Attrition"]]], axis = 1)
df_quanti_total["YearsSinceLastPromotion"].value_counts()
fig,ax = plt.subplots(9,9, figsize = (30,30))

fig.tight_layout(pad = 4.0)

axes = ax.flatten()

col_list = df_quanti.columns.to_list()

i = 0

for xfeatures in col_list:

    for yfeatures in col_list:

        sns.scatterplot(data = df_quanti_total,x = xfeatures, y = yfeatures, ax = axes[i])

        i +=1
corr = df_quanti.corr("spearman")

fig,ax = plt.subplots(figsize = (12,6))

# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))

cmap = sns.diverging_palette(220,10,as_cmap = True)

sns.heatmap(corr, mask = mask, cmap = cmap, annot = True, ax = ax)
corr = final_ordinal.corr("spearman")

fig,ax = plt.subplots(figsize = (12,6))

# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))

cmap = sns.diverging_palette(220,10,as_cmap = True)

sns.heatmap(corr, mask = mask, cmap = cmap, annot = True, ax = ax)
merged_continuous = pd.concat([df_quanti, final_ordinal_df], axis = 1)
from sklearn.model_selection import train_test_split

X = merged_continuous.drop(["Attrition"], axis =1 )

y = merged_continuous["Attrition"]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42)

y_train.value_counts(normalize = True)
from imblearn.over_sampling import SMOTE 

oversample = SMOTE()

X_over,y_over= oversample.fit_resample(X_train,y_train)
from sklearn.pipeline import Pipeline 

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



pipe = Pipeline(steps = [("scaler",MinMaxScaler()),("logerg", LogisticRegression(penalty = "l1",

                                                                                 C = 50,

                                                                                max_iter = 250,

                                                                                solver ="saga",

                                                                                class_weight = "balanced"))])
pipe.fit(X_over,y_over)
from sklearn.metrics import classification_report 

y_over_pred = pipe.predict(X_over)

print(classification_report(y_over, y_over_pred))

y_train_pred = pipe.predict(X_train)

print(classification_report(y_train, y_train_pred))

y_test_pred = pipe.predict(X_test)

print(classification_report(y_test, y_test_pred))

from sklearn.ensemble import GradientBoostingClassifier



pipe_3 = Pipeline(steps = [("scaler",MinMaxScaler()),("gbc", GradientBoostingClassifier(n_estimators = 100,

                                                                                learning_rate = 1.0,

                                                                                        min_samples_split = 30,

                                                                                        max_features = "sqrt",

                                                                                           random_state = 42))])

pipe_3.fit(X_over,y_over)
from sklearn.metrics import classification_report 

y_over_pred = pipe_3.predict(X_over)

print(classification_report(y_over, y_over_pred))

y_train_pred = pipe_3.predict(X_train)

print(classification_report(y_train, y_train_pred))

y_test_pred = pipe_3.predict(X_test)

print(classification_report(y_test, y_test_pred))
df_without_attrition= merged_continuous.drop(["Attrition"],axis =1)

a = pd.DataFrame(dict(zip(df_without_attrition.columns,pipe_3["gbc"].feature_importances_)),index = [0])

a
sns.set()

descending_list = a.iloc[0].sort_values(ascending = False).index.tolist()

a_ordered_df = a[descending_list]

sns.barplot(y = a_ordered_df.columns, 

            x = a_ordered_df.iloc[0],

           orient = 'h')
#important_features = merged_continuous[["Attrition","YearsWithCurrManager","JobSatisfaction","MonthlyIncome","EnvironmentSatisfaction","Age","JobInvolvement","WorkLifeBalance","DistanceFromHome"]].groupby("Attrition").mean()

#features_df= important_features.reset_index()

#features_df
#fig,ax = plt.subplots(3,3, figsize = (12,10))

#fig.tight_layout(pad = 2.0)

#axes = ax.flatten()

#without_attrition= features_df.drop(["Attrition"],axis = 1)

#col_list = without_attrition.columns.to_list()

#i = 0

#

#for features in col_list:

#    sns.barplot(x = "Attrition", y = without_attrition[features] ,data = features_df,ax = axes[i])

#    i +=1

#

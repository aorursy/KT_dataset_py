# Import basic packages

import pandas as pd

import numpy as np

pd.options.display.max_columns = None



import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display

# Output plots in notebook

% matplotlib inline

% config InlineBackend.figure_format = 'retina'



import warnings

warnings.filterwarnings("ignore")



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
Data = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

Data.head()
Data["Attrition"].value_counts()
sns.set(style="whitegrid", font_scale=1.3)

sns.countplot(x="Attrition", data=Data, palette="hls")

sns.plt.title("Attrition Counts")
Data.info()
# columns name list

cols = Data.columns

num_cols = Data._get_numeric_data().columns

cat_cols = cols.drop(num_cols.tolist())
print(num_cols)
Data[num_cols].describe()
Data.drop(["EmployeeCount", "EmployeeNumber", "StandardHours"], axis=1, inplace=True)
for cat_col in cat_cols:

    display(Data[cat_col].value_counts())
# make Business_Travel

Data["Business_Travel"] = Data["BusinessTravel"].map({"Non-Travel":0, "Travel_Rarely":1, "Travel_Frequently":2})



# make Dapartment_JobRole

Data["Department_JobRole"] = Data["Department"] + " : " + Data["JobRole"]

# make binary data

Data["MaritalStatus_Married"] = pd.get_dummies(Data["MaritalStatus"])["Married"]

Data = pd.concat([Data, pd.get_dummies(Data[["Gender", "OverTime", "Attrition"]], drop_first=True)], axis=1)



# drop 

Data.drop(["BusinessTravel", "Department", "JobRole", "MaritalStatus", "Gender", "OverTime", "Attrition", "Over18"], axis=1, inplace=True)
Data.head()
cols = Data.columns

num_cols = Data._get_numeric_data().columns

cat_cols = cols.drop(num_cols.tolist())
print("Numeric data\n", num_cols)

print("Categorical data\n", cat_cols)
sns.set(style="whitegrid", font_scale=0.8)

plt.figure(figsize=(13,13))

corr = round(Data.corr(),2)

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, annot=True, cmap="RdBu", mask=mask, )

plt.title("Correlation between features", fontdict={"fontsize":20})
extract_cols  = ["Age", "JobLevel", "MonthlyIncome", "PercentSalaryHike", "PerformanceRating", "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole","YearsSinceLastPromotion", "OverTime_Yes", "Attrition_Yes"]

sns.set(style="whitegrid", font_scale=1.2)

plt.figure(figsize=(10,7))

corr = round(Data[extract_cols].corr(),2)

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, annot=True, cmap="RdBu", vmin=-1, vmax=1, mask=mask)

plt.title("Correlation between features / important features", fontdict={"fontsize":20})
Data_copy = Data.copy()

scale_cols = Data_copy.columns.drop(["Department_JobRole", "EducationField", "Attrition_Yes"])

Data_copy[scale_cols] = (Data_copy[scale_cols] - Data_copy[scale_cols].mean()) / Data_copy[scale_cols].std()
Att = Data_copy.groupby(["Attrition_Yes"], as_index=False).mean().transpose()

Att.head()
Att_sep = [Att[[x]] for x in range(len(Att.columns))]

Att_plot = pd.DataFrame([], columns=["mean", "feature","kind"])

for (i,data) in enumerate(Att_sep):

    data["feature"] = data.index

    data["kind"] = "Attrition_Yes_" + data.loc["Attrition_Yes"].astype(str).values[0]

    data.rename(columns={i:"mean"}, inplace=True)

    data.drop(["Attrition_Yes"], axis=0, inplace=True)

    Att_plot = pd.concat([Att_plot, data], axis=0)
# organize features

features_1 = ["Age","Gender_Male","MaritalStatus_Married","Education","DistanceFromHome","NumCompaniesWorked"]

features_2 = ["JobInvolvement","JobLevel","JobSatisfaction","EnvironmentSatisfaction",

              "RelationshipSatisfaction", "WorkLifeBalance", "Business_Travel", "OverTime_Yes",]

features_3 = ["HourlyRate","DailyRate","MonthlyRate","MonthlyIncome",

              "PercentSalaryHike","StockOptionLevel","PerformanceRating"]

features_4 = ["TotalWorkingYears","YearsAtCompany","YearsInCurrentRole","YearsWithCurrManager","YearsSinceLastPromotion","TrainingTimesLastYear"]

# make pandas frame for plot

features = [features_1, features_2, features_3, features_4]

Att_plot = Att_plot.loc[features_1 + features_2 + features_3 + features_4]
def feature_plot(input_data, title, palette="hls", size=4, aspect=3, rotation=0, ylim=None):

    ax = sns.factorplot(x="feature", y="mean", hue="kind", data=input_data, palette=palette, size=size, aspect=aspect)

    ax.set(xlabel="", ylim=ylim)

    ax.set_xticklabels(rotation=rotation)

    plt.title(title, fontdict={"fontsize":17})

    sns.despine(left=True, bottom=True)
sns.set(style="whitegrid", font_scale=1.2)

feature_plot(input_data=Att_plot, palette=sns.color_palette("hls",2)[::-1], title="mean distribution / features", size=6, aspect=2, rotation=90, ylim=(-1,1))
important_cols = ["Age", "JobInvolvement", "JobLevel", "MonthlyIncome", "StockOptionLevel", "TotalWorkingYears", "YearsAtCompany", "YearsInCurrManager", "Business_Travel", "OverTime_Yes"]

feature_plot(input_data=Att_plot.loc[important_cols], palette=sns.color_palette("hls",2)[::-1], title="mean distribution / important features", size=4, aspect=3, rotation=90, ylim=(-1,1))
print(cat_cols)
Data_copy["EducationField"].value_counts()
# replace long word to short word.

Data_copy["EducationField"].replace({"Life Sciences":"LifeSc", "Technical Degree":"Technical", "Human Resources":"HR"}, inplace=True)
ax = sns.barplot(x="EducationField", y="Attrition_Yes", data=Data_copy, palette="hls")

ax.set_ylabel("Propotion")

sns.plt.title("Propotion of Attrition_Yes / EducationField")
Education_Att = Data_copy.groupby(["EducationField", "Attrition_Yes"], as_index=False).mean()

Education_Att = Education_Att[Education_Att["Attrition_Yes"] == 1].transpose()

Education_Att.head()
Education_sep = [Education_Att[[x]] for x in Education_Att.columns]

Education_plot = pd.DataFrame([], columns=["mean", "feature", "kind"])

for (col,data) in zip(Education_Att.columns, Education_sep):

    data["feature"] = data.index

    data["kind"] = data.loc["EducationField"].values[0] + "_" + data.loc["Attrition_Yes"].astype(str).values[0]

    data.rename(columns={col:"mean"}, inplace=True)

    data.drop(["EducationField", "Attrition_Yes"], axis=0, inplace=True)

    Education_plot = pd.concat([Education_plot, data], axis=0)

Education_plots = [Education_plot.loc[x] for x in features]
graph_titles = ["Personal Information / EducationField", "Job Information / EducationField", "Evaluation / EducationField", "Working History / EducationField"]

for (i,title) in enumerate(graph_titles):

    feature_plot(input_data=Education_plots[i], title=title, rotation=15, ylim=(-1,1.2))
important_cols = ["Age", "JobInvolvement", "JobLevel", "MonthlyIncome", "StockOptionLevel", "TotalWorkingYears", "YearsAtCompany", "YearsInCurrManager", "Business_Travel", "OverTime_Yes"]

feature_plot(input_data=Education_plot.loc[important_cols], title="Important features / EducationField", size=4, aspect=3, rotation=15, ylim=(-1,1.2))
Data_copy["Department_JobRole"].value_counts()
# replace long word to short word.

Data_copy["Department_JobRole"].replace({"Sales : Sales Executive":"Sales : Executive",

                              "Sales : Sales Representative":"Sales : Representative",

                              "Sales : Manager":"Sales : Manager",

                              "Research & Development : Research Scientist":"R&D : RS",

                              "Research & Development : Laboratory Technician":"R&D : Lab",

                              "Research & Development : Manufacturing Director":"R&D : MD",

                              "Research & Development : Healthcare Representative":"R&D : Health",

                              "Research & Development : Research Director":"R&D : RD",

                              "Research & Development : Manager":"R&D : Manager",

                              "Human Resources : Human Resources":"HR : HR",

                              "Human Resources : Manager":"HR : Manager"}, inplace=True)
sns.set(style="whitegrid", font_scale=1.1)

ax = sns.factorplot(x="Department_JobRole", y="Attrition_Yes", kind="bar", data=Data_copy, size=4, aspect=3, palette="hls")

ax.set(xlabel="", ylabel="Propotion")

ax.set_xticklabels(rotation=15)

plt.title("Propotion of Attrition_Yes / Job Role", fontdict={"fontsize":16})
JobRole_Att = Data_copy.groupby(["Department_JobRole", "Attrition_Yes"], as_index=False).mean()

JobRole_Att = JobRole_Att[JobRole_Att["Attrition_Yes"] == 1].transpose()

JobRole_Att.head()
JobRole_sep = [JobRole_Att[[x]] for x in JobRole_Att.columns]

JobRole_plot = pd.DataFrame([], columns=["mean", "feature", "kind"])

for (col,data) in zip(JobRole_Att.columns, JobRole_sep):

    data["feature"] = data.index

    data["kind"] = data.loc["Department_JobRole"].values[0] + "_" + data.loc["Attrition_Yes"].astype(str).values[0]

    data.rename(columns={col:"mean"}, inplace=True)

    data.drop(["Department_JobRole", "Attrition_Yes"], axis=0, inplace=True)

    JobRole_plot = pd.concat([JobRole_plot, data], axis=0)

JobRole_plots = [JobRole_plot.loc[x] for x in features]
# make color palette

HR_color = sns.color_palette("Reds",1)

RD_color = sns.color_palette("Blues", 10)

RD_color_sorted = [RD_color[1], RD_color[8], RD_color[2], RD_color[5], RD_color[4], RD_color[9]]

Sales_color = sns.color_palette("Greens", 3)

job_color = HR_color + RD_color_sorted + Sales_color

sns.palplot(job_color)
graph_titles = ["Personal Information / JobRole", "Job Information / JobRole", "Evaluation / JobRole", "Working History / JobRole"]

for (i,title) in enumerate(graph_titles):

    feature_plot(input_data=JobRole_plots[i], palette=job_color, title=title, size=4.5, aspect=2.5,rotation=15)
important_cols = ["Age", "JobInvolvement", "JobLevel", "MonthlyIncome", "StockOptionLevel", "TotalWorkingYears", "YearsAtCompany", "YearsInCurrManager", "Business_Travel", "OverTime_Yes"]

feature_plot(input_data=JobRole_plot.loc[important_cols], palette=job_color, title="important features / EducationField", size=4.5, aspect=2.5, rotation=15)
from sklearn.cross_validation import cross_val_predict, KFold

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, f1_score

from sklearn.model_selection import train_test_split
Data.head()
Data = pd.concat([Data, pd.get_dummies(Data[["EducationField", "Department_JobRole"]])], axis=1)

Data.drop(["EducationField", "Department_JobRole"], axis=1, inplace=True)

Data.head()
cols = Data.columns.drop("Attrition_Yes")

features = Data[cols]

target = Data[["Attrition_Yes"]]
kf = KFold(features.shape[0], random_state=1, n_folds=10)
target = target.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=1)
# define score function

def print_clf_score(input_predictions):

    pd.DataFrame(confusion_matrix(Data["Attrition_Yes"], input_predictions), index=["true_0", "true_1"], columns=["pred_0","pred_1"])

    print(classification_report(Data["Attrition_Yes"], input_predictions))

    print("accuracy: ", accuracy_score(Data["Attrition_Yes"], input_predictions))

    print("f1_score: ", f1_score(Data["Attrition_Yes"], input_predictions))

    print("roc_auc: ", roc_auc_score(Data["Attrition_Yes"], input_predictions))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

predictions = cross_val_predict(lr, features, target, cv=kf)

print_clf_score(pd.Series(predictions))
lr.fit(X_train, y_train)
lr_coef = pd.DataFrame(lr.coef_, columns=X_train.columns, index=["feature"]).transpose()

lr_coef.reindex([lr_coef["feature"].abs().sort_values(ascending=False).index]).head(10)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=5)

predictions1 = cross_val_predict(rf, features, target, cv=kf)

print_clf_score(pd.Series(predictions1))
rf.fit(X_train, y_train)
rf_imp = pd.DataFrame(rf.feature_importances_, index=X_train.columns, columns=["feature"])

rf_imp.reindex([rf_imp["feature"].abs().sort_values(ascending=False).index]).head(10)
features_copy = features.copy()
ordinal_cols = features_copy.columns[:24]

features_copy[ordinal_cols] = (features_copy[ordinal_cols] - features_copy[ordinal_cols].mean()) / features_copy[ordinal_cols].std()
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(features_copy, target, test_size=0.1, random_state=1)
tuned_parameters = {

 'C': [pow(2, x) for x in range(-5, 16)] , 'gamma': [pow(2, x) for x in range(-15, 4)], 'kernel': ['rbf']

                   }
from sklearn import svm

from sklearn.grid_search import GridSearchCV

svc = svm.SVC()

model = GridSearchCV(svc, tuned_parameters, cv=10, scoring="f1", n_jobs=8)

model.fit(X_train_c, y_train_c)

print("model.best_score_", model.best_score_)

print("model.best_params_", model.best_params_)
svc = svm.SVC(kernel="rbf", C=256, gamma=pow(2,-11))

predictions2 = cross_val_predict(svc, features_copy, target, cv=kf)

print_clf_score(pd.Series(predictions2))
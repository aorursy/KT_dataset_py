%matplotlib inline

import pandas as pd; pd.options.display.float_format = '{:,.3f}'.format

import numpy as np

import matplotlib.pyplot as plt; plt.rcParams['figure.figsize'] = (8,6)

import seaborn as sns; sns.set_style("dark")
data = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

columns_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'DailyRate', 

                   'HourlyRate', 'MonthlyRate', "EmployeeNumber"]



data = data.drop(columns_to_drop, axis=1)

columns = ["Attrition"] + [col for col in data.columns if col != "Attrition"]

data = data[columns].replace({"Yes": 1, "No": 0})

data.info()
sns.jointplot(x="TotalWorkingYears", y="YearsAtCompany", data=data).plot_joint(sns.kdeplot)

plt.title("Figure 1.1: Employees seniority relative to their working years", y=-.2)

plt.subplots_adjust(bottom=.2)

data[["Age", "YearsAtCompany", "TotalWorkingYears"]].describe()
fig = plt.figure(figsize=(15, 6))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122, sharey=ax1)



for i, (ax, label) in enumerate(zip([ax1, ax2], ["EducationField", "Education"])):

    vc = data[label].value_counts()

    if i == 1: vc = vc.sort_index()

    sns.barplot(x=vc.index, y=vc.values, palette="Blues_d", ax=ax)

    ax.set_xlabel("")

    [(label.set_fontsize(9), label.set_rotation(45)) for label in ax.get_xticklabels()]



study_fields = ["Biology", "Medical", "Marketing", "Technical", "Other", "H.R."]; ax1.set_xticklabels(study_fields)

diplomas = ["No degree", "College", "Bachelor", "Master", "Ph.D"]; ax2.set_xticklabels(diplomas)

ax2.yaxis.set_visible(False)

plt.suptitle("Figure 1.2: Educational Backgrounds of the Employees", size=14)

plt.subplots_adjust(bottom=.15);
dept_field = (data.pivot_table(data, index="Department", columns="EducationField", aggfunc='size'))

plt.figure(figsize=(12,6))

sns.heatmap(dept_field, annot=True, cmap="Blues", fmt=".0f")

ax = plt.gca()

study_fields = ["H.R.", "Biology", "Marketing", "Medical", "Other", "Technical"]; ax.set_xticklabels(study_fields)

ax.set_xlabel(""); ax.set_ylabel("")

plt.subplots_adjust(left=.2, bottom=.2)

plt.title("Figure 1.3: Repartition of Employees in Departments", y=-.2);
performance = data.pivot_table(index=['EducationField', 'JobRole'], 

                               columns='PerformanceRating', 

                               values='Attrition', aggfunc='count')



performance['PercentageTopPerformers'] = performance[4] / (performance[3] + performance[4])

performance.sort_values('PercentageTopPerformers', ascending=False).drop([3,4], axis=1).head(10)
def plot_gender_repartition(data, **kwargs):

    caption = kwargs["caption"]; del kwargs["caption"]

    gender_repartition = data.groupby([kwargs["index"], kwargs["columns"]], as_index=False).count().pivot(**kwargs)

    gender_repartition["Total"] = gender_repartition["Male"] + gender_repartition["Female"]

    gender_repartition["Female"] = gender_repartition["Female"] / gender_repartition["Total"] * 100

    gender_repartition["Male"] = gender_repartition["Male"] / gender_repartition["Total"] * 100

    gender_repartition[["Female", "Male"]].plot(kind="barh", stacked=True, legend=False)

    ax = plt.gca()

    ax.axvline(50, color="k", zorder=0)

    format_chart(ax, True)

    label_barh_chart(ax)

    plt.title("{}: Gender Repartition by {}".format(caption, kwargs["index"]), y=-.2)

    plt.legend(bbox_to_anchor=(.7, 1.1), ncol=2)

    plt.subplots_adjust(left=.16, bottom=.2)

    

def format_chart(ax, multiline_labels=False, ticklabel_size=10):

    [spine.set_visible(False) for spine in ax.spines.values()]

    #[tl.set_visible(False) for tl in ax.get_xticklabels()]

    ax.yaxis.set_label_text("")

    [tl.set(fontsize=ticklabel_size) for tl in ax.get_yticklabels()]

    if multiline_labels:

        ylabels = ax.get_yticklabels()

        new_labels = [label.get_text()[::-1].replace(" ", "\n", 1)[::-1] for label in ylabels]

        ax.set_yticklabels(new_labels)

        

def label_barh_chart(ax):

    text_settings = dict(fontsize=9, fontweight='bold', color="w")

    rects = ax.patches

    for i, rect in enumerate(rects):

        width = rect.get_width()

        x_pos = width / 2 if i in range(len(rects) // 2) else 100 - width / 2

        #color = "pink" if i in range(len(rects) // 2) else "#2C6388"

        #rect.set_facecolor(color)

        label = "{:.1f}%".format(rect.get_width())

        ax.text(x_pos, rect.get_y() + rect.get_height()/2, label, ha='center', va='center', **text_settings)

        

def label_barchart(ax):

    text_settings = dict(fontsize=9, fontweight='bold', color="w")

    rects = ax.patches

    for i, rect in enumerate(rects):

        x_pos = rect.get_x() + rect.get_width() / 2

        label = "{:.1%}".format(rect.get_height())

        ax.text(x_pos, .05, label, ha='center', va='center', **text_settings)
plot_gender_repartition(data, index="Department", columns="Gender", values="Attrition", caption="Figure 1.4")
plot_gender_repartition(data, index="JobLevel", columns="Gender", values="Attrition", caption="Figure 1.5")

plt.gca().set_ylabel("JobLevel");
plt.figure()

sns.barplot(x="JobLevel", y="MonthlyIncome", hue="Gender", data=data, ci=99)

plt.title("Figure 1.6: Income gap between men and women", y=-.2)

plt.subplots_adjust(bottom=.2);
attrition_education = data.pivot_table(index="Department", values="Attrition", columns="EducationField", aggfunc="mean")

plt.figure(figsize=(12,6))

sns.heatmap(attrition_education, annot=True, cmap="Blues")

ax1, ax2 = plt.gcf().get_axes()

ax1.set_xlabel(""); ax1.set_ylabel("")

study_fields = ["H.R.", "Biology", "Marketing", "Medical", "Other", "Technical"]; ax1.set_xticklabels(study_fields)

plt.title("Figure 1.7: Turnover By Deparment and Education", y=-.2)

[el.set_text("{:.1%}".format(float(el.get_text()))) for el in ax1.texts]

ax2.set_yticklabels(["{:.0%}".format(float(tl.get_text())) for tl in ax2.get_yticklabels()]);
data = data.replace({"Male": 1, "Female": 0, "Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2})

X, y = pd.get_dummies(data.iloc[:, 1:].copy()), data.iloc[:, 0].copy()
df = X.copy()

df["Attrition"] = y

df = df[["Attrition"] + list(X.columns)]

corr = df.corr()

fig, ax = plt.subplots(figsize=(12,10))

sns.heatmap(corr, cmap="RdYlBu_r", vmin=-1, vmax=1, ax=ax)

plt.title('Figure 1.10: How are the variables correlated?', size= 14);
from sklearn.model_selection import train_test_split

# Train / Test split (size of training set: 75 %)

X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#X_train["MonthlyIncome"] = scaler.fit_transform(X_train[["MonthlyIncome"]])[0]

#X_test.loc[:, "MonthlyIncome"] = scaler.transform(X_test[["MonthlyIncome"]])

#X_train["MonthlyIncome"]

X_train = X_train.copy(); X_test = X_test.copy()

X_train["MonthlyIncome"] = scaler.fit_transform(X_train[["MonthlyIncome"]])[:,0]

X_test["MonthlyIncome"] = scaler.transform(X_test[["MonthlyIncome"]])[:, 0]
from sklearn.feature_selection import SelectKBest



# Confidence Threshold

alpha = .1



np.random.seed(42)  # To make sure our results are reproducible

anova_filter = SelectKBest()

anova_filter.fit(X_train, y_train)





anova_scores = pd.DataFrame(index=X.columns)



anova_scores["Fisher"] = anova_filter.scores_

anova_scores["p-value"] = anova_filter.pvalues_

anova_scores = anova_scores.sort_values("Fisher", ascending=False)

selected_features = list(anova_scores.loc[anova_scores["p-value"] < 1 - alpha, :].index)

if len(selected_features) == X.shape[1]:

    print("No discarded feature")

X = X[selected_features]

anova_scores.style.apply(lambda f: ["color: red"] * 2 if f["p-value"] > 1-alpha else ["color: black"]*2, axis=1)
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

from sklearn.metrics import precision_recall_fscore_support

from sklearn.dummy import DummyClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict



scoring = "roc_auc"



models = [

    

    ("Dummy", DummyClassifier(strategy="most_frequent")),

    ("SVM", SVC()), 

    ("LR", LogisticRegression()),

    ("NB", GaussianNB()),

    ("kNN", KNeighborsClassifier()),

    ("DT", DecisionTreeClassifier()), 

    ("RF", RandomForestClassifier())



]



results = OrderedDict()

for name, model in models:

    kfold = KFold(n_splits=3)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results[name] = cv_results

    

results = pd.DataFrame(results)

plt.figure(figsize=(10, 6))

plt.bar(range(results.shape[1]), results.mean(), yerr=results.std())

plt.gca().set_xticklabels([""] + list(results.columns) + [""])

plt.title("Figure 2.1: Comparison of AUC score for multiple classifiers", y=-.2)

plt.subplots_adjust(bottom=.2)

label_barchart(plt.gca())

n_features = X_train.shape[1]



classifiers = [

    ("LogisticRegression", LogisticRegression()), 

    ("RandomForest", RandomForestClassifier(n_estimators=50)),

    ("GaussianNB", GaussianNB())

]



all_params = {

    "LogisticRegression": {"penalty": ["l1", "l2"], "C": np.logspace(-3, 3, 7), "class_weight":["balanced", None]}, 

    "RandomForest": {

        "max_features": range(5, n_features, (n_features - 5) // 3), 

        "max_depth": range(3, 6, 2),

        "min_samples_split": range(5, 101, 25)

    },

    "GaussianNB": {"priors": [None, [.161, .839]]}

}

results = pd.DataFrame(index=[item[0] for item in classifiers], 

                       columns=["name", "params", "accuracy", "auc_score_tr", "auc_score_te", 

                                "precision", "recall", "fscore", "support", "TP", "FP", "FN", "TN"])





best_models, scores = [], []

for i, ((name, clf)) in enumerate(classifiers):

    params = all_params[name]

    gs = GridSearchCV(clf, params).fit(X_train, y_train)

    best_models.append(gs.best_estimator_)

    y_pred = gs.predict(X_test)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)

    auc_score_te = roc_auc_score(y_test, y_pred)

    auc_score_tr = gs.best_score_

    accuracy = (y_pred == y_test).mean()

    params = gs.best_params_

    [[TP, FN], [FP, TN]] = confusion_matrix(y_test, y_pred)

    results.loc[name, :] = (name, params, accuracy, auc_score_tr, auc_score_te, precision, 

                            recall, fscore, support, TP, FP, FN, TN)

    

    scores.append(roc_auc_score(y_test, y_pred))

    gs_results = pd.DataFrame(gs.cv_results_).drop("params", axis=1).sort_values("rank_test_score")

    print("\n{}:\n".format(name))

    print("\tAccuracy: {:.2%}".format((y_pred == y_test).mean()))

    print("\tAUC Score (Train set): {:.2%}".format(gs.best_score_))

    print("\tAUC Score (Test set): {:.2%}\n".format(scores[-1]))

    print(classification_report(y_test, y_pred))

    print(best_models[-1], "\n")

    if i + 1 < len(classifiers): print("#" * 90)

    

#results

lr_scores = best_models[0].predict_proba(X_test)[:, 1]

rf_scores = best_models[1].predict_proba(X_test)[:, 1]

lr_fpr, lr_tpr, _ = roc_curve(y_test.ravel(), lr_scores.ravel())

rf_fpr, rf_tpr, _ = roc_curve(y_test.ravel(), rf_scores.ravel())

plt.plot(lr_fpr, lr_tpr, 'b', label='LogisticRegression (AUC={:.2%})'.format(scores[0]))

plt.plot(rf_fpr, rf_tpr, 'g', label='RandomForest (AUC={:.2%})'.format(scores[1]))

plt.title('Figure 2.2: Receiver Operating Characteristic')

plt.plot([0,1],[0,1],'r--', label="Random predictions")

plt.legend(loc=4)

plt.ylabel('True Positive Rate'); plt.xlabel('False Positive Rate');
clf = best_models[0]

coef = pd.DataFrame(index=X_train.columns)

coef["Coefficients"] = clf.coef_[0]

coef.sort_values("Coefficients", ascending=False)
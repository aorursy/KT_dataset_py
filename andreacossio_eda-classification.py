import os

# Pandas
import pandas as pd
pd.set_option('display.max_colwidth', None)

# Plotly
import plotly.graph_objects as go

# Markdown print
from IPython.display import Markdown, display

# Sklearn split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
    
# Functions
def printmd(string):
    display(Markdown(string))

def mycatplot(data, name):
    fig = go.Figure()
    fig.add_traces([go.Bar(x=data.index, y=data.total, name="Total", visible=False),
                    go.Bar(x=data.index, y=data["<=50K"], name="< $50K"),
                    go.Bar(x=data.index, y=data[">50K"], name="> $50K"),
                    go.Bar(x=data.index, y=data.less_ratio, name="< $50K", visible=False),
                    go.Bar(x=data.index, y=1 - data.less_ratio, name="> $50K", visible=False)])
    fig.update_layout(title=name, xaxis_title=name, yaxis_title="Count", legend_title_text="Income", showlegend=True, updatemenus=[
        dict(type="buttons", direction="right", active=1, x=1, y=1.25, buttons=list([
            dict(label="Total", method="update", args=[{"visible": [True, False, False, False, False]}, {"barmode": "group"}]),
            dict(label="Per income", method="update", args=[{"visible": [False, True, True, False, False]}, {"barmode": "group"}]),
            dict(label="Per income ratio", method="update", args=[{"visible": [False, False, False, True, True]}, {"barmode": "stack"}]),
        ]))
    ])
    return fig

def mycatpivot(data, name):
    res = data.pivot_table(index=name, columns="income", values="age", aggfunc="count", fill_value=0)
    res["total"] = res["<=50K"] + res[">50K"]
    res["less_ratio"] = (res["<=50K"] / (res["<=50K"] + res[">50K"])).round(2)
    return res.sort_values("total")
# Read the full dataset
census = pd.read_csv("/kaggle/input/adult-census-income/adult.csv", header=0, names=["age", "workclass", "final_weight", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital", "cap_loss", "work_hours", "country", "income"])

# Merge capital columns
census.capital = census.capital - census.cap_loss
census.drop(columns=["cap_loss"], inplace=True)

# Info
printmd(f"The dataset is characterized by **{census.shape[0]}** rows and **{census.shape[1]}** features.")
census.head(10)
printmd("### Duplicated rows\n\n"
        "As you can see from the following table, there are some rows that are **duplicated**. All the duplicated rows appear only twice (except for one appearing three times). "
        "Of course, the duplicated rows have been **removed** (keeping only one copy).")

if len(census[census.duplicated()]) != 0:
    duplicated = census[census.duplicated(keep=False)].sort_values(by=list(census.columns)).pivot_table(index=list(census.columns), aggfunc="size").reset_index(name="repetitions")
    census.drop_duplicates(ignore_index=True, inplace=True)
    
duplicated.head(len(duplicated))
# Income
printmd("### Income\n\n"
        "Income is our **binary target variable** that indicates whether a person makes over \$50K per year or not. The first thing to notice is that the dataset is a little bit *unbalanced*: "
        f"most of the records belong to the `<$50K` class ({census.income.value_counts()[0] * 100.0 / len(census):.2f}% -> baseline accuracy for the models).")

# Plot
fig = go.Figure(go.Histogram(x=census.income[census.income == "<=50K"], name="< $50K"))
fig.add_trace(go.Histogram(x=census.income[census.income == ">50K"], name="> $50K"))
fig.update_layout(title="Income", xaxis_title="Income", yaxis_title= "Count", legend_title_text="Income", showlegend=True)
fig.show()
# Age
printmd("### Age\n\n"
        "Age is a discrete *numerical* feature that indicates the age of the individuals. The boxplot shows that:\n"
        " - most of the individuals are less than 50 years old\n"
        " - older individuals tend to make more money")

# Plot
fig = go.Figure()
fig.add_traces([go.Box(x=census.age, name="Total", visible=False),
                go.Box(x=census.age[census.income == "<=50K"], name="< $50K"),
                go.Box(x=census.age[census.income == ">50K"], name="> $50K")])
fig.update_layout(title="Age", xaxis_title="Age", yaxis_title= "Income", showlegend=False, updatemenus=[
    dict(type="buttons", direction="right", active=1, x=1, y=1.25, buttons=list([
        dict(label="Total", method="update", args=[{"visible": [True, False, False]}]),
        dict(label="Per income", method="update", args=[{"visible": [False, True, True]}]),
    ]))
])
fig.show()
# Workclass
workclass = mycatpivot(census, "workclass")

printmd("### Workclass\n\n"
        "Workclass is a *categorical* feature indicating the job sector of the individuals. The barplot shows that:\n"
        f" - most of the individuals work in the *Private* sector ({workclass.total.loc['Private'] * 100.0 / workclass.total.sum():.2f}%)\n"
        f" - for a lot of individuals the workclass is unknown *?* ({workclass.total.loc['?'] * 100.0 / workclass.total.sum():.2f}%) (addressed [here](#Workclass---Occupation))\n"
        " - the classes *Never-worked* and *Without-pay* count a very small amount of records and are all related to income `<$50K` (addressed [here](#Workclass---Occupation))")

mycatplot(workclass, "Workclass").show()
# Occupation
occupation = mycatpivot(census, "occupation")

printmd("### Occupation\n\n"
        "Occupation is a *categorical* feature indicating the specific occupation of the individual. The barplot shows that:\n"
        " - there is not a predominant occupation\n"
        f" - for a lot of individuals the occupation is unknown *?* ({occupation.total.loc['?'] * 100.0 / occupation.total.sum():.2f}%) (addressed [here](#Workclass---Occupation))")

mycatplot(occupation, "Occupation").show()
# Workclass - Occupation
printmd("### Workclass - Occupation\n\n"
        "Analysing the features workclass and occupation together, it is possible to notice that:\n"
        " - both have unknown values (*?*) with almost a 1-to-1 relationship between them\n"
        " - the workclass classes *Never-worked* and *Without-pay* are always related to an income `<$50K` (addressed [here](#Outliers))")

workclass_occupation = census.pivot_table(index="workclass", columns="occupation", values="age", aggfunc="count", fill_value=0)
workclass_occupation.head(len(workclass_occupation))
# Education - Education num
education = mycatpivot(census, "education")
education["education_num"] = census.pivot_table(index="education", values="education_num").sort_values(by="education_num").education_num
education.sort_values("education_num", inplace=True)

printmd("### Education - Education number\n\n"
        "Education is a *categorical* feature indicating the heighest education achieved by the individuals. "
        "Each education is associated with an ordinal number going from the lowest level of education to the heighest. The barplot shows that:\n"
        " - most individuals have at least an high-school degree\n"
        " - individuals with an higher level of education tend to make more money")

mycatplot(education, "Education").show()
# Marital status
marital_status = mycatpivot(census, "marital_status")

printmd("### Marital status\n\n"
        "Marital status is a *categorical* feature indicating the marital status of the individual. The barplot shows that:\n"
        " - married individuals tend to make more money")

mycatplot(marital_status, "Marital status").show()
# Relationship
relationship = mycatpivot(census, "relationship")

printmd("### Relationship\n\n"
        "Relationship is a *categorical* feature indicating the relationship status of the individual. As seen before, the barplot shows that:\n"
        " - married individuals tend to make more money")

mycatplot(relationship, "Relationship").show()
# Marital status - Relationship
printmd("### Marital status - Relationship\n\n"
        "Analysing marital status and relationship together, the most important thing to notice is that, if you are an husband or a wife, you of course are married. "
        "On the other hand, if you are unmarried, you cannot be married. Also, notice how differentiating between husband and wife is redundant with the *Sex* feature.")
marital_relationship = census.pivot_table(index="marital_status", columns="relationship", values="age", aggfunc="count", fill_value=0)
marital_relationship
# Race
race = mycatpivot(census, "race")

printmd("### Race\n\n"
        "Race is a *categorical* feature indicating the race of the individual.")

mycatplot(race, "Race").show()
# Sex
sex = mycatpivot(census, "sex")

printmd("### Sex\n\n"
        "Sex is a *categorical* feature indicating the sex of the individual. The barplot shows that:\n"
        " - male individuals tend to make more money")

mycatplot(sex, "Sex").show()
# Country
country = mycatpivot(census, "country")

printmd("### Country\n\n"
        "Country is a *categorical* feature indicating the country of the individual.")

mycatplot(country, "Country").show()
# Capital
printmd("### Capital\n\n"
        "Capital gain and capital loss are *numerical* features that indicate how much an individual has gained or lost through investing. "
        "For simplifying the data, I have reduced the two features to a single column that is the difference of the two. (There were no records with both loss and gain different than 0). "
        " The distribution plot shows that:\n"
        f" - most of the individuals do not invest ({len(census.capital[census.capital == 0])*100.0/len(census):.2f}%)\n"
        " - if you earn from investments, you tend to earn more")

# Plot
fig = go.Figure()
fig.add_traces([go.Box(x=census.capital[census.capital != 0], visible=False),
                go.Box(x=census.capital[(census.income == "<=50K") & (census.capital != 0)], name="< $50K"),
                go.Box(x=census.capital[(census.income == ">50K") & (census.capital != 0)], name="> $50K")])
fig.update_layout(title="Capital", xaxis_title="Capital gain", yaxis_title= "Income", showlegend=False, updatemenus=[
    dict(type="buttons", direction="right", active=1, x=1, y=1.25, buttons=list([
        dict(label="Total (!=0)", method="update", args=[{"visible": [True, False, False]}]),
        dict(label="Per income (!=0)", method="update", args=[{"visible": [False, True, True]}]),
    ]))
])
fig.show()
# Work hours
printmd("### Work hours\n\n"
        "Work hours is a *numerical* feature that indicates the number of work hours per week of the individuals. The distribution plot shows that:\n"
        " - most of the individuals work 40 hours per week (25% and 50% quartiles coincide on 40: at least 25% of the individauls work 40h/week)\n"
        " - individuals that work more tend to make more money")

# Plot
fig = go.Figure()
fig.add_traces([go.Box(x=census.work_hours, name="Total", boxpoints=False, visible=False),
                go.Box(x=census.work_hours[census.income == "<=50K"], name="< $50K", boxpoints=False),
                go.Box(x=census.work_hours[census.income == ">50K"], name="> $50K", boxpoints=False)])
fig.update_layout(title="Work hours", xaxis_title="Work hours", yaxis_title= "Income", showlegend=False, updatemenus=[
    dict(type="buttons", direction="right", active=1, x=1, y=1.25, buttons=list([
        dict(label="Total", method="update", args=[{"visible": [True, False, False]}]),
        dict(label="Per income", method="update", args=[{"visible": [False, True, True]}]),
    ]))
])
fig.show()
def clean(df):
    # Cap
    df.capital = df.capital - df.cap_loss
    df.drop(columns=["cap_loss"], inplace=True)

    # Duplicates
    df.drop_duplicates(ignore_index=True, inplace=True)

    # No fnlwgt and only one education
    df.drop(columns=["final_weight", "education"], inplace=True)
    df.rename(columns={"education_num": "education"}, inplace=True)

    # No never-without workclass
    df = df[~((df.workclass == "Never-worked") | (df.workclass == "Without-pay"))]
    return df

def impute(df):
    df.workclass = df.workclass.map(lambda x: "Private" if x == "?" else x)
    df.occupation = df.occupation.map(lambda x: "Other" if x == "?" else x)
    df.country = df.country.map(lambda x: "United-States" if x == "?" else x)
    return df
    
def drop(df):
    return df[(df.workclass != "?") & (df.occupation != "?") & (df.country != "?")]

def binning(df):
    df.workclass = df.workclass.map(lambda x: "Private" if x == "Private" else "Gov" if x in ["Federal-gov", "Local-gov", "State-gov"] else "Self")
    df.marital_status = df.marital_status.map(lambda x: "Single" if x in ["Widowed", "Divorced", "Never-married"] else "Married")
    df.relationship = df.relationship.map(lambda x: "Spouse" if x in ["Husband", "Wife"] else "Other" if x in ["Unmarried", "Not-in-family"] else "Relative")
    df.race = df.race.map(lambda x: "White" if x == "White" else "Other")
    df.country = df.country.map(lambda x: "US" if x == "United-States" else "Other")
    return df

def discretize(df):
    df.age = df.age // 10
    df.work_hours = df.work_hours // 10
    df.capital = df.capital.map(lambda x: -1 if x < 0 else 0 if x == 0 else 1)
    return df

# Read again the original dataset
census = pd.read_csv("/kaggle/input/adult-census-income/adult.csv", header=0, names=["age", "workclass", "final_weight", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital", "cap_loss", "work_hours", "country", "income"])
census_train, census_test = train_test_split(census, train_size=0.75, random_state=0, stratify=census.income)

# Save original splits
folder = os.path.join("/kaggle/working/original")
if not os.path.isdir(folder):
    os.mkdir(folder)
census_train.to_csv("/kaggle/working/original/train.csv", index=False)
census_test.to_csv("/kaggle/working/original/test.csv", index=False)

datasets = {
    "clean": {"operations": [clean]},
    "drop": {"operations": [clean, drop]},
    "drop_bin": {"operations": [clean, drop, binning]},
    "drop_discr": {"operations": [clean, drop, discretize]},
    "drop_bin_discr": {"operations": [clean, drop, binning, discretize]},
    "impute": {"operations": [clean, impute]},
    "impute_bin": {"operations": [clean, impute, binning]},
    "impute_discr": {"operations": [clean, impute, discretize]},
    "impute_bin_discr": {"operations": [clean, impute, binning, discretize]},
}

# Generation of datasets
for key in datasets:
    temp_train = census_train.copy()
    temp_test = census_test.copy()
    for op in datasets[key]["operations"]:
        temp_train = op(temp_train)
        temp_test = op(temp_test)
    folder = os.path.join("/kaggle/working", key)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    scaler = StandardScaler().fit(temp_train[["age", "education", "capital", "work_hours"]])
    temp_train[["age", "education", "capital", "work_hours"]] = scaler.transform(temp_train[["age", "education", "capital", "work_hours"]])
    temp_test[["age", "education", "capital", "work_hours"]] = scaler.transform(temp_test[["age", "education", "capital", "work_hours"]])
    temp_train.to_csv(os.path.join(folder, "train.csv"), index=False)
    temp_test.to_csv(os.path.join(folder, "test.csv"), index=False)
# Pandas
import pandas as pd
pd.set_option('display.max_colwidth', None)

# Numpy
import numpy as np

# Plotly
import plotly.graph_objects as go

# Scikit learn
from sklearn import set_config
set_config(display='diagram')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

# Functions
def train_and_test(datasets, classifier, param_grid, scoring='accuracy', n_jobs=-1, return_train_score=True):
    model = {}
    results = {}
    
    for key in datasets:
        model[key] = {}
        results[key] = {}
        
        # Grid search
        model[key]["model_ohe_grid"] = GridSearchCV(classifier, param_grid, scoring=scoring, n_jobs=n_jobs, return_train_score=return_train_score).fit(datasets[key]["X_train_ohe"], datasets[key]["Y_train"])
        model[key]["model_le_grid"] = GridSearchCV(classifier, param_grid, scoring=scoring, n_jobs=n_jobs, return_train_score=return_train_score).fit(datasets[key]["X_train_le"], datasets[key]["Y_train"])
        model[key]["model_ohe"] = model[key]["model_ohe_grid"].best_estimator_
        model[key]["model_le"] = model[key]["model_le_grid"].best_estimator_

        # Test
        results[key]["ohe"] = model[key]['model_ohe'].score(datasets[key]['X_test_ohe'], datasets[key]['Y_test'])
        results[key]["le"] = model[key]['model_le'].score(datasets[key]['X_test_le'], datasets[key]['Y_test'])
    
    return model, results

def plot_results(results, title):
    fig = go.Figure()
    results = pd.DataFrame(results)
    for col in results:
        fig.add_traces(go.Bar(y=results.index, x=results[col], orientation="h", name=col))
    fig.update_layout(title=title, xaxis = dict(range=[0.5, 0.9]))
    return fig 

def acc_analysis(datasets, model_dict, param_name):
    analysis = {param_name: [], 'enc': [], 'test': [], 'train': []}
    for key in datasets:
        for param in model_dict[key]["model_ohe_grid"].cv_results_["params"]:
            analysis[param_name].append(param[param_name])
            analysis['enc'].append('ohe')
            analysis['test'].append(model_dict[key]["model_ohe_grid"].cv_results_["mean_test_score"][model_dict[key]["model_ohe_grid"].cv_results_["params"].index(param)])
            analysis['train'].append(model_dict[key]["model_ohe_grid"].cv_results_["mean_train_score"][model_dict[key]["model_ohe_grid"].cv_results_["params"].index(param)])
        for param in model_dict[key]["model_le_grid"].cv_results_["params"]:
            analysis[param_name].append(param[param_name])
            analysis['enc'].append('le')
            analysis['test'].append(model_dict[key]["model_le_grid"].cv_results_["mean_test_score"][model_dict[key]["model_le_grid"].cv_results_["params"].index(param)])
            analysis['train'].append(model_dict[key]["model_le_grid"].cv_results_["mean_train_score"][model_dict[key]["model_le_grid"].cv_results_["params"].index(param)])

    analysis_train = pd.pivot_table(pd.DataFrame(analysis), index=param_name, columns="enc", values="train", aggfunc=np.mean)
    analysis_test = pd.pivot_table(pd.DataFrame(analysis), index=param_name, columns="enc", values="test", aggfunc=np.mean)

    fig = go.Figure()
    for enc in analysis_train:
        fig.add_traces(go.Scatter(x=analysis_train.index, y=analysis_train[enc], name=f"Train - {enc}"))

    for enc in analysis_test:
        fig.add_traces(go.Scatter(x=analysis_test.index, y=analysis_test[enc], name=f"Test - {enc}"))

    return fig
    
def my_print_tree(tree, feature_names):
    print(export_text(tree, feature_names=feature_names))
# Read the datasets
datasets = {}
datasets_keys = ["original", "clean", "drop", "drop_bin", "drop_discr", "drop_bin_discr", "impute", "impute_bin", "impute_discr", "impute_bin_discr"]

# Read and encode
for key in datasets_keys:
    datasets[key] = {}
    datasets[key]["X_train"] = pd.read_csv(f"/kaggle/working/{key}/train.csv")
    datasets[key]["X_test"] = pd.read_csv(f"/kaggle/working/{key}/test.csv")
    
    # Save target variable as 0 / 1 codes
    datasets[key]["Y_train"] = datasets[key]["X_train"].income.astype("category").cat.codes
    datasets[key]["Y_test"] = datasets[key]["X_test"].income.astype("category").cat.codes
    
    # One Hot Encoding
    datasets[key]["X_train_ohe"] = datasets[key]["X_train"].copy().drop(columns=["income"])
    datasets[key]["X_test_ohe"] = datasets[key]["X_test"].copy().drop(columns=["income"])
    for col in datasets[key]["X_train_ohe"].select_dtypes("object").columns:
        if len(datasets[key]["X_train_ohe"][col].unique()) == 2:
            datasets[key]["X_train_ohe"][col] = datasets[key]["X_train_ohe"][col].astype("category").cat.codes
            datasets[key]["X_test_ohe"][col] = datasets[key]["X_test_ohe"][col].astype("category").cat.codes
    datasets[key]["X_train_ohe"] = pd.get_dummies(datasets[key]["X_train_ohe"])
    datasets[key]["X_test_ohe"] = pd.get_dummies(datasets[key]["X_test_ohe"])
    
    # Label Encoding
    datasets[key]["X_train_le"] = datasets[key]["X_train"].copy().drop(columns=["income"])
    datasets[key]["X_test_le"] = datasets[key]["X_test"].copy().drop(columns=["income"])
    for col in datasets[key]["X_train_le"].select_dtypes("object").columns:
        datasets[key]["X_train_le"][col] = datasets[key]["X_train_le"][col].astype("category").cat.codes
        datasets[key]["X_test_le"][col] = datasets[key]["X_test_le"][col].astype("category").cat.codes
    
    del datasets[key]["X_train"]
    del datasets[key]["X_test"]
# Logistic Regression
logistic_regression, logistic_regression_results = train_and_test(
    datasets,
    LogisticRegression(max_iter=500),
    [{'C': [0.001, 0.01, 0.1, 1, 2, 5, 10, 100, 1000], 'solver': ['liblinear', 'lbfgs']}]
)
acc_analysis(datasets, logistic_regression, 'C').update_layout(xaxis_type="log", xaxis_title="C", yaxis_title="Balanced accuracy", title="Logistic regression accuracy against C")
# Plot
plot_results(logistic_regression_results, "Logistic regression test accuracy").show()
print("Best model:")
logistic_regression["clean"]["model_ohe"]
# Linear Discriminant Analysis
lda, lda_results = train_and_test(
    datasets,
    LinearDiscriminantAnalysis(),
    [{'solver': ['svd']}]
)
# Plot
plot_results(lda_results, "Linear discriminant analysis test accuracy").show()
print("Best model: (default params)")
lda["original"]["model_ohe"]
# KNN
knn, knn_results = train_and_test(
    datasets,
    KNeighborsClassifier(),
    [{'n_neighbors': [3, 5, 7, 9, 13, 17, 21, 25, 30, 40]}]
)
acc_analysis(datasets, knn, 'n_neighbors').update_layout(xaxis_title="K", yaxis_title="Balanced accuracy", title="K-Nearest Neighbors accuracy against K")
# Plot
plot_results(knn_results, "K-Nearest Neighbors test accuracy").show()
print("Best model:")
knn["impute"]["model_ohe"]
# Decision trees
dt, dt_results = train_and_test(
    datasets,
    DecisionTreeClassifier(),
    [{'criterion': ['gini', 'entropy'], 'max_depth': [1, 2, 3, 5, 6, 8, 12, 15, 18, 22, 26]}]
)
acc_analysis(datasets, dt, 'max_depth').update_layout(xaxis_title="max_depth", yaxis_title="Balanced accuracy", title="Decision tree accuracy against max_depth")
# Plot
plot_results(dt_results, "Decision tree test accuracy").show()
print("Best model:")
dt["impute_bin"]["model_le"]
# Random forest
rf, rf_results = train_and_test(
    datasets,
    RandomForestClassifier(max_depth=8),
    [{'n_estimators': [1, 10, 50, 100, 150], 'criterion': ['gini', 'entropy']}]
)
acc_analysis(datasets, rf, 'n_estimators').update_layout(xaxis_title="n_estimators", yaxis_title="Balanced accuracy", title="Random forest accuracy against n_estimators")
# Plot
plot_results(rf_results, "Random forest test accuracy").show()
print("Best model:")
dt["impute_bin"]["model_ohe"]
# Random forest depth
rf_depth, rf_results_depth = train_and_test(
    datasets,
    RandomForestClassifier(n_estimators=100),
    [{'max_depth': [1, 3, 6, 8, 12, 15], 'criterion': ['gini', 'entropy']}]
)
acc_analysis(datasets, rf_depth, "max_depth").update_layout(xaxis_title="max_depth", yaxis_title="Balanced accuracy", title="Random forest accuracy against max_depth")
# Plot
plot_results(rf_results_depth, "Random forest").show()
print("Best model:")
dt["clean"]["model_le"]

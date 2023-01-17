# Preparing essentials environment
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn import preprocessing
from itertools import cycle
pd.options.mode.chained_assignment = None
# Reading CSV data
df = pd.read_csv('../input/give-me-some-credit-dataset/cs-training.csv')
# Checking the data types and column names
df.dtypes
# Checking the data top 15 rows 
df.head(15)
# Checking basic stats
rs = round(df.describe(), 2)
rs
# Renaming the first column to "id"
df.rename(
    columns = {'Unnamed: 0':'id'},
    inplace = True
)
# Counting empty cells in each column
df.isnull().sum()
# Checking the issues with RevolvingUtilizationOfUnsecuredLines
df[df["RevolvingUtilizationOfUnsecuredLines"] > 2].sample(n = 30)
# There seems to be no clear source of the issue (i.e. no connection with other columns) for the wrong values
# Checking how many RevolvingUtilizationOfUnsecuredLines values are over 2
df["id"][df["RevolvingUtilizationOfUnsecuredLines"] >= 2].count()
# Checking RevolvingUtilizationOfUnsecuredLines for outliers (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
plt.figure(figsize = (4,8))
a = sns.boxplot(
    y = "RevolvingUtilizationOfUnsecuredLines", 
    x = "SeriousDlqin2yrs",
    data = df
)
a.set(
    ylim = (0, 2)
);
df.sort_values(by = ["age"]).head(5)
df.sort_values(by = ["age"], ascending = False).head(5)
df['NumberOfTime30-59DaysPastDueNotWorse'].value_counts().sort_index()
df['NumberOfTime60-89DaysPastDueNotWorse'].value_counts().sort_index()
df['NumberOfTimes90DaysLate'].value_counts().sort_index()
# Checking the data when MonthlyIncome is null 
rs = round(df[df["MonthlyIncome"].isnull()].describe(), 2)
rs
# Seems like when MonthlyIncome is null, DebtRatio is 100% wrong (i.e. way over 1, which should be it's max value by definition)...
plt.figure(figsize = (16,5))

a = sns.boxplot(
    y = "DebtRatio",
    x = pd.qcut((df[df["MonthlyIncome"].isnull()]["age"]), 15),
    data = df[df["MonthlyIncome"].isnull()]
)

a.set(
    ylim = (0, 9000)
)

plt.setp(
    a.get_xticklabels(), 
    rotation = 55
);
# ... and when MonthlyIncome is not null, DebtRatio is behaving as expected and could be considered as "correct"
plt.figure(figsize = (16,10))

a = sns.boxplot(
    y = "DebtRatio",
    x = pd.qcut((df[df["MonthlyIncome"] > 0]["age"]), 15),
    data = df[df["MonthlyIncome"] > 0]
)

a.set(
    ylim = (0, 10)
)

plt.setp(
    a.get_xticklabels(), 
    rotation = 55
);
df[df["MonthlyIncome"] > 0]['DebtRatio'].value_counts().sort_index()
df[df["MonthlyIncome"] > 0]['DebtRatio'].describe()
df[df["MonthlyIncome"] > 0]['MonthlyIncome'].quantile(.02)
df[df["MonthlyIncome"] > 0]['MonthlyIncome'].quantile(.98)
# Checking NumberOfOpenCreditLinesAndLoans for outliers (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
plt.figure(figsize = (4,8))

a = sns.boxplot(
    y = "NumberOfOpenCreditLinesAndLoans",
    x ="SeriousDlqin2yrs",  
    data = df
);
# Checking NumberRealEstateLoansOrLines for outliers (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
plt.figure(figsize = (4,8))

a = sns.boxplot(
    y = "NumberRealEstateLoansOrLines",
    x ="SeriousDlqin2yrs",  
    data = df
);
# Checking NumberOfDependents for outliers (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
plt.figure(figsize = (4,8))

a = sns.boxplot(
    y = "NumberOfDependents",
    x ="SeriousDlqin2yrs",  
    data = df
);
# Creating a clean dataset for EDA according to the Actions memo
df_clean = df[
    df['RevolvingUtilizationOfUnsecuredLines'].notnull() & 
    (df['RevolvingUtilizationOfUnsecuredLines'] <= 1.9) &
    (df['age'] > 0) &
    (df['DebtRatio'] < 50) &
    (df['MonthlyIncome'] > 0) &
    (df['MonthlyIncome'].notnull())
]

df_clean["RevolvingUtilizationOfUnsecuredLines"] = df_clean["RevolvingUtilizationOfUnsecuredLines"].clip(upper = 1.2)
df_clean["age"] = df_clean["age"].clip(upper = 95)
df_clean["NumberOfTime30-59DaysPastDueNotWorse"] = df_clean["NumberOfTime30-59DaysPastDueNotWorse"].clip(upper = 5)
df_clean["NumberOfTime60-89DaysPastDueNotWorse"] = df_clean["NumberOfTime60-89DaysPastDueNotWorse"].clip(upper = 5)
df_clean["NumberOfTimes90DaysLate"] = df_clean["NumberOfTimes90DaysLate"].clip(upper = 6)
df_clean["MonthlyIncome"] = df_clean["MonthlyIncome"].clip(upper = 20000, lower = 800)
df_clean["DebtRatio"] = df_clean["DebtRatio"].clip(upper = 1.2)
df_clean["NumberOfOpenCreditLinesAndLoans"] = df_clean["NumberOfOpenCreditLinesAndLoans"].clip(upper = 20)
df_clean["NumberRealEstateLoansOrLines"] = df_clean["NumberRealEstateLoansOrLines"].clip(upper = 5)
df_clean["NumberOfDependents"] = df_clean["NumberOfDependents"].clip(upper = 5)
df_clean["NumberOfDependents"].fillna(0, inplace = True)
# Adding a custom predictor
df_clean["Custom1"] = df_clean["NumberOfTime30-59DaysPastDueNotWorse"] + df_clean["NumberOfTime60-89DaysPastDueNotWorse"] * 1.6 + df_clean["NumberOfTimes90DaysLate"] * 2
# Checking the clean dataset 
rs = round(df_clean.describe(), 2)
rs
# Checking the distribution of the target variable SeriousDlqin2yrs
sns.countplot(
    x = "SeriousDlqin2yrs",
    data = df_clean
);
# Seems like Accuracy is not a good metric for ML models
# Plotting the pair grid to better understand connections between columns 
grid = sns.pairplot(
    df_clean[["SeriousDlqin2yrs",
              "RevolvingUtilizationOfUnsecuredLines",
              "age",
              "DebtRatio",
              "MonthlyIncome",
              "NumberOfOpenCreditLinesAndLoans",
              "NumberRealEstateLoansOrLines",
              "NumberOfDependents"
             ]].sample(n = 3000),
    hue = "SeriousDlqin2yrs",
    height = 3,
    kind = "reg",
    plot_kws = {'scatter_kws': {'alpha': 0}}
)
grid = grid.map_upper(plt.scatter)
grid = grid.map_lower(
    sns.kdeplot, 
    shade = True,
    shade_lowest = False,
    alpha = 0.6,
    n_levels = 5
);
# Checking most highly correlated variables
def highestcorrelatedpairs (df, top_num):
    correl_matrix = df.corr()
    correl_matrix *=np.tri(*correl_matrix.values.shape, k = -1).T
    correl_matrix = correl_matrix.stack()
    correl_matrix = correl_matrix.reindex(correl_matrix.abs().sort_values(ascending = False).index).reset_index()
    correl_matrix.columns = [
        "Variable 1",
        "Variable 2",
        "Correlation"
    ]
    return correl_matrix.head(top_num)

highestcorrelatedpairs(df_clean, 16)
# Preparations for ECDF plot 
def ecdf_plot(df, col, split):
    x0 = np.sort(df[(df[split] == 0) | (df[split] == -1)][col])
    x1 = np.sort(df[df[split] == 1][col])
    y0 = np.arange(1, len(x0)+1) / len(x0)
    y1 = np.arange(1, len(x1)+1) / len(x1)
    _ = plt.plot(x0, y0, marker = '.', linestyle = 'none')
    _ = plt.plot(x1, y1, marker = '.', linestyle = 'none')
    plt.margins(0.04) 
    plt.legend([split + ": 0", split + ": 1"])
    plt.xlabel(col, fontsize = 12)
    plt.grid()
    plt.show()
# 1st variable for ECDF: Age
plt.figure(figsize = (8.5,6))
ecdf_plot(df_clean, "age", "SeriousDlqin2yrs")
# 2nd pair for ECDF: RevolvingUtilizationOfUnsecuredLines
plt.figure(figsize = (8.5,6))
ecdf_plot(df_clean, "RevolvingUtilizationOfUnsecuredLines", "SeriousDlqin2yrs")
# Defining ageRange for easier visualization
ageRange = pd.interval_range(
    start = 20, 
    freq = 10, 
    end = 90
)
df_clean['ageRange'] = pd.cut(df_clean['age'], bins = ageRange)
# Exploring the connections between RevolvingUtilizationOfUnsecuredLines and age
plt.figure(figsize = (16,8))
sns.violinplot(
    y = "RevolvingUtilizationOfUnsecuredLines",
    x = "ageRange",
    data = df_clean
);
# Explorning the RevolvingUtilizationOfUnsecuredLines by age groups for both categories of the target variable
plt.figure(figsize = (16,8))
sns.boxplot(
    y = "RevolvingUtilizationOfUnsecuredLines",
    x = "ageRange",
    hue ="SeriousDlqin2yrs",  
    data = df_clean
);
# Explorning the DebtRatio by age groups for both categories of the target variable
plt.figure(figsize = (16,8))
sns.boxplot(
    y = "DebtRatio",
    x = "ageRange",
    hue ="SeriousDlqin2yrs",  
    data = df_clean
);
# Checking the RevolvingUtilizationOfUnsecuredLines distribution differences by age group and both categories of the target variable
plt.figure(figsize = (16,8))
sns.violinplot(
    y = "RevolvingUtilizationOfUnsecuredLines",
    x = "ageRange",
    hue = "SeriousDlqin2yrs",  
    data = df_clean,
    split = True,
    inner = "quart"
);
# Checking the RevolvingUtilizationOfUnsecuredLines distribution differences by NumberOfTime30-59DaysPastDueNotWorse and both categories of the target variable
plt.figure(figsize = (16,8))
sns.violinplot(
    y = "RevolvingUtilizationOfUnsecuredLines",
    x = "NumberOfTime30-59DaysPastDueNotWorse",
    hue = "SeriousDlqin2yrs",  
    data = df_clean,
    split = True,
    inner = "quart"
);
g = sns.FacetGrid(
    df_clean,
    col = "SeriousDlqin2yrs", 
    row = "ageRange", 
    height = 2.5,
    aspect = 1.6
)
g.map(sns.kdeplot, "RevolvingUtilizationOfUnsecuredLines", "DebtRatio");
plt.ylim(-0.5, 1.5);
# Defining MonthlyIncomeRanges for easier visualization
incomeRange = pd.interval_range(
    start = 0, 
    freq = 2500, 
    end = 25000
)
df_clean['MonthlyIncomeRanges'] = pd.cut(df_clean['MonthlyIncome'], bins = incomeRange)
# Explorning the NumberOfOpenCreditLinesAndLoans by income groups for both categories of the target variable
plt.figure(figsize = (16,8))
a = sns.boxplot(
    y = "DebtRatio",
    x = "MonthlyIncomeRanges",
    hue ="SeriousDlqin2yrs",  
    data = df_clean
)
plt.setp(
    a.get_xticklabels(), 
    rotation = 55
);
sns.jointplot(
    "MonthlyIncome",
    "NumberRealEstateLoansOrLines",
    data = df_clean.sample(n = 3000),
    kind = 'kde'
);
# Checking medium and strong correlations in preparation for ML 
corr = df_clean.corr()
plt.subplots(figsize = (11, 9))
sns.heatmap(
    corr[(corr >= 0.25) | (corr <= -0.25)], 
    cmap = 'viridis', 
    vmax = 1.0, 
    vmin = -1.0, 
    linewidths = 0.1,
    annot = True, 
    annot_kws = {"size": 10}, 
    square = True
);
# ML environment for Random Forest with Random Search optimization 

from numpy import arange
from numpy import argmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import itertools

# Set a random seed
seed = 87

# Define evaluation function (source: https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/Random%20Forest%20Tutorial.ipynb)
def evaluate_model(predictions, probs, train_predictions, train_probs):
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 12
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rt'); plt.ylabel('True Positive Rt'); plt.title('ROC Curves');
    
# Define confusion matrix function (source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)
def plot_confusion_matrix(cm,
                          classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Oranges):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (4, 4))
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title, size = 10)
    plt.colorbar(aspect = 3)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45, size = 10)
    plt.yticks(tick_marks, classes, size = 10)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 12,
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 10)
    plt.xlabel('Predicted label', size = 10)
# Test data preparations
# a) Dropping MonthlyIncome and id
df_ml = df_clean.drop(
    columns = [
        "MonthlyIncome",
        "id",
        "MonthlyIncomeRanges",
        "ageRange",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfDependents"
    ]
);
# b) Splitting the dataset (30%)
labels = np.array(df_ml.pop('SeriousDlqin2yrs'))
train, test, train_labels, test_labels = train_test_split(
    df_ml, 
    labels, 
    stratify = labels,
    test_size = 0.3, 
    random_state = seed
);
# c) Saving features
features = list(train.columns);
train.shape
test.shape
model = RandomForestClassifier(
    n_estimators = 230, 
    random_state = seed, 
    max_features = 'sqrt',
    n_jobs = -1,
    verbose = 1
)

# Fitting on the Train dataset, predicting
model.fit(train, train_labels)

train_rf_pred = model.predict(train)
train_rf_probs = model.predict_proba(train)[:, 1]

rf_pred = model.predict(test)
rf_probs = model.predict_proba(test)[:, 1]
# Evaluating
evaluate_model(rf_pred, rf_probs, train_rf_pred, train_rf_probs)
roc_auc_score(test_labels, rf_probs)
cm = confusion_matrix(test_labels, rf_pred)
plot_confusion_matrix(
    cm, 
    classes = ['0', '1'],
    normalize = True,
    title = 'Confusion Matrix'
);
plt.grid(None);
print(classification_report(test_labels, rf_pred))
# Checking variables importance
rf_model = pd.DataFrame({'feature': features,
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)
rf_model.head(10)
# Hyperparameters
params = {
    'n_estimators': np.linspace(100, 210).astype(int),
    'max_depth': [None] + list(np.linspace(4, 24).astype(int)),
    'max_features': ['auto', 'sqrt', None] + list(np.arange(0.2, 0.4)),
    'max_leaf_nodes': [None] + list(np.linspace(16, 48, 80).astype(int)),
    'min_samples_split': [1, 2, 3],
    'bootstrap': [True, False]
}

# Estimator
estimator = RandomForestClassifier(random_state = seed)

# Random search model
rs = RandomizedSearchCV(
    estimator,
    params,
    n_jobs = -1, 
    scoring = 'recall',
    cv = 3, 
    n_iter = 10, 
    verbose = 1,
    random_state = seed
)

# Fitting
rs.fit(train, train_labels)
# Predicting
train_rs_pred = rs.predict(train)
train_rs_probs = rs.predict_proba(train)[:, 1]

rs_pred = rs.predict(test)
rs_probs = rs.predict_proba(test)[:, 1]
# Evaluating
evaluate_model(rs_pred, rs_probs, train_rs_pred, train_rs_probs)
roc_auc_score(test_labels, rs_probs)
rs.best_params_
cm = confusion_matrix(test_labels, rs_pred)
plot_confusion_matrix(
    cm, 
    classes = ['0', '1'],
    normalize = True,
    title = 'Confusion Matrix'
);
plt.grid(None);
print(classification_report(test_labels, rs_pred))
# Checking variables importance
rs_model = pd.DataFrame({'feature': features,
                   'importance': rs.best_estimator_.feature_importances_}).\
                    sort_values('importance', ascending = False)
rs_model.head(10)
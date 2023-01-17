import os
import re
import math
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
np.warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
sns.set_palette(["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"])
INPUT_DIR = "../input"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
# Columns name in order of appearance
ID_COLUMNS = ["PassengerId"]
FEATURE_COLUMNS = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
TARGET_COLUMNS = ["Survived"]

TRAIN_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS
TEST_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS
TRAIN_DATA = pd.read_csv(TRAIN_CSV, usecols = TRAIN_COLUMNS, index_col = ID_COLUMNS)
TEST_DATA = pd.read_csv(TEST_CSV, usecols = TEST_COLUMNS, index_col = ID_COLUMNS)

ALL_DATA = pd.concat([TRAIN_DATA[FEATURE_COLUMNS], TEST_DATA[FEATURE_COLUMNS]])
TRAIN_COUNT = TRAIN_DATA.shape[0]
TEST_COUNT = TEST_DATA.shape[0]
ALL_COUNT = TRAIN_COUNT + TEST_COUNT

print("{:,d} total passengers | {:,d} passengers tagged ({:,.0%}) | {:,d} passengers untagged ({:,.0%})".format(
    ALL_COUNT, TRAIN_COUNT, TRAIN_COUNT / ALL_COUNT, TEST_COUNT, TEST_COUNT / ALL_COUNT
))
TRAIN_DATA.head()
TRAIN_DATA.dtypes
def plot_missing_values(data):
    missing_values = data.isnull().sum().to_frame("Count")
    missing_values = missing_values[missing_values["Count"] > 0]
    missing_values = missing_values.sort_values(by = "Count", ascending = False)
    
    plt.figure(figsize = (15, 10))
    d = sns.barplot(x = missing_values.index, y = missing_values["Count"])
    
    total = len(data)
    for p in d.patches:
        y = p.get_height()
        x = p.get_x()
        
        d.text(x + p.get_width() / 2, y, "{:.1%}".format(y / total), va = "bottom", ha = "center") 
    
    plt.title("Missing Values By Feature")
    plt.xlabel("Feature")
    plt.ylabel("Frequency")
    plt.show()

plot_missing_values(ALL_DATA)
ALL_DATA[ALL_DATA["Fare"].isnull()]
def impute_missing_fares(data):
    missing_fare_rows = data[data["Fare"].isnull()]
    
    for index, row in missing_fare_rows.iterrows():
        similars = data[(data["Pclass"] == row["Pclass"]) &
                        (data["SibSp"]  + data["Parch"] == row["SibSp"] + row["Parch"]) &
                        (data["Embarked"] == row["Embarked"])]
        data.loc[index, "Fare"] = similars["Fare"].mean()
    
    return data

ALL_DATA = impute_missing_fares(ALL_DATA)
AGE_MISSING_COUNT = ALL_DATA["Age"].isnull().sum()

print("{:,d} total passengers > {:,d} missing age values ({:,.0%})".format(
    ALL_COUNT, AGE_MISSING_COUNT, AGE_MISSING_COUNT / ALL_COUNT
))
def plot_age_and_sex_distribution(data, subtitle, axis):
    copy = data.copy()
    copy["All"] = ""
    
    sns.violinplot(
        hue = "Sex",
        y = "Age",
        x = "All",
        data = copy,
        scale = "width",
        inner = "quartile",
        split = True,
        ax = axis
    )
    axis.set_xlabel("")
    axis.set_title("Age Distribution ({:s})".format(subtitle))
def plot_age_imputed_dist(data):
    fig, ax = plt.subplots(3, 2, figsize = (15, 15))
    plot_age_and_sex_distribution(data, "Original", ax[0][0])
    plot_age_and_sex_distribution(data.fillna(data["Age"].mean()), "Impute With Mean", ax[0][1])
    plot_age_and_sex_distribution(data.fillna(stats.gmean(data["Age"].dropna())), "Impute With GMean", ax[1][0])
    plot_age_and_sex_distribution(data.fillna(stats.hmean(data["Age"].dropna())), "Impute With HMean", ax[1][1])
    plot_age_and_sex_distribution(data.fillna(data["Age"].mode()[0]), "Impute With Mode", ax[2][0])
    plot_age_and_sex_distribution(data.fillna(data["Age"].median()), "Impute With Median", ax[2][1])
    sns.despine(trim = True)
    plt.tight_layout()
    plt.show()

plot_age_imputed_dist(ALL_DATA)
def impute_missing_ages_with_similars(data):
    missing_age_rows = data[data["Age"].isnull()]
    
    for index, row in missing_age_rows.iterrows():
        similars = data[(data["SibSp"] == row["SibSp"]) &
                        (data["Parch"] == row["Parch"] &
                        (data["Pclass"] == row["Pclass"]))]
        data.loc[index, "Age"] = similars["Age"].mean()
        
    return data
def plot_age_imputed_dist(data):
    fig, ax = plt.subplots(1, 2, figsize = (15, 5))
    plot_age_and_sex_distribution(data, "Original", ax[0])
    
    filled = impute_missing_ages_with_similars(data.copy())
    plot_age_and_sex_distribution(filled, "Impute With Similars Mean", ax[1])
    sns.despine(trim = True)
    plt.tight_layout()
    plt.show()
    
plot_age_imputed_dist(ALL_DATA)
def impute_missing_ages_with_ml(data):
    target_cols = ["Age"]
    feature_cols = ["SibSp", "Parch", "Pclass"]
    
    missing_values = data[data["Age"].isnull()]
    present_values = data[~data["Age"].isnull()]
    model = ElasticNet()
    model.fit(pd.get_dummies(present_values[feature_cols]), present_values[target_cols])
    
    missing_values["Age"] = model.predict(missing_values[feature_cols])

    return pd.merge(missing_values, present_values, how = "outer")
def plot_age_imputed_dist(data):
    fig, ax = plt.subplots(1, 2, figsize = (15, 5))
    plot_age_and_sex_distribution(data, "Original", ax[0])
    
    filled = impute_missing_ages_with_ml(data.copy())
    plot_age_and_sex_distribution(filled, "Impute With ML (ElasticNet)", ax[1])
    sns.despine(trim = True)
    plt.tight_layout()
    plt.show()
    
plot_age_imputed_dist(ALL_DATA)
ALL_DATA = impute_missing_ages_with_ml(ALL_DATA)
ALL_DATA["Name"].head()
def get_title(name):
    return name.split(",")[1].split(".")[0].strip().replace("the", "").strip()
def create_title_feature(data):
    data["Title"] = data["Name"].apply(lambda name : get_title(name))
    
    return data

ALL_DATA = create_title_feature(ALL_DATA)
ALL_DATA["Title"].unique()
def plot_title_dist(data):
    plt.figure(figsize = (15, 10))
    sns.countplot(
        data["Title"],
        order = data["Title"].value_counts().index
    )
    plt.xticks(rotation = 45)
    plt.title("Title Distribution")
    plt.show()
    
plot_title_dist(ALL_DATA)
ALL_DATA["Title"].unique()
SOCIAL_GROUP_BY_TITLES = {
    ("Major", "Col", "Capt") : "Officer",
    ("Lady", "Don", "Jonkheer", "Countess", "Dona") : "Royal",
    ("Dr",) : "Academic",
    ("Rev",) : "Clergy",
}

SOCIAL_GROUP_BY_TITLE = {}
for titles, social_group in SOCIAL_GROUP_BY_TITLES.items():
    for title in titles:
        SOCIAL_GROUP_BY_TITLE[title] = social_group
def create_socialGroup_feature(data):
    data["SocialGroup"] = data["Title"].map(SOCIAL_GROUP_BY_TITLE)
    
    return data
    
ALL_DATA = create_socialGroup_feature(ALL_DATA)
ALL_DATA["SocialGroup"].unique()
def plot_socialGroup_dist(data):
    plt.figure(figsize = (15, 10))
    sns.countplot(
        data["SocialGroup"],
        order = data["SocialGroup"].value_counts().index
    )
    plt.xticks(rotation = 45)
    plt.title("SocialGroup Distribution")
    plt.show()
    
plot_socialGroup_dist(ALL_DATA)
ALL_DATA["Cabin"].unique()
def create_deck_feature(data):
    data["Deck"] = data["Cabin"].str[0]
    
    return data

ALL_DATA = create_deck_feature(ALL_DATA)
ALL_DATA["Deck"].unique()
def plot_deck_dist(data):
    plt.figure(figsize = (15, 10))
    sns.countplot(
        data["Deck"],
        order = data["Deck"].value_counts().index
    )
    plt.xticks(rotation = 45)
    plt.title("Deck Assigned Distribution")
    plt.show()
    
plot_deck_dist(ALL_DATA)
ALL_DATA["Cabin"].unique()
def create_cabinCount_feature(data):
    data["CabinCount"] = data["Cabin"].str.split().str.len()
    data["CabinCount"] = data["CabinCount"].fillna(0).astype(int)
    
    return data

ALL_DATA = create_cabinCount_feature(ALL_DATA)
ALL_DATA["CabinCount"].unique()
def plot_cabinCount_dist(data):
    plt.figure(figsize = (15, 10))
    sns.countplot(
        data["CabinCount"],
        order = data["CabinCount"].value_counts().index
    )
    plt.xticks(rotation = 45)
    plt.title("CabinCount Distribution")
    plt.show()
    
plot_cabinCount_dist(ALL_DATA)
ALL_DATA.columns
def create_familySize_feature(data):
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    
    return data

ALL_DATA = create_familySize_feature(ALL_DATA)
ALL_DATA["FamilySize"].unique()
def plot_familySize_dist(data):
    plt.figure(figsize = (15, 10))
    sns.countplot(
        data["FamilySize"],
        order = data["FamilySize"].value_counts().index
    )
    plt.xticks(rotation = 45)
    plt.title("FamilySize Distribution")
    plt.show()
    
plot_familySize_dist(ALL_DATA)
ALL_DATA.columns
def create_isAlone_feature(data):
    data["TravelingAlone"] = (data["SibSp"] + data["Parch"] == 0) * 1
    
    return data

ALL_DATA = create_isAlone_feature(ALL_DATA)
ALL_DATA["TravelingAlone"].describe()
ALL_DATA["Ticket"].head()
def create_ticketPrefix_feature(data):
    data["TicketPrefix"] = data["Ticket"].str.extract("(.*)\s+\d*$")
    data["TicketPrefix"] = data["TicketPrefix"].str.replace("\W", "", regex = True)
    data["TicketPrefix"] = data["TicketPrefix"].str.upper().str.strip()
    
    return data

ALL_DATA = create_ticketPrefix_feature(ALL_DATA)
ALL_DATA["TicketPrefix"].unique()
def plot_ticketPrefix_dist(data):
    plt.figure(figsize = (15, 10))
    sns.countplot(
        data["TicketPrefix"], 
        order = data["TicketPrefix"].value_counts().index
    )
    plt.xticks(rotation = 45)
    plt.title("TicketPrefix Distribution")
    plt.show()
    
plot_ticketPrefix_dist(ALL_DATA)
ALL_DATA["Sex"].unique()
def convert_sex_feature(data):
    data["IsMale"] = (data["Sex"] == "male") * 1
    
    return data

ALL_DATA = convert_sex_feature(ALL_DATA)
ALL_DATA["Embarked"].unique()
PORTS_MAP = {
    "S" : "Southampton",
    "C" : "Cherbourg",
    "Q" : "Queenstown"
}
def convert_embarked_feature(data):
    data["Port"] = data["Embarked"].map(PORTS_MAP)
    
    return data

ALL_DATA = convert_embarked_feature(ALL_DATA)
ALL_DATA.head()
ALL_DATA = ALL_DATA.drop(["Name", "Sex", "Ticket", "Cabin", "Embarked"], axis = 1)
TRAIN_DATA = ALL_DATA[:TRAIN_COUNT].join(TRAIN_DATA[TARGET_COLUMNS], how = "inner")
TEST_DATA = ALL_DATA[TRAIN_COUNT:]

FEATURE_COLUMNS = TRAIN_DATA.columns.difference(TARGET_COLUMNS)
TRAIN_DATA.head()
FEATURE_COUNT = pd.get_dummies(ALL_DATA).shape[1]

print("{:,d} passengers tagged X {:,d} features = {:,d} entries".format(
    TRAIN_COUNT, FEATURE_COUNT, TRAIN_COUNT * FEATURE_COUNT
))
# To be able to create more readable plots...
TRAIN_DATA["AgeRange"] = pd.cut(TRAIN_DATA["Age"], range(0, 90, 10))
TRAIN_DATA["FareRange"] = pd.cut(TRAIN_DATA["Fare"], range(0, 550, 25))
def plot_target_dist(data):
    plt.figure(figsize = (14, 7))
    sns.countplot(data["Survived"])
    plt.title("Survavibility Distribution")
    
plot_target_dist(TRAIN_DATA)
def plot_univariate_dist(data, feature_name, target_name):
    fig = sns.factorplot(
        data = data,
        x = feature_name,
        y = target_name,
        kind = "bar",
        height = 7,
        aspect = 2
    )
    sns.despine(trim = True)
    fig.set_xticklabels(rotation = 45)
    plt.title("Survavibility by {:s}".format(feature_name))
plot_univariate_dist(TRAIN_DATA, "Pclass", "Survived")
plot_univariate_dist(TRAIN_DATA, "AgeRange", "Survived")
plot_univariate_dist(TRAIN_DATA, "SibSp", "Survived")
plot_univariate_dist(TRAIN_DATA, "Parch", "Survived")
plot_univariate_dist(TRAIN_DATA, "Title", "Survived")
plot_univariate_dist(TRAIN_DATA, "SocialGroup", "Survived")
plot_univariate_dist(TRAIN_DATA, "CabinCount", "Survived")
plot_univariate_dist(TRAIN_DATA, "Deck", "Survived")
plot_univariate_dist(TRAIN_DATA, "TravelingAlone", "Survived")
plot_univariate_dist(TRAIN_DATA, "FamilySize", "Survived")
plot_univariate_dist(TRAIN_DATA, "TicketPrefix", "Survived")
plot_univariate_dist(TRAIN_DATA, "Port", "Survived")
plot_univariate_dist(TRAIN_DATA, "IsMale", "Survived")
plot_univariate_dist(TRAIN_DATA, "FareRange", "Survived")
def plot_multivariate_dist(data, x, y, hue):
    fig = sns.factorplot(
        data = data,
        x = x,
        y = y,
        hue = hue,
        kind = "bar",
        legend_out = False,
        aspect = 2,
        height = 7,
    )
    sns.despine(trim = True)
    fig.set_xticklabels(rotation = 45)
    plt.title("Survavibility by {:s} and {:s}".format(x, hue))
    plt.show()
plot_multivariate_dist(TRAIN_DATA, "AgeRange", "Survived", "IsMale")
plot_multivariate_dist(TRAIN_DATA, "AgeRange", "Survived", "SocialGroup")
plot_multivariate_dist(TRAIN_DATA, "Deck", "Survived", "CabinCount")
plot_multivariate_dist(TRAIN_DATA, "Deck", "Survived", "Pclass")
plot_multivariate_dist(TRAIN_DATA, "Pclass", "Survived", "IsMale")
plot_multivariate_dist(TRAIN_DATA, "Port", "Survived", "Pclass")
plot_multivariate_dist(TRAIN_DATA, "FareRange", "Survived", "Port")
plot_multivariate_dist(TRAIN_DATA, "AgeRange", "Survived", "TravelingAlone")
plot_multivariate_dist(TRAIN_DATA, "TravelingAlone", "Survived", "IsMale")
plot_multivariate_dist(TRAIN_DATA, "TicketPrefix", "Survived", "Port")
plot_multivariate_dist(TRAIN_DATA, "TicketPrefix", "Survived", "Pclass")
def plot_correlation_heatmap(data):
    corr = pd.get_dummies(data).corr()
    
    plt.figure(figsize = (25, 20))

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype = np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap = True)

    sns.heatmap(corr, cmap = cmap, mask = mask, square = True, center = 0, robust = True, linewidths = .2)
    plt.show()
plot_correlation_heatmap(TRAIN_DATA[FEATURE_COLUMNS])
X_TRAIN = pd.get_dummies(TRAIN_DATA[FEATURE_COLUMNS])
X_TEST = pd.get_dummies(TEST_DATA[FEATURE_COLUMNS])
Y_TRAIN = TRAIN_DATA[TARGET_COLUMNS]
RANDOM_SEED = 123
K_FOLDS = StratifiedKFold(n_splits = 10, random_state = RANDOM_SEED)
SCALERS = [
    ("Standard", StandardScaler()),
    ("Robust", RobustScaler()),
    ("MinMax", MinMaxScaler()),
    ("Normalizer", Normalizer())
]
def get_scaled_models(model_tuples, scaler_tuple):
    scaler_name, scaler = scaler_tuple
    
    scaled_tuples = []
    for model_name, model in model_tuples:
        scaled_tuples.append((model_name, Pipeline([(scaler_name, scaler), (model_name, model)])))
            
    return scaled_tuples
def get_scaled_models_results(model_tuples, scaler_name, x, y, kfolds):
    results = pd.DataFrame([])
    
    for model_name, model in model_tuples:
        results[model_name] = cross_val_score(model, x, y, cv = kfolds)
        
    results = results.melt(var_name = "Model", value_name = "Precision")
    results["Scaler"] = scaler_name

    return results
def get_all_models_combinaisons_results(model_tuples, scaler_tuples, x, y, kfolds):
    results = get_scaled_models_results(model_tuples, "None", x, y, kfolds)

    for scaler_tuple in scaler_tuples:
        scaled_models = get_scaled_models(model_tuples, scaler_tuple)
        results = results.append(get_scaled_models_results(scaled_models, scaler_tuple[0], x, y, kfolds))
        
    return results
def plot_models_results(results):  
    plt.figure(figsize = (20, 10))
    plt.title('Algorithm Comparison')
    sns.boxplot(data = results, x = "Model", y = "Precision", hue = "Scaler")
    sns.despine(trim = True)
    plt.show()
BASE_MODELS = [
    ("SVM", SVC()), 
    ("CART", DecisionTreeClassifier()),
    ("KNN", KNeighborsClassifier()),
    ("LR", LogisticRegression()),
    ("LDA", LinearDiscriminantAnalysis()),
    ("MLP", MLPClassifier()),
    ("GPC", GaussianProcessClassifier()),
]
RESULTS = get_all_models_combinaisons_results(BASE_MODELS, SCALERS, X_TRAIN, Y_TRAIN, K_FOLDS)
plot_models_results(RESULTS)
BASE_ENSEMBLES = [
    ("ADABOOST", AdaBoostClassifier()),
    ("GRDBOOST", GradientBoostingClassifier()),
    ("BAGGING", BaggingClassifier()),
    ("FOREST", RandomForestClassifier()),
    ("VOTING", VotingClassifier(BASE_MODELS)),
    ("EXTRATREE", ExtraTreesClassifier()),
    ("XGBOOST", XGBClassifier()),
]
RESULTS = get_all_models_combinaisons_results(BASE_ENSEMBLES, SCALERS, X_TRAIN, Y_TRAIN, K_FOLDS)
plot_models_results(RESULTS)
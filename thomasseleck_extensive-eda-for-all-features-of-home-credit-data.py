import time
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
import warnings
import seaborn as sns
color = sns.color_palette()
import pickle
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")
rcParams['figure.figsize'] = 12, 8
np.random.seed(23)
# Create a dictionary with features types to reduce memory consumption; this roughly halves the needed memory
main_features_dtypes_dict = {}

# Add np.int8 cols
for col in ["CNT_CHILDREN", "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL", 
            "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY", "HOUR_APPR_PROCESS_START", "REG_REGION_NOT_LIVE_REGION", 
            "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", 
            "LIVE_CITY_NOT_WORK_CITY", "OWN_CAR_AGE", "CNT_CREDIT_PROLONG", "MONTHS_BALANCE", "CNT_DRAWINGS_CURRENT",
            "HOUR_APPR_PROCESS_START", "NFLAG_LAST_APPL_IN_DAY"]:
    main_features_dtypes_dict[col] = np.int8

# Add np.int16 cols
for col in ["DAYS_BIRTH", "DAYS_ID_PUBLISH", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "SK_DPD", "SK_DPD_DEF", "DAYS_DECISION",
            "NUM_INSTALMENT_NUMBER"]:
    main_features_dtypes_dict[col] = np.int16

# Add np.int32 cols
for col in ["SK_ID_CURR", "DAYS_EMPLOYED", "SK_ID_BUREAU", "DAYS_CREDIT_UPDATE", "SK_ID_BUREAU", "SK_ID_PREV", 
            "AMT_CREDIT_LIMIT_ACTUAL", "SELLERPLACE_AREA"]:
    main_features_dtypes_dict[col] = np.int32

# Add np.float16 cols ; these features are integers, but as they contains NAs, they only can be casted to float
for col in ["OWN_CAR_AGE", "CNT_FAM_MEMBERS", "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE",
            "DEF_60_CNT_SOCIAL_CIRCLE", "DAYS_LAST_PHONE_CHANGE", "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", 
            "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
            "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "CNT_DRAWINGS_ATM_CURRENT", "CNT_DRAWINGS_OTHER_CURRENT", 
            "CNT_DRAWINGS_POS_CURRENT", "CNT_INSTALMENT_MATURE_CUM", "CNT_PAYMENT", "NFLAG_INSURED_ON_APPROVAL",
            "NUM_INSTALMENT_VERSION", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"]:
    main_features_dtypes_dict[col] = np.float16

# Add np.float32 cols
for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "REGION_POPULATION_RELATIVE", "DAYS_REGISTRATION",
            "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG", "COMMONAREA_AVG", 
            "ELEVATORS_AVG", "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", 
            "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG", "APARTMENTS_MODE", "BASEMENTAREA_MODE", 
            "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_MODE", "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE", 
            "FLOORSMAX_MODE", "FLOORSMIN_MODE", "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE", 
            "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE", "APARTMENTS_MEDI", "BASEMENTAREA_MEDI", 
            "YEARS_BEGINEXPLUATATION_MEDI", "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI", 
            "FLOORSMAX_MEDI", "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI", "LIVINGAREA_MEDI", 
            "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI", "TOTALAREA_MODE", "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT", 
            "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE", 
            "AMT_ANNUITY", "AMT_BALANCE", "AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_CURRENT", "AMT_DRAWINGS_OTHER_CURRENT", 
            "AMT_DRAWINGS_POS_CURRENT", "AMT_INST_MIN_REGULARITY", "AMT_PAYMENT_CURRENT", "AMT_PAYMENT_TOTAL_CURRENT", 
            "AMT_RECEIVABLE_PRINCIPAL", "AMT_RECIVABLE", "AMT_TOTAL_RECEIVABLE", "AMT_APPLICATION", "AMT_CREDIT", 
            "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE", "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", 
            "DAYS_LAST_DUE", "DAYS_TERMINATION", "AMT_INSTALMENT", "AMT_PAYMENT"]:
    main_features_dtypes_dict[col] = np.float32

for i in range(2, 22):
    main_features_dtypes_dict["FLAG_DOCUMENT_" + str(i)] = np.int8

print("    Loading: ../input/application_train.csv ...")
training_set_df = pd.read_csv("../input/application_train.csv", dtype = main_features_dtypes_dict)
print("    Loading: ../input/application_test.csv ...")
testing_set_df = pd.read_csv("../input/application_test.csv", dtype = main_features_dtypes_dict)
print("    Loading: ../input/bureau.csv ...")
bureau_data_df = pd.read_csv("../input/bureau.csv", dtype = main_features_dtypes_dict)
print("    Loading: ../input/bureau_balance.csv ...")
bureau_balance_data_df = pd.read_csv("../input/bureau_balance.csv", dtype = main_features_dtypes_dict)
print("    Loading: ../input/credit_card_balance.csv ...")
credit_card_balance_data_df = pd.read_csv("../input/credit_card_balance.csv", dtype = main_features_dtypes_dict)
print("    Loading: ../input/installments_payments.csv ...")
installments_payments_data_df = pd.read_csv("../input/installments_payments.csv", dtype = main_features_dtypes_dict)
print("    Loading: ../input/POS_CASH_balance.csv ...")
pos_cash_balance_data_df = pd.read_csv("../input/POS_CASH_balance.csv", dtype = main_features_dtypes_dict)
print("    Loading: ../input/previous_application.csv ...")
previous_application_data_df = pd.read_csv("../input/previous_application.csv", dtype = main_features_dtypes_dict)

# Put ID as index
print("    Put 'SK_ID_CURR' as index...")
training_set_df.index = training_set_df["SK_ID_CURR"]
training_set_df.drop("SK_ID_CURR", axis = 1, inplace = True)
testing_set_df.index = testing_set_df["SK_ID_CURR"]
testing_set_df.drop("SK_ID_CURR", axis = 1, inplace = True)
sns.countplot(x = training_set_df["TARGET"])
plt.title("Count plot of each level of the target")
print("Percentage of positive target:", round((training_set_df["TARGET"].loc[training_set_df["TARGET"] == 1].shape[0] / training_set_df["TARGET"].shape[0]) * 100, 4), "%")
print("Percentage of negative target:", round((training_set_df["TARGET"].loc[training_set_df["TARGET"] == 0].shape[0] / training_set_df["TARGET"].shape[0]) * 100, 4), "%")
nb_levels_sr = training_set_df.nunique()
binary_features_lst = nb_levels_sr.loc[nb_levels_sr == 2].index.tolist()
categorical_features_lst = list(set(training_set_df.select_dtypes(["object"]).columns.tolist()) - set(binary_features_lst))
numerical_features_lst = list(set(training_set_df.columns.tolist()) - set(categorical_features_lst) - set(binary_features_lst))
binary_features_lst = list(set(binary_features_lst) - {"TARGET"})

print("Binary features:", binary_features_lst)
print("\n")
print("Categorical features:", categorical_features_lst)
print("\n")
print("Numerical features:", numerical_features_lst)
binary_features_lst.sort()

fig, ax = plt.subplots(6, 6, sharex = False, sharey = False, figsize = (20, 20))
i = 0
j = 0
for idx in range(len(binary_features_lst)):
    if idx % 6 == 0 and idx != 0:
        j += 1
        
    i = idx % 6
    feature = binary_features_lst[idx]
    table_df = pd.crosstab(training_set_df["TARGET"], training_set_df[feature], normalize = True)
    # Normalize statistics to remove target unbalance
    table_df = table_df.div(table_df.sum(axis = 1), axis = 0)
    table_df = table_df.div(table_df.sum(axis = 0), axis = 1)
    sns.heatmap(table_df, annot = True, square = True, ax = ax[j, i], cbar = False, fmt = '.2%')
for feature in categorical_features_lst:
    fig, ax = plt.subplots(1, 2, sharex = False, sharey = False, figsize = (20, 10))
    # Plot levels distribution
    if training_set_df[feature].nunique() < 10:
        sns.countplot(x = training_set_df[feature], ax = ax[0], order = training_set_df[feature].value_counts().index.tolist())
    else:
        sns.countplot(y = training_set_df[feature], ax = ax[0], order = training_set_df[feature].value_counts().index.tolist())
    ax[0].set_title("Count plot of each level of the feature: " + feature)

    # Plot target distribution among levels
    table_df = pd.crosstab(training_set_df["TARGET"], training_set_df[feature], normalize = True)
    table_df = table_df.div(table_df.sum(axis = 0), axis = 1)
    table_df = pd.crosstab(training_set_df["TARGET"], training_set_df[feature], normalize = True)
    table_df = table_df.div(table_df.sum(axis = 0), axis = 1)
    table_df = table_df.transpose().reset_index()
    order_lst = table_df.sort_values(by = 1)[feature].tolist()
    table_df = table_df.melt(id_vars = [feature])
    if training_set_df[feature].nunique() < 10:
        ax2 = sns.barplot(x = table_df[feature], y = table_df["value"] * 100, hue = table_df["TARGET"], ax = ax[1], order = order_lst)
        for p in ax2.patches:
            height = p.get_height()
            ax2.text(p.get_x() + p.get_width() / 2., height + 1, "{:1.2f}".format(height), ha = "center")
    else:
        ax2 = sns.barplot(x = table_df["value"] * 100, y = table_df[feature], hue = table_df["TARGET"], ax = ax[1], order = order_lst)
        for p in ax2.patches:
            width = p.get_width()
            ax2.text(width + 3.1, p.get_y() + p.get_height() / 2. + 0.35, "{:1.2f}".format(width), ha = "center")

    ax[1].set_title("Target distribution among " +  feature + " levels")
    ax[1].set_ylabel("Percentage")
# generate the linkage matrix
numerical_features_df = training_set_df[numerical_features_lst + ["TARGET"]]
numerical_features_df.fillna(-1, inplace = True) # We need to impute missing values before creating the dendrogram
numerical_features_df = numerical_features_df.transpose()
Z = linkage(numerical_features_df, "ward")
plt.figure(figsize = (20, 15))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("feature")
plt.ylabel("distance")
dend = dendrogram(
    Z,
    leaf_rotation = 90.,  # rotates the x axis labels
    leaf_font_size = 8.,  # font size for the x axis labels
    labels = numerical_features_df.index.tolist()
)
plt.figure(figsize = (20, 20))
sns.heatmap(training_set_df[dend["ivl"]].corr(), annot = False, square = True)
plt.title("Correlation plot between numerical features and target")
for feature in numerical_features_lst:
    fig, ax = plt.subplots(1, 2, sharex = False, sharey = False, figsize = (20, 7))
    
    # Plot feature distribution
    plot_df = training_set_df[[feature, "TARGET"]].dropna()
    sns.distplot(plot_df[feature], kde = False, bins = 100, ax = ax[0])
    ax[0].set_title("Histogram of the feature: " + feature)

    # Plot feature against target
    sns.boxplot(x = training_set_df["TARGET"], y = training_set_df[feature], ax = ax[1])
    ax[1].set_title("Boxplot of the feature: " + feature + " wrt TARGET")
missing_values_sr = training_set_df.isnull().sum()
missing_values_df = missing_values_sr.loc[missing_values_sr > 0].sort_values(ascending = False).reset_index()
missing_values_df.columns = ["Feature", "Number of missing values"]
missing_values_df["Percentage of missing values"] = (missing_values_df["Number of missing values"] / training_set_df.shape[0]) * 100

sns.barplot(x = missing_values_df["Feature"], y = missing_values_df["Percentage of missing values"])
plt.xticks(rotation = 90)
plt.title("Percentage of missing values in the application data")
is_missing_df = training_set_df.isnull().astype(np.int8)
is_missing_df = is_missing_df[missing_values_df["Feature"]]

fig, ax = plt.subplots(9, 8, sharex = False, sharey = False, figsize = (26, 30))
i = 0
j = 0
for idx in range(len(missing_values_df["Feature"])):
    if idx % 8 == 0 and idx != 0:
        j += 1
        
    i = idx % 8
    feature = is_missing_df.columns.tolist()[idx]
    table_df = pd.crosstab(training_set_df["TARGET"], is_missing_df[feature], normalize = True)
    # Normalize statistics to remove target unbalance
    table_df = table_df.div(table_df.sum(axis = 1), axis = 0)
    table_df = table_df.div(table_df.sum(axis = 0), axis = 1)
    sns.heatmap(table_df, annot = True, square = True, ax = ax[j, i], cbar = False, fmt = '.2%')
merged_df = bureau_data_df.merge(bureau_balance_data_df, how = "left", on = "SK_ID_BUREAU")
merged_df.info()
categorical_features_lst = merged_df.select_dtypes(["object"]).columns.tolist()

for feature in categorical_features_lst:
    fig, ax = plt.subplots(1, 1, sharex = False, sharey = False, figsize = (20, 10))
    # Plot levels distribution
    if merged_df[feature].nunique() < 10:
        sns.countplot(x = merged_df[feature], ax = ax, order = merged_df[feature].value_counts().index.tolist())
    else:
        sns.countplot(y = merged_df[feature], ax = ax, order = merged_df[feature].value_counts().index.tolist())
    ax.set_title("Count plot of each level of the feature: " + feature)
numerical_features_lst = list(set(merged_df.columns.tolist()) - set(categorical_features_lst))

# generate the linkage matrix
numerical_features_df = merged_df[numerical_features_lst]
numerical_features_df.fillna(-1, inplace = True) # We need to impute missing values before creating the dendrogram
numerical_features_df = numerical_features_df.transpose()
Z = linkage(numerical_features_df, "ward")
plt.figure(figsize = (20, 15))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("feature")
plt.ylabel("distance")
dend = dendrogram(
    Z,
    leaf_rotation = 90.,  # rotates the x axis labels
    leaf_font_size = 8.,  # font size for the x axis labels
    labels = numerical_features_df.index.tolist()
)
plt.figure(figsize = (20, 20))
sns.heatmap(merged_df[dend["ivl"]].corr(), annot = True, square = True)
plt.title("Correlation plot between numerical features")
for feature in numerical_features_lst:
    fig, ax = plt.subplots(1, 1, sharex = False, sharey = False, figsize = (20, 7))
    
    # Plot feature distribution
    sns.distplot(merged_df[feature].dropna(), kde = False, bins = 100, ax = ax)
    ax.set_title("Histogram of the feature: " + feature)
missing_values_sr = merged_df.isnull().sum()
missing_values_df = missing_values_sr.loc[missing_values_sr > 0].sort_values(ascending = False).reset_index()
missing_values_df.columns = ["Feature", "Number of missing values"]
missing_values_df["Percentage of missing values"] = (missing_values_df["Number of missing values"] / merged_df.shape[0]) * 100

sns.barplot(x = missing_values_df["Feature"], y = missing_values_df["Percentage of missing values"])
plt.xticks(rotation = 90)
plt.title("Percentage of missing values in the bureau and bureau balance data")
categorical_features_lst = credit_card_balance_data_df.select_dtypes(["object"]).columns.tolist()

for feature in categorical_features_lst:
    fig, ax = plt.subplots(1, 1, sharex = False, sharey = False, figsize = (20, 10))
    # Plot levels distribution
    if credit_card_balance_data_df[feature].nunique() < 10:
        sns.countplot(x = credit_card_balance_data_df[feature], ax = ax, order = credit_card_balance_data_df[feature].value_counts().index.tolist())
    else:
        sns.countplot(y = credit_card_balance_data_df[feature], ax = ax, order = credit_card_balance_data_df[feature].value_counts().index.tolist())
    ax.set_title("Count plot of each level of the feature: " + feature)
numerical_features_lst = list(set(credit_card_balance_data_df.columns.tolist()) - set(categorical_features_lst))

# generate the linkage matrix
numerical_features_df = credit_card_balance_data_df[numerical_features_lst]
numerical_features_df.fillna(-1, inplace = True) # We need to impute missing values before creating the dendrogram
numerical_features_df = numerical_features_df.transpose()
Z = linkage(numerical_features_df, "ward")
plt.figure(figsize = (20, 15))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("feature")
plt.ylabel("distance")
dend = dendrogram(
    Z,
    leaf_rotation = 90.,  # rotates the x axis labels
    leaf_font_size = 8.,  # font size for the x axis labels
    labels = numerical_features_df.index.tolist()
)
plt.figure(figsize = (20, 20))
sns.heatmap(credit_card_balance_data_df[dend["ivl"]].corr(), annot = True, square = True)
plt.title("Correlation plot between numerical features")
for feature in numerical_features_lst:
    fig, ax = plt.subplots(1, 1, sharex = False, sharey = False, figsize = (20, 7))
    
    # Plot feature distribution
    sns.distplot(credit_card_balance_data_df[feature].dropna(), kde = False, bins = 100, ax = ax)
    ax.set_title("Histogram of the feature: " + feature)
missing_values_sr = credit_card_balance_data_df.isnull().sum()
missing_values_df = missing_values_sr.loc[missing_values_sr > 0].sort_values(ascending = False).reset_index()
missing_values_df.columns = ["Feature", "Number of missing values"]
missing_values_df["Percentage of missing values"] = (missing_values_df["Number of missing values"] / credit_card_balance_data_df.shape[0]) * 100

sns.barplot(x = missing_values_df["Feature"], y = missing_values_df["Percentage of missing values"])
plt.xticks(rotation = 90)
plt.title("Percentage of missing values in the credit card balance data")
numerical_features_lst = installments_payments_data_df.columns.tolist()

# generate the linkage matrix
numerical_features_df = installments_payments_data_df[numerical_features_lst]
numerical_features_df.fillna(-1, inplace = True) # We need to impute missing values before creating the dendrogram
numerical_features_df = numerical_features_df.transpose()
Z = linkage(numerical_features_df, "ward")
plt.figure(figsize = (20, 15))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("feature")
plt.ylabel("distance")
dend = dendrogram(
    Z,
    leaf_rotation = 90.,  # rotates the x axis labels
    leaf_font_size = 8.,  # font size for the x axis labels
    labels = numerical_features_df.index.tolist()
)
plt.figure(figsize = (15, 15))
sns.heatmap(installments_payments_data_df[dend["ivl"]].corr(), annot = True, square = True)
plt.title("Correlation plot between numerical features")
for feature in numerical_features_lst:
    fig, ax = plt.subplots(1, 1, sharex = False, sharey = False, figsize = (20, 7))
    
    # Plot feature distribution
    sns.distplot(installments_payments_data_df[feature].dropna(), kde = False, bins = 100, ax = ax)
    ax.set_title("Histogram of the feature: " + feature)
missing_values_sr = installments_payments_data_df.isnull().sum()
missing_values_df = missing_values_sr.loc[missing_values_sr > 0].sort_values(ascending = False).reset_index()
missing_values_df.columns = ["Feature", "Number of missing values"]
missing_values_df["Percentage of missing values"] = (missing_values_df["Number of missing values"] / installments_payments_data_df.shape[0]) * 100

sns.barplot(x = missing_values_df["Feature"], y = missing_values_df["Percentage of missing values"])
plt.xticks(rotation = 90)
plt.title("Percentage of missing values in the installments payments data")
categorical_features_lst = pos_cash_balance_data_df.select_dtypes(["object"]).columns.tolist()

for feature in categorical_features_lst:
    fig, ax = plt.subplots(1, 1, sharex = False, sharey = False, figsize = (20, 10))
    # Plot levels distribution
    if pos_cash_balance_data_df[feature].nunique() < 10:
        sns.countplot(x = pos_cash_balance_data_df[feature], ax = ax, order = pos_cash_balance_data_df[feature].value_counts().index.tolist())
    else:
        sns.countplot(y = pos_cash_balance_data_df[feature], ax = ax, order = pos_cash_balance_data_df[feature].value_counts().index.tolist())
    ax.set_title("Count plot of each level of the feature: " + feature)
numerical_features_lst = list(set(pos_cash_balance_data_df.columns.tolist()) - set(categorical_features_lst))

# generate the linkage matrix
numerical_features_df = pos_cash_balance_data_df[numerical_features_lst]
numerical_features_df.fillna(-1, inplace = True) # We need to impute missing values before creating the dendrogram
numerical_features_df = numerical_features_df.transpose()
Z = linkage(numerical_features_df, "ward")
plt.figure(figsize = (20, 15))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("feature")
plt.ylabel("distance")
dend = dendrogram(
    Z,
    leaf_rotation = 90.,  # rotates the x axis labels
    leaf_font_size = 8.,  # font size for the x axis labels
    labels = numerical_features_df.index.tolist()
)
plt.figure(figsize = (15, 15))
sns.heatmap(pos_cash_balance_data_df[dend["ivl"]].corr(), annot = True, square = True)
plt.title("Correlation plot between numerical features")
for feature in numerical_features_lst:
    fig, ax = plt.subplots(1, 1, sharex = False, sharey = False, figsize = (20, 7))
    
    # Plot feature distribution
    sns.distplot(pos_cash_balance_data_df[feature].dropna(), kde = False, bins = 100, ax = ax)
    ax.set_title("Histogram of the feature: " + feature)
missing_values_sr = pos_cash_balance_data_df.isnull().sum()
missing_values_df = missing_values_sr.loc[missing_values_sr > 0].sort_values(ascending = False).reset_index()
missing_values_df.columns = ["Feature", "Number of missing values"]
missing_values_df["Percentage of missing values"] = (missing_values_df["Number of missing values"] / pos_cash_balance_data_df.shape[0]) * 100

sns.barplot(x = missing_values_df["Feature"], y = missing_values_df["Percentage of missing values"])
plt.xticks(rotation = 90)
plt.title("Percentage of missing values in the pos cash balance data")
categorical_features_lst = previous_application_data_df.select_dtypes(["object"]).columns.tolist()

for feature in categorical_features_lst:
    fig, ax = plt.subplots(1, 1, sharex = False, sharey = False, figsize = (20, 10))
    # Plot levels distribution
    if previous_application_data_df[feature].nunique() < 10:
        sns.countplot(x = previous_application_data_df[feature], ax = ax, order = previous_application_data_df[feature].value_counts().index.tolist())
    else:
        sns.countplot(y = previous_application_data_df[feature], ax = ax, order = previous_application_data_df[feature].value_counts().index.tolist())
    ax.set_title("Count plot of each level of the feature: " + feature)
numerical_features_lst = list(set(previous_application_data_df.columns.tolist()) - set(categorical_features_lst))

# generate the linkage matrix
numerical_features_df = previous_application_data_df[numerical_features_lst]
numerical_features_df.fillna(-1, inplace = True) # We need to impute missing values before creating the dendrogram
numerical_features_df = numerical_features_df.transpose()
Z = linkage(numerical_features_df, "ward")
plt.figure(figsize = (20, 15))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("feature")
plt.ylabel("distance")
dend = dendrogram(
    Z,
    leaf_rotation = 90.,  # rotates the x axis labels
    leaf_font_size = 8.,  # font size for the x axis labels
    labels = numerical_features_df.index.tolist()
)
plt.figure(figsize = (15, 15))
sns.heatmap(previous_application_data_df[dend["ivl"]].corr(), annot = True, square = True)
plt.title("Correlation plot between numerical features")
for feature in numerical_features_lst:
    fig, ax = plt.subplots(1, 1, sharex = False, sharey = False, figsize = (20, 7))
    
    # Plot feature distribution
    sns.distplot(previous_application_data_df[feature].dropna(), kde = False, bins = 100, ax = ax)
    ax.set_title("Histogram of the feature: " + feature)
missing_values_sr = previous_application_data_df.isnull().sum()
missing_values_df = missing_values_sr.loc[missing_values_sr > 0].sort_values(ascending = False).reset_index()
missing_values_df.columns = ["Feature", "Number of missing values"]
missing_values_df["Percentage of missing values"] = (missing_values_df["Number of missing values"] / previous_application_data_df.shape[0]) * 100

sns.barplot(x = missing_values_df["Feature"], y = missing_values_df["Percentage of missing values"])
plt.xticks(rotation = 90)
plt.title("Percentage of missing values in the previous applications data")

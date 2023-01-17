import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib

from mpl_toolkits.mplot3d import Axes3D

import sklearn

import sys



from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import ParameterGrid

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

from sklearn.metrics import f1_score, fbeta_score, accuracy_score, confusion_matrix, roc_curve, auc

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC, SVC

from sklearn.model_selection import StratifiedKFold



from imblearn.combine import SMOTEENN

from imblearn.under_sampling import EditedNearestNeighbours

from imblearn.over_sampling import ADASYN, SMOTE





from pandas.api.types import is_numeric_dtype



palette = ["#9b59b6", "#5497db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

sns.set_palette(palette)

# Columns of the dataset (data comes without an header)

columns = ["existing_account", "month_duration", "credit_history",\

           "purpose", "credit_amount", "saving_bonds",\

           "employment_status", "installment_rate", "status_sex", \

           "debts_status", "resident_since", "property", \

           "age", "installment_plans", "housing_status", \

           "credit_number", "job", "people_liability", \

           "telephone", "foreign", "result"]



numerical_attributes = ["month_duration", "credit_amount", "installment_rate", "resident_since", "age",\

                        "credit_number", "people_liability"]
df = pd.read_csv("../input/german-credit-data/german.data", sep=" ", header=None, names=columns)
df.head()
df.info()
df.describe()


# Positive labels

eligible= df["result"] == 1

print(f"The proportion of individuals which are eligible for credit (good risk) is: {df[eligible].shape[0]/df.shape[0]}")



# Negative labels

not_eligible= df["result"] == 2

print(f"The proportion of individuals which are not eligible for credit (bad risk) is: {df[not_eligible].shape[0]/df.shape[0]}")



# We can note a sligthly umbalance
df_numerical = df.copy()



dummy_columns = ["credit_history", "purpose", "status_sex", "debts_status", "property", "installment_plans", "housing_status", \

            "foreign", "existing_account", "saving_bonds", "telephone", "job", "employment_status"]



df_numerical = pd.get_dummies(df_numerical, columns=dummy_columns, drop_first=True)
df_numerical.head()
corr = df.corr()



fig, ax = plt.subplots(figsize=(10,10)) 

ax = sns.heatmap(corr, annot=True, linewidth=0.5, xticklabels=numerical_attributes + ["result"])

ax.set_yticklabels(labels=numerical_attributes + ["result"], rotation=0)

ax.set_title("Correlation matrix for the numerical features")

plt.show()
corr = df_numerical.corr()



fig, ax = plt.subplots(figsize=(10,10)) 

ax = sns.heatmap(corr, xticklabels=corr.columns)

ax.set_yticklabels(labels=corr.columns, rotation=0) 

ax.set_title("Correlation Matrix considering also dummy encoded variable")

plt.show()
corr_unstacked = corr.unstack().abs()

ranked_corr = corr_unstacked.sort_values(kind="quicksort", ascending=False)

ranked_corr = ranked_corr[ranked_corr != 1]



ranked_corr.head(10)
pairplot_df = df.loc[:, numerical_attributes + ["result"]].replace({"result": {1:"good", 2:"bad"}})

ax = sns.pairplot(pairplot_df, hue="result", plot_kws={"alpha":0.5}, diag_kind="hist")

ax.fig.suptitle("Pairplots for the numberical variables", y=1.08)
# Utility function that given the dataset prints a table pivoted for two values of interest with the counts of each one

def entries_per_couple_variables(df, first_var, second_var, first_level_indexes=None, second_level_indexes=None):

    # perform the renaming on the column values

    renamed_df = df.loc[:,[first_var, second_var]]

    if first_level_indexes is not None and second_level_indexes is not None:

        renamed_df = renamed_df.replace({first_var:first_level_indexes, second_var:second_level_indexes})

    # group for the two columns

    grouped_df = renamed_df.groupby([first_var, second_var]).size().reset_index()

    

    # pivot the result



    pivoted_df = pd.pivot_table(grouped_df, index=first_var, columns=second_var, values=0)

    pivoted_df["Total Column Sum"] = pivoted_df.sum(axis=1)

    for index in second_level_indexes.values():

        pivoted_df[f"proportion_{index}"] = pivoted_df[index]/pivoted_df["Total Column Sum"]

    

    df_total_rows = pivoted_df.sum(axis=0)

    df_total_rows = df_total_rows.rename("Total Row Sum")

    

    

    pivoted_df = pivoted_df.append(df_total_rows)

    for index in first_level_indexes.values():

        prop_df = pivoted_df.loc[index]/df_total_rows

        prop_df = prop_df.rename(f"proportion_{index}")

        prop_df[len(second_level_indexes):] = "-"

        pivoted_df = pivoted_df.append(prop_df)

            

    df_total_rows[len(second_level_indexes)+1:] = "-"

    pivoted_df.loc["Total Row Sum"] = df_total_rows

    

    return pivoted_df
saving_bonds_level_indexes = {"A61": "A61: < 100DM",

                       "A62":"A62: 100 DM<=...< 500DM",

                       "A63":"A63: 500 DM <= ... < 1000 DM",

                       "A64":"A64: >= 1000 DM",

                       "A65":"A65: unknown/ no savings account"}

second_level_indexes = {1:"good", 2:"bad"}

entries_per_couple_variables(df, "saving_bonds", "result", saving_bonds_level_indexes, second_level_indexes)
existing_account_level_indexes = {"A11": "A11: .. < 0 DM",

                       "A12": "A12: 0 <= ... < 200 DM",

                       "A13": "A13: ... >= 200 DM / salary assignments for at least 1 year",

                       "A14": "A14: no checking account"}



entries_per_couple_variables(df, "existing_account", "result", existing_account_level_indexes, second_level_indexes)
entries_per_couple_variables(df, "existing_account", "saving_bonds", existing_account_level_indexes, saving_bonds_level_indexes)
housing_level_indexes = {"A151":"A151: rent",\

                       "A152":"A152: own", \

                       "A153":"A153: for free"}

entries_per_couple_variables(df, "housing_status", "result", housing_level_indexes, second_level_indexes)
property_level_indexes = {"A121": "A121: real estate",\

                          "A122": "A122: if not A121 : building society savings agreement/ life insurance",\

                          "A123": "A123: if not A121/A122 : car or other, not in attribute 6",\

                          "A124": "A124: unknown / no property"}



entries_per_couple_variables(df, "housing_status", "property", housing_level_indexes, property_level_indexes)
first_level_indexes = {"A91":"A91: male:divorced/separated",\

                       "A92":"A92: female:divorced/separated/married",\

                       "A93":"A93: male:single",\

                       "A94":"A94: male:married/widowed"}

entries_per_couple_variables(df, "status_sex", "result", first_level_indexes, second_level_indexes)
first_level_indexes = {"A40":"A40: car (new)",\

                       "A41":"A41: car (used)",\

                       "A42":"A42: furniture/equipment",\

                       "A43":"A43: radio/television",\

                       "A44":"A44: domestic appliances",\

                       "A45":"A45: repairs",\

                       "A46":"A46: education",\

                       "A48" : "A48: retraining",\

                       "A49":"A49: business",\

                       "A410" : "A410: others"}

entries_per_couple_variables(df, "purpose", "result", first_level_indexes, second_level_indexes)
job_level_indexes = {"A171": "unemployed/ unskilled - non-resident",\

                     "A172": "A172: unskilled - resident",\

                     "A173": "A173: skilled employee / official",\

                     "A174": "A174: management/ self-employed/highly qualified employee/ officer"}

entries_per_couple_variables(df, "job", "result", job_level_indexes, second_level_indexes)
fig, axs = plt.subplots(3, 3, figsize=(15, 15))



i = 0

j = 0



for category in numerical_attributes:

    sns.boxplot(y=df[category], x=df["result"].replace({1:"good", 2:"bad"}), ax=axs[i, j], orient="v", showmeans=True)

    j += 1

    if j%3 == 0:

        j = 0

        i += 1

        

axs[2, 1].set_visible(True)

fig.delaxes(axs[2, 1])

axs[2, 2].set_visible(True)

fig.delaxes(axs[2, 2])
X = np.array(df_numerical.loc[:, df_numerical.columns != "result"])

print(f"Shape of the features of the dataset: {X.shape}")



y = np.array(df_numerical.loc[:, "result"].replace({1:0, 2:1}))

print(f"Shape of the labels of the dataset: {y.shape}")



X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, stratify=y)



print(f"Shape of the training set: {X_train.shape}")

print(f"Shape of the test set: {X_test.shape}")
# Standardize data

scaler = StandardScaler()

X_train_raw = X_train

X_test_raw = X_test



X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Default beta=1 and default k=5

ada = ADASYN()



X_train_ada, y_train_ada = ada.fit_resample(X_train, y_train)
print(f"Shape of the balanced ADASYN dataset is: {X_train_ada.shape}")



pos_ratio = y_train_ada[y_train_ada==1].shape[0]/y_train_ada.shape[0]

neg_ratio = y_train_ada[y_train_ada==0].shape[0]/y_train_ada.shape[0]



print(f"Proportion of positive samples in the balanced training set: {pos_ratio:.2f}")

print(f"Proportion of negative samples in the balanced training set: {neg_ratio:.2f}")



pos_ratio_test = y_test[y_test==1].shape[0]/y_test.shape[0]

neg_ratio_test = y_test[y_test==0].shape[0]/y_test.shape[0]



print(f"Proportion of positive samples in the test set: {pos_ratio_test}")

print(f"Proportion of negative samples in the test set: {neg_ratio_test}")
smoteenn = SMOTEENN(smote=SMOTE(), enn=EditedNearestNeighbours())



X_train_smoteenn, y_train_smoteenn = smoteenn.fit_resample(X_train, y_train)
print(f"Shape of the balanced SMOTEEN dataset is: {X_train_smoteenn.shape}")



pos_ratio = y_train_smoteenn[y_train_smoteenn==1].shape[0]/y_train_smoteenn.shape[0]

neg_ratio = y_train_smoteenn[y_train_smoteenn==0].shape[0]/y_train_smoteenn.shape[0]



print(f"Proportion of positive samples in the balanced training set: {pos_ratio:.2f}")

print(f"Proportion of negative samples in the balanced training set: {neg_ratio:.2f}")



pos_ratio_test = y_test[y_test==1].shape[0]/y_test.shape[0]

neg_ratio_test = y_test[y_test==0].shape[0]/y_test.shape[0]



print(f"Proportion of positive samples in the test set: {pos_ratio_test}")

print(f"Proportion of negative samples in the test set: {neg_ratio_test}")
pca = PCA(n_components=2)



X_pca_visualization = pca.fit_transform(X)



fig, ax = plt.subplots(figsize=(10, 10))

sns.scatterplot(x=X_pca_visualization[:, 0], y=X_pca_visualization[:, 1], hue=df["result"].replace({1:"good", 2:"bad"}))

ax.set_title("Original Dataset reduced to two components with PCA")

ax.set_xlabel("First PC")

ax.set_ylabel("Second PC")

plt.show()



print(f"The percentage of variance explained by each components is: {pca.explained_variance_ratio_}")



pca = PCA(n_components=2)

X_ada, y_ada =  ada.fit_resample(scaler.fit_transform(X), y)

X_pca_visualization_ada = pca.fit_transform(X_ada)



fig, ax = plt.subplots(figsize=(10, 10))

sns.scatterplot(x=X_pca_visualization_ada[:, 0], y=X_pca_visualization_ada[:, 1], \

                hue=pd.Series(y_ada).replace({0:"good", 1:"bad"}))

ax.set_title("ADASYN rebalanced Dataset reduced to two components with PCA")

ax.set_xlabel("First PC")

ax.set_ylabel("Second PC")

plt.show()



print(f"The percentage of variance explained by each components for the ADASYN dataset is: {pca.explained_variance_ratio_}")



pca = PCA(n_components=2)

X_smoteenn, y_smoteenn =  smoteenn.fit_resample(scaler.fit_transform(X), y)

X_pca_visualization_smoteenn = pca.fit_transform(X_smoteenn)



fig, ax = plt.subplots(figsize=(10, 10))

sns.scatterplot(x=X_pca_visualization_smoteenn[:, 0], y=X_pca_visualization_smoteenn[:, 1], \

                hue=pd.Series(y_smoteenn).replace({0:"good", 1:"bad"}))

ax.set_title("SMOTEENN rebalanced Dataset reduced to two components with PCA")

ax.set_xlabel("First PC")

ax.set_ylabel("Second PC")

plt.show()



print(f"The percentage of variance explained by each components for the SMOTEENN dataset is: {pca.explained_variance_ratio_}")
pca = PCA(n_components=3)

X_pca_visualization = pca.fit_transform(scaler.fit_transform(X))



fig, ax = plt.subplots(figsize=(10, 10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca_visualization[df["result"] == 1, 0], \

           X_pca_visualization[df["result"] == 1, 1], \

           X_pca_visualization[df["result"] == 1, 2], label="bad")



ax.scatter(X_pca_visualization[df["result"] == 2, 0], \

           X_pca_visualization[df["result"] == 2, 1], \

           X_pca_visualization[df["result"] == 2, 2], label="good")

ax.legend()

ax.set_title("Original Dataset reduced to three components with PCA")

ax.set_xlabel("First PC")

ax.set_ylabel("Second PC")

ax.set_zlabel("Third PC")

plt.show()



print(f"The percentage of variance explained by each components is: {pca.explained_variance_ratio_}")



X_ada, y_ada =  ada.fit_resample(scaler.fit_transform(X), y)

X_pca_visualization_ada = pca.fit_transform(X_ada)



fig, ax = plt.subplots(figsize=(10, 10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca_visualization_ada[y_ada==1, 0], \

           X_pca_visualization_ada[y_ada==1, 1], \

           X_pca_visualization_ada[y_ada==1, 2], label="bad")



ax.scatter(X_pca_visualization_ada[y_ada==0, 0], \

           X_pca_visualization_ada[y_ada==0, 1], \

           X_pca_visualization_ada[y_ada==0, 2], label="good")

ax.legend()

ax.set_title("ADASYN Dataset reduced to three components with PCA")

ax.set_xlabel("First PC")

ax.set_ylabel("Second PC")

ax.set_zlabel("Third PC")

plt.show()



print(f"The percentage of variance explained by each components for the ADASYN dataset is: {pca.explained_variance_ratio_}")



X_smoteenn, y_smoteenn =  smoteenn.fit_resample(scaler.fit_transform(X), y)

X_pca_visualization_smoteenn = pca.fit_transform(X_smoteenn)



fig, ax = plt.subplots(figsize=(10, 10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca_visualization_smoteenn[y_smoteenn==1, 0], \

           X_pca_visualization_smoteenn[y_smoteenn==1, 1], \

           X_pca_visualization_smoteenn[y_smoteenn==1, 2], label="bad")



ax.scatter(X_pca_visualization_smoteenn[y_smoteenn==0, 0], \

           X_pca_visualization_smoteenn[y_smoteenn==0, 1], \

           X_pca_visualization_smoteenn[y_smoteenn==0, 2], label="good")

ax.legend()

ax.set_title("SMOTEENN Dataset reduced to three components with PCA")

ax.set_xlabel("First PC")

ax.set_ylabel("Second PC")

ax.set_zlabel("Third PC")

plt.show()



print(f"The percentage of variance explained by each components for the SMOTEENN dataset is: {pca.explained_variance_ratio_}")
pca = PCA(n_components=X_train.shape[1])

c_analyzed = 30



pca.fit(X_train)

fig, ax = plt.subplots(figsize=(10, 10))

ax = sns.lineplot(x=range(1, X_train.shape[1]+1), y=pca.explained_variance_ratio_, label="Variance explained by each component")

ax = sns.lineplot(x=range(1, X_train.shape[1]+1), y=np.cumsum(pca.explained_variance_ratio_), label="Cumulative variance explained")

ax = sns.scatterplot(x=[c_analyzed], y=[np.cumsum(pca.explained_variance_ratio_)[c_analyzed-1]], s=100)

ax.text(c_analyzed+1, np.cumsum(pca.explained_variance_ratio_)[c_analyzed-1], f"{np.cumsum(pca.explained_variance_ratio_)[c_analyzed-1]:.2f}",\

       horizontalalignment='left')



ax = sns.scatterplot(x=[c_analyzed], y=[pca.explained_variance_ratio_[c_analyzed-1]], s=100)

ax.text(c_analyzed+1, pca.explained_variance_ratio_[c_analyzed-1], f"{pca.explained_variance_ratio_[c_analyzed-1]:.2f}",\

       horizontalalignment='left')

plt.axvline(c_analyzed, color="black", ls="--")

ax.set_title("number of componet used against PVE for the original Dataset")

ax.set_xlabel("Principal component")

ax.set_ylabel("PVE")

plt.show()
pca = PCA(n_components=X_train_ada.shape[1])



pca.fit(X_train_ada)

fig, ax = plt.subplots(figsize=(10, 10))

ax = sns.lineplot(x=range(1, X_train_ada.shape[1]+1), y=pca.explained_variance_ratio_, label="Variance explained by each component")

ax = sns.lineplot(x=range(1, X_train_ada.shape[1]+1), y=np.cumsum(pca.explained_variance_ratio_), label="Cumulative variance explained")

ax = sns.scatterplot(x=[c_analyzed], y=[np.cumsum(pca.explained_variance_ratio_)[c_analyzed-1]], s=100)

ax.text(c_analyzed+1, np.cumsum(pca.explained_variance_ratio_)[c_analyzed-1], f"{np.cumsum(pca.explained_variance_ratio_)[c_analyzed-1]:.2f}",\

       horizontalalignment='left')



ax = sns.scatterplot(x=[c_analyzed], y=[pca.explained_variance_ratio_[c_analyzed-1]], s=100)

ax.text(c_analyzed+1, pca.explained_variance_ratio_[c_analyzed-1], f"{pca.explained_variance_ratio_[c_analyzed-1]:.2f}",\

       horizontalalignment='left')

plt.axvline(c_analyzed, color="black", ls="--")

ax.set_title("number of componet used against PVE for the ADASYN re-balanced Dataset")

ax.set_xlabel("Principal component")

ax.set_ylabel("PVE")

plt.show()
pca = PCA(n_components=X_train_smoteenn.shape[1])



pca.fit(X_train_smoteenn)

fig, ax = plt.subplots(figsize=(10, 10))

ax = sns.lineplot(x=range(1, X_train_smoteenn.shape[1]+1), y=pca.explained_variance_ratio_, label="Variance explained by each component")

ax = sns.lineplot(x=range(1, X_train_smoteenn.shape[1]+1), y=np.cumsum(pca.explained_variance_ratio_), label="Cumulative variance explained")

ax = sns.scatterplot(x=[c_analyzed], y=[np.cumsum(pca.explained_variance_ratio_)[c_analyzed-1]], s=100)

ax.text(c_analyzed+1, np.cumsum(pca.explained_variance_ratio_)[c_analyzed-1], f"{np.cumsum(pca.explained_variance_ratio_)[c_analyzed-1]:.2f}",\

       horizontalalignment='left')



ax = sns.scatterplot(x=[c_analyzed], y=[pca.explained_variance_ratio_[c_analyzed-1]], s=100)

ax.text(c_analyzed+1, pca.explained_variance_ratio_[c_analyzed-1], f"{pca.explained_variance_ratio_[c_analyzed-1]:.2f}",\

       horizontalalignment='left')

plt.axvline(c_analyzed, color="black", ls="--")

ax.set_title("number of componet used against PVE for the SMOTEENN re-balanced Dataset")

ax.set_xlabel("Principal component")

ax.set_ylabel("PVE")

plt.show()
pca = PCA(c_analyzed)



X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)
default_weighting = {"TP":0,\

                        "TN":0,\

                        "FP":1,\

                        "FN":5}



adjusted_weighting = {"TP":0,\

                        "TN":-1,\

                        "FP":1,\

                        "FN":10}                        



def compute_penalty(y, y_hat, weight_matrix):

    

    penalty = 0

    count = 0

    for y_i, y_hat_i in zip(y, y_hat):

        

        # True Positive Case

        if(y_i == 1 and y_hat_i == 1):

            penalty += weight_matrix["TP"]

        

        # True Negative Case

        if(y_i == 0 and y_hat_i == 0):

            penalty += weight_matrix["TN"]

            

        # False Positive Case

        if(y_i == 0 and y_hat_i == 1):

            penalty += weight_matrix["FP"]        

            

        # False Negative Case

        if(y_i == 1 and y_hat_i == 0):

            penalty += weight_matrix["FN"]

        

        count += 1

            

    return penalty / count       
results_df = pd.DataFrame(columns=["model_name", "parameters", "training_score (f2)","f1_test", "f2_test", "accuracy_test", \

                                   "penalty_default_matrix", "penalty_adj_matrix"])
y_hat_baseline = np.ones((300, 1))



# Train score

f2_train = fbeta_score(y_train, np.ones((700, 1)), beta=2)



# Train score

f2_test = fbeta_score(y_test, y_hat_baseline, beta=2)



# f1 score

f1_test = f1_score(y_test, y_hat_baseline)



# Accuracy

accuracy_test = accuracy_score(y_test, y_hat_baseline)



# Penalty default weight

penalty_default = compute_penalty(y_test, y_hat_baseline, weight_matrix=default_weighting)



# Penalty adjusted weight 

penalty_adjusted = compute_penalty(y_test, y_hat_baseline, weight_matrix=adjusted_weighting)



output_df = pd.DataFrame(data=[[f1_test, f2_test, accuracy_test, penalty_default, penalty_adjusted]],\

        columns=["f1_test", "f2_test", "accuracy_test", "penalty_default_matrix", "penalty_adj_matrix"])



conf_matrix = confusion_matrix(y_test, y_hat_baseline)



fig, ax = plt.subplots(figsize=(7, 5))

sns.heatmap(conf_matrix, annot=True, linewidth=0.5, xticklabels=["good", "bad"], fmt="g")

ax.set_yticklabels(labels=["good", "bad"], rotation=0) 

ax.set_xlabel("predicted")

ax.set_ylabel("actual")

ax.set_title("Confusion matrix for the baseline model")

plt.show()
classifier_df = pd.DataFrame(data = [["Baseline model", "N/A" , f2_train]], \

        columns=["model_name", "parameters", "training_score (f2)"])

results_df = results_df.append(pd.concat([classifier_df, output_df], axis=1, sort=False))
results_df
summary_df = results_df.copy()
def trainClassifier(X_train, y_train, model_name, classifier, params, score, score_parameters=None,\

                    greater_is_better=True, verbose=False, adasyn=None, smoteenn=None, pca=None, standardize=None):



    kf = StratifiedKFold(10)

    

    

    train_scores = []



    if greater_is_better:

        best_score = 0

    else:

        best_score = sys.float_info.max

        

    for config in ParameterGrid(params):

        train_scores_run = []

        counts = []

        for train_indices, valid_indices in kf.split(X_train, y_train):

            counts.append(len(train_indices))

            X_train_kf = X_train[train_indices]

            y_train_kf = y_train[train_indices]

            X_valid_kf = X_train[valid_indices]

            y_valid_kf = y_train[valid_indices]

                       

            if standardize is not None: 

                X_train_kf = standardize.fit_transform(X_train_kf)

                X_valid_kf = standardize.transform(X_valid_kf)

            

            if adasyn is not None:

                X_train_kf, y_train_kf = adasyn.fit_resample(X_train_kf, y_train_kf)

                

            if smoteenn is not None:

                X_train_kf, y_train_kf = smoteenn.fit_resample(X_train_kf, y_train_kf)

            

            if pca is not None:

                X_train_kf = pca.fit_transform(X_train_kf)

                X_valid_kf = pca.transform(X_valid_kf)



            # keep track of the number of elements in each split

            model = classifier(**config)

            model.fit(X_train_kf, y_train_kf)

            y_hat = model.predict(X_valid_kf)

            train_score = score(y_valid_kf, y_hat, **score_parameters)

            train_scores_run.append(train_score)



        if(greater_is_better):

            if np.average(train_scores_run, weights=counts) > best_score:

                best_score = np.average(train_scores_run, weights=counts)

                best_config = config

                if(verbose):

                    print("New best score obtained")

                    print(f"Training with: {config}")

                    print(f"Total Score obtained with cross validation: {best_score}\n")

        else:

            if np.average(train_scores_run, weights=counts) < best_score:

                best_score = np.average(train_scores_run, weights=counts)

                if(verbose):

                    print("New best score obtained")

                    print(f"Training with: {config}")

                    print(f"Total Score obtained with cross validation: {best_score}\n")



        train_scores.append(np.average(train_scores_run, weights=counts))



    output_df = pd.DataFrame(data = [[model_name, best_config ,best_score]], \

        columns=["model_name", "parameters", "training_score (f2)"])



    return output_df



def testClassifier(X_train, y_train, X_test, y_test, classifier, best_params, adasyn=None, smoteenn=None, pca=None, standardize=None, proba=False):

    # Train model obtained with best hyperparameters              

    if standardize is not None: 

        X_train = standardize.fit_transform(X_train)

        X_test = standardize.transform(X_test)

    

    if adasyn is not None:

        X_train, y_train = ada.fit_resample(X_train, y_train)

        

    if smoteenn is not None:

        X_train, y_train = smoteenn.fit_resample(X_train, y_train)

        

    if pca is not None:

        X_train = pca.fit_transform(X_train)

        X_test = pca.transform(X_test)

    

    best_model = classifier(**best_params)

    best_model.fit(X_train, y_train)

    y_hat = best_model.predict(X_test)



    # f1 score

    f1_test = f1_score(y_test, y_hat)

    

    # f2 score

    f2_test = fbeta_score(y_test, y_hat, beta=2)



    # Accuracy

    accuracy_test = accuracy_score(y_test, y_hat)# Train model obtained with best hyperparameters



    # Penalty default weight

    penalty_default = compute_penalty(y_test, y_hat, weight_matrix=default_weighting)



    # Penalty adjusted weight 

    penalty_adjusted = compute_penalty(y_test, y_hat, weight_matrix=adjusted_weighting)



    output_df = pd.DataFrame(data=[[f1_test, f2_test, accuracy_test, penalty_default, penalty_adjusted]],\

        columns=["f1_test", "f2_test", "accuracy_test", "penalty_default_matrix", "penalty_adj_matrix"])



    conf_matrix = confusion_matrix(y_test, y_hat)

    

    if proba == False:

        y_score = best_model.decision_function(X_test)

    else:

        y_score = best_model.predict_proba(X_test)[:, 1]

        

    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)

    roc_auc = auc(fpr, tpr)

    roc = [fpr, tpr]  

    

    return output_df, conf_matrix, roc, roc_auc



def performAnalysis(train_Xs, y_train, test_Xs, y_test, model_names, classifier,\

                    params, score, score_parameters, greater_is_better=True, verbose=False,\

                    adasyn=None, smoteenn=None, pca=None, standardize=None, proba=False):

    

    # Find best parameters with cross validation

    conf_matrices = []

    rocs = []

    roc_aucs = []

    

    results_df = pd.DataFrame(columns=["model_name", "parameters", "training_score (f2)", "f1_test", "f2_test",\

                                       "accuracy_test", "penalty_default_matrix", "penalty_adj_matrix"])

    

    for X_train, X_test, model_name in zip(train_Xs, test_Xs, model_names):

        classifier_df = trainClassifier(X_train, y_train, model_name, classifier, params=params, score=score,\

             score_parameters=score_parameters, greater_is_better=True,verbose=False, adasyn=adasyn, pca=pca, smoteenn=smoteenn, standardize=standardize)

        

        tests_df, conf_matrix, roc, auc = testClassifier(X_train, y_train, X_test, y_test, classifier, \

                                                classifier_df["parameters"].to_dict()[0], adasyn=adasyn, smoteenn=smoteenn, pca=pca, standardize=standardize, proba=proba)

        conf_matrices.append(conf_matrix)

        rocs.append(roc)

        roc_aucs.append(auc)

                                                                         

        results_df = results_df.append(pd.concat([classifier_df, tests_df], axis=1, sort=False))

                                                                

    return results_df, conf_matrices, rocs, roc_aucs



def draw_rocs(rocs, roc_aucs, roc_titles):

    fig, axs = plt.subplots(2, 3, figsize=(25, 10))

    i = 0

    j = 0



    for roc, roc_auc, roc_title in zip(rocs, roc_aucs, roc_titles):

        lw=2

        sns.lineplot(roc[0], roc[1], color='darkorange',

             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc, ax=axs[j, i])

        sns.lineplot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', ax=axs[j, i])

        plt.xlim([0.0, 1.0])

        plt.ylim([0.0, 1.05])

        axs[j, i].set_xlabel('False Positive Rate')

        axs[j, i].set_ylabel('True Positive Rate')

        axs[j, i].set_title(roc_title)

        axs[j, i].legend(loc="lower right")



        i += 1

        if i%3 == 0 and i != 0:

            j += 1

            i = 0



    plt.show()   

    

def draw_conf_matrices(conf_matrices, plot_titles):

    fig, axs = plt.subplots(2, 3, figsize=(25, 10))

    i = 0

    j = 0



    for conf_matrix, plot_title in zip(conf_matrices, plot_titles):

        sns.heatmap(conf_matrix, annot=True, linewidth=0.5, xticklabels=["good", "bad"], ax=axs[j, i], fmt="g")

        axs[j, i].set_yticklabels(labels=["good", "bad"], rotation=0) 

        axs[j, i].set_xlabel("predicted")

        axs[j, i].set_ylabel("actual")

        axs[j, i].set_title(plot_title)



        i += 1

        if i%3 == 0 and i != 0:

            j += 1

            i = 0

    plt.show()



def run_classifier(proba=True):

    results_df, conf_matrices, rocs, roc_aucs = performAnalysis(\

                                                [X_train_raw], y_train, [X_test_raw], y_test, model_names=[model_names[0]], \

                                                classifier=classifier, params=params, score=fbeta_score, score_parameters={"beta":2},\

                                                greater_is_better=True, verbose=False, proba=proba, standardize=scaler)



    results_df_pca, conf_matrices_pca, rocs_pca, roc_aucs_pca, = performAnalysis(\

                                                [X_train_raw], y_train, [X_test_raw], y_test, model_names=[model_names[1]], \

                                                classifier=classifier, params=params, score=fbeta_score, score_parameters={"beta":2},\

                                                greater_is_better=True, verbose=False, proba=proba, pca=pca, standardize=scaler)



    rocs.append(rocs_pca[0])

    roc_aucs.append(roc_aucs_pca[0])

    results_df = results_df.append(results_df_pca)

    conf_matrices.append(conf_matrices_pca[0])



    results_df_ada, conf_matrices_ada, rocs_ada, roc_aucs_ada = performAnalysis(\

                                                [X_train_raw], y_train, [X_test_raw], y_test, model_names=[model_names[2]],\

                                                classifier=classifier, params=params, score=fbeta_score, score_parameters={"beta":2},\

                                                greater_is_better=True, verbose=False, proba=proba, adasyn=ada, pca=None, standardize=scaler)



    results_df = results_df.append(results_df_ada)

    conf_matrices.append(conf_matrices_ada[0])

    rocs.append(rocs_ada[0])

    roc_aucs.append(roc_aucs_ada[0])



    results_df_ada_pca, conf_matrices_ada_pca, rocs_ada_pca, roc_aucs_ada_pca = performAnalysis(\

                                                [X_train_raw], y_train, [X_test_raw], y_test, model_names=[model_names[3]],\

                                                classifier=classifier, params=params, score=fbeta_score, score_parameters={"beta":2}, \

                                                greater_is_better=True, verbose=False, proba=proba, adasyn=ada, pca=pca, standardize=scaler)



    results_df = results_df.append(results_df_ada_pca)

    conf_matrices.append(conf_matrices_ada_pca[0])

    rocs.append(rocs_ada_pca[0])

    roc_aucs.append(roc_aucs_ada_pca[0])



    results_df_smoteenn, conf_matrices_smoteenn, rocs_smoteenn, roc_aucs_smoteenn = performAnalysis(\

                                                [X_train_raw], y_train, [X_test_raw], y_test, model_names=[model_names[4]],\

                                                classifier=classifier, params=params, score=fbeta_score, score_parameters={"beta":2}, \

                                                greater_is_better=True, verbose=False, proba=proba, smoteenn=smoteenn, pca=None, standardize=scaler)



    results_df = results_df.append(results_df_smoteenn)

    conf_matrices.append(conf_matrices_smoteenn[0])

    rocs.append(rocs_smoteenn[0])

    roc_aucs.append(roc_aucs_smoteenn[0])



    results_df_smoteenn_pca, conf_matrices_smoteenn_pca, rocs_smoteenn_pca, roc_aucs_smoteenn_pca = performAnalysis(\

                                                [X_train_raw], y_train, [X_test_raw], y_test, model_names=[model_names[5]],\

                                                classifier=classifier, params=params, score=fbeta_score, score_parameters={"beta":2}, \

                                                greater_is_better=True, verbose=False, proba=proba, smoteenn=smoteenn, pca=pca, standardize=scaler)



    results_df = results_df.append(results_df_smoteenn_pca)

    conf_matrices.append(conf_matrices_smoteenn_pca[0])

    rocs.append(rocs_smoteenn_pca[0])

    roc_aucs.append(roc_aucs_smoteenn_pca[0])

    

    return results_df, conf_matrices, rocs, roc_aucs
params = {

    "n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15]

}

classifier = KNeighborsClassifier

model_names = ["K-NN", "K-NN PCA", "K-NN ADASYN", "K-NN ADASYN PCA", "K-NN SMOTEENN", "K-NN SMOTEENN PCA"]

conf_matr_titles = ["Confusion Matrix K-NN not reduced dimension", "Confusion Matrix K-NN PCA",\

               "Confusion Matrix K-NN ADASYN", "Confusion Matrix K-NN ADASYN PCA",\

               "Confusion Matrix K-NN SMOTEENN", "Confusion Matrix K-NN SMOTEENN PCA"]



roc_titles = ["ROC K-NN not reduced dimension", "ROC K-NN PCA",\

              "ROC K-NN ADASYN", "ROC K-NN ADASYN PCA",\

              "ROC K-NN SMOTEENN", "ROC K-NN SMOTEENN PCA"]

              
results_df, conf_matrices, rocs, roc_aucs = run_classifier(proba=True)



summary_df = summary_df.append(results_df)



results_df
draw_conf_matrices(conf_matrices, conf_matr_titles)
draw_rocs(rocs, roc_aucs, roc_titles)
params = {

    "C": [1e-3, 1e-2, 1e-1, 1],

    "max_iter": [30000]

}

classifier = LinearSVC

model_names = ["Linear SVM", "Linear SVM PCA",\

               "Linear SVM ADASYN", "Linear SVM ADASYN PCA",\

               "Linear SVM SMOTEENN", "Linear SVM SMOTEENN PCA"]

conf_matr_titles = ["Confusion Matrix Linear SVM not reduced dimension", "Confusion Matrix Linear SVM PCA",\

                    "Confusion Matrix Linear SVM ADASYN", "Confusion Matrix Linear SVM ADASYN PCA",\

                    "Confusion Matrix Linear SVM SMOTEENN", "Confusion Matrix Linear SVM SMOTEENN PCA"]



roc_titles = ["ROC Linear SVM not reduced dimension", "ROC Linear SVM PCA",\

              "ROC Linear SVM ADASYN", "ROC Linear SVM ADASYN PCA",\

              "ROC Linear SVM SMOTEENN", "ROC Linear SVM SMOTEENN PCA"]
results_df, conf_matrices, rocs, roc_aucs = run_classifier(proba=False)



summary_df = summary_df.append(results_df)



results_df
draw_conf_matrices(conf_matrices, conf_matr_titles)
draw_rocs(rocs, roc_aucs, roc_titles)
params = {

    "kernel" : ["rbf"],

    "C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],

    "gamma": [1e-3, 1e-2, 1e-1, 1, 10]

}

classifier = SVC

model_names = ["RBF SVM", "RBF SVM PCA",\

               "RBF SVM ADASYN", "RBF SVM ADASYN PCA",\

               "RBF SVM SMOTEENN", "RBF SVM SMOTEENN PCA"]

conf_matr_titles = ["Confusion Matrix RBF SVM not reduced dimension", "Confusion Matrix RBF SVM PCA",\

                    "Confusion Matrix RBF SVM ADASYN", "Confusion Matrix RBF SVM ADASYN PCA",\

                    "Confusion Matrix RBF SVM SMOTEENN", "Confusion Matrix RBF SVM SMOTEENN PCA"]



roc_titles = ["ROC RBF SVM not reduced dimension", "ROC RBF SVM PCA",\

              "ROC RBF SVM ADASYN", "ROC RBF SVM ADASYN PCA",\

              "ROC RBF SVM SMOTEENN", "ROC RBF SVM SMOTEENN PCA"]
results_df, conf_matrices, rocs, roc_aucs = run_classifier(proba=False)



summary_df = summary_df.append(results_df)



results_df
draw_conf_matrices(conf_matrices, conf_matr_titles)
draw_rocs(rocs, roc_aucs, roc_titles)
params = {

    "C": [1e-3, 1e-2, 1e-1, 1, 10]

}

classifier = LogisticRegression

model_names = ["Logistic Regression", "Logistic Regression PCA",\

               "Logistic Regression ADASYN", "Logistic Regression ADASYN PCA",\

               "Logistic Regression SMOTEENN", "Logistic Regression SMOTEENN PCA"]

conf_matr_titles = ["Confusion Matrix Logistic Regression not reduced dimension", "Confusion Matrix Logistic Regression PCA",\

               "Confusion Matrix Logistic Regression ADASYN", "Confusion Matrix Logistic Regression ADASYN PCA",\

                   "Confusion Matrix Logistic Regression SMOTEENN", "Confusion Matrix Logistic Regression SMOTEENN PCA"]



roc_titles = ["ROC Logistic Regression not reduced dimension", "ROC Logistic Regression not reduced dimension PCA",\

              "ROC Logistic Regression ADASYN", "ROC Logistic Regression ADASYN PCA",\

              "ROC Logistic Regression SMOTEENN", "ROC Logistic Regression SMOTEENN PCA"]

results_df, conf_matrices, rocs, roc_aucs = run_classifier(proba=True)



summary_df = summary_df.append(results_df)



results_df
draw_conf_matrices(conf_matrices, conf_matr_titles)
draw_rocs(rocs, roc_aucs, roc_titles)
params = {

    "max_depth": [None, 4, 6, 8, 10],

    "min_samples_split": [2, 0.2, 0.4],

    "min_samples_leaf": [1, 0.2]

}

classifier = DecisionTreeClassifier

model_names = ["Decision Tree", "Decision Tree PCA",\

               "Decision Tree ADASYN", "Decision Tree ADASYN PCA",\

               "Decision Tree SMOTEENN", "Decision Tree SMOTEENN PCA"]

conf_matr_titles = ["Confusion Matrix Decision Tree not reduced dimension", "Confusion Matrix Decision Tree PCA",\

                    "Confusion Matrix Decision Tree ADASYN", "Confusion Matrix Decision Tree ADASYN PCA",\

                    "Confusion Matrix Decision Tree SMOTEENN", "Confusion Matrix Decision Tree SMOTEENN PCA"]

roc_titles = ["ROC Decision Tree not reduced dimension", "ROC Decision Tree PCA",\

              "ROC Decision Tree ADASYN", "ROC Decision Tree ADASYN PCA",\

              "ROC Decision Tree SMOTEENN", "ROC Decision Tree SMOTEENN PCA"]
results_df, conf_matrices, rocs, roc_aucs = run_classifier(proba=True)



summary_df = summary_df.append(results_df)



results_df
draw_conf_matrices(conf_matrices, conf_matr_titles)
draw_rocs(rocs, roc_aucs, roc_titles)
params = {"max_depth": [3,5, 7, 10,None],

          "n_estimators":[3,5,10,25,50],

          "max_features": [4,7,15,20, "auto"]}

classifier = RandomForestClassifier

model_names = ["Random Forest", "Random Forest PCA",\

               "Random Forest ADASYN", "Random Forest ADASYN PCA",\

               "Random Forest SMOTEENN", "Random Forest SMOTEENN PCA"]

conf_matr_titles = ["Confusion Matrix Random Forest not reduced dimension", "Confusion Matrix Random Forest PCA",\

                    "Confusion Matrix Random Forest ADASYN", "Confusion Matrix Random Forest ADASYN PCA",\

                    "Confusion Matrix Random Forest SMOTEENN", "Confusion Matrix Random Forest SMOTEENN PCA"]

roc_titles = ["ROC Random Forest not reduced dimension", "ROC Random Forest PCA",\

              "ROC Random Forest ADASYN", "ROC Random Forest ADASYN PCA",\

              "ROC Random Forest SMOTEENN", "ROC Random Forest SMOTEENN PCA"]
results_df, conf_matrices, rocs, roc_aucs = run_classifier(proba=True)



summary_df = summary_df.append(results_df)



results_df
draw_conf_matrices(conf_matrices, conf_matr_titles)
draw_rocs(rocs, roc_aucs, roc_titles)
summary_df = summary_df.reset_index(drop=True)

f2_df = summary_df[["f2_test", "model_name"]].copy()

f2_df.loc[0, "model_type"] = "Baseline model"

f2_df.loc[1:len(model_names)+1, "model_type"] = "K-NN"

f2_df.loc[len(model_names)+1:2*len(model_names)+1, "model_type"] = "Linear SVM"

f2_df.loc[2*len(model_names)+1:3*len(model_names)+1, "model_type"] = "RBF SVM"

f2_df.loc[3*len(model_names)+1:4*len(model_names)+1, "model_type"] = "Logistic Regression"

f2_df.loc[4*len(model_names)+1:5*len(model_names)+1, "model_type"] = "Decision Tree"

f2_df.loc[5*len(model_names)+1:6*len(model_names)+1, "model_type"] = "Random Forest"



f2_df = f2_df.sort_values(by=["f2_test"], ascending=False)



fig, ax = plt.subplots(figsize=(15, 15))



ax = sns.barplot(y=f2_df["model_name"], x=f2_df["f2_test"], hue=f2_df["model_type"])

ax.set_title("Final results for the different classification models")

plt.show()
summary_df
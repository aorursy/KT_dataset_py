import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.estimator import LinearRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
print(tf.__version__)
from tqdm import tqdm
import seaborn as sns
import os

%matplotlib inline
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (7, 6)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

os.chdir('/kaggle/input/fda-potential-renal-failure/DATA/')


drug_vocab = pd.read_csv("drug_vocabulary.csv")

df = pd.read_csv('non_renal-reason_ov3-renal-symptoms_ov1-drug.csv', index_col = 0)

df.loc[:, 'Normalised Suspect Product Active Ingredients'] = df['Suspect Product Active Ingredients']


df.columns
# Normalising Drug Vocabulary

for d in tqdm(drug_vocab.iterrows()):
    standard_drug = d[1]["Drug"]
    
    #print(d[1]["Terms"])
    
    drug_alts = set(d[1]["Terms"].split(";"))
    
    drug_query = None
    
    # find smallest term
    
    for D in drug_alts:
        try:
            if len(D.split()) < len(drug_query.split()):
                drug_query = D
        except:
            drug_query = D
        
    drug_subset_idxs = np.array(df[df['Suspect Product Active Ingredients'].str.contains(drug_query)].index)
    
    for i in drug_subset_idxs:
        
        entry_drugs = set(df.at[i,'Suspect Product Active Ingredients'].split(";"))
        entry_drugs = entry_drugs - drug_alts
        entry_drugs = list(entry_drugs)
        entry_drugs.append(standard_drug)
        entry_drugs = ";".join(entry_drugs)
        df.at[i,'Normalised Suspect Product Active Ingredients'] = entry_drugs
drug_ref_vocab = []

for l in df['Normalised Suspect Product Active Ingredients'].str.split(";").values:
    drug_ref_vocab.extend(l)
    drug_ref_vocab = list(set(drug_ref_vocab))
    
drug_ref_vocab = sorted(drug_ref_vocab)
    
print("Unique drugs: {}".format(len(drug_ref_vocab)))
# Confirming that database only contains drug combinations per case and not a single drug. 

df.loc[:, "Number of Drugs"] = df["Normalised Suspect Product Active Ingredients"].str.split(";").apply(len)

df = df[df["Number of Drugs"] > 1]


# Drug counts and distribution

d_counts = df["Normalised Suspect Product Active Ingredients"].str.split(";").values
drug_counts = []

for l in d_counts:
    drug_counts.extend(l)
    
del d_counts

drug_counts = pd.DataFrame(np.array(drug_counts))
drug_counts = drug_counts[0].value_counts().reset_index()
drug_counts.columns = ["Drug", "Frequency"]

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (14, 12))

# ax1 ---
sns.distplot(drug_counts["Frequency"], ax = ax1)
ax1.set_title("Distribution of Drug Frequencies in the dataset (n = {})".format(len(df)))

# ax2 ---
c = drug_counts.sort_values("Frequency", ascending = False)
sns.barplot(y = "Drug", x = "Frequency", data = c.head(15), ax = ax2, orient="h")
ax2.set_title("Top 15 occuring drugs in the dataset (n = {})".format(len(df)))
ax2.set_yticklabels(c["Drug"], rotation = 0)

# ax3 ---
sns.distplot(df["Normalised Suspect Product Active Ingredients"].str.split(";").apply(len).values, ax = ax3)
ax3.set_title("Distribution of Number of Drugs per case in the dataset (n = {})".format(len(df)))
ax3.set_ylabel("Frequency Distribution")
ax3.set_xlabel("Number of Drugs per case")

plt.tight_layout()
plt.show()
# Drugs vs Number of Drugs per case

drug_per_case = None

df.loc[:, "Number of Drugs"] = df["Normalised Suspect Product Active Ingredients"].str.split(";").apply(len)

for d in tqdm(drug_ref_vocab):
    try:
        _ = pd.DataFrame(df[df["Normalised Suspect Product Active Ingredients"].str.contains(d)]["Number of Drugs"])
        _.loc[:, "Drug"] = d
        
        if _.shape[0] > 1:
            try:
                drug_per_case = pd.concat([drug_per_case, _])
            except:
                drug_per_case = _
    except:
        print(d)
plt.figure(figsize = (6, 12))
c = drug_per_case.groupby("Drug").mean().reset_index().sort_values("Number of Drugs", ascending = False).head(30)
sns.barplot(y = "Drug", x = "Number of Drugs", data = c, orient = "h")
plt.xlabel("Average Number of Drugs in Combination")
plt.show()
c = drug_per_case.groupby("Drug").mean().reset_index()#.sort_values("Number of Drugs", ascending = False)
c = pd.merge(left = drug_counts, right = c, on = "Drug")

f, (ax1, ax2) = plt.subplots(1,2, figsize = (14, 6))

# ax1 ---
sns.scatterplot(y = "Number of Drugs", x = "Frequency", data = c, ax = ax1)
ax1.set_xlabel("Frequency in Dataset")
ax1.set_ylabel("Average number of drugs per case")

# ax2 ---
#sns.barplot(y = "Drug", x = "Number of Drugs" , hue = "Frequency", data = c[c["Frequency"] > 3].sort_values(["Number of Drugs", "Frequency"]).head(15), ax = ax2)
drug_combo_counts = df["Normalised Suspect Product Active Ingredients"].value_counts().reset_index()
sns.barplot(y = "index", x = "Normalised Suspect Product Active Ingredients", data = drug_combo_counts.head(10), ax = ax2, orient = "h")
ax2.set_xticks(np.arange(19))
ax2.set_xlabel("Drug Combination")

plt.tight_layout()
plt.show()
drug_combo_counts = df[df["Normalised Suspect Product Active Ingredients"].isin(drug_combo_counts[drug_combo_counts["Normalised Suspect Product Active Ingredients"] > 1]["index"].values)]
non_overlap_drug_combos = drug_combo_counts.groupby("Normalised Suspect Product Active Ingredients").sum()
dc = non_overlap_drug_combos.index.values
#dc = np.array([d[:50] + "..." for d in dc])
non_overlap_drug_combos = non_overlap_drug_combos.values[:, 2:6]
non_overlap_drug_combos = pd.DataFrame(non_overlap_drug_combos / drug_combo_counts.groupby("Normalised Suspect Product Active Ingredients").size().values.reshape(-1,1), index = dc)
# Removed those combinations where each set of symptoms perfectly overlapped over cases.
non_overlap_drug_combos = non_overlap_drug_combos[(non_overlap_drug_combos.iloc[:, :].sum(axis = 1) < 3) &
                                                  (non_overlap_drug_combos.iloc[:, :].mean(axis = 1) != 0.5)]

plt.figure(figsize = (10, 11))
sns.heatmap(non_overlap_drug_combos)

plt.yticks(np.arange(len(non_overlap_drug_combos)) + 0.5, non_overlap_drug_combos.index.values)
plt.xticks(np.arange(4) + 0.5, df.columns[12:16])
plt.title("Proportion of Cases with specific Drug Combinations with renal failure symptoms")

plt.tight_layout()
plt.show()

non_overlap_drug_combos_df = df[df["Normalised Suspect Product Active Ingredients"].isin(non_overlap_drug_combos.index.values)]
c = non_overlap_drug_combos_df.groupby(["Sex", "Normalised Suspect Product Active Ingredients"]).size()
c = (c/c.groupby(["Normalised Suspect Product Active Ingredients"]).sum()).reset_index()


f, (ax1, ax2) = plt.subplots(1,2, figsize = (14, 9), sharey = True)

# ax1 ---
sns.barplot(y = "index", x = "Normalised Suspect Product Active Ingredients",
            data = non_overlap_drug_combos_df["Normalised Suspect Product Active Ingredients"].value_counts().reset_index(),
            orient = "h", ax = ax1)
ax1.set_xlabel("Frequency")
ax1.set_ylabel("Drug Combination")

# ax2 ---
sns.barplot(y = "Normalised Suspect Product Active Ingredients", x = 0, hue = "Sex", data = c, orient = "h", ax = ax2)
ax2.set_xlabel("Proportion of Sex")
ax2.set_ylabel("")


plt.tight_layout()
plt.show()
all_df = pd.read_csv("fda_all.csv", usecols = ["Case ID", "Patient Weight", "Patient Age"])

all_df.head()

# Cleaning 

all_df.loc[:, "Age"] = all_df["Patient Age"].str.extract(r"(\d+)", flags = 0, expand = False)
all_df.loc[:, "Weight"] = all_df["Patient Weight"].str.extract(r"(\d+\.{0,1}\d+)", flags = 0, expand = False)

all_df = all_df.loc[:, ["Case ID", "Age", "Weight"]]

df = pd.merge(right = df, left = all_df, on = "Case ID")

df["Age"] = df["Age"].astype(float)
df["Weight"] = df["Weight"].astype(float)
non_overlap_drug_combos_df = df[df["Normalised Suspect Product Active Ingredients"].isin(non_overlap_drug_combos.index.values)]
f, (ax1, ax2) = plt.subplots(1,2, figsize = (14, 12), sharey = True)

# ax1 --
c = non_overlap_drug_combos_df[~non_overlap_drug_combos_df["Age"].isna()]
sns.boxplot(y = "Normalised Suspect Product Active Ingredients", x = "Age",
            data = c, ax = ax1, orient = "h")
ax1.set_title("""Distribution of Patient Age by 
Non-overlapping Drug Combinations
(n = {})""".format(len(c)))
ax1.set_ylabel("Drug Combination")
ax1.set_xlabel("Patient Age")

# ax1 --
c = non_overlap_drug_combos_df[~non_overlap_drug_combos_df["Weight"].isna()]
sns.boxplot(y = "Normalised Suspect Product Active Ingredients", x = "Weight",
            data = c, ax = ax2, orient = "h")
ax2.set_title("""Distribution of Patient Age by 
Non-overlapping Drug Combinations
(n = {})""".format(len(c)))
ax2.set_ylabel("")
ax2.set_xlabel("Patient Weight")

plt.tight_layout()
plt.show()
df.loc[:, "Patient Age"] = df.groupby("Normalised Suspect Product Active Ingredients")['Age'].apply(lambda x:x.fillna(x.mean()))
df.loc[:, "Patient Weight"] = df.groupby("Normalised Suspect Product Active Ingredients")['Weight'].apply(lambda x:x.fillna(x.mean()))
df.loc[df[df["Patient Age"].isna()].index.values, "Patient Age"] = df["Patient Age"].mean()
df.loc[df[df["Patient Weight"].isna()].index.values, "Patient Weight"] = df["Patient Weight"].mean()
c = df.loc[:, "oliguria":"decreased appetite"]

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 6))

# ax1 ---
c_ = c.sum(axis = 0).reset_index()
sns.barplot(y = "index", x = 0, data = c_, ax = ax1, orient="h")
ax1.set_title("""Distribution of renal failure-related symptoms
(n = {})""".format(len(df)))
ax1.set_yticklabels(c_["index"], rotation = 0)
ax1.set_ylabel("Symptoms")
ax1.set_xlabel("Frequency")

# ax2 ---
c_ = c.sum(axis = 1).reset_index()
sns.distplot(c_[0], ax = ax2)
ax2.set_title("""Distribution of Number of Symptoms per case in the dataset
(n = {})""".format(len(df)))
ax2.set_xlabel("Number of Symptoms")
ax2.set_ylabel("Frequency Density")



plt.tight_layout()
plt.show()
def one_hot_encode(drug_entry):
    """
    One hot encoding given string drug entry
    """
    
    drugs = drug_entry.split(";")
    
    one_hot = np.zeros(len(drug_ref_vocab))
    
    for d in drugs:
        try:
            idx = drug_ref_vocab.index(d)
            one_hot[idx] = 1
        except:
            continue
        
    return one_hot
from sklearn.preprocessing import StandardScaler

def prepare_dataset(age_weight_flag = True):
    """
    Prepares dataset for the model.
    
    :param age_weight_flag: (Boolean) include Patient Age and Weight as features
    """

    X = None

    for entry in tqdm(df["Normalised Suspect Product Active Ingredients"].values):

        v = one_hot_encode(entry)
        try:
            X = np.vstack((X, v))
        except:
            X = v

    
    # Adding Patient Age and Weight

    scaler_age = StandardScaler()
    scaler_weight = StandardScaler()

    scaled_age = scaler_age.fit_transform(df["Patient Age"].values.reshape(-1,1))#[:, 0]
    scaled_weight = scaler_weight.fit_transform(df["Patient Weight"].values.reshape(-1,1))#[:, 0]

    if age_weight_flag:
        X = np.hstack((X, scaled_age))
        X = np.hstack((X, scaled_weight))

    print("X shape: {}".format(X.shape))
    
    y = df.loc[:, "oliguria":"decreased appetite"].values

    print("y shape: {}".format(y.shape))
    
    X_trn , X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.1)
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=0.2)

    for i, lab in zip([X_trn , X_val, X_tst], ["Train", "Validation", "Test"]):
        print("{} set examples: {}".format(lab, i.shape))
    
    return X_trn, X_val, X_tst, y_trn, y_val, y_tst
def test_acc(model):
    predictions = model.predict(X_tst)
    tst_pred = np.round(predictions)
    acc = tst_pred - y_tst
    acc = np.round((np.where(acc.sum(axis = 1) == 0)[0].size / acc.shape[0]) * 100, 2)
    print("Test accuracy : {}%".format(acc))
def plot_loss(histories, labels):
    
    f, (ax1, ax2) = plt.subplots(1,2, figsize = (14,6))
    
    for history, label, n in zip(histories, labels, np.arange(len(labels))):
        
        # ax1 ---
        # Use a log scale to show the wide range of values.
        ax1.semilogy(history.epoch,  history.history['loss'],
                   color=colors[n], label='Train '+label, alpha = 0.5)
        ax1.semilogy(history.epoch,  history.history['val_loss'],
              color=colors[n], label='Val '+label,
              linestyle="--", alpha = 0.5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')



        # ax2 ---
        ax2.semilogy(history.epoch,  history.history['accuracy'],
                   color=colors[n], label='Train '+label, alpha = 0.5)
        ax2.semilogy(history.epoch,  history.history['val_accuracy'],
              color=colors[n], label='Val '+label,
              linestyle="--", alpha = 0.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')

    plt.legend()
X_trn, X_val, X_tst, y_trn, y_val, y_tst = prepare_dataset(age_weight_flag = False)

baseline = keras.Sequential([
    keras.layers.Dense(units=800, activation='elu'),
    keras.layers.Dense(units=800, activation='elu'),
    keras.layers.Dense(units=y_tst.shape[1], activation='sigmoid')
], name = "baseline")

baseline.compile(optimizer='adam', 
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

baseline.build(X_trn.shape)

baseline.summary()
bs_history = baseline.fit(
    X_trn,
    y_trn,
    epochs=100,
    validation_data = (X_val, y_val),
    batch_size = 32
)
X_trn, X_val, X_tst, y_trn, y_val, y_tst = prepare_dataset(age_weight_flag = True)

baseline_inc_feat = keras.Sequential([
    keras.layers.Dense(units=800, activation='elu'),
    keras.layers.Dense(units=800, activation='elu'),
    keras.layers.Dense(units=y_tst.shape[1], activation='sigmoid')
], name = "baseline_inc_feat")

baseline_inc_feat.compile(optimizer='adam', 
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

baseline_inc_feat.build(X_trn.shape)

baseline_inc_feat.summary()
baseline_inc_feat_history = baseline_inc_feat.fit(
    X_trn,
    y_trn,
    epochs=100,
    validation_data = (X_val, y_val),
    batch_size = 32
)
X_trn, X_val, X_tst, y_trn, y_val, y_tst = prepare_dataset(age_weight_flag = True)

baseline_inc_feat_reg = keras.Sequential([
    keras.layers.Dense(units=800, activation='elu'),
    keras.layers.Dropout(0.15),
    keras.layers.Dense(units=800, activation='elu', kernel_regularizer='l2'),
    keras.layers.Dense(units=y_tst.shape[1], activation='sigmoid')
], name = "baseline_inc_feat_reg")

baseline_inc_feat_reg.compile(optimizer='adam', 
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

baseline_inc_feat_reg.build(X_trn.shape)

baseline_inc_feat_reg.summary()
baseline_inc_feat_reg_history = baseline_inc_feat_reg.fit(
    X_trn,
    y_trn,
    epochs=100,
    validation_data = (X_val, y_val),
    batch_size = 32
)
X_trn, X_val, X_tst, y_trn, y_val, y_tst = prepare_dataset(age_weight_flag = True)

baseline_inc_feat_reg_drop = keras.Sequential([
    keras.layers.Dense(units=800, activation='elu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(units=800, activation='elu'),
    keras.layers.Dense(units=y_tst.shape[1], activation='sigmoid')
], name = "baseline_inc_feat_reg_drop")

baseline_inc_feat_reg_drop.compile(optimizer='adam', 
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

baseline_inc_feat_reg_drop.build(X_trn.shape)

baseline_inc_feat_reg_drop.summary()
baseline_inc_feat_reg_drop_history = baseline_inc_feat_reg_drop.fit(
    X_trn,
    y_trn,
    epochs=100,
    validation_data = (X_val, y_val),
    batch_size = 32
)
plot_loss([bs_history, baseline_inc_feat_history, baseline_inc_feat_reg_drop_history, baseline_inc_feat_reg_history],
          ["Baseline", "Baseline Inc Feature", "Baseline Drop Inc Feature", "Baseline Reg Inc Feature"])
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier

clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=4))

clf.fit(X_trn, y_trn)
print("""One-vs-rest XGBoost classifier
Validation accuracy: {}""".format(clf.score(X_val, y_val)))
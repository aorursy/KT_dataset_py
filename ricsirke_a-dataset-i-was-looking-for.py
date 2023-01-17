import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.tree import export_graphviz
import sklearn.decomposition
import sklearn.naive_bayes

import os
print(os.listdir("../input"))
df = pd.read_csv('../input/train_ajEneEa.csv')
df.head()
df.describe()
df.info()
sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(3,2, figsize=(11,11))

df_married = df["ever_married"].value_counts()
df_work = df["work_type"].value_counts()
df_resid = df["Residence_type"].value_counts()
df_smoke = df["smoking_status"].value_counts()
df_gender = df["gender"].value_counts()

sns.barplot(x=df_gender.index, y=df_gender, ax=axes[0,0])
sns.despine(left=True)

sns.distplot(df["age"], color="m", ax=axes[0,1])
sns.despine(left=True)

sns.barplot(x=df_married.index, y=df_married, ax=axes[1,0])
sns.despine(left=True)

sns.barplot(x=df_work.index, y=df_work, ax=axes[1,1])
sns.despine(left=True)

sns.barplot(x=df_resid.index, y=df_resid, ax=axes[2,0])
sns.despine(left=True)

sns.barplot(x=df_smoke.index, y=df_smoke, ax=axes[2,1])
sns.despine(left=True)

axes[0,0].set_title("Gender")
axes[0,1].set_title("Age distribution - histogram and kernel estimate")
axes[1,0].set_title("Ever married")
axes[1,1].set_title("Work type")
axes[2,0].set_title("Residence type")
axes[2,1].set_title("Smoking status")

for i in range(3):
    for j in range(2):
        axes[i,j].set_ylabel("")

plt.tight_layout()
f, axes = plt.subplots(2,3, figsize=(11,11))

df_hyp = df["hypertension"].value_counts()
df_heart = df["heart_disease"].value_counts()
df_stroke = df["stroke"].value_counts()

sns.barplot(x=df_hyp.index, y=df_hyp, ax=axes[0,0])
sns.despine(left=True)

sns.barplot(x=df_heart.index, y=df_heart, ax=axes[0,1])
sns.despine(left=True)

sns.barplot(x=df_stroke.index, y=df_stroke, ax=axes[0,2])
sns.despine(left=True)

sns.distplot(df["avg_glucose_level"], ax=axes[1,0])
sns.despine(left=True)

sns.distplot(df["bmi"].dropna(), ax=axes[1,1])
sns.despine(left=True)

for i in range(2):
    for j in range(3):
        axes[i,j].set_ylabel("")
        
axes[0,0].set_title("Hypertension")
axes[0,1].set_title("Heart disease")
axes[0,2].set_title("Stroke")
axes[1,0].set_title("Distribution of average glucose level")
axes[1,1].set_title("Body mass index")
axes[1,2].set_axis_off()

plt.tight_layout()
df_attrs = df[["id","gender","ever_married", "Residence_type", "smoking_status"]]
df_attrs.head()
df_attrs[["gender", "ever_married", "id"]].groupby(["gender", "ever_married"]).count()
table = sm.stats.Table.from_data(df[["gender", "ever_married"]])
table.table_orig
table.fittedvalues
table.resid_pearson
rslt = table.test_nominal_association()
rslt.pvalue
table.chi2_contribs
df_hyptens = df[["id", "gender", "ever_married", "Residence_type", "smoking_status", "hypertension"]]
df_hyptens.head()
(df["smoking_status"].value_counts(dropna=False)/len(df["smoking_status"])).plot.bar()
plt.ylabel("percentage")
sns.despine(left=True)
plt.show()
df_smoke_inj = df[["gender", "ever_married", "Residence_type", "work_type", "smoking_status"]]
df_smoke_inj = df_smoke_inj[df_smoke_inj["smoking_status"].notnull()]
df_smoke_inj.head()
def encodeAllCol(df_in, isUseBinarizer):
    le = sklearn.preprocessing.LabelEncoder()
    lb = sklearn.preprocessing.LabelBinarizer()
    df_enc = pd.DataFrame()
    for i, col in enumerate(df_in):
        le.fit(df_in[col])
        encoded_col = pd.DataFrame(le.fit_transform(df_in[col]), columns=[col + "_enc"])
        if isUseBinarizer[i]:
            lb.fit(encoded_col)
            if len(lb.classes_) == 2:
                # set the column names
                columns = [col]
            else:
                columns = [col + str(i) for i in range(len(lb.classes_))]
            encoded_col = pd.DataFrame(lb.transform(encoded_col), columns=columns)
        df_enc = pd.concat([df_enc, encoded_col], axis=1)
    return df_enc
features = ["gender_enc", "ever_married_enc", "Residence_type_enc", "work_type_enc"]
target = "smoking_status_enc"
df_smoke_enc = encodeAllCol(df_smoke_inj, isUseBinarizer=[True, True, True, True, False])
df_smoke_enc.head()
pca = sklearn.decomposition.PCA()
pca.fit(df_smoke_enc.drop("smoking_status_enc", axis=1))
pd.Series(pca.explained_variance_ratio_).plot.bar()
plt.title("Explained variance ratio")
plt.show()
pca = sklearn.decomposition.PCA(2)

X = pca.fit_transform(df_smoke_enc.drop("smoking_status_enc", axis=1))
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
plt.scatter(x=X[:,0], y=X[:,1], s=200, c=df_smoke_enc["smoking_status_enc"])
plt.title("2D projection of the dataset based explained variance maximiazion by PCA\nthe different colors encode the smoking status variable")
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")
_ = plt.plot()
df_train_X, df_test_X, df_train_y, df_test_y = sklearn.model_selection.train_test_split(df_smoke_enc.drop(target, axis=1), df_smoke_enc[target], test_size=0.3, random_state=1)
rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=10, class_weight="balanced", max_features=None)
rfc.fit(df_train_X, df_train_y)
print("score:", rfc.score(df_test_X, df_test_y))
#print(rfc.estimators_)
pd.Series(rfc.predict(df_test_X)).value_counts()
pd.Series(rfc.feature_importances_).plot.bar()
plt.show()
export_graphviz(rfc.estimators_[0])
os.system('dot -Tpng tree.dot -o tree.png')
pca_model = sklearn.decomposition.PCA()
smoking_enc_red_pca_full = pca_model.fit_transform(df_smoke_enc.drop("smoking_status_enc", axis=1))
plt.plot(pca_model.explained_variance_ratio_, 'o')
plt.title("explained variance ratio for each component")
plt.show()
pca_model = sklearn.decomposition.PCA(5)
smoking_enc_red = pca_model.fit_transform(df_smoke_enc.drop("smoking_status_enc", axis=1))
pd.DataFrame(smoking_enc_red).head()
df_train_X, df_test_X, df_train_y, df_test_y = sklearn.model_selection.train_test_split(smoking_enc_red, df_smoke_enc[target], test_size=0.3)
rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=100, class_weight="balanced", max_features=None)
rfc.fit(df_train_X, df_train_y)
print("score:", rfc.score(df_test_X, df_test_y))
export_graphviz(rfc.estimators_[0])
os.system('dot -Tpng tree.dot -o tree.png')
smoking_enc_red_pca_small_expl_var = smoking_enc_red_pca_full[:,5:]
df_train_X, df_test_X, df_train_y, df_test_y = sklearn.model_selection.train_test_split(smoking_enc_red_pca_small_expl_var, df_smoke_enc[target], test_size=0.3)
rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=100, class_weight="balanced", max_features=None, oob_score=True)
rfc.fit(df_train_X, df_train_y)
print("mean accuracy score:", rfc.score(df_test_X, df_test_y))
print("oob-scoe:", rfc.oob_score_ )
#print(sklearn.model_selection.cross_val_score(rfc, smoking_enc_red, df_smoke_enc[target], cv=10))
df_smoke_inj_enc_full = encodeAllCol(df_smoke_inj, isUseBinarizer=[False, False, False, False, False])
df_smoke_inj_X = df_smoke_inj_enc_full[features]
df_smoke_inj_y = df_smoke_inj_enc_full[target]
mnnb = sklearn.naive_bayes.MultinomialNB()
mnnb.fit(df_smoke_inj_X, df_smoke_inj_y)
print("cross validation score:", sklearn.model_selection.cross_val_score(mnnb, df_smoke_inj_X, df_smoke_inj_y))
print("class log probabilities:", np.exp(mnnb.class_log_prior_) )
features = [ "gender", "ever_married", "Residence_type", "work_type" ]
target = "smoking_status"
df_smoke_inj = df_smoke_inj[df_smoke_inj["gender"] != "Other"]
df_feat = df_smoke_inj[features]
df_targ = df_smoke_inj[target]

for feat in features:
    table = sm.stats.Table.from_data(df_smoke_inj[[target, feat]])
    rslt = table.test_nominal_association()
    print(target, "vs", feat)
    print("p-value:", rslt.pvalue)
    print()
    print(table.resid_pearson)
    print("-----------------------")
    #table.table_orig
    
sm.stats.Table.from_data(df_smoke_inj[[target,  "gender"]]).table_orig
sm.stats.Table.from_data(df_smoke_inj[[target,  "ever_married"]]).table_orig
sm.stats.Table.from_data(df_smoke_inj[[target,  "Residence_type"]]).table_orig
sm.stats.Table.from_data(df_smoke_inj[[target,  "work_type"]]).table_orig

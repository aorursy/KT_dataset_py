import os

import random

import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from mlxtend.plotting import plot_confusion_matrix



from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import StandardScaler



import lightgbm as lgb



def seed_everything(seed=2020):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    # tf.random.set_seed(seed)

seed_everything(0)



sns.set_style("whitegrid")

palette_ro = ["#ee2f35", "#fa7211", "#fbd600", "#75c731", "#1fb86e", "#0488cf", "#7b44ab"]



ROOT = "../input/breast-cancer-wisconsin-data"
df = pd.read_csv(ROOT + "/data.csv")



print("Data shape: ", df.shape)

df.head()
df.info()
df.isnull().sum()
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

sns.countplot(x="diagnosis", ax=ax, data=df, palette=palette_ro[6::-5], alpha=0.9)



ax.annotate(len(df[df["diagnosis"]=="M"]), xy=(-0.05, len(df[df["diagnosis"]=="M"])+5),

            size=16, color=palette_ro[6])

ax.annotate(len(df[df["diagnosis"]=="B"]), xy=(0.95, len(df[df["diagnosis"]=="B"])+5),

            size=16, color=palette_ro[1])



fig.suptitle("Distribution of diagnosis", fontsize=18);
scaler = StandardScaler()

columns = df.columns.drop(["id", "Unnamed: 32", "diagnosis"])



data_s = pd.DataFrame(scaler.fit_transform(df[columns]), columns=columns)

data_s = pd.concat([df["diagnosis"], data_s.iloc[:, 0:10]], axis=1)

data_s = pd.melt(data_s, id_vars="diagnosis", var_name="features", value_name="value")



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))

sns.violinplot(x="features", y="value", hue="diagnosis", ax=ax1,

               data=data_s, palette=palette_ro[6::-5], split=True,

               scale="count", inner="quartile")



sns.swarmplot(x="features", y="value", hue="diagnosis", ax=ax2,

              data=data_s, palette=palette_ro[6::-5])



fig.suptitle("Mean values distribution", fontsize=18);
data_s = pd.DataFrame(scaler.fit_transform(df[columns]), columns=columns)

data_s = pd.concat([df["diagnosis"], data_s.iloc[:, 10:20]], axis=1)

data_s = pd.melt(data_s, id_vars="diagnosis", var_name="features", value_name="value")



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))

sns.violinplot(x="features", y="value", hue="diagnosis", ax=ax1,

               data=data_s, palette=palette_ro[6::-5], split=True,

               scale="count", inner="quartile")



sns.swarmplot(x="features", y="value", hue="diagnosis", ax=ax2,

              data=data_s, palette=palette_ro[6::-5])



fig.suptitle("Standard error values distribution", fontsize=18);
data_s = pd.DataFrame(scaler.fit_transform(df[columns]), columns=columns)

data_s = pd.concat([df["diagnosis"], data_s.iloc[:, 20:30]], axis=1)

data_s = pd.melt(data_s, id_vars="diagnosis", var_name="features", value_name="value")



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))

sns.violinplot(x="features", y="value", hue="diagnosis", ax=ax1,

               data=data_s, palette=palette_ro[6::-5], split=True,

               scale="count", inner="quartile")



sns.swarmplot(x="features", y="value", hue="diagnosis", ax=ax2,

              data=data_s, palette=palette_ro[6::-5])



fig.suptitle("Worst values distribution", fontsize=18);
df_c = df.reindex(columns=["radius_mean", "radius_se", "radius_worst", "texture_mean", "texture_se", "texture_worst",

                           "perimeter_mean", "perimeter_se", "perimeter_worst", "area_mean", "area_se", "area_worst",

                           "smoothness_mean", "smoothness_se", "smoothness_worst", "compactness_mean", "compactness_se", "compactness_worst",

                           "concavity_mean", "concavity_se", "concavity_worst", "concave points_mean", "concave points_se", "concave points_worst",

                           "symmetry_mean", "symmetry_se", "symmetry_worst", "fractal_dimension_mean", "fractal_dimension_se", "fractal_dimension_worst",

                           "diagnosis"])

df_c = df_c.replace({"M":1, "B":0})



print("Correlation coefficient against diagnosis")

df_c.corr().sort_values("diagnosis", ascending=False)["diagnosis"]
fig, ax = plt.subplots(1, 1, figsize=(18, 12))



sns.heatmap(df_c.corr(), ax=ax, vmax=1, vmin=-1, center=0,

            annot=True, fmt=".2f",

            cmap=sns.diverging_palette(220, 10, as_cmap=True),

            mask=np.triu(np.ones_like(df_c.corr(), dtype=np.bool)))



_, labels = plt.yticks()

labels[30].set_color(palette_ro[0])



fig.suptitle("Diagonal correlation matrix", fontsize=18);
X = df.copy()

y = X["diagnosis"].replace({"M":1, "B":0})

X = X.drop(["id", "Unnamed: 32", "diagnosis"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=0)

X_train.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)

X_train.head()
NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

imp = np.zeros((NFOLD, len(X_train.columns)))

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = lgb.LGBMClassifier(objective="binary",

                             metric="binary_logloss")

    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],

            early_stopping_rounds=10,

            verbose=-1)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)

    imp[fold_id] = clf.feature_importances_



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test, num_iteration=clf.best_iteration_)

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))



print(f"\nOut-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")



feature_imp = pd.DataFrame(sorted(zip(np.mean(imp, axis=0), X_train.columns), reverse=True), columns=["values", "features"])



fig, ax = plt.subplots(1, 1, figsize=(16, 6))

sns.barplot(x="values", y="features", data=feature_imp, palette="Blues_r")

plt.title("Feature importance of default LightGBM", fontsize=18);
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of default LightGBM", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1-score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), ["Benign", "Malignant"], fontsize=16)

plt.yticks(np.arange(2), ["Benign", "Malignant"], fontsize=16);
drop_features1 = ["radius_mean", "radius_se", "radius_worst", "texture_mean", "texture_se",

                  "perimeter_mean", "perimeter_se", "area_mean", "area_worst",

                  "smoothness_mean", "smoothness_se", "compactness_mean", "compactness_se", "compactness_worst",

                  "concavity_mean", "concavity_se", "concavity_worst", "concave points_worst",

                  "symmetry_mean", "symmetry_se", "fractal_dimension_mean", "fractal_dimension_se"]

X_1 = X.drop(drop_features1, axis=1)



fig, ax = plt.subplots(1, 1, figsize=(12, 8))

sns.heatmap(pd.concat([X_1, y], axis=1).corr(), ax=ax, vmax=1, vmin=-1, center=0,

            annot=True, fmt=".2f",

            cmap=sns.diverging_palette(220, 10, as_cmap=True),

            mask=np.triu(np.ones_like(pd.concat([y, X_1], axis=1).corr(), dtype=np.bool)))



_, labels = plt.yticks()

labels[8].set_color(palette_ro[0])



fig.suptitle("Diagonal correlation matrix", fontsize=18);



X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.3, shuffle=True, stratify=y, random_state=0)

X_train.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)
NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

imp = np.zeros((NFOLD, len(X_train.columns)))

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = lgb.LGBMClassifier(objective="binary",

                             metric="binary_logloss",

                             min_child_samples=10,

                             reg_alpha=0.1)

    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],

            early_stopping_rounds=10,

            verbose=-1)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)

    imp[fold_id] = clf.feature_importances_



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test, num_iteration=clf.best_iteration_)

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))

y_pred_gbm = np.mean(y_preds, axis=1)



print(f"\nOut-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")



feature_imp = pd.DataFrame(sorted(zip(np.mean(imp, axis=0), X_train.columns), reverse=True), columns=["values", "features"])



fig, ax = plt.subplots(1, 1, figsize=(16, 6))

sns.barplot(x="values", y="features", data=feature_imp, palette="Blues_r")

plt.title("Feature importance of optimized LightGBM", fontsize=18);
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of optimized LightGBM", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1 score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), ["Benign", "Malignant"], fontsize=16)

plt.yticks(np.arange(2), ["Benign", "Malignant"], fontsize=16);
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=0)

X_train.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)
NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

imp = np.zeros((NFOLD, len(X_train.columns)))

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    # print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = ExtraTreesClassifier(random_state=0)

    clf.fit(X_tr, y_tr)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)

    imp[fold_id] = clf.feature_importances_



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test)

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))



print(f"Out-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")



feature_imp = pd.DataFrame(sorted(zip(np.mean(imp, axis=0), X_train.columns), reverse=True), columns=["values", "features"])



fig, ax = plt.subplots(1, 1, figsize=(16, 6))

sns.barplot(x="values", y="features", data=feature_imp, palette="Blues_r")

plt.title("Feature importance of default Extremely randomized trees", fontsize=18);
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of default Extremely randomized trees", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1 score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), ["Benign", "Malignant"], fontsize=16)

plt.yticks(np.arange(2), ["Benign", "Malignant"], fontsize=16);
drop_features2 = ["radius_mean", "radius_se", "radius_worst", "texture_mean", "texture_se",

                  "perimeter_mean", "perimeter_se", "area_mean", "area_worst",

                  "smoothness_mean", "smoothness_se", "compactness_mean", "compactness_se", "compactness_worst",

                  "concavity_mean",  "concavity_worst", "concave points_mean", "concave points_se",

                  "symmetry_mean", "symmetry_se", "fractal_dimension_mean", "fractal_dimension_se"]

X_2 = X.drop(drop_features2, axis=1)



fig, ax = plt.subplots(1, 1, figsize=(12, 8))

sns.heatmap(pd.concat([X_2, y], axis=1).corr(), ax=ax, vmax=1, vmin=-1, center=0,

            annot=True, fmt=".2f",

            cmap=sns.diverging_palette(220, 10, as_cmap=True),

            mask=np.triu(np.ones_like(pd.concat([y, X_2], axis=1).corr(), dtype=np.bool)))



_, labels = plt.yticks()

labels[8].set_color(palette_ro[0])



fig.suptitle("Diagonal correlation matrix", fontsize=18);



X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size=0.3, shuffle=True, stratify=y, random_state=0)

X_train.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)
NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

imp = np.zeros((NFOLD, len(X_train.columns)))

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    # print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = ExtraTreesClassifier(random_state=0,

                               n_estimators=200,

                               min_samples_split=5)

    clf.fit(X_tr, y_tr)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)

    imp[fold_id] = clf.feature_importances_



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test)

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))

y_pred_ert = np.mean(y_preds, axis=1)



print(f"Out-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")



feature_imp = pd.DataFrame(sorted(zip(np.mean(imp, axis=0), X_train.columns), reverse=True), columns=["values", "features"])



fig, ax = plt.subplots(1, 1, figsize=(16, 6))

sns.barplot(x="values", y="features", data=feature_imp, palette="Blues_r")

plt.title("Feature importance of optimized Extremely randomized trees", fontsize=18);
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of optimized Extremely randomized trees", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1 score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), ["Benign", "Malignant"], fontsize=16)

plt.yticks(np.arange(2), ["Benign", "Malignant"], fontsize=16);
scaler = StandardScaler()

X_s = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)



X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.3, shuffle=True, stratify=y, random_state=0)

X_train.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)
NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    # print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = LogisticRegression(random_state=0)

    clf.fit(X_tr, y_tr)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test)

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))



print(f"Out-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of default linear model", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1 score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), ["Benign", "Malignant"], fontsize=16)

plt.yticks(np.arange(2), ["Benign", "Malignant"], fontsize=16);
drop_features3 = ["radius_mean", "radius_se", "radius_worst", "texture_mean", "texture_se",

                  "perimeter_mean", "perimeter_se", "area_mean", "area_worst",

                  "smoothness_mean", "smoothness_se", "compactness_mean", "compactness_se", 

                  "concavity_se", "concavity_worst", "concave points_mean", "concave points_se",

                  "symmetry_mean", "symmetry_se", "fractal_dimension_mean", "fractal_dimension_se", "fractal_dimension_worst"]

X_3 = X_s.drop(drop_features3, axis=1)



fig, ax = plt.subplots(1, 1, figsize=(12, 8))

sns.heatmap(pd.concat([X_3, y], axis=1).corr(), ax=ax, vmax=1, vmin=-1, center=0,

            annot=True, fmt=".2f",

            cmap=sns.diverging_palette(220, 10, as_cmap=True),

            mask=np.triu(np.ones_like(pd.concat([y, X_3], axis=1).corr(), dtype=np.bool)))



_, labels = plt.yticks()

labels[8].set_color(palette_ro[0])



fig.suptitle("Diagonal correlation matrix", fontsize=18);



X_train, X_test, y_train, y_test = train_test_split(X_3, y, test_size=0.3, shuffle=True, stratify=y, random_state=0)

X_train.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)
NFOLD = 10



skf = StratifiedKFold(n_splits=NFOLD)

models = []

oof = np.zeros((len(X_train), ))

y_preds = np.zeros((len(X_test), NFOLD))



for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):

    # print(f"FOLD {fold_id+1}")

    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]

    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    

    clf = LogisticRegression(random_state=0)

    clf.fit(X_tr, y_tr)

    oof[va_idx] = clf.predict(X_va)

    models.append(clf)



for fold_id, clf in enumerate(models):

    pred_ = clf.predict(X_test)

    y_preds[:, fold_id] = pred_

y_pred = np.rint(np.mean(y_preds, axis=1))

y_pred_lm = np.mean(y_preds, axis=1)



print(f"Out-of-fold accuracy: {accuracy_score(y_train, oof)}")

print(f"Out-of-fold F1 score: {f1_score(y_train, oof)}")

print(f"Test accuracy:        {accuracy_score(y_test, y_pred)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred)}")
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of optimized linear model", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1 score={:0.4f}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)), fontsize=14)

plt.xticks(np.arange(2), ["Benign", "Malignant"], fontsize=16)

plt.yticks(np.arange(2), ["Benign", "Malignant"], fontsize=16);
y_pred_em = y_pred_gbm*2 +  y_pred_ert*2 + y_pred_lm

y_pred_em = (y_pred_em > 3).astype(int)



print(f"Test accuracy:        {accuracy_score(y_test, y_pred_em)}")

print(f"Test F1 score:        {f1_score(y_test, y_pred_em)}")
fig, ax = plot_confusion_matrix(confusion_matrix(y_test, y_pred_em), figsize=(12,8), hide_ticks=True, colorbar=True, class_names=["true", "false"])



plt.title("Confusion Matrix of the ensembled model", fontsize=18)

plt.ylabel("True label", fontsize=14)

plt.xlabel("Predicted label\naccuracy={:0.4f}, F1-score={:0.4f}".format(accuracy_score(y_test, y_pred_em), f1_score(y_test, y_pred_em)), fontsize=14)

plt.xticks(np.arange(2), ["Benign", "Malignant"], fontsize=16)

plt.yticks(np.arange(2), ["Benign", "Malignant"], fontsize=16);
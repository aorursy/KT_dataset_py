import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats

from sklearn.metrics import classification_report

from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, PolynomialFeatures

from category_encoders import WOEEncoder, BinaryEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier
train = pd.read_csv("/kaggle/input/mobile-price-classification/train.csv")

test = pd.read_csv("/kaggle/input/mobile-price-classification/test.csv")

sub = pd.DataFrame(test["id"])

sub["price_range"] = 2

test.drop("id", axis=1, inplace=True)

print(f"train data :{train.shape} test data :{test.shape}")
train.head()
test.head()
cat_var = ["blue","dual_sim","four_g","three_g","touch_screen","wifi"]

con_var = ['px_height', 'sc_h', 'sc_w', 'clock_speed', 'battery_power', 'int_memory', 'talk_time', 'pc',

           'n_cores', 'px_width', 'fc', 'mobile_wt', 'm_dep', 'ram']
def con_plot(var):

    fig, ax = plt.subplots(int(np.ceil(len(con_var)/3)), 3, figsize=(16,16))

    ax = ax.flatten()

    i = 0

    for col in var:

        skew = train[col].skew()

        sns.distplot(train[col], fit = stats.norm, ax=ax[i])

        ax[i].set_title("Variable %s skew : %.4f"%(col, skew))

        i+=1

    plt.tight_layout()

    plt.show()

    

con_plot(con_var)
def cat_plot(var):

    fig, ax = plt.subplots(int(np.ceil(len(var)/3)), 3, figsize=(16,8))

    ax = ax.flatten()

    i = 0

    for col in var:

        sns.countplot(train[col], ax=ax[i])

        ax[i].set_title("devices in each category for %s"%(col))

        i+=1

    plt.tight_layout()

    plt.show()

    

cat_plot(cat_var)
train.isna().sum()
test.isna().sum()
train.price_range.value_counts().plot(kind='bar')

plt.show()
train.skew()
sns.pairplot(train, hue='price_range', diag_kind='hist')

plt.show()
X = train.drop(["price_range"], axis=1)

Y = train["price_range"]
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

vif_val = pd.DataFrame({"Col":X.columns})

vif_val["VIF"] = [vif(X.values, i) for i in range(X.shape[1])]

vif_val
model_rf = RandomForestClassifier(random_state=1, n_jobs=-1)

model_logr = LogisticRegression(random_state=1, n_jobs=-1, multi_class='multinomial')

model_lgbm = LGBMClassifier(random_state=1, n_jobs=-1)

model_xgb = XGBClassifier(random_state=1, n_jobs=-1)

model_gbr = GradientBoostingClassifier(random_state=1)

model_cat = CatBoostClassifier(random_state=1, verbose=0)



models = []

models.append(('LR',model_logr))

models.append(('RF',model_rf))

models.append(('GBR',model_gbr))

models.append(('XGB',model_xgb))

models.append(('LGB',model_lgbm))

models.append(('CAT',model_cat))
scaler = StandardScaler()

onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)

cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

feature = SelectFromModel(model_rf, threshold=0.001)

ct = ColumnTransformer([('onehot', onehot, cat_var),

                        ('scaler', scaler, con_var)], remainder='passthrough', n_jobs=-1)
results = []

names = []

for name, model in models:

    #pipe = Pipeline([('ct', ct), ('fselect', feature), (name, model)]) # including feature selection step using RF

    pipe = Pipeline([('ct', ct), (name, model)])

    scores = cross_val_score(pipe, X, Y, scoring='accuracy', cv=cv, n_jobs=-1, verbose=0)

    names.append(name)

    results.append(scores)

    print("model %s accuracy: %.4f variance: %.4f"%(name, np.mean(scores), np.std(scores)))
plt.figure(figsize=(12,5))

plt.boxplot(results)

plt.xticks(np.arange(1,len(names)+1),names)

plt.title("Accuracy for different machine learning algorithms")

plt.xlabel("Model Name")

plt.ylabel("Cross val Accuracies")

plt.show()
logr_pipe = Pipeline([('ct', ct), ('LR', model_logr)])

logr_pipe.fit(X, Y)

trainpred = logr_pipe.predict(X)
print(classification_report(Y, trainpred))
prediction = logr_pipe.predict(test)
def submission(prediction, model):

    sub["price_range"] = prediction

    sub.price_range.value_counts()

    sub.to_csv("model_"+model+"_mobile_price.csv", index=False)
submission(prediction, 'logr')
onehot_categories = logr_pipe.named_steps['ct'].transformers_[0][1].categories_

onehot_features = [f"{col}__{val}" for col, vals in zip(cat_var, onehot_categories) for val in vals]

all_features = onehot_features + con_var

print(all_features)
coeff = pd.DataFrame(logr_pipe['LR'].coef_, columns=all_features)

coeff.T
import shap

pd.set_option("display.max_columns",None)

shap.initjs()

import xgboost

import eli5
ct.fit(X)

X_shap = ct.fit_transform(X)

test_shap  = ct.transform(test)

explainer = shap.LinearExplainer(logr_pipe.named_steps['LR'], X_shap, feature_perturbation="interventional")

shap_values = explainer.shap_values(test_shap)
shap.summary_plot(shap_values, test_shap, feature_names=all_features)
# prediction class 2, shap values for class 2

shap.force_plot(explainer.expected_value[2], shap_values[2][2], test_shap[2], feature_names=all_features)
# prediction class 2, shap values for class 3

shap.force_plot(explainer.expected_value[3], shap_values[3][2], test_shap[2], feature_names=all_features)
# prediction class 0, shap values for class 0

shap.force_plot(explainer.expected_value[0], shap_values[0][997], test_shap[997], feature_names=all_features)
# prediction class 0, shap values for class 3

shap.force_plot(explainer.expected_value[3], shap_values[3][997], test_shap[997], feature_names=X.columns)
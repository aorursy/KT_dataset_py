

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from scipy import stats 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import collections

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("../input/av-healthcare-analytics-ii/healthcare/train_data.csv")
train.head()
test = pd.read_csv("../input/av-healthcare-analytics-ii/healthcare/test_data.csv")
test.head()
train.shape, test.shape
train.info()
plt.rcParams["figure.figsize"] = 12, 8
sum_ad_deposit = train.groupby("Stay").agg({"Admission_Deposit": "sum"})
sum_ad_deposit.plot(kind = "bar")
plt.title("Sum of Admission deposit")
plt.show()
collections.Counter(train["Stay"])
sns.boxenplot(train["Stay"], train["Visitors with Patient"])
train_X = train.drop("Stay", axis = 1) #dropping the dependent variable for preprocessing
full_data = pd.concat([train_X, test], ignore_index= True)
full_data.isnull().sum()
simple_Impute_median = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
full_data["Bed Grade"] = simple_Impute_median.fit_transform(full_data[["Bed Grade"]]).ravel()
full_data["City_Code_Patient"] = simple_Impute_median.fit_transform(full_data[["City_Code_Patient"]]).ravel()
full_data.isnull().sum()
Lab_enc = LabelEncoder()

for i in full_data.columns:
    if full_data[i].dtype == "object":
        full_data[i] = Lab_enc.fit_transform(full_data[i])
y = Lab_enc.fit_transform(train["Stay"])
def metric(model, pred, y_valid ):
    if hasattr(model, 'oob_score_'): 
        return (accuracy_score(y_valid, pred)) * 100, model.oob_score_
    else:
        return (accuracy_score(y_valid, pred)) * 100

def get_sample(df,y, number):
    df = df.sample(number)
    return df, y[df.index]

def split(X, y, pct = 0.2):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=pct, stratify = y )
    return X_train, X_valid, y_train, y_valid


def feat_imp(model, cols):
    return pd.DataFrame({"Col_names": cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)

def plot_i(fi, x, y):
    return fi.plot(x, y, "barh", figsize = (12,8))

def create_csv(preds):
    cols = ["case_id", "Stay"]
    sub = pd.DataFrame({"case_id": test["case_id"], "Stay": preds})
    sub = sub[cols]
    
    return sub.to_csv("submission.csv", index = False)

#using the helper functions above

sample_X, sample_y = get_sample(full_data[:train.shape[0]], y, 30000)
sample_X.shape, sample_y.shape
x_train , x_valid, y_train, y_valid = split(sample_X, sample_y)
%%time
Rf = RandomForestClassifier(oob_score=True) 
model = Rf.fit(x_train, y_train)
preds = model.predict(x_valid)
print(metric(model, preds, y_valid))
%%time
Rf = RandomForestClassifier(n_estimators=160, max_features=0.5, min_samples_leaf= 5, oob_score=True) 
model = Rf.fit(x_train, y_train)
preds = model.predict(x_valid)
print(metric(model, preds, y_valid))
feat10 = feat_imp(model, sample_X.columns)
feat10[:10]
file4 = pd.read_csv("../input/av-healthcare-analytics-ii/healthcare/train_data_dictionary.csv")
file4
plot_i(feat10, "Col_names", "Importance")
#Removing caseid and patient id from sample_x

sample_X = sample_X.drop(["case_id", "patientid"], axis = 1 )
sample_X.shape
x_train , x_valid, y_train, y_valid = split(sample_X, sample_y)
%%time
Rf = RandomForestClassifier(n_estimators=160, max_features=0.5, min_samples_leaf= 5, oob_score=True) 
model = Rf.fit(x_train, y_train)
preds = model.predict(x_valid)
print(metric(model, preds, y_valid))
feat10 = feat_imp(model, sample_X.columns)
feat10[:12]
from lightgbm import LGBMClassifier
cat= ['Hospital_code', 'Hospital_code', 'Hospital_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 
              'City_Code_Patient', 'Type of Admission', 'Severity of Illness', 'Age']


model_Lgm = LGBMClassifier(random_state=45)
model_Lgm.fit(x_train, y_train, categorical_feature=cat)
preds = model_Lgm.predict(x_valid)
print(metric(model_Lgm, preds, y_valid))
X = full_data[:train.shape[0]]
y = y
X.shape, y.shape
x_train, x_valid, y_train, y_valid = split(X, y)
x_train.shape, x_valid.shape, y_train.shape, y_valid.shape
%%time
model_Lgm = LGBMClassifier(n_estimators=160, num_leaves=32, max_depth=5, reg_lambda= 0.3, random_state=46, n_jobs = -1)
model_Lgm.fit(x_train, y_train, categorical_feature=cat)
preds = model_Lgm.predict(x_valid)
print(metric(model_Lgm, preds, y_valid))

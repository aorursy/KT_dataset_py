import numpy as np 

import pandas as pd



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss



from pdpbox import pdp

from plotnine import *
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_json('../input/two-sigma-connect-rental-listing-inquiries/train.json.zip', convert_dates=['created'])

test_data = pd.read_json('../input/two-sigma-connect-rental-listing-inquiries/test.json.zip', convert_dates=['created'])
train_data.shape
test_data.shape
train_data.info()
test_data.info()
train_data.head()
train_data["num_photos"] = train_data["photos"].apply(len)

train_data["num_features"] = train_data["features"].apply(len)

train_data["num_description_words"] = train_data["description"].apply(lambda x: len(x.split(" ")))

train_data["created"] = pd.to_datetime(train_data["created"])

train_data["created_year"] = train_data["created"].dt.year

train_data["created_month"] = train_data["created"].dt.month

train_data["created_day"] = train_data["created"].dt.day
num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",

             "num_photos", "num_features", "num_description_words",

             "created_year", "created_month", "created_day"]

X = train_data[num_feats]

y = train_data["interest_level"]

X.head()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1,max_depth=15,bootstrap=True,random_state=42)

clf.fit(X_train, y_train)

y_val_pred = clf.predict_proba(X_val)

log_loss(y_val, y_val_pred)
print(clf.score(X_train,y_train))

print(clf.score(X_val,y_val))
def get_sample(df,n):

    

    idxs = sorted(np.random.permutation(len(df))[:n])

    return df.iloc[idxs].copy()
x_all = get_sample(X_train[X_train.bedrooms > 0], 60000)
ggplot(x_all, aes('bedrooms', 'price'))+stat_smooth(se=True, method='lowess' )
def plot_pdp(feat, clusters=None, feat_name=None):

    feat_name = feat_name or feat

    p = pdp.pdp_isolate(clf, x_all, x_all.columns, feat)

    return pdp.pdp_plot(p, feat_name, plot_lines=True,

                        cluster=clusters is not None,

                        n_cluster_centers=clusters)
plot_pdp('bedrooms')
df_ext = X.copy()

df_ext['is_valid'] = 1
df_ext.is_valid[:5021]=0
df_ext.head()
print(X.shape,X_train.shape, X_val.shape)
X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(df_ext, y, test_size=0.33)
print(X.shape,X_train_1.shape, X_val_1.shape)
df_ext.info()
X_df_ext = df_ext[num_feats]

y_df_ext = df_ext["is_valid"]

X_df_ext.head()
m = RandomForestClassifier(n_estimators=30, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_df_ext, y_df_ext);

m.oob_score_
def rf_feat_importance(m, df):

    return pd.DataFrame({'columns':df.columns, 'importance':m.feature_importances_}

                       ).sort_values('importance', ascending=False)
fi= rf_feat_importance(m,X_df_ext); fi[:4]
df_ext.info()
df_ext.drop(['created_month'], axis=1, inplace=True)

X_df_ext.drop(['created_month'], axis=1, inplace=True)
X_train_1.drop(['created_month'], axis=1, inplace=True)

X_val_1.drop(['created_month'], axis=1, inplace=True)
t = RandomForestClassifier(n_estimators=100, n_jobs=-1,max_depth=15,bootstrap=True,random_state=42)

t.fit(X_train_1, y_train_1)

y_val_pred = t.predict_proba(X_val_1)

log_loss(y_val_1, y_val_pred)
print(t.score(X_train_1,y_train_1))

print(t.score(X_val_1,y_val_1))
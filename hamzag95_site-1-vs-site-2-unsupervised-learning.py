import pandas as pd
import numpy as np
from collections import Counter
fnc = pd.read_csv('../input/trends-assessment-prediction/fnc.csv')
loading = pd.read_csv('../input/trends-assessment-prediction/loading.csv')
sites = pd.read_csv('../input/trends-assessment-prediction/reveal_ID_site2.csv')
df = loading
# fnc_features, loading_features = list(fnc.columns[1:]), list(loading.columns[1:])
# df = fnc.merge(loading, on="Id")
sites = np.array(sites).reshape(sites.shape[0])
def get_test_train(df):
    labels = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv')
    labels["is_train"] = True
    df = df.merge(labels, on="Id", how="left")

    
    test_df = df[df["is_train"] != True].copy()
    df = df[df["is_train"] == True].copy()
    df = df.drop(['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2', 'is_train'], axis=1)
    test_df = test_df.drop(['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2', 'is_train'], axis=1)
    return df, test_df
train_df, test_df = get_test_train(df)
site_unknown = test_df[~test_df['Id'].isin(set(sites))]
site2 = test_df[test_df['Id'].isin(set(sites))]
site_unknown.shape
site2.shape
site_1_2 = pd.concat([train_df, site2], axis=0)
site_1_2.head()
from sklearn.cluster import KMeans
import numpy as np
kmeans = KMeans(n_clusters=2, random_state=0).fit(site_1_2)
kmeans.labels_
from collections import Counter
Counter(kmeans.labels_)
site2_preds = kmeans.predict(site2)
Counter(site2_preds)
site_unknown_preds = kmeans.predict(site_unknown)
Counter(site_unknown_preds)
from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination=(510/(5877+510)),random_state=0).fit(site_1_2)
site_unknown_preds = clf.predict(site_unknown)
Counter(site_unknown_preds)
site_unknown_preds = clf.predict(test_df)
Counter(site_unknown_preds)

from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

import matplotlib

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
data = pd.read_csv('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv')
data.head()
targeted_data = data.loc[data['target_class'] == 1]
targeted_data
for name in data.columns:

    if name == "target_class":

        continue

    else:

        data[name + "_quantile_10_diff"] = data[name] - data[name].quantile(0.10)

        data[name + "_quantile_30_diff"] = data[name] - data[name].quantile(0.30)

        data[name + "_median_diff"] = data[name] - data[name].quantile(0.50)

        data[name + "_quantile_70-diff"] = data[name] - data[name].quantile(0.70)

        data[name + "_quantile_90-diff"] = data[name] - data[name].quantile(0.90)

        data[name + "_mean_diff"] = data[name] - data[name].mean()

        data[name + "_min_diff"] = data[name] - data[name].min()

        data[name + "_max_diff"] = data[name] - data[name].max()

        

        data[name + "_quantile_10_diff_targeted"] = data[name] - targeted_data[name].quantile(0.10)

        data[name + "_quantile_30_diff_targeted"] = data[name] - targeted_data[name].quantile(0.30)

        data[name + "_median_diff_targeted"] = data[name] - targeted_data[name].quantile(0.50)

        data[name + "_quantile_70-diff_targeted"] = data[name] - targeted_data[name].quantile(0.70)

        data[name + "_quantile_90-diff_targeted"] = data[name] - targeted_data[name].quantile(0.90)

        data[name + "_mean_diff_targeted"] = data[name] - targeted_data[name].mean()

        data[name + "_min_diff_targeted"] = data[name] - targeted_data[name].min()

        data[name + "_max_diff_targeted"] = data[name] - targeted_data[name].max()
for name_1 in data.columns[0:50]:

    for name_2 in data.columns[0:50]:

        if name_1 == "target_class" or name_2 == "target_class":

            continue

        else:

            if name_1 != name_2:

                data[name_1 + "_mult_" + name_2] = (data[name_1]+0.001) * (data[name_2]+0.001)

                

                data[name_1 + "_other_quantile_10_diff"] = data[name_1] - data[name_2].quantile(0.10)

                data[name_1 + "_other_quantile_30_diff"] = data[name_1] - data[name_2].quantile(0.30)

                data[name_1 + "_other_median_diff"] = data[name_1] - data[name_2].quantile(0.50)

                data[name_1 + "_other_quantile_70-diff"] = data[name_1] - data[name_2].quantile(0.70)

                data[name_1 + "_other_quantile_90-diff"] = data[name_1] - data[name_2].quantile(0.90)

                data[name_1 + "_other_mean_diff"] = data[name_1] - data[name_2].mean()

                data[name_1 + "_other_min_diff"] = data[name_1] - data[name_2].min()

                data[name_1 + "_other_max_diff"] = data[name_1] - data[name_2].max()



              
for name_1 in data.columns[0:50]:

    for name_2 in targeted_data.columns[0:50]:

        if name_1 == "target_class" or name_2 == "target_class":

            continue

        else:



            data[name_1 + "_other_quantile_10_diff_targeted"] = data[name_1] - targeted_data[name_2].quantile(0.10)

            data[name_1 + "_other_quantile_30_diff_targeted"] = data[name_1] - targeted_data[name_2].quantile(0.30)

            data[name_1 + "_other_median_diff_targeted"] = data[name_1] - targeted_data[name_2].quantile(0.50)

            data[name_1 + "_other_quantile_70-diff_targeted"] = data[name_1] - targeted_data[name_2].quantile(0.70)

            data[name_1 + "_other_quantile_90-diff_targeted"] = data[name_1] - targeted_data[name_2].quantile(0.90)

            data[name_1 + "_other_mean_diff_targeted"] = data[name_1] - targeted_data[name_2].mean()

            data[name_1 + "_other_min_diff_targeted"] = data[name_1] - targeted_data[name_2].min()

            data[name_1 + "_other_max_diff_targeted"] = data[name_1] - targeted_data[name_2].max()
data.head()
X = data.drop('target_class',axis = 1)

y = data['target_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = RandomForestClassifier()

clf.fit(X_train,y_train)

predictions = clf.predict(X_test)
print(roc_auc_score(y_test,predictions))

print(accuracy_score(y_test,predictions))
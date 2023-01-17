import numpy as np

import pandas as pd

import seaborn as sns

from operator import add

from pprint import pprint

import matplotlib.pyplot as plt



from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

from sklearn.metrics import confusion_matrix, classification_report, f1_score, make_scorer



from sklearn.cluster import Birch

from sklearn.naive_bayes import GaussianNB

from sklearn.tree  import ExtraTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier





%matplotlib inline
def process_data(data, train_data=False, cols_to_keep=None, scaler=None):

    

    # Handle the issue with col2

    pprint("Processing col2")

    data.col2 = data.col2.replace("Silver", 0).replace("Gold", 1).replace("Diamond", 2).replace("Platinum", 3)

    

    # Handle the issue with col56

    pprint("Processing col56")

    data.col56 = data.col56.replace("Low", 0).replace("Medium", 1).replace("High", 2)

    

    # Handle the issue with col11

    pprint("Processing col11")

    data.col11 = data.col11.replace("Yes", 1).replace("No", 0)

    

    # Handle the issue with col44

    pprint("Processing col44")

    data.col44 = data.col44.replace("Yes", 1).replace("No", 0)

    

    # Handle the issue with col37

    pprint("Processing col37")

    data.col37 = data.col37.replace("Male", 0).replace("Female", 1)

        

    

    if train_data:

        # Scaling

        scaler = RobustScaler()

        data.loc[:, :"col63"] = pd.DataFrame(scaler.fit_transform(data.loc[:, :"col63"]), columns=data.columns[:-1])

        

        # Handle corrrelations

        pprint("Removing highly correlated columns")

        correlation_threshold = 0.90

        correlations = abs(data.corr())

        cols_to_keep = set(data.columns)

        cols_to_keep.remove("Class")

        pprint(f"Number of columns before removal: {len(cols_to_keep)}")

        for col_a in data.columns:

            for col_b in data.columns:

                if correlations[col_a][col_b] >= correlation_threshold and col_a in cols_to_keep and col_b in cols_to_keep and col_a != col_b:

                    if correlations[col_a]["Class"] > correlations[col_b]["Class"]:

                        cols_to_keep.remove(col_b)

                    else:

                        cols_to_keep.remove(col_a)

        pprint("Removing columns uncorrelated with Class")

        tmp = cols_to_keep.copy()

        for col in tmp:

            if correlations["Class"][col] < 0.01:

                cols_to_keep.remove(col)

        del tmp

        pprint(f"Number of columns after removal: {len(cols_to_keep)}")

        data = data.loc[:, cols_to_keep.union({"Class"})]

        return data, cols_to_keep, scaler

    else:

        pprint("Scaling and removing columns based on correlation")

        data = pd.DataFrame(scaler.transform(data), columns=data.columns)

        try:

            data = data.loc[:, cols_to_keep]

        except KeyError:

            pprint(cols_to_keep)

        return data





train = pd.read_csv("../input/data-mining-assignment-2/train.csv", index_col="ID")

train, cols_to_keep, scaler = process_data(train, train_data=True)
plt.figure(figsize=[5, 20])

sns.heatmap(pd.DataFrame(abs(train.corr().Class.sort_values(ascending=False))), vmin=0, vmax=1, center=0, annot=True)
plt.figure(figsize=[15, 15])

sns.heatmap(abs(train.corr()), vmin=0, vmax=1, center=0)
pprint(train.Class.value_counts())

pprint((len(train), len(train.columns)))
X, y = train.loc[:, :"col63"], train["Class"]
print(f"X: {len(X), len(X.columns)}")

print(f"y: {len(y), 1}")
rf = RandomForestClassifier()



# Note: this parameter space was iteratively shirinked after RandomSearchCV

n_estimators = [x for x in range(101, 250, 50)]

criterion = ["entropy"]

max_depth = [x for x in range(19, 30, 2)]

min_samples_split = [x for x in range(2, 11, 2)]

min_samples_leaf = [x for x in range(2, 11, 2)]

min_weight_fraction_leaf = list(np.arange(start=0.01, stop=0.5, step=0.1))

max_features = list(np.arange(start=0.1, stop=1.1, step=0.1))

min_impurity_decrease = list(np.arange(start=0.0, stop=0.5, step=0.1))

bootstrap = [True]

class_weight = [None]

max_samples = list(np.arange(start=0.1, stop=1, step=0.1))

ccp_alpha = list(np.arange(start=0.001, stop=0.02, step=0.05))



param_distributions = {"n_estimators":n_estimators,

                       "criterion":criterion,

                       "max_depth":max_depth,

                       "min_samples_split":min_samples_split,

#                        "min_samples_leaf":min_samples_leaf,

#                        "min_weight_fraction_leaf":min_weight_fraction_leaf,

#                        "max_features":max_features,

#                        "min_impurity_decrease":min_impurity_decrease,

                       "bootstrap":bootstrap,

                       "class_weight":class_weight,

#                        "max_samples":max_samples,

#                        "ccp_alpha":ccp_alpha

                      }



rf_best_rand = GridSearchCV(estimator=rf, 

                            param_grid=param_distributions,

                            cv=2,

                            verbose=1,

                            scoring=make_scorer(f1_score, average="weighted"))

rf_best_rand.fit(X, y)



pprint(rf_best_rand.best_params_)

pprint(rf_best_rand.best_score_)
# Last model obtained after multiple runs of the cell above

final_model = rf_best_rand

final_model.fit(X, y)
model = rf_best_rand

test = pd.read_csv("../input/data-mining-assignment-2/test.csv", index_col="ID")

test = process_data(test, train_data=False, cols_to_keep=cols_to_keep, scaler=scaler)

test = test.reindex(X.columns, axis=1)

pd.DataFrame(data={"ID":test.index + 700, "Class":model.predict(test)}).to_csv("submission.csv", index=False)
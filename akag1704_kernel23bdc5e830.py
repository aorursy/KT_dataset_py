import numpy as np 

import pandas as pd 
df = pd.read_csv("../input/exl-train/EXL_EQ_2020_Train_datasets.csv")

df.head(2)
target = df['self_service_platform'].replace({'Desktop':1, 'Mobile App':2, 'Mobile Web': 3, 'STB': 4})

features = df.drop(['cust_id','self_service_platform','var30'] , axis = 1)

features.head(3) 
features.isnull().any()
features.var24.fillna(0, inplace = True)

features.var24.isnull().any()
features.var24 = features.var24.astype('int64')

features.dtypes
for col in features.columns:

    print(col, " \n", features[col].value_counts(), '\n\n')
features_final = pd.get_dummies(features, columns = ['var33','var34','var35','var36','var37','var38','var39','var40'],drop_first = True)
features_final.columns

my_columns = features_final.columns

my_columns
features_final.isnull().any()
features_final.dtypes
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(features_final, target, test_size = 0.1, random_state = 42)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=15, random_state = 42).fit(x_train,y_train)
model.score(x_train, y_train)
model.score(x_test, y_test)
from sklearn.ensemble import GradientBoostingClassifier

xgb = GradientBoostingClassifier()
xgb.fit(x_train, y_train)



print(xgb.score(x_train, y_train))

print(xgb.score(x_test, y_test))
importances = model.feature_importances_



std = np.std([tree.feature_importances_ for tree in model.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(features_final.shape[1]):

    print("%d. %s %d (%f)" % (f + 1, list[indices[f]], indices[f], importances[indices[f]]))

from sklearn.base import clone



# Reduce the feature space

X_train_reduced = features_final[features_final.columns.values[(np.argsort(importances)[::-1])[:10]]]

X_test_reduced = x_test_final[x_test_final.columns.values[(np.argsort(importances)[::-1])[:10]]]

X_train_reduced.shape

# Train on the "best" model found from grid search earlier

#clf = (clone(best_clf)).fit(X_train_reduced, y_train)



# Make new predictions

model1 = RandomForestClassifier(random_state = 42).fit(X_train_reduced, target) 

reduced_predictions = model1.predict(X_train_reduced)



from sklearn.metrics import accuracy_score



# Report scores from the final model using both versions of data

print("Final Model trained on full data\n------")

print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))

print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))

print("\nFinal Model trained on reduced data\n------")

print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))

print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))
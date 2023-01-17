import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 



%matplotlib inline
df = pd.read_csv("../input/minor-project-2020/train.csv")
# df.head()
# df.info(verbose=True, null_counts=True)
df.describe()
#prints correlation matrix

# mask = np.zeros_like(df.corr(), dtype=np.bool)

# mask[np.triu_indices_from(mask)] = True

# sns.set_style('whitegrid')

# plt.subplots(figsize = (80,80))

# sns.heatmap(df.corr(), 

#             annot=True,

#             mask = mask,

#             cmap = 'RdBu', ## in order to reverse the bar replace "RdBu" with "RdBu_r"

#             linewidths=.9, 

#             linecolor='white',

#             fmt='.2g',

#             center = 0,

#             square=True)
X=df

y=df[['target']]



X.drop('target',axis=1,inplace=True)



X.drop('id',axis=1,inplace=True)



# X.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=121)
print(len(X_train), len(X_test))
from sklearn.preprocessing import StandardScaler



scalar = StandardScaler()

scaled_X_train = scalar.fit_transform(X_train)

scaled_X_test = scalar.transform(X_test)
# from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text



# dt = DecisionTreeClassifier(class_weight="balanced")



# dt.fit(scaled_X_train, y_train)



# y_pred = dt.predict(scaled_X_test)



# print("Accuracy is : {}".format(dt.score(scaled_X_test, y_test)))



# column_names = X_train.columns

# feature_importances = pd.DataFrame(dt.feature_importances_,

#                                    index = column_names,

#                                     columns=['importance'])

# feature_importances.sort_values(by='importance', ascending=False).head(10)



# # print(export_text(dt))



# from sklearn.metrics import classification_report, confusion_matrix



# print("Confusion Matrix: ")



# print(confusion_matrix(y_test, y_pred))
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.model_selection import GridSearchCV
# parameters = {'criterion': ("gini", "entropy"), 'max_depth': (50,300)}



# rf_cv = RandomForestClassifier()



# clf = GridSearchCV(rf_cv, parameters, verbose=1)



# clf.fit(scaled_X_train, y_train)
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(class_weight='balanced')

# rf.fit(scaled_X_train, y_train)

# print(rf.score(scaled_X_test, y_test.values.ravel()))
# plot_confusion_matrix(rf, scaled_X_test, y_test, cmap = plt.cm.Blues)
# from sklearn.linear_model import LogisticRegression

# lr = LogisticRegression(random_state = 0,class_weight='balanced') 

# lr.fit(scaled_X_train, y_train) 

from sklearn.linear_model import LogisticRegressionCV



clf = LogisticRegressionCV(cv=10,max_iter=1000,class_weight='balanced')

#clf = RandomForestClassifier(max_depth=10, random_state=0,class_weight={0:1,1:1},n_jobs=-1)

clf.fit(scaled_X_train, y_train)
model=clf
test = pd.read_csv("../input/minor-project-2020/test.csv")

# test.head()

test_no_id = test.drop('id', axis=1)



test_predictions = model.predict(test_no_id)



# test_predictions



ID = test['id']

submission_df_1 = pd.DataFrame({

                  "id": ID, 

                  "target": test_predictions})



# submission_df_1



submission_df_1.to_csv('submission_7.csv', index=False)
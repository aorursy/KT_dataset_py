# This notebook anaylzes the dataset provided by a hiring firm in india, in order to help solve the firms attendance problem for interviews.
# Through several machine learning methods, we can say that around 70% of applicants will attend their interview.
# In addition, if we make the assumption that all applicants applying to routine/non-challenge jobs will not appear
# and applicants with challenging jobs will appears, we would be around 64% right
import numpy as np # linear algebra
np.random.seed(0)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Interview.csv')
data = data.drop(['Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27'], axis=1)
data.info()
clean_na = data.dropna()
print(clean_na)
from collections import defaultdict
column_name_dict = defaultdict(str)
for column in clean_na.columns:
    column_name_dict[column] = ''
new_names = ['Date', 'Client', 'Industry', 'Location', 'Position', 'Skillset', 'Interview_Type', 'ID', 'Gender', 'Candidate_Loc', 'Job_Location', 'Venue', 'Native_Loc', 'Permission', 'Hope', '3_hour_call', 'Alt_Number', 'Resume_Printout', 'Clarify_Venue', 'Share_Letter', 'Expected', 'Attendance', 'Marital_Status']
for idx,key in enumerate(column_name_dict):
    column_name_dict[key] = new_names[idx]

clean_na = clean_na.rename(columns=column_name_dict)
clean_na = clean_na.apply(lambda x: x.astype(str).str.upper() if isinstance(x, str) else x)
clean_na = clean_na.apply(lambda x: x.astype(str).str.replace(" ", "") if isinstance(x, str) else x)
clean_na.info()
print(clean_na)
columns = ['Permission', 'Hope', '3_hour_call', 'Share_Letter', 'Clarify_Venue', 'Resume_Printout', 'Alt_Number', 'Expected', 'Attendance', 'Marital_Status', 'Gender', 'Interview_Type']
def changeData(clean, arr=[]):
    for column in arr:
        clean[column] = clean[column].map({'yes': 1, 'Yes':1, 'YES': 1, 'Married': 1, 'female': 1, 'Scheduled': 1, 'no': 0, 'No': 0, 'NO': 0, "Not yet": 0, 'Havent Checked':0, 'Uncertain':0, 'Single':0, 'Male': 0, 'Scheduled Walkin': 0, 'Scheduled Walk In': 0})
        
    clean = clean.dropna()
changeData(clean_na, columns)
print(clean_na)
columns_to_drop = ["Date", "ID", "Expected"]
clean_df = clean_na.drop(columns_to_drop, axis=1)
df = pd.get_dummies(clean_df)
df = df.dropna()
# labels = df["Attendance"]
print(df)
df.dropna()
labels = df["Attendance"]
df = df.drop("Attendance", axis=1)
print(labels)
print(labels)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(df, labels)
clf = DecisionTreeClassifier()
# Call clf.fit(), and pass in X_train and y_train as parameters
clf.fit(X_train, y_train)

# Use the .predict() method to have our model create predictions on the X_test variable
test_pred = clf.predict(X_test)

# Finally, pass in test_pred and y_test
print(accuracy_score(test_pred, y_test))

clf_depth_1 = DecisionTreeClassifier(max_depth=1)
clf_depth_1.fit(X_train, y_train)
clf_depth_1_pred = clf_depth_1.predict(X_test)
print(accuracy_score(clf_depth_1_pred, y_test))
import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(clf_depth_1, out_file=None, feature_names=list(df.columns.values), class_names=["Absent", "Non-Absent"])
graph = graphviz.Source(dot_data)  
graph
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
#Fit the scaler to iris.data
scaler.fit(df)
# call scaler.transform() on iris.data and store the result in scaled_X
scaled_X = scaler.transform(df)

# Store the labels contained in iris.targets below
labels2 = labels

# Create a PCA() object
pca = PCA()

#Fit the pca object to scaled_X
pca.fit(scaled_X)

# Call pca.transform() on scaled_X and store the results below
X_with_pca = pca.transform(scaled_X)

# Enumerate through pca.explained_variance_ratio_ to see the amount of variance captured by each Principal Component
for ind, var in enumerate(pca.explained_variance_ratio_):
    print("Explained Variance for Principal Component {}: {}".format(ind, var))
X_with_pca2 = pd.DataFrame(data=X_with_pca)
X_with_pca2.drop(columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63], axis=1, inplace=True)
pca_X_train, pca_X_test, pca_y_train, pca_y_test = train_test_split(X_with_pca2, labels)
reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(scaled_X, labels)
clf = DecisionTreeClassifier()
clf_for_pca = DecisionTreeClassifier()

# Fit both models on the appropriate datasets
clf.fit(reg_X_train, reg_y_train)
clf_for_pca.fit(pca_X_train, pca_y_train)


# Use each fitted model to make predictions on the appropriate test sets
reg_pred = clf.predict(reg_X_test)
pca_pred = clf_for_pca.predict(pca_X_test)

print("Accuracy for regular model: {}".format(accuracy_score(reg_y_test, reg_pred)))
print("Accuracy for model with PCA: {}".format(accuracy_score(pca_y_test, pca_pred)))
from sklearn import tree
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=list(df.columns.values), class_names=["Absent", "Non-Absent"])
graph = graphviz.Source(dot_data)  
graph
import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(clf_for_pca, out_file=None, class_names=["Absent", "Non-Absent"])
graph = graphviz.Source(dot_data)  
graph
from sklearn.ensemble import RandomForestClassifier as RFC

clf = RFC()
clf2 = RFC()

clf.fit(pca_X_train, pca_y_train)
predict = clf.predict(pca_X_test)


clf2.fit(reg_X_train, reg_y_train)
predict2 = clf2.predict(reg_X_test)


print("Accuracy for regular model: {}".format(accuracy_score(predict2, reg_y_test)))
print("Accuracy for model with PCA: {}".format(accuracy_score(predict, pca_y_test)))



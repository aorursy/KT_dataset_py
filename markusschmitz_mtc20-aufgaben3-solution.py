from IPython.display import Image
from IPython.core.display import HTML 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns # Library for Plotting
sns.set # make plots look nicer
sns.set_palette("husl")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
def plot_results(X_test, y_test, clf):
    transformer = KernelPCA(n_components=2, kernel='linear')
    X_transformed = transformer.fit_transform(X_test)
    X_transformed = pd.DataFrame(X_transformed)
    X_transformed["pred"] = clf.predict(X_test)
    X_transformed["Outcome"] = y_test.values
    X_transformed["correct"] = X_transformed["pred"] == X_transformed["Outcome"] 
    score = f1_score(X_transformed["Outcome"], X_transformed["pred"], average='macro')
    print("You have achieved an f1 score of: ", score)
    print("This is your confusion matrix:")
    plot_confusion_matrix(clf, X_test, y_test, normalize = "true")
    print("Here is the plot of your prediction:")
    fig, ax = plt.subplots(figsize=(25, 10)) # Size of Plot
    ax = sns.scatterplot(x=0, y=1, data=X_transformed, hue="pred", style = "Outcome") # Makes a LinePlot
    plt.rcParams.update({'font.size': 32})
    plt.show()

    
    
    
# Read in Data
data = pd.read_csv("/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv")
# Code the gender numerically for later use
data["Gender"] = np.where(data['Gender'] == "Female", 1, 0)
# code the target in binary
data["Dataset"] = np.where(data['Dataset'] == 1, 1, 0)
# rename columns for better readability
data = data.rename(columns={"Dataset": "Outcome", "Gender" : "Female"})
data = data.dropna()

data.head()
data.Outcome.value_counts()
# Describe the data
data.describe()
# Plot a correlation matrix
# calculate the correlations
corr = data.corr()
# plot the matrix as heatmap
f, ax = plt.subplots(figsize=(25, 25))
sns.heatmap(corr, annot=True, linewidths=.5, ax=ax)
plt.show()
# Plot two important features Kombinations and highlight the data with diabetes
fig, ax = plt.subplots(figsize=(15, 10)) # Size of Plot
ax = sns.scatterplot(x="Direct_Bilirubin", y="Alkaline_Phosphotase", hue = "Outcome", data=data) # Makes a Scatterplot
plt.show()
fig, ax = plt.subplots(figsize=(15, 10)) # Size of Plot
ax = sns.scatterplot(x="Direct_Bilirubin", y="Albumin", hue = "Outcome", data=data) # Makes a Scatterplot
plt.show()
# Define a threshold
threshold = 4

# Add a new column that is 1 if the patients Glucose is above the Threshold and 0 otherwise
data["thresh"] = np.where(data['Direct_Bilirubin'] > threshold, 1, 0)

# Add another column that describes if your threshold prediction is the same as Outcome
data["correct"] = data["thresh"] == data["Outcome"]

# Read out how many 
data["correct"].value_counts()
# Read in Data again:
# Read in Data
data = pd.read_csv("/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv")
data["Gender"] = np.where(data['Gender'] == "Female", 1, 0)
data["Dataset"] = np.where(data['Dataset'] == 1, 1, 0)
data = data.rename(columns={"Dataset": "Outcome", "Gender" : "Female"})
data = data.dropna()

# Split the data in train and test set
X_train, X_test, y_train, y_test = train_test_split(data.drop(["Outcome"], axis = 1), data["Outcome"], test_size=.25, random_state=42, stratify=data['Outcome'])

# Print size of each Dataset
print(len(X_train), len(X_test))
# Set clf to the Decision Tree Clasiifier
clf = DecisionTreeClassifier(random_state=42)

# fit the classifier with your train data
clf.fit(X_train, y_train)

# plot your test data with "plot_results"
plot_results(X_test, y_test, clf)
# Plot the Tree layout
fig, ax = plt.subplots(figsize=(35, 20))
tree.plot_tree(clf, filled=True, ax = ax, fontsize = 15)
plt.show()
# train a KNN classifier and evaluate results
clf = KNeighborsClassifier(n_neighbors = 6)
clf.fit(X_train, y_train)
plot_results(X_test, y_test, clf)
# Select importan features
selected_features = ['Age','Direct_Bilirubin','Total_Protiens','Albumin', 'Female']

# Generate new train and test data
X_train, X_test, y_train, y_test = train_test_split(data[selected_features], data["Outcome"], test_size=.25, random_state=42, stratify=data['Outcome'])

# run your algorithm
clf = KNeighborsClassifier(n_neighbors = 6)
clf.fit(X_train, y_train)
plot_results(X_test, y_test, clf)
# example for logistic Regression
clf = LogisticRegression()
clf.fit(X_train, y_train)
plot_results(X_test, y_test, clf)
# example for Gaussian Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)
plot_results(X_test, y_test, clf)
# example for Random Forest
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
plot_results(X_test, y_test, clf)
# example for adaboost
clf = AdaBoostClassifier(random_state=0)
clf.fit(X_train, y_train)
plot_results(X_test, y_test, clf)
# example for Support Vector Classifier
clf = SVC(random_state=0, kernel="rbf")
clf.fit(X_train, y_train)
plot_results(X_test, y_test, clf)
# example for Multilayer Perceptron (Neural Net)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 4), random_state=1)
clf.fit(X_train, y_train)
plot_results(X_test, y_test, clf)
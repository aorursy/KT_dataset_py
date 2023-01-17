# Based on Kaggle template and built with help from many other PIMA Kaggle forks:
#   kaggle/python docker image: https://github.com/kaggle/docker-python

# Load foundation packages
import numpy as np     # linear algebra
import pandas as pd    # data processing, CSV file I/O (e.g. pd.read_csv)
import os              # operating system
import matplotlib.pyplot as plt  # plotting
import seaborn as sns            # plotting

# Load Sci-Kit basic packages
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Load Sci-Kit models
from sklearn import svm  
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Default Kaggle file location is the "../input/" directory.
# Any results you write to the current directory are saved as output.
print(os.listdir("../input"))  # list files in the input directory
# Read the PIMA Indian Diabetes Data, Rename cols
df = pd.read_csv('../input/diabetes.csv')
#type(df)           # pandas.core.frame.DataFrame
#df.isnull().sum()  # Any NULL entries?  No!
#df.columns         # Get col names
df.columns = ['Preg', 'Glucose', 'D_BP', 'Skin', 'Insulin', 'BMI', 'DPF', 'Age', 'Outcome']
df.shape
df.head()
df.describe().round(2)
# Plot good size histograms, .shape suppresses return of memory addr
df.hist(figsize=(10,8)).shape
# Plot all pairwise scatterplots, stacked histo on the diagonal
#sns.pairplot(data=df, hue="Outcome", dropna=True)
sns.pairplot(data=df, hue="Outcome", # palette="husl",
             vars=['Preg', 'Glucose', 'D_BP', 'Skin', 'Insulin', 'BMI', 'DPF', 'Age'])
#plt.savefig("PIMA_Diab_pair_plot.png") # Save figure to default directory
plt.show()
# Count non-zeros per col. Zeros seem to represent both real and missing values
df.astype(bool).sum(axis=0)
# Drop rows with sparse zeros
df2 = df.drop(df[(df.Glucose == 0) | (df.D_BP == 0) | (df.BMI == 0)].index)

# Drop cols with many zeros. (Not using the [inplace=True] option)
#clean = df2.drop(['Skin', 'Insulin'], axis=1)  # Keith version
clean = df2.drop(['Skin'],            axis=1)  # Tara version

# Split data into training and test sets
train, test = train_test_split(clean, test_size=0.2, random_state=0, stratify=clean['Outcome']) # stratify the outcome
train.shape
# Break out data into features-x and outcome-y
train_Y = train['Outcome']
train_X = train[train.columns[:-1]]
train_X = (train_X-train_X.mean()) / train_X.std()  # Normalize data

test_Y = test['Outcome']
test_X = test[test.columns[:-1]]
test_X = (test_X-test_X.mean()) / test_X.std()  # Normalize data

test_X.head()
# Try a model
model = LogisticRegression()
model.fit(train_X, train_Y)
prediction = model.predict(test_X)
print('Logistic Regr Accuracy: %0.2f' % metrics.accuracy_score(test_Y, prediction))
print('Logistic Regr AUC: %0.2f' % metrics.roc_auc_score(test_Y, prediction))
# Set up multiple models, aka classifiers
names = ["K Neighbors", 
         "Logistic Regr",
         "Decision Tree", 
         "Random Forest", 
         #"Neural Net", 
         "Naive Bayes", 
         "Linear SVM", 
         "RBF SVM", 
         "Gaussian Proc", 
         "AdaBoost", 
         "QDA"
]

classifiers = [KNeighborsClassifier(), 
               LogisticRegression(),
               DecisionTreeClassifier(), 
               RandomForestClassifier(),
               #MLPClassifier(), 
               GaussianNB(), 
               svm.SVC(kernel="linear"), 
               svm.SVC(kernel="rbf"),
               GaussianProcessClassifier(), 
               AdaBoostClassifier(), 
               QuadraticDiscriminantAnalysis()
]

# Run classifiers, record their accuracy or ROC AUC scores
results = {}
for name, clf in zip(names, classifiers):
    #scores = cross_val_score(clf, train_X, train_Y, cv=10, scoring='accuracy')
    scores = cross_val_score(clf, train_X, train_Y, cv=10, scoring='roc_auc' )    
    results[name] = scores
    
# Print score summaries
for name, scores in results.items():
    print("%20s | %0.1f%% (+/- %0.1f%%)" % (name, 100*scores.mean(), 100*scores.std()*2/(len(scores)**0.5)))

# Importing fundamental datasets
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
import itertools #iterators for efficient looping
import seaborn as sns # for quickly plotting heatmaps
dataset = pd.read_csv("../input/train.csv", index_col = "PassengerId") # reading input
dataset.sample(5) # printing the first five rows to get an impression of the data
# printing datatypes and missing values
overview = pd.concat([dataset.dtypes, dataset.isnull().sum()], axis=1)
overview.columns = ["type", "NAs"]
overview
# Creating a new feature: If a family member survived: 1, else: 0
dataset["Family"] = dataset["Name"].str.split(',').str.get(0) # extract family name
dataset["FamSurv"] = 0
for index, row in dataset.iterrows():
    for index2, row2 in dataset.iterrows():
        if not index2 == index:
            if row["Family"] == row2["Family"] and (row["Survived"] == 1 or row2["Survived"] == 1):
                dataset.loc[index, "FamSurv"] = 1

lstTextCategories = ["Sex", "Ticket", "Embarked", "Family"] # list of categorial variables
# recode cabin: 1 if there is an entry, 0 else
dataset["Cabin"] = dataset["Cabin"].where(dataset["Cabin"].isnull(), 1).fillna(0)
dataset.sample(5)
for cat in lstTextCategories: # replacing categorial words with numbers
    dataset[cat] = pd.Categorical(dataset[cat])
    dataset[cat] = dataset[cat].cat.codes
    
corr = dataset.corr() # calculating correlation matrix
plt.figure(figsize=(10,8)) # defining plot size
sns.heatmap(corr, # creating the heatmap plot of the correlation matrix
            xticklabels = corr.columns.values,
            yticklabels = corr.columns.values, 
            annot = True, # for showing the values
            vmin=-1, vmax=1, center=0) # min and max at [-1, 1]
plt.show() # actually plotting
# create age-classes
dataset.hist(column='Age')
dataset['ageClass'] = dataset['Age']/15
dataset['ageClass'] = dataset['ageClass'].apply(np.floor)
dataset.hist(column='ageClass')
strTarget = "Survived" # defining target variable
lstExclude = [strTarget, "Name", "Ticket", "Family", "Age"] # variables to be excluded from the feature matrix
lstCategorials = ["Sex", "Pclass", "Cabin", "SibSp", "Parch"] # categorial variables

lstInclude = [column for column in dataset.columns if column not in lstExclude] # variables to be included

y = dataset[strTarget] # extracting target
X = dataset.loc[:, lstInclude] # extracting features
X.describe() # sumarizing variables - note some outliers in the "fare" column
# creating dummy variables
from sklearn.preprocessing import StandardScaler

X = X.fillna(X.mean()) # Imputing NA with average
X = pd.get_dummies(X, drop_first=True, columns=lstCategorials) # creating dummy variables
heads = X.columns # finding (new) column heads

X = pd.DataFrame(StandardScaler().fit_transform(X), columns=heads) # Feature scaling
X.describe() # sumarizing variables
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# importing the individual classification models from scikit learn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# definig models and parameter grids which will be tested for best results
dicModels = {"Logistic Regression": {"classifier":LogisticRegression(),
                                     "param_grid": {"solver": ["lbfgs"]}},
             "KN Classifier": {"classifier":KNeighborsClassifier(2), 
                               "param_grid":{"n_neighbors": [4, 6, 8], 
                                             "weights": ["uniform", "distance"]}},
             "Naive Bays": {"classifier":GaussianNB()}, 
             "SVM": {"classifier":SVC(random_state=0), 
                     "param_grid":{"kernel" : ["rbf", "linear", "poly"],
                                   "C": [0.01, 0.1, 1, 10, 100], 
                                   "gamma": [0.001, 0.01]}},
             "Random Forest": {"classifier":RandomForestClassifier(max_depth=None, min_samples_split=2, random_state=0),
                               "param_grid":{"n_estimators": [150], 
                                             "max_features": ["auto", "sqrt", "log2"], 
                                             "criterion": ["entropy", "gini"], 
                                             "min_samples_split": [2, 4]}},
             "Gradient Boosting": {"classifier":GradientBoostingClassifier(max_depth=None, min_samples_split=2, random_state=0),
                                   "param_grid":{"n_estimators": [150], 
                                                 "learning_rate": [0.05, 0.1, 0.5], 
                                                 "loss": ["deviance", "exponential"], 
                                                 "min_samples_split": [2, 4]}}}
dicResults = {} # result dictionary
dicCMs = {} # confusion matrix dictionary
for name, model in dicModels.items(): # loop over all modes
    dicCMs[name] = np.array([[0,0],[0,0]]) # creating empty confusion matrix
    print(name) # printing name sothat we know where the algorithm stands
    for split in range(1): # loop over 1 random split of the dataset 
        # creating split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = split)
        
        if "param_grid" in model: # if a parameter grid is defined: find optimal configuration
            clf = GridSearchCV(estimator=model["classifier"], 
                               param_grid=model["param_grid"], 
                               iid=True,
                               cv=5)
            clf.fit(X_train, y_train)
            model["classifier"] = clf.best_estimator_ # takes bast parameters to the model
                
        model["classifier"].fit(X_train, y_train) # fit model to training data

        y_pred = model["classifier"].predict(X_test) # Predicting the Test set results

        dicCMs[name] += confusion_matrix(y_test, y_pred) # Making the Confusion Matrix
    
    dicResults[name] = {"correct":dicCMs[name][0][0]+dicCMs[name][1][1], # creating result table
                        "false":dicCMs[name][0][1]+dicCMs[name][1][0],
                        "correct positives":dicCMs[name][0][0], 
                        "correct negatives":dicCMs[name][1][1],
                        "false positives":dicCMs[name][0][1],
                        "false negatives":dicCMs[name][1][0],
                        "parameters": model["classifier"].get_params()}

# bringing results to a datafram (it's just more beautiful) and sorting it by accuracy
results = pd.DataFrame(dicResults).T.sort_values(["correct", "correct positives", "correct negatives"], 
                                                 ascending=[0, 0, 0])
# plotting result tables with bars
results.style.bar(subset=["correct", "correct positives", "correct negatives"], 
                  color="#2876dd", 
                  align="mid")
for name, cm in dicCMs.items(): # looping over matrices
    cm = cm / cm.sum(axis=1)[:,None] # making percentages of absolute values
    plt.figure() # creating new figure
    ax = plt.axes() 
    sns.heatmap(cm, # creating heat map
                xticklabels = ["pred. positive", "pred. negative"],
                yticklabels = ["true positive", "true negative"], 
                annot = True, # plotting values
                cmap = sns.color_palette("Blues"),
                vmin = 0, vmax = 1)
    for number in ax.texts: 
        number.set_text("{0:.0f}%".format(float(number.get_text())*100)) # convert values [0, 1] to [0%, 100&]
    ax.set_title(name)
    plt.show() #plotting
from sklearn.preprocessing import StandardScaler
train = pd.read_csv("../input/train.csv", index_col="PassengerId")
test = pd.read_csv("../input/test.csv", index_col="PassengerId")
dataset = train.append(test)
dataset["Family"] = dataset["Name"].str.split(',').str.get(0) # extract family name
lstTextCategories = ["Sex", "Ticket", "Cabin", "Embarked", "Family"] # list of categorial variables
dataset.describe()
# Creating a new feature: If a family member survived: 1, else: 0
dataset["Family"] = dataset["Name"].str.split(',').str.get(0) # extract family name
dataset["FamSurv"] = 0
for index, row in dataset.iterrows():
    for index2, row2 in dataset.iterrows():
        if not index2 == index:
            if row["Family"] == row2["Family"] and (row["Survived"] == 1 or row2["Survived"] == 1):
                dataset.loc[index, "FamSurv"] = 1

lstTextCategories = ["Sex", "Ticket", "Cabin", "Embarked", "Family"] # list of categorial variables
# recode cabin: 1 if there is an entry, 0 else
dataset["Cabin"] = dataset["Cabin"].where(dataset["Cabin"].isnull(), 1).fillna(0)
dataset.sample(5)
for cat in lstTextCategories:
    dataset[cat] = pd.Categorical(dataset[cat])
    dataset[cat] = dataset[cat].cat.codes

dataset['ageClass'] = dataset['Age']/15
dataset['ageClass'] = dataset['ageClass'].apply(np.floor)
    
strTarget = "Survived" # defining target variable
lstExclude = [strTarget, "Name", "Ticket", "Family", "Age"] # variables to be excluded from the feature matrix
lstCategorials = ["Sex", "Pclass", "Cabin", "SibSp", "Parch"] # categorial variables
lstInclude = [column for column in dataset.columns if column not in lstExclude] # variables to be included

y = dataset[strTarget]
X = dataset.loc[:, lstInclude]

X = X.fillna(X.mean())
X = pd.get_dummies(X, drop_first=True, columns=lstCategorials)
X = pd.get_dummies(X, drop_first=True)
heads = X.columns
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=heads)

X_train, X_test, y_train = X.loc[:y.notnull().sum()-1,:], X.loc[y.notnull().sum():,:], y[y.notnull()]

print(len(X_train), len(y_train))

bestModel = dicModels[results.index[0]]
if "param_grid" in bestModel:
    clf = GridSearchCV(estimator=bestModel["classifier"], 
                       param_grid=bestModel["param_grid"], 
                       cv=5)
    clf.fit(X_train, y_train)
    bestModel["classifier"] = clf.best_estimator_
    
bestModel["classifier"].fit(X_train, y_train)
y_pred = pd.DataFrame(bestModel["classifier"].predict(X_test), 
                      index=test.index.values, 
                      columns=["Survived"]).astype(int)
y_pred.index.names=["PassengerId"]
y_pred.to_csv("submission14.csv")
print(results.index[0], bestModel["classifier"].get_params())
y_pred.head(10)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('dark_background')
sns.set_style("whitegrid")

from IPython.display import display

%matplotlib inline
from sklearn import datasets
iris = datasets.load_iris()


iris = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                     columns= ["sepalLength", "sepalWidth", "petalLength", "petalWidth", "target"])

# Target Descriptor

# 0 -> Sentosa
# 1 -> versicolor
# 2 -> virginica

def getTargetName(row):
    target = row["target"]
    if int(target) == 0:
        return "Sentosa"
    elif int(target) == 1:
        return "Versicolor"
    else:
        return "virginica"
    
iris["target_name"] = iris.apply(getTargetName, axis = 1)

display(iris.sample(n = 5))
iris["target_name"].value_counts().plot(kind = 'pie', figsize = (5, 5), explode = [0, 0.1, 0.1], autopct = '%1.1f%%', shadow = True)
plt.ylabel("")
plt.show()
# Individual Feature Analysis.
f, ax = plt.subplots(1, 2 , figsize=(20, 7))

sns.factorplot(x = "target_name", y = "sepalLength", data = iris, ax = ax[0])
ax[0].set_xlabel("Species")
ax[0].set_ylabel("Mean Sepal Length")

sns.violinplot(x = "target_name", y = "sepalLength", data = iris, ax = ax[1])
ax[1].set_xlabel("Species")
ax[1].set_ylabel("Mean Sepal Length")

plt.show()
f, ax = plt.subplots(1, 2 , figsize=(20,7 ))

sns.factorplot(x = "target_name", y = "sepalWidth", data = iris, ax = ax[0])
ax[0].set_xlabel("Species")
ax[0].set_ylabel("Mean Sepal Width")

sns.violinplot(x = "target_name", y = "sepalWidth", data = iris, ax = ax[1])
ax[1].set_xlabel("Species")
ax[1].set_ylabel("Mean Sepal Width")

plt.show()
sns.lmplot(x = "sepalLength", y = "sepalWidth", hue = "target_name", data = iris, fit_reg = True)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()
f,ax=plt.subplots(1,2,figsize=(20, 7))

sns.factorplot(x = "target_name", y = "petalLength", data = iris, ax = ax[0])
ax[0].set_xlabel("Species")
ax[0].set_ylabel("Mean Petal Length")

sns.violinplot(x = "target_name", y = "petalLength", data = iris, ax = ax[1])
ax[1].set_xlabel("Species")
ax[1].set_ylabel("Mean Petal Length")

plt.show()
f,ax=plt.subplots(1,2,figsize=(20, 7))

sns.factorplot(x = "target_name", y = "petalWidth", data = iris, ax = ax[0])
ax[0].set_xlabel("Species")
ax[0].set_ylabel("Mean Petal Width")

sns.violinplot(x = "target_name", y = "petalWidth", data = iris, ax = ax[1])
ax[1].set_xlabel("Species")
ax[1].set_ylabel("Mean Petal Width")

plt.show()
pd.scatter_matrix(iris, figsize = (20, 10))
plt.show()
corr = iris.corr()
sns.heatmap(corr, annot = True, cmap = 'RdYlGn', linewidths = 0.2)
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.show()
sns.lmplot(x = "petalLength", y = "petalWidth", hue = "target_name", data = iris, fit_reg = True)
plt.show()
sns.pairplot(iris, hue = "target_name", size = 3)
plt.show()
# Lets build a decision tree and find out the features it considers as important
features = iris.drop(labels = ["target", "target_name"], axis = 1)
target = iris["target"]

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state = 0)

classifier.fit(features, target)

pd.Series(classifier.feature_importances_, features.columns).sort_values(ascending = True).plot.barh(width=0.8)
plt.show()
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2)
results = pd.DataFrame(columns = ['Model_Name', 'Model', 'Params', 'Test_Score', 'CV_Mean', 'CV_Std'])
def save(grid, modelName):
    global results
    cv_scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv = 10, scoring = 'accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    test_score = grid.score(X_test, y_test)
    
    print("Best model parameter are\n", grid.best_estimator_)
    print("Saving model {}\n".format(modelName))
    print("Mean Cross validation score is {} with a Standard deviation of {}\n".format(cv_mean, cv_std))
    print("Test Score for the model is {}\n".format(test_score))
    
    cm = confusion_matrix(y_test, grid.best_estimator_.predict(X_test))
    
    cm_df = pd.DataFrame(cm, index = ["Sentosa", "versicolor", "Virginica"], columns = ["Sentosa", "versicolor", "Virginica"])
    sns.heatmap(cm_df, annot = True)
    plt.show()
    
    results = results.append({'Model_Name' : modelName, 'Model' : grid.best_estimator_, 'Params' : grid.best_params_, 'Test_Score' : test_score, 'CV_Mean' : cv_mean, 'CV_Std' : cv_std}
                                       , ignore_index=True)
def doGridSearch(classifier, params):
    score_fn = make_scorer(accuracy_score)
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
    grid = GridSearchCV(classifier, params, scoring = score_fn, cv = cv)
    grid = grid.fit(X_train, y_train)
    return grid    
from sklearn.svm import SVC
svc = SVC(random_state = 0)
params = {"C" : [0.1, 1, 1.1, 1.2], "gamma" : [0.01, 0.02, 0.03, 0.04, 0.08, 0.1, 1], 
          "kernel" : ["linear", "poly", "rbf", "sigmoid"]}

grid = doGridSearch(svc, params)
save(grid, 'SVC')
from sklearn.linear_model import LogisticRegression

logisticModel = LogisticRegression(random_state = 0)
params = {}

grid = doGridSearch(logisticModel, params)
save(grid, 'LogisticRegression')
from sklearn.naive_bayes import GaussianNB

naiveModel = GaussianNB()
params = {}

grid = doGridSearch(naiveModel, params)
save(grid, 'NaiveBayes')
display(results.sort_values(by = 'Test_Score', ascending = False))
sns.barplot(x = "Model_Name", y = "Test_Score", data = results)
plt.xlabel("Model")
plt.ylabel("Test Score")
plt.yticks(np.arange(0, 1, 0.05))
fig = plt.gcf()
fig.set_size_inches(8, 8)
plt.show()

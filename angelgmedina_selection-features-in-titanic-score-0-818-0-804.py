import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Split
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold    


# preprocessiong
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel, RFECV

# machine learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

#cluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# import files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# preview the data
train.head()
IDtest = test["PassengerId"]
train_len = len(train)

# preview the data
total =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
total.info()
# Embarked (two missing values)
total[total[["Embarked"]].isnull().any(axis=1)]
embark_perc = total[["Embarked", "Pclass"]].groupby(["Embarked"]).count()
a = total.groupby(["Embarked", "Pclass"]).size().reset_index(name='count')
a
b = total[["Embarked", "Pclass", "Fare"]].groupby(["Embarked", "Pclass"]).mean()
print(b)
c = total[["Embarked", "Pclass", "Fare"]].groupby(["Embarked", "Pclass"]).count()
print(c)
sns.set(style="whitegrid")
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=total, palette="husl")
plt.ylim(0,150)
plt.show()
sns.countplot(x="Embarked", data=total, hue="Pclass", palette="Greens_d");
total["Embarked"] = total["Embarked"].fillna("S")
g = sns.catplot(x="Embarked", y="Survived",  data=train,  hue="Pclass", height=5, kind="bar", 
                   palette="Greens_d")
g.despine(left=True)
g = g.set_ylabels("survival probability")
total = pd.get_dummies(total, columns = ["Embarked"], prefix="Em")
total[total[["Fare"]].isnull().any(axis=1)]
from scipy import stats
total[["Pclass", "Fare"]].groupby(['Pclass']).agg(lambda x: stats.mode(x)[0][0])
total["Fare"] = total["Fare"].fillna(8.05)
sns.distplot(total["Fare"], color="g")
plt.show()
g = sns.barplot(x="Sex",y="Survived",data=total, palette="Greens_d")
g = g.set_ylabel("Survival Probability")
# convert Sex into categorical value 0 for male and 1 for female
total["Sex"] = total["Sex"].map({"male": 0, "female":1})
g = sns.catplot(x="Pclass",y="Survived",data=train, kind="bar", height = 5 , palette = "Purples")
g = g.set_ylabels("survival probability")
g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train,
                   height=5, kind="bar", palette="Purples")
g = g.set_ylabels("survival probability")
total["CabinYN"] = pd.Series([1 if not pd.isnull(i) else 0 for i in total['Cabin'] ])

g = sns.catplot(x="CabinYN",y="Survived",data=total, kind="bar", height = 5 , palette = "Oranges")
g = g.set_ylabels("survival probability")
total["Cabin1"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in total['Cabin'] ])
ordinal = {"A": 1, "B": 2, "C": 3, "D":4, "E": 5, "F": 6, "G": 7, "T": 8, "X": 10}
total["Cabin_ord"] = total.Cabin1.map(ordinal)
total.drop(labels = ["Cabin1"], axis = 1, inplace = True)
total[["CabinYN", "Cabin_ord"]].head(6)
g = sns.catplot(x="Cabin_ord",y="Survived",data=total, kind="bar", height = 5 , palette = "Oranges")
g = g.set_ylabels("survival probability")
# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in total["Name"]]
total["Title"] = pd.Series(dataset_title)

# Convert to categorical values Title 
total["Title"] = total["Title"].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev'], 'Officer')
total["Title"] = total["Title"].replace(['Mme'], 'Mrs')
total["Title"] = total["Title"].replace(['Ms', 'Mlle'], 'Miss')
total["Title"] = total["Title"].replace(['Dona', 'Lady', 'the Countess','Sir', 'Jonkheer'], 'Royalty')

g = sns.catplot(x="Title", y="Survived", data=total, kind="bar", palette="Blues_d")
g = g.set_ylabels("survival probability")
g = sns.countplot(total["Title"], palette="Blues_d")
total[["Title", "Age"]].groupby(["Title"]).mean()
# Dummies fot Title
total = pd.get_dummies(total, columns = ["Title"])
# Drop Name variable
total.drop(labels = ["Name"], axis = 1, inplace = True)
total["Fsize"] = total["SibSp"] + total["Parch"] + 1

#data = total[:train.shape[0]]
g = sns.catplot(x="Fsize",y="Survived", data = total, kind='point')
g = g.set_ylabels("Survival Probability")
# Create new feature of family size
total['Single'] = total['Fsize'].map(lambda s: 1 if s == 1 else 0)
total['SmallF'] = total['Fsize'].map(lambda s: 1 if 2 <= s <= 4  else 0)
total['LargeF'] = total['Fsize'].map(lambda s: 1 if s >= 5 else 0)
g = sns.catplot(x="Single",y="Survived",data=total,kind="bar", height = 3, palette="YlOrBr")
g = g.set_ylabels("Survival Probability")
g = sns.catplot(x="SmallF",y="Survived",data=total,kind="bar", height = 3, palette="YlOrBr")
g = g.set_ylabels("Survival Probability")
g = sns.catplot(x="LargeF",y="Survived",data=total,kind="bar", height = 3, palette="YlOrBr")
g = g.set_ylabels("Survival Probability")
# Tiquets together and near
Torder = total["Ticket"].unique()
Torder = np.sort(Torder)
Index_T  = np.arange(len(Torder))
mapear = pd.DataFrame(data={"Torder": Torder, "index_T":Index_T})
total["Ticket_tog"] = total["Ticket"].map(mapear.set_index("Torder")["index_T"])

# the same for cabins
total["Cabin"] = total["Cabin"].fillna("X")
Corder = total["Cabin"].unique()
Corder = np.sort(Corder)
Index_C  = np.arange(len(Corder))
mapear = pd.DataFrame(data={"Corder": Corder, "index_C":Index_C})
total["Cabin_tog"] = total["Cabin"].map(mapear.set_index("Corder")["index_C"])
# for the cluster i will use the following features
df = total[["Cabin_tog", "Ticket_tog", "Em_C", "Em_S", "Em_Q", "Fare", 
              "Pclass", "Fsize"]]
# help me to select a n_cluster
range_n_clusters = np.arange(200, 600, 50)
for n_clusters in range_n_clusters:
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(df)
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(df, cluster_labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
kmeans = KMeans(n_clusters=450)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_
total["Groups"] = labels
df = total[["Sex", "Age", "Cabin_tog", "Ticket_tog", "Em_C", "Em_S", "Em_Q", "Fare", 
              "Pclass", "Fsize", "Groups", "Survived"]]
# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
plt.figure(figsize = (10,5))
g = sns.heatmap(df.corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
Ticket = []
for i in list(total.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")

total["Ticket"] = Ticket
total = pd.get_dummies(total, columns = ["Ticket"], prefix="T")
# Explore Age vs Survived. Initial
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")
X_age = total[["Fare", "Parch", "SibSp", "Sex", "Pclass",  
        "Title_Master", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Officer", "Title_Royalty"]]

Y_age = total[["Age"]]
index = Y_age.Age.isnull()
X_age_train = X_age[~index]
Y_age_train = Y_age[~index]
X_age_test = X_age[index]
Y_age_test = Y_age[index]

clf = LinearRegression()
clf.fit(X_age_train, Y_age_train)

p_age = clf.predict(X_age_test).round()
total.loc[index, "Age"] = p_age
# Explore Age vs Survived. Final
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")
# Drop useless variables 
total.drop(labels = ["PassengerId"], axis = 1, inplace = True)
total.drop(labels = ["Cabin"], axis = 1, inplace = True)

total= total.astype(float)
y = total["Survived"].copy()
X = total.drop(labels=["Survived"], axis=1)

T = preprocessing.StandardScaler().fit_transform(X)
 
X_train = T[:train_len]
X_test  = T[train_len:]
y_train = y[:train_len]
# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10, random_state=1)
# Modeling step Test differents algorithms 
random_state = 1
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", 
                                      cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,
                       "Algorithm":["SVC", "RandomForest","ExtraTrees","GradientBoosting",
                       "MultipleLayerPerceptron", "KNeighboors","LogisticRegression"]})
cv_res = cv_res.sort_values(["CrossValMeans"], ascending=False)
g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="RdYlBu",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
# Gradient Boosting
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }
gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsGBC.fit(X_train,y_train)
print ("Score: ", gsGBC.best_score_)
# Score:  0.833894500561
# Random Forest
RFC = RandomForestClassifier()
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsRFC.fit(X_train,y_train)
print ("Score: ", gsRFC.best_score_)
# Score:  0.833894500561
# Extra Trees Classifier
ExtC = ExtraTreesClassifier()
ex_param_grid = {"max_features": [3, 5, 8, 'auto'],
              "min_samples_split": [2, 6, 10],
              "min_samples_leaf": [1, 3, 10],
              "n_estimators" :[100, 150, 200]}
gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsExtC.fit(X_train, y_train)
print ("Score: ", gsExtC.best_score_)
# Score:  Score:  0.838383838384
# Logistic Regression
LG = LogisticRegression()
lg_param_grid = {'C': [1, 10, 50, 100,200,300, 1000]}
gsLG = GridSearchCV(LG,param_grid = lg_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsLG.fit(X_train,y_train)
print ("Score: ", gsLG.best_score_)
# Score:  0.820426487093
# Support Vector Machine.
SVMC = SVC()
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1, 3, 10],
                  'C': [1, 10, 50, 100,200,300]}
gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsSVMC.fit(X_train,y_train)
print ("Score: ", gsSVMC.best_score_)
# Score:  0.832772166105
gsExtC.best_estimator_
ExtC = ExtraTreesClassifier(min_samples_split=10, n_estimators=100 , random_state=1)
ExtC.fit(X_train, y_train)
importances = ExtC.feature_importances_
indices = np.argsort(importances)[::-1]
col = X.columns[indices]

# Top 25 features
plt.figure(figsize = (10,5))
g = sns.barplot(y=col[:25], x = importances[indices][:25] , orient='h')
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("Feature importance")
plt.show()
X_train.shape[1]
best_score=0
var=[]
scor=[]
for i in np.arange(0.1, 1.9, 0.1):

    str_t = np.str(i) + "*mean"
    ExtC = ExtraTreesClassifier(min_samples_split=10, n_estimators=100 , random_state=1)
    
    model = SelectFromModel(ExtC, threshold = str_t )
    model.fit(X_train, y_train)
    X_train_new = model.transform(X_train)
    X_test_new = model.transform(X_test)
     
    cvs = cross_val_score(ExtC, X_train_new, y = y_train, scoring = "accuracy", cv = kfold,
                          n_jobs=4)
    var.append(X_train_new.shape[1])
    scor.append(cvs.mean())
    score=round(cvs.mean(),3)
    print ("The cv accuracy is: ", round(score,4), " - i = ", i, 
           " - features = ", X_train_new.shape[1])
    if score>best_score:
        best_score = score
        print("*** BEST : Score:", score, " features selected: ", X_train_new.shape[1])
        X_train_best = X_train_new
        X_test_best = X_test_new
plt.plot(var, scor, lw=6, c="blue")
plt.plot(var, scor, "ro", ms=10, alpha=0.8)
plt.xlabel('Number of features')
plt.ylabel('Score')
plt.title('Selection Features')
plt.show()
#Selected features
g = sns.barplot(y=col[:X_test_best.shape[1]], x = importances[indices][:X_test_best.shape[1]] , orient='h')
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("Feature importance")
plt.show()
ExtC.fit(X_train_best, y_train)
test_Survived = pd.Series(ExtC.predict(X_test_best), name="Survived")
r = pd.DataFrame(test_Survived, dtype="int64")
results = pd.concat([IDtest,r],axis=1)
results.to_csv("result.csv",index=False)
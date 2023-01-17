# Import of libraries

import pandas as pd

import numpy as np

import matplotlib as mtl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy as sc

import sklearn



import warnings

warnings.filterwarnings('ignore') 



plt.rcParams['figure.figsize'] = (10, 10)



#Import of data

data = pd.read_csv("../input/bank.csv", sep = ";")





import sys

print ("Python version: {}".format(sys.version))

print ("Numpy version: {}".format(np.__version__))

print ("Pandas version: {}".format(pd.__version__))

print ("Matplotlib version: {}".format(mtl.__version__))

print ("Seaborn version: {}".format(sns.__version__))

print ("Scipy version: {}".format(sc.__version__))

print ("Scikit version: {}".format(sklearn.__version__))
data.head()
del data['duration']
np.unique(data.day)
Class_0 = data.day[data.y == 'no']

Class_1 = data.day[data.y == 'yes']

data_to_plot = [Class_0, Class_1]

plt.boxplot(x = data_to_plot)

plt.show()
del data['day']
cross = pd.crosstab(data.month, data.y,)

cross = cross.reindex(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])

cross.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
months = [['dec', 'jan', 'feb'], 

          ['mar', 'apr', 'may'], 

          ['jun', 'jul', 'aug'], 

          ['sep', 'oct', 'nov']]



seasons = ['winter', 'spring','summer', 'fall']



# Changing values of months for seasons 

for i, k in zip(months, seasons):

    data['month'].replace([i], k , inplace = True)



data.rename(columns = {'month':'season'}, inplace = True)
# Absolute values

cross = pd.crosstab(data.season, data.y)

cross = cross.reindex(['winter', 'spring', 'summer', 'fall'])

cross.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
#Relative values

cross = pd.crosstab(data.season, data.y)

cross = cross.div(cross.sum(1).astype(float), axis=0)

cross = cross.reindex(['winter', 'spring', 'summer', 'fall'])

cross.plot(kind='bar', stacked=True,  grid=False)
print('The dataset has {0} rows and {1} columns'.format(data.shape[0], data.shape[1]))
data.dtypes
data.isnull().any()
for i in data.columns: 

    if data[i].dtype == "O":

        print(np.unique(data[i].values))
#Calculating the percentage of missing values

data.replace("unknown", np.nan, inplace = True)

percentage = (data.isnull().sum()/len(data))*100

percentage.sort_values(inplace=True)

percentage
pd.crosstab(data.poutcome, data.y)
#Relative values

cross = pd.crosstab(data.poutcome, data.y)

cross = cross.div(cross.sum(1).astype(float), axis=0)

cross.plot(kind='bar', stacked=True,grid=False)
del data["poutcome"]
data.groupby('y').size() 
# Percentage of value "yes" in "default"

plt.pie([pd.value_counts(data.y)[0], pd.value_counts(data.y)[1]], 

        labels=["NO", "YES"], 

        startangle=90, 

        shadow=True, 

        explode=(0,0.1),

        autopct='%1.1f%%')

plt.title('Distribution of "default"')
data.describe()
cont_feat = ["age", "balance", "campaign", "pdays", "previous"]

for i in cont_feat:

    plt.title(i)

    plt.hist(data[i], bins = 100)

    plt.show()
data.previous = data.previous.map(lambda x: 0 if x == 0 else 1)
cross = pd.crosstab(data.previous, data.pdays)

cross
cross = pd.crosstab(data.previous, data.pdays)

cross
data.rename(columns = {'previous':'ContactedBefore'}, inplace = True)

del data["pdays"]
# FILTERING 

data=data[abs(data['balance']-data['balance'].mean())<= 3*data['balance'].std()]
# Violinplot 

sns.violinplot(y=data["balance"], x=data["y"])
from scipy import stats

stats.probplot(data['balance'], plot=plt)
# FILTERING 

data=data[abs(data['age']-data['age'].mean())<= 3*data['age'].std()]



#Violinplot

sns.violinplot(y=data["age"], x=data["y"])
#Probability plot



stats.probplot(data['age'], plot=plt)
# Test for normality 

from scipy.stats import mstats

mstats.normaltest(data.age)
cont_feat = ["age", "balance", "campaign"]

sns.pairplot(data[cont_feat])
cat_feat = ["job", "marital", "education", "default", "housing", "loan", "contact", "ContactedBefore", "season", "y"]

# Histrograms for categorical features

for i in cat_feat:

    fig, ax = plt.subplots(figsize=(3, 3))

    sns.set(style="whitegrid", color_codes=True)

    sns.countplot(ax = ax, x=i, data=data, palette="Greens_d")

    plt.xticks(rotation=90)

    plt.show()
del data["default"]
data[["age", "balance", "campaign"]].corr(method='pearson')
colormap = plt.cm.viridis

plt.figure(figsize=(7,7))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(data[["age", "balance", "campaign"]].corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
for i in data.columns:

    data[i]=data[i].replace(np.nan, data[i].mode()[0])
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



cat_feat = ["job", "marital", "education", "housing", "loan","season", "contact", "y"]



for i in cat_feat:

    data[i] = le.fit_transform(data[i])
data.head()
data = pd.get_dummies(data, columns = ["job", "marital", "education"])
data.head()
# X are predictors

# Y is the targer variable

X = data[data.columns.difference(['y'])]

Y = data.y
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size = 0.7, stratify = Y, random_state = 0)
from sklearn.preprocessing import StandardScaler

col_to_scale = ["age", "balance", "campaign"]

for i in col_to_scale:

    sc = StandardScaler().fit(X_train[i].values.reshape(-1, 1))

    X_train[i] = sc.transform(X_train[i].values.reshape(-1, 1))

    X_test[i] = sc.transform(X_test[i].values.reshape(-1, 1))
# Feature Extraction with RFE



from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

rfe = RFE(model, 15)

fit = rfe.fit(X_train, Y_train)

selected_feature = fit.support_



print("Num Features: %d", fit.n_features_) 

print("Selected Features: %s",  fit.support_)

print("Feature Ranking: %s", fit.ranking_) 
# Dropping unimportant features

col_to_drop=[]

for i in range(len(X.columns)-1):

    if selected_feature[i] == False:

        col_to_drop.append(i)



X_train.drop(X.iloc[:, col_to_drop], axis=1, inplace = True)

X_test.drop(X.iloc[:, col_to_drop], axis=1, inplace = True)
# Remaining features

X_train.columns
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
models = [

    SVC(kernel="rbf", class_weight = "balanced"), 

    KNeighborsClassifier(),

    DummyClassifier(strategy = 'most_frequent'), 

    LogisticRegression( class_weight="balanced"), 

]
scores = pd.DataFrame(columns= ['Model','Accuracy', 'F1-weighted', 'Precision', 'Recall'])

for model in models: 

    classifier=model.fit(X_train, Y_train)

    name = str(classifier).rsplit('(', 1)[0]

    accuracy = np.average(cross_val_score(classifier, X_test, Y_test, scoring= "accuracy"))

    f1 = np.average(cross_val_score(classifier, X_test, Y_test, scoring= "f1_weighted"))

    precision = np.average(cross_val_score(classifier, X_test, Y_test, scoring='precision_weighted'))

    recall = np.average(cross_val_score(classifier, X_test, Y_test, scoring='recall_weighted'))

    scores = scores.append({'Model': name,'Accuracy': accuracy,'F1-weighted': f1,

                             'Precision': precision, 'Recall': recall}, ignore_index=True)

    

scores.set_index("Model")  
scores.plot(kind='bar', title='Scores' )
from sklearn.grid_search import GridSearchCV

                    

parameters = {'n_neighbors': np.arange(10)+1}



model = GridSearchCV(KNeighborsClassifier(),

                    parameters,

                    n_jobs               = 5,

                    cv                   = 5,

                    scoring              = 'f1_weighted'

                    ) 



model.fit(X_train, Y_train)

print("\nThe best estimator: ", model.best_estimator_)

print("\nThe best precision score: ", model.best_score_)

print("\nHighest scoring parameter set: ", model.best_params_)

model = model.best_estimator_
accuracy = cross_val_score(model, X, Y, scoring='accuracy', cv=5) 

print( "Accuracy: "+ str(round(100*accuracy.mean(), 2)))

f1 = cross_val_score(model, X, Y, scoring='f1_weighted', cv=5) 



print("F1: " + str(round(100*f1.mean(), 2)))

precision = cross_val_score(model, X, Y, scoring='precision_weighted', cv=5) 

print("Precision: " + str(round(100*precision.mean(), 2)))

recall = cross_val_score(model, X, Y, scoring='recall_weighted', cv=5)

print("Recall: " + str(round(100*recall.mean(), 2)), "\n")
from sklearn.metrics import confusion_matrix



Y_pred = model.predict(X_test)

conf_m = confusion_matrix(Y_test,Y_pred)



Y_test_0 = Y_test.value_counts()[0]

Y_test_1 = Y_test.value_counts()[1]



conf_m_norm = np.array([[1.0 / Y_test_0, 1.0/Y_test_0],[1.0/Y_test_1, 1.0/Y_test_1]])

norm_conf_matrix = conf_m  * conf_m_norm





fig = plt.figure(figsize=(5,5))

sns.heatmap(norm_conf_matrix, cmap='coolwarm_r',linewidths=0.5,annot=True)

plt.title('Confusion Matrix')

plt.ylabel('Real Classes')

plt.xlabel('Predicted Classes')
# Confusion matrix

conf_m_norm
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from scipy.stats import spearmanr, chi2_contingency, mannwhitneyu, shapiro
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import missingno as msn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from warnings import filterwarnings
filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train["Survived"].value_counts()
test.head()
dataset=pd.concat([train, test], ignore_index=True)
dataset.info()
dataset.Pclass.value_counts()
dataset.Sex.value_counts()
dataset.SibSp.value_counts()
dataset.Parch.value_counts()
dataset.Ticket.value_counts()
dataset.Cabin.value_counts()
dataset.Embarked.value_counts()
train.isnull().any()
msn.matrix(train);
msn.heatmap(train);
train.Age.isnull().sum()
print(train.Age.skew())
print(train.Age.kurtosis())
train.Age.fillna(train.Age.median(), inplace=True)
train.Embarked.value_counts()
train.Embarked.isnull().sum()
train.Embarked.fillna("S", inplace=True)
train.Cabin.isnull().sum()
train.Cabin.value_counts()
train.Cabin=np.where(train.Cabin.isnull(), "unknown", train.Cabin)
train.isnull().any()
train.head()

test.isnull().any()
msn.matrix(test);
msn.heatmap(test);
test.select_dtypes(exclude="object").describe().T
test.Age.isnull().sum()
print(test.Age.skew())
print(test.Age.kurtosis())
test.Age.fillna(test.Age.median(), inplace=True)
test.Fare.isnull().sum()
test.Fare.fillna(test.Fare.median(), inplace=True)
test.Cabin.isnull().sum()
test.Cabin=np.where(test.Cabin.isnull(), "unknown", test.Cabin)
test.isnull().any()
test.head()
train.head()
train.select_dtypes(exclude="object").describe().T
sns.set(rc={'figure.figsize': (8, 8)})
sns.boxplot(x=train.Fare,orient="v");
Q1=train.Fare.quantile(0.25)
Q3=train.Fare.quantile(0.75)
IQR=Q3-Q1

lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR
IQR,lower_limit,upper_limit
freq_outliers = ((train.Fare < lower_limit) | (train.Fare > upper_limit)).value_counts()
freq_outliers
outliers_bool = ((train.Fare < lower_limit) | (train.Fare > upper_limit))
outliers_bool.head()
outliers=train.Fare[outliers_bool]
outliers.head()
train.Fare[outliers_bool]=train.Fare.median()
train.Fare[outliers_bool]
train.describe().T
train[["Age", "SibSp", "Parch", "Fare"]].corr(method="spearman")
spearmanr(train.SibSp, train.Fare)
spearmanr(train.Fare, train.Parch)
spearmanr(train.SibSp, train.Parch)
X=train.copy()
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor    
from joblib import Parallel, delayed

# Defining the function that you will run later
def calculate_vif_(X, thresh=5.0):
    variables = [X.columns[i] for i in range(X.shape[1])]
    dropped=True
    while dropped:
        dropped=False
        print(len(variables))
        vif = Parallel(n_jobs=-1,verbose=0)(delayed(variance_inflation_factor)(X[variables].values, ix) for ix in range(len(variables)))

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print(time.ctime() + ' dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables.pop(maxloc)
            dropped=True

    print('Remaining variables:')
    print([variables])
    return X[[i for i in variables]]

X = X[["Age", "SibSp", "Parch", "Fare"]] # Selecting your data

X2 = calculate_vif_(X,5) # Actually running the function
pd.crosstab(train.Survived, train.Pclass, margins=True, margins_name="Total")
obs=np.matrix(pd.crosstab(train.Survived, train.Pclass))
chi2_contingency(obs)
pd.crosstab(train.Survived, train.Sex, margins=True, margins_name="Total")
obs=np.matrix(pd.crosstab(train.Survived, train.Sex))
chi2_contingency(obs)
pd.crosstab(train.Survived, train.SibSp, margins=True, margins_name="Total")
obs=np.matrix(pd.crosstab(train.Survived, train.SibSp))
chi2_contingency(obs)
pd.crosstab(train.Survived, train.Parch, margins=True, margins_name="Total")
obs=np.matrix(pd.crosstab(train.Survived, train.Parch))
chi2_contingency(obs)
train["Cabin"]=train["Cabin"].map(lambda x: str(x)[:1])
pd.crosstab(train.Survived, train.Cabin, margins=True, margins_name="Total")
obs=np.matrix(pd.crosstab(train.Survived, train.Cabin))
chi2_contingency(obs)
pd.crosstab(train.Survived, [train.Pclass, train.Cabin], margins=True, margins_name="Total")
pd.crosstab(train.Survived, train.Embarked, margins=True, margins_name="Total")
obs=np.matrix(pd.crosstab(train.Survived, train.Embarked))
chi2_contingency(obs)
train.groupby("Survived")["Age"].describe()
plt.figure(figsize=(10,8))

sns.boxplot(x=train.Survived, y=train.Age);
Age_0=train[train["Survived"]==0]["Age"]
Age_1=train[train["Survived"]==1]["Age"]
shapiro(Age_0)
shapiro(Age_1)
mannwhitneyu(Age_0, Age_1)
train.groupby("Survived")["Fare"].describe()
plt.figure(figsize=(10,8))

sns.boxplot(x=train.Survived, y=train.Fare);
Fare_0=train[train["Survived"]==0]["Fare"]
Fare_1=train[train["Survived"]==1]["Fare"]
shapiro(Fare_0)
shapiro(Fare_1)
mannwhitneyu(Fare_0, Fare_0)
train.head()
test.head()
# dataset is updated 

dataset=pd.concat([train, test])
dataset = dataset[[train, test][0].columns]
dataset.head()
dataset["Pclass"]=np.where(dataset["Pclass"]==1, "1st", dataset["Pclass"])
dataset["Pclass"]=np.where(dataset["Pclass"]=="2", "2nd", dataset["Pclass"])
dataset["Pclass"]=np.where(dataset["Pclass"]=="3", "3th", dataset["Pclass"])
dataset["Cabin"]=dataset["Cabin"].map(lambda x: str(x)[:1])
dataset.drop(["Name", "Ticket"], axis=1, inplace=True)
dataset.head(10)
df=dataset.copy()
dms=pd.get_dummies(df[['Pclass', 'Sex', 'Cabin', 'Embarked']])
dms.head()
y=df["Survived"]
y.dropna(inplace=True)
y.isnull().sum()
y.shape
df.head()
X=df.drop(["Survived", "PassengerId", 'Pclass', 'Sex', 'Cabin','Embarked'], axis=1).astype("float64")
x = pd.concat([X, dms[['Pclass_2nd', "Pclass_3th", 'Sex_female',"Cabin_B","Cabin_C","Cabin_D", "Cabin_E","Cabin_F", "Cabin_G" ,"Cabin_T", "Cabin_u", 'Embarked_S', "Embarked_C"]]], axis=1)
x.head()
x_scaled=scale(x)
x_sc=pd.DataFrame(x_scaled, columns=['Age', 'SibSp', 'Parch', 'Fare',
       'Pclass_2nd', 'Pclass_3th', 'Sex_female', 'Cabin_B', 'Cabin_C',
       'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_u',
       'Embarked_S', 'Embarked_C'])
x_sc.head()
train_data=x_sc[0:len(train)]
test_data=x_sc[len(train):]
test_data.reset_index(drop=True, inplace=True)
train_data.tail()
test_data.head()
x_train, x_test, y_train, y_test = train_test_split(train_data, y, 
                                                    test_size = 0.30, 
                                                    random_state = 42)
loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(x_train,y_train)
loj_model
y_pred=loj_model.predict(x_test)
accuracy_score(y_test, loj_model.predict(x_test))
cross_val_score(loj_model, x_test, y_test, cv = 10).mean()
nb = GaussianNB()
nb_model = nb.fit(x_train, y_train)
nb_model
y_pred = nb_model.predict(x_test)
print(accuracy_score(y_test, y_pred))
cross_val_score(nb_model, x_test, y_test, cv = 10).mean()
knn = KNeighborsClassifier(23)
knn_tuned = knn.fit(x_train, y_train)
knn_tuned
y_pred = knn_tuned.predict(x_test)
accuracy_score(y_test, y_pred)
svc_tuned_linear=SVC(kernel = "linear", C = 1, probability=True, random_state=1)
svc_tuned_linear.fit(x_train, y_train)
y_pred = svc_tuned_linear.predict(x_test)
accuracy_score(y_test, y_pred)
svc_tuned=SVC(C = 1, gamma = 0.1, probability=True,random_state=1)
svc_tuned.fit(x_train, y_train)
y_pred=svc_tuned.predict(x_test)
accuracy_score(y_test, y_pred)
mlpc_tuned = MLPClassifier(activation = "relu", 
                           alpha = 0.0001, 
                           hidden_layer_sizes = (100, 100, 100),
                           solver = "sgd", random_state=1)
mlpc_tuned.fit(x_train, y_train)
y_pred = mlpc_tuned.predict(x_test)
accuracy_score(y_test, y_pred)
cart = tree.DecisionTreeClassifier(max_depth = 6, min_samples_split = 3, random_state=1)
cart_tuned = cart.fit(x_train, y_train)
cart_tuned
y_pred = cart_tuned.predict(x_test)
accuracy_score(y_test, y_pred)
rf_tuned = RandomForestClassifier(max_depth = 8, 
                                  max_features = 5, 
                                  min_samples_split = 2,
                                  n_estimators = 1000, random_state=1)

rf_tuned.fit(x_train, y_train)
y_pred = rf_tuned.predict(x_test)
accuracy_score(y_test, y_pred)
gbm = GradientBoostingClassifier(learning_rate = 0.1, 
                                 max_depth = 3,
                                min_samples_split = 5,
                                n_estimators = 100, random_state=1)
gbm_tuned =  gbm.fit(x_train,y_train)
gbm_tuned
y_pred = gbm_tuned.predict(x_test)
accuracy_score(y_test, y_pred)
xgb = XGBClassifier(learning_rate = 0.02, 
                    max_depth = 3,
                    min_samples_split = 2,
                    n_estimators = 2000,
                    subsample = 1)
xgb_tuned =  xgb.fit(x_train,y_train)
xgb_tuned
y_pred = xgb_tuned.predict(x_test)
accuracy_score(y_test, y_pred)
lgbm = LGBMClassifier(learning_rate = 0.02, 
                       max_depth = 4,
                       subsample = 0.6,
                       n_estimators = 500,
                       min_child_samples = 20,random_state=1)
lgbm_tuned = lgbm.fit(x_train,y_train)
lgbm_tuned
y_pred = lgbm_tuned.predict(x_test)
accuracy_score(y_test, y_pred)
catb = CatBoostClassifier(iterations = 200, 
                          learning_rate = 0.1, 
                          depth = 5, verbose=False, random_seed=1)

catb_tuned = catb.fit(x_train, y_train)
catb_tuned
y_pred = catb_tuned.predict(x_test)
accuracy_score(y_test, y_pred)
models_importance = [
{
    'label': 'DecisionTreeClassifier',
    'model': cart_tuned,
},
{
    'label': 'RandomForestClassifier',
    'model': rf_tuned,
},
{
    'label': 'GradientBoostingClassifier',
    'model': gbm_tuned,
},
{
    'label': 'XGBClassifier',
    'model': xgb_tuned,
},
{
    'label': 'LGBMClassifier',
    'model': lgbm_tuned,
},
{
    'label': 'CatBoostClassifier',
    'model': catb_tuned,
}
]

for model in models_importance:
    names = model['label']
    model = model["model"]    
    model_imp=pd.DataFrame(model.feature_importances_, index = x_train.columns)
    model_imp.columns=["imp"]
    model_imp.sort_values(by = "imp", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")
    plt.title(names)
    plt.xlabel("Importance Values")
models_importance = [
{
    'label': 'LogisticRegression',
    'model': loj_model,
},
{
    'label': 'SVC_LINEAR',
    'model': svc_tuned_linear,
}
]


for model in models_importance:
    names = model['label']
    model = model["model"]
    coef=np.array(model.coef_)
    model_imp=pd.DataFrame(coef.flatten(), index = x_train.columns)
    model_imp.columns=["imp"]
    model_imp.sort_values(by = "imp", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")
    plt.title(names)
    plt.xlabel("Importance Values")
models = [
{
    'label': 'KNeighborsClassifier',
    'model': knn_tuned,
},
{
    'label': 'LogisticRegression',
    'model': loj_model,
},
{
    'label': 'SVC_LINEAR',
    'model': svc_tuned_linear,
},
{
    'label': 'SVC_RBF',
    'model': svc_tuned,
},
{
    'label': 'GaussianNB',
    'model': nb_model,
},
{
    'label': 'MLPClassifier',
    'model': mlpc_tuned,
},
{
    'label': 'DecisionTreeClassifier',
    'model': cart_tuned,
},
{
    'label': 'RandomForestClassifier',
    'model': rf_tuned,
},
{
    'label': 'GradientBoostingClassifier',
    'model': gbm_tuned,
},
{
    'label': 'XGBClassifier',
    'model': xgb_tuned,
},
{
    'label': 'LGBMClassifier',
    'model': lgbm_tuned,
},
{
    'label': 'CatBoostClassifier',
    'model': catb_tuned,
}
]

for model in models:
    names = model['label']
    model = model["model"]
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusionmatrix=confusion_matrix(y_test, y_pred)
    (TN, FP, FN, TP) = confusionmatrix.ravel()
    TPR = TP/(TP+FN) 
    TNR = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    print("-"*28)
    print(names + ":" )
    print("Accuracy: {:.4%}".format(accuracy))
    print("TPR     : {:.4%}".format(TPR))
    print("TNR     : {:.4%}".format(TNR))
    print("PPV     : {:.4%}".format(PPV))
    print("NPV     : {:.4%}".format(NPV))
    print("FPR     : {:.4%}".format(FPR))
    print("FNR     : {:.4%}".format(FNR))
result = []

results = pd.DataFrame(columns= ["Models","Accuracy"])

for model in models:
    names = model['label']
    model = model["model"]
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)    
    result = pd.DataFrame([[names, accuracy*100]], columns= ["Models","Accuracy"])
    results = results.append(result)


    
results=results.sort_values(by="Accuracy", ascending=False).reset_index()
results.drop(["index"], axis=1, inplace=True)
g=sns.barplot(x= 'Accuracy', y = 'Models', data=results, color="r")
g.set_xlabel("Accuracy",fontsize=20)
g.set_ylabel("Models",fontsize=20)

for index, row in results.iterrows():
    g.text(row.Accuracy, row.name, round(row.Accuracy,2), color='black', horizontalalignment='left', fontsize=13)

plt.xlabel('Accuracy %')
plt.title('Accuracy Scores of Models');  
from sklearn import metrics
import matplotlib.pyplot as plt

plt.figure()


# Below for loop iterates through your models list
for m in models:
    model = m['model'] # select the model
# Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(x_test)[:,1])
# Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test,model.predict(x_test))
# Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (AUC = %0.4f)' % (m['label'], auc))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5) , fontsize=15)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.show()   # Display
from sklearn import metrics

plt.figure()

# Add the models to the list that you want to view on the ROC plot
modelsROC = [
{
    'label': 'LogisticRegression',
    'model': loj_model,
},
{
    'label': 'GradientBoostingClassifier',
    'model': gbm_tuned,
}
]

# Below for loop iterates through your models list
for m in modelsROC:
    model = m['model'] # select the model
# Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(x_test)[:,1])
# Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test,model.predict(x_test))
# Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (AUC = %0.4f)' % (m['label'], auc))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='right corner', fontsize=15)
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.show()   # Display
test_data_y_pred=gbm_tuned.predict(test_data)
survived=pd.concat([test["PassengerId"], (pd.DataFrame(test_data_y_pred, columns=["Survived"]))], axis=1)
survived["Survived"]=survived["Survived"].astype("int64")
survived
survived.to_csv("Survived_Prediction.csv")

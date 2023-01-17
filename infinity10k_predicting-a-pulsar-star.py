import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



warnings.filterwarnings("ignore")



%matplotlib inline
df = pd.read_csv("../input/predicting-a-pulsar-star/pulsar_stars.csv")

df.head()
plt.figure(figsize=(12,6))

plt.subplot(121)

ax = sns.countplot(y = df["target_class"],

                   palette=["r","g"],

                   linewidth=1,

                   edgecolor="k"*2)

for i,j in enumerate(df["target_class"].value_counts().values):

    ax.text(.7,i,j,weight = "bold",fontsize = 27)

plt.title("Count for target variable in dataset")





plt.subplot(122)

plt.pie(df["target_class"].value_counts().values,

        labels=["not pulsar stars","pulsar stars"],

        autopct="%1.0f%%",wedgeprops={"linewidth":2,"edgecolor":"white"})

my_circ = plt.Circle((0,0),.7,color = "white")

plt.gca().add_artist(my_circ)

plt.subplots_adjust(wspace = .2)

plt.title("Proportion of target variable in dataset")

plt.show()
df.info()
df.isnull().sum()
#Renaming columns

df = df.rename(columns={' Mean of the integrated profile':"mean_profile",

       ' Standard deviation of the integrated profile':"std_profile",

       ' Excess kurtosis of the integrated profile':"excess_kurtosis_profile",

       ' Skewness of the integrated profile':"skewness_profile", 

        ' Mean of the DM-SNR curve':"mean_dmsnr_curve",

       ' Standard deviation of the DM-SNR curve':"std_dmsnr_curve",

       ' Excess kurtosis of the DM-SNR curve':"excess_kurtosis_dmsnr_curve",

       ' Skewness of the DM-SNR curve':"skewness_dmsnr_curve",

       })
df.describe()
plt.figure(figsize=(16,12))

sns.heatmap(data=df.corr(), annot=True, cmap="magma", linewidth=1, fmt=".2f")

plt.title("Correlation Map",fontsize=20)

plt.tight_layout()

plt.show()
sns.pairplot(df ,hue = "target_class")
import itertools



columns = [x for x in df.columns if x not in ["target_class"]]

length  = len(columns)



plt.figure(figsize=(13,25))



for i,j in itertools.zip_longest(columns, range(length)):

    plt.subplot(length/2 ,length/4, j+1)

    sns.violinplot(x=df["target_class"], y=df[i],

                   palette=["Orangered","lime"], alpha=.5)

    plt.title(i)
X = df.drop('target_class', axis = 1)

y = df.target_class
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc



def model(algorithm, dtrain_x, dtrain_y, dtest_x, dtest_y, of_type):

    

    print ("*****************************************************************************************")

    print ("MODEL - OUTPUT")

    print ("*****************************************************************************************")

    algorithm.fit(dtrain_x,dtrain_y)

    predictions = algorithm.predict(dtest_x)

    

    print (algorithm)

    print ("\naccuracy_score :",accuracy_score(dtest_y,predictions))

    

    print ("\nclassification report :\n",(classification_report(dtest_y,predictions)))

        

    plt.figure(figsize=(13,10))

    plt.subplot(221)

    sns.heatmap(confusion_matrix(dtest_y,predictions),annot=True,fmt = "d",linecolor="k",linewidths=3)

    plt.title("CONFUSION MATRIX",fontsize=20)

    

    predicting_probabilites = algorithm.predict_proba(dtest_x)[:,1]

    fpr,tpr,thresholds = roc_curve(dtest_y,predicting_probabilites)

    plt.subplot(222)

    plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")

    plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")

    plt.legend(loc = "best")

    plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=20)

    

    if  of_type == "feat":

        

        dataframe = pd.DataFrame(algorithm.best_estimator_.feature_importances_,dtrain_x.columns).reset_index()

        dataframe = dataframe.rename(columns={"index":"features",0:"coefficients"})

        dataframe = dataframe.sort_values(by="coefficients",ascending = False)

        plt.subplot(223)

        ax = sns.barplot(x = "coefficients" ,y ="features",data=dataframe,palette="husl")

        plt.title("FEATURE IMPORTANCES",fontsize =20)

        for i,j in enumerate(dataframe["coefficients"]):

            ax.text(.011,i,j,weight = "bold")

    

    elif of_type == "coef" :

        

        dataframe = pd.DataFrame(algorithm.coef_.ravel(),dtrain_x.columns).reset_index()

        dataframe = dataframe.rename(columns={"index":"features",0:"coefficients"})

        dataframe = dataframe.sort_values(by="coefficients",ascending = False)

        plt.subplot(223)

        ax = sns.barplot(x = "coefficients" ,y ="features",data=dataframe,palette="husl")

        plt.title("FEATURE IMPORTANCES",fontsize =20)

        for i,j in enumerate(dataframe["coefficients"]):

            ax.text(.011,i,j,weight = "bold")

            

    elif of_type == "none" :

        return (algorithm)
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
dt = DecisionTreeClassifier(random_state = 42)
param_grid = {'criterion': ['gini', 'entropy'],

    'max_depth': range(1, 9),

    'min_samples_split': range(25, 31),

    'min_samples_leaf': range(2, 6)}
dt_search = GridSearchCV(dt, param_grid, cv = 5, n_jobs = -1)
model(dt_search, X_train, y_train, X_test, y_test, "feat")
dt_search.best_params_
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV
rt = RandomForestClassifier(n_estimators = 100, random_state = 42)
rt_search = RandomizedSearchCV(rt, param_grid, cv = 5, n_jobs = -1)
model(rt_search, X_train, y_train, X_test, y_test, "feat")
rt_search.best_params_
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42, n_jobs=-1, C=1.6)
model(lr, X_train, y_train, X_test, y_test, "coef")
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

model(gbc, X_train, y_train, X_test, y_test, "feat")
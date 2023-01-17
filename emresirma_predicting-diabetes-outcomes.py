#installation of libraries

import numpy as np

import pandas as pd 

import statsmodels.api as sm

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold
#any warnings that do not significantly impact the project are ignored.

import warnings

warnings.simplefilter(action = "ignore") 
#reading the dataset

df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

#selection of the first 5 observations

df.head() 
#return a random sample of items from an axis of object

df.sample(3) 
#makes random selection from dataset at the rate of written value

df.sample(frac = 0.01) 
#size information

df.shape
#dataframe's index dtype and column dtypes, non-null values and memory usage information

df.info()
#explanatory statistics values of the observation units corresponding to the specified percentages

df.describe([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T

#transposition of the df table. This makes it easier to evaluate.
#correlation between variables

df.corr()
#get a histogram of the Glucose column for both classes



col = 'Glucose'

plt.hist(df[df['Outcome']==0][col], 10, alpha=0.5, label='non-diabetes')

plt.hist(df[df['Outcome']==1][col], 10, alpha=0.5, label='diabetes')

plt.legend(loc='upper right')

plt.xlabel(col)

plt.ylabel('Frequency')

plt.title('Histogram of {}'.format(col))

plt.show()
for col in ['BMI', 'BloodPressure']:

    plt.hist(df[df['Outcome']==0][col], 10, alpha=0.5, label='non-diabetes')

    plt.hist(df[df['Outcome']==1][col], 10, alpha=0.5, label='diabetes')

    plt.legend(loc='upper right')

    plt.xlabel(col)

    plt.ylabel('Frequency')

    plt.title('Histogram of {}'.format(col))

    plt.show()
def plot_corr(df,size = 9): 

    corr = df.corr() #corr = variable, where we assign the correlation matrix to a variable

    fig, ax = plt.subplots(figsize = (size,size)) 

    #fig = the column to the right of the chart, subplots (figsize = (size, size)) = determines the size of the chart

    ax.matshow(corr) # prints the correlation, which draws the matshow matrix directly

    cax=ax.matshow(corr, interpolation = 'nearest') #plotting axis, code that makes the graphic like square or map

    fig.colorbar(cax) #plotting color

    plt.xticks(range(len(corr.columns)),corr.columns,rotation=65) 

    # draw xticks, rotation = 17 is for inclined printing of expressions written for each top column

    plt.yticks(range(len(corr.columns)),corr.columns) #draw yticks
#we draw the dataframe using the function.

plot_corr(df) 
#correlation matrix in seaborn library

import seaborn as sb

sb.heatmap(df.corr());
#this way we can see the correlations

sb.heatmap(df.corr(),annot =True); 
#proportions of classes 0 and 1 in Outcome

df["Outcome"].value_counts()*100/len(df)
#how many classes are 0 and 1

df.Outcome.value_counts()
#histogram of the Age variable

df["Age"].hist(edgecolor = "black");
#Age, Glucose and BMI means according to Outcome variable

df.groupby("Outcome").agg({"Age":"mean","Glucose":"mean","BMI":"mean"})
#no missing data in dataset

df.isnull().sum()
#zeros in the corresponding variables mean NA, so 0 is assigned instead of NA

df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.NaN)
#exclusive values

df.isnull().sum()
def median_target(var):   

    

    temp = df[df[var].notnull()] 

    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index() #reset_index; solved problems in indices

    

    return temp

#Non-nulls are selected from within df and assigned to a dataframe named temp, ignoring the observation units filled.
#median of glucose taken according to Outcome's value of 0 and 1

median_target("Glucose")
#median values of diabetes and non-diabetes were given for incomplete observations.



columns = df.columns



columns = columns.drop("Outcome")



for col in columns:

    

    df.loc[(df['Outcome'] == 0 ) & (df[col].isnull()), col] = median_target(col)[col][0]

    df.loc[(df['Outcome'] == 1 ) & (df[col].isnull()), col] = median_target(col)[col][1]

    #select the outcome value 0 and the relevant variable blank, select the relevant variable

#It refers to pre-comma filtering operations, it is used for column selection after comma.
#according to BMI, some ranges were determined and categorical variables were assigned.

NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype = "category")



df["NewBMI"] = NewBMI



df.loc[df["BMI"] < 18.5, "NewBMI"] = NewBMI[0]



df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]

df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]

df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]

df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]

df.loc[df["BMI"] > 39.9 ,"NewBMI"] = NewBMI[5]
df.head()
#categorical variable creation according to the insulin value

def set_insulin(row):

    if row["Insulin"] >= 16 and row["Insulin"] <= 166:

        return "Normal"

    else:

        return "Abnormal"     
df.head()
#NewInsulinScore variable added with set_insulin

df["NewInsulinScore"] = df.apply(set_insulin, axis=1)
df.head()
#some intervals were determined according to the glucose variable and these were assigned categorical variables.



NewGlucose = pd.Series(["Low", "Normal", "Overweight", "Secret", "High"], dtype = "category")



df["NewGlucose"] = NewGlucose



df.loc[df["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]



df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = NewGlucose[1]



df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), "NewGlucose"] = NewGlucose[2]



df.loc[df["Glucose"] > 126 ,"NewGlucose"] = NewGlucose[3]
df.head()
#categorical variables were converted into numerical values by making One Hot Encoding transform

#it is also protected from the Dummy variable trap

df = pd.get_dummies(df, columns =["NewBMI","NewInsulinScore", "NewGlucose"], drop_first = True)
df.head()
#categorical variables

categorical_df = df[['NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',

                     'NewInsulinScore_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret']]
#categorical variables deleted from df

y = df["Outcome"]

X = df.drop(["Outcome",'NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',

                     'NewInsulinScore_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret'], axis = 1)

cols = X.columns

index = X.index
y.head()
X.head()
#by standardizing the variables in the dataset, the performance of the models is increased.

from sklearn.preprocessing import RobustScaler

transformer = RobustScaler().fit(X)

X = transformer.transform(X)

X = pd.DataFrame(X, columns = cols, index = index)
X.head()
#combining non-categorical and categorical variables

X = pd.concat([X, categorical_df], axis = 1)
X.head()
models = []

models.append(('LR', LogisticRegression(random_state = 12345)))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier(random_state = 12345)))

models.append(('RF', RandomForestClassifier(random_state = 12345)))

models.append(('SVM', SVC(gamma='auto', random_state = 12345)))

models.append(('XGB', GradientBoostingClassifier(random_state = 12345)))

models.append(("LightGBM", LGBMClassifier(random_state = 12345)))



#evaluate each model in turn

results = []

names = []



for name, model in models:

        

        cv_results = cross_val_score(model, X, y, cv = 10, scoring= "accuracy")

        results.append(cv_results)

        names.append(name)

        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        print(msg)

        

#boxplot algorithm comparison

fig = plt.figure(figsize=(15,10))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results,

            vert=True, #vertical box alignment

            patch_artist=True) #fill with color

                         

ax.set_xticklabels(names)

plt.show()
rf_params = {"n_estimators" :[100,200,500,1000], 

             "max_features": [3,5,7], 

             "min_samples_split": [2,5,10,30],

            "max_depth": [3,5,8,None]}
rf_model = RandomForestClassifier(random_state = 12345)
gs_cv = GridSearchCV(rf_model, 

                    rf_params,

                    cv = 10,

                    n_jobs = -1,

                    verbose = 2).fit(X, y)
gs_cv.best_params_
rf_tuned = RandomForestClassifier(**gs_cv.best_params_)
rf_tuned = rf_tuned.fit(X,y)
cross_val_score(rf_tuned, X, y, cv = 10).mean()
feature_imp = pd.Series(rf_tuned.feature_importances_,

                        index=X.columns).sort_values(ascending=False)



sns.barplot(x=feature_imp, y=feature_imp.index, palette="Blues_d")

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Feature Severity Levels")

plt.show()
xgb = GradientBoostingClassifier(random_state = 12345)
xgb_params = {

    "learning_rate": [0.01, 0.1, 0.2, 1],

    "min_samples_split": np.linspace(0.1, 0.5, 3),

    "max_depth":[3,5,8],

    "subsample":[0.5, 0.9, 1.0],

    "n_estimators": [100,500]}
xgb_cv = GridSearchCV(xgb,xgb_params, cv = 10, n_jobs = -1, verbose = 2).fit(X, y)
xgb_cv.best_params_
xgb_tuned = GradientBoostingClassifier(**xgb_cv.best_params_).fit(X,y)
cross_val_score(xgb_tuned, X, y, cv = 10).mean()
feature_imp = pd.Series(xgb_tuned.feature_importances_,

                        index=X.columns).sort_values(ascending=False)



sns.barplot(x=feature_imp, y=feature_imp.index, palette="Blues_d")

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Feature Severity Levels")

plt.show()
lgbm = LGBMClassifier(random_state = 12345)
lgbm_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.5],

              "n_estimators": [500, 1000, 1500],

              "max_depth":[3,5,8]}
gs_cv = GridSearchCV(lgbm, 

                     lgbm_params, 

                     cv = 10, 

                     n_jobs = -1, 

                     verbose = 2).fit(X, y)
gs_cv.best_params_
lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X,y)
cross_val_score(lgbm_tuned, X, y, cv = 10).mean()
feature_imp = pd.Series(lgbm_tuned.feature_importances_,

                        index=X.columns).sort_values(ascending=False)



sns.barplot(x=feature_imp, y=feature_imp.index, palette="Blues_d")

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Feature Severity Levels")

plt.show()
models = []



models.append(('RF', RandomForestClassifier(random_state = 12345, max_depth = 8, max_features = 7, min_samples_split = 2, n_estimators = 500)))

models.append(('XGB', GradientBoostingClassifier(random_state = 12345, learning_rate = 0.1, max_depth = 5, min_samples_split = 0.1, n_estimators = 100, subsample = 1.0)))

models.append(("LightGBM", LGBMClassifier(random_state = 12345, learning_rate = 0.01,  max_depth = 3, n_estimators = 1000)))



results = []

names = []
for name, model in models:

    

        cv_results = cross_val_score(model, X, y, cv = 10, scoring= "accuracy")

        results.append(cv_results)

        names.append(name)

        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        print(msg)

        

# boxplot algorithm comparison

fig = plt.figure(figsize=(15,10))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results,

            vert=True, #vertical box alignment

            patch_artist=True) #fill with color

                         

ax.set_xticklabels(names)

plt.show()
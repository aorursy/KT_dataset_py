import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import warnings  
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/StudentsPerformance.csv')
df.head()
#Check for missing values
if df.isnull().any().unique() == False:
    print("No missing values")
else: 
    print(df.isnull().any())
sns.pairplot(df)
#Skewness & Kurtosis
#The values for asymmetry and kurtosis between -2 and +2 are considered acceptable in order to prove normal univariate distribution (George & Mallery, 2010)
print("Skewness \n", df.skew(), "\n\nKurtosis\n", df.kurtosis())
#Outliers
f, axes = plt.subplots(1, 3, figsize=(15,5))
sns.boxplot(df["math score"], orient='v' ,ax = axes[0])
sns.boxplot(df["reading score"],orient='v' , ax = axes[1])
sns.boxplot(df["writing score"],orient='v' , ax = axes[2])
#Remove Outliers
def remove_outlier(col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3-q1
    #Cut-Off
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    #Remove Outliers
    df2 = df.loc[(df[col] > fence_low) & (df[col] < fence_high)]
    return df2
num_rows_org = df.shape[0]
columns = ["math score", "reading score", "writing score"]
for i in columns:
    df = remove_outlier(i) 
num_rows_new = df.shape[0]
print(num_rows_org-num_rows_new,"Outliers Removed")
f, axes = plt.subplots(1, 3, figsize=(15,5))
sns.boxplot(df["math score"], orient='v' ,ax = axes[0])
sns.boxplot(df["reading score"],orient='v' , ax = axes[1])
sns.boxplot(df["writing score"],orient='v' , ax = axes[2])
#Skewness & Kurtosis
#The values for asymmetry and kurtosis between -2 and +2 are considered acceptable in order to prove normal univariate distribution (George & Mallery, 2010)
print("Skewness \n", df.skew(), "\n\nKurtosis\n", df.kurtosis())
sns.pairplot(df)
#Encode Variables
from sklearn.preprocessing import LabelEncoder
df['gender'] = LabelEncoder().fit_transform(df["gender"]) #'male' or 'female'
df['race/ethnicity'] = LabelEncoder().fit_transform(df["race/ethnicity"]) #Group A to E
df['parental level of education'] = LabelEncoder().fit_transform(df["parental level of education"])#'bachelor's degree', 'some college', "master's degree","associate's degree", 'high school' or 'some high school'
df['lunch'] = LabelEncoder().fit_transform(df["lunch"]) #'standard' or 'free/reduced'
df['test preparation course'] = LabelEncoder().fit_transform(df["test preparation course"])#'none' or 'completed'

df.head(33)
df["overall_score"] = ((df["math score"] + df["reading score"] + df["writing score"]) / 3).astype(int)
median = df["overall_score"].median()
df.head()
f, axes = plt.subplots(1, 5, figsize=(25,5))
sns.countplot(df["gender"], ax = axes[0])
sns.countplot(df["race/ethnicity"], ax = axes[1])
sns.countplot(df["parental level of education"], ax = axes[2])
sns.countplot(df["lunch"], ax = axes[3])
sns.countplot(df["test preparation course"], ax = axes[4])
corr = df.corr()
sns.heatmap(corr)
print("Boys with Lunch:", len(df[(df["gender"] == 0)  & (df["lunch"] == 0)]) / len(df[df["gender"] == 0]))
print("Girls with Lunch:", len(df[(df["gender"] == 1)  & (df["lunch"] == 0)]) / len(df[df["gender"] == 1]))

print("Boys with Preparation:", len(df[(df["gender"] == 0)  & (df["test preparation course"] == 0)]) / len(df[df["gender"] == 0]))
print("Girls with Preparation:", len(df[(df["gender"] == 1)  & (df["test preparation course"] == 0)]) / len(df[df["gender"] == 1]))

print("Preparation with Lunch:", len(df[(df["lunch"] == 0)  & (df["test preparation course"] == 0)]) / len(df[df["test preparation course"] == 0]))
print("Preparation without Lunch:", len(df[(df["lunch"] == 1)  & (df["test preparation course"] == 0)]) / len(df[df["test preparation course"] == 1]))
from scipy.stats import ttest_ind
w_lunch_mean = df[(df["lunch"] == 0)]
wo_lunch_mean = df[(df["lunch"] == 1)]

print(ttest_ind(w_lunch_mean['overall_score'], wo_lunch_mean['overall_score'], nan_policy='omit'))

print('With Lunch:', df[(df["lunch"] == 0)].overall_score.mean())
print('Without Lunch:', df[(df["lunch"] == 1)].overall_score.mean())
from scipy.stats import ttest_ind
w_course_mean = df[(df["test preparation course"] == 0)]
wo_course_mean = df[(df["test preparation course"] == 1)]

print(ttest_ind(w_course_mean['overall_score'], wo_course_mean['overall_score'], nan_policy='omit'))

print('With Course:', df[(df["test preparation course"] == 0)].overall_score.mean())
print('Without Course:', df[(df["test preparation course"] == 1)].overall_score.mean())
from scipy.stats import ttest_ind
n_lunch_n_course = df[(df["lunch"] == 1)  & (df["test preparation course"] == 1)]
lunch_course = df[(df["lunch"] == 0)  & (df["test preparation course"] == 0)]

print(ttest_ind(n_lunch_n_course['overall_score'], lunch_course['overall_score'], nan_policy='omit'))

print('With Course:', df[(df["lunch"] == 0)  & (df["test preparation course"] == 0)].overall_score.mean())
print('Without Course:', df[(df["lunch"] == 0)  & (df["test preparation course"] == 0)].overall_score.mean())
from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression

x = df.drop(["math score","overall_score"], axis = 1)
y = df["math score"]

score_lr = cross_val_score(LinearRegression(), x, y, cv = RepeatedKFold(n_repeats = 10))
score_rf = cross_val_score(RandomForestRegressor(n_estimators = 100), x, y, cv = RepeatedKFold(n_repeats = 10))
print("Math Score:\nLinear Regression:",np.mean(score_lr),"\nRandom Forest:",np.mean(score_rf))
x = df.drop(["math score", "writing score", "reading score", "overall_score"], axis = 1)
y = df["overall_score"]

score_lr = cross_val_score(LinearRegression(), x, y, cv = RepeatedKFold(n_repeats = 10))
score_rf = cross_val_score(RandomForestRegressor(n_estimators = 100), x, y, cv = RepeatedKFold(n_repeats = 10))
print("Overall_Score:\nLinear Regression:",np.mean(score_lr),"\nRandom Forest:",np.mean(score_rf))
df["overall_median"] = df["overall_score"]
df.overall_median[(df["overall_score"] >= median)] = 1
df.overall_median[(df["overall_score"] < median)] = 0
df.head()
x = df.drop(["math score", "writing score", "reading score", "overall_score", "overall_median"], axis = 1)
y = df["overall_median"]

score_lr = cross_val_score(LinearRegression(), x, y, cv = RepeatedKFold(n_repeats = 10))
score_rf = cross_val_score(RandomForestClassifier(n_estimators = 100), x, y, cv = RepeatedKFold(n_repeats = 10))
print("Good/Bad Student:\nLinear Regression:",np.mean(score_lr),"\nRandom Forest:",np.mean(score_rf))
x = df.drop(["math score", "writing score", "reading score", "overall_score", "overall_median", "parental level of education"], axis = 1)
y = df["overall_median"]

score_lr = cross_val_score(LinearRegression(), x, y, cv = RepeatedKFold(n_repeats = 10))
score_rf = cross_val_score(RandomForestClassifier(n_estimators = 100), x, y, cv = RepeatedKFold(n_repeats = 10))
print("Good/Bad Student:\nLinear Regression:",np.mean(score_lr),"\nRandom Forest:",np.mean(score_rf))
sns.countplot(x = "overall_median", hue = "race/ethnicity", data = df)
df["ethnicity"] = df["race/ethnicity"]
df.ethnicity[(df["race/ethnicity"] <= 2)] = 0
df.ethnicity[(df["race/ethnicity"] > 2)] = 1
df.head()
x = df.drop(["math score", "writing score", "reading score", "overall_score", "overall_median", "parental level of education", "race/ethnicity"], axis = 1)
y = df["overall_median"]

score_lr = cross_val_score(LinearRegression(), x, y, cv = RepeatedKFold(n_repeats = 10))
score_rf = cross_val_score(RandomForestClassifier(n_estimators = 100), x, y, cv = RepeatedKFold(n_repeats = 10))
print("Good/Bad Student:\nLinear Regression:",np.mean(score_lr),"\nRandom Forest:",np.mean(score_rf))
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

x = df.drop(["math score", "writing score", "reading score", "overall_score", "overall_median", "parental level of education", "race/ethnicity"], axis = 1)
y = df["overall_median"]

from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(x)
x = scaler.transform(x)
x = np.clip(x, a_min=-1, a_max=1)

models = [LogisticRegression(), GradientBoostingClassifier(), RandomForestClassifier(),AdaBoostClassifier()]

scores = []
names = []

for alg in models: 
    score = cross_val_score(alg, x, y, cv = RepeatedKFold(n_repeats = 3))
    scores.append(np.mean(score))    
    names.append(alg.__class__.__name__)
    print(alg.__class__.__name__, "trained")
    
sns.set_color_codes("muted")
sns.barplot(x=scores, y=names, color="g")

plt.xlabel('Accuracy')
plt.title('Classifier Scores')
plt.show()
lr_params = dict(     
    C = [n for n in range(1, 10)],     
    tol = [0.0001, 0.001, 0.001, 0.01, 0.1, 1],  
)

lr = LogisticRegression(solver='liblinear')
lr_cv = GridSearchCV(estimator=lr, param_grid=lr_params, cv=5) 
lr_cv.fit(x, y)
lr_est = lr_cv.best_estimator_
print(lr_cv.best_score_)
gb_params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(2, 5)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 40, 5)],
)

gb = GradientBoostingClassifier()
gb_cv = GridSearchCV(estimator=gb, param_grid=gb_params, cv=5) 
gb_cv.fit(x, y)
gb_est = gb_cv.best_estimator_
print(gb_cv.best_score_)
forest_params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 40, 5)],
)

forest = RandomForestClassifier()
forest_cv = GridSearchCV(estimator=forest, param_grid=forest_params, cv=5) 
forest_cv.fit(x, y)
forest_est = forest_cv.best_estimator_
print(forest_cv.best_score_)
ada_params = dict(          
    learning_rate = [0.05, 0.1, 0.15, 0.2],  
    n_estimators = [n for n in range(10, 40, 5)],
)

ada = AdaBoostClassifier()
ada_cv = GridSearchCV(estimator=ada, param_grid=ada_params, cv=5) 
ada_cv.fit(x, y)
ada_est = ada_cv.best_estimator_
print(ada_cv.best_score_)
cv_models = [lr_est, gb_est, forest_est, ada_est]

scores_cv = []
names_cv = []

for alg in cv_models: 
    score = cross_val_score(alg, x, y, cv = RepeatedKFold(n_repeats = 3))
    scores_cv.append(np.mean(score))    
    names_cv.append(alg.__class__.__name__)
    print(alg.__class__.__name__, "trained")
    
sns.set_color_codes("muted")
sns.barplot(x=scores_cv, y=names_cv, color="b")
sns.barplot(x=scores, y=names, color="g")

plt.xlabel('Accuracy')
plt.title('Classifier Scores')
plt.show()
col = df.drop(["math score", "writing score", "reading score", "overall_score", "overall_median", "parental level of education", "race/ethnicity"], axis = 1)
f = plt.figure(figsize=(5,10))

plt.subplot(3, 1, 1)
(pd.Series(forest_est.feature_importances_, index=col.columns).nlargest(10).plot(kind='barh') )

plt.subplot(3, 1, 2)
(pd.Series(gb_est.feature_importances_, index=col.columns).nlargest(10).plot(kind='barh') )

plt.subplot(3, 1, 3)
(pd.Series(ada_est.feature_importances_, index=col.columns).nlargest(10).plot(kind='barh') )
#ROC Curve
from sklearn.metrics import roc_curve, precision_recall_curve
fpr_model, tpr_model, thresholds_model = roc_curve(y, lr_cv.predict_proba(x)[:,1])
plt.plot(fpr_model, tpr_model, label = "LogisticRegression")

fpr_knn, tpr_knn, thresholds_knn = roc_curve(y, gb_cv.predict_proba(x)[:,1])
plt.plot(fpr_knn, tpr_knn, label = "GB")

fpr_knn, tpr_knn, thresholds_knn = roc_curve(y, forest_cv.predict_proba(x)[:,1])
plt.plot(fpr_knn, tpr_knn, label = "RFC")

fpr_rfc, tpr_rfc, thresholds_rfc = roc_curve(y, ada_cv.predict_proba(x)[:,1])
plt.plot(fpr_rfc, tpr_rfc, label = "ADA")
plt.xlabel("P(FP)")
plt.ylabel("P(TP)")
plt.legend(loc = "best")
#Recall Curve
precision_model, recall_model, thresholds_model = precision_recall_curve(y, lr_cv.predict_proba(x)[:,1])
plt.plot(precision_model, recall_model, label = "LogisticRegression")

precision_gb, recall_gb, thresholds_gb = precision_recall_curve(y, gb_cv.predict_proba(x)[:,1])
plt.plot(precision_gb, recall_gb, label = "GB")

precision_rfc, recall_rfc, thresholds_rfc = precision_recall_curve(y, forest_cv.predict_proba(x)[:,1])
plt.plot(precision_rfc, recall_rfc, label = "RFC")

precision_ada, recall_ada, thresholds_ada = precision_recall_curve(y, ada_cv.predict_proba(x)[:,1])
plt.plot(precision_ada, recall_ada, label = "Ada")
#Learning Curve
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle
x, y = shuffle(x, y)

train_sizes_abs, train_scores, test_scores = learning_curve(RandomForestClassifier(), x, y)
plt.plot(train_sizes_abs, np.mean(train_scores, axis = 1))
plt.plot(train_sizes_abs, np.mean(test_scores, axis = 1))
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# Read the file
data = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
data.head(5)
# Checking the available features
data.columns
# Replacing the ordinal data with categories to avoid misinterpretation by Model as Numerical data.

Education = {1: "Below College", 2: "College", 3:"Bachelor", 4:"Master", 5:"Doctor"}
EnvironmentSatisfaction = {1: "Low", 2: "Medium", 3:"High", 4:"Very High"}
JobInvolvement = {1: "Low", 2: "Medium", 3:"High", 4:"Very High"}
JobSatisfaction = {1: "Low", 2: "Medium", 3:"High", 4:"Very High"}
PerformanceRating = {1: "Low", 2: "Good", 3:"Excellent", 4:"Outstanding"}
RelationshipSatisfaction = {1: "Low", 2: "Medium", 3:"High", 4:"Very High"}
WorkLifeBalance = {1: "Bad", 2: "Good", 3:"Better", 4:"Best"}

data.replace({"Education": Education, "JobInvolvement":JobInvolvement, "JobSatisfaction": JobSatisfaction, 
              "PerformanceRating":PerformanceRating, "RelationshipSatisfaction":RelationshipSatisfaction,
             "WorkLifeBalance":WorkLifeBalance, "EnvironmentSatisfaction":EnvironmentSatisfaction}, inplace=True)
features = ['Age', 'DailyRate', 'DistanceFromHome', 'EmployeeCount','EmployeeNumber', 'HourlyRate', 
            'JobLevel', 'MonthlyIncome', 'MonthlyRate']
data[features].describe().loc[['min','max','mean']]

#### Observations
# Employee Count is same for all. This can be eliminated.
# EmployeeNumber is unique for all. Can be eliminated.
features = ['NumCompaniesWorked', 'PercentSalaryHike','StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
           'StandardHours']
data[features].describe().loc[['min','max','mean']]

#### Observations
# StandardHours is 80 for every record. Can be eliminated
features = ['YearsAtCompany', 'YearsInCurrentRole','YearsSinceLastPromotion', 'YearsWithCurrManager']
data[features].describe().loc[['min','max','mean']]
data.isna().any()
# None of the fields have null values
categoricalCols = [col for col in data.columns if data[col].dtype=='object']
for i in categoricalCols:
    print(str(data[i].value_counts()))
    print()
target = data.Attrition
# Columns with only one value for all rows, Columns with unique values per row and target column are dropped
predictors = data.drop(['Attrition', 'EmployeeCount', 'Over18','EmployeeNumber','StandardHours'], axis=1)
predictors.head()
#Separating the Categorical and Numerical data for further analysis.

categoricalCols = [col for col in predictors.columns if predictors[col].dtype=='object']
predictorsCategorical = predictors[categoricalCols]

numericCols = predictors.columns.difference(categoricalCols)
predictorsNumerical = predictors[numericCols]
num_data = pd.concat([predictorsNumerical.reset_index(drop=True),target],axis=1)
num_data.columns
fig, axarr = plt.subplots(4, 2, figsize=(12, 15))

# Age vs Attrition - Most attrition is at lower age
num_data[num_data['Attrition']=='Yes']['Age'].plot.hist(
    ax=axarr[0][0], alpha = 0.7, lw=3, color= 'b', label='Yes'
)
num_data[num_data['Attrition']=='No']['Age'].plot.hist(
    ax=axarr[0][0],  alpha = 0.3, lw=3, color= 'g', label='No'
)
axarr[0][0].set_xlabel("Age")
axarr[0][0].legend()
sns.boxplot(x='Age', y="Attrition", data=num_data, ax=axarr[0][1], palette="muted")


# TotalWorkingYears vs Attrition -- 
num_data[num_data['Attrition']=='Yes']['TotalWorkingYears'].plot.hist(
    ax=axarr[2][0], alpha = 0.7, lw=3, color= 'b', label='Yes'

)
num_data[num_data['Attrition']=='No']['TotalWorkingYears'].plot.hist(
    ax=axarr[2][0], alpha = 0.3, lw=3, color= 'g', label='No'
)
axarr[2][0].set_xlabel("TotalWorkingYears")
axarr[2][0].legend()
sns.boxplot(x='TotalWorkingYears', y="Attrition", data=num_data, ax=axarr[2][1], palette="muted")


# DistanceFromHome vs Attrition -- attrition at larger distances from home
num_data[num_data['Attrition']=='Yes']['DistanceFromHome'].plot.hist(
    ax=axarr[1][0], alpha = 0.7, lw=3, color= 'b', label='Yes'

)
num_data[num_data['Attrition']=='No']['DistanceFromHome'].plot.hist(
    ax=axarr[1][0], alpha = 0.3, lw=3, color= 'g', label='No'
)
axarr[1][0].set_xlabel("DistanceFromHome")
axarr[1][0].legend()
sns.boxplot(x='DistanceFromHome', y="Attrition", data=num_data, ax=axarr[1][1], palette="muted")


# YearsAtCompany vs Attrition -- Leave at younger age, but there are many outliers indicating that people leave at other ages also
num_data[num_data['Attrition']=='Yes']['YearsAtCompany'].plot.hist(
    ax=axarr[3][0], alpha = 0.7, lw=3, color= 'b', label='Yes'

)
num_data[num_data['Attrition']=='No']['YearsAtCompany'].plot.hist(
    ax=axarr[3][0], alpha = 0.3, lw=3, color= 'g', label='No'
)
axarr[3][0].set_xlabel("YearsAtCompany")
axarr[3][0].legend()
sns.boxplot(x='YearsAtCompany', y="Attrition", data=num_data, ax=axarr[3][1], palette="muted")
fig, axarr = plt.subplots(2,2, figsize=(12, 8))

# Job level vs Attrition -- High attrition at lower Job levels
num_data[num_data['Attrition']=='Yes']['JobLevel'].plot.hist(
    ax=axarr[0][0], alpha = 0.7, lw=3, color= 'b', label='Yes'
)
num_data[num_data['Attrition']=='No']['JobLevel'].plot.hist(
    ax=axarr[0][0],  alpha = 0.3, lw=3, color= 'g', label='No'
)
axarr[0][0].set_xlabel("JobLevel")
axarr[0][0].legend()
sns.boxplot(x='JobLevel', y="Attrition", data=num_data, ax=axarr[0][1], palette="muted")

# MonthlyIncome vs Attrition -- Higher attrition at lower monthly income
num_data[num_data['Attrition']=='Yes']['MonthlyIncome'].plot.hist(
    ax=axarr[1][0], alpha = 0.7, lw=3, color= 'b', label='Yes'

)
num_data[num_data['Attrition']=='No']['MonthlyIncome'].plot.hist(
    ax=axarr[1][0], alpha = 0.3, lw=3, color= 'g', label='No'
)
axarr[1][0].set_xlabel("MonthlyIncome")
axarr[1][0].legend()
sns.boxplot(x='MonthlyIncome', y="Attrition", data=num_data, ax=axarr[1][1], palette="muted")
fig,ax = plt.subplots(figsize=(10, 8))

y= [0 if i =='No' else 1 for i in num_data['Attrition']]
sns.kdeplot(data=num_data['YearsAtCompany'], data2=num_data['YearsInCurrentRole'])
plt.scatter(num_data['YearsAtCompany'], num_data['YearsInCurrentRole'], alpha=0.3, s=20, c=y, cmap='bwr')
plt.title('YearsAtCompany, YearsInCurrentRole, Attrition Comparison')
sns.jointplot(x=num_data["YearsInCurrentRole"], y=num_data["YearsAtCompany"], kind='kde')
sns.jointplot(x=num_data["Age"], y=num_data["YearsAtCompany"], kind='kde')
cat_data = pd.concat([predictorsCategorical.reset_index(drop=True),target],axis=1)
cat_data.columns
fig, axarr = plt.subplots(3, 2, figsize=(18, 15))
sns.countplot(x="BusinessTravel", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[0][0],palette="muted") 
# Higher attrition in travel Frequently
sns.countplot(x="OverTime", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[0][1],palette="muted")
# Higher attrition in Overtime
sns.countplot(x="Education", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[1][0],palette="muted")
sns.countplot(x="EducationField", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[1][1],palette="muted")
# Higher percentage attrition in merketing and tech degree
sns.countplot(x="JobInvolvement", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[2][0],palette="muted")
sns.countplot(x="JobSatisfaction", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[2][1],palette="muted")
# Extremely low attrition in Very High Job satisfaction
# higher attrition in "High" vs ["Very High and "Medium"]
fig, axarr = plt.subplots(2, 2, figsize=(15, 10))
sns.countplot(x="Gender", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[0][0],palette="muted")
sns.countplot(x="MaritalStatus", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[0][1],palette="muted")
# More attrition among single people
sns.countplot(x="RelationshipSatisfaction", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[1][0],palette="muted")
# More attrition among people with low relationship satisfaction
sns.countplot(x="WorkLifeBalance", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[1][1],palette="muted")
# High attrition among people with bad Work Life Balance, even though very few people have reported bad worklife balance
fig, axarr = plt.subplots(2, figsize=(19, 10))
# Higher percentage attrition in Sales
sns.countplot(x="Department", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[0],palette="muted")
# Higher attrition in Sales roles and Lab technician role
sns.countplot(x="JobRole", hue = "Attrition",dodge= True ,data=cat_data, ax=axarr[1],palette="muted")
# Overtime vs Department
sns.factorplot(x="OverTime", col="Department", col_wrap=4,hue="Attrition",
                   data=cat_data, kind ="count",palette="muted")
# JobInvolvement vs JobSatisfaction
sns.factorplot(x="JobInvolvement", col="JobSatisfaction", col_wrap=4,hue="Attrition",
                   data=cat_data, kind ="count",palette="muted")
total_Data = pd.concat([predictorsCategorical.reset_index(drop=True),predictorsNumerical.reset_index(drop=True),target],axis=1)
fig, ax = plt.subplots(figsize=(12, 8))
sns.violinplot(x="Department", y="MonthlyIncome", hue="Attrition", data=total_Data,split=True,inner="quartile",palette="muted");
fig, ax = plt.subplots(figsize=(19, 10))
sns.violinplot(x="JobRole", y="MonthlyIncome", hue="Attrition", data=total_Data,split=True,inner="quartile",palette="muted");
fig, ax = plt.subplots(figsize=(15, 10))
sns.violinplot(x="WorkLifeBalance", y="Age", hue="Attrition", data=total_Data,split=True,inner="quartile",palette="muted");
predictors = pd.concat([predictorsCategorical.reset_index(drop=True),predictorsNumerical.reset_index(drop=True)],axis=1)
corr = predictors.corr(method='pearson')
sns.set(style="white")
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(10, 8))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
from sklearn.preprocessing import LabelEncoder

# One hot encoding Categorical and Ordinal Variables
predictorsCategorical_encoded  = pd.get_dummies(predictorsCategorical)

# Encoding Target to binary 0 & 1
number = LabelEncoder()
y = number.fit_transform(target.astype('str'))
import statsmodels.api as sm
X = pd.concat([predictorsCategorical_encoded.reset_index(drop=True),predictorsNumerical.reset_index(drop=True)],axis=1)

X2 = sm.add_constant(X)
results = sm.OLS(y,X2).fit()
#results.pvalues[results.pvalues < 0.06].index
results.pvalues[results.pvalues > 0.5].index
predictorsNumerical_new = predictorsNumerical.drop(['DailyRate',
       'HourlyRate','JobLevel','MonthlyIncome','MonthlyRate',
       'PercentSalaryHike','StockOptionLevel','TotalWorkingYears','YearsAtCompany','TrainingTimesLastYear', 'YearsSinceLastPromotion',
       'YearsWithCurrManager'], axis=1)
predictorsCategorical_encoded = predictorsCategorical_encoded.drop(['Department_Human Resources', 'Education_Below College',
       'Education_Doctor', 'EnvironmentSatisfaction_High',
       'EnvironmentSatisfaction_Medium', 'JobRole_Manager',
       'JobRole_Sales Executive', 'MaritalStatus_Divorced',
       'MaritalStatus_Married', 'RelationshipSatisfaction_Medium',
       'RelationshipSatisfaction_Very High', 'WorkLifeBalance_Best',
       'WorkLifeBalance_Good'],axis=1)
X = pd.concat([predictorsCategorical_encoded.reset_index(drop=True),predictorsNumerical_new.reset_index(drop=True)],axis=1)
X.shape
value , counts = np.unique(y, return_counts=True)
dict(zip(value, counts))
from imblearn.over_sampling import SMOTE

oversampler=SMOTE(random_state=0)
smote_X, smote_y = oversampler.fit_sample(X,y)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import zero_one_loss
from sklearn.linear_model import LogisticRegression

train_X, val_X, train_y, val_y = train_test_split(smote_X,smote_y,train_size=0.8, 
                                                    test_size=0.2,random_state=0)
model1 = LogisticRegression()
model1.fit(train_X, train_y)
predicted = model1.predict(val_X)

print("Accuracy : "+ str(accuracy_score(predicted, val_y)))
print("Precision : "+str(precision_score(predicted, val_y)))
print("Recall : "+str(recall_score(predicted, val_y)))
print("F1-Score : "+str(f1_score(predicted, val_y)))
print("Confusion Matrix :")
print(confusion_matrix(predicted, val_y))
from sklearn.naive_bayes import BernoulliNB 

model2 = BernoulliNB ()
model2.fit(train_X, train_y)
predicted = model2.predict(val_X)

print("Accuracy : "+ str(accuracy_score(predicted, val_y)))
print("Precision : "+str(precision_score(predicted, val_y)))
print("Recall : "+str(recall_score(predicted, val_y)))
print("F1-Score : "+str(f1_score(predicted, val_y)))
print("Confusion Matrix :")
print(confusion_matrix(predicted, val_y))
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

model3 = DecisionTreeClassifier(criterion = "gini",min_samples_leaf=10)
model3.fit(train_X, train_y)
predicted = model3.predict(val_X)

print("Accuracy : "+ str(accuracy_score(predicted, val_y)))
print("Precision : "+str(precision_score(predicted, val_y)))
print("Recall : "+str(recall_score(predicted, val_y)))
print("F1-Score : "+str(f1_score(predicted, val_y)))
print("Confusion Matrix :")
print(confusion_matrix(predicted, val_y))
import operator
imp_dict = { X.columns[i]:imp for i,imp in enumerate(model3.feature_importances_)}
sorted_imp_dict = sorted(imp_dict.items(), key=operator.itemgetter(1), reverse=True)
sorted_imp_dict
from sklearn.ensemble import RandomForestClassifier

forest_model = RandomForestClassifier()
forest_model.fit(train_X, train_y)
predicted = forest_model.predict(val_X)
print("Accuracy : "+ str(accuracy_score(predicted, val_y)))
print("Precision : "+str(precision_score(predicted, val_y)))
print("Recall : "+str(recall_score(predicted, val_y)))
print("F1-Score : "+str(f1_score(predicted, val_y)))
print("Confusion Matrix :")
print(confusion_matrix(predicted, val_y))
imp_dict = { X.columns[i]:imp for i,imp in enumerate(forest_model.feature_importances_)}
sorted_imp_dict = sorted(imp_dict.items(), key=operator.itemgetter(1), reverse=True)
sorted_imp_dict
from xgboost import XGBClassifier
from xgboost import plot_importance

xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.05)
xgb_model.fit(train_X, train_y, verbose=False, eval_set=[(val_X, val_y)], early_stopping_rounds=5)
predicted = xgb_model.predict(val_X)

print("Accuracy : "+ str(accuracy_score(predicted, val_y)))
print("Precision : "+str(precision_score(predicted, val_y)))
print("Recall : "+str(recall_score(predicted, val_y)))
print("F1-Score : "+str(f1_score(predicted, val_y)))
print("Confusion Matrix :")
print(confusion_matrix(predicted, val_y))
plot_importance(xgb_model)
from sklearn.model_selection import KFold
from sklearn import svm

# Tuning to find the Penalty parameter on error term with least error on validation set
# This prevents us from overfitting to the training test
# 30 numbers evenly spaced between 10^-4 and 10^(2)
Cs = np.logspace(-4, 2, 30) 
print(Cs)

test_X = val_X
test_y = val_y

# Number of Folds
n_folds = 5
k_fold = KFold(n_folds)

c_scores = []
for C in Cs:
    fold_scores = []
    print("C = " + str(C))
    for k, (train, val) in enumerate(k_fold.split(train_X, train_y)):
        clf = svm.LinearSVC(C=C)
        clf.fit(train_X[train], train_y[train])
        ypred = clf.predict(train_X[val])
        yval = train_y[val]
        accuracy = np.sum(ypred==yval)/len(ypred)
        fold_scores.append(accuracy)
        
        print("\t[fold {0}] C: {1:.5f}, accuracy: {2:.5f}".
              format(k, C, accuracy))
    
    c_score = np.mean(fold_scores)
    c_scores.append(c_score)
    print("\tMean k-Fold score: " + str(c_score))
best_score_idx = np.argmax(c_scores)
best_c = Cs[best_score_idx]
print("Best C: " + str(best_c) + " with score: " + str(c_scores[best_score_idx]))
# Training th efull model with best C
svm_clf = svm.LinearSVC(C=best_c)
svm_clf.fit(train_X, train_y)
predicted = clf.predict(test_X)

print("Accuracy : "+ str(accuracy_score(predicted, val_y)))
print("Precision : "+str(precision_score(predicted, val_y)))
print("Recall : "+str(recall_score(predicted, val_y)))
print("F1-Score : "+str(f1_score(predicted, val_y)))
print("Confusion Matrix :")
print(confusion_matrix(predicted, val_y))
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

# Selecting the Hyperparamente K in K nearest neighbours classifier
Ks = list(range(1,20))

# Number of folds for cross-validation
n_folds = 5
k_fold = KFold(n_folds)

k_scores = []
for C in Ks:
    fold_scores = []
    print("K = " + str(C))
    for k, (train, val) in enumerate(k_fold.split(train_X, train_y)):
        clf = KNeighborsClassifier(n_neighbors=C)
        clf.fit(train_X[train], train_y[train])
        ypred = clf.predict(train_X[val])
        yval = train_y[val]
        accuracy = np.sum(ypred==yval)/len(ypred)
        fold_scores.append(accuracy)
        
        print("\t[fold {0}] K: {1:.5f}, accuracy: {2:.5f}".
              format(k, C, accuracy))
    
    k_score = np.mean(fold_scores)
    k_scores.append(k_score)
    print("\tMean k-Fold score: " + str(k_scores))
best_score_idx = np.argmax(k_scores)
best_k = Ks[best_score_idx]
print("Best K: " + str(best_k) + " with score: " + str(k_scores[best_score_idx]))
# Training the full model with best K
knn_clf = KNeighborsClassifier(n_neighbors=best_k)
knn_clf.fit(train_X, train_y)
predicted = knn_clf.predict(test_X)

print("Accuracy : "+ str(accuracy_score(predicted, val_y)))
print("Precision : "+str(precision_score(predicted, val_y)))
print("Recall : "+str(recall_score(predicted, val_y)))
print("F1-Score : "+str(f1_score(predicted, val_y)))
print("Confusion Matrix :")
print(confusion_matrix(predicted, val_y))
from keras import models
from keras import layers

oversampler=SMOTE(random_state=0)
smote_X, smote_y = oversampler.fit_sample(X,y)

train_X, val_X, train_y, val_y = train_test_split(smote_X,smote_y,train_size=0.8, 
                                                    test_size=0.2,random_state=0)

shallow_single_layer_model = models.Sequential()
shallow_single_layer_model.add(layers.Dense(1, activation='sigmoid', input_shape=(46,)))
shallow_single_layer_model.summary()
shallow_single_layer_model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                                   metrics=['accuracy'])
shallow_single_layer_model.fit(train_X, train_y, epochs=50, batch_size=10)
test_loss, test_acc = shallow_single_layer_model.evaluate(val_X, val_y)
print('Test accuracy:', test_acc)
# two-layer model
two_layer_model = models.Sequential()
two_layer_model.add(layers.Dense(20, activation='relu', input_shape=(46,)))
two_layer_model.add(layers.Dense(6, activation='relu'))
two_layer_model.add(layers.Dense(1, activation='sigmoid'))
two_layer_model.compile(optimizer='rmsprop',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
two_layer_model.summary()
two_layer_model.fit(train_X, train_y, epochs=50, batch_size=10,validation_split=0.8)
test_loss, test_acc = two_layer_model.evaluate(val_X, val_y)
print('Test accuracy:', test_acc)
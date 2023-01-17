# Import libraries necessary for this project

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.linear_model import LogisticRegression, RidgeClassifierCV

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier



from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.metrics import accuracy_score, fbeta_score, roc_auc_score, make_scorer



from xgboost import XGBClassifier, plot_importance



# Pretty display for notebooks

%matplotlib inline



sns.set_palette("Set2")
# Load the Census dataset

df_train = pd.read_csv("../input/udacity-mlcharity-competition/census.csv")



# Total number of records

n_records = df_train.shape[0]



# Print the results

print("Total number of records in training set: {}".format(n_records))
# Load the Census dataset

df_test = pd.read_csv("../input/udacity-mlcharity-competition/test_census.csv")



# Total number of records

n_records = df_test.shape[0]



# Print the results

print("Total number of records in testing set: {}".format(n_records))
data = [df_train, df_test]
df_train.head()
df_train.describe()
plt.figure(figsize=[10, 5])



fig = sns.countplot(data=df_train, x="income")



plt.title("Distribution of Income Class")

plt.xlabel("Income Class")

plt.ylabel("Number of Records")

plt.ylim(0, 40000)



income_means = df_train[df_train.income == '<=50K'].shape[0] / df_train.shape[0], df_train[df_train.income == '>50K'].shape[0] / df_train.shape[0]

i = 0



for bar in fig.patches:

    fig.annotate("{} ({:.2f}%)".format(bar.get_height(), 100 * income_means[i]), 

                 (bar.get_x() + bar.get_width() / 2., bar.get_height()), 

                 ha='center', 

                 va='center',

                 xytext=(0, 10),

                 textcoords = 'offset points',

                 fontsize=12,

                 fontweight='bold')

    i += 1
# Number of records where individual's income is more than $50,000

n_greater_50k = df_train[df_train['income'] == '>50K'].shape[0]



# Number of records where individual's income is at most $50,000

n_at_most_50k = df_train[df_train['income'] == '<=50K'].shape[0]



# Percentage of individuals whose income is more than $50,000

greater_percent = (n_greater_50k / n_records) * 100



print("Individuals making more than $50,000: {}".format(n_greater_50k))

print("Individuals making at most $50,000: {}".format(n_at_most_50k))

print("Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent))
# Encode the 'income' data to numerical values

df_train.income = df_train.income.replace(to_replace=['<=50K', '>50K'], value=[0, 1])
# get the education_level indexes by income average

educ_levels = df_train.groupby('education_level')['education-num'].mean().sort_values(ascending=False).index.tolist()



# grouping by education-num and calculating the income average

df_temp = df_train.groupby('education-num').income.mean().sort_index(ascending=False)



# get education-num indexes

educ_nums = df_temp.index



# get income average

income_means = df_temp.values



# creating a dataframe

df_temp = pd.DataFrame({'education_level': educ_levels, 'education-num': educ_nums, 'income average': income_means})



# sorting by education-num

df_temp = df_temp.sort_values('education-num', ascending=False)

df_temp
plt.figure(figsize=[10, 7])



# formating labels

labels = ["{} ({})".format(level, int(num)) for level, num in zip(df_temp['education_level'].values, df_temp['education-num'].values)]



sns.barplot(data=df_temp, x='income average', y='education_level', color=sns.color_palette()[0])



plt.ylabel('Education level and completed years of study')

plt.xlabel('% of people making more than 50k annually')

plt.title('Average of people that makes more than 50k annually in each education level')



plt.xlim(0, 1)

plt.yticks(np.arange(0, 16, 1), labels)

plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"]);
plt.figure(figsize=[10, 5])



fig = sns.barplot(data=df_train.groupby('income')['hours-per-week'].mean().reset_index(), x="income", y="hours-per-week")



plt.title("Mean hours worked per week of each income class")

plt.xlabel("Income Class")

plt.ylabel("Hours worked per week")

plt.ylim(0, df_train['hours-per-week'].max())





plt.xticks([0, 1], ["<=50k", ">50k"])



for bar in fig.patches:

    fig.annotate("{:.2f} hours per week".format(bar.get_height()), 

                 (bar.get_x() + bar.get_width() / 2., bar.get_height()), 

                 ha='center', 

                 va='center',

                 xytext=(0, 10),

                 textcoords = 'offset points',

                 fontsize=12,

                 fontweight='bold')
plt.figure(figsize=[12, 6])



income_by_hours = df_train.groupby('hours-per-week').income.mean().reset_index()

income_by_hours['hours-per-week'] = income_by_hours['hours-per-week'].astype(int)



ax = sns.barplot(data=income_by_hours, x='hours-per-week', y="income", palette=[sns.color_palette()[0] if x != 39 else 'red' for x in np.arange(0, 99, 1)],)

plt.errorbar(y=df_train.income.mean(), x=np.arange(-1, 97, 1), linestyle="--", color=sns.color_palette()[1])



ax.margins(0)



plt.tight_layout()

plt.xticks([0, 9, 19, 29, 39, 49, 59, 69, 79, 89], [1, 10, 20, 30, 40, 50, 60, 70, 80, 90])

plt.xlabel('Hours worked per week')

plt.ylabel('% of people making more than 50k annually')

plt.title('Average of people that makes more than 50k annually by hours worked per week');
plt.figure(figsize=[10, 5])



fig = sns.barplot(data=df_train.groupby('income')['age'].mean().reset_index(), x="income", y="age")



plt.title("Mean age of each income class")

plt.xlabel("Income Class")

plt.ylabel("Age")

plt.ylim(0, df_train['age'].max())





plt.xticks([0, 1], ["<=50k", ">50k"])



for bar in fig.patches:

    fig.annotate("{:.2f}".format(bar.get_height()), 

                 (bar.get_x() + bar.get_width() / 2., bar.get_height()), 

                 ha='center', 

                 va='center',

                 xytext=(0, 10),

                 textcoords = 'offset points',

                 fontsize=12,

                 fontweight='bold')
fig = sns.FacetGrid(df_train, hue='income', height=4, aspect=3)

fig.map(sns.kdeplot, 'age', shade=True)



fig.set(xlim=(0, df_train['age'].max()))



plt.title('Age density curve of each income class')

plt.xlabel('Age')

plt.ylabel('Probability density')

plt.legend(title='Income Class', labels=['<=50K', '>50K'])
print('Training set')

df_train.isna().sum()
print('Testing set')

df_test.isna().sum()
df_test.drop('Unnamed: 0', axis=1, inplace=True)
plt.figure(figsize = [13, 6])

base_color = sns.color_palette()[0]



labels = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]

labels_names = ['0', '250', '500', '750', '1000', '1250', '1500', '1750', '>=2000']



plt.subplot(1, 2, 1)

plt.hist(data=df_train, x='capital-gain', bins=25)

plt.ylim(0, 2000)

plt.ylabel('Number of Records')

plt.xlabel('Values')

plt.title('Capital Gain Distribution')

plt.yticks(labels, labels_names)



plt.subplot(1, 2, 2)

plt.hist(data=df_train, x='capital-loss', bins=25)

plt.ylim(0, 2000)

plt.xlabel('Values')

plt.title('Capital Loss Distribution')

plt.yticks(labels, labels_names)



plt.suptitle('Skewed Distributions of Continuous Census Data Features');
# Log-transform the skewed features

skewed = ['capital-gain', 'capital-loss']



for df in data:

    df[skewed] = df[skewed].apply(lambda x: np.log(x + 1))
plt.figure(figsize = [13, 6])

base_color = sns.color_palette()[0]



labels = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]

labels_names = ['0', '250', '500', '750', '1000', '1250', '1500', '1750', '>=2000']



plt.subplot(1, 2, 1)

plt.hist(data=df_train, x='capital-gain', bins=25)

plt.ylim(0, 2000)

plt.ylabel('Number of Records')

plt.xlabel('Values')

plt.title('Capital Gain Distribution')

plt.yticks(labels, labels_names)



plt.subplot(1, 2, 2)

plt.hist(data=df_train, x='capital-loss', bins=25)

plt.ylim(0, 2000)

plt.xlabel('Values')

plt.title('Capital Loss Distribution')

plt.yticks(labels, labels_names)



plt.suptitle('Log-Transformed Distributions of Continuous Census Data Features');
for df in data:

    df.drop('education_level', axis=1, inplace=True)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']



df_test[numerical] = df_test[numerical].transform(lambda x: x.fillna(x.mean()))



categorical=['workclass', 'marital-status', 'occupation', 'relationship', 'sex', 'race', 'native-country']

for cat in categorical:

    df_test[cat].fillna(df_test[cat].mode()[0], inplace=True)



print('Testing set')

df_test.isna().sum()
for df in data:

    df.sex = df.sex.replace(to_replace=['Male', 'Female'], value=[1, 0])
# One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()

df_train = pd.get_dummies(data=df_train)

df_test = pd.get_dummies(data=df_test)



# Print the number of features after one-hot encoding

encoded = list(df_train.columns)

print("{} total features after one-hot encoding.".format(len(encoded)))
# Import train_test_split

from sklearn.model_selection import train_test_split



# Split the 'features' and 'income' data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(df_train.drop('income', axis=1), 

                                                    df_train.income, 

                                                    test_size = 0.3, 

                                                    random_state = 0)



# Show the results of the split

print("Training set has {} samples.".format(X_train.shape[0]))

print("Testing set has {} samples.".format(X_test.shape[0]))
# Initialize a scaler, then apply it to the features

scaler = MinMaxScaler()



X_train[numerical] = scaler.fit_transform(X_train[numerical])

X_test[numerical] = scaler.transform(X_test[numerical])

df_test[numerical] = scaler.transform(df_test[numerical])
X_train[numerical].describe()
MODELS = [

    #Ensemble Methods

    GradientBoostingClassifier(random_state=42),

    RandomForestClassifier(random_state=42),

    

    #GLM

    RidgeClassifierCV(),

    

    #Navies Bayes

    GaussianNB(),

    

    #Nearest Neighbor

    KNeighborsClassifier(),

    

    #Discriminant Analysis

    LinearDiscriminantAnalysis(),



    #xgboost

    XGBClassifier(random_state=42)    

]



columns = ['Model Name', 'Train AUC Mean', 'Test AUC Mean', 'Test AUC STD * 3', 'Time']

models = pd.DataFrame(columns=columns)



row_index = 0

for ml in MODELS:

    model_name = ml.__class__.__name__

    models.loc[row_index, 'Model Name'] = model_name

    

    cv_results = cross_validate(ml, df_train.drop('income', axis=1), df_train.income, scoring=make_scorer(roc_auc_score), cv=5, return_train_score=True, return_estimator=True)

    

    models.loc[row_index, 'Time'] = cv_results['fit_time'].mean()

    models.loc[row_index, 'Train AUC Mean'] = cv_results['train_score'].mean()

    models.loc[row_index, 'Test AUC Mean'] = cv_results['test_score'].mean()

    models.loc[row_index, 'Test AUC STD * 3'] = cv_results['test_score'].std() * 3

    

    row_index+=1



models.sort_values(by=['Test AUC Mean'], ascending=False, inplace=True)

models.reset_index(drop=True, inplace=True)

models
plt.figure(figsize=[10, 7])



sns.barplot(x='Test AUC Mean', y='Model Name', data=models, color=sns.color_palette()[0])



plt.title('Machine Learning AUC Scores')

plt.xlabel('ROC AUC')

plt.ylabel('Machine Learning Model')



plt.xlim(0, 1);
importances = []

for i in range(10):

    rf = XGBClassifier()

    rf.fit(df_train.drop('income', axis=1), df_train.income)

    if len(importances) > 0:

        importances = [x + y for x, y in zip(importances, rf.feature_importances_)]

    else:

        importances = rf.feature_importances_



importances = [x / 10 for x in importances]

importances = pd.DataFrame({'feature': X_train.columns, 'importance': importances})

importances.sort_values('importance', ascending=False, inplace=True)



acc = []

for i in importances.importance.values:

    acc.append(i + acc[-1] if len(acc) > 0 else i)

importances['acc'] = acc
plt.figure(figsize=[15, 6])



sns.barplot(data=importances.loc[:10, :], x='feature', y='importance', color=sns.color_palette()[0])

plt.plot(importances.loc[:10, 'feature'], importances.loc[:10, 'acc'], '--', color=sns.color_palette()[1])

plt.ylabel('Importance')

plt.xlabel('Features')

plt.title('Accumulative Feature Importances')

plt.xticks(rotation=90)

plt.ylim(0, 1)



plt.show()
parameters = {

    'min_child_weight': [1, 5, 10],

    'gamma': [0.5, 1, 1.5, 2, 5],

    'subsample': [0.6, 0.8, 1.0],

    'colsample_bytree': [0.6, 0.8, 1.0],

    'max_depth': [3, 4, 5]

}



scorer = make_scorer(roc_auc_score)

grid_obj = GridSearchCV(estimator=XGBClassifier(random_state=42), param_grid=parameters, scoring=scorer, cv=5)

grid_fit = grid_obj.fit(X_train, y_train)

best_model = grid_fit.best_estimator_
best_predictions = best_model.predict(X_test)



# Report the before-and-afterscores

print("Unoptimized model")

# print("ROC/AUC on the testing data: {:.4f}".format(models.loc[0]['Test AUC Mean']))



print("\nOptimized Model")

print("Final ROC/AUC on the testing data: {:.4f}".format(roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])))
best_model
df_submission = pd.read_csv('../input/udacity-mlcharity-competition/test_census.csv')

df_submission.rename(columns={"Unnamed: 0": "id"}, inplace=True)
final_pred = best_model.predict_proba(df_test)[:, 1]
submission_data = pd.DataFrame(df_submission["id"])

submission_data["income"] = final_pred
submission_data.head()
submission_data.to_csv("./submission.csv", index=False)
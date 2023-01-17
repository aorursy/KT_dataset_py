import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
survey_data = pd.read_csv('../input/bank_customer_survey.csv')
#Sample of dataset
survey_data.head()
#Listing column names in the dataset
survey_data.columns
#number of rows in the dataset
survey_data.count()[0]
#Check if there are any missing values
survey_data.isnull().sum()
survey_data.describe()
survey_data[['job', 'y']].groupby("job").mean().reset_index()
some_data = survey_data[['job', 'y']].groupby("job").mean().reset_index().sort_values("y", ascending=False)
sns.barplot(y = "job", x = 'y',data = some_data)
some_data = survey_data[['marital', 'y']].groupby("marital").mean().reset_index().sort_values("y", ascending=False)
sns.barplot(y = "marital", x = 'y',data = some_data)
fig, axes = plt.subplots(1,1, figsize = (15,5))
sns.countplot(x = survey_data['education'], hue = survey_data["y"])
some_data = survey_data[['education', 'y']].groupby("education").mean().reset_index().sort_values("y", ascending=False)
sns.barplot(y = "education", x = 'y',data = some_data)
fig, axes = plt.subplots(1,1, figsize = (15,5))
sns.countplot(x = survey_data['default'], hue = survey_data["y"])
some_data = survey_data[['default', 'y']].groupby("default").mean().reset_index().sort_values("y", ascending=False)
sns.barplot(y = "default", x = 'y',data = some_data)
some_data = survey_data[['housing', 'y']].groupby("housing").mean().reset_index().sort_values("y", ascending=False)
sns.barplot(y = "housing", x = 'y',data = some_data)
some_data = survey_data[['loan', 'y']].groupby("loan").mean().reset_index().sort_values("y", ascending=False)
sns.barplot(y = "loan", x = 'y',data = some_data)
yes_summary = survey_data.groupby("y")
yes_summary.mean().reset_index()
pd.DataFrame(abs(survey_data.corr()['y']).reset_index().sort_values('y',ascending = False))
# Compute the correlation matrix
corr = survey_data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})
survey_data['education'].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in survey_data.columns:
    if(survey_data[col].dtype == 'object'):
        survey_data.loc[:,col] = le.fit_transform(survey_data.loc[:,col])
survey_data.head()
X = survey_data.iloc[:,:-1].values
y = survey_data.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed, n_jobs=-1)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print(results.mean())
from sklearn.ensemble import RandomForestClassifier
seed = 7
num_trees = 100
max_features = 3
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features, n_jobs=-1)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print(results.mean())
from sklearn.ensemble import ExtraTreesClassifier
seed = 7
num_trees = 100
max_features = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features, n_jobs=-1)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print(results.mean())
from sklearn.ensemble import AdaBoostClassifier
seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, y, cv=kfold, n_jobs=-1)
print(results.mean())
from sklearn.ensemble import GradientBoostingClassifier
seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print(results.mean())
from xgboost import XGBClassifier
seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = XGBClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print(results.mean())
from catboost import CatBoostClassifier
categorical_features_indices = [1, 2, 3, 4, 6, 7, 8, 9, 10, 15]
model=CatBoostClassifier(iterations=50, depth=10, learning_rate=0.1, loss_function='Logloss')
model.fit(x_train, y_train,cat_features=categorical_features_indices,eval_set=(x_test, y_test),plot=True)

print(model.get_best_score())

# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import VotingClassifier
# seed = 7
# kfold = model_selection.KFold(n_splits=10, random_state=seed)
# # create the sub models
# estimators = []
# model1 = LogisticRegression()
# estimators.append(('logistic', model1))
# model2 = DecisionTreeClassifier()
# estimators.append(('cart', model2))
# model3 = SVC()
# estimators.append(('svm', model3))
# # create the ensemble model
# ensemble = VotingClassifier(estimators)
# results = model_selection.cross_val_score(ensemble, X, y, cv=kfold, n)
# print(results.mean())
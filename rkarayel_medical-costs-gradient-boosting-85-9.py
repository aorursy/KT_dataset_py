%matplotlib inline

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from IPython.display import Image
data = pd.read_csv('../input/insurance/insurance.csv')
data.head(2)
data.shape
data.info()
data.isna().sum()
data.describe().T
data['Age_Bin'] = pd.cut(data['age'], bins=5, precision=0)
data['Age_Bin_str'] = data['Age_Bin'].astype(str).str.replace("(", "").str.replace("]", "").str.replace(", ", "-").str.replace(".0", "")+" years"
fig, axs = plt.subplots(2, figsize=(16,6))
axs[0].grid()
axs[0].axhline(y=0, color='grey', linewidth=4) 
axs[0].set_title('Charges by Age Bin', fontsize=16)
sns.barplot(x=data['Age_Bin_str'], y=data['charges'], ax=axs[0])

axs[1].grid()
axs[1].axhline(y=0, color='grey', linewidth=4) 
axs[1].set_title('Charges divided by Smoker and by Age Bins', fontsize=16)
sns.barplot(x=data['Age_Bin_str'], y=data['charges'], hue=data['smoker'], palette={'yes':'goldenrod', 'no':'seagreen'}, ax=axs[1])

plt.tight_layout()
plt.show()
plt.figure(figsize=(16,4))
sns.lineplot(x='age', y='charges', data=data, hue='smoker', err_style=None)
plt.show()
f = plt.figure(figsize=(16,4))
f.add_subplot(1,2,1)
sns.countplot(x=data['smoker'], hue=data['sex'], palette={'male':'royalblue', 'female':'violet'})

f.add_subplot(1,2,2)
data_smoker = data.query('smoker == "yes"')
order = data_smoker['Age_Bin_str'].value_counts().sort_index().index
sns.countplot(x=data_smoker['Age_Bin_str'], order=order)
plt.xlabel('Age Bin (Smoker==True)')

plt.grid()
plt.tight_layout()
plt.show()
filt_young_smokers = (data['Age_Bin_str']=="18-27 years") & (data['smoker']=='yes')
df_young_smokers = data.loc[filt_young_smokers]
sns.countplot(df_young_smokers['sex'])
plt.show()
values = data.groupby('region')['charges'].mean()
colors=['lightblue', 'lightgreen', 'bisque', 'lightcoral']
plt.figure(figsize=(6,6))
plt.pie(values, labels=values.index, autopct='%1.1f%%', colors=colors)
plt.title('Charges by Region', {'fontsize':16})
plt.show()
def bmi_categorizer(bmi):
    if (bmi < 18.5):
        return "Underweight"
    elif (bmi >= 18.5) and (bmi < 25):
        return "Normal weight"
    elif (bmi >= 25) and (bmi < 30):
        return "Overweight"
    elif (bmi >= 30) and (bmi < 35):
        return "Obese 1"
    elif (bmi >= 35) and (bmi < 40):
        return "Obese 2"
    elif (bmi >= 40):
        return "Obese 3"
data['BMI_Category'] = data['bmi'].apply(bmi_categorizer)
f = plt.figure(figsize=(16,4))
f.add_subplot(1,2,1)
sns.countplot(data['BMI_Category'], order=['Underweight', 'Normal weight', 'Overweight', 'Obese 1', 'Obese 2', 'Obese 3'])
f.add_subplot(1,2,2)
sns.boxplot(x=data['BMI_Category'], y=data['charges'], order=['Underweight', 'Normal weight', 'Overweight', 'Obese 1', 'Obese 2', 'Obese 3'])
plt.tight_layout()
plt.show()
plt.figure(figsize=(16,4))
sns.swarmplot(x='children', y='charges',
              data=data, hue='smoker',
              palette={'yes':'tab:orange', 'no':'tab:blue'}, edgecolor="gray")
plt.grid(b=False)
plt.show()
def get_livestyle(smoker, bmi_cat):
    if (smoker=='no') & (bmi_cat == 'Normal weight'):
        return 'Healthy'
    else:
        return 'Unhealthy'
data['livestyle'] = data.apply(lambda x: get_livestyle(x['smoker'], x['BMI_Category']), axis=1)
data.head()
order = data['Age_Bin_str'].value_counts().sort_index().index
plt.figure(figsize=(16,4))

ax = sns.countplot(x='Age_Bin_str', order=order, data=data[data['livestyle']=='Unhealthy'])
for p in ax.patches:
    ax.annotate(f'\n\n{p.get_height()}', (p.get_x()+0.4, p.get_height()), ha='center', va='top', color='white', size=18)
ax.grid(b=False)
ax.set_title("Livestyle == Unhealthy", fontsize=16)
plt.show()
plt.figure(figsize=(16,4))
ax = sns.countplot(x='region', data=data[data['livestyle']=='Unhealthy'], palette='Set2')
for p in ax.patches:
    ax.annotate(f'\n\n{p.get_height()}', (p.get_x()+0.4, p.get_height()), ha='center', va='top', color='black', size=18)
ax.grid(b=False)
ax.set_title("Livestyle == Unhealthy", fontsize=16)
plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(20,8))
sns.lineplot(x='age', y='charges', hue='livestyle', data=data, err_style=None, ax=ax1)
ax1.set_title('Charges Healthy vs Unhealthy (Male and Female)', fontsize=20, color='Red')
ax1.grid(b=False)

sns.lineplot(x='age', y='charges', hue='livestyle', data=data[data['sex']=='male'], err_style=None, ax=ax2)
ax2.set_title('Charges Only Males Healthy vs Unhealthy', fontsize=20, color='Red')
ax2.grid(b=False)

sns.lineplot(x='age', y='charges', hue='livestyle', data=data[data['sex']=='female'], err_style=None, ax=ax3)
ax3.set_title('Charges Only Females Healthy vs Unhealthy', fontsize=20, color='Red')
ax3.grid(b=False)

plt.grid(b=False)
plt.tight_layout()
plt.show()
female_df = data.loc[data['sex']=='female']
male_df = data.loc[data['sex']=='male']
order = data['Age_Bin_str'].value_counts().sort_index().index

fig, axs = plt.subplots(2,2, figsize=(20,6))

sns.barplot(x='Age_Bin_str', y='charges', data=female_df[female_df['livestyle']=='Healthy'], order=order, ax=axs[0, 0], ci=None)
axs[0, 0].set_title('Female Charges - Category: Healthy', fontsize=20, color='Red')

sns.barplot(x='Age_Bin_str', y='charges', data=male_df[male_df['livestyle']=='Healthy'], order=order, ax=axs[0, 1], ci=None)
axs[0, 1].set_title('Male Charges - Category: Healthy', fontsize=20, color='Red')

sns.barplot(x='Age_Bin_str', y='charges', data=female_df[female_df['livestyle']=='Unhealthy'], order=order, ax=axs[1, 0], ci=None)
axs[1, 0].set_title('Female Charges - Category: Unhealthy', fontsize=20, color='Red')
axs[1, 0].set_ylim(0, 20000)

sns.barplot(x='Age_Bin_str', y='charges', data=male_df[male_df['livestyle']=='Unhealthy'], order=order, ax=axs[1, 1], ci=None)
axs[1, 1].set_title('Male Charges - Category: Unhealthy', fontsize=20, color='Red')


plt.tight_layout(h_pad=2, w_pad=2)
plt.show()
avg_charges_female = np.mean(data.loc[data['sex']=='female', 'charges'])
avg_charges_male = np.mean(data.loc[data['sex']=='male', 'charges'])

print(f"Average charges for females: {round(avg_charges_female, 2)}$")
print(f"Average charges for males: {round(avg_charges_male, 2)}$")
X = data.drop(['charges', 'Age_Bin'], axis=1)
y = data['charges']
print(X.shape)
print(y.shape)
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
# low cardinality
lc_cols = [cname for cname in X.columns if X[cname].dtype == 'object' and X[cname].nunique()<10]
# high cardinality
hc_cols = [cname for cname in X.columns if X[cname].dtype == 'object' and X[cname].nunique()>10]
print(f"Numerical Columns: {numerical_cols}")
print(f"Categorical Low Cardinality Columns: {lc_cols}")
print(f"Categorical High Cardinality Columns: {hc_cols}")
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
num_transformer = Pipeline(steps=[
    ("Scale", StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('OHE', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numerical_cols),
    ('cat', cat_transformer, lc_cols)
])
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
models = [
    ('Lasso', Lasso(random_state=42)),
    ('ElasticNet', ElasticNet(random_state=42)),
    ('Linear Regression', LinearRegression()),
    ('KNeighborsRegressor', KNeighborsRegressor()),
    ('GradientBoostingRegressor', GradientBoostingRegressor(random_state=42)),
    ('DecisionTree', DecisionTreeRegressor(random_state=42)),
    ('RandomForest', RandomForestRegressor(random_state=42))
]
model_name=[]
mae_score_lst=[]
avg_mae_scores=[]
avg_r2_scores = []
for name, model in models:
    pipe = Pipeline(steps=[
        ('prep', preprocessor), ('model', model)
    ])
    mae_lst = -1*(cross_val_score(pipe, X, y, cv=10, scoring='neg_mean_absolute_error'))
    avg_mae = np.mean(mae_lst)
    avg_r2 = np.mean((cross_val_score(pipe, X, y, cv=10, scoring='r2')))*100
    model_name.append(name)
    mae_score_lst.append(mae_lst)
    avg_mae_scores.append(round(avg_mae, 2))
    avg_r2_scores.append(round(avg_r2, 2))
fig, axs = plt.subplots(1, 2, figsize=(18,4))

sns.boxplot(x=model_name, y=mae_score_lst, ax=axs[0])
axs[0].set_xticklabels(labels=model_name, rotation=30, horizontalalignment='right')
axs[0].set_title("Mean absolute errors", fontsize=20, color='red')

sns.barplot(x=model_name, y=avg_r2_scores, ax=axs[1])
axs[1].set_xticklabels(labels=model_name, rotation=30, horizontalalignment='right')
for p in axs[1].patches:
    axs[1].annotate(f'\n\n{p.get_height()}', (p.get_x()+0.4, p.get_height()), ha='center', va='top', color='white', size=18)
axs[1].set_title("Average R2 Scores", fontsize=20, color='red')
    
plt.tight_layout()
plt.show()
from sklearn.model_selection import GridSearchCV
gbc = GradientBoostingRegressor()
pipe = Pipeline(steps=[
        ('prep', preprocessor), 
        ('model', gbc)
    ])
parameters = {
    "model__n_estimators": [50, 100, 500, 1000],
    "model__max_depth": [1, 2, 4],
    "model__learning_rate": [0.001, 0.01, 0.1],
    'model__subsample':[0.68, 0.7, 0.72],
    'model__random_state':[1]
}
gridsearch = GridSearchCV(pipe, param_grid=parameters, cv=3, n_jobs=-1, scoring='r2')
gridsearch.fit(X_train, y_train)
gridsearch.best_params_
gridsearch.best_score_
best_model = gridsearch.best_estimator_
avg_mae_best_model = -1*(np.mean((cross_val_score(best_model, X, y, cv=10, scoring='neg_mean_absolute_error'))))
avg_r2_best_model = np.mean((cross_val_score(best_model, X, y, cv=10, scoring='r2')))
print(f"Best Model's average MAE: {avg_mae_best_model}")
print(f"Best Model's average R2: {avg_r2_best_model}")
y_pred_grid = best_model.predict(X_valid)
fig, ax = plt.subplots()
ax.scatter(y_valid, y_pred_grid,edgecolors=(0, 0, 1), s=40, alpha=0.6, c='c')
ax.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], color='red', lw=2)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()
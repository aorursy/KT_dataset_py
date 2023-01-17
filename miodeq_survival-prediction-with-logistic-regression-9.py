import operator
from fancyimpute import KNN 
from sklearn.preprocessing import LabelBinarizer
import math
from operator import itemgetter 
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, classification_report, r2_score, make_scorer, roc_curve, auc
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, StratifiedKFold, KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression

%matplotlib inline

plt.style.use('bmh')
columns = [
    # nominal
    'gender', #0-1
    'symptoms', #0-1
    'alcohol', #0-1
    'hepatitis b surface antigen', #0-1
    'hepatitis b e antigen', #0-1
    'hepatitis b core antibody', #0-1
    'hepatitis c virus antibody', #0-1
    'cirrhosis', #0-1
    'endemic countries', #0-1
    'smoking', #0-1
    'diabetes', #0-1
    'obesity', #0-1
    'hemochromatosis', #0-1
    'arterial hypertension', #0-1
    'chronic renal insufficiency', #0-1
    'human immunodeficiency virus', #0-1
    'nonalcoholic steatohepatitis', #0-1
    'esophageal varices', #0-1
    'splenomegaly', #0-1
    'portal hypertension', #0-1
    'portal vein thrombosis', #0-1
    'liver metastasis', #0-1
    'radiological hallmark', #0-1
    
    # integer
    'age', # age at diagnosis
    
    # continuous
    'grams of alcohol per day',
    'packs of cigarets per year',
    
    # ordinal
    'performance status',
    'encephalopathy degree',
    'ascites degree',
     
    # continuous   
    'international normalised ratio',
    'alpha-fetoprotein',
    'haemoglobin',
    'mean corpuscular volume',
    'leukocytes',
    'platelets',
    'albumin',
    'total bilirubin',
    'alanine transaminase',
    'aspartate transaminase',
    'gamma glutamyl transferase',
    'alkaline phosphatase',
    'total proteins',
    'creatinine',
    
    # integer
    'number of nodules',
    
    # continuous
    'major dimension of nodule cm',
    'direct bilirubin mg/dL',
    'iron',
    'oxygen saturation %',
    'ferritin',
        
    #nominal
    'class attribute', #0-1
]

columns = list([x.replace(' ', '_').strip() for x in columns])
df = pd.read_csv(
    '../input/hcc-data.csv', 
    names=columns, 
    header=None, 
    na_values=['?']
)
data = df.copy()
data.head()
data.info()
data.isnull().sum(axis=0)
data.describe()
data['age'].isnull().sum()
print('The oldest patient: {} years.'.format(data['age'].max()))
print('The youngest patient: {} years.'.format(data['age'].min()))
print('Average age: {} years.'.format(data['age'].mean()))
print('Median age: {} years.'.format(data['age'].median(skipna=True)))
plt.figure(figsize=(15,8))

sns.kdeplot(
    data.age[data.class_attribute == 1], 
    color="darkturquoise", 
    shade=True
)

sns.kdeplot(
    data.age[data.class_attribute == 0], 
    color="lightcoral", 
    shade=True
)

plt.legend(['Live', 'Died'])
plt.title('age vs class_attribute')
plt.xlim(0,100)
plt.show()
bins = [0, 20, 50, 75, 100]

out = pd.cut(
    data.age, 
    bins=bins,
    include_lowest=True
)

ax = out.value_counts(sort=False).plot.bar(
    rot=0, 
    color="g", 
    figsize=(20,10)
)

plt.xlabel('Age bins')
plt.ylabel('Count')
plt.show()
ax = data.age.hist(
    bins=15,
    color='teal', 
    alpha=0.8
)

ax.set(
    xlabel='Age', 
    ylabel='Count'
)

plt.show()
data['grams_of_alcohol_per_day'].isnull().sum()
max(data["grams_of_alcohol_per_day"])
min(data["grams_of_alcohol_per_day"])
data["grams_of_alcohol_per_day"].mean()
data["grams_of_alcohol_per_day"].median()
plt.figure(figsize=(15,8))

sns.kdeplot(
    data.grams_of_alcohol_per_day[data.class_attribute == 1], 
    color="darkturquoise", 
    shade=True
)

sns.kdeplot(
    data.grams_of_alcohol_per_day[data.class_attribute == 0], 
    color="lightcoral", 
    shade=True
)

plt.legend(['Live', 'Died'])
plt.title('grams_of_alcohol_per_day vs class_attribute')
plt.show()
bins = [
    0, 50, 100, 
    150, 200, 250, 
    300, 350, 400,
    450, 500
]

out = pd.cut(
    data.grams_of_alcohol_per_day, 
    bins=bins,
    include_lowest=True
)

ax = out.value_counts(sort=False).plot.bar(
    rot=0, 
    color='c', 
    figsize=(20,10)
)

plt.xlabel('Bins - grams_of_alcohol_per_day')
plt.ylabel('Count')
plt.show()
data['packs_of_cigarets_per_year'].isnull().sum()
max(data['packs_of_cigarets_per_year'])
min(data['packs_of_cigarets_per_year'])
data['packs_of_cigarets_per_year'].mean()
plt.figure(figsize=(15,8))

sns.kdeplot(
    data["packs_of_cigarets_per_year"][data.class_attribute == 1], 
    color="darkturquoise", 
    shade=True
)

sns.kdeplot(
    data["packs_of_cigarets_per_year"][data.class_attribute == 0], 
    color="lightcoral", 
    shade=True
)

plt.legend(['Live', 'Died'])
plt.title('packs_of_cigarets_per_year vs class_attribute')
plt.xlim(-45, 510)
plt.show()
data['class_attribute'].value_counts()
lives = 'lives'
died = 'died'

fig, axes = plt.subplots(
    nrows=1, 
    ncols=2,
    figsize=(10, 4)
)

# 0-women, 1-men
women = data[data['gender'] == 0]
men = data[data['gender'] == 1]

ax = sns.distplot(
    women[women['class_attribute'] == 1].age.dropna(), 
    bins=5, 
    label=lives, 
    ax=axes[0], 
    kde=False
)

ax = sns.distplot(
    women[women['class_attribute'] == 0].age.dropna(),
    bins=5, 
    label=died, 
    ax=axes[0], 
    kde=False
)

ax.legend()
ax.set_title('Women')

ax = sns.distplot(
    men[men['class_attribute'] == 1].age.dropna(), 
    bins=5, 
    label=lives, 
    ax=axes[1], 
    kde=False
)

ax = sns.distplot(
    men[men['class_attribute'] == 0].age.dropna(), 
    bins=5, 
    label = died, 
    ax = axes[1], 
    kde = False
)

ax.legend()
_ = ax.set_title('Men')
data['class_attribute'].isnull().sum()
data['gender'].isnull().sum()
data.groupby(['gender','class_attribute'])['class_attribute'].count()
f, ax=plt.subplots(
    1,
    2,
    figsize=(18,8)
)

data['class_attribute'].value_counts().plot.pie(
    explode=[0,0.1],
    autopct='%1.1f%%',
    ax=ax[0],
    shadow=True
)

ax[0].set_title('0-died, 1-lives')
ax[0].set_ylabel('')

sns.countplot(
    'class_attribute',
    data=data,ax=ax[1]
)

ax[1].set_title('0-died, 1-lives')

plt.show()
total_missing_data = data.isnull().sum().sort_values(ascending=False)

percent_of_missing_data = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat(
    [
        total_missing_data, 
        percent_of_missing_data
    ], 
    axis=1, 
    keys=['Total', 'Percent']
)

missing_data.head(15)
cons = data.loc[:, :]

cons['null_values'] = cons.isnull().sum(axis=1)

null_values = cons.drop('null_values', axis=1).isnull().sum()

sns.set_style("darkgrid")

plt.figure(figsize=(15,10))

pbar = null_values.plot.bar()

plt.xticks(
    list(range(0,len(null_values.index),6)), 
    list(null_values.index[0::6]), 
    rotation=45, 
    ha='left'
)

plt.show()
data.groupby('class_attribute').count()
data2 = data.drop(columns=['null_values'])
corr = data2.corr()

sns.heatmap(
    data=corr,
    annot=True,
    fmt='.2f',
    linewidths=.5,
    cmap='RdYlGn',
    xticklabels=corr.columns.values,
    yticklabels=corr.columns.values
)

fig = plt.gcf()
fig.set_size_inches(40, 20)

plt.show()
def prepare_missing_values_for_nans(df=None, columns=None):
    """
    Looking for the most frequent value for both decision classes outputs - 0,1.
    """
    
    to_update_nans_dict = {}
    
    if columns:
        for decision_class in [0, 1]:
            for column in columns:
                vals = df[df.class_attribute == decision_class][column].value_counts()
                
                to_update_nans_dict['{decision_class}_{column}'.format(
                    decision_class=decision_class,
                    column=column
                )] = vals.idxmax()
                
        return to_update_nans_dict
def replace_missing_values(df=None, columns=None, to_update_nans_dict=None):
    """
    Replacing NaN with the most frequent values for both decission classes outputs - 0,1.
    """
    
    df_list = []
    
    if columns:
        for decision_class in [0, 1]:
            _df = df[df.class_attribute == decision_class].reset_index(drop=True)

            for column in columns:        
                _df[column] = _df[column].fillna(
                    to_update_nans_dict['{}_{}'.format(decision_class, column)]
            )

            df_list.append(_df)

        return df_list
# replacing NaNs with the most frequent value in column
nominal_indexes = [
    1, 3, 4, 5, 
    6, 8, 9, 10, 
    11, 12, 13, 
    14, 15, 16, 
    17, 18, 19, 
    20, 21, 22
]

nominal_columns_to_discretize = list(itemgetter(*nominal_indexes)(columns))
# prepare missing values
nominal_dict = prepare_missing_values_for_nans(
    df=data2, 
    columns=nominal_columns_to_discretize
)

# replace NaN
missing_nominal_values_list = replace_missing_values(
    df=data2,
    columns=nominal_columns_to_discretize,
    to_update_nans_dict=nominal_dict

)

# data2[nominal_columns_to_discretize] = data2[nominal_columns_to_discretize].apply(
#     lambda x:x.fillna(x.value_counts().index[0])
# )
data2 = pd.concat(missing_nominal_values_list).reset_index(drop=True)
# KNN imputation
# Nearest neighbor imputations which weights samples 
# using the mean squared difference on features 
# for which two rows both have observed data.
continuous_indexes = [
    24,25,29,30,
    31,32,33,34,
    35,36,37,38,
    39,40,41,42,
    44,45,46,47,
    48]


continuous_columns_to_discretize = list(
    itemgetter(*continuous_indexes)(columns)
)

continuous_data = data2[continuous_columns_to_discretize].as_matrix()
# method 1 - KNN neighbours
X_filled_knn = KNN(k=3).complete(continuous_data)

data2[continuous_columns_to_discretize] = X_filled_knn
X_filled_knn.shape
integer_columns = ['age', 'number_of_nodules']

# prepare missing integer values
integer_dict = prepare_missing_values_for_nans(
    df=data2, 
    columns=integer_columns
)
integer_dict
# replace NaN
missing_integer_values_list = replace_missing_values(
    df=data2,
    columns=integer_columns,
    to_update_nans_dict=integer_dict

)
data2 = pd.concat(missing_integer_values_list).reset_index(drop=True)
data2['ascites_degree'].value_counts()
ordinal_columns = ['encephalopathy_degree', 'ascites_degree', 'performance_status']
# prepare missing ordinal values
ordinal_dict = prepare_missing_values_for_nans(
    df=data2, 
    columns=ordinal_columns
)
ordinal_dict
# replace NaN
missing_ordinal_values_list = replace_missing_values(
    df=data2,
    columns=ordinal_columns,
    to_update_nans_dict=ordinal_dict

)
data2 = pd.concat(missing_ordinal_values_list).reset_index(drop=True)
data2[data2.isnull().any(axis=1)]
ordinal_columns
binarized_data = []

for c in ordinal_columns:
    lb = LabelBinarizer()
    
    lb.fit(data2[c].values)
    
    binarized = lb.transform(data2[c].values)
    binarized_data.append(binarized)
binarized_ordinal_matrix_data = np.hstack(binarized_data)
binarized_ordinal_matrix_data
list(set(data2.number_of_nodules.values))
lb = LabelBinarizer()

lb.fit(data2.number_of_nodules.values)

binarized_number_of_nodules = lb.transform(data2.number_of_nodules.values)
data2['age_'] = data2.age.apply(lambda x: x / data2.age.max())
data2['age_'].head(10)
age_ = data2.age_.values.reshape(-1,1)
corr = data2.corr()

sns.heatmap(
    data=corr,
    annot=True,
    fmt='.2f',
    linewidths=.5,
    cmap='RdYlGn',
    xticklabels=corr.columns.values,
    yticklabels=corr.columns.values
)

fig = plt.gcf()
fig.set_size_inches(40, 20)

plt.show()
f, ax = plt.subplots(
    figsize=(12, 9)
)

sns.heatmap(
    data2.corr(), 
    vmax=.8, 
    square=True
)
to_drop_columns = [
    'age', 
    'encephalopathy_degree', 
    'ascites_degree', 
    'performance_status', 
    'number_of_nodules'
]

columns_set = set(columns)

_columns = list(columns_set.difference(to_drop_columns))
len(columns)
len(_columns)
X = data2[_columns].as_matrix()
y = data2.class_attribute.values
X_new = np.hstack((X, binarized_ordinal_matrix_data, age_, binarized_number_of_nodules))
X_new.shape
std_scaler = StandardScaler() #StandardScaler() # RobustScaler
X_new = std_scaler.fit_transform(X_new)
X_train, X_test, y_train, y_test = train_test_split(
    X_new,
    y,
    random_state=42,
    test_size=0.20
)
log_reg = LogisticRegression(
    solver='lbfgs',
    random_state=42,
    C=0.1,
    multi_class='ovr',
    penalty='l2',
)
log_reg.fit(X_train, y_train)
log_reg_predict = log_reg.predict(X_test)
log_reg.score(X_test, y_test)
preds = log_reg.predict(X_test)
print('\nLogistic Regression Accuracy: {:.2f}%'.format(accuracy_score(y_test, log_reg_predict) * 100))
print('Logistic Regression AUC: {:.2f}%'.format(roc_auc_score(y_test, log_reg_predict) * 100))
print('Logistic Regression Classification report:\n\n', classification_report(y_test, log_reg_predict))
kfold = StratifiedKFold(
    n_splits=5, 
    shuffle=True, 
    random_state=42
)

predicted = cross_val_predict(
    log_reg, 
    X_new, 
    y, 
    cv=kfold
)

scores = cross_val_score(
    log_reg, 
    X_new, 
    y, 
    cv=kfold,
    scoring='f1'
)

print('Cross-validated scores: {}\n'.format(scores))

print(classification_report(y, predicted))

print("LogisticRegression: F1 after 5-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    scores.mean() * 100,
    scores.std() * 2
))

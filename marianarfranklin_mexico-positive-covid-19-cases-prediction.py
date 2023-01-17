import pandas as pd
import numpy as np
import warnings

pd.set_option('display.max_columns', None) # shows all columns
warnings.filterwarnings('ignore')
# Read file into memory

# It's best to try first with the lastest remote file from repository
data = pd.read_csv('https://raw.githubusercontent.com/marianarf/covid19_mexico_analysis/master/mexico_covid19.csv')

# Local data
# data = pd.read_csv('../input/mexico-covid19-clinical-data/mexico_covid19.csv')
# Show first 5 rows
data.head(5)
# Quick glance at the data
data.describe()
# Make a copy of the data frame so that we don't override the original dataframe
df = data.copy()
# Take a glance at the variables
df.keys()
# Check dtypes for each column
# df.dtypes
# Exclude features as described above
df = df[df.columns[~df.columns.isin(
    ['id', 'ID_REGISTRO',
     'FECHA_ARCHIVO', 'FECHA_ACTUALIZACION', 'FECHA_DEF', 'FECHA_INGRESO','FECHA_SINTOMAS',
     'ABR_ENT', 'ENTIDAD', 'MIGRANTE', 'NACIONALIDAD', 'ORIGEN', 'PAIS_NACIONALIDAD', 'PAIS_ORIGEN',
     'INTUBADO', 'UCI'] # remove features that are only available while hospitalized
)]]
# Show the number of missing (NAN, NaN, na) data for each column
# data.isnull().sum()
# Various ways to check for NaN, NA and NULL
# df.isnull()
# df.isnull().sum()
# df.isnull().values.any()
# df.isnull().values.sum()
# df.isnull().any()
# There are a few rows without the city code attribute - so let's remove them
df = df[~df.isnull().any(axis=1)]
# We have data that contains either negative or positive results (i.e, excludes tests that are in process)
print(df['RESULTADO'].unique())
# The original data has different codes, but let's follow convention and refactor them (0=negative, 1=positive)
df['RESULTADO'] = df['RESULTADO'].astype(str).str.replace('2','0') # negative
df['RESULTADO'] = df['RESULTADO'].astype(str).str.replace('1','1') # positive
# Convert whole df to numeric
df = df.apply(pd.to_numeric, errors='ignore')
# We want every remaining column to be of numeric type
df.info()
# Rename target column as 'target' for clarity
df.rename(
    columns={'RESULTADO': 'target'},
    inplace=True
)
# Remove target variable to move it to the first position of dataframe
col_name = 'target'
first_col = df.pop(col_name)
# Now we can use Pandas insert() function and insert the opped column into first position of the dataframe
# The first argument of insert() function is the location we want to insert, here it is 0
df.insert(0, col_name, first_col)
# Now response variable is at the start of the data frame
df.head()
# Let's see how many observations and features we have
df.shape
import seaborn as sns 
import matplotlib.pyplot as plt

corrmat = df.corr() 
  
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax=ax, cmap='viridis', linewidths = 0.1)
# Print target incidence proportions and round to 3 decimal places
df.target.value_counts(normalize=True).round(3)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (15,6))

plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16

five_thirty_eight = [
    '#30a2da',
    '#fc4f30',
    '#e5ae38',
    '#6d904f',
    '#8b8b8b',
]

sns.set_palette(sns.color_palette(five_thirty_eight))

total = float(len(data))
results = ['Negative', 'Positive']  
ax = sns.countplot(x='target', data=df)
ax.set_xticklabels(results)

# add percentages above each bar
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            p.get_height(),
            '{:1.2f}'.format(height/total),
            ha='center')
    
plt.title('Incidence of COVID-19 in Mexico', fontsize=20)
plt.xlabel('Result')
plt.ylabel('Patients')

sns.set(style='ticks')
sns.despine()
import seaborn as sns
import matplotlib.pyplot as plt

df_positive = df.loc[df['target'] == 0]
df_negative = df.loc[df['target'] == 1]

five_thirty_eight = [
    '#30a2da',
    '#fc4f30',
    '#e5ae38',
    '#6d904f',
    '#8b8b8b',
]

plt.figure(figsize = (15,6))
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16

sns.set_palette(sns.color_palette(five_thirty_eight))

sns.distplot(df_positive['EDAD'])
sns.distplot(df_negative['EDAD'])

sns.set(style='ticks')
sns.despine()
import seaborn as sns
import matplotlib.pyplot as plt

five_thirty_eight = [
    '#30a2da',
    '#fc4f30',
    '#e5ae38',
    '#6d904f',
    '#8b8b8b',
]
    
var = df.columns.values

i = 0
t0 = df.loc[df['target'] == 0]
t1 = df.loc[df['target'] == 1]

plt.figure()
fig, ax = plt.subplots(8,4,figsize=(16,28))
plt.title('Class imbalance for features in data')

for feature in var:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(t0[feature], bw=0.5, label="Negative = 0")
    sns.kdeplot(t1[feature], bw=0.5, label="Positive = 1")
    sns.set_palette(sns.color_palette(five_thirty_eight))
    sns.set(style='ticks')
    sns.despine()
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();
from sklearn.model_selection import train_test_split

X = df.drop('target',axis=1)
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# First create dataframe to store the results
cols = ['Case','RndForest','LogReg','LGB']
result_tbl = pd.DataFrame(columns=cols)
result_tbl.set_index('Case',inplace=True)
result_tbl.loc['Standard'] = [0,0,0]
result_tbl.loc['GridSearch'] = [0,0,0]
result_tbl.loc['RandomSearch'] = [0,0,0]
result_tbl.loc['Hyperopt'] = [0,0,0]
result_tbl.head()
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# Instantiate models with default parameters

rf    = RandomForestClassifier(n_estimators=10)
lr    =  LogisticRegression(solver='liblinear')
lgg   = lgb.LGBMClassifier()
models = [rf,lr,lgg]

col = 0

for model in models:
    model.fit(X_train,y_train.values.ravel())
    result_tbl.iloc[0,col] = model.score(X_test,y_test)
    col += 1
result_tbl.head()
# Parameters for GridSearchCV

# Random Forest
n_estimators = [10, 100, 1000,10000]
max_features = ['sqrt', 'log2']
rf_grid = dict(n_estimators=n_estimators, max_features=max_features)

# Logistic Regrresion
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
lr_grid = dict(solver=solvers, penalty=penalty, C=c_values)

# LGB
scale_pos_weight = (161217/103095) # change this to not-manual input -_- pls
boosting_type = ['gbdt', 'goss', 'dart']
num_leaves = [30,50,100,150] #list(range(30, 150)),
learning_rate = list(np.logspace(np.log(0.005), np.log(0.2), base = np.exp(1), num = 10)) #1000
lgg_grid = dict(scale_pos_weight=scale_pos_weight, boosting_type=boosting_type, num_leaves=num_leaves, learning_rate=learning_rate)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

# This configuration depends on one's computing capacity and scope of time.
# The parameters in the previous cell are only some parameters available for the models.
# I haven't figured how to use GPU for the following optimization techniques, so I will use LightXGB.

models = [rf,lr,lgg]
grids = [rf_grid,lr_grid,lgg_grid]

col = 0
for ind in range(0,len(models)):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, 
                                 random_state=1)
    grid_search = GridSearchCV(estimator=models[col], 
                  param_grid=grids[col], n_jobs=-1, cv=cv,  
                  scoring='accuracy',error_score=0)
    grid_clf_acc = grid_search.fit(X_train, y_train)
    result_tbl.iloc[1,col] = grid_clf_acc.score(X_test,y_test)
    col += 1
result_tbl.head()
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

# This configuration depends on one's computing capacity and scope of time.
# The parameters in the previous cell are only some parameters available for the models.
# I haven't figured how to use GPU for the following optimization techniques, so I will use LightXGB.

models = [rf,lr,lgg]
grids = [rf_grid,lr_grid,lgg_grid]

col = 0
for ind in range(0,len(models)):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, 
                                 random_state=1)
    n_iter_search = 3
    random_search = RandomizedSearchCV(models[col],
    param_distributions=grids[col],n_iter=n_iter_search, cv=cv)
    random_search.fit(X_train,y_train)
    result_tbl.iloc[2,col] = random_search.score(X_test, y_test)
    col += 1
result_tbl.head()
import lightgbm as lgb

# Instantiate model

lgg = lgb.LGBMClassifier()
lgg.fit(X_train,y_train.values.ravel())
print(lgg.score(X_test,y_test))
# Predict the results
y_hat = lgg.predict(X_test)
from sklearn.metrics import accuracy_score

# View accuracy
accuracy = accuracy_score(y_hat, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_hat)))
# Compare train and test set accuracy
y_hat_train = lgg.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_hat_train)))
# Check for overfitting
print('Training set score: {:.4f}'.format(lgg.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(lgg.score(X_test, y_test)))
from sklearn.metrics import classification_report

# Classification metrics
print(classification_report(y_test, y_hat))
from sklearn.metrics import confusion_matrix

# Confussion matrix
cm = confusion_matrix(y_test, y_hat)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])
# Calculate probability again
y_hat = lgg.predict_proba(X_test) # Use this instead of `predict()`so that we can retrieve probabilities
pos_probs = y_hat[:,1] # Retrieve just the probabilities for the positive class
from sklearn.metrics import roc_curve

# Plot No Skill ROC Curve
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')

# Calculate ROC Curve for model
fpr, tpr, _ = roc_curve(y_test, pos_probs)

# Plot model ROC Curve
plt.plot(fpr, tpr, marker='.', label='LightGBM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

# Create No Skill classifier
dummy_model = DummyClassifier(strategy='stratified')
dummy_model.fit(X_train, y_train)
y_hat_dummy = dummy_model.predict_proba(X_test)
pos_probs_dummy = y_hat_dummy[:, 1]

# Calculate ROC AUC for No Skill
roc_auc = roc_auc_score(y_test, pos_probs_dummy)
print('No Skill ROC AUC %.3f' % roc_auc)

# Calculate ROC AUC for our model
roc_auc = roc_auc_score(y_test, pos_probs)
print('LightGBM ROC AUC %.3f' % roc_auc)
from sklearn.metrics import precision_recall_curve

# Calculate the no skill line as the proportion of the positive class
no_skill = len(y[y==1])/len(y)

# Plot the no skill precision-recall curve
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

# Calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, pos_probs)

# Plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='LightGBM')

# Calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, pos_probs)

# Call the plot
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

# Calculate the Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_test, pos_probs_dummy)
auc_score = auc(recall, precision)
print('No Skill PR AUC: %.3f' % auc_score)

# Calculate the Precision-Recall AUC for our model
precision, recall, _ = precision_recall_curve(y_test, pos_probs)
auc_score = auc(recall, precision)
print('LightGBM PR AUC: %.3f' % auc_score)
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# styling
pd.set_option('display.max_columns',150)
plt.style.use('bmh')
from IPython.display import display

# a bit of machine learning
from sklearn.metrics import recall_score, precision_score
from sklearn.cross_validation import KFold, train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV

%matplotlib inline
young = pd.read_csv('../input/responses.csv')
young.head(2)
young.describe()
nulls = young.isnull().sum().sort_values(ascending=False)
nulls.plot(
    kind='bar', figsize=(23, 5))
print('Number of girls who omitted weight field: {:.0f}'.format(
    young[young['Gender'] == 'female']['Weight'].isnull().sum()))
print('Number of boys who omitted weight field: {:.0f}'.format(
    young[young['Gender'] == 'male']['Weight'].isnull().sum()))
print('Number of girls who omitted height field: {:.0f}'.format(
    young[young['Gender'] == 'female']['Height'].isnull().sum()))
print('Number of boys who omitted height field: {:.0f}'.format(
    young[young['Gender'] == 'male']['Height'].isnull().sum()))
omitted = young[(young['Weight'].isnull()) | young['Height'].isnull()]
print('Number of people with omitted weight or height: {:.0f}'.format(omitted.shape[0]))
nas = omitted.drop(['Weight', 'Height', 'Number of siblings', 'Age'], 1).isnull().sum().sum()
print('Number of fields that were omitted by people who did not fill Weight or Height: {:.0f}'.format(nas))
var_of_interest = 'Village - town'
mapping = {var_of_interest: {'city': 0, 'village': 1}}
young.dropna(subset=[var_of_interest], inplace=True)
# to be able to use hue parameter for better comparison in seaborn
young["all"] = ""
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
sns.countplot(y=var_of_interest, data=young, ax=ax[0])
sns.countplot(y=var_of_interest, hue='Gender', data=young, ax=ax[1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
data = young.dropna(subset=['Height'])
sns.violinplot(x='Height', y = "all", hue=var_of_interest, data=data, split=True, ax = ax[0]);
data = young.dropna(subset=['Weight'])
sns.violinplot(x='Weight', y = "all", hue=var_of_interest, data=data, split=True, ax = ax[1]);

var_of_int_ser = young[var_of_interest]
sns.distplot(young[var_of_int_ser=='village'].Age.dropna(),
             label='village', ax=ax[2], kde=False, bins=30);

sns.distplot(young[var_of_int_ser=='city'].Age.dropna(),
             label='city', ax=ax[2], kde=False, bins=30);
ax[2].legend()

display(young[young['Height']<70][['Age', 'Height', 'Weight', 'Gender', var_of_interest]])
display(young[young['Weight']>120][['Age', 'Height', 'Weight', 'Gender', var_of_interest]])
young.drop([676,885,992, 859], inplace = True)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
data = young.dropna(subset=['Height'])
sns.violinplot(x='Height', y="all", hue=var_of_interest, data=data, 
                   split=True, ax=ax[0], inner='quartile');

data = young.dropna(subset=['Weight'])
sns.violinplot(x='Weight', y="all", hue=var_of_interest, data=data, 
                   split=True, ax=ax[1], inner='quartile');
young['BMI'] = round(young['Weight']/((young['Height']/100)**2),1)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,5))
data = young.dropna(subset=['BMI'])
sns.violinplot(x='BMI', hue=var_of_interest, y='Gender', data=data, 
               split=True, inner='quartile', ax=ax);

import scipy.stats as stats
city_bmi = data[data[var_of_interest]=='city'].BMI
village_bmi  = data[data[var_of_interest]=='village'].BMI
t, p = stats.ttest_ind(village_bmi, city_bmi, axis=0, equal_var=False)
print(' t-stat = {t} \n p-value = {p}'.format(t=t,p=p/2))

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,6))
data_under = data[data['Age']<20]
data_above = data[data['Age']>=20]
sns.violinplot(x='BMI', hue=var_of_interest, y='Gender', data=data_under, split=True, 
                   inner = 'quartile', ax=ax[0], hue_order=['village', 'city']);
sns.violinplot(x='BMI', hue=var_of_interest, y='Gender', data=data_above, split=True, 
                   inner = 'quartile', ax=ax[1], hue_order=['village', 'city']);

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,6))
sns.violinplot(x='BMI', hue=var_of_interest, y='Gender', data=data_under, 
               split=True, inner='stick', ax=ax, hue_order=['village', 'city']);

def do_ploting(x, y, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Correlation coefficient of the variables")
    sns.barplot(x=x, y=y, ax=ax)
    ax.set_ylabel("Correlation coefficients")


def correlation_plot(var_of_interest, df_main, mapping, figsize=(10, 30)):
    def calc_corr(var_of_interest, df, cols, figsize):
        lbls = []
        vals = []
        for col in cols:
            lbls.append(col)
            vals.append(np.corrcoef(df[col], df[var_of_interest])[0, 1])
        corrs = pd.DataFrame({'features': lbls, 'corr_values': vals})
        corrs = corrs.sort_values(by='corr_values')
        do_ploting(corrs.corr_values, corrs['features'], figsize)
        return corrs

    #imputing the set
    df = copy.deepcopy(df_main)
    df.replace(mapping, inplace=True)
    mean_values = df.mean(axis=0)
    df.fillna(mean_values, inplace=True)

    #correlating non-categorical varibales
    cols_floats = [col for col in df.columns if df[col].dtype != 'object']
    cols_floats.remove(var_of_interest)
    corrs_one = calc_corr(var_of_interest, df, cols_floats, figsize)

    #correlating categorical variables
    cols_cats = [col for col in df.columns if df[col].dtype == 'object']
    if cols_cats:
        df_dummies = pd.get_dummies(df[cols_cats])
        cols_cats = df_dummies.columns
        df_dummies[var_of_interest] = df[var_of_interest]
        corrs_two = calc_corr(var_of_interest, df_dummies, cols_cats, (5, 10))
    else:
        corrs_two = 0
    return [corrs_one, corrs_two]
corrs_area = correlation_plot(var_of_interest, young, mapping)
#The strongest correlations that we have are  
corr_num = corrs_area[0]
corr_cats = corrs_area[1]
display(corr_num[corr_num.corr_values == max(corr_num.corr_values)])
display(corr_num[corr_num.corr_values == min(corr_num.corr_values)])
display(corr_cats[corr_cats.corr_values == max(corr_cats.corr_values)])
display(corr_cats[corr_cats.corr_values == min(corr_cats.corr_values)])
good_columns = ['Dance', 'Folk', 'Techno, Trance','Religion', 'Medicine', 'Countryside, outdoors',
                'Spending on gadgets','Hypochondria','Western', 'Eating to survive', 
                'God', 'Chemistry', 'Gardening', 'Politics','Economy Management',
                'Branded clothing', 'Friends versus money','Number of siblings', 'Snakes',
                'Storm', 'Rats', 'Country', 'Dangerous dogs', 'Finances', 'Spiders', 
                'Entertainment spending', 'Horror', 'Pets', 'Prioritising workload', 'Dancing',
                'Biology', 'Final judgement', 'Sci-fi', 'Spending on looks']
fig, ax = plt.subplots(nrows=5, ncols=7 ,figsize=(30,40), sharex=True)
start = 0
for j in range(5):
    for i in range(7):
        if start == len(good_columns):
            break
        sns.barplot(y=good_columns[start], x=var_of_interest, data=young, ax=ax[j,i])
        ax[j,i].set_ylabel('')
        ax[j,i].set_xlabel('')
        ax[j,i].set_title(good_columns[start], fontsize=25)
        start += 1

features_cats = [col for col in young.columns if young[col].dtype=='object']
features_cats.remove(var_of_interest)
fig, ax = plt.subplots(nrows=2, ncols=5,figsize=(20,10), sharex=True)
start = 0
for j in range(2):
    for i in range(5):
        tab = pd.crosstab(young[var_of_interest], young[features_cats[start]])
        tab_prop = tab.div(tab.sum(1).astype(float), axis=0)
        tab_prop.plot(kind="bar", stacked=True, ax=ax[j,i] )
        start += 1
corr = young.corr()
# code: https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
# the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)

os = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False))
display(os.head(10))
display(os.tail(10))
corr = young[good_columns].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmax=.3,
    center=0,
    square=True,
    linewidths=.5,
    cbar_kws={"shrink": .5})
print (os['Final judgement'][0:2])
print (os['Religion'][0:2])
print (os['Entertainment spending'])
corr = young.corr()
os = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False))
display(os[abs(os)>0.5])
drop_colinera_cols = os[abs(os)>0.5].reset_index()['level_1']
clean_data = young.dropna(subset=[var_of_interest])
features_int = [col for col in clean_data.columns if clean_data[col].dtype!='object']
features_cats = [col for col in clean_data.columns if clean_data[col].dtype=='object']

features_int = list(set(features_int) - set(drop_colinera_cols))
print ('Number of features {:.0f}'.format(len(features_int)))
X = clean_data[features_int]
mean_values = X.mean(axis=0)
X = X.apply(lambda x: x.fillna(x.mean()),axis=0)
#X_cats = clean_data[features_cats].drop(var_of_interest, 1)
#X_cats = X_cats.drop('House - block of flats', 1)
#X_cats = pd.get_dummies(X_cats)
#print(X.shape)
#print(X_cats.shape)
Y = clean_data[var_of_interest]
for key, val in mapping[var_of_interest].items():
    Y.replace(key,val, inplace = True)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=100)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,5))

sns.kdeplot(X.Height,label = 'Before imputation', ax = ax[0]);
sns.kdeplot(clean_data.Height, label = 'After imputation', ax = ax[0]);
ax[0].set_title('Height');

sns.kdeplot(X.Age,label = 'Before imputation', ax = ax[1]);
sns.kdeplot(clean_data.Age, label = 'After imputation', ax = ax[1]);
ax[1].set_title('Age');

#sns.kdeplot(X.Weight,label = 'Before imputation', ax = ax[2])
#sns.kdeplot(clean_data.Weight, label = 'After imputation' , ax = ax[2])
#ax[2].set_title('Weight')
# standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# gridsearch for parameter tuning
from sklearn.linear_model import LogisticRegression
clr = LogisticRegression()
KF = KFold(len(x_train), n_folds=5)
param_grid = {'C':[.001,.01,.03,.1,.3,1,3,10]}
grsearch = GridSearchCV(clr, param_grid=param_grid, cv=KF, scoring = 'f1')
grsearch.fit(x_train, y_train)
print(grsearch.best_params_)

# fitting logistic regression and evaluating
clr = LogisticRegression(C=grsearch.best_params_['C'])
clr.fit(x_train, y_train)

mean_accuracy = np.mean(cross_val_score(clr, x_train, y_train, cv=KF))
print('Average accuracy score on CV set: {:.2f}'.format(mean_accuracy))

mean_f1 = np.mean(cross_val_score(clr, x_train, y_train, cv=KF, scoring = 'f1'))
print('Average f1 on CV set: {:.2f}'.format(mean_f1))
print('')
print('Accuracy score on test set is: {:.2f}'.format(clr.score(x_test, y_test)))
recall = recall_score(y_test, clr.predict(x_test))
print ('Recall on test: {:.2f}'.format(recall))
precision = precision_score(y_test, clr.predict(x_test))
print ('Presicion on test: {:.2f}'.format(precision))
print ('F1 score on test: {:.2f}'.format((2*recall*precision /(recall + precision))))

feat_coeff = pd.DataFrame({'features': X.columns,'impacts': clr.coef_[0]})
feat_coeff = feat_coeff.sort_values('impacts', ascending=False)

fig, ax1 = plt.subplots(1,1, figsize=(30,6));
sns.barplot(x=feat_coeff.features, y=feat_coeff.impacts, ax=ax1);
ax1.set_title('All features', size=30);
ax1.set_xticklabels(labels=feat_coeff.features, size=20, rotation=90);
ax1.set_ylabel('Impact', size=30);
top10 = pd.concat([feat_coeff.head(6),feat_coeff.tail(6)])
fig, ax1 = plt.subplots(1,1, figsize=(10,6))
sns.barplot(y=top10.features, x=top10.impacts, ax=ax1);
ax1.set_title('Top 12 features', size=20);
ax1.set_yticklabels(labels=top10.features, size=15);
ax1.set_xlabel('Impact', size=20);
display(os.tail(3))
cols_to_keep = ['Life struggles', 'Romantic', 'Shopping', 
                'Reading', 'Weight', 'Height', 'PC', 
                'Cars', 'Gender']
gender_map = {'Gender': {'female': 0, 'male': 1}}
corrs_dfs_gender = correlation_plot('Gender', young[cols_to_keep], gender_map, figsize=(5,5))
clean_data = young.dropna(subset=['Gender'])
features_int = [col for col in clean_data.columns if clean_data[col].dtype!='object']
X = clean_data[features_int]
mean_values = X.mean(axis=0)
X = X.apply(lambda x: x.fillna(x.mean()),axis=0)
Y = clean_data['Gender']
Y.replace('female',0, inplace = True)
Y.replace('male',1, inplace = True)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

clr = LogisticRegression()
clr.fit(X, Y)
feat_coeff = pd.DataFrame({'features': features_int,'impacts': clr.coef_[0]})
feat_coeff = feat_coeff.sort_values('impacts', ascending=False)

top10 = pd.concat([feat_coeff.head(8),feat_coeff.tail(8)])
fig, ax1 = plt.subplots(1,1, figsize=(10,6))
sns.barplot(y=top10.features, x=top10.impacts, ax=ax1);
ax1.set_title('Top 16 features', size=20);
ax1.set_yticklabels(labels=top10.features, size=15);
ax1.set_xlabel('Impact', size=20);
fig, ax = plt.subplots(nrows=15, ncols=8, figsize=(30, 70), sharex=True)
start = 0
for j in range(15):
    for i in range(8):
        sns.barplot(
            y=features_int[start], x=var_of_interest, data=young, ax=ax[j, i])
        ax[j, i].set_ylabel('')
        ax[j, i].set_xlabel('')
        ax[j, i].set_title(features_int[start], fontsize=25)
        start += 1

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(7, 2), sharex=True)

for i in range(4):
    sns.barplot(y=features_int[start], x=var_of_interest, data=young, ax=ax[i])
    ax[i].set_ylabel('')
    ax[i].set_xlabel('')
    ax[i].set_title(features_int[start], fontsize=10)
    start += 1
fig, ax = plt.subplots(nrows=15, ncols=9, figsize=(30, 70), sharex=True)
start = 0
for j in range(15):
    for i in range(9):
        sns.barplot(
            y=features_int[start],
            x='Left - right handed',
            data=young,
            ax=ax[j, i])
        ax[j, i].set_ylabel('')
        ax[j, i].set_xlabel('')
        ax[j, i].set_title(features_int[start], fontsize=25)
        start += 1

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(7, 2), sharex=True)
for i in range(4):
    sns.barplot(
        y=features_int[start], x='Left - right handed', data=young, ax=ax[i])
    ax[i].set_ylabel('')
    ax[i].set_xlabel('')
    ax[i].set_title(features_int[start], fontsize=10)
    start += 1
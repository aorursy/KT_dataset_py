import pandas as pd

import numpy as np

from scipy.stats import pearsonr, pointbiserialr

from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.base import clone

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from IPython.core.display import display, HTML

import missingno as msno

from itertools import combinations_with_replacement, combinations

from collections import defaultdict

import warnings

warnings.filterwarnings('ignore')



display(HTML("<style>.container { width:100% !important; }</style>"))
#train = pd.read_csv("D:/Kaggle_Data/Safe Driver/train.csv")  # could have used the na_values=-1 argument for automatic replacement of -1 with NaNs

#test = pd.read_csv("D:/Kaggle_Data/Safe Driver/test.csv")

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.shape
train.target.value_counts(normalize=True)*100
plt.figure(figsize=(4,4))

ax = sns.countplot(train.target)

ax.set_facecolor('white')
plt.figure(figsize=(4,4))

ax = sns.countplot(train.dtypes)

ax.set_facecolor('white')
pd.Series(train.columns)[:10]  # a trivial conversion to Series to see the names without the ugly single quotes
# Find the first underscore from the left and keep the remaining characters of the col.name

new_col_names = [s[s.find("_")+1:] for s in train.columns]
test_new_col_names = new_col_names[:]

test_new_col_names.remove('target')
pd.DataFrame.from_dict({"New Names":new_col_names, "Old Names":list(train.columns)})
train.columns = new_col_names

test.columns = test_new_col_names
# checking how many prefixes exist

prefixes=  set([s[:s.find("_")] for s in train.columns if "_" in s])

print(prefixes)
grouped_cols = defaultdict(list)



for prefix in prefixes:

    grouped_cols[prefix]=[col for col in train.columns if prefix in col]
grouped_cols['reg']
train.replace(-1,np.nan, inplace=True)

test.replace(-1,np.nan, inplace=True)
def display_nans(df):

    '''

    returns a dataframe with Number of NaNs in each column and also as a percentage of all rows in that column



    :param df: DataFrame containing NaNs. Type: pandas.DataFrame

    :return: DataFrame with indices as column names and columns as no. of NaN values and their percentage of # of rows.

    '''

    nans = pd.concat([df.isnull().sum(), (df.isnull().sum() / df.shape[0]) * 100], axis=1,

                     keys=['Num_NaN', 'NaN_Percent'])

    return nans[nans.Num_NaN > 0]
# Train NaNs

display_nans(train)
# Test NaNs

display_nans(test)
nans = pd.concat([train.isnull().sum(), (train.isnull().sum() / train.shape[0]) * 100], axis=1, keys=['Num_NaN', 'NaN_Percent'])

cols_with_nans = nans[nans.Num_NaN > 0].index

msno.matrix(df=train.loc[:,cols_with_nans], figsize=(20, 20), color=(0.24, 0.77, 0.77))
cols_with_nans_ind = [col for col in cols_with_nans if "ind" in col]



# ind_04_cat has a lot of target 1.0s than one might expect when its value is Null. One option 

# could be to create a new category "2" or something.



for col1, col2 in combinations(cols_with_nans_ind, 2):

    print(col1,col2, ":", end=" ")

    count_of_both_nans = len(train[train[col1].isnull()].index & train[train[col2].isnull()].index)

    print(count_of_both_nans, 'common indices')
train['ind_04_cat'].value_counts(dropna=False)
# Since 0 occurs more often and the values are categorical, we will impute it with the mode

train['ind_04_cat'].fillna(value=train['ind_04_cat'].mode()[0], inplace=True)  # dont forget the damn [0]

test['ind_04_cat'].fillna(value=train['ind_04_cat'].mode()[0], inplace=True)  # Test NaNs are filled with Train mode values



pd.DataFrame({'Train':train['ind_04_cat'].value_counts(dropna=False), 'Test':test['ind_04_cat'].value_counts(dropna=False)})
print('For ind_02_cat:', '\n', train.loc[train.ind_04_cat==0.0,'ind_02_cat'].value_counts())

print("*"*30)

print('For ind_05_cat:', '\n', train.loc[train.ind_04_cat==0.0,'ind_05_cat'].value_counts())
# So for both the mode value holds.

train.ind_02_cat.fillna(train.ind_02_cat.mode()[0], inplace=True)

train.ind_05_cat.fillna(train.ind_05_cat.mode()[0], inplace=True)



test.ind_02_cat.fillna(train.ind_02_cat.mode()[0], inplace=True)

test.ind_05_cat.fillna(train.ind_05_cat.mode()[0], inplace=True)
train.drop(['car_03_cat','car_05_cat'], axis=1, inplace=True)

test.drop(['car_03_cat','car_05_cat'], axis=1, inplace=True)
#Train

display_nans(train)
#Test

display_nans(test)
# starting with the easiest ones i.e. with the fewest values and seem to be ordinal

#train.car_01_cat.value_counts(dropna=False)  # uncheck one at a time to see the value counts

#train.car_02_cat.value_counts(dropna=False)

train.car_11.value_counts(dropna=False)
train.car_01_cat.fillna(train.car_01_cat.mode()[0], inplace=True)

test.car_01_cat.fillna(train.car_01_cat.mode()[0], inplace=True)



train.car_02_cat.fillna(train.car_02_cat.mode()[0], inplace=True)

test.car_02_cat.fillna(train.car_02_cat.mode()[0], inplace=True)



train.car_11.fillna(train.car_11.mode()[0], inplace=True)  # assuming car_11_cat column has nothing to do with this one

test.car_11.fillna(train.car_11.mode()[0], inplace=True)  # assuming car_11_cat column has nothing to do with this one



train.car_12.fillna(train.car_12.mean(), inplace=True)  # mean since car_12 is continuous

test.car_12.fillna(train.car_12.mean(), inplace=True)  # mean since car_12 is continuous
f, axarr = plt.subplots(1,4, figsize=(16,5))

train.plot(x="target", y="reg_03", ax=axarr[0], kind="scatter");

train.plot(x="target", y="car_09_cat",ax=axarr[1], kind="scatter");

train.plot(x="target", y="car_07_cat",ax=axarr[2], kind="scatter");

train.plot(x="target", y="car_14",ax=axarr[3], kind="scatter");
train.reg_03.fillna(train.reg_03.mean(),inplace=True)

test.reg_03.fillna(train.reg_03.mean(),inplace=True)



train.car_09_cat.fillna(train.car_09_cat.mode()[0],inplace=True)

test.car_09_cat.fillna(train.car_09_cat.mode()[0],inplace=True)



train.car_07_cat.fillna(train.car_07_cat.mode()[0], inplace=True)

test.car_07_cat.fillna(train.car_07_cat.mode()[0], inplace=True)



train.car_14.fillna(train.car_14.mode()[0], inplace=True)

test.car_14.fillna(train.car_14.mode()[0], inplace=True)
# Train

display_nans(train)
# Test

display_nans(test)
def draw_heatmap(filtered_cols, train, fmt='.1f', calc_corr=True):

    sub_train = train.loc[:,filtered_cols]

    f,ax = plt.subplots(figsize=(len(filtered_cols),len(filtered_cols)))

    if calc_corr:

        sns.heatmap(sub_train.corr(), annot=True, fmt= '.1f',ax=ax, vmin=0, vmax=1);

    else:

        sns.heatmap(train, annot=True, fmt=fmt,ax=ax);
prefix='calc'

filtered_cols = [col for col in grouped_cols[prefix] if ('bin' not in col) and ('cat' not in col)] + ['target']

draw_heatmap(filtered_cols, train)
prefix='reg'

filtered_cols = [col for col in grouped_cols[prefix] if ('bin' not in col) and ('cat' not in col)] + ['target']

draw_heatmap(filtered_cols, train)
prefix='ind'

filtered_cols = [col for col in grouped_cols[prefix] if ('bin' not in col) and ('cat' not in col)] + ['target']

draw_heatmap(filtered_cols, train)
prefix='ind'

filtered_cols = [col for col in grouped_cols[prefix] if ('bin' not in col) and ('cat' not in col)] + ["target"]

sub_train = train.loc[:,filtered_cols]

sns.pairplot(sub_train,size=2.5,hue="target");
prefix='car'

filtered_cols = [col for col in grouped_cols[prefix] if ('bin' not in col) and ('cat' not in col)] + ['target']

draw_heatmap(filtered_cols, train)
prefix='car'

filtered_cols = [col for col in grouped_cols[prefix] if ('bin' not in col) and ('cat' not in col)] + ["target"]

sub_train = train.loc[:,filtered_cols]

sns.pairplot(train,size=2.5, vars=filtered_cols,hue="target", plot_kws={'alpha':0.3});
sns.lmplot(x="car_13", y="car_15", hue="target", data=train,scatter_kws={'alpha':0.3});
sns.lmplot(x="car_12", y="car_15", hue="target", data=train,scatter_kws={'alpha':0.7});
[s for s in train.columns if "_bin" in s]
for column in [col for col in train.columns if "bin" in col]:

    train[column] = train[column].astype(bool)

    test[column] = test[column].astype(bool)



train['target'] = train['target'].astype(bool)
#defining a correlation dataframe maker



def correl_df_maker(filtered_cols, train, round_to=2):



    coeff_df = pd.DataFrame(columns=filtered_cols,index=filtered_cols)

    for idx,col in combinations_with_replacement(filtered_cols,2):



        if train[idx].dtype == bool and train[col].dtype == bool:

            coeff_df.loc[idx,col] = np.round_(np.sum(train[idx]==train[col])/train.shape[0],round_to)

            coeff_df.loc[col,idx] = coeff_df.loc[idx,col]

        elif train[idx].dtype == bool:

            coeff_df.loc[idx,col] = np.round_(pointbiserialr(train[idx].values, train[col].values)[0],round_to)

            coeff_df.loc[col,idx] = coeff_df.loc[idx,col]

        elif train[col].dtype == bool:

            coeff_df.loc[idx,col] = np.round_(pointbiserialr(train[col].values, train[idx].values)[0],round_to)

            coeff_df.loc[col,idx] = coeff_df.loc[idx,col]

        else:

            coeff_df.loc[idx,col] = np.round_(pearsonr(train[idx].values, train[col].values)[0],round_to)

            coeff_df.loc[col,idx] = coeff_df.loc[idx,col]

            

    return coeff_df.astype(float)
prefix='ind'



# first comparing all binary variables with one another

filtered_cols = [col for col in grouped_cols[prefix] if ('cat' not in col)] + ['target']

coeff_ind_df = correl_df_maker(filtered_cols, train)

coeff_ind_df
draw_heatmap(filtered_cols, coeff_ind_df, fmt='.2f', calc_corr=False)
prefix='calc'



# first comparing all binary variables with one another

filtered_cols = [col for col in grouped_cols[prefix] if ('cat' not in col)] + ['target']

correl_car_df = correl_df_maker(filtered_cols,train)

correl_car_df
draw_heatmap(filtered_cols, correl_car_df, fmt='.2f', calc_corr=False)
for column in ['ind_10_bin', 'ind_11_bin', 'ind_12_bin', 'ind_13_bin']:



    print("*"*15,column,"*"*15)

    print(classification_report(train.target.values, train[column].values))
train.drop(['ind_10_bin', 'ind_11_bin', 'ind_13_bin'], axis=1, inplace=True)

test.drop(['ind_10_bin', 'ind_11_bin', 'ind_13_bin'], axis=1, inplace=True)
for column in ['calc_18_bin', 'calc_20_bin']:



    print("*"*15,column,"*"*15)

    print(classification_report(train.target.astype(int).values, train[column].astype(int).values))
train.drop(['calc_18_bin'], axis=1, inplace=True)  # since its recall of 1s is better even though it has an overall worse f1-score

test.drop(['calc_18_bin'], axis=1, inplace=True)
for col in [col for col in train.columns if "cat" in col]:

    print(col, end="|")  

    df = pd.get_dummies(train[col],prefix=col).astype(bool)

    train.drop([col],axis=1,inplace=True)  # dropping the original columns

    train = pd.concat([train, df], axis=1)

    

    df = pd.get_dummies(test[col],prefix=col).astype(bool)

    test.drop([col],axis=1,inplace=True)  # dropping the original columns

    test = pd.concat([test, df], axis=1)
train.shape, test.shape  # train has target so all is well!
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,2:].values, train.iloc[:,1].values, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=50, class_weight="balanced", random_state=42)
rf_clf.fit(X_train, y_train)
y_train_pred = rf_clf.predict(X_train)

confusion_matrix(y_train,y_train_pred)
y_test_pred = rf_clf.predict(X_test)

confusion_matrix(y_test, y_test_pred)
dict(zip(train.columns[2:], np.round(rf_clf.feature_importances_*100,3)))
train.select_dtypes(include=[int,float]).describe()
columns_for_violin = list(train.iloc[:,1:].select_dtypes(include=[int,float]).columns) + ['target']  # target for hue

data = train[columns_for_violin]

data = pd.melt(data, id_vars='target', var_name='feature', value_name="value")

plt.figure(figsize=(len(columns_for_violin), len(columns_for_violin)))

sns.violinplot(x="feature", y="value", hue="target", data=data,split=True,inner="quart")

plt.xticks(rotation=90);
calc01_dist_df = pd.concat([train[train.target==True].calc_01.value_counts(),train[train.target==False].calc_01.value_counts()], axis=1)

calc01_dist_df.columns = ['True','False']

calc01_dist_df
f, axarr = plt.subplots(3,2, figsize=(16,5))

sns.distplot(train.car_13,ax=axarr[0,1])

sns.distplot(train.car_11,ax=axarr[1,1])

sns.distplot(train.car_15,ax=axarr[2,1])

sns.distplot(train.reg_02,ax=axarr[0,0])

sns.distplot(train.calc_01,ax=axarr[1,0])

sns.distplot(train.car_12,ax=axarr[2,0]);
sns.distplot(train.car_13);  # a long tail
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)
# To preserve the original train DataFrame, I will apply log on the split ones



for df in [train_df, val_df]:

    

    df.car_13 = df.car_13.apply(np.log2)
std_scaler = StandardScaler()



for col in train.select_dtypes(include=["int64","float64"]).columns:

    

    clone_scalr = clone(std_scaler)

    print(col, end=' | ')

    

    np_data_train = train_df[col].astype(np.float32).values.reshape(-1, 1)

    assert np.sum(np.isnan(np_data_train)) == 0, 'NaNs exist in Series converted to ndarray for train'

    np_data_val = val_df[col].astype(np.float32).values.reshape(-1, 1)

    assert np.sum(np.isnan(np_data_val)) == 0, 'NaNs exist in Series converted to ndarray for validation'



    np_data_train_t = np.round(clone_scalr.fit_transform(np_data_train).ravel(),4)

    assert np.sum(np.isnan(np_data_train_t)) == 0, 'NaNs exist in transformed ndarray for train'

    np_data_val_t = np.round(clone_scalr.transform(np_data_val).ravel(),4)

    assert np.sum(np.isnan(np_data_val_t)) == 0, 'NaNs exist in transformed ndarray for validation'

    

    train_df[col] = pd.Series(np_data_train_t, name=col, index=train_df.index)

    assert not train_df[col].isnull().any(), "NaNs exist in conversion of transformed ndarray to Series for train"

    val_df[col] = pd.Series(np_data_val_t, name=col, index=val_df.index)

    assert not val_df[col].isnull().any(), "NaNs exist in conversion of transformed ndarray to Series for validation"
sns.distplot(train_df.car_13);  # looks good!
columns_for_violin = list(train_df.select_dtypes(include=["float32"]).columns) +['target'] # target for hue

data = train_df[columns_for_violin]

data = pd.melt(data, id_vars='target', var_name='feature', value_name="value")

plt.figure(figsize=(len(columns_for_violin), len(columns_for_violin)))

sns.violinplot(x="feature", y="value", hue="target", data=data,split=True,inner="quart")

plt.xticks(rotation=90);
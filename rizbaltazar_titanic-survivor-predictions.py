# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import shap



import eli5

from eli5.sklearn import PermutationImportance



from itertools import combinations



from sklearn import preprocessing



from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LogisticRegression, LinearRegression



from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error, accuracy_score, balanced_accuracy_score, explained_variance_score

from sklearn.model_selection import train_test_split



# Categorical encoders

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import category_encoders as ce

import xgboost as xgb



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from pdpbox import pdp, get_dataset, info_plots



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sns.set(style="darkgrid")



train_full = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

test_full = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')



train_NA = train_full.isna().sum()

test_NA = test_full.isna().sum()



pd.concat([train_NA, test_NA], axis=1, sort = False, keys=['Train NA', 'Test NA']).transpose()
null_fare = test_full[test_full.Fare.isnull()].index[0]

test_full.loc[null_fare, 'Fare'] = 0
train_full.head()
y = train_full.Survived

train_X_full = train_full[train_full.columns.drop('Survived')]

combined = pd.concat([train_full, test_full], axis=0)
combined.info()
def alone(df):

    if ('SibSp' in df.columns) and ('Parch' in df.columns):

        df['Family'] = df['SibSp'] + df['Parch'] + 1

    

    le = LabelEncoder()

    

    df['le_Ticket'] = le.fit_transform(df['Ticket'])

    df['Same_Ticket'] = df.duplicated(['le_Ticket'])

    df['Alone'] = np.where((df['Family'] > 1) | (df['Same_Ticket']), False, True)

        

    plt.figure(figsize=(20, 12))

    sns.catplot(x="Alone", kind="count", palette="ch:.25", data=df)

    plt.title('Number of passengers travelling alone or not')

    

    df = df[df.columns.drop('SibSp', errors='ignore')]

    df = df[df.columns.drop('Parch', errors='ignore')]

    df = df[df.columns.drop('le_Ticket', errors='ignore')]

    df = df[df.columns.drop('Same_Ticket', errors='ignore')]

    return df
combined = alone(combined)

train_X_full = combined[combined.Survived.notnull()]

test_full = combined[combined.Survived.isnull()]
# For data visualization

temp = train_X_full.copy()

temp['Survived'] = y



sa = temp[(temp.Alone==1) & (temp.Survived==1)].Survived.count()

da = temp[(temp.Alone==1) & (temp.Survived==0)].Survived.count()

st = temp[(temp.Alone==0) & (temp.Survived==1)].Survived.count()

dt = temp[(temp.Alone==0) & (temp.Survived==0)].Survived.count()

da

plotdata = pd.DataFrame({

        'Survived': [sa, st],

        'Died': [da, dt],

    }, 

    index=["Yes", "No"]

)

print(plotdata)

plotdata.plot(kind="bar", stacked=True)

plt.title("Survivors travelling alone (or not)")

plt.xlabel("Alone")

plt.ylabel("Number of survivors")
stacked_data = plotdata.apply(lambda x: x*100/sum(x), axis=1)

print(stacked_data)

stacked_data.plot(kind="bar", stacked=True)

plt.title("Survivors travelling alone (or not) as a percentage")

plt.xlabel("Alone")

plt.ylabel("Number of survivors")
combined.head()
def fix_cabin(df):

    t = df.Cabin.fillna('U')

    df['Cabin'] = t.str.slice(0,1)

    

    plt.figure(figsize=(20, 15))

    sns.catplot(x="Cabin", kind="count", palette="ch:.25", data=df)

    plt.title('Number of passengers per cabin')

    

#     le = LabelEncoder()

#     df['Cabin'] = le.fit_transform(df['Cabin'])
# fix_cabin(train_X_full)



fix_cabin(combined)

train_X_full = combined[combined.Survived.notnull()]

test_full = combined[combined.Survived.isnull()]
temp = train_X_full.copy()

temp['Survived'] = y

temp['Cabin'].replace({8: 'U', 2: 'C', 4: 'E', 6: 'G', 3: 'D', 0: 'A', 1: 'B', 5: 'F', 7: 'T'}, inplace=True)

plotdata = pd.DataFrame({'Yes': [], 'No': []})

yes = []

no = []



for i in temp.Cabin.unique():

    yes.append(temp[(temp.Survived==1) & (temp.Cabin==i)].Cabin.count())

    no.append(temp[(temp.Survived==0) & (temp.Cabin==i)].Cabin.count())



plotdata['Yes'] = yes

plotdata['No'] = no

plotdata.rename(index={0: 'U', 1: 'C', 2: 'E', 3: 'G', 4: 'D', 5: 'A', 6: 'B', 7: 'F', 8: 'T'}, inplace=True)

plotdata.transpose()
plotdata.plot(kind="bar", stacked=True)

plt.title("Survivors by Cabin")

plt.xlabel("Cabin")

plt.ylabel("Passengers")
stacked_data = plotdata.apply(lambda x: x*100/sum(x), axis=1)

print(stacked_data.sort_values('Yes', ascending=False))

stacked_data.plot(kind="bar", stacked=True)

plt.title("Survivors by cabin (as a percentage)")

plt.xlabel("Cabin")

plt.ylabel("Percentage of survivors")
temp = train_X_full.copy()

temp['Survived'] = y

temp['Cabin'].replace({8: 'U', 2: 'C', 4: 'E', 6: 'G', 3: 'D', 0: 'A', 1: 'B', 5: 'F', 7: 'T'}, inplace=True)

plotdata = pd.DataFrame({'P1': [], 'P2': [], 'P3': []})

p1 = []

p2 = []

p3 = []



for i in temp.Cabin.unique():

    p1.append(temp[(temp.Pclass==1) & (temp.Cabin==i)].Cabin.count())

    p2.append(temp[(temp.Pclass==2) & (temp.Cabin==i)].Cabin.count())

    p3.append(temp[(temp.Pclass==3) & (temp.Cabin==i)].Cabin.count())



plotdata['P1'] = p1

plotdata['P2'] = p2

plotdata['P3'] = p3

plotdata.rename(index={0: 'U', 1: 'C', 2: 'E', 3: 'G', 4: 'D', 5: 'A', 6: 'B', 7: 'F', 8: 'T'}, inplace=True)

plotdata.transpose()
plotdata.plot(kind="bar", stacked=True)

plt.title("Classes by Cabin")

plt.xlabel("Cabin")

plt.ylabel("Passenger classes")
stacked_data = plotdata.apply(lambda x: x*100/sum(x), axis=1)

print(stacked_data)

stacked_data.plot(kind="bar", stacked=True)

plt.title("Classes by cabin (as a percentage)")

plt.xlabel("Cabin")

plt.ylabel("Percentage of survivors")
temp = train_X_full.copy()

temp['Survived'] = y

temp['Cabin'].replace({8: 'U', 2: 'C', 4: 'E', 6: 'G', 3: 'D', 0: 'A', 1: 'B', 5: 'F', 7: 'T'}, inplace=True)

plotdata = pd.DataFrame({'S': [], 'C': [], 'Q': []})

s = []

c = []

q = []



for i in temp.Cabin.unique():

    s.append(temp[(temp.Embarked=='S') & (temp.Cabin==i)].Cabin.count())

    c.append(temp[(temp.Embarked=='C') & (temp.Cabin==i)].Cabin.count())

    q.append(temp[(temp.Embarked=='Q') & (temp.Cabin==i)].Cabin.count())



plotdata['S'] = s

plotdata['C'] = c

plotdata['Q'] = q

plotdata.rename(index={0: 'U', 1: 'C', 2: 'E', 3: 'G', 4: 'D', 5: 'A', 6: 'B', 7: 'F', 8: 'T'}, inplace=True)

plotdata.transpose()
plotdata.plot(kind="bar", stacked=True)

plt.title("Embark location by Cabin")

plt.xlabel("Cabin")

plt.ylabel("Passenger embark location")
stacked_data = plotdata.apply(lambda x: x*100/sum(x), axis=1)

print(stacked_data)

stacked_data.plot(kind="bar", stacked=True)

plt.title("Embark location by cabin (as a percentage)")

plt.xlabel("Cabin")

plt.ylabel("Percentage of survivors")
def fix_embark(df):

    most_freq = df.Embarked.mode().iloc[0]

    df['Embarked'] = df.Embarked.fillna(most_freq)

    plt.figure(figsize=(20, 12))

    sns.catplot(x="Embarked", kind="count", palette="ch:.25", data=df)

    plt.title('Number of passengers by embark location')
# fix_embark(train_X_full)



fix_embark(combined)

train_X_full = combined[combined.Survived.notnull()]

test_full = combined[combined.Survived.isnull()]
temp = train_X_full.copy()

temp['Survived'] = y

plotdata = pd.DataFrame({'Yes': [], 'No': []})

yes = []

no = []



for i in temp.Embarked.unique():

    yes.append(temp[(temp.Survived==1) & (temp.Embarked==i)].Embarked.count())

    no.append(temp[(temp.Survived==0) & (temp.Embarked==i)].Embarked.count())



plotdata['Yes'] = yes

plotdata['No'] = no

plotdata.rename(index={0: 'S', 1: 'C', 2: 'Q'}, inplace=True)

plotdata.transpose()
plotdata.plot(kind="bar", stacked=True)

plt.title("Survivors by embark location")

plt.xlabel("Embark location")

plt.ylabel("Passengers")
stacked_data = plotdata.apply(lambda x: x*100/sum(x), axis=1)

print(stacked_data.sort_values('Yes', ascending=False))

stacked_data.plot(kind="bar", stacked=True)

plt.title("Survivors by embark location (as a percentage)")

plt.xlabel("Embark location")

plt.ylabel("Percentage of survivors")
# def encode_nominal(df, cols):

#     oh_encoder = OneHotEncoder()

#     oh_cols = pd.DataFrame(oh_encoder.fit_transform(df[cols]).toarray())

#     oh_cols.columns = oh_encoder.get_feature_names(cols)

#     oh_cols.index = df.index

#     df = df.join(oh_cols)

#     df = df.drop(columns=cols)

#     return df
# train_X_full = encode_nominal(train_X_full, ['Embarked', 'Sex', 'Alone'])

# train_X_full.head()
# Replace 'Name' with 'Title'

def extract_title(df):

    if 'Name' in df.columns:

        df['Title'] = df['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0].str.strip()

        df = df[df.columns.drop('Name')]

        

    df['Title'] = df['Title'].replace({'Ms': 'Miss', 'Mlle': 'Miss',

                                       'Mme': 'Mrs', 'Lady': 'Mrs', 'the Countess': 'Mrs', 'Dona': 'Mrs',

                                       'Don': 'Other', 'Rev': 'Other', 'Jonkheer': 'Other', 'Dr': 'Other',

                                       'Sir': 'Other', 'Col': 'Other', 'Capt': 'Other', 'Major': 'Other'})

        

    sns.catplot(kind="count", y="Title", palette="ch:.25", data=df, order=df['Title'].value_counts().index)

    plt.title('Number of passengers per Title')

#     le = LabelEncoder()

#     df['Title'] = le.fit_transform(df['Title'])

    return df
# train_X_full = extract_title(train_X_full)



combined = extract_title(combined)

train_X_full = combined[combined.Survived.notnull()]

test_full = combined[combined.Survived.isnull()]
temp = train_X_full.copy()

temp['Survived'] = y

print(temp['Title'].unique())

# titles = temp['Title'].unique()

# print(titles)
temp.Title.unique()
plotdata = pd.DataFrame({'Yes': [], 'No': []})

yes = []

no = []



for i in temp.Title.unique():

    yes.append(temp[(temp.Survived==1) & (temp.Title==i)].Title.count())

    no.append(temp[(temp.Survived==0) & (temp.Title==i)].Title.count())



plotdata['Yes'] = yes

plotdata['No'] = no

plotdata.rename(index={0: 'Mr', 1: 'Mrs', 2: 'Miss', 3: 'Master', 4: 'Other'}, inplace=True)

plotdata.transpose()
plotdata.plot(kind="bar", stacked=True)

plt.title("Survivors by title")

plt.xlabel("Title")

plt.ylabel("Passengers")
stacked_data = plotdata.apply(lambda x: x*100/sum(x), axis=1)

print(stacked_data.sort_values('Yes', ascending=False))

stacked_data.plot(kind="bar", stacked=True)

plt.title("Survivors by title (as a percentage)")

plt.xlabel("Title")

plt.ylabel("Percentage of survivors")
def extract_ticket(df):

    df['Ticket_Letters'] = df['Ticket'].str.replace('\d+', '')

#     df['Ticket_Numbers'] = df['Ticket'].str.replace('[^0-9]', '')

    df.loc[df['Ticket_Letters']=='','Ticket_Letters'] = 'NA'

    df.drop(columns=['Ticket'], inplace=True)

    return df
combined = extract_ticket(combined)

train_X_full = combined[combined.Survived.notnull()]

test_full = combined[combined.Survived.isnull()]

train_X_full.head()
combined['FareBin'] = pd.qcut(combined['Fare'], 4, labels=['cheap', 'average', 'expensive', 'costly'])

combined.head()
sns.catplot(data=combined, x='FareBin', y='Age', hue='Survived')
train_X_full = combined[combined.Survived.notnull()]

test_full = combined[combined.Survived.isnull()]
temp = train_X_full.copy()

temp['Survived'] = y

plotdata = pd.DataFrame({'Yes': [], 'No': []})

yes = []

no = []



for i in temp.FareBin.unique():

    yes.append(temp[(temp.Survived==1) & (temp.FareBin==i)].FareBin.count())

    no.append(temp[(temp.Survived==0) & (temp.FareBin==i)].FareBin.count())



plotdata['Yes'] = yes

plotdata['No'] = no

plotdata.rename(index={0: 'cheap', 1: 'average', 2: 'expensive', 3: 'costly'}, inplace=True)

plotdata.transpose()



plotdata.plot(kind="bar", stacked=True)

plt.title("Survivors by fare")

plt.xlabel("Fare")

plt.ylabel("Passengers")
stacked_data = plotdata.apply(lambda x: x*100/sum(x), axis=1)

print(stacked_data.sort_values('Yes', ascending=False))

stacked_data.plot(kind="bar", stacked=True)

plt.title("Survivors by fare (as a percentage)")

plt.xlabel("Fare")

plt.ylabel("Percentage of survivors")
r = combined.corr()

plt.figure(figsize=(10, 10))

sns.heatmap(r, annot=True, fmt='.2f', cmap="YlGnBu")

plt.title('Correlations before one-hot encoding')
temp = combined.copy()

to_one_hot = ['Sex', 'Embarked', 'Ticket_Letters', 'Cabin', 'Title', 'Alone', 'FareBin']



temp = pd.concat([temp, pd.get_dummies(temp[to_one_hot])], axis=1)

temp.drop(columns=to_one_hot, axis=1, inplace=True)

combined = temp

combined.head()
train_X_full = combined[combined.Survived.notnull()]

test_full = combined[combined.Survived.isnull()]
combined.info()
# Convert categorical features to numerical features using label encoder

def convert_cat(df):

    le = LabelEncoder()



    le_train_X = df.copy()



    # Encode categorical features

    s = df.dtypes=='object'

    cat_features = list(s[s].index)



    for col in cat_features:

        le_train_X[col] = le.fit_transform(df[col])



    return le_train_X
def predict_age(df):

    age_X = df[df.Age.notnull()]

    

    age_y = age_X.Age.astype(int)

    age_X = age_X[age_X.columns.drop('Age')]



    ma_X, va_X, ma_y, va_y = train_test_split(age_X, age_y, random_state=1)

#     rf_ma_model = RandomForestRegressor(random_state=0).fit(ma_X, ma_y)

    rf_ma_model = RandomForestRegressor().fit(ma_X, ma_y)



    pred_valid = rf_ma_model.predict(va_X)

    print('Mean absolute error: %.2f' %mean_absolute_error(pred_valid, va_y))

    sns.distplot(explained_variance_score(va_y, pred_valid))

    sns.distplot(pred_valid, kde=False)

    plt.title('Explained variance')

    return rf_ma_model
without_survived = combined.copy()

without_survived.drop(columns=['Survived'], inplace=True)

# without_survived.loc[null_fare]

without_survived.head()
missing_age_model = predict_age(without_survived)
missing_age = without_survived[without_survived.Age.isnull()]

# missing_age = missing_age[missing_age.columns.drop('Survived')]

missing_age = missing_age[missing_age.columns.drop('Age')]

pred_age = missing_age_model.predict(missing_age)

output_age = pd.DataFrame({'Age': pred_age}, index=missing_age.index)

output_age.transpose()
sns.distplot(output_age.Age, kde=False)

plt.title('Predicted missing ages')
sns.distplot(combined.Age, kde=False)

plt.title('Non-missing ages in training data set')
pd.DataFrame(combined.Age).transpose()
test = combined.copy()

test = test.combine_first(output_age)

pd.DataFrame(test.Age).transpose()
combined = test

train_X_full = combined[combined.Survived.notnull()]

test_full = combined[combined.Survived.isnull()]
y = train_X_full.Survived

train_X_full = train_X_full.drop(columns=['Survived'], axis=1)

train_X, valid_X, train_y, valid_y = train_test_split(train_X_full, y, random_state=5)
results = {}



for i in range(1,9):

    rf_model = RandomForestClassifier(n_estimators=i*50,random_state=0).fit(train_X, train_y)

    pred_valid = rf_model.predict(valid_X)

    results[i*50] = accuracy_score(pred_valid, valid_y)

    print('%d Mean absolute error: \t%.4f' %(i*50, mean_absolute_error(pred_valid, valid_y)))

    print('Accuracy score: \t%.4f' %accuracy_score(valid_y, pred_valid))

    

plt.plot(list(results.keys()), list(results.values()))

plt.xlabel('# of trees')

plt.ylabel('Accuracy score')

plt.show()
def scores(results):

    key_min = min(results.keys(), key=(lambda k: results[k]))

    key_max = max(results.keys(), key=(lambda k: results[k]))

    

    print('Highest score at %d of %.4f' %(key_max, results[key_max]))

    print('Lowest score at %d of %.4f' %(key_min, results[key_min]))

    return key_max, key_min
print('Number of trees: ')

high, low = scores(results)
best_rf_model = RandomForestClassifier(n_estimators=high,random_state=0).fit(train_X, train_y)

pred_valid1 = best_rf_model.predict(valid_X)

print('Mean absolute error: \t%.4f' %mean_absolute_error(pred_valid1, valid_y))

print('Accuracy score: \t%.4f' %accuracy_score(valid_y, pred_valid1))
perm = PermutationImportance(best_rf_model, random_state=1).fit(valid_X, valid_y)

eli5.show_weights(perm, feature_names=valid_X.columns.tolist())
row_to_show = 4

data_for_prediction = valid_X.iloc[row_to_show]

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)



best_rf_model.predict_proba(data_for_prediction_array)



# Create object that can calculate shap values

explainer = shap.TreeExplainer(best_rf_model)



# Calculate Shap values

shap_values = explainer.shap_values(data_for_prediction)



shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
row_to_show = 6

data_for_prediction = valid_X.iloc[row_to_show]

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)



best_rf_model.predict_proba(data_for_prediction_array)



# Create object that can calculate shap values

explainer = shap.TreeExplainer(best_rf_model)



# Calculate Shap values

shap_values = explainer.shap_values(data_for_prediction)



shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
# Calculate Shap values

shap_values_all = explainer.shap_values(valid_X)



shap.summary_plot(shap_values_all[1], valid_X)
feature_cols = train_X.columns

selector = SelectKBest(f_classif, k=len(train_X.columns))

X_new = selector.fit_transform(train_X[feature_cols], train_y)

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=train_X.index, 

                                 columns=feature_cols)

selected_features.head()

selected_cols = selected_features.columns[selected_features.var() != 0]

selected_cols
results1 = {}



for i in range(1, len(train_X.columns)):

    selector = SelectKBest(f_classif, k=i)

    X_new = selector.fit_transform(train_X[feature_cols], train_y)

    selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                     index=train_X.index, 

                                     columns=feature_cols)

    selected_cols = selected_features.columns[selected_features.var() != 0]

    kbest_X = train_X[selected_cols]

    kval_X = valid_X[selected_cols]

    rf_model = RandomForestClassifier(n_estimators=high,random_state=0).fit(kbest_X, train_y)

    pred_valid = rf_model.predict(kval_X)

    results1[i] = accuracy_score(pred_valid, valid_y)

    

plt.plot(list(results1.keys()), list(results1.values()))

plt.xlabel('# of features')

plt.ylabel('Accuracy score')

plt.show()
high1, low1 = scores(results1)
results2 = {}



for i in range(1, len(train_X.columns)):

    selector = SelectKBest(f_classif, k=i)

    X_new = selector.fit_transform(train_X[feature_cols], train_y)

    selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                     index=train_X.index, 

                                     columns=feature_cols)

    selected_cols = selected_features.columns[selected_features.var() != 0]

    kbest_X = train_X[selected_cols]

    kval_X = valid_X[selected_cols]

    rf_model = RandomForestClassifier(n_estimators=low,random_state=0).fit(kbest_X, train_y)

    pred_valid = rf_model.predict(kval_X)

    results2[i] = accuracy_score(pred_valid, valid_y)

    

plt.plot(list(results2.keys()), list(results2.values()))

plt.xlabel('# of features')

plt.ylabel('Accuracy score')

plt.show()
high2, low2 = scores(results2)
results3 = {}



for i in range(1, len(train_X.columns)):

    selector = SelectKBest(f_classif, k=i)

    X_new = selector.fit_transform(train_X[feature_cols], train_y)

    selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                     index=train_X.index, 

                                     columns=feature_cols)

    selected_cols = selected_features.columns[selected_features.var() != 0]

    kbest_X = train_X[selected_cols]

    kval_X = valid_X[selected_cols]

    log_model = LogisticRegression(random_state=0).fit(kbest_X, train_y)

    pred_valid = log_model.predict(kval_X)

    results3[i] = accuracy_score(pred_valid, valid_y)

    

plt.plot(list(results3.keys()), list(results3.values()))

plt.xlabel('# of features')

plt.ylabel('Accuracy score')

plt.show()
high3, low3 = scores(results3)
xg_model = xgb.XGBClassifier()



results4 = {}



for i in range(1, len(train_X.columns)):

    selector = SelectKBest(f_classif, k=i)

    X_new = selector.fit_transform(train_X[feature_cols], train_y)

    selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                     index=train_X.index, 

                                     columns=feature_cols)

    selected_cols = selected_features.columns[selected_features.var() != 0]

    kbest_X = train_X[selected_cols]

    kval_X = valid_X[selected_cols]

    xg_model = xgb.XGBClassifier().fit(kbest_X, train_y)

    pred_valid = xg_model.predict(kval_X)

    results4[i] = accuracy_score(pred_valid, valid_y)

    

plt.plot(list(results4.keys()), list(results4.values()))

plt.xlabel('# of features')

plt.ylabel('Accuracy score')

plt.show()
high4, low4 = scores(results4)
feature_cols = train_X.columns



selector = SelectKBest(f_classif, k=high1)

X_new = selector.fit_transform(train_X[feature_cols], train_y)

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=train_X.index, 

                                 columns=feature_cols)

selected_cols = selected_features.columns[selected_features.var() != 0]

kbest_X = train_X[selected_cols]

kval_X = valid_X[selected_cols]



best_k_model = RandomForestClassifier(n_estimators=high,random_state=0).fit(kbest_X, train_y)

# best_k_model = LogisticRegression(random_state=0).fit(kbest_X, train_y)

pred_valid1 = best_k_model.predict(kval_X)

print('Mean absolute error: \t%.4f' %mean_absolute_error(pred_valid1, valid_y))

print('Accuracy score: \t%.4f' %accuracy_score(valid_y, pred_valid1))
feature_cols = train_X.columns



selector = SelectKBest(f_classif, k=high4)

X_new = selector.fit_transform(train_X[feature_cols], train_y)

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=train_X.index, 

                                 columns=feature_cols)

selected_cols2 = selected_features.columns[selected_features.var() != 0]

kbest_X = train_X[selected_cols2]

kval_X = valid_X[selected_cols2]



xg_best_model = xgb.XGBClassifier().fit(kbest_X, train_y)

# best_k_model = LogisticRegression(random_state=0).fit(kbest_X, train_y)

pred_valid1 = xg_best_model.predict(kval_X)

print('Mean absolute error: \t%.4f' %mean_absolute_error(pred_valid1, valid_y))

print('Accuracy score: \t%.4f' %accuracy_score(valid_y, pred_valid1))
selected_cols
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_score

# cv_X = pd.concat([train_X[selected_cols], valid_X[selected_cols]])

# cv_y = pd.concat([train_y, valid_y])

cv_X = train_X_full[selected_cols]

cv_y = y

cv_X.head()
cv_results = {}

for i in range(2, 10):

    cv_score = cross_val_score(xg_best_model, cv_X, cv_y, cv=i)

#     print('Cross-validation score: %s' %(cv_score))

    cv_results[i] = cv_score.mean()
plt.plot(list(cv_results.keys()), list(cv_results.values()))

plt.xlabel('# of folds')

plt.ylabel('Score (average)')

plt.show()
cv_high, cv_low = scores(cv_results)
cv_score = cross_val_score(best_k_model, cv_X, cv_y, cv=cv_high)

print('Mean cross-validation score: %.2f' %(cv_score.mean()*100))
feature_cols = train_X.columns



selector = SelectKBest(f_classif, k=high3)

X_new = selector.fit_transform(train_X[feature_cols], train_y)

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=train_X.index, 

                                 columns=feature_cols)

selected_cols1 = selected_features.columns[selected_features.var() != 0]

kbest_X1 = train_X[selected_cols1]

kval_X1 = valid_X[selected_cols1]



best_k_model1 = LogisticRegression(random_state=0).fit(kbest_X1, train_y)

pred_valid1 = best_k_model1.predict(kval_X1)

print('Mean absolute error: \t%.4f' %mean_absolute_error(pred_valid1, valid_y))

print('Accuracy score: \t%.4f' %accuracy_score(valid_y, pred_valid1))



cv_X = pd.concat([train_X, valid_X])

cv_y = pd.concat([train_y, valid_y])

cv_X.head()
cv_results = {}

for i in range(2, 15):

    cv_score = cross_val_score(best_k_model1, cv_X, cv_y, cv=i)

#     print('Cross-validation score: %s' %(cv_score))

    cv_results[i] = cv_score.mean()

    

plt.plot(list(cv_results.keys()), list(cv_results.values()))

plt.xlabel('# of folds')

plt.ylabel('Score (average)')

plt.show()
cv_high, cv_low = scores(cv_results)
cv_score = cross_val_score(best_k_model1, cv_X, cv_y, cv=cv_high)

print('Mean cross-validation score: %.2f' %(cv_score.mean()*100))
test_full.drop(columns=['Survived'], axis=0, inplace=True)
test_full.info()
len(selected_cols)
ktest_X = test_full[selected_cols]

selected_cols
ktest_X.head()
pred = best_k_model.predict(ktest_X).astype('int')

pred
output = pd.DataFrame({'PassengerId': test_full.index,

                       'Survived': pred})

output.to_csv('survived.csv', index=False)
ktest_X1 = test_full[selected_cols1]

pred1 = best_k_model1.predict(ktest_X1).astype('int')

output = pd.DataFrame({'PassengerId': test_full.index,

                       'Survived': pred1})

output.to_csv('survived1.csv', index=False)
pred1
ktest_X2 = test_full[selected_cols2]

pred2 = xg_best_model.predict(ktest_X2).astype('int')

output = pd.DataFrame({'PassengerId': test_full.index,

                       'Survived': pred2})

output.to_csv('survived2.csv', index=False)
pred2
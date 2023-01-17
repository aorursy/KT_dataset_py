import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report



from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import SVC, LinearSVC
import lightgbm as lgbm

import xgboost as xgb
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



pd.set_option('display.max_columns', 100)
df = pd.read_csv(r"../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.head()
df.shape
df.info()
df.dtypes.value_counts()
df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
df['Churn'].value_counts()
df['Churn'].str.replace('No', '0').str.replace('Yes', '1').astype(int).plot.hist()
df.describe()
df.duplicated().sum()
df.isnull().sum()
df = df.drop(columns=['customerID'])
# example for the record strip non digit values

#test = pd.Series(["U$ 192.01"])

#test.str.replace('^[^\d]*', '').astype(float)



#df.TotalCharges = df.TotalCharges.str.replace('^[^\d]*', '')
df.iloc[0, df.columns.get_loc("TotalCharges")]
float(df.iloc[0, df.columns.get_loc("TotalCharges")])
df.iloc[488, df.columns.get_loc("TotalCharges")]
len(df[df['TotalCharges'] == ' '])
# replace missing values by 0

df.TotalCharges = df.TotalCharges.replace(" ",np.nan)



# drop missing values - side note: it represents only 11 out of 7043 rows which is not significant...

df = df.dropna()



# now we can convert the column type

df.TotalCharges = df.TotalCharges.astype('float')



df.shape
num_feat = df.select_dtypes(include=['float', 'int']).columns.tolist()

num_feat.remove('SeniorCitizen')    # SeniorCitizen is only a boolean

num_feat
sns.pairplot(data=df[num_feat])

plt.show()
plt.figure(figsize=(16, 10))



plt.subplot(2, 3, 1)

sns.distplot(df['tenure'])

plt.title('tenure')



plt.subplot(2, 3, 2)

sns.distplot(df['MonthlyCharges'])

plt.title('MonthlyCharges')



plt.subplot(2, 3, 3)

sns.distplot(df['TotalCharges'])

plt.title('TotalCharges')



plt.subplot(2, 3, 4)

sns.kdeplot(df.loc[df['Churn'] == 'No', 'tenure'], shade=True,label = 'Churn == 0')

sns.kdeplot(df.loc[df['Churn'] == 'Yes', 'tenure'], shade=True,label = 'Churn == 1')



plt.subplot(2, 3, 5)

sns.kdeplot(df.loc[df['Churn'] == 'No', 'MonthlyCharges'], shade=True,label = 'Churn == 0')

sns.kdeplot(df.loc[df['Churn'] == 'Yes', 'MonthlyCharges'], shade=True,label = 'Churn == 1')



plt.subplot(2, 3, 6)

sns.kdeplot(df.loc[df['Churn'] == 'No', 'TotalCharges'], shade=True,label = 'Churn == 0')

sns.kdeplot(df.loc[df['Churn'] == 'Yes', 'TotalCharges'], shade=True,label = 'Churn == 1')

corr = df.corr()

corr
# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(6, 4))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
for c in num_feat:

    plt.figure(figsize=(12, 1))

    sns.boxplot(df[c])

    plt.title(c)

    plt.show()
cat_features = df.select_dtypes('object').columns.tolist()

cat_features
plt.figure(figsize=(16, 20))

plt.subplots_adjust(hspace=0.4)



for i in range(len(cat_features)):

    plt.subplot(6, 3, i+1)

    sns.countplot(df[cat_features[i]])

    #plt.title(cat_features[i])



plt.show()
cat_features.remove('Churn')



plt.figure(figsize=(16, 20))

plt.subplots_adjust(hspace=0.4)



for i in range(len(cat_features)):

    plt.subplot(6, 3, i+1)

    sns.countplot(df[cat_features[i]], hue=df['Churn'])

    #plt.title(cat_features[i])



plt.show()
y = df.Churn.str.replace('No', '0').str.replace('Yes', '1').astype(int)
X = pd.get_dummies(data=df, columns=cat_features, drop_first=True)

X = X.drop(columns=['Churn'])
X.shape, y.shape
X['average_charges'] = X['TotalCharges'] / X['tenure']

X.loc[X['tenure'] == 0, 'average_charges'] = X['MonthlyCharges']

X.head()
num_feat.append('average_charges')

scaler = MinMaxScaler()

X[num_feat] = scaler.fit_transform(X[num_feat])
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)

rnd_clf.fit(X, y)
feature_importances = pd.DataFrame(rnd_clf.feature_importances_, index = X.columns,

                                    columns=['importance']).sort_values('importance', ascending=False)

feature_importances[:10]
plt.figure(figsize=(8, 10))

sns.barplot(x="importance", y=feature_importances.index, data=feature_importances)

plt.show()
# f1_score binary by default

def get_f1_scores(clf, model_name):

    y_train_pred, y_pred = clf.predict(X_train), clf.predict(X_test)

    print(model_name, f'\t - Training F1 score = {f1_score(y_train, y_train_pred) * 100:.2f}% / Test F1 score = {f1_score(y_test, y_pred)  * 100:.2f}%')
model_list = [RandomForestClassifier(),

    LogisticRegression(),

    SVC(),

    LinearSVC(),

    SGDClassifier(),

    lgbm.LGBMClassifier(),

    xgb.XGBClassifier()

             ]
model_names = [str(m)[:str(m).index('(')] for m in model_list]
for model, name in zip(model_list, model_names):

    model.fit(X_train, y_train)

    get_f1_scores(model, name)
rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

get_f1_scores(rfc, 'RandomForest')
y.sum(), len(y) - y.sum()
rfc = RandomForestClassifier(class_weight={1:1869, 0:5174})

rfc.fit(X_train, y_train)

get_f1_scores(rfc, 'RandomForest weighted')
lgbm_w = lgbm.LGBMClassifier(n_jobs = -1, class_weight={0:1869, 1:5174})

lgbm_w.fit(X_train, y_train)

get_f1_scores(lgbm_w, 'LGBM weighted')
ratio = ((len(y) - y.sum()) - y.sum()) / y.sum()

ratio
xgb_model = xgb.XGBClassifier(objective="binary:logistic", scale_pos_weight=ratio)

xgb_model.fit(X_train, y_train)

get_f1_scores(xgb_model, 'XGB with ratio')
abc = AdaBoostClassifier()

abc.fit(X_train, y_train)

get_f1_scores(abc, 'Adaboost')
print(classification_report(y_test, xgb_model.predict(X_test)))
from sklearn.model_selection import GridSearchCV
params = {'learning_rate':[0.175, 0.167, 0.165, 0.163, 0.17], 

          'max_depth':[1, 2, 3],

          'scale_pos_weight':[1.70, 1.73, 1.76, 1.79]}

clf_grid = GridSearchCV(xgb.XGBClassifier(), param_grid=params, cv=5, scoring='f1', n_jobs=-1, verbose=1)

clf_grid.fit(X_train, y_train)
clf_grid.best_score_
clf_grid.best_params_
lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,

          verbose=0, warm_start=False)
lr.fit(X_train, y_train)

get_f1_scores(lr, 'Logistic Reg')
xgb_model = xgb.XGBClassifier(objective="binary:logistic", learning_rate=0.167, max_depth=2, scale_pos_weight=1.73)

xgb_model.fit(X_train, y_train)

get_f1_scores(xgb_model, 'XGB with ratio')
y_pred_lr = lr.predict_proba(X_test)
lgbm_w = lgbm.LGBMClassifier(n_jobs = -1, class_weight={0:1869, 1:5174})

lgbm_w.fit(X_train, y_train)

y_pred_lgbm = lgbm_w.predict_proba(X_test)
# y_pred with predict_proba returns 2 cols, one for each class

y_pred_xgb[:5, 1]
y_pred_lgbm[:5, 1]
test = np.vstack((y_pred_lgbm[:5, 1], y_pred_xgb[:5, 1]))

test
np.mean(test, axis=0)
y_pred_mean = np.mean(np.vstack((y_pred_lgbm[:, 1], y_pred_xgb[:, 1])), axis=0)

y_pred_mean[:5]
y_pred_mean[y_pred_mean < 0.5] = 0

y_pred_mean[y_pred_mean > 0.5] = 1

y_pred_mean[:5]
print(f'F1 score of models combined on the test dataset = {f1_score(y_test, y_pred_mean)  * 100:.2f}%')
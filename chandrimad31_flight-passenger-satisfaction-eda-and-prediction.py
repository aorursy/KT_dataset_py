# Importing data

import pandas as pd

train = pd.read_csv('../input/airline-passenger-satisfaction/train.csv')

test = pd.read_csv('../input/airline-passenger-satisfaction/test.csv')
# Get row and column count

train.shape
# Get a snapshot of data

train.head(10)
# Drop unnecessary columns

train = train.drop('Unnamed: 0', axis=1)

train = train.drop('id', axis=1)
# Check size of the data set

train.info()
test.shape
test.head(10)
test = test.drop('Unnamed: 0', axis=1)

test = test.drop('id', axis=1)
test.info()
# Replace spaces in the column names with underscore

train.columns = [c.replace(' ', '_') for c in train.columns]
test.columns = [c.replace(' ', '_') for c in test.columns]
train['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1},inplace = True)
test['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1},inplace = True)
# Checking the nature of data set: balanced or imbalanced?

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8,5))

train.satisfaction.value_counts(normalize = True).plot(kind='bar', color= ['darkorange','steelblue'], alpha = 0.9, rot=0)

plt.title('Satisfaction Indicator (0) and (1) in the Dataset')

plt.show()
# Missing data

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing.head()
# Imputing missing value with mean

train['Arrival_Delay_in_Minutes'] = train['Arrival_Delay_in_Minutes'].fillna(train['Arrival_Delay_in_Minutes'].mean())
test['Arrival_Delay_in_Minutes'] = test['Arrival_Delay_in_Minutes'].fillna(test['Arrival_Delay_in_Minutes'].mean())
# Check the list of categorical variables

train.select_dtypes(include=['object']).columns
# Replace NaN with mode for categorical variables

train['Gender'] = train['Gender'].fillna(train['Gender'].mode()[0])

train['Customer_Type'] = train['Customer_Type'].fillna(train['Customer_Type'].mode()[0])

train['Type_of_Travel'] = train['Type_of_Travel'].fillna(train['Type_of_Travel'].mode()[0])

train['Class'] = train['Class'].fillna(train['Class'].mode()[0])
test['Gender'] = test['Gender'].fillna(test['Gender'].mode()[0])

test['Customer_Type'] = test['Customer_Type'].fillna(test['Customer_Type'].mode()[0])

test['Type_of_Travel'] = test['Type_of_Travel'].fillna(test['Type_of_Travel'].mode()[0])

test['Class'] = test['Class'].fillna(test['Class'].mode()[0])
import seaborn as sns

with sns.axes_style(style='ticks'):

    g = sns.catplot("satisfaction", col="Gender", col_wrap=2, data=train, kind="count", height=2.5, aspect=1.0)  

    g = sns.catplot("satisfaction", col="Customer_Type", col_wrap=2, data=train, kind="count", height=2.5, aspect=1.0)
with sns.axes_style('white'):

    g = sns.catplot("Age", data=train, aspect=3.0, kind='count', hue='satisfaction', order=range(5, 80))

    g.set_ylabels('Age vs Passenger Satisfaction')
with sns.axes_style('white'):

    g = sns.catplot(x="Flight_Distance", y="Type_of_Travel", hue="satisfaction", col="Class", data=train, kind="bar", height=4.5, aspect=.8)
with sns.axes_style('white'):

    g = sns.catplot(x="Departure/Arrival_time_convenient", y="Online_boarding", hue="satisfaction", col="Class", data=train, kind="bar", height=4.5, aspect=.8)
with sns.axes_style('white'):

    g = sns.catplot(x="Class", y="Departure_Delay_in_Minutes", hue="satisfaction", col="Type_of_Travel", data=train, kind="bar", height=4.5, aspect=.8)

    g = sns.catplot(x="Class", y="Arrival_Delay_in_Minutes", hue="satisfaction", col="Type_of_Travel", data=train, kind="bar", height=4.5, aspect=.8)
with sns.axes_style('white'):

    g = sns.catplot(x="Gate_location", y="Baggage_handling", hue="satisfaction", col="Class", data=train, kind="box", height=4.5, aspect=.8)
with sns.axes_style('white'):

    g = sns.catplot(x="Inflight_wifi_service", y="Inflight_entertainment", hue="satisfaction", col="Class", data=train, kind="box", height=4.5, aspect=.8)
with sns.axes_style(style='ticks'):

    g = sns.catplot("satisfaction", col="Ease_of_Online_booking", col_wrap=6, data=train, kind="count", height=2.5, aspect=.9)
with sns.axes_style(style='ticks'):

    g = sns.catplot("satisfaction", col="Seat_comfort", col_wrap=6, data=train, kind="count", height=2.5, aspect=.8)
with sns.axes_style(style='ticks'):

    g = sns.catplot("satisfaction", col="Cleanliness", col_wrap=6, data=train, kind="count", height=2.5, aspect=.8)
with sns.axes_style(style='ticks'):

    g = sns.catplot("satisfaction", col="Food_and_drink", col_wrap=6, data=train, kind="count", height=2.5, aspect=.8)
import matplotlib.pyplot as plt 

fig, axarr = plt.subplots(2, 2, figsize=(12, 8))



table1 = pd.crosstab(train['satisfaction'], train['Checkin_service'])

sns.heatmap(table1, cmap='Oranges', ax = axarr[0][0])

table2 = pd.crosstab(train['satisfaction'], train['Inflight_service'])

sns.heatmap(table2, cmap='Blues', ax = axarr[0][1])

table3 = pd.crosstab(train['satisfaction'], train['On-board_service'])

sns.heatmap(table3, cmap='pink', ax = axarr[1][0])

table4 = pd.crosstab(train['satisfaction'], train['Leg_room_service'])

sns.heatmap(table4, cmap='bone', ax = axarr[1][1])
from sklearn.preprocessing import LabelEncoder

lencoders = {}

for col in train.select_dtypes(include=['object']).columns:

    lencoders[col] = LabelEncoder()

    train[col] = lencoders[col].fit_transform(train[col])
lencoders_t = {}

for col in test.select_dtypes(include=['object']).columns:

    lencoders_t[col] = LabelEncoder()

    test[col] = lencoders_t[col].fit_transform(test[col])
Q1 = train.quantile(0.25)

Q3 = train.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
# Removing outliers from dataset

train = train[~((train < (Q1 - 1.5 * IQR)) |(train > (Q3 + 1.5 * IQR))).any(axis=1)]

train.shape
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

corr = train.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))

f, ax = plt.subplots(figsize=(20, 20))

cmap = sns.diverging_palette(150, 1, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .9})
from sklearn import preprocessing

r_scaler = preprocessing.MinMaxScaler()

r_scaler.fit(train)

#modified_data = pd.DataFrame(r_scaler.transform(train), index=train['id'], columns=train.columns)

modified_data = pd.DataFrame(r_scaler.transform(train), columns=train.columns)

modified_data.head()
from sklearn.feature_selection import SelectKBest, chi2

X = modified_data.loc[:,modified_data.columns!='satisfaction']

y = modified_data[['satisfaction']]

selector = SelectKBest(chi2, k=10)

selector.fit(X, y)

X_new = selector.transform(X)

print(X.columns[selector.get_support(indices=True)])
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier as rf



X = train.drop('satisfaction', axis=1)

y = train['satisfaction']

selector = SelectFromModel(rf(n_estimators=100, random_state=0))

selector.fit(X, y)

support = selector.get_support()

features = X.loc[:,support].columns.tolist()

print(features)

print(rf(n_estimators=100, random_state=0).fit(X,y).feature_importances_)
import warnings

warnings.filterwarnings("ignore")
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(rf(n_estimators=100, random_state=0).fit(X,y),random_state=1).fit(X,y)

eli5.show_weights(perm, feature_names = X.columns.tolist())
features = ['Type_of_Travel','Inflight_wifi_service','Online_boarding','Seat_comfort','Flight_Distance',

            'Inflight_entertainment','On-board_service','Leg_room_service','Cleanliness','Checkin_service', 

            'Inflight_service', 'Baggage_handling']

target = ['satisfaction']



# Split into test and train

X_train = train[features]

y_train = train[target].to_numpy()

X_test = test[features]

y_test = test[target].to_numpy()



# Normalize Features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
import time

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, plot_confusion_matrix, plot_roc_curve

from matplotlib import pyplot as plt 

def run_model(model, X_train, y_train, X_test, y_test, verbose=True):

    t0=time.time()

    if verbose == False:

        model.fit(X_train,y_train.ravel(), verbose=0)

    else:

        model.fit(X_train,y_train.ravel())

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_pred) 

    time_taken = time.time()-t0

    print("Accuracy = {}".format(accuracy))

    print("ROC Area under Curve = {}".format(roc_auc))

    print("Time taken = {}".format(time_taken))

    print(classification_report(y_test,y_pred,digits=5))

    plot_confusion_matrix(model, X_test, y_test,cmap=plt.cm.pink, normalize = 'all')

    plot_roc_curve(model, X_test, y_test)                     

    

    return model, accuracy, roc_auc, time_taken
from sklearn.linear_model import LogisticRegression



params_lr = {'penalty': 'elasticnet', 'l1_ratio':0.5, 'solver': 'saga'}



model_lr = LogisticRegression(**params_lr)

model_lr, accuracy_lr, roc_auc_lr, tt_lr = run_model(model_lr, X_train, y_train, X_test, y_test)
import statsmodels.api as sm

logit_model=sm.Logit(y_train,X_train)

result=logit_model.fit()

print(result.summary())
from sklearn.naive_bayes import GaussianNB



params_nb = {}



model_nb = GaussianNB(**params_nb)

model_nb, accuracy_nb, roc_auc_nb, tt_nb = run_model(model_nb, X_train, y_train, X_test, y_test)
from sklearn.neighbors import KNeighborsClassifier



params_kn = {'n_neighbors':10, 'algorithm': 'kd_tree', 'n_jobs':4}



model_kn = KNeighborsClassifier(**params_kn)

model_kn, accuracy_kn, roc_auc_kn, tt_kn = run_model(model_kn, X_train, y_train, X_test, y_test)
from sklearn.tree import DecisionTreeClassifier

params_dt = {'max_depth': 12,    

             'max_features': "sqrt"}



model_dt = DecisionTreeClassifier(**params_dt)

model_dt, accuracy_dt, roc_auc_dt, tt_dt = run_model(model_dt, X_train, y_train, X_test, y_test)
import graphviz

from sklearn.tree import DecisionTreeClassifier, export_graphviz



features_n = ['Type_of_Travel', 'Inflight_wifi_service', 'Online_boarding', 'Seat_comfort']

X_train_n = scaler.fit_transform(train[features_n])

data = export_graphviz(DecisionTreeClassifier(max_depth=3).fit(X_train_n, y_train), out_file=None, 

                       feature_names = features_n,

                       class_names = ['Dissatisfied (0)', 'Satisfied (1)'], 

                       filled = True, rounded = True, special_characters = True)

# we have intentionally kept max_depth short here to accommodate the entire visual-tree, best result comes with max_depth = 12

# we have taken only really important features here to accommodate the entire tree picture

graph = graphviz.Source(data)

graph
from sklearn.neural_network import MLPClassifier



params_nn = {'hidden_layer_sizes': (30,30,30),

             'activation': 'logistic',

             'solver': 'lbfgs',

             'max_iter': 100}



model_nn = MLPClassifier(**params_nn)

model_nn, accuracy_nn, roc_auc_nn, tt_nn = run_model(model_nn, X_train, y_train, X_test, y_test)
from sklearn.ensemble import RandomForestClassifier



params_rf = {'max_depth': 16,

             'min_samples_leaf': 1,

             'min_samples_split': 2,

             'n_estimators': 100,

             'random_state': 12345}



model_rf = RandomForestClassifier(**params_rf)

model_rf, accuracy_rf, roc_auc_rf, tt_rf = run_model(model_rf, X_train, y_train, X_test, y_test)
import numpy as np

%matplotlib inline



trees=range(100)

accuracy=np.zeros(100)



for i in range(len(trees)):

    clf = RandomForestClassifier(n_estimators = i+1)

    model1 = clf.fit(X_train, y_train.ravel())

    y_predictions = model1.predict(X_test)

    accuracy[i] = accuracy_score(y_test, y_predictions)



plt.plot(trees,accuracy)
import xgboost as xgb

params_xgb ={'n_estimators': 500,

            'max_depth': 16}



model_xgb = xgb.XGBClassifier(**params_xgb)

model_xgb, accuracy_xgb, roc_auc_xgb, tt_xgb = run_model(model_xgb, X_train, y_train, X_test, y_test)
from sklearn.ensemble import AdaBoostClassifier as adab

params_adab ={'n_estimators': 500,

              'random_state': 12345}



model_adab = adab(**params_adab)

model_adab, accuracy_adab, roc_auc_adab, tt_adab = run_model(model_adab, X_train, y_train, X_test, y_test)
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import itertools

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from sklearn.ensemble import AdaBoostClassifier

from mlxtend.classifier import EnsembleVoteClassifier

from mlxtend.plotting import plot_decision_regions



value = 1.70

width = 0.85



clf1 = LogisticRegression(random_state=12345)

clf2 = GaussianNB()

clf3 = KNeighborsClassifier()

clf4 = DecisionTreeClassifier(random_state=12345) 

clf5 = MLPClassifier(random_state=12345, verbose = 0)

clf6 = RandomForestClassifier(random_state=12345)

clf7 = xgb.XGBClassifier(random_state=12345)

clf8 = AdaBoostClassifier(random_state=12345)

eclf = EnsembleVoteClassifier(clfs=[clf6, clf7, clf8], weights=[1, 1, 1], voting='soft')



X_list = train[["Type_of_Travel", "Inflight_wifi_service", "Online_boarding", "Seat_comfort"]] #took only really important features

X = np.asarray(X_list, dtype=np.float32)

y_list = train["satisfaction"]

y = np.asarray(y_list, dtype=np.int32)



# Plotting Decision Regions

gs = gridspec.GridSpec(3,3)

fig = plt.figure(figsize=(18, 14))



labels = ['Logistic Regression',

          'Naive Bayes',

          'KNN',

          'Decision Tree',

          'Neural Network',

          'Random Forest',

          'XGBoost',

          'AdaBoost',

          'Ensemble']



for clf, lab, grd in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, eclf],

                         labels,

                         itertools.product([0, 1, 2],

                         repeat=2)):

    clf.fit(X, y)

    ax = plt.subplot(gs[grd[0], grd[1]])

    fig = plot_decision_regions(X=X, y=y, clf=clf, 

                                filler_feature_values={2: value, 3: value}, 

                                filler_feature_ranges={2: width, 3: width}, 

                                legend=2)

    plt.title(lab)



plt.show()
roc_auc_scores = [roc_auc_lr, roc_auc_nb, roc_auc_kn, roc_auc_dt, roc_auc_nn, roc_auc_rf, roc_auc_xgb, roc_auc_adab]

tt = [tt_lr, tt_nb, tt_kn, tt_dt, tt_nn, tt_rf, tt_xgb, tt_adab]



model_data = {'Model': ['Logistic Regression','Naive Bayes','K-NN','Decision Tree','Neural Network','Random Forest','XGBoost','AdaBoost'],

              'ROC_AUC': roc_auc_scores,

              'Time taken': tt}

data = pd.DataFrame(model_data)



fig, ax1 = plt.subplots(figsize=(14,8))

ax1.set_title('Model Comparison: Area under ROC Curve and Time taken for execution by Various Models', fontsize=13)

color = 'tab:blue'

ax1.set_xlabel('Model', fontsize=13)

ax1.set_ylabel('Time taken', fontsize=13, color=color)

ax2 = sns.barplot(x='Model', y='Time taken', data = data, palette='Blues_r')

ax1.tick_params(axis='y')

ax2 = ax1.twinx()

color = 'tab:orange'

ax2.set_ylabel('ROC_AUC', fontsize=13, color=color)

ax2 = sns.lineplot(x='Model', y='ROC_AUC', data = data, sort=False, color=color)

ax2.tick_params(axis='y', color=color)
import pandas as pd

import numpy as np



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, validation_curve

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn import metrics

import itertools

from mlxtend.evaluate import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix





import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium.plugins import MarkerCluster

import warnings

warnings.filterwarnings('ignore')



pd.set_option('display.max_rows', 194673)

pd.set_option('display.max_columns', 37)

df = pd.read_csv('../input/seattle-sdot-collisions-data/Collisions.csv')
df.columns
df.rename(columns={'SEVERITYCODE': 'severity_code', 'X':'longitude', 'Y': 'latitude',

                   'ADDRTYPE':'addr_type', 'LOCATION': 'location','SEVERITYDESC':'severity_desc', 'COLLISIONTYPE':'collision_type',

                   'PERSONCOUNT':'person_count', 'PEDCOUNT': 'ped_count', 'PEDCYLCOUNT': 'ped_cycle_count', 'VEHCOUNT': 'veh_count',

                   'INCDTTM': 'inc_dt', 'JUNCTIONTYPE': 'junc_type', 'SDOT_COLCODE': 'case_code', 'SDOT_COLDESC': 'case_desc',

                   'UNDERINFL':'under_infl', 'WEATHER': 'weather', 'ROADCOND': 'roadcond', 'LIGHTCOND': 'light_cond',

                   'ST_COLCODE': 'st_code', 'ST_COLDESC': 'st_desc', 'HITPARKEDCAR':'hit_parked_car', 'SPEEDING':'speeding', 

                   'FATALITIES':'fatalities', 'INJURIES':'injuries', 'SERIOUSINJURIES':'serious_injuries'}, inplace=True)
df.shape
df.head()
df.info()
map = folium.Map(location=[47.606209, -122.332069], zoom_start=10)

map_clust = MarkerCluster().add_to(map)

location = df[['latitude', 'longitude']][df['longitude'].notnull()][:5000]

loc = location.values.tolist()

for i in range(len(loc)):

  folium.Marker(loc[i]).add_to(map_clust)

map
df['severity_code'].value_counts().to_frame('counts')
df['severity_desc'].value_counts().to_frame('counts')
df['collision_type'].value_counts().to_frame('counts')
df['addr_type'].value_counts().to_frame('counts')
df['junc_type'].value_counts().to_frame('counts')
df['weather'].value_counts().to_frame('counts')
df['roadcond'].value_counts().to_frame()
df['light_cond'].value_counts().to_frame('counts')
df[['person_count', 'ped_count', 'ped_cycle_count', 'veh_count']].describe()
df = df[['longitude', 'latitude','location','severity_code',

        'severity_desc','collision_type', 'person_count', 'ped_count', 'ped_cycle_count',

       'veh_count','inc_dt','addr_type', 'junc_type', 'case_code', 'case_desc','under_infl',

       'speeding', 'weather', 'roadcond', 'light_cond','st_code', 'st_desc',

       'hit_parked_car', 'injuries', 'serious_injuries', 'fatalities']]
df.isnull().sum()
df1 = df[['latitude', 'longitude', 'severity_code', 'weather', 'roadcond', 'light_cond', 

          'speeding', 'under_infl', 'person_count', 'ped_count', 'ped_cycle_count', 'veh_count', 

          'injuries', 'serious_injuries', 'severity_desc', 'fatalities']]
df1['speeding'].replace(np.nan,0,inplace=True)

df1['speeding'].replace('Y', 1, inplace=True)

df1['speeding'].value_counts().to_frame('counts')
df1.replace(to_replace={'Unknown': np.nan, 

                        'Other':np.nan}, inplace=True)
df1.dropna(inplace=True)
df1.isnull().sum()
df1['under_infl'].replace(to_replace={'Y':1, 'N':0, '1':1, '0':0}, inplace=True)
df1['under_infl'].value_counts().to_frame('counts')
df1['severity_code'].replace(to_replace={'2b':'4'}, inplace=True)
plt.style.use('ggplot')

ax = sns.countplot(df1['weather'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
plt.style.use('seaborn')

ax = sns.countplot(df1['severity_desc'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.xlabel('severity')

plt.show()
ax = sns.countplot(df1['roadcond'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
ax = sns.countplot(df1['light_cond'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
plt.figure(figsize=(6, 4))

sns.countplot(df1['under_infl'])
ax = plt.scatter(df1['veh_count'], df1['person_count'])

plt.xlabel('vehicle_count')

plt.ylabel('person_couont')

plt.show()
plt.scatter(df1['ped_count'], df1['person_count'])

plt.xlabel('pedestrian_count')

plt.ylabel('person_count')

plt.show()
plt.scatter(df1['veh_count'], df1['injuries'])

plt.xlabel('vehicle count')

plt.ylabel('injuries count')

plt.show()
sns.heatmap(df1.corr(), cmap='YlGnBu_r')

plt.show()
df2 = pd.concat([df1.drop(['weather', 'roadcond', 'light_cond','severity_desc'], axis=1),

                 pd.get_dummies(df1['weather']),

                 pd.get_dummies(df1['roadcond']),

                 pd.get_dummies(df1['light_cond'])], axis=1)

df2.reset_index(drop=True, inplace=True)
df2.head().T
sns.heatmap(df2.corr(), cmap='YlGnBu_r')

plt.show()
x = df2.drop('severity_code', axis=1).values

y = df2['severity_code'].values

X = StandardScaler().fit(x).transform(x)

y = y.astype(int)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def cnf_mx(preds):

    cm = confusion_matrix(y_target=y_test, 

                          y_predicted=preds, 

                          binary=False)

    fig, ax = plot_confusion_matrix(conf_mat=cm)

    plt.show()


def validate_models(model):

    kfold = KFold(n_splits=10, random_state=42)

    results_1 = cross_val_score(model, X, y, cv=kfold)

    print("kfold cross_val_score: %.2f%%" % (results_1.mean()*100.0))

    

    skfold = StratifiedKFold(n_splits=3, random_state=100)

    results_2  = cross_val_score(model, X, y, cv=skfold)

    print("stratified kfold cross_val _score: %.2f%%" % (results_2.mean()*100.0))
tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=5)

tree_model.fit(x_train, y_train)

print(tree_model)

yhat1 = tree_model.predict(x_test)

print('The accuracy of the decision tree classifier is {} with a max_depth of 5'.format(accuracy_score(y_test, yhat1)))
print(classification_report(y_test, yhat1))
cnf_mx(preds=yhat1)
validate_models(DecisionTreeClassifier(criterion='entropy', max_depth=5))
forest_model = RandomForestClassifier(n_estimators=75)

forest_model.fit(x_train, y_train)

print(forest_model)

yhat2 = forest_model.predict(x_test)

print('the accuracy score for Random Forest Classifier is {}'.format(accuracy_score(y_test, yhat2)))
print(classification_report(y_test, yhat2))
cnf_mx(preds=yhat2)
validate_models(RandomForestClassifier(n_estimators=75))
log_reg_model = LogisticRegression(C=0.06)

log_reg_model.fit(x_train, y_train)

print(log_reg_model)

yhat3 = log_reg_model.predict(x_test)

print('The accuracy score for logistic regression is {}'.format(accuracy_score(y_test, yhat3)))
print(classification_report(y_test, yhat3))
cnf_mx(yhat3)
validate_models(LogisticRegression())
plt.bar(['DecisionTreeClassifier', 'RandomForestClassifier', 'LogisticRegression'], [1.,1.,1.])

plt.ylabel('accuracy')

plt.xlabel('machine learning models')

plt.show()
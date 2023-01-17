

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import rcParams

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings('ignore')



color = sns.color_palette()

rcParams['figure.figsize'] = 10, 6

lbl = LabelEncoder()





%matplotlib inline

pd.set_option('display.max_columns', 50)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#s1 = pd.Series([4,5,6,20,42])

#s2 = pd.Series([1,2,3,5,42])



#s1[s1.isin(s2)]
f=open(os.path.join('/kaggle/input', 'bank-marketing/bank-additional-names.txt'), "r")

if f.mode == 'r':

    contents =f.read()

    print(contents)
df = pd.read_csv('../input/bank-marketing/bank-additional-full.csv', sep = ';')
df.shape
df[df['y'] == 'yes'].shape[0]
df[df['y'] == 'no'].shape[0]
df.head(4)
df.describe()
df.columns
print(pd.crosstab(df['marital'],df['y'], normalize='index'))
plt.figure(figsize=(10, 6))

sns.countplot(x='marital', hue='y', data=df)

plt.xticks(rotation=45)

plt.show()
age_target_yes = df[(df['age'] > 0) & (df['y'] == 'yes')]

age_target_no = df[(df['age'] > 0) & (df['y'] == 'no')]



plt.figure(figsize=(10, 6))

sns.distplot(age_target_yes['age'], bins=25, color='g')

sns.distplot(age_target_no['age'], bins=25, color='r')

plt.show()
group = pd.DataFrame()

group['job_count'] = df.groupby(['job'])['job'].count()

group['job_index'] = group.index



group_top = group.sort_values(by='job_count', ascending=False).head(15)



plt.figure(figsize=(10, 6))

sns.barplot(x='job_index', y ='job_count', data =group_top)

plt.xticks(rotation=45)

plt.title('Count job types')

plt.xlabel('Job types')

plt.ylabel('Number of job each type')



plt.show()
df.groupby(['poutcome'])['poutcome'].count()
print(pd.crosstab(df['poutcome'], df['y']))

sns.countplot(x='poutcome', hue='y', data=df)

plt.xticks(rotation=45)

plt.show()
print(pd.crosstab(df['contact'], df['y']))

sns.countplot(x='contact', hue='y', data=df)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(10, 6))

sns.boxenplot(x='education', y='cons.price.idx', hue='y', data=df )

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(10, 6))

g = sns.distplot(df[df['y'] == 'yes']['nr.employed'], label='Yes')

g= sns.distplot(df[df['y'] == 'no']['nr.employed'], label='No')

g.legend()

plt.show()
plt.figure(figsize=(10, 6))

g = sns.distplot(df[df['y'] == 'yes']['euribor3m'], label='Yes')

g= sns.distplot(df[df['y'] == 'no']['euribor3m'], label='No')

g.legend()

plt.show()
plt.figure(figsize=(10, 6))

g = sns.distplot(df[df['y'] == 'yes']['cons.conf.idx'], label='Yes')

g= sns.distplot(df[df['y'] == 'no']['cons.conf.idx'], label='No')

g.legend()

plt.show()
plt.figure(figsize=(10, 6))

g = sns.distplot(df[df['y'] == 'yes']['emp.var.rate'], label='Yes')

g= sns.distplot(df[df['y'] == 'no']['emp.var.rate'], label='No')

g.legend()

plt.show()
plt.figure(figsize=(10, 6))

g = sns.distplot(df[df['y'] == 'yes']['cons.conf.idx'], label='Yes')

g= sns.distplot(df[df['y'] == 'no']['cons.conf.idx'], label='No')

g.legend()

plt.show()
plt.figure(figsize=(10, 6))

g = sns.distplot(df[df['y'] == 'yes']['cons.price.idx'], label='Yes')

g= sns.distplot(df[df['y'] == 'no']['cons.price.idx'], label='No')

g.legend()

plt.show()
plt.figure(figsize=(16, 8))

tmp = pd.crosstab(df['marital'], df['y'], normalize='index') * 100

tmp = tmp.reset_index()

plt.subplot(221)

g = sns.countplot(x='marital', data=df, order=list(tmp.marital.values))

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,

            height + 3, '{:1.2f}%'.format(height/df.shape[0]*100),

            ha="center",fontsize=14) 

    

plt.xticks(rotation=45)

plt.subplot(222)

g1 = sns.countplot(x='marital', hue='y', data=df,order=list(tmp.marital.values))

plt.subplots_adjust(hspace = 0.6, top = 1.4)

plt.xticks(rotation=45)

plt.show()
df.head(2)
plt.figure(figsize=(16, 8))

tmp = pd.crosstab(df['education'], df['y'], normalize='index') * 100

tmp = tmp.reset_index()

plt.subplot(221)

g = sns.countplot(x='education', data=df, order=list(tmp.education.values))

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,

            height + 3, '{:1.2f}%'.format(height/df.shape[0]*100),

            ha="center",fontsize=14) 

    

plt.xticks(rotation=45)

plt.subplot(222)

g1 = sns.countplot(x='education', hue='y', data=df,order=list(tmp.education.values))

plt.subplots_adjust(hspace = 0.6, top = 1.4)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(16, 8))

tmp = pd.crosstab(df['month'], df['y'], normalize='index') * 100

tmp = tmp.reset_index()

plt.subplot(221)

g = sns.countplot(x='month', data=df, order=list(tmp.month.values))

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,

            height + 3, '{:1.2f}%'.format(height/df.shape[0]*100),

            ha="center",fontsize=14) 

    

plt.xticks(rotation=45)

plt.subplot(222)

g1 = sns.countplot(x='month', hue='y', data=df,order=list(tmp.month.values))

plt.subplots_adjust(hspace = 0.6, top = 1.4)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(16, 8))

tmp = pd.crosstab(df['job'], df['y'], normalize='index') * 100

tmp = tmp.reset_index()

plt.subplot(221)

g = sns.countplot(x='job', data=df, order=list(tmp.job.values))

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,

            height + 3, '{:1.2f}%'.format(height/df.shape[0]*100),

            ha="center",fontsize=14) 

    

plt.xticks(rotation=45)

plt.subplot(222)

g1 = sns.countplot(x='job', hue='y', data=df,order=list(tmp.job.values))

plt.subplots_adjust(hspace = 0.6, top = 1.4)

plt.xticks(rotation=45)

plt.show()
df.dtypes
#plt.plot(df['cons.price.idx'])

#plt.plot(df['euribor3m'].unique())
df['y'] = df['y'].map({'yes':1, 'no':0})
df['marital'].unique()
df['marital'] = df['marital'].map({'married':1, 'single':2, 'divorced':3, 'unknown':4})
df['education'].unique()
df['education'].unique()
df['education'] = df['education'].map({'basic.4y':1, 'high.school':2, 'basic.6y':3, 'basic.9y':4,

       'professional.course':5, 'unknown':6, 'university.degree':6, 'illiterate':7})
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier



from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve, auc

import xgboost as xgb

import lightgbm as lgb
#g = sns.FacetGrid(df, hue="y", col="housing", margin_titles=True)

#g=g.map(plt.scatter, "age", "marital",edgecolor="w").add_legend();
'''age_target_yes = df[(df['age'] > 0) & (df['y'] == 1)]

age_target_no = df[(df['age'] > 0) & (df['y'] == 0)]



plt.scatterplot(age_target_yes['age'], age_target_yes.index, color='red')

plt.scatter(age_target_no['age'], age_target_no.index, color='green')

plt.show()'''
'''sns.scatterplot(x='age', y= df.index, hue='y',  data=df)

plt.show()'''
df['poutcome']=  df['poutcome'].map({'nonexistent':1, 'failure':2, 'success':3})

df['contact'] = df['contact'].map({'telephone':1, 'cellular':2})

df['housing'] = df['housing'].map({'no':1, 'yes':2, 'unknown':3})

df['loan'] = df['loan'].map({'no':1, 'yes':2, 'unknown':3})

df['default'] = df['default'].map({'no':1, 'yes':2, 'unknown':3})

df['job'] = df['job'].map({'housemaid':1, 'services':2, 'admin.':3, 'blue-collar':4, 'technician':5,

       'retired':6, 'management':7, 'unemployed':8, 'self-employed':9, 'unknown':10,

       'entrepreneur':11, 'student':12})
del df['day_of_week']

del df['month']
df.head(5)
feature_col = ['age', 'job','marital', 'education', 'default','housing','loan', 'contact','duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx','euribor3m','nr.employed']

label_col = ['y']
X_train, X_test, Y_train, Y_test = train_test_split(df[feature_col],df[label_col], test_size=0.2, random_state=52)
randomforest = RandomForestClassifier()

randomforest.fit(X_train,Y_train)

randomforest_score = round(randomforest.score(X_train,Y_train)*100, 2)

#model_name.append('RandomForestClassifier')

#model_score.append(randomforest_score)

randomforest_score
predict_y = randomforest.predict(X_test)
'''predict_y.shape

predict_y

Y_test.shape

Y_test['y'].shape[0]

np.array(Y_test['y'])'''
def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
#prob = predict_y[predict_y == 1]
auc_result = roc_auc_score(Y_test['y'], predict_y)

print('AUC: %.2f' % auc_result)
fpr, tpr, thresholds = roc_curve(Y_test['y'], predict_y)
plot_roc_curve(fpr, tpr)
#roc_auc = auc(fpr, tpr)

#roc_auc
#X_train, X_test, Y_train, Y_test 



n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]



train_results =[]

test_results =[]



for estimator in n_estimators:

    rf = RandomForestClassifier(n_estimators=estimator, n_jobs= -1)

    rf.fit(X_train,Y_train)

    

    train_pred = rf.predict(X_train)

    

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, train_pred)



    roc_auc = auc(false_positive_rate,true_positive_rate)

    train_results.append(roc_auc)

    

    y_pred = rf.predict(X_test)

   

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    test_results.append(roc_auc)

    

    

from matplotlib.legend_handler import HandlerLine2D



line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC" )

line2, = plt.plot(n_estimators, test_results, 'r', label ="Test AUC")



plt.legend(handler_map = {line1: HandlerLine2D(numpoints = 2)})



plt.ylabel('AUC score')

plt.xlabel('n_estimators')

plt.show()
#np.arange(1, 33, 1)

#max_depths = np.linspace(1, 32, 32, endpoint=True)

#max_depths

#np.linspace(1, 32, 32, endpoint=True)


max_depths = np.arange(1, 33, 1)

train_results = []

test_results = []

for max_depth in max_depths:

    rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)

    rf.fit(X_train, Y_train)

    train_pred = rf.predict(X_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)



    train_results.append(roc_auc)

    y_pred = rf.predict(X_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")

line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('Tree depth')

plt.show()
#X_train, X_test, Y_train, Y_test

min_samples_splits = np.linspace(0.1, 1.0, 10,  endpoint=True)



train_results = []

test_results = []



for min_samples_split in min_samples_splits:

    rf = RandomForestClassifier(min_samples_split= min_samples_split )

    rf.fit(X_train, Y_train)

    

    train_pred = rf.predict(X_train)

    

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)

    

    y_pred = rf.predict(X_test)

    

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    test_results.append(roc_auc)

    

from matplotlib.legend_handler import HandlerLine2D



line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')

line2, = plt.plot(min_samples_splits, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1:HandlerLine2D(numpoints=2)})



plt.ylabel('AUC Score')

plt.xlabel('Min Samples split')

plt.show()
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)



train_results = []

test_results =[]



for min_samples_leaf in min_samples_leafs:

    rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf)

    rf.fit(X_train, Y_train)

    

    train_pred = rf.predict(X_train)

    

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, train_pred)

    

    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)

    

    y_pred = rf.predict(X_test)

    

    false_positive_rate, true_positive_rate,thresholds = roc_curve(Y_test, y_pred)

    

    roc_auc = auc(false_positive_rate, true_positive_rate)

    test_results.append(roc_auc)



from matplotlib.legend_handler import HandlerLine2D    

line1, = plt.plot(min_samples_leafs,train_results, 'b', label="Train AUC")

line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')

plt.legend(handler_map={line1:HandlerLine2D(numpoints=2)})



plt.ylabel('AUC Score')

plt.xlabel('Min Samples leaf')

plt.show()
max_features = list(range(1, df.shape[1]))

#print(max_features)



train_results =[]

test_results =[]



for max_feature in max_features:

    rf = RandomForestClassifier(max_features = max_feature)

    rf.fit(X_train, Y_train)

    

    train_pred = rf.predict(X_train)

    

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)

    

    y_pred = rf.predict(X_test)

    

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    

    test_results.append(roc_auc)



from matplotlib.legend_handler import HandlerLine2D       

line1, =plt.plot(max_features, train_results, 'b', label="Train AUC")

line2, =plt.plot(max_features, test_results, 'r', label="Test AUC")



plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})



plt.ylabel('AUC Score')

plt.xlabel('Max features')

plt.show()
kn = KNeighborsClassifier()

kn.fit(X_train, Y_train)



y_pred = kn.predict(X_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

print("ROC AUC",round(roc_auc,2))

print("Score", round(kn.score(X_train,Y_train)*100, 2))
plot_roc_curve(false_positive_rate, true_positive_rate)
neighbors = list(range(1, 30))



train_results = []

test_results =[]



for n in neighbors:

    kn = KNeighborsClassifier(n_neighbors=n)

    kn.fit(X_train, Y_train)

    

    train_pred = kn.predict(X_train)

    

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)

    

    y_pred = kn.predict(X_test)

    

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    test_results.append(roc_auc)

    

line1, = plt.plot(neighbors, train_results, 'b', label='Train AUC')

line2, = plt.plot(neighbors, test_results, 'r', label='Test AUC')



plt.legend(handler_map ={line1:HandlerLine2D(numpoints=2)})



plt.ylabel('AUC Score')

plt.xlabel('n_neigbors')

plt.show()

distances = [1, 2, 3, 4, 5]



train_results = []

test_results = []



for p in distances:

    kn = KNeighborsClassifier(p=p)

    kn.fit(X_train, Y_train)

    

    train_pred = kn.predict(X_train)

    

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)

    

    y_pred = kn.predict(X_test)

    

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

    

    roc_auc = auc(false_positive_rate, true_positive_rate)

    test_results.append(roc_auc)

    

line1, = plt.plot(distances, train_results, 'b', label='Train AUC')

line2, = plt.plot(distances, test_results, 'r', label='Test AUC')



plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})



plt.ylabel('AUC Score')

plt.xlabel('P')

plt.show()
gbclassifier = GradientBoostingClassifier()

gbclassifier.fit(X_train, Y_train)

y_pred = gbclassifier.predict(X_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

print("AUC ",round(roc_auc, 2))

print("Score", round(gbclassifier.score(X_train,Y_train)*100, 2))
learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]



train_results = []

test_results = []



for eta in learning_rates:

    gbclassifier = GradientBoostingClassifier(learning_rate=eta)

    gbclassifier.fit(X_train, Y_train)

    

    train_pred = gbclassifier.predict(X_train)

    

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)

    

    y_pred = gbclassifier.predict(X_test)

    

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    test_results.append(roc_auc)

    

line1, = plt.plot(learning_rates, train_results, 'b', label="Train AUC")

line2, = plt.plot(learning_rates, test_results, 'r', label="Test AUC")



plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC Score')

plt.xlabel('Learn')

    
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]



train_results = []

test_results = []



for estimator in n_estimators:

    gbclassifier = GradientBoostingClassifier(n_estimators=estimator)

    gbclassifier.fit(X_train, Y_train)

    

    train_pred = gbclassifier.predict(X_train)

    

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, train_pred)

    roc_auc =auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)

    

    y_pred = gbclassifier.predict(X_test)

    

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    test_results.append(roc_auc)

    

line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")

line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('n_estimators')

plt.show()
'''max_depths = np.linspace(1, 32, 32, endpoint=True)



train_results =[]

test_results = []



for max_depth in max_depths:

    gbclassifier = GradientBoostingClassifier(max_depth=max_depth)

    gbclassifier.fit(X_train, Y_train)

    

    train_pred = gbclassifier.predict(X_train)

    

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    train_results.append(roc_auc)

    

    y_pred = gbclassifier.predict(X_test)

    

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

    

    roc_auc = auc(false_positive_rate, true_positive_rate)

    test_results.append(roc_auc)

    

line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")

line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('n_estimators')

plt.show()'''
import os
print(os.listdir("../input"))
# Data Processing
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Data Visualizing
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Data Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC, NuSVC
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Data Evalutation
from sklearn import metrics
from sklearn.metrics import roc_curve

# Warning Removal
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
df = pd.read_csv('../input/diabetes.csv')
df.head(5)
df.describe(include="all")
df.info(verbose=True)
df.corr()['Outcome'].sort_values(ascending = False)
zero_count = (df == 0).sum() # (df.isnull()).sum()
zero_count_df = pd.DataFrame(zero_count)
zero_count_df.drop('Outcome', axis=0, inplace=True)
zero_count_df.columns = ['count_0']

# https://stackoverflow.com/questions/31859285/rotate-tick-labels-for-seaborn-barplot/60530167#60530167
sns.set(style='whitegrid')
plt.figure(figsize=(13,8))
sns.barplot(x=zero_count_df.index, y=zero_count_df['count_0'])
plt.xticks(rotation=70)
plt.show()
plt.figure(figsize=(13,8))
sns.boxplot(data=df.drop('Outcome', axis=1))
plt.show()
cols = ['Glucose','BloodPressure','Insulin','BMI','SkinThickness']
df[cols] = df[cols].replace(0, np.nan)
# Check:  df[df['Glucose'].isnull()]
df.isnull().sum()
cols = ['Pregnancies','Glucose','BloodPressure','Insulin','BMI','SkinThickness','DiabetesPedigreeFunction', 'Age']
arr_median = []
def median_target(cols):
    for col in cols:
        temp = df[df[col].notnull()]
        temp = temp[[col, 'Outcome']].groupby('Outcome').median()
        arr_median.append(temp)
    return arr_median
median_target(cols)
a = pd.concat(arr_median, axis=1)
arr_mean=[]
def mean_target(cols):
    for col in cols:
        temp = df[df[col].notnull()]
        temp = temp[[col, 'Outcome']].groupby('Outcome').mean()
        arr_mean.append(temp)
    return arr_mean
mean_target(cols)
b = pd.concat(arr_mean, axis=1)
c = pd.concat([a,b], axis=0)
c
c.loc[0].iloc[0]['Pregnancies']
g = sns.FacetGrid(df, col="Outcome")
g = g.map(plt.hist, "Pregnancies")
g = sns.FacetGrid(df, col="Outcome")
g = g.map(plt.hist, "Glucose")
g = sns.FacetGrid(df, col="Outcome")
g = g.map(plt.hist, "BloodPressure")
g = sns.FacetGrid(df, col="Outcome")
g = g.map(plt.hist, "Insulin")
g = sns.FacetGrid(df, col="Outcome")
g = g.map(plt.hist, "BMI")
g = sns.FacetGrid(df, col="Outcome")
g = g.map(plt.hist, "SkinThickness")
df.loc[(df['Outcome'] == 0 ) & (df['Glucose'].isnull()), 'Glucose'] = 107
df.loc[(df['Outcome'] == 1 ) & (df['Glucose'].isnull()), 'Glucose'] = 140

df.loc[(df['Outcome'] == 0 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 70
df.loc[(df['Outcome'] == 1 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 74.5

df.loc[(df['Outcome'] == 0 ) & (df['Insulin'].isnull()), 'Insulin'] = 102.5
df.loc[(df['Outcome'] == 1 ) & (df['Insulin'].isnull()), 'Insulin'] = 169.5

df.loc[(df['Outcome'] == 0 ) & (df['BMI'].isnull()), 'BMI'] = 30.1
df.loc[(df['Outcome'] == 1 ) & (df['BMI'].isnull()), 'BMI'] = 34.3

df.loc[(df['Outcome'] == 0 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 27
df.loc[(df['Outcome'] == 1 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 32
df.isnull().sum()
f, ax = plt.subplots(figsize=(11, 15))
ax.set(xlim=(-.05, 300))
sns.boxplot(data=df.drop('Outcome', axis=1), orient = 'h')
plt.show()
sns.boxplot(df['Pregnancies'])
plt.show()
df['Pregnancies'].value_counts()
df.loc[(df['Outcome'] == 0 ) & (df['Pregnancies']>12), 'Pregnancies'] = 2
df.loc[(df['Outcome'] == 1 ) & (df['Pregnancies']>12), 'Pregnancies'] = 4
sns.boxplot(df['BloodPressure'])
plt.show()
df.loc[(df['Outcome'] == 0 ) & (df['BloodPressure']<40), 'BloodPressure'] = 70
df.loc[(df['Outcome'] == 1 ) & (df['BloodPressure']<40), 'BloodPressure'] = 74.5

df.loc[(df['Outcome'] == 0 ) & (df['BloodPressure']>110), 'BloodPressure'] = 70
df.loc[(df['Outcome'] == 1 ) & (df['BloodPressure']>110), 'BloodPressure'] = 74.5
sns.boxplot(df['Insulin'])
plt.show()
df.loc[(df['Outcome'] == 0 ) & (df['Insulin']>270), 'Insulin'] = 102.5
df.loc[(df['Outcome'] == 1 ) & (df['Insulin']>270), 'Insulin'] = 169.5
sns.boxplot(df['BMI'])
plt.show()
df.loc[(df['Outcome'] == 0 ) & (df['BMI']>50), 'BMI'] = 30.1
df.loc[(df['Outcome'] == 1 ) & (df['BMI']>50), 'BMI'] = 34.3
sns.boxplot(df['SkinThickness'])
plt.show()
df.loc[(df['Outcome'] == 0 ) & (df['SkinThickness']>42), 'SkinThickness'] = 27
df.loc[(df['Outcome'] == 1 ) & (df['SkinThickness']>42), 'SkinThickness'] = 32

df.loc[(df['Outcome'] == 0 ) & (df['SkinThickness']<15), 'SkinThickness'] = 27
df.loc[(df['Outcome'] == 1 ) & (df['SkinThickness']<15), 'SkinThickness'] = 32
sns.boxplot(df['DiabetesPedigreeFunction'])
plt.show()
df.loc[(df['Outcome'] == 0 ) & (df['DiabetesPedigreeFunction']>1.1), 'DiabetesPedigreeFunction'] = 0.336
df.loc[(df['Outcome'] == 1 ) & (df['DiabetesPedigreeFunction']>1.1), 'DiabetesPedigreeFunction'] = 0.449
sns.boxplot(df['Age'])
plt.show()
df.loc[(df['Outcome'] == 0 ) & (df['Age']>62), 'Age'] = 27
df.loc[(df['Outcome'] == 1 ) & (df['Age']>62), 'Age'] = 36
df.describe()
x = StandardScaler().fit_transform(df.drop('Outcome', axis=1))
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.3, random_state=0)
pwd
models_score = []

# LOGISTICREGRESSION
logistic = LogisticRegression(random_state=0).fit(x_train, y_train)
models_score.append([logistic.__class__.__name__, logistic.score(x_test, y_test)])

# KNN
max = 0
num_neighbor = 0

train_scores = []
test_scores = []

for i in range (1, 100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    
    train_scores.append(knn.score(x_train, y_train))
    test_scores.append(knn.score(x_test, y_test))   
    # source: https://stackoverflow.com/questions/455612/limiting-floats-to-two-decimal-points/28142318#28142318
models_score.append([knn.__class__.__name__, np.max(test_scores)])
    
# NAIVE BAYES
NB = GaussianNB().fit(x_train, y_train)
models_score.append([NB.__class__.__name__, NB.score(x_test, y_test)])

# LINEAR - NONLINEAR SVM
linearSVC = LinearSVC(random_state=0, penalty='l2', loss='hinge', dual=True, C=1, multi_class='ovr', max_iter=1000).fit(x_train, y_train)
RbfSVC = SVC(C=1, kernel='rbf', gamma='scale', coef0=1, decision_function_shape='ovr').fit(x_train, y_train)
models_score.append([linearSVC.__class__.__name__, linearSVC.score(x_test, y_test)])
models_score.append([RbfSVC.__class__.__name__, RbfSVC.score(x_test, y_test)])


# DEEP LEARNING
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
    # choose the best model
mc = ModelCheckpoint('./input/best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    # stop traning for not accuracy improvement 
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
history = model.fit(
    x_train,
    y_train,
    epochs=60,
    batch_size=10,
    #validation_split=0.1,
    validation_data=(x_test, y_test),
    verbose = 0,
    shuffle=True,
    callbacks = [mc,es]
)
# evaluate the keras model
_, accuracy = model.evaluate(x_test, y_test)
models_score.append(['Deep Learning', accuracy])
    # load the best model for prediction
# saved_model = load_model('best_model.h5')
# test_loss, test_acc = saved_model.evaluate(x_test, y_test)
# models_score.append(['Deep Learning', test_acc])


# Unscaling Data
x = df.drop('Outcome', axis=1)
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.3, random_state=0)

# DECISION TREE
dt = DecisionTreeClassifier(random_state=0).fit(x_train,y_train)
models_score.append([dt.__class__.__name__, dt.score(x_test, y_test)])

# RANDOM FOREST
rdf=RandomForestClassifier(random_state=0).fit(x_train,y_train)
models_score.append([rdf.__class__.__name__, rdf.score(x_test, y_test)])

# EXTRA TREE
etc = ExtraTreeClassifier(criterion='gini', max_depth=7, max_features=None, random_state=0).fit(x_train,y_train)
models_score.append([etc.__class__.__name__, etc.score(x_test, y_test)])

# ADABOOST
adb = AdaBoostClassifier(base_estimator=None, n_estimators=90, algorithm='SAMME', random_state=0).fit(x_train,y_train)
models_score.append([adb.__class__.__name__, adb.score(x_test, y_test)])

# GRADIENTBOOSTING
gbc = GradientBoostingClassifier(loss='deviance', n_estimators=100, subsample=1, random_state=0).fit(x_train,y_train)
models_score.append([gbc.__class__.__name__, gbc.score(x_test, y_test)])

# XGBOOST
XGB = XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.05, 
                    subsample=0.96, objective='binary:logistic', 
                    booster='gbtree', tree_method='exact', random_state=0).fit(x_train,y_train)
models_score.append([XGB.__class__.__name__, XGB.score(x_test, y_test)])

# LIGHT GBM
LGBM = LGBMClassifier(boosting='gbdt', n_estimators=100, 
                      colsample_bytree=1.0, max_depth=10, 
                      min_child_samples=3, objective='binary', 
                      reg_alpha=0.0, subsample=1.0).fit(x_train,y_train)
models_score.append([LGBM.__class__.__name__, LGBM.score(x_test, y_test)])
model_evaluation = pd.DataFrame(models_score, columns=['Model','Score'])
model_evaluation.sort_values(by=['Score'],
                            ascending=False,
                            inplace=True)
model_evaluation
plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,100), train_scores,marker='*', label='Train Score')
p = sns.lineplot(range(1,100), test_scores,marker='o', label='Test Score')
y_pred = knn.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Knn(n_neighbors=13) ROC curve')
plt.show()
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(18,5))
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Source: https://stackoverflow.com/questions/27817994/visualizing-decision-tree-in-scikit-learn/54836424#54836424
from sklearn import tree
import graphviz 
from graphviz import Source
dot_data = tree.export_graphviz(dt, out_file=None, feature_names=x.columns)
graph = graphviz.Source(dot_data, format='png')
graph.render("decision tree",view = True)
# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'decision tree.png')
# Source: https://www.kaggle.com/parulpandey/intrepreting-machine-learning-models
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(dt, random_state=1).fit(x_train,y_train)
eli5.show_weights(perm, feature_names = x_test.columns.tolist())
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=dt, dataset=x_test, model_features=x_test.columns, feature='Glucose')

# plot it
pdp.pdp_plot(pdp_goals, 'Glucose')
plt.show()
row_to_show = 1
data_for_prediction = x_test.iloc[row_to_show]

import shap
# Create object that can calculate shap values
explainer = shap.TreeExplainer(dt)
# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)

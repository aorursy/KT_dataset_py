# imports and loading the dataset

from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt

import math

from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

from sklearn import feature_selection as fs
heart_filename = '/kaggle/input/heart-disease-uci/heart.csv'

heart_df = pd.read_csv(heart_filename, sep=',', decimal = '.', header = None, 

                    names=['age','sex','cp', 

                           'trestbps', 'chol', 'fbs', 

                           'restecg', 'thalach', 'exang', 

                           'oldpeak', 'slope', 'ca', 'thal', 'target'])

#first row loads in as column names so remove the first row of values

heart_df = heart_df.ix[1:]

print(heart_df.head())

# we are now using The data from the UCI repository known as preprocessed.cleveland.data
heart_df.isna().sum()
#check data set

heart_df.columns
# check load in

heart_df.head() # no obvious nans or weird values
heart_df.head(2)
# check the shape (should be 303 lines)

X = heart_df['age'].shape

X
Y = heart_df[['age', 'cp']]

Y.head(5)
Y.shape
Z = heart_df[['exang', 'oldpeak']]

Z.head(5)
# seems to be no odd shapes

Z.shape
# check for NAN values



heart_df.isnull().sum()
#check data types

heart_df.dtypes
#change data types, loaded in as objects

#some need to be changed to int, categorical and some to float therefore use a for loop

col_list = ['trestbps', 'chol', 'thalach', 'oldpeak']

for item in col_list:

    heart_df[item] = heart_df[item].astype(str).astype(float)

    

col_list2 = ['age']

for item2 in col_list2:

    heart_df[item2] = heart_df[item2].astype(str).astype(np.int64)

    

col_list3 = ['sex','cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']

for item3 in col_list3:

    heart_df[item3] = heart_df[item3].astype(str).astype('category')

    

heart_df.dtypes
# re-check the data

heart_df.head(5)
heart_df['age'].describe()
heart_df['trestbps'].describe()
heart_df['chol'].describe()
heart_df['thalach'].describe()
heart_df['oldpeak'].describe()
# check values for sanity in each column

#sanity check

age = pd.Series(heart_df['age'])

# if every age is between these values it will print true

age.between(18,120).value_counts()



# age range seems real

# all values returned true so we can continue
#sanity check

sex = heart_df['sex']

sex.value_counts() 



# binary values correct
#sanity check

cp = heart_df['cp']

cp.value_counts() 



# all defined nominal values exists with no extras
#sanity check

trestbps = heart_df['trestbps']

# if every trestbps is between these values it will print true (highest/lowest bps taken from healthline)

trestbps.between(40,220).value_counts()

# all blood pressure seem to indicate real values

#all values returned true so we can continue
#sanity check

chol = heart_df['chol']

# if every chol is between these values it will print true (highest/lowest bps taken from healthline)

chol.between(110,600).value_counts()



#all values returned true so we can continue
#sanity check

fbs = heart_df['fbs']

fbs.value_counts() 



# binary value met
restecg = heart_df['restecg']

restecg.value_counts() 



# no non-nominal values, therefore we can continue
#sanity check

thalach = heart_df['thalach']

# if every thalach (max heart rate) is between these values it will print true (highest/lowest bps taken from healthline)

thalach.between(40,220).value_counts()

# all heart rates seem to indicate real values

#all values returned true so we can continue
exang = heart_df['exang']

exang.value_counts() 



# no non-binary values, therefore we can continue
restecg = heart_df['restecg']

restecg.value_counts() 



# no non-nominal values, therefore we can continue
#sanity check

oldpeak = heart_df['oldpeak']

# if every oldpeak (ratio) is between these values it will print true (highest/lowest bps taken from definition of dataset)

oldpeak.between(0,10).value_counts()

#all values returned true so we can continue
slope = heart_df['slope']

slope.value_counts() 



# no non-nominal values, therefore we can continue
ca = heart_df['ca']

ca.value_counts() 
thal = heart_df['thal']

thal.value_counts() 

# thal is correct per summaries on kaggle
heart_df.info()
heart_df['target'].value_counts().plot(kind='pie', autopct='%.2f', legend = True)

plt.title('Heart Disease in Data Set')

plt.show()
# check age range

heart_df['age'].plot(kind='hist', bins=20, figsize =(10,5))

plt.title('Age of Patients', fontsize=14)

plt.xlabel('Age',fontsize=14)

plt.ylabel('Frequency',fontsize=14)

plt.show()

# most patients seem to be around 60 years old
# lets see if age visually has any kind of relationship with heart disease

age_bins = [20,30,40,50,60,70,80]

heart_df['bin_age']=pd.cut(heart_df['age'], bins=age_bins)

ageVsTarget=sns.countplot(x='bin_age',data=heart_df ,hue='target',linewidth=3)

ageVsTarget.set_title("Age vs Heart Disease")

heart_df.drop(['bin_age'],axis=1,inplace=True)

# It seems as though the patients with ages around 40-50 are more likely than not to have heart disease. The big peak at 50-60 

# is about equal to one another (looks like a 50/50 shot)

heart_df['sex'].value_counts().plot(kind='pie', autopct='%.2f', legend = True)

plt.title('Gender in the Data Set')

plt.show()
# check sex with heart disease

fig,ax=plt.subplots(figsize=(16,6))

plt.subplot(121)

s1=sns.boxplot(x='sex',y='age',hue='target',data=heart_df,palette='bright',linewidth=3)

s1.set_title("Gender vs Heart Disease")



# the plot shows us that heart disease tends to affect women more broadly than men, whereas the age range of non affected women is more limited

# men affected tend to be younger than women
# cp and target



cpVsTarget=sns.countplot(x='cp',data=heart_df ,hue='target',linewidth=3, palette = 'pastel')

cpVsTarget.set_title("CP vs Heart Disease")



# chest pains 1, 2 and 3 may indicate heart disease, whereas 0 may not
# trestbps and target



heart_df['trestbps'].plot(kind='hist', bins=20, figsize =(10,5))

plt.title('trestbps of Patients', fontsize=14)

plt.xlabel('trestbps',fontsize=14)

plt.ylabel('Frequency',fontsize=14)

plt.show()

# most trestbps seem to be around 130 with outliers around 200
trestbps_bins = [80,100,120,140,160,180,200]

heart_df['trestbps_bins']=pd.cut(heart_df['trestbps'], bins=trestbps_bins)

treVsTarget=sns.countplot(x='trestbps_bins',data=heart_df ,hue='target',linewidth=3)

treVsTarget.set_title("trestbps vs Heart Disease")

heart_df.drop(['trestbps_bins'],axis=1,inplace=True)



# it seems as though the lower the resting blood pressure, the higher the chance of heart disease
# chol and target



heart_df['chol'].plot(kind='hist', bins=20, figsize =(10,5))

plt.title('chol of Patients', fontsize=14)

plt.xlabel('chol',fontsize=14)

plt.ylabel('Frequency',fontsize=14)

plt.show()

# most chol seem to be around 240 with outliers around 400-500
chol_bins = [80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420]

fig,ax=plt.subplots(figsize=(24,8))

heart_df['chol_bins']=pd.cut(heart_df['chol'], bins=chol_bins)

cholVsTarget=sns.countplot(x='chol_bins',data=heart_df ,hue='target',linewidth=3 )

cholVsTarget.set_title("chol vs Heart Disease")

heart_df.drop(['chol_bins'],axis=1,inplace=True)

# fbs and target



fbsVsTarget=sns.countplot(x='fbs',data=heart_df ,hue='target',linewidth=3, palette = 'pastel')

fbsVsTarget.set_title("fbs vs Heart Disease")



# fbs 0 may indicate more of a chance of heart disease, whereas 1 may not (seems to be close to 50-50)
# restecg and target



restecgVsTarget=sns.countplot(x='restecg',data=heart_df ,hue='target',linewidth=3, palette = 'bright')

restecgVsTarget.set_title("restecg vs Heart Disease")



# restecg of 1 may indicate heart disease whereas 0 and 2 may not
# thalach and target

# find bins

heart_df['thalach'].plot(kind='hist', bins=20, figsize =(10,5))

plt.title('thalach of Patients', fontsize=14)

plt.xlabel('thalach',fontsize=14)

plt.ylabel('Frequency',fontsize=14)

plt.show()

#right skewewed, most seem to be around 160bpm (max heart rate)
thalach_bins = [60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]

fig,ax=plt.subplots(figsize=(24,8))

heart_df['thalach_bins']=pd.cut(heart_df['thalach'], bins=thalach_bins)

thalachVsTarget=sns.countplot(x='thalach_bins',data=heart_df ,hue='target',linewidth=3 )

thalachVsTarget.set_title("thalach vs Heart Disease")

heart_df.drop(['thalach_bins'],axis=1,inplace=True)

#max heart rate above 150 seems to indicate heart disease
# exang and target (exercise induced angina)



exangVsTarget=sns.countplot(x='exang',data=heart_df ,hue='target',linewidth=3, palette = 'bright')

exangVsTarget.set_title("exang vs Heart Disease")



# exang of 0 may indicate heart disease whereas 1 may not

# oldpeak and target

# find bins

heart_df['oldpeak'].plot(kind='hist', bins=20, figsize =(10,5))

plt.title('oldpeak of Patients', fontsize=14)

plt.xlabel('oldpeak',fontsize=14)

plt.ylabel('Frequency',fontsize=14)

plt.show()

# extremely left skewed
oldpeak_bins = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]

fig,ax=plt.subplots(figsize=(24,8))

heart_df['oldpeak_bins']=pd.cut(heart_df['oldpeak'], bins=oldpeak_bins)

oldVsTarget=sns.countplot(x='oldpeak_bins',data=heart_df ,hue='target',linewidth=3 )

oldVsTarget.set_title("oldpeak vs Heart Disease")

heart_df.drop(['oldpeak_bins'],axis=1,inplace=True)

# an oldpeak below 1.5 may indicate heart disease
# slope and target



slopeVsTarget=sns.countplot(x='slope',data=heart_df ,hue='target',linewidth=3, palette = 'bright')

slopeVsTarget.set_title("slope vs Heart Disease")



# slope of 2 (downsloping) may indicate heart disease whereas 0 (upsloping) and 1 (flat) may not
# ca and target



caVsTarget=sns.countplot(x='ca',data=heart_df ,hue='target',linewidth=3, palette = 'muted')

caVsTarget.set_title("ca vs Heart Disease")



# no coloured cells may indicate no heart disease, whilst coloured cells may indicate no heart disease
# thal and target



thalVsTarget=sns.countplot(x='thal',data=heart_df ,hue='target',linewidth=3, palette = 'muted')

thalVsTarget.set_title("thal vs Heart Disease")



# thal of 2 seems to indicate heart disease whilst the others don't
heart = heart_df.drop(['target'], axis=1)

target = heart_df['target']

target.value_counts()
# one hot encoding

heart.dtypes
col_list = ['trestbps', 'chol', 'thalach', 'oldpeak']

for item in col_list:

    heart[item] = heart[item].astype(str).astype(float)

    

col_list2 = ['age']

for item2 in col_list2:

    heart[item2] = heart[item2].astype(str).astype(np.int64)

    

col_list3 = ['sex','cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

for item3 in col_list3:

    heart[item3] = heart[item3].astype(str).astype('object')

    

heart.dtypes
categorical_cols = heart.columns[heart.dtypes==object].tolist()

categorical_cols
for col in categorical_cols:

    n = len(heart[col].unique())

    if (n == 2):

        heart[col] = pd.get_dummies(heart[col], drop_first=True)

   

# use one-hot-encoding for categorical features with >2 levels

heart = pd.get_dummies(heart)
heart.columns
heart.sample(5, random_state=999)
from sklearn import preprocessing



heart_copy = heart.copy()

df = heart.copy()

Data_scaler = preprocessing.MinMaxScaler()

Data_scaler.fit(heart)

heart = Data_scaler.fit_transform(heart)
pd.DataFrame(heart, columns=heart_copy.columns).sample(5, random_state=999)
number_f = 10
fs_Fscore = fs.SelectKBest(fs.f_classif, k=number_f)

fs_Fscore.fit_transform(heart, target)

fs_colNum_fscore = np.argsort(fs_Fscore.scores_)[::-1][0:number_f]

fs_colNum_fscore
best_Features_Fscore = heart_copy.columns[fs_colNum_fscore].values

best_Features_Fscore
fs_importance = fs_Fscore.scores_[fs_colNum_fscore]

fs_importance
import altair as alt



def plot_importance(best_features, scores, method_name, color):

    

    df = pd.DataFrame({'features': best_features, 

                       'importances': scores})

    

    chart = alt.Chart(df,height = 100, width = 200, 

                      title=method_name + ' Feature Importances'

                     ).mark_bar(opacity=0.85, 

                                color=color).encode(

        alt.X('features', title='Feature', sort=None, axis=alt.AxisConfig(labelAngle=45)),

        alt.Y('importances', title='Importance')

    )

    

    return chart
plot_importance(best_Features_Fscore, fs_importance, "F-Score", "red")
heart_sample = pd.DataFrame(heart).values

target_sample = pd.DataFrame(target).values



print(heart_sample.shape)

print(target_sample.shape)
from sklearn.model_selection import train_test_split



heart_train, heart_test, \

target_train, target_test = train_test_split(heart_sample, target_sample, 

                                                    test_size = 0.3, random_state=999,

                                                    stratify = target_sample)



print(heart_train.shape)

print(heart_test.shape)
from sklearn.model_selection import StratifiedKFold, GridSearchCV



cross_val_method = StratifiedKFold(n_splits=5, random_state=999)
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier



pipe_KNN = Pipeline([('fselector', fs.SelectKBest()), 

                     ('knn', KNeighborsClassifier())])



params_pipe_KNN = {'fselector__k': [5,10, 20, heart.shape[1]],

                   'knn__n_neighbors': [1, 5, 10, 15, 20, 25, 30 ,35, 40, 60],

                   'knn__p': [1, 2, 3, 4, 5]}

 

GridSearch_KNN = GridSearchCV(estimator=pipe_KNN, 

                           param_grid=params_pipe_KNN, 

                           cv=cross_val_method,

                           scoring='roc_auc',

                           verbose=1) 

GridSearch_KNN.fit(heart_train, target_train);
GridSearch_KNN.best_params_
GridSearch_KNN.best_score_
def get_search_results(gs):



    def model_result(scores, params):

        scores = {'mean_score': np.mean(scores),

             'std_score': np.std(scores),

             'min_score': np.min(scores),

             'max_score': np.max(scores)}

        return pd.Series({**params,**scores})



    models = []

    scores = []



    for i in range(gs.n_splits_):

        key = f"split{i}_test_score"

        r = gs.cv_results_[key]        

        scores.append(r.reshape(-1,1))



    all_scores = np.hstack(scores)

    for p, s in zip(gs.cv_results_['params'], all_scores):

        models.append((model_result(s, p)))



    pipe_results = pd.concat(models, axis=1).T.sort_values(['mean_score'], ascending=False)



    columns_first = ['mean_score', 'std_score', 'max_score', 'min_score']

    columns = columns_first + [c for c in pipe_results.columns if c not in columns_first]



    return pipe_results[columns]
KNN_results = get_search_results(GridSearch_KNN)

KNN_results.head()
import altair as alt



def plot_GS_KNN(number_features):

    results_KNN_x_features = KNN_results[KNN_results['fselector__k'] == number_features]



    chart = alt.Chart(results_KNN_x_features, height = 100, width = 150,

          title='KNN Performance Comparison with ' + str(number_features) + " Features"

         ).mark_line(point=True).encode(

    alt.X('knn__n_neighbors', title='Number of Neighbors'),

    alt.Y('mean_score', title='AUC Score', scale=alt.Scale(zero=False)),

    alt.Color('knn__p:N', title='p')

    )

    return chart

plot_GS_KNN(20)
from sklearn.preprocessing import PowerTransformer

heart_train_transformed = PowerTransformer().fit_transform(heart_train)
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import RandomizedSearchCV



pipe_NB = Pipeline([('fselector', fs.SelectKBest()), 

                     ('nb', GaussianNB())])



params_pipe_NB = {'fselector__k': [5,10, 20, heart.shape[1]],

                  'nb__var_smoothing': np.logspace(1,-2, num=200)}





gs_NB = RandomizedSearchCV(estimator=pipe_NB, 

                          param_distributions=params_pipe_NB, 

                          cv=cross_val_method,

                          refit=True,

                          n_jobs=-2,

                          scoring='roc_auc',

                          verbose=1) 



gs_NB.fit(heart_train_transformed, target_train);
gs_NB.best_params_
gs_NB.best_score_
results_NB = get_search_results(gs_NB)

results_NB.head()
def plot_GS_NB(number_features):

    results_NB_x_features = results_NB[results_NB['fselector__k'] == number_features]



    chart = alt.Chart(results_NB_x_features,height = 100, width = 150,

          title='NB Performance Comparison with ' + str(number_features) + " Features"

         ).mark_line(point=True).encode(

    alt.X('nb__var_smoothing', title='Var. Smoothing'),

    alt.Y('mean_score', title='AUC Score', scale=alt.Scale(zero=False)),

    )

    return chart

plot_GS_NB(10)
from sklearn.naive_bayes import BernoulliNB
# Check for binary values

heart_copy.head()
heart_B = heart_copy.drop(['age', 'trestbps','chol','thalach', 'oldpeak'], axis=1)
heart_B_sample = pd.DataFrame(heart_B).values

target_B_sample = pd.DataFrame(target).values



print(heart_B_sample.shape)

print(target_B_sample.shape)
heart_B_train, heart_B_test, \

target_B_train, target_B_test = train_test_split(heart_B_sample, target_B_sample, 

                                                    test_size = 0.3, random_state=999,

                                                    stratify = target_sample)

print(heart_B_train.shape)

print(heart_B_test.shape)
pipe_b_NB = Pipeline([('fselector', fs.SelectKBest()), 

                     ('b_NB', BernoulliNB(binarize = None))])



params_pipe_b_NB = {'fselector__k': [5, 10, 15, 20, heart_B.shape[1]]}



gs_b_NB = RandomizedSearchCV(estimator=pipe_b_NB, 

                          param_distributions=params_pipe_b_NB, 

                          cv=cross_val_method,

                          refit=True,

                          n_jobs=-2,

                          scoring='roc_auc',

                          verbose=1) 



gs_b_NB.fit(heart_B_train, target_B_train);

gs_b_NB.best_params_
gs_b_NB.best_score_
results_b_NB = get_search_results(gs_b_NB)

results_b_NB.head()
def plot_GS_b_NB(results):



    chart = alt.Chart(results, height = 100, width = 150,

          title='Bernoulli NB Performance Comparison'

         ).mark_line(point=True).encode(

    alt.X('fselector__k', title='Features'),

    alt.Y('mean_score', title='AUC Score', scale=alt.Scale(zero=False)),

    )

    return chart
plot_GS_b_NB(results_b_NB)
heart_bin_B = heart_copy
heart_bin_B['age'].describe()
# Set up bins variables

heart_bin_B['age_bins'] = pd.cut(heart_bin_B['age'], 5)
heart_bin_B['trestbps'].describe()
heart_bin_B['trestbps_bins'] = pd.cut(heart_bin_B['trestbps'], 5)
heart_bin_B['chol'].describe()
heart_bin_B['chol_bins'] = pd.cut(heart_bin_B['chol'], 5)
heart_bin_B['thalach'].describe()
heart_bin_B['thalach_bins'] = pd.cut(heart_bin_B['thalach'], 5)
heart_bin_B['oldpeak'].describe()
heart_bin_B['oldpeak_bins'] = pd.cut(heart_bin_B['oldpeak'], 4)
heart_bin_B = pd.get_dummies(heart_bin_B)

heart_bin_B = heart_bin_B.drop(['age', 'trestbps','chol','thalach', 'oldpeak'], axis=1)
heart_B_bins = pd.DataFrame(heart_bin_B).values

target_B_bins = pd.DataFrame(target).values



print(heart_B_bins.shape)

print(target_B_bins.shape)
heart_B_bin_train, heart_B_bin_test, \

target_B_bin_train, target_B_bin_test = train_test_split(heart_B_bins, target_B_bins, 

                                                    test_size = 0.3, random_state=999,

                                                    stratify = target_sample)

print(heart_B_bin_train.shape)

print(heart_B_bin_test.shape)
pipe_b_bins_NB = Pipeline([('fselector', fs.SelectKBest()), 

                     ('b_NB', BernoulliNB(binarize = None))])



params_pipe_b_bins_NB = {'fselector__k': [5, 10, 15, 20, 25, 30, 35, 40, heart_B.shape[1]]}



gs_b_NB_binned = RandomizedSearchCV(estimator=pipe_b_bins_NB, 

                          param_distributions=params_pipe_b_bins_NB, 

                          cv=cross_val_method,

                          refit=True,

                          n_jobs=-2,

                          scoring='roc_auc',

                          verbose=1) 



gs_b_NB_binned.fit(heart_B_bin_train, target_B_bin_train);
gs_b_NB_binned.best_params_
gs_b_NB_binned.best_score_
results_b__binned_NB = get_search_results(gs_b_NB_binned)

results_b__binned_NB.head()
plot_GS_b_NB(results_b__binned_NB)
from sklearn.tree import DecisionTreeClassifier
DT_pipe = Pipeline([('fselector', fs.SelectKBest()), 

                     ('DT', DecisionTreeClassifier())])



DT_pipe_params = {'fselector__k' : [5, 10, 15, 20,  heart.shape[1]],

                  'DT__criterion': ['entropy', 'gini'],

                 'DT__max_depth' : [2,3,5,9,10],

                 'DT__min_samples_split' : [2,3,5,10,50,100]}



GridSearch_DT = GridSearchCV(estimator=DT_pipe, 

                          param_grid=DT_pipe_params, 

                          cv=cross_val_method,

                          refit=True,

                          n_jobs=-2,

                          scoring='roc_auc',

                          verbose=0) 
GridSearch_DT.fit(heart_train, target_train)
GridSearch_DT.best_params_
GridSearch_DT.best_score_
results_DT = get_search_results(GridSearch_DT)

results_DT.head()
def plot_DT_MaxDepth(number_of_features):

    results_DT_x_features = results_DT[results_DT['fselector__k'] == number_of_features]



    chart = alt.Chart(results_DT_x_features, height = 100, width = 150, 

          title='DT Performance Comparison with ' + str(number_of_features) +' Features'

             ).mark_line(point=True).encode(

        alt.X('DT__min_samples_split', title='Min Samples for Split'),

        alt.Y('mean_score', title='AUC Score', scale=alt.Scale(zero=False)),

        alt.Color('DT__max_depth:N', title='Max Depth')

    )

    return chart
plot_DT_MaxDepth(20)
def plot_DT_Cr(number_of_features):

    results_DT_x_features = results_DT[results_DT['fselector__k'] == number_of_features]



    chart = alt.Chart(results_DT_x_features, height = 100, width = 150,

          title='DT Performance Comparison with ' + str(number_of_features) +' Features'

             ).mark_line(point=True).encode(

        alt.X('DT__min_samples_split', title='Min Samples for Split'),

        alt.Y('mean_score', title='AUC Score', scale=alt.Scale(zero=False)),

        alt.Color('DT__criterion')

    )

    return chart
plot_DT_Cr(10)
from sklearn.linear_model import LogisticRegression
lr_pipe = Pipeline([('fselector', fs.SelectKBest()), 

                     ('lr', LogisticRegression())])



lr_pipe_params = {'fselector__k' : [5, 10, 15, 20,  heart.shape[1]],

                 'lr__penalty' : ['l1','l2']}



GridSearch_lr = GridSearchCV(estimator=lr_pipe, 

                          param_grid=lr_pipe_params, 

                          cv=cross_val_method,

                          refit=True,

                          n_jobs=-2,

                          scoring='roc_auc',

                          verbose=1) 
GridSearch_lr.fit(heart_train,target_train)
GridSearch_lr.best_params_
GridSearch_lr.best_score_
results_lr = get_search_results(GridSearch_lr)

results_lr.head()
def plot_lr(results):

    chart = alt.Chart(results, height = 100, width = 150,

          title='Logistic Regression Performance'

             ).mark_line(point=True).encode(

        alt.X('fselector__k', title='Number of Features'),

        alt.Y('mean_score', title='AUC Score', scale=alt.Scale(zero=False)),

        alt.Color('lr__penalty')

    )

    return chart
plot_lr(results_lr)
heart_lr = heart_bin_B.copy()

heart_lr = heart_lr.rename(columns={'age_bins_(28.952, 38.6]' : 'age0', 'age_bins_(38.6, 48.2]' : 'age1',

                                   'age_bins_(48.2, 57.8]' : 'age2', 'age_bins_(57.8, 67.4]' : 'age3',

       'age_bins_(67.4, 77.0]': 'age4', 'trestbps_bins_(93.894, 115.2]' : 'trest0',

       'trestbps_bins_(115.2, 136.4]' : 'trest1', 'trestbps_bins_(136.4, 157.6]' : 'trest2',

       'trestbps_bins_(157.6, 178.8]' : 'trest3', 'trestbps_bins_(178.8, 200.0]' : 'trest4',

       'chol_bins_(125.562, 213.6]' : 'chol0', 'chol_bins_(213.6, 301.2]' : 'chol1',

       'chol_bins_(301.2, 388.8]' : 'chol2', 'chol_bins_(388.8, 476.4]' : 'chol3',

       'chol_bins_(476.4, 564.0]' : 'chol4', 'thalach_bins_(70.869, 97.2]' : 'thalach0',

       'thalach_bins_(97.2, 123.4]' : 'thalach1', 'thalach_bins_(123.4, 149.6]' : 'thalach2',

       'thalach_bins_(149.6, 175.8]' : 'thalach3', 'thalach_bins_(175.8, 202.0]' : 'thalach4',

       'oldpeak_bins_(-0.0062, 1.55]' : 'old0', 'oldpeak_bins_(1.55, 3.1]' : 'old1',

       'oldpeak_bins_(3.1, 4.65]' : 'old2', 'oldpeak_bins_(4.65, 6.2]' : 'old3'

    

})

dummies = pd.get_dummies(target)

heart_lr['target1'] = dummies['1']
heart_lr.columns
import statsmodels.formula.api as smf

import statsmodels.api as sm

mylogit = smf.glm(formula='target1 ~  sex + fbs + exang + cp_0 + cp_1 + cp_2 + cp_3 + restecg_0 + restecg_1 + restecg_2 + slope_0 + slope_1 + slope_2 + ca_0 + ca_1 + ca_2 + ca_3 + thal_0 + thal_1 + thal_2 + thal_3 + age0 + age1 + age2 + age3 + age4 + trest0 + trest1 + trest2 + trest3 + trest4 + chol0 + chol1 + chol2 + chol3 + chol4 + thalach0 + thalach1 + thalach2 + thalach3 + thalach4 + old0 + old1 + old2 + old3', data=heart_lr, family=sm.families.Binomial())

print(mylogit.fit().summary())
import statsmodels.formula.api as smf

import statsmodels.api as sm

mylogit = smf.glm(formula='target1 ~  sex + fbs + exang + cp_0 + cp_1 + cp_2 + cp_3 + restecg_0 + restecg_1 + restecg_2 + slope_0 + slope_1 + slope_2 + ca_0 + ca_1 + ca_2 + ca_3 + thal_0 + thal_1 + thal_2 + thal_3', data=heart_lr, family=sm.families.Binomial())

print(mylogit.fit().summary2())
heart_lr1 = heart_lr.copy()

heart_lr1 = heart_lr1.drop(['fbs','cp_1','cp_3','restecg_0','restecg_2','slope_0','ca_1', 'ca_3','thal_0','thal_1','thal_2','thal_3'], axis = 1)
mylogit = smf.glm(formula='target1 ~  sex + exang + cp_0 + cp_2 + slope_1 + slope_2 + ca_0 + ca_2', data=heart_lr1, family=sm.families.Binomial())

print(mylogit.fit().summary2())
mylogit = smf.glm(formula='target1 ~  sex + exang + cp_0  + slope_2 + ca_0', data=heart_lr1, family=sm.families.Binomial())

print(mylogit.fit().summary2())
target1 = heart_lr1['target1']

heart_lr1 = heart_lr1.drop(['target1', 'cp_2', 'restecg_1', 'slope_1',

        'ca_2', 'age0', 'age1', 'age2', 'age3', 'age4', 'trest0',

       'trest1', 'trest2', 'trest3', 'trest4', 'chol0', 'chol1', 'chol2',

       'chol3', 'chol4', 'thalach0', 'thalach1', 'thalach2', 'thalach3',

       'thalach4', 'old0', 'old1', 'old2', 'old3'], axis = 1)

heart_lr_sample = pd.DataFrame(heart_lr1).values

target_lr_sample = pd.DataFrame(target1).values



print(heart_lr_sample.shape)

print(target_lr_sample.shape)
X_train, X_test, y_train, y_test = train_test_split(heart_lr_sample, target_lr_sample, test_size=0.3, random_state=0)
lr_simple_pipe = Pipeline([('fselector', fs.SelectKBest()), 

                     ('lr', LogisticRegression())])



lr_simple_pipe_params = {'fselector__k' : [1,2,3, heart_lr1.shape[1]],

                 'lr__penalty' : ['l1','l2']}



GridSearch_lr_simple = GridSearchCV(estimator=lr_simple_pipe, 

                          param_grid=lr_simple_pipe_params, 

                          cv=cross_val_method,

                          refit=True,

                          n_jobs=-2,

                          scoring='roc_auc',

                          verbose=1) 

GridSearch_lr_simple.fit(X_train, y_train)
GridSearch_lr_simple.best_params_
GridSearch_lr_simple.best_score_
results_s_lr = get_search_results(GridSearch_lr_simple)

results_s_lr.head()
plot_lr(results_s_lr)
from sklearn.model_selection import cross_val_score



ttest_cv = StratifiedKFold(n_splits=10, random_state=1)

results_b_NB_cv = cross_val_score(estimator=gs_b_NB.best_estimator_,

                                 X=heart_test,

                                 y=target_test, 

                                 cv=ttest_cv, 

                                 n_jobs=-2,

                                 scoring='roc_auc')

results_b_NB_cv.mean()
heart_test_transformed = PowerTransformer().fit_transform(heart_test)



results_NB_cv = cross_val_score(estimator=gs_NB.best_estimator_,

                                 X=heart_test_transformed,

                                 y=target_test, 

                                 cv=ttest_cv, 

                                 n_jobs=-2,

                                 scoring='roc_auc')

results_NB_cv.mean()
results_s_lr_cv = cross_val_score(estimator=GridSearch_lr_simple.best_estimator_,

                                 X=heart_test,

                                 y=target_test, 

                                 cv=ttest_cv, 

                                 n_jobs=-2,

                                 scoring='roc_auc')

results_s_lr_cv.mean()
results_LR_cv = cross_val_score(estimator=GridSearch_lr.best_estimator_,

                                 X=heart_test,

                                 y=target_test, 

                                 cv=ttest_cv, 

                                 n_jobs=-2,

                                 scoring='roc_auc')

results_LR_cv.mean()
results_DT_cv = cross_val_score(estimator=GridSearch_DT.best_estimator_,

                                 X=heart_test,

                                 y=target_test, 

                                 cv=ttest_cv, 

                                 n_jobs=-2,

                                 scoring='roc_auc')

results_DT_cv.mean()
results_KNN_cv = cross_val_score(estimator=GridSearch_KNN.best_estimator_,

                                 X=heart_test,

                                 y=target_test, 

                                 cv=ttest_cv, 

                                 n_jobs=-2,

                                 scoring='roc_auc')

results_KNN_cv.mean()
from scipy import stats

# round 1

print("t-test for Bernoulli NB vs Gaussian NB: \n" + str(stats.ttest_rel(results_b_NB_cv, results_NB_cv)) + "\n")

print("t-test for simple logistic regression vs full logistic regression: \n" + str(stats.ttest_rel(results_LR_cv, results_s_lr_cv)))
# round 2:

print("KNN vs DT: \n" + str(stats.ttest_rel(results_KNN_cv, results_DT_cv)) + "\n")

print("KNN vs Logistic Regression: \n" +str(stats.ttest_rel(results_KNN_cv, results_LR_cv))+ "\n")

print("Logistic Regression vs Bernoulli Naive Bayes: \n" +str(stats.ttest_rel(results_LR_cv, results_b_NB_cv))+ "\n")

print("KNN vs Bernoulli NB: \n" +str(stats.ttest_rel(results_KNN_cv, results_b_NB_cv))+ "\n")

print("DT vs Logistic Regression: \n" +str(stats.ttest_rel(results_DT_cv, results_LR_cv))+ "\n")

print("DT vs Bernoulli NB: \n" +str(stats.ttest_rel(results_DT_cv, results_b_NB_cv))+ "\n")
# KNN 

KNN_pred = GridSearch_KNN.predict(heart_test)
# Gaussian NB

G_NB_pred = gs_NB.predict(heart_test_transformed)
# Bernoulli NB

B_NB_pred = gs_b_NB.predict(heart_B_test)
# DT

pred_DT = GridSearch_DT.predict(heart_test)
# Simple Logistic Regression

pred_S_LR = GridSearch_lr_simple.predict(X_test)
# Full Logistic Regression

pred_FullLR = GridSearch_lr.predict(heart_test)
from sklearn import metrics

print("\nClassification report for K-Nearest Neighbor") 

print(metrics.classification_report(target_test, KNN_pred))

print("\nClassification report for Gaussian Naive Bayes") 

print(metrics.classification_report(target_test, G_NB_pred))

print("\nClassification report for Decision Tree") 

print(metrics.classification_report(target_test, pred_DT))

print("\nClassification report for Simple Logistic Regression") 

print(metrics.classification_report(y_test, pred_S_LR))

print("\nClassification report for Full Logistic Regression") 

print(metrics.classification_report(target_test, pred_FullLR))

print("\nClassification report for Bernoulli NB") 

print(metrics.classification_report(target_test, B_NB_pred))
from sklearn import metrics

print("\nConfusion Matrix report for K-Nearest Neighbor") 

print(metrics.confusion_matrix(target_test, KNN_pred))

print("\nConfusion Matrix for Gaussian Naive Bayes") 

print(metrics.confusion_matrix(target_test, G_NB_pred))

print("\nConfusion Matrix for Decision Tree") 

print(metrics.confusion_matrix(target_test, pred_DT))

print("\nConfusion Matrix for Simple Logistic Regression") 

print(metrics.confusion_matrix(y_test, pred_S_LR))

print("\nConfusion Matrix for Full Logistic Regression") 

print(metrics.confusion_matrix(target_test, pred_FullLR))

print("\nConfusion Matrix for Bernoulli NB") 

print(metrics.confusion_matrix(target_test, B_NB_pred))
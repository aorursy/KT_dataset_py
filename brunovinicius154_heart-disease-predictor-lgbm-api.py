# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# General Propose

import pickle

import requests

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import shapiro

from scikitplot import metrics as mt



# Hipo Test

from scipy import stats

from scipy.stats import f_oneway

from scipy.stats import ttest_ind





# Pre-processing

from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.compose import ColumnTransformer

from sklearn.feature_selection import VarianceThreshold

from imblearn.combine import SMOTETomek

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder



# Modelling

import lightgbm as lgb

from catboost import CatBoostClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans





# Evaluation

from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score, precision_score, f1_score, recall_score

from yellowbrick.classifier.threshold import discrimination_threshold



# Set options and warnings

import warnings

warnings.filterwarnings('ignore')

pd.set_option('MAX_ROWS', None)




# Helper Functions

def balanced_target(target, dataset, hue=None):

    """

    Function to check the balancing of the target variable.



    :target:  An pd.Series of the target variable that will be checked.

    :dataset: An Dataframe object. 

    """

    sns.set(style='darkgrid', palette='Accent')

    ax = sns.countplot(x=target, hue=hue, data=dataset)

    ax.figure.set_size_inches(10, 6)

    ax.set_title('Cardio Distribution', fontsize=18, loc='left')

    ax.set_xlabel(target, fontsize=14)

    ax.set_ylabel('Count', fontsize=14)

    ax=ax





def univariate_analysis(target, df):

    """

    Function to perform univariate analysis.



    df: DataFrame

    """

    for col in df.columns.to_list():



        fig = sns.displot(x=col, hue=target, data=df, kind='hist')

        fig.set_titles(f'{col}\n distribuition', fontsize=16)

        fig.set_axis_labels(col, fontsize=14)





def multi_histogram(data: pd.DataFrame, variables: list) -> None:



    # set of initial plot posistion

    plt.figure(figsize=(18, 10))

    n = 1

    for column in data[variables].columns:

        plt.subplot(3, 3, n)

        _ = sns.distplot(a=data[column], bins=50, hist=True)

        n += 1



    plt.subplots_adjust(hspace=0.3)



    plt.show()







def multi_boxplot(data: pd.DataFrame, variables: list) -> None:



    """

    Function to check for outliers visually through a boxplot



    data: DataFrame



    variable: list of numerical variables

    """



    # set of initial plot posistion

    plt.figure(figsize=(18, 10))

    n = 1

    for column in data[variables].columns:

        plt.subplot(3, 3, n)

        _ = sns.boxplot(x=column, data=data)

        n += 1



    plt.subplots_adjust(hspace=0.3)



    plt.show()





def hipo_test(*samples):



    samples = samples



    try:

        if len(samples) == 2:

            stat, p = ttest_ind(*samples)

        elif len(samples) > 2:

            stat, p = f_oneway(*samples)

    except:

        raise Exception("Deve ser fornecido pelo menos duas samples!!!")



    if p < 0.05:

        print(f'O valor de p é: {p}')

        print('Provável haver diferença')

    else:

        print(f'O valor de p é: {p}')

        print('Provável que não haja diferença')



    return stat, p





def point_bi_corr(a, b):



    """

    Function to calculate point biserial correlation coefficient heatmap function

    Credits: Bruno Santos - Comunidade DS



    :a: input dataframe with binary variable

    :b: input dataframe with continous variable

    """



    # Get column name

    a = a.values.reshape(-1)

    b = b.columns.reshape(-1)



    # apply scipys point-biserial

    stats.pointbiserialr(a, b)



    # correlation coefficient array

    c = np.corrcoef(a, b)



    # dataframe for heatmap

    df = pd.DataFrame(c, columns=[a, b], index=[a, b])



    # return heatmap

    return sns.heatmap(df, annot=True).set_title('{} x {} correlation heatmap'.format(a, b));





def change_threshold_lgbm(X, y, model, n_splits, thresh):



    # cross-validação

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)



    acc = []

    kappa = []

    recall = []

    for linhas_treino, linhas_valid in skf.split(X, y):



        X_treino, X_valid = X.iloc[linhas_treino], X.iloc[linhas_valid]

        y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]



        pred_prob = model.predict_proba(X_valid)



        for i in range(0, len(pred_prob)):

            if pred_prob[i, 1] >= thresh:

                pred_prob[i, 1] = 1

            else:

                pred_prob[i, 1] = 0



        Acc = accuracy_score(y_valid, pred_prob[:, 1])

        Kappa =  cohen_kappa_score(y_valid, pred_prob[:, 1])

        Recall = recall_score(y_valid, pred_prob[:, 1])

        acc.append(Acc)

        kappa.append(Kappa)

        recall.append(Recall)



    print('####### Bussines Metrics #######')

    print('\n')

    acc_inc = np.mean(acc) - 0.50

    prc_inc = round((acc_inc/0.05)*500, 2)

    print(f'Increased precision: {round(acc_inc,2)}')

    print(f'Price Increased in: {prc_inc}')

    print(f'Percentual of Price increassing: {round(prc_inc/500,2)}')

    print('\n')



    # print classification report

    print('####### Machine Learning Metrics #######\n')

    print(classification_report(y_valid, pred_prob[:,1], digits=2))



    # Confusion Matrix

    mt.plot_confusion_matrix(y_valid, pred_prob[:,1], normalize=False, figsize=(10,8))



    return pred_prob[:, 1]





def change_threshold_lr(X, y, model, n_splits, thresh):

    # cross-validação

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)



    acc = []

    kappa = []

    recall = []

    for linhas_treino, linhas_valid in skf.split(X, y):



        X_treino, X_valid = X.iloc[linhas_treino], X.iloc[linhas_valid]

        y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]



        y_scores_final = model.decision_function(X_valid)

        y_pred_recall = (y_scores_final > thresh)



        Acc = accuracy_score(y_valid, y_pred_recall)

        Kappa =  cohen_kappa_score(y_valid, y_pred_recall)

        Recall = recall_score(y_valid, y_pred_recall)

        acc.append(Acc)

        kappa.append(Kappa)

        recall.append(Recall)



    print('####### Bussines Metrics #######\n')



    acc_inc = np.mean(acc) - 0.50

    prc_inc = round((acc_inc/0.05)*500, 2)

    print(f'Increased precision: {round(acc_inc,2)}')

    print(f'Price Increased in: {prc_inc}')

    print(f'Percentual of Price increassing: {round(prc_inc/500,2)}')

    print('\n')



    print('####### Machine Learning Metrics #######\n')

    print(f'New kappa: {cohen_kappa_score(y_valid,y_pred_recall)}\n')

    print(classification_report(y_valid, y_pred_recall, digits=2))





    return y_pred_recall





################################################# Custons Transformers ###########################################################



class PreProcessingTransformer(BaseEstimator, TransformerMixin):



    def __init__(self):

        pass



    def fit(self, X, y=None):

        return self



    def transform(self, X, y=None):

        Xtemp = X.copy()



        # Height

        index_height = Xtemp.loc[Xtemp['height'] > 230, ['height']].index

        Xtemp.drop(index_height, inplace=True)

        index_height1 = Xtemp.loc[Xtemp['height'] < 112, ['height']].index

        Xtemp.drop(index_height1, inplace=True)



        # Weight

        index_weight = Xtemp.loc[Xtemp['weight'] < 40, ['weight']].index

        Xtemp.drop(index_weight, inplace=True)



        # ap_hi

        index_ap_hi = Xtemp.loc[Xtemp['ap_hi'] < 10, ['ap_hi']].index

        Xtemp.drop(index_ap_hi, inplace=True)



        # ap_lo

        index_ap_lo = Xtemp.loc[Xtemp['ap_lo'] < 5, ['ap_lo']].index

        Xtemp.drop(index_ap_lo, inplace=True)



        # SMOTE + TOMEKLINK

        X = Xtemp.drop('cardio', axis=1)

        y = Xtemp['cardio']



        smt = SMOTETomek(random_state=42)

        Xres, yres = smt.fit_resample(X, y)

        Xtemp = pd.concat([Xres, yres], axis=1)



        return Xtemp





class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):



    def __init__(self):

        pass



    def fit(self, X, y=None):

        return self



    def transform(self, X, y=None):

        Xtemp = X.copy()



        # Cluster based var

        kmeans = KMeans(n_clusters=2, init='k-means++',n_init=20, random_state=0).fit(Xtemp)

        Xtemp['kmeans_cat'] = kmeans.labels_



        # # Cluster GMM

        # gmm = GaussianMixture(n_components=3).fit(Xtemp)

        # Xtemp['gauss_cat'] = gmm.predict(Xtemp)



        # Year_age

        Xtemp['year_age'] = Xtemp['age'] / 365



        # drop 'id' and 'age' 'smoke','alco','gluc', 'ap_lo', 'cholesterol', 'height', 'active', 'weight'

        Xtemp.drop(['id', 'age'], inplace=True, axis=1)



        # IMC

        Xtemp['imc'] = Xtemp['weight']/(Xtemp['height']/100)**2



        # cat_dwarfism

        Xtemp['cat_Dwarfism'] = [1 if value < 145 else 0 for value in Xtemp['height']]



        # ap_hi divide 10

        Xtemp.loc[Xtemp['ap_hi'] > 220, ['ap_hi']] = Xtemp.loc[Xtemp['ap_hi'] > 220, ['ap_hi']]/10



        # ap_lo divide 10

        Xtemp.loc[Xtemp['ap_lo'] > 190, ['ap_lo']] = Xtemp.loc[Xtemp['ap_lo'] > 190, ['ap_lo']]/10



        # ap_hi divide 10

        Xtemp.loc[Xtemp['ap_hi'] > 220, ['ap_hi']] = Xtemp.loc[Xtemp['ap_hi'] > 220,['ap_hi']]/10



        # ap_hi divide 10

        Xtemp.loc[Xtemp['ap_hi'] > 220, ['ap_hi']] = Xtemp.loc[Xtemp['ap_hi'] > 220,['ap_hi']]/10



        # ap_lo divide 10

        Xtemp.loc[Xtemp['ap_lo'] > 190, ['ap_lo']] = Xtemp.loc[Xtemp['ap_lo'] > 190,['ap_lo']]/10



        return Xtemp





class CatBloodPressureTransformer(BaseEstimator, TransformerMixin):



    def __init__(self):

        pass



    def fit(self, X, y=None):

        return self



    def transform(self, X, y=None):

        Xtemp = X.copy()



        # cat_bloodpressure

        def cat_bloodpressure(df):



            if df['ap_hi'] < 90 and df['ap_lo'] < 60:

                return 1 #Hipotensão

            elif 90 <= df['ap_hi'] < 140 and 60 <= df['ap_lo'] < 90:

                return 2    # Pré-Hipotensão

            elif 140 <= df['ap_hi'] < 160 and 90 <= df['ap_lo'] < 100:

                return 3  # 'Hipertensão estagio1'

            elif df['ap_hi'] >= 160 and df['ap_lo'] >= 100:

                return 4 # 'Hipertensão estagio2'

            else:

                return 5 # 'no_cat'



        # cat_bloodpressure

        Xtemp['cat_bloodpressure'] = Xtemp.apply(cat_bloodpressure, axis=1)



        return Xtemp





class TotalPressureTransformer(BaseEstimator, TransformerMixin):



    def __init__(self):

        pass



    def fit(self, X, y=None):



        return self



    def transform(self, X, y=None):



        Xtemp = X.copy()



        # total_preassure

        Xtemp['total_pressure'] = Xtemp['ap_hi'] + Xtemp['ap_lo']



        return Xtemp





class MyRobustScalerTransformer(BaseEstimator, TransformerMixin):



    def __init__(self):

        pass



    def fit(self, X, y=None):

        return self



    def transform(self, X, y=None):



        Xtemp = X.copy()



        scaler = RobustScaler()

        Xscaled = scaler.fit_transform(Xtemp)

        Xtemp = pd.DataFrame(Xscaled, columns=Xtemp.columns.to_list())



        return Xtemp
train = pd.read_csv('/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv', sep=';')

print(f'The number of columns are {train.shape[1]}')

print(f'The number of rows are {train.shape[0]}')
# Sneak look

train.head()
# Details of the data

pd.DataFrame({'missingPerc': train.isna().mean(),

              'uniques': train.nunique(),

              '%uniquePerc': round((train.nunique()/train.shape[0])*100, 2),

              'data_types': train.dtypes,

              'mean': round(train.mean(), 2),

              'median': round(train.median(), 2),

              'std': round(train.std(), 2),

              'min': round(train.min(), 2),

              'max': round(train.max(), 2)})
# splitting data

dfTrain = train[:60000]



# Validação

dfValid = train[60000:]

X = dfValid.drop(['cardio'], axis=1)

y = dfValid['cardio']
# Checking all numerical vars

numerical_vars = ['age','height', 'weight', 'ap_hi', 'ap_lo']

multi_histogram(data=train, variables=numerical_vars)
# Target Analysys

balanced_target(target='cardio',dataset=train)
# changing the variable 'Age' to years instead of days

age_df = train.copy()

age_df['age_years'] = age_df['age'] / 365



print(age_df['age_years'].describe())

print('\n')



# Plot distribution

fig = sns.distplot(a=age_df['age_years'], hist=True)
# Describe

print(train['height'].describe())

print('\n')



# Plot distribution

fig = sns.distplot(a=train['height'], hist=True)
# Describe

print(train['weight'].describe())

print('\n')



# Plot distribution

fig = sns.distplot(a=train['weight'], hist=True)
# count below 10 values

train.loc[train['ap_hi']<10,:].size
# count values over 220

train.loc[train['ap_hi']>220,:].size
# Dropping

df = train.copy()

index = df.loc[(df['ap_hi']<10)|(df['ap_hi']>220),['ap_hi']].index

df.drop(index, inplace=True)



# Describe

print(df['ap_hi'].describe())

print('\n')



# Plot distribution

fig = sns.distplot(a=df['ap_hi'], hist=True)
# Dropping

df = train.copy()

index = df.loc[(df['ap_lo']<5)|(df['ap_lo']>190),['ap_lo']].index

df.drop(index, inplace=True)



# Describe

print(df['ap_lo'].describe())

print('\n')



# Plot distribution

fig = sns.distplot(a=df['ap_lo'], hist=True)
out_train = train.drop(['id','gender','gluc','smoke','cholesterol','active','alco','cardio'],axis=1)

vars = out_train.columns.to_list()



multi_boxplot(data=train, variables=vars)
# Set of Variables

bi_train = train.drop(['id','gender','gluc','smoke','cholesterol','active','alco'],axis=1)
# 'age' vs 'weight'

sns.scatterplot(x='age',y='weight',hue='cardio',data=bi_train)
# 'age' vs 'height'

sns.scatterplot(x='age',y='height',hue='cardio',data=bi_train)
sns.scatterplot(x='age',y='ap_lo',hue='cardio',data=bi_train)
sns.scatterplot(x='age',y='ap_hi',hue='cardio',data=bi_train)
# cardio distribution by 'cholesterol'

balanced_target(target='cholesterol',hue='cardio', dataset=train)
# cardio distribution by 'gluc'

balanced_target(target='gluc',hue='cardio', dataset=train)
# cardio distribution by 'alco'

balanced_target(target='alco',hue='cardio', dataset=train)
# cardio distribution by 'active'

balanced_target(target='active',hue='cardio', dataset=train)
sample_class0 = train.loc[train['cardio']==0,'age']

sample_class1 = train.loc[train['cardio']==1,'age']



print(f'The Mean to class 0: {np.mean(sample_class0)}')

print(f'The Mean to class 1: {np.mean(sample_class1)}')

print('\n')



# Testing the Hypothesis

hipo_test(sample_class0, sample_class1)
sample_class0 = train.loc[train['cardio']==0,'weight']

sample_class1 = train.loc[train['cardio']==1,'weight']



print(f'The Mean to class 0: {np.mean(sample_class0)}')

print(f'The Mean to class 1: {np.mean(sample_class1)}')

print('\n')



# Testing the Hypothesis

hipo_test(sample_class0, sample_class1)
sample_class0 = train.loc[train['cardio']==0,'height']

sample_class1 = train.loc[train['cardio']==1,'height']



print(f'The Mean to class 0: {np.mean(sample_class0)}')

print(f'The Mean to class 1: {np.mean(sample_class1)}')

print('\n')



# Testing the Hypothesis

hipo_test(sample_class0, sample_class1)
transf = PreProcessingTransformer()

df1 = transf.fit_transform(dfTrain)

df1.head()
# Testing the FeatureEng

tranf = FeatureEngineeringTransformer()

df = tranf.fit_transform(dfTrain)

df.head()
pipeline = Pipeline(steps=[('feateng', FeatureEngineeringTransformer()),

                           ('totalpress', TotalPressureTransformer()),

                           ('catpress', CatBloodPressureTransformer())

                           ])



ct = ColumnTransformer([('numerical',MyRobustScalerTransformer(), ['year_age','height','weight', 'ap_hi', 'ap_lo','total_pressure', 'imc']),

                       ('categorical',OneHotEncoder(drop='first'), ['cholesterol', 'gluc', 'active', 'gender', 'smoke', 'alco','cat_bloodpressure','cat_Dwarfism','kmeans_cat'])])





pipeline_final = Pipeline(steps= [('geral',pipeline),

                                  ('num_cat', ct)])



exemp = pipeline_final.fit_transform(dfTrain)

pd.DataFrame(exemp).head()
# Training Model

def train_LR(X, n_iter, n_splits=10):



    print('> Iniciando Treinamento...')



    cleaning_pipeline = Pipeline(steps=[('preproc', PreProcessingTransformer())])



    pipeline = Pipeline(steps=[('feateng', FeatureEngineeringTransformer()),

                               ('totalpress', TotalPressureTransformer()),

                               ('catpress', CatBloodPressureTransformer())

                               ])



    ct = ColumnTransformer([('numerical',MyRobustScalerTransformer(),  ['year_age', 'height', 'weight', 'ap_hi', 'ap_lo', 'total_pressure', 'imc']),

                           ('categorical',OneHotEncoder(drop='first'), ['cholesterol', 'gluc', 'active', 'gender', 'smoke', 'alco', 'cat_bloodpressure', 'cat_Dwarfism', 'kmeans_cat'])])



    pipeline_final = Pipeline(steps= [('geral',pipeline),

                                      ('num_cat', ct),

                                      ('model', LogisticRegression(C=0.5, n_jobs=-1))])



    data = cleaning_pipeline.fit_transform(X)

    X = data.drop(['cardio'], axis=1)

    y = data['cardio']



    # Fit the random search model

    print('> Fitting Modelo...')



    # cross-validação

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)



    acc = []

    kappa = []

    recall = []

    for linhas_treino, linhas_valid in skf.split(X, y):



        X_treino, X_valid = X.iloc[linhas_treino], X.iloc[linhas_valid]

        y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]





        pipeline_final.fit(X_treino, y_treino)

        pred = pipeline_final.predict(X_valid)

        Acc = accuracy_score(y_valid, pred)

        Kappa =  cohen_kappa_score(y_valid, pred)

        Recall = recall_score(y_valid, pred)

        acc.append(Acc)

        kappa.append(Kappa)

        recall.append(Recall)



    print('####### Bussines Metrics #######')



    print('\n')

    acc_inc = np.mean(acc) - 0.50

    prc_inc = round((acc_inc/0.05)*500, 2)

    print(f'Increased accuracy: {round(acc_inc,2)}')

    print(f'Price Increased in: {prc_inc}')

    print(f'Percentual of Price increassing: {round((prc_inc/500),2)}')

    print('\n')



    print('####### Machine Learning Metrics #######')

    print(f'Accuracy mean: {np.mean(acc)}')

    print(f'Accuracy std: {np.std(acc)}')



    print('\n')



    print(f'kappa mean: {np.mean(kappa)}')

    print(f'kappa std: {np.std(kappa)}')



    print('\n')



    print(f'Recall mean: {np.mean(recall)}')

    print(f'Recall std: {np.std(recall)}')



    print('> Treinamento realizado...')

    return pipeline_final



model_lr = train_LR(X=dfTrain, n_iter=5)
# Discrimination_threshold

discrimination_threshold(model_lr, X, y)
# Changing the Threshold - Increasing Precision

new_predictions = change_threshold_lr(X=X, y=y, model=model_lr, n_splits=10, thresh=0.5)
# Changing the Threshold - Increasing Recall

new_predictions = change_threshold_lr(X=X, y=y, model=model_lr, n_splits=10, thresh=-0.58)
# Training Model

def train_RF(X, n_iter, n_splits=10):



    print('> Iniciando Treinamento...')



    cleaning_pipeline = Pipeline(steps=[('preproc', PreProcessingTransformer())])



    pipeline = Pipeline(steps=[('feateng', FeatureEngineeringTransformer()),

                               ('totalpress', TotalPressureTransformer()),

                               ('catpress', CatBloodPressureTransformer())

                               ])



    ct = ColumnTransformer([('numerical',MyRobustScalerTransformer(), ['year_age', 'height', 'weight', 'ap_hi', 'ap_lo', 'total_pressure', 'imc']),

                           ('categorical',OneHotEncoder(drop='first'), ['cholesterol', 'gluc', 'active', 'gender', 'smoke', 'alco', 'cat_bloodpressure', 'cat_Dwarfism', 'kmeans_cat'])])





    pipeline_final = Pipeline(steps= [('geral',pipeline),

                                      ('num_cat', ct),

                                      ('model', RandomForestClassifier(n_jobs=-1))])





    data = cleaning_pipeline.fit_transform(X)

    X = data.drop(['cardio'], axis=1)

    y = data['cardio']





    # Fit the random search model

    print('> Fitting Modelo...')



    # cross-validação

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)



    acc = []

    kappa = []

    recall = []

    for linhas_treino, linhas_valid in skf.split(X, y):



        X_treino, X_valid = X.iloc[linhas_treino], X.iloc[linhas_valid]

        y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]





        pipeline_final.fit(X_treino, y_treino)

        pred = pipeline_final.predict(X_valid)

        Acc = accuracy_score(y_valid, pred)

        Kappa =  cohen_kappa_score(y_valid, pred)

        Recall = recall_score(y_valid, pred)

        acc.append(Acc)

        kappa.append(Kappa)

        recall.append(Recall)



    print('####### Bussines Metrics #######')



    print('\n')

    acc_inc = np.mean(acc) - 0.50

    prc_inc = round((acc_inc/0.05)*500, 2)

    print(f'Increased accuracy: {round(acc_inc,2)}')

    print(f'Price Increased in: {prc_inc}')

    print(f'Percentual of Price increassing: {round((prc_inc/500),2)}')

    print('\n')



    print('####### Machine Learning Metrics #######')

    print(f'Accuracy mean: {np.mean(acc)}')

    print(f'Accuracy std: {np.std(acc)}')



    print('\n')



    print(f'kappa mean: {np.mean(kappa)}')

    print(f'kappa std: {np.std(kappa)}')



    print('\n')



    print(f'Recall mean: {np.mean(recall)}')

    print(f'Recall std: {np.std(recall)}')



    print('> Treinamento realizado...')



    return pipeline_final



model_rf = train_RF(X=dfTrain, n_iter=5)
# Training LGBM



def train_lightGBM(X, n_iter, n_splits=10):



    print('> Iniciando Treinamento...')



    cleaning_pipeline = Pipeline(steps=[('preproc', PreProcessingTransformer())])



    pipeline = Pipeline(steps=[('feateng', FeatureEngineeringTransformer()),

                               ('totalpress', TotalPressureTransformer()),

                               ('catpress', CatBloodPressureTransformer())

                               ])



    ct = ColumnTransformer([('numerical',MyRobustScalerTransformer(), ['year_age', 'height', 'weight', 'ap_hi', 'ap_lo', 'total_pressure', 'imc']),

                           ('categorical',OneHotEncoder(drop='first'), ['cholesterol', 'gluc', 'active', 'gender', 'smoke', 'alco', 'cat_bloodpressure', 'cat_Dwarfism', 'kmeans_cat'])])



    pipeline_final = Pipeline(steps= [('geral',pipeline),

                                      ('num_cat', ct),

                                      ('feature_selection', VarianceThreshold(threshold=0.1)),

                                      ('model', lgb.LGBMClassifier())])



    data = cleaning_pipeline.fit_transform(X)

    X = data.drop(['cardio'], axis=1)

    y = data['cardio']



    # Fit the model

    print('> Fitting Modelo...')



    # cross-validação

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)



    acc = []

    kappa = []

    recall = []

    for linhas_treino, linhas_valid in skf.split(X, y):



        X_treino, X_valid = X.iloc[linhas_treino], X.iloc[linhas_valid]

        y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]



        pipeline_final.fit(X_treino, y_treino)

        pred = pipeline_final.predict(X_valid)

        Acc = accuracy_score(y_valid, pred)

        Kappa =  cohen_kappa_score(y_valid, pred)

        Recall = recall_score(y_valid, pred)

        acc.append(Acc)

        kappa.append(Kappa)

        recall.append(Recall)



    print('####### Bussines Metrics #######')



    print('\n')

    acc_inc = np.mean(acc) - 0.50

    prc_inc = round((acc_inc/0.05)*500, 2)

    print(f'Increased accuracy: {round(acc_inc,2)}')

    print(f'Price Increased in: {prc_inc}')

    print(f'Percentual of Price increassing: {round((prc_inc/500),2)}')

    print('\n')



    print('####### Machine Learning Metrics #######')

    print(f'Accuracy mean: {np.mean(acc)}')

    print(f'Accuracy std: {np.std(acc)}')



    print('\n')



    print(f'kappa mean: {np.mean(kappa)}')

    print(f'kappa std: {np.std(kappa)}')



    print('\n')



    print(f'Recall mean: {np.mean(recall)}')

    print(f'Recall std: {np.std(recall)}')



    print('> Treinamento realizado...')



    return pipeline_final



model_lgbm = train_lightGBM(X=dfTrain, n_iter=5)
# Tunnig Parameters - GridSearch



from sklearn.model_selection import GridSearchCV



model = model_lgbm





param_grid = {'learning_rate': [0.1, 1, 1.2],

              'n_estimators': [100, 500, 1000],

              'num_leaves': [10, 31, 50],

              'min_child_samples' : [10, 20, 30]

             }



# Function to tune the parameters

def tunnig_gridsearch(Xtrain, model, param_grid, cv, scoring, refit):



    """

    Função para tunnig de parâmetros utilizando o GridSearchCV



    """





    cleaning_pipeline = Pipeline(steps=[('preproc', PreProcessingTransformer())])



    data = cleaning_pipeline.fit_transform(Xtrain)

    X_ = data.drop(['cardio'], axis=1)

    y_ = data['cardio']





    search = GridSearchCV(estimator=model.named_steps['model'],

                          param_grid=param_grid,

                          scoring=scoring,

                          refit=refit,

                          cv=cv,

                          verbose=1,

                          n_jobs=-1,

                          return_train_score=True)

    search.fit(X_, y_)



    return search.best_params_, search.cv_results_



# Function to retrain with the new parameters and over the all data

def retrain_lightGBM(X,valid, params, n_iter, n_splits=10):



    print('> Iniciando Treinamento...')



    cleaning_pipeline = Pipeline(steps=[('preproc', PreProcessingTransformer())])



    pipeline = Pipeline(steps=[('feateng', FeatureEngineeringTransformer()),

                               ('totalpress', TotalPressureTransformer()),

                               ('catpress', CatBloodPressureTransformer())

                               ])



    ct = ColumnTransformer([('numerical',MyRobustScalerTransformer(), ['year_age', 'height', 'weight', 'ap_hi', 'ap_lo', 'total_pressure', 'imc']),

                           ('categorical',OneHotEncoder(drop='first'), ['cholesterol', 'gluc', 'active', 'gender', 'smoke', 'alco', 'cat_bloodpressure', 'cat_Dwarfism', 'kmeans_cat'])])



    pipeline_final = Pipeline(steps= [('geral',pipeline),

                                      ('num_cat', ct),

                                      ('feature_selection', VarianceThreshold(threshold=0.1)),

                                      ('model', lgb.LGBMClassifier(learning_rate=params['learning_rate'],

                                                                   n_estimators=params['n_estimators'],

                                                                   num_leaves=params['num_leaves'],

                                                                   min_child_samples=params['min_child_samples']

                                                                   ))])



    data = cleaning_pipeline.fit_transform(X)

    X = data.drop(['cardio'], axis=1)

    y = data['cardio']



    # Fit the model

    print('> Fitting Modelo...')



    # cross-validação

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)



    acc = []

    kappa = []

    recall = []

    for linhas_treino, linhas_valid in skf.split(X, y):



        X_treino, X_valid = X.iloc[linhas_treino], X.iloc[linhas_valid]

        y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]



        pipeline_final.fit(X_treino, y_treino)

        pred = pipeline_final.predict(X_valid)

        Acc = accuracy_score(y_valid, pred)

        Kappa =  cohen_kappa_score(y_valid, pred)

        Recall = recall_score(y_valid, pred)

        acc.append(Acc)

        kappa.append(Kappa)

        recall.append(Recall)



    # modelo treinado somente nas 60000 obs



    Xvalid = valid.drop(['cardio'], axis=1)

    yvalid = valid['cardio']



    tunned_pipe = pipeline_final.fit(Xvalid, yvalid)



    # retrain over all data

    tunned_model = pipeline_final.fit(X, y)





    print('####### Bussines Metrics #######')



    print('\n')

    acc_inc = np.mean(acc) - 0.50

    prc_inc = round((acc_inc/0.05)*500, 2)

    print(f'Increased accuracy: {round(acc_inc,2)}')

    print(f'Price Increased in: {prc_inc}')

    print(f'Percentual of Price increassing: {round((prc_inc/500),2)}')

    print('\n')



    print('####### Machine Learning Metrics #######')

    print(f'Accuracy mean: {np.mean(acc)}')

    print(f'Accuracy std: {np.std(acc)}')



    print('\n')



    print(f'kappa mean: {np.mean(kappa)}')

    print(f'kappa std: {np.std(kappa)}')



    print('\n')



    print(f'Recall mean: {np.mean(recall)}')

    print(f'Recall std: {np.std(recall)}')



    print('> Treinamento realizado...')



    return tunned_model, tunned_pipe







# Calling the Functions

best_params, cv_results = tunnig_gridsearch(Xtrain=dfTrain,

                                            model=model,

                                            param_grid=param_grid,

                                            scoring='accuracy',

                                            refit='accuracy',

                                            cv=5)



tunned_model, tunned_pipe = retrain_lightGBM(train, dfValid, params=best_params, n_iter=5)
# Best params

best_params
# Discrimination_threshold

discrimination_threshold(tunned_pipe, X, y)
# changing the threshold - Recall

new_pred_prob = change_threshold_lgbm(X=X, y=y, model=tunned_pipe, n_splits=10, thresh=0.40)
# changing the threshold - Precision

new_pred_prob = change_threshold_lgbm(X=X, y=y, model=tunned_pipe, n_splits=10, thresh=0.6)
# CatBoost

def train_catBoost(X, n_iter, n_splits=10):



    print('> Iniciando Treinamento...')

    cleaning_pipeline = Pipeline(steps=[('preproc', PreProcessingTransformer())])



    pipeline = Pipeline(steps=[('feateng', FeatureEngineeringTransformer()),

                               ('totalpress', TotalPressureTransformer()),

                               ('catpress', CatBloodPressureTransformer())

                               ])



    ct = ColumnTransformer([('numerical',MyRobustScalerTransformer(), ['year_age', 'height', 'weight', 'ap_hi', 'ap_lo', 'total_pressure', 'imc']),

                           ('categorical',OneHotEncoder(drop='first'), ['cholesterol', 'gluc', 'active', 'gender', 'smoke', 'alco', 'cat_bloodpressure', 'cat_Dwarfism', 'kmeans_cat'])

                           ])





    pipeline_final = Pipeline(steps= [('geral',pipeline),

                                      ('num_cat', ct),

                                      ('model', CatBoostClassifier(

                                                            n_estimators=100,

                                                            depth=6,

                                                            l2_leaf_reg=0.0,

                                                            bagging_temperature=1,

                                                            early_stopping_rounds=100,

                                                            loss_function='Logloss',

                                                            eval_metric='Accuracy',

                                                            verbose=False))])



    data = cleaning_pipeline.fit_transform(X)

    X = data.drop(['cardio'], axis=1)

    y = data['cardio']



    # Fit the random search model

    print('> Fitting Modelo...')



    # cross-validação

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)



    acc = []

    kappa = []

    recall = []

    for linhas_treino, linhas_valid in skf.split(X, y):



        X_treino, X_valid = X.iloc[linhas_treino], X.iloc[linhas_valid]

        y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]



        pipeline_final.fit(X_treino, y_treino)

        pred = pipeline_final.predict(X_valid)

        Acc = accuracy_score(y_valid, pred)

        Kappa =  cohen_kappa_score(y_valid, pred)

        Recall = recall_score(y_valid, pred)

        acc.append(Acc)

        kappa.append(Kappa)

        recall.append(Recall)



    print('####### Bussines Metrics #######')



    print('\n')

    acc_inc = np.mean(acc) - 0.50

    prc_inc = round((acc_inc/0.05)*500, 2)

    print(f'Increased accuracy: {round(acc_inc,2)}')

    print(f'Price Increased in: {prc_inc}')

    print(f'Percentual of Price increassing: {round((prc_inc/500),2)}')

    print('\n')



    print('####### Machine Learning Metrics #######')

    print(f'Accuracy mean: {np.mean(acc)}')

    print(f'Accuracy std: {np.std(acc)}')



    print('\n')



    print(f'kappa mean: {np.mean(kappa)}')

    print(f'kappa std: {np.std(kappa)}')



    print('\n')



    print(f'Recall mean: {np.mean(recall)}')

    print(f'Recall std: {np.std(recall)}')



    print('> Treinamento realizado...')



    return pipeline_final



model_cat = train_catBoost(X=train, n_iter=5)
# Discrimination_threshold

discrimination_threshold(model_lgbm, X, y)
# changing the threshold - Recall

new_pred_prob = change_threshold_lgbm(X=X, y=y, model=model_cat , n_splits=10, thresh=0.35)
# changing the threshold - Precission

new_pred_prob = change_threshold_lgbm(X=X, y=y, model=model_cat , n_splits=10, thresh=0.6)
# Example Data

test_data = X.sample()



# Tunnig into a json

df_json = test_data.to_json(orient='records')

df_json
# url = 'http://0.0.0.0:5000/predict'

url = 'https://pa001-app.herokuapp.com/predict'

data = df_json

header = {'Content-type': 'application/json'}



# Request

r = requests.post(url=url, data=data, headers=header)

print(r.status_code)

r.json()
# url = 'http://0.0.0.0:5000/predict_thresh'

url = 'https://pa001-app.herokuapp.com/predict_thresh'

data = df_json

header = {'Content-type': 'application/json'}



# Request

r = requests.post(url=url, data=data, headers=header)

print(r.status_code)

r.json()
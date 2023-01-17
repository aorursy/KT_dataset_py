import os
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option('display.max_columns', 50)
pd.options.display.float_format = '{:,.2f}'.format
general = pd.read_csv('/kaggle/input/covid19-mx/covid-19_general_MX.csv')
general.rename( columns={'Unnamed: 0' :'#'}, inplace=True )

print(general.shape)
display(general.head())
print(general.info())
from sklearn.preprocessing import StandardScaler
features = general.copy()

# Parse dates
features['FECHA_INGRESO'] = pd.to_datetime(features['FECHA_INGRESO'])
features['FECHA_SINTOMAS'] = pd.to_datetime(features['FECHA_SINTOMAS'])
features['MUERTE'] = (features['FECHA_DEF'] != '9999-99-99').replace([True, False], [1, 0])
features['FECHA_DEF'].replace({'9999-99-99': '2099-01-01'}, inplace=True)
features['FECHA_DEF'] = pd.to_datetime(features['FECHA_DEF'])
features['DIAS'] = (features['FECHA_DEF'] - features['FECHA_SINTOMAS']).dt.days

# Select inputs and autputs
cond_headers = ['DIABETES', 'EPOC', 'ASMA', 'INMUSUPR', 'HIPERTENSION', 'CARDIOVASCULAR', 'OBESIDAD', 'RENAL_CRONICA', 'TABAQUISMO', 'OTRA_CON']
outcome_headers = ['NEUMONIA', 'INTUBADO', 'UCI', 'MUERTE']

# Format binary features
binary_headers = list(set(features.columns) - set(['index', 'EDAD']))
features[binary_headers] = features[binary_headers].replace({1: 1, 2: 0, 3: np.nan, 97: np.nan, 98: np.nan, 99: np.nan})
features.loc[:] = features.dropna()

#features['EDAD'] = StandardScaler().fit_transform(features['EDAD'].values.reshape(-1,1))

# Select only confirmed COVID cases, drop unused columns
infectados = features[features['RESULTADO'] == 1]
infectados = infectados.drop(columns=['SECTOR', 'ENTIDAD_UM', 'ENTIDAD_RES', 'TIPO_PACIENTE', 'NACIONALIDAD', 
                                'OTRO_CASO', 'FECHA_INGRESO', 'FECHA_SINTOMAS', 'FECHA_DEF', 'RESULTADO', '#'])

feature_headers = list(set(infectados.columns) - set(outcome_headers))

display(infectados.info())
fig, ax = plt.subplots(1, len(outcome_headers), figsize=(25,5))
for i, target in enumerate(outcome_headers):
    groups = infectados[cond_headers + [target, 'SEXO']].groupby([target, 'SEXO']).sum()
    groups = groups.unstack().unstack().reset_index(name='count')
    groups.replace('SEXO', {0:'male', 1:'female'}, inplace=True)

    alive = groups[groups[target]==0]
    dead = groups[groups[target]==1].copy()
    rate = dead['count'] / (dead['count'] + alive['count'].values)
    dead['rate']=rate

    sns.barplot(data=dead.sort_values('rate', axis='rows', ascending=False),
                ax=ax[i],
                x='level_0',
                y='rate',
                hue='SEXO',
                palette=sns.color_palette("BrBG", 2))
    ax[i].tick_params(labelrotation=60)
    ax[i].set_title('Percent of patients that resulted in ' + target)
    ax[i].set_xlabel('')
    ax[i].set_ylabel('')
plt.show()
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].grid()
ax[1].grid()

# Days histogram
data = infectados[infectados['MUERTE']==1]['DIAS']
sns.distplot(data, ax = ax[0]).set_title('Days from symptoms to death')
ax[0].set_xlabel('days')
plt.sca(ax[0])
plt.xlim(0, 40)

# Age histogram
data = infectados['EDAD']
sns.distplot(data,
             hist=True, 
             kde=False, 
             ax=ax[1], 
             label='Total cases').set_title('Death distribution by age')
data = infectados[infectados['MUERTE']==1]['EDAD']
sns.distplot(data, 
             hist=True,
             kde=False,
             ax=ax[1],
             label='Deaths').set_title('Cases distribution by age')
ax[1].legend()
ax[1].set_xlabel('age')

# Death histogram
sns.lineplot(data=infectados.replace('SEXO', {0: 'Male', 1: 'Female'}),
             x='EDAD',
             y='MUERTE',
             hue='SEXO',
             ci=None,
             ax=ax[2]).set_title('Death rate by age and sex')
ax[2].set_xlabel('age')
ax[2].set_ylabel('')
plt.sca(ax[2])
plt.xlim(20, 80)
plt.ylim(0, .6)

plt.show()
from sklearn.preprocessing import PowerTransformer

def plot_death_rate(column, ax, scale=False):
    df = infectados.groupby(['EDAD', 'MUERTE']).mean().reset_index()
    
    plt.sca(ax)
    plt.xlim(20, 80)
#    plt.ylim(0, .15)
    if (scale):
        df[column] = PowerTransformer().fit_transform(df[column].values.reshape(-1,1))
    else: 
        ax.set_ylabel('Percent of ' + column)

    ax.scatter(data=df[df['MUERTE']==0], x='EDAD', y=column, label='non fatal')
    ax.scatter(data=df[df['MUERTE']==1], x='EDAD', y=column, label='fatal')
    ax.set_xlabel('age')
    ax.set_title(column + ' by age group')
    ax.legend()
    

headers = ['DIABETES', 'HIPERTENSION', 'OBESIDAD']
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
for i, condition in enumerate(headers):
    plot_death_rate(condition, ax[i])
plt.show()    

headers = list(set(cond_headers) - set(headers))
fig, ax = plt.subplots(1, 7, figsize=(30, 5))
for i, condition in enumerate(headers):
    plot_death_rate(condition, ax[i], True)
plt.show()    
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
def grid_search_report(estimator, params, X_train, y_train):
    cv = GridSearchCV(estimator, params, cv=3).fit(X_train, y_train)
    results = pd.DataFrame(cv.cv_results_)
    test_scores = results[['split0_test_score', 'split1_test_score', 'split2_test_score']]#, 'split3_test_score', 'split4_test_score']]
    test_scores = test_scores.transpose()
    sns.barplot(data=test_scores)
    plt.xlabel('combinations')
    plt.ylabel('train score')
    plt.show()
    print('Best param combination: ', cv.best_params_)
    return cv
def print_classification_results(estimator, X, y, plot_proba=False, debug=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    
    if (debug):
        print('\nPositives count: ' + str(len(positives)) + '; Negatives count: ' + str(len(negatives)), '\n')
        print('INPUT:\n')
        display(X_train.info())
        print('\nCONFUSION MATRIX:\n\n' + str(metrics.confusion_matrix(y_test, y_pred)))
        print('\nCLASSIFICATION REPORT:\n\n', metrics.classification_report(y_test, y_pred))

    y_proba = estimator.predict_proba(X_test)
    comparison = pd.DataFrame(columns=['EDAD', 'prediction', 'actual result'])
    comparison['EDAD'] = X_test['EDAD'].reset_index(drop=True)
    comparison['prediction'] = pd.DataFrame(y_proba)[1]
    comparison['actual result'] = y_test.reset_index(drop=True)

    if (plot_proba):
        plt.figure(figsize=(10, 10))
#        plt.xlim(20,85)
        sns.scatterplot(data=comparison, x='EDAD', y='prediction', hue='actual result')
        plt.ylabel('predicted probability for ' + target)
        plt.xlabel('normalized age')
        plt.show()
        
    return comparison
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import fbeta_score, make_scorer

def get_cv_results(estimator, X, y):
    scoring = {
            'f1': make_scorer(metrics.f1_score),
            'recall': make_scorer(metrics.recall_score),
            'accuracy': make_scorer(metrics.accuracy_score)
            }
    cv_results = cross_validate(estimator, X, y, scoring=scoring)
    mean_f1 = np.mean(cv_results['test_f1'])
    mean_accuracy = np.mean(cv_results['test_accuracy'])
    mean_recall = np.mean(cv_results['test_recall'])
    
    df = pd.DataFrame([[type(estimator).__name__, mean_accuracy, mean_recall, mean_f1]], columns=['Estimator', 'Accuracy Score', 'Recall Score', 'F1 Score'])
    return df
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import ensemble

# Balance dataset
positives = infectados[infectados['MUERTE']==1]
negatives = infectados[infectados['MUERTE']==0]
min_samples = min(2000, len(positives), len(negatives))
negatives = negatives.sample(min_samples)
positives = positives.sample(min_samples)
balanced_data = pd.concat([negatives, positives])

# Split
input_headers = ['EDAD', 'SEXO', 'OBESIDAD', 'CARDIOVASCULAR', 'RENAL_CRONICA', 'INMUSUPR']
X = balanced_data[input_headers]
y = balanced_data['MUERTE']

# Scale
X = pd.DataFrame(PowerTransformer().fit_transform(X), columns=input_headers)

# Grid Search
#grid_search_report(RadiusNeighborsClassifier(),{}, X, y)

# Report cross validation results
pd.options.display.float_format = '{:,.4f}'.format
df = pd.DataFrame()
df = df.append(get_cv_results(KNeighborsClassifier(n_neighbors=500), X, y))
df = df.append(get_cv_results(MLPClassifier(activation='logistic'), X, y))
df = df.append(get_cv_results(linear_model.LogisticRegression(), X, y))
df = df.append(get_cv_results(linear_model.RidgeClassifier(normalize=True), X, y))
df = df.append(get_cv_results(SVC(C=.01, kernel='poly', degree=1), X, y))
#df = df.append(get_cv_results(DecisionTreeClassifier(), X, y))
#df = df.append(get_cv_results(RadiusNeighborsClassifier(p=1, weights='distance', radius=10), X, y))
display(df.reset_index().sort_values(by='F1 Score', ascending=False))

selected_estimator = KNeighborsClassifier(n_neighbors=500)
results = print_classification_results(selected_estimator, X, y, True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Configs
pd.options.display.float_format = '{:,.4f}'.format
sns.set(style="whitegrid")
plt.style.use('seaborn')
seed = 42
np.random.seed(seed)
file_path = '/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)
print("DataSet = {:,d} rows and {} columns".format(df.shape[0], df.shape[1]))

print("\nAll Columns:\n=>", df.columns.tolist())

quantitative = [f for f in df.columns if df.dtypes[f] != 'object']
qualitative = [f for f in df.columns if df.dtypes[f] == 'object']

print("\nStrings Variables:\n=>", qualitative,
      "\n\nNumerics Variables:\n=>", quantitative)

df.head(3)
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score

this_labels = ['No Churn','Churn']
scoress = {}

def class_report(y_real, y_my_preds, name="", labels=this_labels):
    if(name != ''):
        print(name,"\n")
    print(confusion_matrix(y_real, y_my_preds), '\n')
    print(classification_report(y_real, y_my_preds, target_names=labels))
    scoress[name] = [accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro')]
import time

def time_spent(time0):
    t = time.time() - time0
    t_int = int(t) // 60
    t_min = t % 60
    if(t_int != 0):
        return '{} min {:.3f} s'.format(t_int, t_min)
    else:
        return '{:.3f} s'.format(t_min)
# statistics
from scipy import stats
from scipy.stats import norm, skew, boxcox_normmax #for some statistics
from scipy.special import boxcox1p

def test_normal_distribution(serie, series_name='series', thershold=0.4):
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6), sharex=False)
    f.suptitle('{} is a Normal Distribution?'.format(series_name), fontsize=18)
    ax1.set_title("Histogram to " + series_name)
    ax2.set_title("Q-Q-Plot to "+ series_name)
    mu, sigma = norm.fit(serie)
    print('Normal dist. (mu= {:,.2f} and sigma= {:,.2f} )'.format(mu, sigma))
    skewness = serie.skew()
    kurtoise = serie.kurt()
    print("Skewness: {:,.2f} | Kurtosis: {:,.2f}".format(skewness, kurtoise))
    pre_text = '\t=> '
    if(skewness < 0):
        text = pre_text + 'negatively skewed or left-skewed'
    else:
        text =  pre_text + 'positively skewed or right-skewed\n'
        text += pre_text + 'in case of positive skewness, log transformations usually works well.\n'
        text += pre_text + 'np.log(), np.log1(), boxcox1p()'
    if(skewness < -1 or skewness > 1):
        print("Evaluate skewness: highly skewed")
        print(text)
    if( (skewness <= -0.5 and skewness > -1) or (skewness >= 0.5 and skewness < 1)):
        print("Evaluate skewness: moderately skewed")
        print(text)
    if(skewness >= -0.5 and skewness <= 0.5):
        print('Evaluate skewness: approximately symmetric')
    print('evaluate kurtoise')
    if(kurtoise > 3 + thershold):
        print(pre_text + 'Leptokurtic: anormal: Peak is higher')
    elif(kurtoise < 3 - thershold):
        print(pre_text + 'Platykurtic: anormal: The peak is lower')
    else:
        print(pre_text + 'Mesokurtic: normal: the peack is normal')
    sns.distplot(serie , fit=norm, ax=ax1)
    ax1.legend(['Normal dist. ($\mu=$ {:,.2f} and $\sigma=$ {:,.2f} )'.format(mu, sigma)],
            loc='best')
    ax1.set_ylabel('Frequency')
    stats.probplot(serie, plot=ax2)
    plt.show()
def plot_top_bottom_rank_correlation(my_df, column_target, top_rank=5, title=''):
    corr_matrix = my_df.corr()
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7), sharex=False)
    if(title):
        f.suptitle(title)

    ax1.set_title('Top {} Positive Corr to {}'.format(top_rank, column_target))
    ax2.set_title('Top {} Negative Corr to {}'.format(top_rank, column_target))
    
    cols_top = corr_matrix.nlargest(top_rank+1, column_target)[column_target].index
    cm = np.corrcoef(my_df[cols_top].values.T)
    mask = np.zeros_like(cm)
    mask[np.triu_indices_from(mask)] = True
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 11}, yticklabels=cols_top.values,
                     xticklabels=cols_top.values, mask=mask, ax=ax1)
    
    cols_bot = corr_matrix.nsmallest(top_rank, column_target)[column_target].index
    cols_bot  = cols_bot.insert(0, column_target)
    cm = np.corrcoef(my_df[cols_bot].values.T)
    mask = np.zeros_like(cm)
    mask[np.triu_indices_from(mask)] = True
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                     annot_kws={'size': 11}, yticklabels=cols_bot.values,
                     xticklabels=cols_bot.values, mask=mask, ax=ax2)
    
    plt.show()
def check_balanced_train_test_binary(x_train, y_train, x_test, y_test, original_size, labels):
    """ To binary classification
    each paramethes is pandas.core.frame.DataFrame
    @total_size = len(X) before split
    @labels = labels in ordem [0,1 ...]
    """
    train_unique_label, train_counts_label = np.unique(y_train, return_counts=True)
    test_unique_label, test_counts_label = np.unique(y_test, return_counts=True)

    prop_train = train_counts_label/ len(y_train)
    prop_test = test_counts_label/ len(y_test)

    print("Original Size:", '{:,d}'.format(original_size))
    print("\nTrain: must be 80% of dataset:\n", 
          "the train dataset has {:,d} rows".format(len(x_train)),
          'this is ({:.2%}) of original dataset'.format(len(x_train)/original_size),
                "\n => Classe 0 ({}):".format(labels[0]), train_counts_label[0], '({:.2%})'.format(prop_train[0]), 
                "\n => Classe 1 ({}):".format(labels[1]), train_counts_label[1], '({:.2%})'.format(prop_train[1]),
          "\n\nTest: must be 20% of dataset:\n",
          "the test dataset has {:,d} rows".format(len(x_test)),
          'this is ({:.2%}) of original dataset'.format(len(x_test)/original_size),
                  "\n => Classe 0 ({}):".format(labels[0]), test_counts_label[0], '({:.2%})'.format(prop_test[0]),
                  "\n => Classe 1 ({}):".format(labels[1]),test_counts_label[1], '({:.2%})'.format(prop_test[1])
         )
def eda_categ_feat_desc_plot(series_categorical, title = "", fix_labels=False):
    """Generate 2 plots: barplot with quantity and pieplot with percentage. 
       @series_categorical: categorical series
       @title: optional
       @fix_labels: The labes plot in barplot in sorted by values, some times its bugs cuz axis ticks is alphabethic
           if this happens, pass True in fix_labels
       @bar_format: pass {:,.0f} to int
    """
    series_name = series_categorical.name
    val_counts = series_categorical.value_counts()
    val_counts.name = 'quantity'
    val_percentage = series_categorical.value_counts(normalize=True)
    val_percentage.name = "percentage"
    val_concat = pd.concat([val_counts, val_percentage], axis = 1)
    val_concat.reset_index(level=0, inplace=True)
    val_concat = val_concat.rename( columns = {'index': series_name} )
    
    fig, ax = plt.subplots(figsize = (12,4), ncols=2, nrows=1) # figsize = (width, height)
    if(title != ""):
        fig.suptitle(title, fontsize=18)
        fig.subplots_adjust(top=0.8)

    s = sns.barplot(x=series_name, y='quantity', data=val_concat, ax=ax[0])
    if(fix_labels):
        val_concat = val_concat.sort_values(series_name).reset_index()
    
    for index, row in val_concat.iterrows():
        s.text(row.name, row['quantity'], '{:,d}'.format(int(row['quantity'])), color='black', ha="center")

    s2 = val_concat.plot.pie(y='percentage', autopct=lambda value: '{:.2f}%'.format(value),
                             labels=val_concat[series_name].tolist(), legend=None, ax=ax[1],
                             title="Percentage Plot")

    ax[1].set_ylabel('')
    ax[0].set_title('Quantity Plot')

    plt.show()
def eda_numerical_feat(series, title="", with_label=True, number_format="", show_describe=False, size_labels=10):
    # Use 'series_remove_outiliers' to filter outiliers
    """ Generate series.describe(), bosplot and displot to a series
    @with_label: show labels in boxplot
    @number_format: 
        integer: 
            '{:d}'.format(42) => '42'
            '{:,d}'.format(12855787591251) => '12,855,787,591,251'
        float:
            '{:.0f}'.format(91.00000) => '91' # no decimal places
            '{:.2f}'.format(42.7668)  => '42.77' # two decimal places and round
            '{:,.4f}'.format(1285591251.78) => '1,285,591,251.7800'
            '{:.2%}'.format(0.09) => '9.00%' # Percentage Format
        string:
            ab = '$ {:,.4f}'.format(651.78) => '$ 651.7800'
    def swap(string, v1, v2):
        return string.replace(v1, "!").replace(v2, v1).replace('!',v2)
    # Using
        swap(ab, ',', '.')
    """
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 5), sharex=False)
    if(show_describe):
        print(series.describe())
    if(title != ""):
        f.suptitle(title, fontsize=18)
    sns.distplot(series, ax=ax1)
    sns.boxplot(series, ax=ax2)
    if(with_label):
        describe = series.describe()
        labels = { 'min': describe.loc['min'], 'max': describe.loc['max'], 
              'Q1': describe.loc['25%'], 'Q2': describe.loc['50%'],
              'Q3': describe.loc['75%']}
        if(number_format != ""):
            for k, v in labels.items():
                ax2.text(v, 0.3, k + "\n" + number_format.format(v), ha='center', va='center', fontweight='bold',
                         size=size_labels, color='white', bbox=dict(facecolor='#445A64'))
        else:
            for k, v in labels.items():
                ax2.text(v, 0.3, k + "\n" + str(v), ha='center', va='center', fontweight='bold',
                     size=size_labels, color='white', bbox=dict(facecolor='#445A64'))
    plt.show()
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
df.duplicated().sum()

df['TotalCharges'] = df['TotalCharges'].replace(" ", 0).astype('float32')
df.head()
eda_categ_feat_desc_plot(df['gender'], title = "gender distribution")
eda_categ_feat_desc_plot(df['SeniorCitizen'], title = "SeniorCitizen distribution")
eda_categ_feat_desc_plot(df['Partner'], title = "Partner distribution")
# PhoneService
eda_categ_feat_desc_plot(df['PhoneService'], title = "PhoneService distribution")
# PhoneService
eda_categ_feat_desc_plot(df['MultipleLines'], title = "MultipleLines distribution")
# PhoneService
eda_categ_feat_desc_plot(df['InternetService'], title = "InternetService distribution")
# PhoneService
eda_categ_feat_desc_plot(df['OnlineSecurity'], title = "OnlineSecurity distribution")
# InternetService

# InternetService
eda_categ_feat_desc_plot(df['InternetService'], title = "InternetService distribution")
eda_numerical_feat(df['MonthlyCharges'], title="MonthlyCharges distribution")
eda_categ_feat_desc_plot(df['Churn'], title = "Churn distribution")
df['StreamingMovies'].value_counts()
yes_no = {'No':0, 'Yes': 1}
gender = {'Female':0, 'Male':1}

df1 = df.copy().drop(['customerID'], axis=1)

df1['Churn'] = df1['Churn'].replace(yes_no)
df1['PaperlessBilling'] = df1['PaperlessBilling'].replace(yes_no)
df1['Partner'] = df1['Partner'].replace(yes_no)
df1['Dependents'] = df1['Dependents'].replace(yes_no)
df1['PhoneService'] = df1['PhoneService'].replace(yes_no)

df1['gender'] = df1['gender'].replace(gender)

multiple_lines = pd.get_dummies(df1['MultipleLines'], prefix='ML')
internet_service = pd.get_dummies(df1['InternetService'], prefix='IS')
online_security = pd.get_dummies(df1['OnlineSecurity'], prefix='OS')
online_backup = pd.get_dummies(df1['OnlineBackup'], prefix='OB')

device_protection = pd.get_dummies(df1['DeviceProtection'], prefix='DP')
tech_support = pd.get_dummies(df1['TechSupport'], prefix='TS')
streaming_tv = pd.get_dummies(df1['StreamingTV'], prefix='ST')
streaming_movies = pd.get_dummies(df1['StreamingMovies'], prefix='SM')

contract = pd.get_dummies(df1['Contract'], prefix='Contr')
payment_method = pd.get_dummies(df1['PaymentMethod'], prefix='PM')

dummies_columns = [multiple_lines, internet_service, online_security, online_backup,
                  device_protection, tech_support, streaming_tv, streaming_movies,
                  contract, payment_method]

df1['TotalCharges'] = df1['TotalCharges'].replace(" ", 0).astype('float32')

df1 = pd.concat([df1, *dummies_columns], axis=1)

df1 = df1.drop(['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
               'Contract', 'PaymentMethod'], axis=1)

df1
abc = plot_top_bottom_rank_correlation(df1, 'Churn', top_rank=12, title='Top Cors')
df1
from sklearn.model_selection import train_test_split

X = df1.drop(['Churn'], axis=1)

y = df1['Churn']

x_train, x_test, y_train, y_test = train_test_split(X, y.values, test_size=0.20, random_state=42)

check_balanced_train_test_binary(x_train, y_train, x_test, y_test, len(df), ['Response 0', 'Response 1'])
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek # over and under sampling
from imblearn.metrics import classification_report_imbalanced

imb_models = {
    'ADASYN': ADASYN(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'SMOTEENN': SMOTEENN("minority", random_state=42),
    'SMOTETomek': SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'), random_state=42),
    'RandomUnderSampler': RandomUnderSampler(random_state=42)
}

imb_strategy = "None"

if(imb_strategy != "None"):
    before = x_train.shape[0]
    imb_tranformer = imb_models[imb_strategy]
    x_train, y_train = imb_tranformer.fit_sample(x_train, y_train)
    print("train dataset before: {:,d}\nimbalanced_strategy: {}".format(before, imb_strategy),
          "\ntrain dataset after: {:,d}\ngenerate: {:,d}".format(x_train.shape[0], x_train.shape[0] - before))
else:
    print("Dont correct unbalanced dataset")
# Classifier Libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Ensemble Classifiers
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Others Linear Classifiers
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier

# xboost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# scores
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# neural net of sklearn
from sklearn.neural_network import MLPClassifier

# others
import time
import operator
all_classifiers = {
    "NaiveBayes": GaussianNB(),
    "Ridge": RidgeClassifier(),
    "Perceptron": Perceptron(),
    "PassiveAggr": PassiveAggressiveClassifier(),
    "XGBoost": XGBClassifier(),
    "LightGB": LGBMClassifier(boosting_type='gbdt',n_estimators=500,depth=10,learning_rate=0.04,objective='binary',
                 colsample_bytree=0.5,reg_lambda=2,reg_alpha=2,random_state=294,n_jobs=-1),
    "SVM": SVC(),
    "LogisiticR": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
#     "DecisionTree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(), # All 100 features: 48min
    # "SGDC": SGDClassifier(),
    "GBoost": GradientBoostingClassifier(),
#     "Bagging": BaggingClassifier(),
    "RandomForest": RandomForestClassifier(),
    "ExtraTree": ExtraTreesClassifier()
}
metrics = { 'cv_acc': {}, 'acc_test': {}, 'f1_test': {} }
m = list(metrics.keys())
time_start = time.time()
print('CrossValidation, Fitting and Testing')

# Cross Validation, Fit and Test
for name, model in all_classifiers.items():
    print('{:15}'.format(name), end='')
    t0 = time.time()
    # Cross Validation
    training_score = cross_val_score(model, x_train, y_train, scoring="accuracy", cv=4)
    # Fitting
    all_classifiers[name] = model.fit(x_train, y_train) 
    # Testing
    y_pred = all_classifiers[name].predict(x_test)
    t1 = time.time()
    # Save metrics
    metrics[m[0]][name] = training_score.mean()
    metrics[m[1]][name] = accuracy_score(y_test, y_pred)
    metrics[m[2]][name] = f1_score(y_test, y_pred, average="macro") 
    # Show metrics
    print('| {}: {:6,.4f} | {}: {:6,.4f} | {}: {:6.4f} | took: {:>15} |'.format(
        m[0], metrics[m[0]][name], m[1], metrics[m[1]][name],
        m[2], metrics[m[2]][name], time_spent(t0) ))
        
print("\nDone in {}".format(time_spent(time_start)))
print("Best cv acc  :", max( metrics[m[0]].items(), key=operator.itemgetter(1) ))
print("Best acc test:", max( metrics[m[1]].items(), key=operator.itemgetter(1) ))
print("Best f1 test :", max( metrics[m[2]].items(), key=operator.itemgetter(1) ))

df_metrics = pd.DataFrame(data = [list(metrics[m[0]].values()),
                                  list(metrics[m[1]].values()),
                                  list(metrics[m[2]].values())],
                          index = ['cv_acc', 'acc_test', 'f1_test' ],
                          columns = metrics[m[0]].keys() ).T.sort_values(by=m[2], ascending=False)
df_metrics
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

name = 'CatBoost'
catb = CatBoostClassifier()

t0 = time.time()
# Fitting
catb = catb.fit(x_train, y_train, eval_set=(x_test, y_test), plot=False, early_stopping_rounds=30,verbose=0)
# catb = catb.fit(x_train, y_train, cat_features=cat_col, eval_set=(x_test, y_test), plot=False, early_stopping_rounds=30,verbose=0) 
# Testing
y_pred = catb.predict(x_test)
t1 = time.time()
# Save metrics
metrics[m[0]][name] = 0.0
metrics[m[1]][name] = accuracy_score(y_test, y_pred)
metrics[m[2]][name] = f1_score(y_test, y_pred, average="macro") 

# Show metrics
print('{:15} | {}: {:6,.4f} | {}: {:6.4f} | took: {:>15} |'.format(
    name, m[1], metrics[m[1]][name],
    m[2], metrics[m[2]][name], time_spent(t0) ))
feat_importances = pd.Series(catb.feature_importances_, index=X.columns)
feat_importances.nlargest(25).plot(kind='barh')
plt.show()
from sklearn.model_selection import GridSearchCV

def optimize_logistic_r(mx_train, my_train, my_hyper_params, hyper_to_search, hyper_search_name, cv=4, scoring='accuracy'):
    """search best param to unic one hyper param
    @mx_train, @my_train = x_train, y_train of dataset
    @my_hyper_params: dict with actuals best_params: start like: {}
      => will be accumulated and modified with each optimization iteration
      => example stater: best_hyper_params = {'random_state': 42, 'n_jobs': -1}
    @hyper_to_search: dict with key @hyper_search_name and list of values to gridSearch:
    @hyper_search_name: name of hyperparam
    """
    if(hyper_search_name in my_hyper_params.keys()):
        del my_hyper_params[hyper_search_name]
    if(hyper_search_name not in hyper_to_search.keys()):
        raise Exception('"hyper_to_search" dont have {} in dict'.format(hyper_search_name))
        
    t0 = time.time()
        
    rf = LogisticRegression(**my_hyper_params)
    
    grid_search = GridSearchCV(estimator = rf, param_grid = hyper_to_search, 
      scoring = scoring, n_jobs = -1, cv = cv)
    grid_search.fit(mx_train, my_train)
    
    print('took', time_spent(t0))
    
    data_frame_results = pd.DataFrame(
        data={'mean_fit_time': grid_search.cv_results_['mean_fit_time'],
        'mean_test_score_'+scoring: grid_search.cv_results_['mean_test_score'],
        'ranking': grid_search.cv_results_['rank_test_score']
         },
        index=grid_search.cv_results_['params']).sort_values(by='ranking')
    
    print('The Best HyperParam to "{}" is {} with {} in {}'.format(
        hyper_search_name, grid_search.best_params_[hyper_search_name], grid_search.best_score_, scoring))
    
    my_hyper_params[hyper_search_name] = grid_search.best_params_[hyper_search_name]
    
    """
    @@my_hyper_params: my_hyper_params appends best param find to @hyper_search_name
    @@data_frame_results: dataframe with statistics of gridsearch: time, score and ranking
    @@grid_search: grid serach object if it's necessary
    """
    return my_hyper_params, data_frame_results, grid_search
best_hyper_params = {'random_state': 42, 'n_jobs': -1} # Stater Hyper Params
search_hyper = {'penalty': ['l1', 'l2', 'elasticnet', 'none']}

best_hyper_params, results, last_grid_search = optimize_logistic_r(
    x_train, y_train, best_hyper_params, search_hyper, 'penalty')
search_hyper = {'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0 , 4.0, 8.0, 16.0, 32.0, 64.0]}

best_hyper_params, results, last_grid_search = optimize_logistic_r(
    x_train, y_train, best_hyper_params, search_hyper, 'C')
search_hyper = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

best_hyper_params, results, last_grid_search = optimize_logistic_r(
    x_train, y_train, best_hyper_params, search_hyper, 'solver')
# last_grid_search

y_pred = all_classifiers['LogisiticR'].predict(x_test)
print(accuracy_score(y_test, y_pred))
class_report(y_test, y_pred, name="LogisiticR")

y_pred = last_grid_search.predict(x_test)
print(accuracy_score(y_test, y_pred))
class_report(y_test, y_pred, name="LogisiticR0")
# all_classifiers['LogisiticR'].get_params()
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from mlens.ensemble import SuperLearner
 
# create a list of base-models
def get_models():
    models = list()
    models.append(LogisticRegression(**best_hyper_params))
    models.append(DecisionTreeClassifier())
    models.append(XGBClassifier())
    models.append(AdaBoostClassifier())
    models.append(CatBoostClassifier(verbose=0))
    models.append(RandomForestClassifier())
    models.append(LGBMClassifier())
    return models
 
# create the super learner
def get_super_learner(X):
    ensemble = SuperLearner(scorer=accuracy_score, folds=5, shuffle=True, sample_size=len(X), verbose=0)
    # add base models
    models = get_models()
    ensemble.add(models)
    # add the meta model
    ensemble.add_meta(LogisticRegression(**best_hyper_params))
    return ensemble
import time
t0 = time.time()

# create the super learner
ensemble = get_super_learner(x_train.values)

# fit the super learner
ensemble.fit(x_train.values, y_train)

# summarize base learners
print(ensemble.data)

# make predictions on hold out set
y_pred = ensemble.predict(x_test.values)

print("took ", time_spent(t0))
class_report(y_test, y_pred, name="SuperLeaner")

# y_probs = ensemble.predict_proba(x_test.values)

# roc_auc_score(y_test, y_probs)
y_pred = all_classifiers['LogisiticR'].predict(x_test)
print(accuracy_score(y_test, y_pred))
class_report(y_test, y_pred, name="LogisiticR")
!pip install pycaret
from pycaret.classification import *
df_pycaret = df.copy().drop(['customerID'],axis=1)

df_pycaret['Churn'] = df_pycaret['Churn'].replace(yes_no)
df_pycaret['SeniorCitizen'] = df_pycaret['SeniorCitizen'].replace({0: 'No', 1: 'Yes'})
df_pycaret.head(1)
categorical_features = [f for f in df_pycaret.columns if df_pycaret.dtypes[f] == 'object']
# categorical_features
# from sklearn.model_selection import train_test_split

# X = df1.drop(['Churn'], axis=1)

# y = df1['Churn']

# x_train, x_test, y_train, y_test = train_test_split(X, y.values, test_size=0.20, random_state=42)

# # https://pycaret.org/classification/
df_pycaret_setup = setup(data = df_pycaret,
                         target = 'Churn',
                         numeric_imputation = 'mean',
                         categorical_features = categorical_features, 
                         train_size = 0.80,
                         session_id = 42,
                         silent = True)
compare_models()
lr_pycaret  = create_model('lr')     
plot_model(estimator = lr_pycaret, plot = 'learning')
plot_model(estimator = lr_pycaret, plot = 'auc')
plot_model(estimator = lr_pycaret, plot = 'confusion_matrix')
plot_model(estimator = lr_pycaret, plot = 'feature')


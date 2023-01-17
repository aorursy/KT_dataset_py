# Data manipulation

import pandas as pd

import numpy as np



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Set a few plotting defaults

%matplotlib inline

plt.style.use('fivethirtyeight')

plt.rcParams['font.size'] = 18

plt.rcParams['patch.edgecolor'] = 'k'
pd.options.display.max_columns = 150



#Read in data

train = pd.read_csv('../input/costa-rican-household-poverty-prediction/train.csv')

test = pd.read_csv('../input/costa-rican-household-poverty-prediction/test.csv')

train.head()
train.info()



#130 개의 정수 열, 8 개의 부동 (숫자) 열 및 5 개의 개체 열이 있음

#정수 열은 부울 변수 (0 또는 1을 사용), 불연속 순서 값을 가진 서수 변수를 나타낸다.

#객체 열은 머신 러닝 모델로 직접 공급 될 수 없기 때문에 문제가 될 수도 있다



#열보다 행이 더 많은 test를 살펴보자.
test.info()

# 타겟이 없기 때문에 열은 하나가 적음 (train int130개 test int129개)



#정수 열

#정수 열에서 고유 한 값의 분포를 살펴보고

#각 열에 대해 고유 값의 수를 세고 결과를 막대 그래프로 확인
train.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color = 'blue',

                                                                                 figsize = (8, 6),                                                                          

                                                                                 edgecolor = 'k', linewidth = 2);



plt.xlabel('Number of Unique Values');plt.ylabel('Count');

plt.title('Count of Unique Values in Integer Columns');



#정수열의 고유한 값의 분포를 살펴본 결과
from collections import OrderedDict



plt.figure(figsize = (20, 16))

plt.style.use('fivethirtyeight')



#Color mapping

colors = OrderedDict({1: 'red', 2:'orange', 3:'blue', 4:'green'})

poverty_mapping = OrderedDict({1:'extreme', 2:'moderate', 3:'vulnerable', 4:'non vulnerable'})



#Interate through the float columns

for i, col in enumerate(train.select_dtypes('float')):

    ax = plt.subplot(4, 2, i+1)

    #Interate throught the poverty levels

    for poverty_level, color in colors.items():

        #plot each poverty level as a separate line

        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(),

                   ax = ax,color = color, label = poverty_mapping[poverty_level])

        

        plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}');plt.ylabel('Desity')

        

        plt.subplots_adjust(top = 2)

        

        #v2al = mothly rent payment 월세 지불

        #b18q1 = number of tablets household owns 세대가 소유한 태블릿 수

        #rez_esc, Years behind in school 학교 몇년뒤에 가냐

        

        #overcrowding = persons per room  방 당 사람

        #SQBovercrowding = overcrowding squared 

      

        

        #dependency = dependency, Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/

                                                #(number of member of household between 19 and 64)

            

            #의존성, 의존성 비율, 계산 된 = (19 세 이하 또는 64 세 이상 가구 구성원 수) / (19-64 세 가구 구성원 수)

        #SQBdependency = dependency squared

        #SQBmeaned = square of the mean years of education of adults (>=18) in the household

        

train.select_dtypes('object').head()
#dependency 오류남 왜인지는 모름



mapping = {"yes" : 1, "no" : 0}



#Apply the same operation to both train and test

for df in [train, test]:

    #Fill in the values with the correct mapping

    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)

    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)

    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)

    

    train[['dependency', 'edjefa', 'edjefe']].describe()

    
plt.figure(figsize = (16, 12))



# Iterate through the float columns

for i, col in enumerate(['dependency', 'edjefa', 'edjefe']):

    ax = plt.subplot(3, 1, i + 1)

    # Iterate through the poverty levels

    for poverty_level, color in colors.items():

        # Plot each poverty level as a separate line

        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(), 

                    ax = ax, color = color, label = poverty_mapping[poverty_level])

        

    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')



plt.subplots_adjust(top = 2)



#dependency, Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)

#edjefe, years of education of male head of household,based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0

#edjefa, years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0





#이런 단계를 거치면 변수는 이제 숫자로 올바르게 표시되며 기계 학습 모델에 제공 될 수 있음



#위와 같은 작업을 좀 더 쉽게하기 위해 교육 및 테스트 데이터 프레임을 결합한다.

#기능 엔지니어링을 시작한 후에는 두 데이터 프레임에 동일한 작업을 적용하여

#동일한 기능을 사용하기 때문에 중요하다. 나중에 Target을 기준으로 세트를 분리 할 수 있다.
# Add null Target column to test

test['Target'] = np.nan

data = train.append(test, ignore_index = True)
# Heads of household

heads = data.loc[data['parentesco1'] == 1].copy()



# Labels for training

train_labels = data.loc[(data['Target'].notnull()) & (data['parentesco1'] == 1), ['Target', 'idhogar']]



# Value counts of target

label_counts = train_labels['Target'].value_counts().sort_index()



# Bar plot of occurrences of each label

label_counts.plot.bar(figsize = (8, 6), 

                      color = colors.values(),

                      edgecolor = 'k', linewidth = 2)



# Formatting

plt.xlabel('Poverty Level'); plt.ylabel('Count'); 

plt.xticks([x - 1 for x in poverty_mapping.keys()], 

           list(poverty_mapping.values()), rotation = 60)

plt.title('Poverty Level Breakdown');



label_counts
# Groupby the household and figure out the number of unique values

all_equal = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)



# Households where targets are not all equal

not_equal = all_equal[all_equal != True]

print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))
train[train['idhogar'] == not_equal.index[0]][['idhogar', 'parentesco1', 'Target']]
households_leader = train.groupby('idhogar')['parentesco1'].sum()



# Find households without a head

households_no_head = train.loc[train['idhogar'].isin(households_leader[households_leader == 0].index), :]



print('There are {} households without a head.'.format(households_no_head['idhogar'].nunique()))
# Find households without a head and where labels are different

#head가 없는 세대주를 찾고 어느 라벨이 다른지 찾기

households_no_head_equal = households_no_head.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

print('{} Households with no head have different labels.'.format(sum(households_no_head_equal == False)))
# Iterate through each household

for household in not_equal.index:

    # Find the correct label (for the head of household)

    true_target = int(train[(train['idhogar'] == household) & (train['parentesco1'] == 1.0)]['Target'])

    

    # Set the correct label for all members in the household

    train.loc[train['idhogar'] == household, 'Target'] = true_target

    

    

# Groupby the household and figure out the number of unique values

all_equal = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)



# Households where targets are not all equal

not_equal = all_equal[all_equal != True]

print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))
# Number of missing in each column

missing = pd.DataFrame(data.isnull().sum()).rename(columns = {0: 'total'})



# Create a percentage missing

missing['percent'] = missing['total'] / len(data)



missing.sort_values('percent', ascending = False).head(10).drop('Target')
def plot_value_counts(df, col, heads_only = False):

    """Plot value counts of a column, optionally with only the heads of a household"""

    # Select heads of household

    if heads_only:

        df = df.loc[df['parentesco1'] == 1].copy()

        

    plt.figure(figsize = (8, 6))

    df[col].value_counts().sort_index().plot.bar(color = 'blue',

                                                 edgecolor = 'k',

                                                 linewidth = 2)

    plt.xlabel(f'{col}'); plt.title(f'{col} Value Counts'); plt.ylabel('Count')

    plt.show();
plot_value_counts(heads, 'v18q1')
heads.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())
data['v18q1'] = data['v18q1'].fillna(0)
# Variables indicating home ownership

own_variables = [x for x in data if x.startswith('tipo')]





# Plot of the home ownership variables for home missing rent payments

data.loc[data['v2a1'].isnull(), own_variables].sum().plot.bar(figsize = (10, 8),

                                                                        color = 'green',

                                                              edgecolor = 'k', linewidth = 2);

plt.xticks([0, 1, 2, 3, 4],

           ['Owns and Paid Off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'],

          rotation = 60)

plt.title('Home Ownership Status for Households Missing Rent Payments', size = 18);
# Fill in households that own the house with 0 rent payment

data.loc[(data['tipovivi1'] == 1), 'v2a1'] = 0



# Create missing rent payment column

data['v2a1-missing'] = data['v2a1'].isnull()



data['v2a1-missing'].value_counts()
data.loc[data['rez_esc'].notnull()]['age'].describe()
# If individual is over 19 or younger than 7 and missing years behind, set it to 0

data.loc[((data['age'] > 19) | (data['age'] < 7)) & (data['rez_esc'].isnull()), 'rez_esc'] = 0



# Add a flag for those between 7 and 19 with a missing value

data['rez_esc-missing'] = data['rez_esc'].isnull()
data.loc[data['rez_esc'] > 5, 'rez_esc'] = 5
def plot_categoricals(x, y, data, annotate = True):

    """Plot counts of two categoricals.

    Size is raw count for each grouping.

    Percentages are for a given value of y."""

    

    # Raw counts 

    raw_counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize = False))

    raw_counts = raw_counts.rename(columns = {x: 'raw_count'})

    

    # Calculate counts for each group of x and y

    counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize = True))

    

    # Rename the column and reset the index

    counts = counts.rename(columns = {x: 'normalized_count'}).reset_index()

    counts['percent'] = 100 * counts['normalized_count']

    

    # Add the raw count

    counts['raw_count'] = list(raw_counts['raw_count'])

    

    plt.figure(figsize = (14, 10))

    # Scatter plot sized by percent

    plt.scatter(counts[x], counts[y], edgecolor = 'k', color = 'lightgreen',

                s = 100 * np.sqrt(counts['raw_count']), marker = 'o',

                alpha = 0.6, linewidth = 1.5)

    

    if annotate:

        # Annotate the plot with text

        for i, row in counts.iterrows():

            # Put text with appropriate offsets

            plt.annotate(xy = (row[x] - (1 / counts[x].nunique()), 

                               row[y] - (0.15 / counts[y].nunique())),

                         color = 'navy',

                         s = f"{round(row['percent'], 1)}%")

        

    # Set tick marks

    plt.yticks(counts[y].unique())

    plt.xticks(counts[x].unique())

    

    # Transform min and max to evenly space in square root domain

    sqr_min = int(np.sqrt(raw_counts['raw_count'].min()))

    sqr_max = int(np.sqrt(raw_counts['raw_count'].max()))

    

    # 5 sizes for legend

    msizes = list(range(sqr_min, sqr_max,

                        int(( sqr_max - sqr_min) / 5)))

    markers = []

    

    # Markers for legend

    for size in msizes:

        markers.append(plt.scatter([], [], s = 100 * size, 

                                   label = f'{int(round(np.square(size) / 100) * 100)}', 

                                   color = 'lightgreen',

                                   alpha = 0.6, edgecolor = 'k', linewidth = 1.5))

        

    # Legend and formatting

    plt.legend(handles = markers, title = 'Counts',

               labelspacing = 3, handletextpad = 2,

               fontsize = 16,

               loc = (1.10, 0.19))

    

    plt.annotate(f'* Size represents raw count while % is for a given y value.',

                 xy = (0, 1), xycoords = 'figure points', size = 10)

    

    # Adjust axes limits

    plt.xlim((counts[x].min() - (6 / counts[x].nunique()), 

              counts[x].max() + (6 / counts[x].nunique())))

    plt.ylim((counts[y].min() - (4 / counts[y].nunique()), 

              counts[y].max() + (4 / counts[y].nunique())))

    plt.grid(None)

    plt.xlabel(f"{x}"); plt.ylabel(f"{y}"); plt.title(f"{y} vs {x}");
plot_categoricals('rez_esc', 'Target', data);
plot_categoricals('escolari', 'Target', data, annotate = False)
plot_value_counts(data[(data['rez_esc-missing'] == 1)], 

                  'Target')
plot_value_counts(data[(data['v2a1-missing'] == 1)], 

                  'Target')
# Model imports

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK

from hyperopt.pyll.stochastic import sample

import csv

import ast

from timeit import default_timer as timer
def objective(hyperparameters, nfolds=5):

    """Return validation score from hyperparameters for LightGBM"""

    

    # Keep track of evals

    global ITERATION

    ITERATION += 1

    

    # Retrieve the subsample

    subsample = hyperparameters['boosting_type'].get('subsample', 1.0)

    subsample_freq = hyperparameters['boosting_type'].get('subsample_freq', 0)

    

    boosting_type = hyperparameters['boosting_type']['boosting_type']

    

    if boosting_type == 'dart':

        hyperparameters['drop_rate'] = hyperparameters['boosting_type']['drop_rate']

    

    # Subsample and subsample frequency to top level keys

    hyperparameters['subsample'] = subsample

    hyperparameters['subsample_freq'] = subsample_freq

    hyperparameters['boosting_type'] = boosting_type

    

    # Whether or not to use limit maximum depth

    if not hyperparameters['limit_max_depth']:

        hyperparameters['max_depth'] = -1

    

    # Make sure parameters that need to be integers are integers

    for parameter_name in ['max_depth', 'num_leaves', 'subsample_for_bin', 

                           'min_child_samples', 'subsample_freq']:

        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])



    if 'n_estimators' in hyperparameters:

        del hyperparameters['n_estimators']

    

    # Using stratified kfold cross validation

    strkfold = StratifiedKFold(n_splits = nfolds, shuffle = True)

    

    # Convert to arrays for indexing

    features = np.array(train_selected)

    labels = np.array(train_labels).reshape((-1 ))

    

    valid_scores = []

    best_estimators = []

    run_times = []

    

    model = lgb.LGBMClassifier(**hyperparameters, class_weight = 'balanced',

                               n_jobs=-1, metric = 'None',

                               n_estimators=10000)

    

    # Iterate through the folds

    for i, (train_indices, valid_indices) in enumerate(strkfold.split(features, labels)):

        

        # Training and validation data

        X_train = features[train_indices]

        X_valid = features[valid_indices]

        y_train = labels[train_indices]

        y_valid = labels[valid_indices]

        

        start = timer()

        # Train with early stopping

        model.fit(X_train, y_train, early_stopping_rounds = 100, 

                  eval_metric = macro_f1_score, 

                  eval_set = [(X_train, y_train), (X_valid, y_valid)],

                  eval_names = ['train', 'valid'],

                  verbose = 400)

        end = timer()

        # Record the validation fold score

        valid_scores.append(model.best_score_['valid']['macro_f1'])

        best_estimators.append(model.best_iteration_)

        

        run_times.append(end - start)

    

    score = np.mean(valid_scores)

    score_std = np.std(valid_scores)

    loss = 1 - score

    

    run_time = np.mean(run_times)

    run_time_std = np.std(run_times)

    

    estimators = int(np.mean(best_estimators))

    hyperparameters['n_estimators'] = estimators

    

    # Write to the csv file ('a' means append)

    of_connection = open(OUT_FILE, 'a')

    writer = csv.writer(of_connection)

    writer.writerow([loss, hyperparameters, ITERATION, run_time, score, score_std])

    of_connection.close()

    

    # Display progress

    if ITERATION % PROGRESS == 0:

        display(f'Iteration: {ITERATION}, Current Score: {round(score, 4)}.')

    

    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,

            'time': run_time, 'time_std': run_time_std, 'status': STATUS_OK, 

            'score': score, 'score_std': score_std}
# Define the search space

space = {

    'boosting_type': hp.choice('boosting_type', 

                              [{'boosting_type': 'gbdt', 

                                'subsample': hp.uniform('gdbt_subsample', 0.5, 1),

                                'subsample_freq': hp.quniform('gbdt_subsample_freq', 1, 10, 1)}, 

                               {'boosting_type': 'dart', 

                                 'subsample': hp.uniform('dart_subsample', 0.5, 1),

                                 'subsample_freq': hp.quniform('dart_subsample_freq', 1, 10, 1),

                                 'drop_rate': hp.uniform('dart_drop_rate', 0.1, 0.5)},

                                {'boosting_type': 'goss',

                                 'subsample': 1.0,

                                 'subsample_freq': 0}]),

    'limit_max_depth': hp.choice('limit_max_depth', [True, False]),

    'max_depth': hp.quniform('max_depth', 1, 40, 1),

    'num_leaves': hp.quniform('num_leaves', 3, 50, 1),

    'learning_rate': hp.loguniform('learning_rate', 

                                   np.log(0.025), 

                                   np.log(0.25)),

    'subsample_for_bin': hp.quniform('subsample_for_bin', 2000, 100000, 2000),

    'min_child_samples': hp.quniform('min_child_samples', 5, 80, 5),

    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),

    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),

    'colsample_bytree': hp.uniform('colsample_by_tree', 0.5, 1.0)

}

sample(space)
algo = tpe.suggest
#여기부터 내가 뛰어넘어서 오류 및 제대로 결과값이 나오지 않음. 본문 보면서 설명하기

# Record results

trials = Trials()



# Create a file and open a connection

OUT_FILE = 'optimization.csv'

of_connection = open(OUT_FILE, 'w')

writer = csv.writer(of_connection)



MAX_EVALS = 100

PROGRESS = 10

N_FOLDS = 5

ITERATION = 0



# Write column names

headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score', 'std']

writer.writerow(headers)

of_connection.close()
%%capture --no-display

display("Running Optimization for {} Trials.".format(MAX_EVALS))



# Run optimization

best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,

            max_evals = MAX_EVALS)
import json



# Save the trial results

with open('trials.json', 'w') as f:

    f.write(json.dumps(str(trials)))
results = pd.read_csv(OUT_FILE).sort_values('loss', ascending = True).reset_index()

results.head()
plt.figure(figsize = (8, 6))

sns.regplot('iteration', 'score', data = results);

plt.title("Optimization Scores");

plt.xticks(list(range(1, results['iteration'].max() + 1, 3)));

import numpy as np               # algebra linear

import pandas as pd              # operadores de dados pandas

import matplotlib.pyplot as plt  # graphics manipulation

import os                        # para visualidacao dos arquivos e caminhos

import seaborn as sns            # biblioteca de vizualisacao de dados



%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
test_data = pd.read_csv("../input/adult-pmr3508/test_data.csv", na_values = '?')

#test_data.set_index("Id", inplace = True)



train_data = pd.read_csv("../input/adult-pmr3508/train_data.csv", na_values = '?')

train_data.set_index("Id", inplace = True)
display(train_data.head())

display(train_data.describe())
display(test_data.head())

display(test_data.describe())
print("train shape: {}".format(train_data.shape))

print("test shape:  {}".format(test_data.shape))
def rename_dot_2_underline(df):

    df.rename(columns={'education.num':'education_num'}, inplace=True)

    df.rename(columns={'marital.status':'marital_status'}, inplace=True)

    df.rename(columns={'capital.gain':'capital_gain'}, inplace=True)

    df.rename(columns={'capital.loss':'capital_loss'}, inplace=True)

    df.rename(columns={ 'hours.per.week':'hours_per_week'}, inplace=True)

    df.rename(columns={ 'native.country':'native_country'}, inplace=True)

    return(df)
train_data = rename_dot_2_underline(train_data)

test_data = rename_dot_2_underline(test_data) 

train_data.head()
train_data['occupation'].isnull().drop_duplicates().head()
def count_nulls(series, column):

    isn = (series.isnull().sum())

    nn =  (series.notnull().sum())

    print(column)

    print ("Null = {}, Not null = {}, Total = {}, %null = {}".format(isn, nn, isn+nn, isn/(isn+nn) * 100))
for column, data in train_data.iteritems():

    count_nulls(data, column)
def fill_missing_vals (df):

    fill = df['workclass'].describe().top

    df['workclass'] = df['workclass'].fillna(fill)



    fill = df['native_country'].describe().top

    df['native_country'] = df['native_country'].fillna(fill)



    fill = df['occupation'].describe().top

    df['occupation'] = df['occupation'].fillna(fill)

    

    return (df)
#train_data = train_data.dropna()

train_data = fill_missing_vals(train_data)
for column, data in train_data.iteritems():

    count_nulls(data, column)
train_data['income'].drop_duplicates()
#remap = {"<=50K": 0,

#        ">50K" : 1}



#train_data["is_greater_50k"] = train_data.income.map(remap)



#display(train_data.head())
cols = ['age', 'fnlwgt', 'education_num', 'hours_per_week']



sns.pairplot(train_data, vars = cols, hue = 'income')
test_data.columns
train_data.groupby('age').count()
# gera um grafico de barras utilizando a coluna 'analysis_col' sendo o diferenciador o 'hue_col' 

def hue_separated_count_histogram(dataframe, analysis_col, hue_col, skip = 0, sort = None):

    

    # o skip deve pular para ambas as categoerias (>50K e <=50K)

    skip = 2*skip

    columns = [analysis_col, hue_col]

    temp = []

    

    grouped_df = dataframe[columns].groupby(columns)

    

    for key,item in grouped_df:

        a_group = grouped_df.get_group(key).count()

        temp.append([key[0], a_group[0], key[1]])

        #print("[{},{},{}]".format(key[0], a_group[0], key[1]))

        

        

    df = pd.DataFrame(temp, columns = [columns[0], 'count', columns[1]])

    if (sort != None):

        df = df.sort_values(sort)

    

    

    plt.figure(figsize=(17,6))

    fig = sns.barplot(x = df[columns[0]][skip:], y = df['count'][skip:], hue = df[columns[1]][skip:], dodge=False)

    plt.xticks(rotation='vertical')

    plt.show()

    

    

# calcula as porcentagens em um de cada categoria dentro da serie column

def show_percentages(column):

    temp = []

    base_query = "{} == ".format(column)

    base_query = base_query + "'{}'"

    for classification in train_data[column].unique():

        less = train_data.query(base_query.format(classification)).query("income == '<=50K'").shape

        more = train_data.query(base_query.format(classification)).query("income == '>50K'").shape

        total = train_data.shape

        temp.append([classification, 100*more[0]/(less[0]+more[0]), 100*(less[0]+more[0])/(total[0])])

        #print("{}: {:.2f}%".format(classification, 100*more[0]/(less[0]+more[0])))

    df = pd.DataFrame(temp, columns = [column, '%50K+', '%total'])

    return(df)
hue_separated_count_histogram(train_data, 'age', 'income')
hue_separated_count_histogram(train_data, 'workclass', 'income')
hue_separated_count_histogram(train_data.query("workclass != 'Private'"), 'workclass', 'income')
df = show_percentages('workclass')

display(df.sort_values('%50K+', ascending = False))
hue_separated_count_histogram(train_data, 'education', 'income')
df = show_percentages('education')

display(df.sort_values('%50K+', ascending = False))
train_data[['education', 'education_num']].drop_duplicates().sort_values('education_num')
hue_separated_count_histogram(train_data, 'marital_status', 'income')
df = show_percentages("marital_status")

display(df.sort_values('%50K+', ascending = False))
hue_separated_count_histogram(train_data, 'occupation', 'income')
df = show_percentages("occupation")

display(df.sort_values('%50K+', ascending = False))
hue_separated_count_histogram(train_data, 'relationship', 'income')
df = show_percentages("relationship")

display(df.sort_values('%50K+', ascending = False))
hue_separated_count_histogram(train_data, 'race', 'income')
hue_separated_count_histogram(train_data.query("race != 'White'"), 'race', 'income')
df = show_percentages("race")

display(df.sort_values('%50K+', ascending = False))
hue_separated_count_histogram(train_data, 'sex', 'income')
df = show_percentages("sex")

display(df.sort_values('%50K+', ascending = False))
hue_separated_count_histogram(train_data, 'capital_gain', 'income', 1)
hue_separated_count_histogram(train_data, 'capital_loss', 'income', 1)
hue_separated_count_histogram(train_data, 'hours_per_week', 'income')
hue_separated_count_histogram(train_data, 'native_country', 'income')
hue_separated_count_histogram(train_data.query("native_country != 'United-States'"), 'native_country', 'income')
df = show_percentages("native_country")

display(df.sort_values('%50K+', ascending = False).head(10))

display(df.sort_values('%total', ascending = False).head(10))
new_train_df = train_data.drop(['fnlwgt', 'marital_status', 'relationship', 'race', 'education'] , axis = 1)
new_train_df.head()
train_data['workclass'].unique()
remap = {'Private': 1,

        'Local-gov': 0,

        'Self-emp-inc' : 0,

        'State-gov' : 0,

        'Self-emp-not-inc' : 0,

        'Federal-gov' : 0, 

        'Without-pay': 0,

        'Never-worked' : 0}





new_train_df['workclass'] = train_data['workclass'].map(remap)



display(new_train_df.head())
train_data['occupation'].unique()
remap = {'Exec-managerial' : 0, 

         'Transport-moving' : 1, 

         'Machine-op-inspct' : 2,

         'Adm-clerical' : 3, 

         'Other-service' : 4, 

         'Sales' : 5, 

         'Handlers-cleaners' : 6,

         'Craft-repair' : 7, 

         'Tech-support' : 8, 

         'Prof-specialty' : 9,

         'Priv-house-serv' : 10, 

         'Farming-fishing' : 11, 

         'Protective-serv': 12,

         'Armed-Forces' : 13}



new_train_df['occupation'] = train_data['occupation'].map(remap)



display(new_train_df.head())
train_data['sex'].unique()
remap = {'Male':0,

        'Female':1}



new_train_df['sex'] = train_data['sex'].map(remap)



display(new_train_df.head())
train_data['native_country'].unique()
remap = {'United-States':1,

         'Hungary':0, 

         'Jamaica':0, 

         'Mexico':0,

         'Guatemala':0,

         'El-Salvador':0, 

         'Ireland':0, 

         'Haiti':0,

         'Philippines':0, 

         'Cuba':0,

         'England':0, 

         'Iran':0, 

         'Puerto-Rico':0, 

         'South':0, 

         'France':0, 

         'Portugal':0,

         'Canada':0, 

         'Scotland':0, 

         'India':0, 

         'Italy':0, 

         'Dominican-Republic':0,

         'Taiwan':0,

         'Germany':0, 

         'Nicaragua':0,

         'Ecuador':0,

         'Columbia':0, 

         'Vietnam':0,

         'Peru':0,

         'Greece':0, 

         'Hong':0, 

         'Japan':0, 

         'Yugoslavia':0,

         'Cambodia':0,

         'China':0,

         'Poland':0, 

         'Honduras':0,

         'Trinadad&Tobago':0,

         'Thailand':0,

         'Laos':0,

         'Outlying-US(Guam-USVI-etc)':0,

         'Holand-Netherlands':0}



new_train_df['native_country'] = train_data['native_country'].map(remap)



display(new_train_df.head())
def remaps(df):

    new_df = df.drop(['fnlwgt', 'marital_status', 'relationship', 'race', 'education'] , axis = 1)

    

    remap = {'Private': 1,

             'Local-gov': 0,

             'Self-emp-inc' : 0,

             'State-gov' : 0,

             'Self-emp-not-inc' : 0,

             'Federal-gov' : 0, 

             'Without-pay': 0,

             'Never-worked' : 0}



    new_df['workclass'] = df['workclass'].map(remap)

    

    

    remap = {'Exec-managerial' : 0, 

             'Transport-moving' : 1, 

             'Machine-op-inspct' : 2,

             'Adm-clerical' : 3, 

             'Other-service' : 4, 

             'Sales' : 5, 

             'Handlers-cleaners' : 6,

             'Craft-repair' : 7, 

             'Tech-support' : 8, 

             'Prof-specialty' : 9,

             'Priv-house-serv' : 10, 

             'Farming-fishing' : 11, 

             'Protective-serv': 12,

             'Armed-Forces' : 13}



    new_df['occupation'] = df['occupation'].map(remap)

    

    

    remap = {'Male':0,

             'Female':1}



    new_df['sex'] = df['sex'].map(remap)

    

    

    remap = {'United-States':1,

             'Hungary':0, 

             'Jamaica':0, 

             'Mexico':0,

             'Guatemala':0,

             'El-Salvador':0, 

             'Ireland':0, 

             'Haiti':0,

             'Philippines':0, 

             'Cuba':0,

             'England':0, 

             'Iran':0, 

             'Puerto-Rico':0, 

             'South':0, 

             'France':0, 

             'Portugal':0,

             'Canada':0, 

             'Scotland':0, 

             'India':0, 

             'Italy':0, 

             'Dominican-Republic':0,

             'Taiwan':0,

             'Germany':0, 

             'Nicaragua':0,

             'Ecuador':0,

             'Columbia':0, 

             'Vietnam':0,

             'Peru':0,

             'Greece':0, 

             'Hong':0, 

             'Japan':0, 

             'Yugoslavia':0,

             'Cambodia':0,

             'China':0,

             'Poland':0, 

             'Honduras':0,

             'Trinadad&Tobago':0,

             'Thailand':0,

             'Laos':0,

             'Outlying-US(Guam-USVI-etc)':0,

             'Holand-Netherlands':0}



    new_df['native_country'] = df['native_country'].map(remap)

    

    return(new_df)
x_raw_def = remaps(train_data)
from sklearn.preprocessing import StandardScaler
new_train_df.head()
used_columns = ['age', 'workclass', 'education_num', 'occupation', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']

x_raw = new_train_df[used_columns]

#[["age", "workclass",	"fnlwgt", "education", "education.num", "marital.status", "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss", "hours.per.week"]]

y = new_train_df.income
scaler = StandardScaler()

scaler.fit(x_raw)

x_scaled = scaler.transform(x_raw)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate
def print_results(results):

  media = results['test_score'].mean()

  desvio_padrao = results['test_score'].std()

  print("Avg accuracy: %.2f" % (media * 100))

  print("Accuracy range: [%.2f, %.2f]" % ((media - 2 * desvio_padrao)*100, (media + 2 * desvio_padrao) * 100))
def kfold_crossvalidade(splits, neighbors, x, y, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None):

    SEED = 301

    np.random.seed(SEED)

    

    #Kfold model

    model = KNeighborsClassifier(n_neighbors=neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs)



    # cross validate

    cv = KFold(n_splits = splits)

    results = cross_validate(model, x, y, cv = cv, return_train_score=False, return_estimator = True)

    #print(results["estimator"])

    

    avg = results['test_score'].mean()

    std = results['test_score'].std()

    

    #print_results(results)

    

    return (avg)
def run_knn_test(x, y, K = [20, 31], splits = [5], weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None):

    

    results = []



    rows = []

    columns = []



    enumerate

    for i, kfold_splits in enumerate(splits):



        rows.append("splits = {}".format(kfold_splits))

        columns.append([])

        results.append([])



        for knn_K in K:

            #print("splits for kfold = {}\nK for knn = {}".format(kfold_splits, knn_K) )

            results[i].append(kfold_crossvalidade(kfold_splits, knn_K, x, y, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs))

            columns[i].append("K = {}".format(knn_K))



    df = pd.DataFrame(data=results, columns = columns[0], index = rows)

    return(df)
for column, data in x_raw.iteritems():

    count_nulls(data, column)
df = run_knn_test(x_raw, y, p = 1, K = [19, 20, 21, 30, 31, 32])

df
df = run_knn_test(x_scaled, y)

df
df = run_knn_test(x_raw, y)

df
df = run_knn_test(x_raw, y, p = 1)

df
df = run_knn_test(x_raw, y, p = 2)

df
df = run_knn_test(x_scaled, y, p = 1)

df
df = run_knn_test(x_raw, y, p = 1, weights='distance')

df
df = run_knn_test(x_scaled, y, p = 1, weights='distance')

df
df = run_knn_test(x_raw, y, p = 1, weights='uniform')

df
df = run_knn_test(x_raw, y, p = 1, algorithm = 'ball_tree')

df
df = run_knn_test(x_raw, y, p = 1, algorithm = 'kd_tree')

df
df = run_knn_test(x_raw, y, p = 1, algorithm = 'brute')

df
model = KNeighborsClassifier(n_neighbors=20, p = 1)

model.fit(x_raw, y)
display(pd.DataFrame(model.predict(x_raw_def[used_columns])))
test_data = fill_missing_vals(test_data)
x_raw_test = remaps(test_data)
for column, data in x_raw_test.iteritems():

    count_nulls(data, column)
test_data.head()
prdiction = model.predict(x_raw_test[used_columns])
submission = pd.DataFrame(columns = ["Id","income"])



submission.Id = x_raw_test["Id"]

submission.income = prdiction
submission.head()
submission.to_csv('submission.csv', index = False)
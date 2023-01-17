import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow as tf

import networkx as nx





from scipy import stats

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from lime import lime_tabular

import lime



# classifiers

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



# regressors

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor



%matplotlib inline

sns.set()
column_names = ['Id', 'age', 'workclass', 'final_weight', 'education', 'education_num', 'marital_status',

                'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',

                'native_country', 'income']
data = pd.read_csv('../input/adult-pmr3508/train_data.csv', names = column_names, na_values='?').drop(0, axis = 0).reset_index(drop = True)
data.head()
data.describe()
def count_null_values(data):

    '''

    Return a DataFrame with count of null values

    '''

    

    counts_null = []

    for column in data.columns:

        counts_null.append(data[column].isnull().sum())

    counts_null = np.asarray(counts_null)



    counts_null = pd.DataFrame({'feature': data.columns, 'count.': counts_null,

                                'freq. [%]': 100*counts_null/data.shape[0]}).set_index('feature', drop = True)

    counts_null = counts_null.sort_values(by = 'count.', ascending = False)

    

    return counts_null
count_null_values(data).head()
def work_missing_values(data):

    '''

    Return new data with no missing values for this problem

    '''

    

    aux = data.copy()

    # select index of rows that workclass is nan

    aux_index = aux[aux['workclass'].isna()].index

    

    # fill nan with 'unknown'

    aux['workclass'].loc[aux_index] = 'unknown'

    aux['occupation'].loc[aux_index] = 'unknown'

    

    # complete missing of native_country and occupation with most frequent

    cols = ['native_country', 'occupation']

    for col in cols:

        top = aux[col].value_counts().index[0]

        aux[col] = aux[col].fillna(top)

    aux.reset_index(drop = True)

    

    return aux
data = work_missing_values(data)
count_null_values(data).head()
numeric_columns = ['age', 'final_weight', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

categoric_columns = ['workclass', 'education', 'marital_status', 'occupation',

                     'relationship', 'race', 'sex', 'native_country', 'income']



# change to number variable

for column in numeric_columns:

    data[column] = pd.to_numeric(data[column])
sns.set()

sns.pairplot(data, vars = numeric_columns, hue = 'income', palette = 'Wistia')
def bar_plot(data, by, hue, normalize_by_index = True):

    '''

    Plot count bar for each unique in data[by], using as reference 'hue'

    

    obs: if normalize_by_index is True, than for each index, the values will be normalized

    '''

    

    index = data[by].unique()

    columns = data[hue].unique()

    

    data_to_plot = pd.DataFrame({'index': index})

    

    for column in columns:

        temp = []

        for unique in index:

            filtered_data = data[data[by] == unique]

            filtered_data = filtered_data[filtered_data[hue] == column]

            

            temp.append(filtered_data.shape[0])

        data_to_plot = pd.concat([data_to_plot, pd.DataFrame({column: temp})], axis = 1)

        

    data_to_plot = data_to_plot.set_index('index', drop = True)

    

    if normalize_by_index:

        print('O gráfico está normalizado por index!')

        for row in index:

            data_to_plot.loc[row] = data_to_plot.loc[row].values/data_to_plot.loc[row].values.sum()

    

    ax = data_to_plot.plot.bar(rot=0, figsize = (14,7), alpha = 0.9, cmap = 'Wistia')
bar_plot(data, 'sex', 'income')
bar_plot(data, 'race', 'income')
bar_plot(data, 'race', 'occupation')
bar_plot(data, 'marital_status', 'income')
bar_plot(data, 'workclass', 'income')
def pie_plot(data, by, n = 5):

    '''

    Plot pie with count of recurrence of each unique of data[by]. 

    

    n: is for how much uniques to be shown, remainer will be treat as 'Others'

    '''

    

    data_counts = data[by].value_counts()

    if len(data_counts) < n:

        n = len(data_counts)

    

    all_counts = data_counts[:n]

    

    if n != len(data_counts):

        other_counts = data_counts[n:]

        

        all_counts = pd.concat([all_counts, pd.Series({'Others': other_counts.sum()})])

        del other_counts

        

    all_counts = all_counts.sort_values(ascending = False)

    all_counts.plot(kind = 'pie', cmap = 'Wistia', figsize = (14,7))

    del all_counts
pie_plot(data, 'native_country', n = 3)
pie_plot(data, 'occupation', n = 13)
pie_plot(data, 'education', n = 13)
pie_plot(data, 'workclass', n = 13)
def box_plot(data, var_x, var_y, orientation = 'v', rotate_x_label = False):



    df = pd.concat([data[var_y], data[var_x]], axis=1)

    f, ax = plt.subplots(figsize=(15, 7))



    sns.boxplot(x=var_x, y=var_y, data=df, notch = True, palette = 'Wistia', orient = orientation)

    plt.title('Boxplot of education num over race')

    if rotate_x_label:

        ax.set_xticklabels(data[var_x].unique(), rotation=90)

    
box_plot(data, 'income', 'education_num')
box_plot(data, 'income', 'age')
box_plot(data, 'income', 'hours_per_week')
box_plot(data, 'marital_status', 'hours_per_week', rotate_x_label=True)
box_plot(data, 'marital_status', 'age', rotate_x_label=True)
def findEdges(data, col1, col2):

    '''

    return list of edges that connects col1 and col2

    '''

    

    edges = []

    for unique1 in data[col1].unique():

        temp1 = data[data[col1] == unique1]

        for unique2 in data[col2].unique():

            temp2 = temp1[temp1[col2] == unique2]

            size = len(temp2)

            if size == 0:

                continue

            edges.append((unique1, unique2, size))

    

    del temp1, temp2

    return edges
def setNodePosition(nodes, r = 7, center = 14):

    '''

    return dictionary of nodes with respected position in a circle

    '''

    

    n = len(nodes)

    positions = {}

    for k, node in enumerate(nodes):

        positions[node] = (r*np.cos(k*(2*np.pi)/n) + center, r*np.sin(k*(2*np.pi)/n) + center)

    return positions
def drawPairNetwork(data, var1, var2):



    G = nx.Graph()



    G.add_nodes_from(data[var1].unique())

    positions = setNodePosition(data[var1].unique(), r = 7)



    G.add_nodes_from(data[var2].unique())

    positions.update(setNodePosition(data[var2].unique(), r = 14))



    edges = findEdges(data, var1, var2)

    for edge in edges:

        G.add_edge(edge[0], edge[1], weight = edge[2])

    del edges



    with plt.style.context('seaborn-whitegrid'):

        colors = range(20)

        

        plt.figure(figsize = (14,14))

        nx.draw_networkx_nodes(G, pos = positions, nodelist = data[var1].unique(), node_color = 'yellow')

        nx.draw_networkx_nodes(G, pos = positions, nodelist = data[var2].unique(), node_color = 'coral')

        

        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

        nx.draw_networkx_edges(G, pos = positions, width = 4.0, edgelist = edges, edge_color = weights, edge_cmap = plt.cm.Oranges)

        nx.draw_networkx_labels(G, pos = positions)

        plt.grid(alpha = 0)

    

    return G
G = drawPairNetwork(data, 'race', 'native_country')
G = drawPairNetwork(data, 'sex', 'occupation')
G = drawPairNetwork(data, 'income', 'race')
G = drawPairNetwork(data, 'income', 'occupation')
G = drawPairNetwork(data, 'income', 'marital_status')
G = drawPairNetwork(data, 'income', 'education')
def usa_column_prepare(value):

    '''

    Return 1 if value is United-States, 0 otherwise

    '''

    

    if value == 'United-States':

        return 1

    return 0



def education_column_prepare(value):

    '''

    Return an integer that correspond to education order:

    

    Preschool < 1st-4th < 5th-6th < 7th-8th < 9th < 10th < 11th 

    < 12th < HS-grad < Prof-school < Assoc-acdm < Assoc-voc 

    < Some-college < Bachelors < Masters < Doctorate

    '''

    

    if value == 'Preschool':

        return 0

    if value == '1st-4th':

        return 1

    if value == '5th-6th':

        return 2

    if value == '7th-8th':

        return 3

    if value == '9th':

        return 4

    if value == '10th':

        return 5

    if value == '11th':

        return 6

    if value == '12th':

        return 7

    if value == 'HS-grad':

        return 8

    if value == 'Prof-school':

        return 9

    if value == 'Assoc-acdm':

        return 10

    if value == 'Assoc-voc':

        return 11

    if value == 'Some-college':

        return 12

    if value == 'Bachelors':

        return 13

    if value == 'Masters':

        return 14

    return 15



def married_present_column_prepare(value):

    '''

    Returns 1 if marriege has both partners present to each other.

    '''

    if value == 'Married-civ-spouse' or value == 'Married-AF-spouse':

        return 1

    return 0
def pipe_of_data_prepare(data, drop_columns = None, kind = 'train'):

    '''

    Return numeric prepared data to train models

    '''

    

    columns = data.columns

    

    # remove missing values

    new_data = work_missing_values(data)

    

    # remove Id column

    new_data = new_data.drop('Id', axis = 1)

    

    # add US or not column

    if 'native_country' in columns:

        new_data['usa'] = new_data['native_country'].apply(usa_column_prepare)

        

    # education ordered column:

    if 'education' in columns:

        new_data['education'] = new_data['education'].apply(education_column_prepare)

        

    # select important correspondence in marital_status

    if 'marital_status' in columns:

        new_data['married_present'] = new_data['marital_status'].apply(married_present_column_prepare)

        

    # label encoder for categorical not ordered features    

    categorical_features = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

    encoder = {}

    for feature in categorical_features:

        if feature in columns:

            encoder[feature] = LabelEncoder()

            new_data[feature] = encoder[feature].fit_transform(new_data[feature])

            

    if drop_columns is not None:

        new_data = new_data.drop(drop_columns, axis = 1)

    

    if kind == 'train':

        X, y = new_data.drop('income', axis = 1), new_data['income']

        y = y.values.reshape(-1,1)

    else:

        X = new_data.copy()

    used_columns = X.columns

    

    X = X.values

    

    del new_data

            

    # normalize features with StandardScaler  

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    

    # train data has y

    if kind == 'train':

        return X, y, used_columns, scaler, encoder

    

    # test data

    return X, used_columns, scaler, encoder
data = {}

data['train'] = pd.read_csv('../input/adult-pmr3508/train_data.csv', names = column_names, na_values='?').drop(0, axis = 0).reset_index(drop = True)

data['test'] = pd.read_csv('../input/adult-pmr3508/test_data.csv', names = column_names[:len(column_names)-1], na_values='?').drop(0, axis = 0).reset_index(drop = True)



drop_columns = ['final_weight', 'native_country', 'marital_status', 'education_num']

X_train, y, used_columns_train, scaler_train, encoder_train = pipe_of_data_prepare(data['train'], drop_columns = drop_columns, kind = 'train')

X_test, used_columns_test, scaler_test, encoder_test = pipe_of_data_prepare(data['test'], drop_columns = drop_columns, kind = 'test')



X_train_unscaled = scaler_train.inverse_transform(X_train)

X_test_unscaled = scaler_test.inverse_transform(X_test)



used_columns_train, used_columns_test = list(used_columns_train), list(used_columns_test)
def explanation_fn(estimator, instance):

    '''

    fixed function for lime explanation for estimator and given example instance

    '''

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, training_labels=y, 

                                                   feature_names=used_columns_train, categorical_features = [1,3,4,5,6,12], 

                                                   class_names = ['<=50K', '>50K'])



    exp = explainer.explain_instance(X_test[instance], estimator.predict_proba, num_features=6, top_labels=None)

    exp.show_in_notebook(show_table=True, show_all=False)
def outputPrediction(ids, predictions):

    data = pd.DataFrame({'Id': ids, 'income': predictions})

    return data
%%time

time_train = [2000]



# train



LogClf = LogisticRegression(solver = 'lbfgs', C = 1.0, penalty = 'l2', warm_start =  True)



LogCV = cross_val_score(LogClf, X_train, y.reshape(-1), cv = 10)



LogClf.fit(X_train, y.reshape(-1))



cv_accuracy = [LogCV.mean()]

cv_std = [LogCV.std()]



cv_values = {}

cv_values['Lin'] = LogCV

print('Logistic Regression CV accuracy: {0:1.4f} +-{1:2.5f}\n'.format(LogCV.mean(), LogCV.std()))
%%time

time_test = [5]



# test



predictions = {}

predictions['Log'] = LogClf.predict(X_test)

predictions['Log'] = outputPrediction(data['test']['Id'].values, predictions['Log'])
explanation_fn(LogClf, 6)
explanation_fn(LogClf, 21)
%%time

time_train.append(30500)



# train



KNNClf = KNeighborsClassifier(n_neighbors = 19, p = 1, weights = 'uniform')



KNNCV = cross_val_score(KNNClf, X_train, y.reshape(-1), cv = 10)



KNNClf.fit(X_train, y.reshape(-1))



cv_accuracy.append(KNNCV.mean())

cv_std.append(KNNCV.std())

cv_values['KNN'] = KNNCV

print('K-Nearest Neighboors CV accuracy: {0:1.4f} +-{1:2.5f}\n'.format(KNNCV.mean(), KNNCV.std()))
%%time

time_test.append(9350)



# test



predictions['KNN'] = KNNClf.predict(X_test)

predictions['KNN'] = outputPrediction(data['test']['Id'].values, predictions['KNN'])
explanation_fn(KNNClf, 6)
explanation_fn(KNNClf, 21)
%%time

time_train.append(170000)



# train



RFClf = RandomForestClassifier(n_estimators = 700, max_depth = 12)



RFCV = cross_val_score(RFClf, X_train, y.reshape(-1), cv = 10)



RFClf.fit(X_train, y.reshape(-1))



cv_accuracy.append(RFCV.mean())

cv_std.append(RFCV.std())

cv_values['RF'] = RFCV

print('Random Forest CV accuracy: {0:1.4f} +-{1:2.5f}\n'.format(RFCV.mean(), RFCV.std()))
%%time

time_test.append(1570)



# test



predictions['RF'] = RFClf.predict(X_test)

predictions['RF'] = outputPrediction(data['test']['Id'].values, predictions['RF'])
explanation_fn(RFClf, 6)
explanation_fn(RFClf, 21)
%%time

time_train.append(60000)



# train



XGBClf = XGBClassifier(max_depth = 4, n_estimators = 250)



XGBCV = cross_val_score(XGBClf, X_train, y.reshape(-1), cv = 10)



XGBClf.fit(X_train, y.reshape(-1))



cv_accuracy.append(XGBCV.mean())

cv_std.append(XGBCV.std())

cv_values['XGB'] = XGBCV

print('XGBoost CV accuracy: {0:1.4f} +-{1:2.5f}\n'.format(XGBCV.mean(), XGBCV.std()))
%%time

time_test.append(155)



# test



predictions['XGB'] = XGBClf.predict(X_test)

predictions['XGB'] = outputPrediction(data['test']['Id'].values, predictions['XGB'])
explanation_fn(XGBClf, 6)
explanation_fn(XGBClf, 21)
results = pd.DataFrame()



results['Estimator'] = ['Logistic Regression', 'K-Nearest Neighboors', 'Random Forest', 'XGBoost']

results['CV accuracy'] = cv_accuracy

results['CV std'] = cv_std

results['~Time (Train) [ms]'] = time_train

results['~Time (Test) [ms]'] = time_test



results = results.set_index('Estimator', drop = True)



# setting CV values for visualization



cv_values = pd.DataFrame(cv_values)

cv_values = np.concatenate([cv_values.columns.values.reshape(-1,1), cv_values.values.transpose()], axis = 1)

temp = {}

for u in range(cv_values.shape[0]):

    temp[u] = cv_values[u, :]

cv_values = pd.DataFrame(temp)

cv_values = cv_values.rename(columns = {0:'Logistic Regression', 1:'K-Nearest Neighboors', 2:'Random Forest', 3:'XGBoost'})

cv_values = cv_values.drop(0, axis = 0)
plt.figure(figsize=(14,7))

plt.title('Cross Validation Accuracy', fontsize = 15)

sns.boxplot(data = cv_values, palette = 'Wistia')
plt.figure(figsize=(14,7))

plt.title('~Time Evaluation [ms]', fontsize = 15)

plt.plot(results.index, results['~Time (Train) [ms]'].values, color = 'gray', lw = 2,

         marker = 'o', markersize = 12, markerfacecolor = 'orange', label = 'Train')

plt.plot(results.index, results['~Time (Test) [ms]'].values, color = 'gray', lw = 2,

         marker = 'o', markersize = 12, markerfacecolor = 'yellow', label = 'Test')

plt.legend()
for classifier in predictions.keys():

    predictions[classifier].to_csv(classifier.lower() + '_predict.csv', index = False)
data = pd.read_csv('../input/atividade-3/train.csv', na_values='?').reset_index(drop = True)

data.head()
data.describe()
def plotMap(data, sizes = None, colors = None, cmap = 'Wistia', alpha = 0.7, title = 'Mapa'):

    '''

    plot on cartesian plan, coordinatedes according to lat long, with circle sizes em color scale

    '''

    v_sizes, v_colors = None, None

    if sizes is not None:

        scaler = MinMaxScaler()

        v_sizes = scaler.fit_transform(data[sizes].values.reshape(-1,1))*100

        v_sizes = v_sizes.reshape(-1)

        

    if colors is not None:

        v_colors = data[colors]

        

    with plt.style.context('seaborn-whitegrid'):

        data.plot.scatter('longitude', 'latitude', s = v_sizes, figsize = (11,7), c = v_colors, cmap = cmap, alpha = alpha)

        plt.title(title)
plotMap(data, sizes = 'median_income', colors = 'median_house_value', title = 'Income and house value visualization map')
plotMap(data, sizes = 'population', colors = 'median_age', title = 'Population and ages visualization map', alpha = 0.7)
plt.figure(figsize=(14,7))

sns.distplot(data['median_age'], color = 'Orange', bins = 20)
plt.figure(figsize=(14,7))

sns.distplot(data['median_income'], color = 'Orange', bins = 20)
plt.figure(figsize=(14,7))

sns.distplot(data['population'], color = 'Orange', bins = 30)
plt.figure(figsize=(14,7))

sns.boxplot(x = data['median_age'], y = data['population'], palette = 'Wistia')
plt.figure(figsize=(14,7))

sns.boxplot(x = data['median_age'], y = data['median_income'], palette = 'Wistia')
corrmat = data.drop('Id', axis = 1).corr()

plt.figure(figsize=(12,9))

sns.heatmap(corrmat, cmap = 'Wistia')
#Retirando outliers da base

data_clean = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]



#Reindexando para ajustar termos faltantes

data_clean = data_clean.assign(index = list(range(0, data_clean.iloc[:,0].size)))

data_clean = data_clean.set_index('index')
selected_columns = ['longitude', 'median_income', 'median_age', 'population']

target = 'median_house_value'



selected_base = pd.concat([data_clean[selected_columns], data_clean[target]], axis = 1)
scaler = {}

for col in selected_columns:

    scaler[col] = StandardScaler()

    selected_base[col] = scaler[col].fit_transform(selected_base[col].values.reshape(-1,1))

    

scaler[target] = MinMaxScaler()

selected_base[target] = scaler[target].fit_transform(selected_base[target].values.reshape(-1,1))
X, y = selected_base[selected_columns].values, selected_base[target].values
# metrica para avaliar os regressores



from sklearn.metrics import mean_squared_log_error, make_scorer



msle = make_scorer(mean_squared_log_error)
%%time

time_train = [40]



# train



LinReg = LinearRegression()



LinCV = cross_val_score(LinReg, X, y.reshape(-1), cv = 10, scoring = msle)



LinReg.fit(X, y)



cv_accuracy = [LinCV.mean()]

cv_std = [LinCV.std()]

cv_values = {}

cv_values['Lin'] = LinCV

print('Linear Regression CV msle: {0:1.4f} +-{1:2.5f}\n'.format(LinCV.mean(), LinCV.std()))
%%time

time_train.append(310)



# train



KNNReg = KNeighborsRegressor(n_neighbors=30)



KNNCV = cross_val_score(KNNReg, X, y, cv = 10, scoring = msle)



KNNReg.fit(X, y)



cv_accuracy.append(KNNCV.mean())

cv_std.append(KNNCV.std())

cv_values['KNN'] = KNNCV

print('KNN Regression CV msle: {0:1.4f} +-{1:2.5f}\n'.format(KNNCV.mean(), KNNCV.std()))
%%time

time_train.append(18700)



# train



RFReg = RandomForestRegressor(n_estimators = 50, max_depth = 14)



RFCV = cross_val_score(RFReg, X, y.reshape(-1), cv = 10, scoring = msle)



RFReg.fit(X, y.reshape(-1))



cv_accuracy.append(RFCV.mean())

cv_std.append(RFCV.std())

cv_values['RF'] = RFCV

print('RF Regression CV msle: {0:1.4f} +-{1:2.5f}\n'.format(RFCV.mean(), RFCV.std()))
results = pd.DataFrame()



results['Estimator'] = ['Linear Regression', 'K-Nearest Neighboors Regressor', 'Random Forest Regressor']

results['CV accuracy'] = cv_accuracy

results['CV std'] = cv_std

results['~Time (Train) [ms]'] = [35, 265, 1440]



results = results.set_index('Estimator', drop = True)



# setting CV values for visualization



cv_values = pd.DataFrame(cv_values)

cv_values = np.concatenate([cv_values.columns.values.reshape(-1,1), cv_values.values.transpose()], axis = 1)

temp = {}

for u in range(cv_values.shape[0]):

    temp[u] = cv_values[u, :]

cv_values = pd.DataFrame(temp)

cv_values = cv_values.rename(columns = {0:'Linear Regression', 1:'K-Nearest Neighboors Regressor', 2:'Random Forest Regressor'})

cv_values = cv_values.drop(0, axis = 0)
plt.figure(figsize=(14,7))

plt.title('Cross Validation MSLE', fontsize = 15)

sns.boxplot(data = cv_values, palette = 'Wistia')
plt.figure(figsize=(14,7))

plt.title('~Time Evaluation [ms]', fontsize = 15)

plt.plot(results.index, results['~Time (Train) [ms]'].values, color = 'gray', lw = 2,

         marker = 'o', markersize = 12, markerfacecolor = 'orange', label = 'Train')

plt.legend()
import pandas as pd

import numpy as np



data = pd.read_csv("../input/train.csv").set_index('PassengerId')

data_test = pd.read_csv("../input/test.csv").set_index('PassengerId')



len(data)

for col in data.columns:

    currSet = set(data[col])

    if len(currSet) < 20:

        printSet = currSet

    else:

        printSet = 'too large to print.'

    print(col, 'set:', printSet)
from pandas.tools.plotting import scatter_matrix

plotRelevant = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

_ = scatter_matrix(data[plotRelevant].dropna(), alpha = 0.2, figsize = (10,10))
numNull = [np.sum(data.isnull() == True, axis = 0)]

print('Number of nulls: \n', numNull)
from sklearn.preprocessing import Imputer

import matplotlib.pyplot as plt



# Imputting features on the test set:

def imput_features(df, impCols, strategy = 'median'):

    from sklearn.preprocessing import Imputer

    df = df.copy()

    # Imputting 'Age' and 'Fare' features:

    medImp = Imputer(strategy = strategy)

    df[impCols] = medImp.fit_transform(df[impCols])

    return df

data_test = imput_features(data_test, impCols = ['Age','Fare'])

data = imput_features(data, impCols = ['Age'])



# Converting categorical variables strings and Pclass:

def extract_categ_dummies(df, df_test, categFeatures, prefix = None):

    df = df.copy()

    df_test = df_test.copy()

    dummiesTrain = (

                    pd.get_dummies(df[categFeatures], prefix = prefix)

                    .reindex(df.index)

                    )

    dummiesTest = (

                   pd.get_dummies(df_test[categFeatures], prefix = prefix)

                   .reindex(index = df_test.index,

                            columns = dummiesTrain.columns, fill_value=0)

                   )

    df = (

          pd.concat([data,dummiesTrain], axis = 1)

          .drop(categFeatures, axis = 1)

          )

    df_test = (

               pd.concat([data_test,dummiesTest], axis = 1)

               .drop(categFeatures, axis = 1)

               )

    return df, df_test

data, data_test = extract_categ_dummies(data, data_test, ['Sex','Embarked'])

data, data_test = extract_categ_dummies(data, data_test, 'Pclass',

                                        prefix = 'Pclass')



# Checking numerical Features for outliers:

def check_summary(series):

    s1 = str(series.name) + ' Min: ' + str(round(np.min(series),2))

    s2 = str(series.name) + ' Median: ' + str(round(np.median(series),2))

    s3 = str(series.name) + ' Mean: ' + str(round(np.mean(series),2))

    s4 = str(series.name) + ' Max: ' + str(round(np.max(series),2))

    s = s1 + '\n' + s2 + '\n' + s3 + '\n' + s4

    return s



plotRelevant = ['Age', 'Fare']

fig, axes = plt.subplots(1,2, sharey = True, figsize = (8,4))

for i, ax in enumerate(axes.flatten()):

    histData = data[plotRelevant[i]]

    ax.hist(histData)

    ax.set_title(plotRelevant[i])

    ax.text(np.max(histData)/3,600, check_summary(histData))

fig.tight_layout()
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

import re

# Extracting the numbers from 'Ticket' feature,

# substituting empty values for '0' and converting to int:

def extract_ticket_num(df):

    import re

    df = df.copy()

    reFunc = lambda x: re.search('[0-9]*$',str(x)).group()

    df['Ticket_num'] = (

                        df.Ticket

                        .apply(reFunc)

                        .replace('', '0').apply(int)

                        )

    return df



data = extract_ticket_num(data)

data_test = extract_ticket_num(data_test)



# Creating a feature for examples that contain any value in 'Cabin':  

def extract_cabin(df):

    df = df.copy()

    df['Has_Cabin'] = df.Cabin.notnull() *1

    cabinClean = df.Cabin.dropna().apply(lambda x: str(x).split())

    getStrFunc = lambda x: re.search('[A-Z]*',x[0].upper()).group()

    df['Cabin_str'] = np.nan

    df.loc[cabinClean.index, 'Cabin_str'] = cabinClean.apply(getStrFunc)

    getNumFunc = lambda x: re.search('[0-9]*$',x[0].upper()).group()

    df['Cabin_num'] = np.nan

    df.loc[cabinClean.index, 'Cabin_num'] = cabinClean.apply(getNumFunc)

    df.Cabin_num.replace('', np.nan, inplace = True)

    df.Cabin_num = pd.to_numeric(df.Cabin_num)

    cleanNumIndex = df.Cabin_num.dropna().index

    cleanNumSeries = df.Cabin_num.dropna()

    df['Cabin_even'] = np.nan

    evenFunc = lambda x: (x % 2)

    df.loc[cleanNumIndex, 'Cabin_even'] = cleanNumSeries.apply(evenFunc)

    df.drop('Cabin', axis = 1, inplace = True)

    return df

data = extract_cabin(data)

data_test = extract_cabin(data_test)



# Extracting dummies from 'Cabin_str','Cabin_num' and 'Cabin_even':

data, data_test = extract_categ_dummies(data, data_test, ['Cabin_str',

                                                          'Cabin_num',

                                                          'Cabin_even'])



# Extracting features from Names:

def extract_name(df):

    """

    Function to create 2 binary features. The first is positive if there are 2

    parenthesis in the 'Name' feature, while the second one is positivie if

    there are 2 double quotes in the same feature.

    """

    import re

    df = df.copy()

    funcPar = lambda x: re.search("\(.*\)", str(x))

    parIndex = df.Name.apply(funcPar).dropna().index

    df['Name_par'] = 0

    df.loc[parIndex,'Name_par'] = 1

    funcQuote = lambda x: re.search("\".*\"", str(x))

    quoteIndex = df.Name.apply(funcQuote).dropna().index

    df['Name_quote'] = 0

    df.loc[quoteIndex, 'Name_quote'] = 1

    rmv1 = lambda x: re.sub("\(.*\)", '', str(x))

    rmv2= lambda x: re.sub("\".*\"", '', str(x))

    rmv3 = lambda x: re.sub("[,].*\.\s",' ', str(x))

    rmv = lambda x: rmv3(rmv2(rmv1(x)))

    df.Name = df.Name.apply(rmv)

    df['Name_len'] = df.Name.apply(len)

    return df

data = extract_name(data)

data_test = extract_name(data_test)

data.drop('Name', axis = 1, inplace = True)

data_test.drop('Name', axis = 1, inplace = True)



# Now, cleaning the string left on the 'Ticket' feature:

def extract_ticket_str(df):

    df = df.copy()

    cleanFunc1 = lambda x: re.sub('[0-9]*$','',x)

    cleanFunc2 = lambda x: re.sub('\s$','',x).upper()

    cleanFunc3 = lambda x: re.sub('\.','',x)

    cleanFunc = lambda x:cleanFunc3(cleanFunc2(cleanFunc1(str(x))))

    df.Ticket = df.Ticket.apply(cleanFunc)

    return df

data = extract_ticket_str(data)

data_test = extract_ticket_str(data_test)



# MinMax scaling 'Age','Fare' and 'Ticket_num' feature:

def min_max_features(df, features):

    df = df.copy()

    scaler = MinMaxScaler()

    scaler.fit(df[features])

    df[features] = scaler.transform(df[features])

    return df

toMinMax = ['Fare', 'Age', 'Ticket_num', 'Name_len', 'Parch', 'SibSp']

data = min_max_features(data, features = toMinMax)

data_test = min_max_features(data_test, features = toMinMax)



#Checking the number of different strings on the set:

print(data.Ticket.value_counts())
from sklearn.cluster import KMeans

# From the strings that repeat at least 10 times, we will create new features to represent the presence or absence

# of this strings on each example:

def extract_ticket_groups(df):

    df = df.copy()

    ticketGroups = [['CA'], ['PC'], ['SOTON', 'STON','SO'], ['PARIS'], ['A/']]

    for i, group in enumerate(ticketGroups):

        name = 'Ticket_str_' + str(i+1)

        df[name] = 0

        inGroupFunc = lambda x,y: any([y[i] in x for i in range(len(y))])

        df[name] = df.Ticket.apply(lambda x: inGroupFunc(x,group)) * 1

    df.drop('Ticket', axis = 1, inplace = True)

    return df

data = extract_ticket_groups(data)

data_test = extract_ticket_groups(data_test)



# Adding polynomial features to 'Age' and 'Fare' features:

def add_poly_features(df, polyCols, degree = 3, **kargs):

    poly = PolynomialFeatures(degree = degree, include_bias = False, **kargs)

    df_poly = df.copy()

    poly.fit(df_poly[polyCols])

    names = poly.get_feature_names(polyCols)

    df_poly = pd.DataFrame(poly.transform(df_poly[polyCols]), columns = names,

                           index = df.index)

    df_poly.drop(polyCols, axis = 1, inplace = True)

    df_poly = pd.concat([df,df_poly], axis = 1)

    return df_poly

polyNames = ['Age', 'Fare', 'Ticket_num']

data = add_poly_features(data, polyCols = polyNames)

data_test = add_poly_features(data_test, polyCols = polyNames)



# Adding interaction features from some binomial features:

polyNames2 = ['Age', 'Sex_female', 'Sex_male', 'Pclass_1', 'Pclass_2',

              'Pclass_3', 'Ticket_num', 'Has_Cabin', 'Embarked_C',

              'Embarked_Q', 'Embarked_S']



data = add_poly_features(data, polyCols = polyNames2, degree = 2,

                         interaction_only = True)

data_test = add_poly_features(data_test, polyCols = polyNames2, degree = 2,

                              interaction_only = True)





# Applying K-Means clustering to create a new feture:

def extract_kmeans_feature(df, df_test, n_clusters):

    from sklearn.cluster import KMeans

    df = df.copy()

    df_test = df_test.copy()

    kmeans = KMeans(n_clusters)

    kmeans.fit(df.iloc[:,1:])

    df['Cluster'] = kmeans.labels_

    df_test['Cluster'] = kmeans.predict(df_test)

    return df, df_test



data, data_test = extract_kmeans_feature(data, data_test, n_clusters = 6)



# Converting the cluster feature to dummies:

data, data_test = extract_categ_dummies(data, data_test, ['Cluster'])
# Plotting histograms of the features before starting the analysis:

plotRelevant = list(data.columns[1:19])



fig, axes = plt.subplots(6, 3, figsize = (6,12))

for i,ax in enumerate(axes.flatten()):

    ax.hist(data[plotRelevant[i]])

    ax.set_title(plotRelevant[i])

fig.tight_layout()
from sklearn.model_selection import train_test_split

#Diving the data:

X = data.iloc[:,1:]

y = data.iloc[:,0]

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.33, random_state = 0)
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

import sklearn.metrics as m



#Defining metrics to save for each model:

def save_metrics(fittedModel, estimatorName, data_test = X_test):

    scoreList = [m.accuracy_score , m.precision_score, m.recall_score,

                 m.f1_score, m.log_loss, m.roc_auc_score]

    names = ['Accuracy','Precision','Recall','f1 score', 'Log loss', 'ROC AUC']

    ans = pd.DataFrame(names, columns = ['Scores']).set_index(['Scores'])

    ans[estimatorName] = 0

    for i,score in enumerate(scoreList):

        ans.iloc[i,0] = (score(y_test, fittedModel.predict(data_test)))

    print(estimatorName, m.classification_report(y_test, fittedModel.predict(data_test)))

    print(estimatorName, ':\n', m.confusion_matrix(y_test, fittedModel.predict(data_test)))

    return ans

models = list()



# Applying different classifier methods:

# Logistic Regression with CV:

lrcv = LogisticRegressionCV(Cs = 10, scoring = 'neg_log_loss')

lrcv.fit(X_train, y_train)

Results = save_metrics(lrcv, 'LRCV')

models.append(lrcv)



# Logistic with L-1 and L-2:

C = np.logspace(-5,5,20)

alpha = np.logspace(-5,5,20)

lrgrid = LogisticRegression()

param_values = {'C': C, 'penalty' : ['l1','l2']}

grid = GridSearchCV(lrgrid, param_grid = param_values, scoring = 'neg_log_loss')

grid.fit(X_train, y_train)

Results['LRGrid'] = save_metrics(grid, 'LRGrid')

lrgrid_best = grid.best_estimator_

models.append(lrgrid_best)



# Applying gridsearch on MLPClassifier:

import warnings

nn = MLPClassifier()

alpha = np.logspace(-5,5,20)

sizes = [(10,10,10),(10,10),(5,5)]

activation = ['logistic', 'tanh', 'relu']

param_values = {'hidden_layer_sizes' : sizes, 'activation' : activation, 'alpha' : alpha}

grid_nn = GridSearchCV(nn, param_grid = param_values, scoring = 'neg_log_loss')

with warnings.catch_warnings():

    warnings.filterwarnings("ignore")

    grid_nn.fit(X_train,y_train)

Results['Neural Grid'] = save_metrics(grid_nn, 'Neural Net Grid NLL')

grid_nn_best = grid_nn.best_estimator_

models.append(grid_nn_best)



# Applying Random Forest Classifier:

rfc = RandomForestClassifier()

n_estimators = np.arange(5,15)

min_samples_split = [3,5,10,15]

param_values = {'n_estimators' : n_estimators, 'min_samples_split' : min_samples_split}

grid_rfc = GridSearchCV(rfc, param_grid = param_values, scoring = 'neg_log_loss')

grid_rfc.fit(X_train, y_train)

Results['RFC Grid'] = save_metrics(grid_rfc, 'RFC Grid')

grid_rfc_best = grid_rfc.best_estimator_

models.append(grid_rfc_best)



# Applying SGD Classifier:

lrSGDC = SGDClassifier()

loss = ['log', 'perceptron']

alpha = np.logspace(-4,4,10)

penalty = ['l2','l1','elasticnet']

param_values = {'loss' : loss, 'alpha' : alpha, 'penalty' : penalty}

grid_sgdc = GridSearchCV(lrSGDC, param_grid = param_values)

grid_sgdc.fit(X_train, y_train)

grid_sgdc_best = grid_sgdc.best_estimator_

Results['Grid SGDC'] = save_metrics(grid_sgdc, 'Grid SGDC')

models.append(grid_sgdc_best)



#Applying KNN:

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

Results['KNN'] = save_metrics(knn, 'KNN')

models.append(knn)



# Applying SVC linear with GridSearch:

svc = SVC()

C = np.logspace(-4,4,20)

kernel = ['linear','rbf']

param_values = { 'C' : C, 'kernel' : kernel}

grid_svc = GridSearchCV(svc, param_grid = param_values)

grid_svc.fit(X_train, y_train)

grid_svc_best = grid_svc.best_estimator_

Results['SVC Grid'] = save_metrics(grid_svc, 'SVC Grid')

models.append(grid_svc_best)



print(Results)
# Logistic with L-1:

C = np.logspace(-5,5,20)

alpha = np.logspace(-5,5,20)

lrgridl1 = LogisticRegression()

param_values = {'C': C, 'penalty' : ['l1']}

grid_l1 = GridSearchCV(lrgridl1, param_grid = param_values, scoring = 'neg_log_loss')

grid_l1.fit(X_train, y_train)

Results['LRGrid L1'] = save_metrics(grid_l1, 'LRGrid L1')

lrgrid_best = grid.best_estimator_

nonZeroIndex = np.where(lrgrid_best.coef_ != 0)[1]



# Reducing the dimension of X using the L-1 coefficients:

X_train_red = X_train.iloc[:,nonZeroIndex]

X_test_red = X_test.iloc[:,nonZeroIndex]



models_red = list()

# Logistic with L-2:

C = np.logspace(-5,5,20)

alpha = np.logspace(-5,5,20)

lrgrid_red = LogisticRegression()

param_values = {'C': C, 'penalty' : ['l2']}

grid_lr_red = GridSearchCV(lrgrid_red, param_grid = param_values, scoring = 'neg_log_loss')

grid_lr_red.fit(X_train_red, y_train)

Results_red = save_metrics(grid_lr_red, 'LRGrid', X_test_red)

lrgrid_best_red = grid.best_estimator_

models_red.append(lrgrid_best_red)



# Applying gridsearch on MLPClassifier:

import warnings

nn_red = MLPClassifier()

alpha = np.logspace(-5,5,20)

sizes = [(10,10,10),(5,5,5),(10,10),(5,5)]

activation = ['logistic', 'tanh', 'relu']

param_values = {'hidden_layer_sizes' : sizes, 'activation' : activation, 'alpha' : alpha}

grid_nn_red = GridSearchCV(nn_red, param_grid = param_values, scoring = 'neg_log_loss')

with warnings.catch_warnings():

    warnings.filterwarnings("ignore")

    grid_nn_red.fit(X_train_red,y_train)

Results_red['Neural Grid'] = save_metrics(grid_nn_red, 'Neural Net Grid', X_test_red)

grid_nn_best_red = grid_nn.best_estimator_

models_red.append(grid_nn_best)



# Applying Random Forest Classifier:

rfc_red = RandomForestClassifier()

n_estimators = np.arange(5,15)

min_samples_split = [3,5,10,15]

param_values = {'n_estimators' : n_estimators, 'min_samples_split' : min_samples_split}

grid_rfc_red = GridSearchCV(rfc_red, param_grid = param_values, scoring = 'neg_log_loss')

grid_rfc_red.fit(X_train_red, y_train)

Results_red['RFC Grid'] = save_metrics(grid_rfc_red, 'RFC Grid', X_test_red)

grid_rfc_red_best = grid_rfc_red.best_estimator_

models_red.append(grid_rfc_best)



# Applying SVC linear with GridSearch:

svc_red = SVC()

C = np.logspace(-4,4,20)

kernel = ['linear','rbf']

param_values = { 'C' : C, 'kernel' : kernel}

grid_svc_red = GridSearchCV(svc_red, param_grid = param_values)

grid_svc_red.fit(X_train_red, y_train)

Results_red['SVC Grid'] = save_metrics(grid_svc_red, 'SVC Grid', X_test_red)

grid_svc_red_best = grid_svc_red.best_estimator_

models_red.append(grid_svc_red_best)



print(Results_red)
modelPredicts = np.empty(shape = (y_test.shape[0],len(models)))

modelProbs = np.empty(shape = (y_test.shape[0],len(models)))

for i,model in enumerate(models):

    modelPredicts[:,i] = model.predict(X_test)

    try:

        modelProbs[:,i] = model.predict_proba(X_test).reshape(-1,1)[1]

    except:

        modelProbs[:,i] = np.zeros(y_test.shape)



anyPositives = np.max(modelPredicts[:,(0,2,3)], axis = 1)

finalAcc = m.accuracy_score(y_test, anyPositives)

finalPrec = m.precision_score(y_test, anyPositives)

finalRec = m.recall_score(y_test, anyPositives)

finalF1 = m.f1_score(y_test, anyPositives)

finalLogLoss = m.log_loss(y_test, anyPositives)

finalROCAUC = m.roc_auc_score(y_test, anyPositives)



Results['Final'] = np.array([finalAcc, finalPrec, finalRec, finalF1,

       finalLogLoss, finalROCAUC])

print(Results)
chosenModelsIndex = [0,2,3]

finalPreds = pd.DataFrame()

for i in chosenModelsIndex:

    with warnings.catch_warnings():

        warnings.filterwarnings("ignore")

        models[i].fit(X,y)

        finalPreds[str(i)] = models[i].predict(data_test)

resultsDf = pd.DataFrame(np.max(finalPreds, axis = 1))

resultsDf.rename(columns = { 0 : 'Survived'}, inplace = True)

resultsDf.set_index(data_test.index, inplace = True)

resultsDf
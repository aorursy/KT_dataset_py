import pandas as pd

import numpy as np

import warnings

from matplotlib import pyplot

#from pandas import read_csv, set_option

from pandas import Series, datetime

from pandas.tools.plotting import scatter_matrix, autocorrelation_plot

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, TimeSeriesSplit

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

import random

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_model import ARIMA

from xgboost import XGBClassifier

import seaborn as sns
# Loading the dataset to a dataframe

sentence_file = "../input/headlinespolarity/combined_stock_data.csv"

sentence_df = pd.read_csv(sentence_file, parse_dates=[1])
sentence_df.head()
# Check the shape and data types of the dataframe

print(sentence_df.shape)

print(sentence_df.dtypes)
# Load the stock prices dataset into a dataframe and check the top 5 rows

stock_prices = "../input/stocknews/DJIA_table.csv"

stock_data = pd.read_csv(stock_prices, parse_dates=[0])

stock_data.head()
# Check the shape and datatypes of the stock prices dataframe

print(stock_data.shape)

print(stock_data.dtypes)
# Create a dataframe by merging the headlines and the stock prices dataframe

merged_dataframe = sentence_df[['Date', 'Label', 'Subjectivity', 'Objectivity', 'Positive', 'Negative', 'Neutral']].merge(stock_data, how='inner', on='Date', left_index=True)

# Check the shape and top 5 rows of the merged dataframe

print(merged_dataframe.shape)

merged_dataframe.head()
# Push the Label column to the end of the dataframe

cols = list(merged_dataframe)

print(cols)

cols.append(cols.pop(cols.index('Label')))

merged_dataframe = merged_dataframe.ix[:, cols]

merged_dataframe.head()
# Change the datatype of the volume column to float

#merged_dataframe['Date'] = pd.to_datetime(merged_dataframe['Date'])

merged_dataframe['Volume'] = merged_dataframe['Volume'].astype(float)

print(cols)

#merged_dataframe = merged_dataframe.set_index(['Date'])

merged_dataframe.index = merged_dataframe.index.sort_values()

merged_dataframe.head()
# Check the statistics of the columns of the merged dataframe and check for outliers

print(merged_dataframe.describe())
# Plot a histogram for all the columns of the dataframe. This shows the frequency of values in all the columns

sns.set()

merged_dataframe.hist(sharex = False, sharey = False, xlabelsize = 4, ylabelsize = 4, figsize=(10, 10))

pyplot.show()
pyplot.scatter(merged_dataframe['Subjectivity'], merged_dataframe['Label'])

pyplot.xlabel('Subjectivity')

pyplot.ylabel('Stock Price Up or Down 0: Down, 1: Up')

pyplot.show()

pyplot.scatter(merged_dataframe['Objectivity'], merged_dataframe['Label'])

pyplot.xlabel('Objectivity')

pyplot.ylabel('Stock Price Up or Down 0: Down, 1: Up')

pyplot.show()

merged_dataframe['Subjectivity'].plot('hist')

pyplot.xlabel('Subjectivity')

pyplot.ylabel('Frequency')

pyplot.show()

merged_dataframe['Objectivity'].plot('hist')

pyplot.xlabel('Subjectivity')

pyplot.ylabel('Frequency')

pyplot.show()

print("Size of the Labels column")

print(merged_dataframe.groupby('Label').size())
md_copy = merged_dataframe

md_copy = md_copy.replace(-1, np.NaN)

import missingno as msno

# Nullity or missing values by columns

msno.matrix(df=md_copy.iloc[:,2:39], figsize=(20, 14), color=(0.42, 0.1, 0.05))
colormap = pyplot.cm.afmhot

pyplot.figure(figsize=(16,12))

pyplot.title('Pearson correlation of continuous features', y=1.05, size=15)

sns.heatmap(merged_dataframe.corr(),linewidths=0.1,vmax=1.0, square=True, 

            cmap=colormap, linecolor='white', annot=True)

pyplot.show()
%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings



bin_col = merged_dataframe.columns

zero_list = []

one_list = []

for col in bin_col:

    zero_count = 0

    one_count = 0

    for ix, val in merged_dataframe[col].iteritems():

        if merged_dataframe.loc[ix, 'Label'] == 0:

            zero_count += 1

        else:

            one_count += 1

    zero_list.append(zero_count)

    one_list.append(one_count)

    

trace1 = go.Bar(

    x=bin_col,

    y=zero_list ,

    name='Zero count'

)

trace2 = go.Bar(

    x=bin_col,

    y=one_list,

    name='One count'

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    title='Count of 1 and 0 in binary variables'

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked-bar')
# Print the datatypes and count of the dataframe

print(merged_dataframe.dtypes)

print(merged_dataframe.count())

# Change the NaN values to the mean value of that column

nan_list = ['Subjectivity', 'Objectivity', 'Positive', 'Negative', 'Neutral']

for col in nan_list:

    merged_dataframe[col] = merged_dataframe[col].fillna(merged_dataframe[col].mean())



# Recheck the count

print(merged_dataframe.count())

# Separate the dataframe for input(X) and output variables(y)

X = merged_dataframe.loc[:,'Subjectivity':'Adj Close']

y = merged_dataframe.loc[:,'Label']

# Set the validation size, i.e the test set to 20%

validation_size = 0.20

# Split the dataset to test and train sets

# Split the initial 70% of the data as training set and the remaining 30% data as the testing set

train_size = int(len(X.index) * 0.7)

print(len(y))

print(train_size)

X_train, X_test = X.loc[0:train_size, :], X.loc[train_size: len(X.index), :]

y_train, y_test = y[0:train_size+1], y.loc[train_size: len(X.index)]

print('Observations: %d' % (len(X.index)))

print('X Training Observations: %d' % (len(X_train.index)))

print('X Testing Observations: %d' % (len(X_test.index)))

print('y Training Observations: %d' % (len(y_train)))

print('y Testing Observations: %d' % (len(y_test)))

pyplot.plot(X_train['Objectivity'])

pyplot.plot([None for i in X_train['Objectivity']] + [x for x in X_test['Objectivity']])

pyplot.show()

num_folds = 10

scoring = 'accuracy'

# Append the models to the models list

models = []

models.append(('LR' , LogisticRegression()))

models.append(('LDA' , LinearDiscriminantAnalysis()))

models.append(('KNN' , KNeighborsClassifier()))

models.append(('CART' , DecisionTreeClassifier()))

models.append(('NB' , GaussianNB()))

models.append(('SVM' , SVC()))

models.append(('RF' , RandomForestClassifier(n_estimators=50)))

models.append(('XGBoost', XGBClassifier()))
# Evaluate each algorithm for accuracy

results = []

names = []

'''

for name, model in models:

    kfold = KFold(n_splits=num_folds, random_state=42)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg) '''



for name, model in models:

    clf = model

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accu_score = accuracy_score(y_test, y_pred)

    print(name + ": " + str(accu_score))
# prepare the model LDA

scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

model_lda = LinearDiscriminantAnalysis()

model_lda.fit(rescaledX, y_train)

# estimate accuracy on validation dataset

rescaledValidationX = scaler.transform(X_test)

predictions = model_lda.predict(rescaledValidationX)

print("accuracy score:")

print(accuracy_score(y_test, predictions))

print("confusion matrix: ")

print(confusion_matrix(y_test, predictions))

print("classification report: ")

print(classification_report(y_test, predictions))



model_xgb = XGBClassifier()

model_xgb.fit(rescaledX, y_train)

# estimate accuracy on validation dataset

rescaledValidationX = scaler.transform(X_test)

predictions = model_xgb.predict(rescaledValidationX)

print("accuracy score:")

print(accuracy_score(y_test, predictions))

print("confusion matrix: ")

print(confusion_matrix(y_test, predictions))

print("classification report: ")

print(classification_report(y_test, predictions))

# Generating the ROC curve

y_pred_proba = model_lda.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

roc_auc = auc(fpr, tpr)



# Plot ROC curve

print("roc auc is :" + str(roc_auc))

pyplot.plot([0, 1], [0, 1], 'k--')

pyplot.plot(fpr, tpr)

pyplot.xlabel('False Positive Rate')

pyplot.ylabel('True Positive Rate')

pyplot.title('ROC Curve')

pyplot.show()



# AUC score using cross validation

kfold_val = KFold(n_splits=num_folds, random_state=42)

auc_score = cross_val_score(model_lda, X_test, y_test, cv=kfold_val, scoring='roc_auc')

print("AUC using cross val: " + str(auc_score))

mean_auc = np.mean(auc_score)

print("Mean AUC score is: " + str(mean_auc))
# Scaling Random Forests



model_rf = RandomForestClassifier(n_estimators=1000)

model_rf.fit(rescaledX, y_train)

# estimate accuracy on validation dataset

rescaledValidationX = scaler.transform(X_test)

predictions = model_rf.predict(rescaledValidationX)

print("accuracy score:")

print(accuracy_score(y_test, predictions))

print("confusion matrix: ")

print(confusion_matrix(y_test, predictions))

print("classification report: ")

print(classification_report(y_test, predictions))

features = merged_dataframe.drop(['Label'],axis=1).columns.values



x, y = (list(x) for x in zip(*sorted(zip(model_xgb.feature_importances_, features), 

                                                            reverse = False)))

trace2 = go.Bar(

    x=x ,

    y=y,

    marker=dict(

        color=x,

        colorscale = 'Viridis',

        reversescale = True

    ),

    name='Feature importance for XGBoost',

    orientation='h',

)



layout = dict(

    title='Barplot of Feature importances for XGBoost',

     width = 1000, height = 1000,

    yaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

#         domain=[0, 0.85],

    ))



fig1 = go.Figure(data=[trace2])

fig1['layout'].update(layout)

py.iplot(fig1, filename='plots')

x, y = (list(x) for x in zip(*sorted(zip(model_rf.feature_importances_, features), 

                                                            reverse = False)))

trace2 = go.Bar(

    x=x ,

    y=y,

    marker=dict(

        color=x,

        colorscale = 'Viridis',

        reversescale = True

    ),

    name='Feature importance for Random Forests',

    orientation='h',

)



layout = dict(

    title='Barplot of Feature importances for Random Forests',

     width = 1000, height = 1000,

    yaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

#         domain=[0, 0.85],

    ))



fig1 = go.Figure(data=[trace2])

fig1['layout'].update(layout)

py.iplot(fig1, filename='plots')

# XGBoost on Stock Price dataset, Tune n_estimators and max_depth

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder

import matplotlib



matplotlib.use('Agg')

model = XGBClassifier()

n_estimators = [150, 200, 250, 450, 500, 550, 1000]

max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

print(max_depth)

best_depth = 0

best_estimator = 0

max_score = 0

for n in n_estimators:

    for md in max_depth:

        model = XGBClassifier(n_estimators=n, max_depth=md)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        score = accuracy_score(y_test, y_pred)

        if score > max_score:

            max_score = score

            best_depth = md

            best_estimator = n

        print("Score is " + str(score) + " at depth of " + str(md) + " and estimator " + str(n))

print("Best score is " + str(max_score) + " at depth of " + str(best_depth) + " and estimator of " + str(best_estimator))

imp_features_df = merged_dataframe[['Low', "Neutral", 'Close', 'Objectivity', 'Date']]

Xi_train, Xi_test = X.loc[0:train_size, :], X.loc[train_size: len(X.index), :]

clf = XGBClassifier(n_estimators=500, max_depth=3)

clf.fit(Xi_train, y_train)

yi_pred = clf.predict(Xi_test)

score = accuracy_score(y_test, yi_pred)

print("Score is "+ str(score))

from sklearn.decomposition import PCA



pca = PCA(n_components=3)

pca.fit(X)

transformed = pca.transform(X)



transformed.shape

print(type(transformed))
pca_df = pd.DataFrame(transformed)



X_train_pca, X_test_pca = pca_df.loc[0:train_size, :], pca_df.loc[train_size: len(X.index), :]



clf = XGBClassifier(n_estimators=500, max_depth=3)

clf.fit(X_train_pca, y_train)

y_pred_pca = clf.predict(X_test_pca)

score = accuracy_score(y_test, y_pred_pca)

print("Score is "+ str(score))

pca_matrix = confusion_matrix(y_test, y_pred_pca)

pca_report = classification_report(y_test, y_pred_pca)

print("Confusion Matrix: \n" + str(pca_matrix))

print("Classification report: \n" + str(pca_report))
# Generating the ROC curve

y_pred_proba_pca = clf.predict_proba(X_test_pca)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_pca)

roc_auc = auc(fpr, tpr)

print("AUC score is " + str(roc_auc))



# Plot ROC curve

print("roc auc is :" + str(roc_auc))

pyplot.plot([0, 1], [0, 1], 'k--')

pyplot.plot(fpr, tpr)

pyplot.xlabel('False Positive Rate')

pyplot.ylabel('True Positive Rate')

pyplot.title('ROC Curve')

pyplot.show()

splits = TimeSeriesSplit(n_splits=3)

pyplot.figure(1)

index = 1

print(splits.split(X))

for X_train_index, X_test_index in splits.split(X):

    X1_train = X.loc[X_train_index, :]

    X1_test = X.loc[X_test_index, :]

    print('Observations: %d' % (len(X1_train) + len(X1_test)))

    print('Training Observations: %d' % (len(X1_train)))

    print('Testing Observations: %d' % (len(X1_test)))

    pyplot.subplot(310 + index)

    pyplot.plot(X1_train['Objectivity'])

    pyplot.plot([None for i in X1_train['Objectivity']] + [x for x in X1_test['Objectivity']])

    index += 1

pyplot.show()



index = 1

#y = np.array(y)

y_ser = list(splits.split(y))

print(y_ser[0])

for y_train_index, y_test_index in y_ser:

    y1_train = y[y_train_index]

    y1_test = y[y_test_index]

    print('Observations: %d' % (len(y1_train) + len(y1_test)))

    print('Training Observations: %d' % (len(y1_train)))

    print('Testing Observations: %d' % (len(y1_test)))

    pyplot.subplot(310 + index)

    pyplot.plot(y1_train)

    pyplot.plot([None for i in y1_train] + [x for x in y1_test])

    index += 1

pyplot.show()

# Evaluate each algorithm for accuracy

results = []

names = []



for name, model in models:

    clf1 = model

    clf1.fit(X1_train, y1_train)

    y1_pred = clf.predict(X1_test)

    accu_score1 = accuracy_score(y1_test, y1_pred)

    print(name + ": " + str(accu_score))
# prepare the model LDA

scaler1 = StandardScaler().fit(X1_train)

rescaledX1 = scaler1.transform(X1_train)

model_lda1 = LinearDiscriminantAnalysis()

model_lda1.fit(rescaledX1, y1_train)

# estimate accuracy on validation dataset

rescaledValidationX1 = scaler1.transform(X1_test)

predictions1 = model_lda1.predict(rescaledValidationX1)

print("accuracy score:")

print(accuracy_score(y1_test, predictions1))

print("confusion matrix: ")

print(confusion_matrix(y1_test, predictions1))

print("classification report: ")

print(classification_report(y1_test, predictions1))
# prepare the model Random Forest

scaler_rf = StandardScaler().fit(X1_train)

rescaledX_rf = scaler_rf.transform(X1_train)

model_rf = RandomForestClassifier(n_estimators=500)

model_rf.fit(rescaledX_rf, y1_train)

# estimate accuracy on validation dataset

rescaledValidationX_rf = scaler_rf.transform(X1_test)

predictions_rf = model_rf.predict(rescaledValidationX_rf)

print("accuracy score:")

print(accuracy_score(y1_test, predictions_rf))

print("confusion matrix: ")

print(confusion_matrix(y1_test, predictions_rf))

print("classification report: ")

print(classification_report(y1_test, predictions_rf))
# Generating the ROC curve

y1_pred_proba = model_lda1.predict_proba(X1_test)[:,1]

fpr1, tpr1, thresholds1 = roc_curve(y1_test, y1_pred_proba)

roc_auc1 = auc(fpr1, tpr1)



# Plot ROC curve

print("roc auc is :" + str(roc_auc1))

pyplot.plot([0, 1], [0, 1], 'k--')

pyplot.plot(fpr1, tpr1)

pyplot.xlabel('False Positive Rate')

pyplot.ylabel('True Positive Rate')

pyplot.title('ROC Curve')

pyplot.show()
pyplot.plot(merged_dataframe['Date'], merged_dataframe['Close'])

pyplot.xlabel('Years')

pyplot.ylabel('Stock Closing Prices')

pyplot.show()

merged_dataframe.info()
merged_dataframe['Date'] = merged_dataframe['Date'].dt.strftime('%Y-%m')
df_dateclose = pd.Series(merged_dataframe['Close'].values, index = merged_dataframe['Date'])

print(df_dateclose.index)

print(df_dateclose.head())

autocorrelation_plot(df_dateclose)

pyplot.show()
merged_dataframe['Close'].head()
plot_pacf(df_dateclose, lags=50)
plot_acf(df_dateclose)
plot_acf(df_dateclose, lags=50)
arima_df = pd.DataFrame(merged_dataframe, index=merged_dataframe['Date'])
arima_df.index


# evaluate an ARIMA model for a given order (p,d,q)

def evaluate_arima_model(X, arima_order):

    # prepare training dataset

    train_size = int(len(X) * 0.66)

    train, test = X[0:train_size], X[train_size:]

    history = [x for x in train]

    # make predictions

    predictions = list()

    for t in range(len(test)):

        model = ARIMA(history, order=arima_order)

        model_fit = model.fit(disp=0)

        yhat = model_fit.forecast()[0]

        predictions.append(yhat)

        history.append(test[t])

    # calculate out of sample error

    error = mean_squared_error(test, predictions)

    return error



# evaluate combinations of p, d and q values for an ARIMA model

def evaluate_models(dataset, p_values, d_values, q_values):

    dataset = dataset.astype('float32')

    best_score, best_cfg = float("inf"), None

    for p in p_values:

        for d in d_values:

            for q in q_values:

                order = (p,d,q)

                try:

                    mse = evaluate_arima_model(dataset, order)

                    if mse < best_score:

                        best_score, best_cfg = mse, order

                    print('ARIMA%s MSE=%.3f' % (order,mse))

                except:

                    continue

    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))



# evaluate parameters

p_values = [0, 1, 2, 4, 6, 8, 10]

d_values = range(0, 3)

q_values = range(0, 3)

warnings.filterwarnings("ignore")

evaluate_models(df_dateclose.values, p_values, d_values, q_values)
df_dateclose.values
X = df_dateclose.values

size = int(len(X) * 0.66)

train, test = X[0:size], X[size:len(X)]

history = [x for x in train]

predictions = list()

for t in range(len(test)):

    model = ARIMA(history, order=(0,1,0))

    model_fit = model.fit(disp=0)

    output = model_fit.forecast()

    yhat = output[0]

    predictions.append(yhat)

    obs = test[t]

    history.append(obs)

    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)

print('Test MSE: %.3f' % error)

# plot

pyplot.plot(test)

pyplot.plot(predictions, color='red')

pyplot.show()
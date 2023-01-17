import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.display import display

from time import time

import warnings

warnings.filterwarnings("ignore")



%matplotlib inline



data = pd.read_csv("../input/weatherAUS.csv")

data.insert(loc=1, column='Month', value = data['Date'].apply(lambda x: x[5:7])) #create column "Month"

data.insert(loc=2, column='Day', value = data['Date'].apply(lambda x: x[5:10])) #create column "Month"

data.insert(loc=3, column='Season', value = data['Month'].replace(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], ['summer','summer', 'summer', 'fall', 'fall', 'fall', 'winter', 'winter', 'winter', 'spring', 'spring', 'spring'])) #create column "Season"

data['RainTomorrow'] = data['RainTomorrow'].replace(['Yes', 'No'], [1,0])

print('Dataset shape:',data.shape)

display(data.head(n=5))
fill = pd.DataFrame(100*data.count().sort_values()/data.shape[0])

fill.reset_index(level=0, inplace=True)

fill.columns = ['Variable','Fill (%)']



sns.set()

plt.figure(figsize=(16, 5))

g = sns.barplot(x = 'Variable', y = 'Fill (%)', data = fill,color = 'orange')

g = plt.xticks(rotation=75)
import missingno as msno

msno.matrix(data)
types = pd.DataFrame(data.dtypes)

types.reset_index(level=0, inplace=True)

types.columns = ['Variable','Type']



numerical_variables = list(types[types['Type'] == 'float64']['Variable'].values)

categorical_variables = list(types[types['Type'] == 'object']['Variable'].values)



print ('numerical_variables:', numerical_variables) 

print ('\ncategorical_variables:', categorical_variables) 
#pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')

f, ax = plt.subplots(figsize=(16, 10))

corr = data.corr()

corr_mtx = sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=False, ax=ax, annot = True)
n_row = 4

n_col = 4

list_of_variables = numerical_variables[0:-1]

f, axes = plt.subplots(n_row, n_col, figsize=(16, 15))

total = n_row*n_col

k = 0

for i in range(n_row):

    for j in range(n_col):

        sns.distplot(data[data['RainTomorrow']==1][list_of_variables[k]].dropna(),hist = False, kde = True, label = 'Yes',ax=axes[i, j])#.set_title(list_of_variables[k])

        sns.distplot(data[data['RainTomorrow']==0][list_of_variables[k]].dropna(), hist = False, kde = True, label = 'No',ax=axes[i, j])#.set_title(list_of_variables[k])

        k = k + 1
print('Probability of Rain Tomorrow:',np.mean(data['RainTomorrow']))
from numpy import mean



n_row = 3

n_col = 2

list_of_variables = ['Month', 'Season', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

f, axes = plt.subplots(n_row, n_col, figsize=(16, 12))

k = 0

for i in list(range(n_row)):

    for j in list(range(n_col)):

        sns.barplot(x = list_of_variables[k], y = 'RainTomorrow', data = data, estimator = mean, color = 'orange', ax=axes[i, j])

        #g.xticks(rotation=45)

        k = k + 1
data_final_variables = data.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','Date', 'Month', 'Day', 'RISK_MM'],axis=1)



data_final_variables = data_final_variables.dropna()



features_raw = data_final_variables.drop(columns = ['RainTomorrow'])

income_raw = data_final_variables['RainTomorrow']
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures



scaler = MinMaxScaler()



types_aux = pd.DataFrame(features_raw.dtypes)

types_aux.reset_index(level=0, inplace=True)

types_aux.columns = ['Variable','Type']

numerical = list(types_aux[types_aux['Type'] == 'float64']['Variable'].values)



features_minmax_transform = pd.DataFrame(data = features_raw)

features_minmax_transform[numerical] = scaler.fit_transform(features_raw[numerical])



features_minmax_transform['RainToday'] = features_minmax_transform['RainToday'].replace(['Yes', 'No'], [1,0])



features_minmax_transform.head()
features_final = pd.get_dummies(features_minmax_transform)



encoded = list(features_final.columns)

print ("{} total features after one-hot encoding.".format(len(encoded)))



# Descomente a linha abaixo para ver as colunas após o encode

print (encoded)
from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier



X = features_final

y = income_raw



clf_a = MultinomialNB()

clf_b = DecisionTreeClassifier(random_state = 0)

clf_c = RandomForestClassifier(random_state = 0)

clf_d = LogisticRegression(random_state = 0)

clf_e = SGDClassifier(random_state = 0)



list_clf = [clf_a, clf_b, clf_c, clf_d, clf_e]



results = []

for clf in list_clf:

    start = time()

    clf_name = clf.__class__.__name__

    scores_f1 = cross_val_score(clf, X, y, cv=5, scoring = 'f1')

    scores_ = cross_val_score(clf, X, y, cv=5)

    end = time()

    train_time = end  - start

    results.append([clf_name, np.mean(scores_f1), np.mean(scores_), train_time])



df_results = pd.DataFrame(np.array(results))

df_results.columns = ['Classifier', 'F1-Score', 'Accuracy', 'Train Time']

df_results.sort_values(by=['F1-Score'], ascending=False)
from sklearn.decomposition import PCA



clf_pca = LogisticRegression(random_state=0)



list_n = [10,20,30,40,50]



results_pca = []

for i in list_n:

    start = time()

    pca = PCA(n_components = i)

    pca.fit(X)

    X_pca = pca.fit_transform(X)

    scores_f1 = cross_val_score(clf_pca, X_pca, y, cv=5, scoring = 'f1')

    scores_ = cross_val_score(clf_pca, X_pca, y, cv=5)

    explained_variance = np.sum(pca.explained_variance_ratio_)

    end = time()

    train_time = end  - start

    results_pca.append([i, explained_variance, np.mean(scores_f1), np.mean(scores_), train_time])



df_results_pca = pd.DataFrame(np.array(results_pca))

df_results_pca.columns = ['Number of components', 'Cumulative Explained Variance Ration','F1-Score', 'Accuracy', 'Train Time']

df_results_pca.sort_values(by=['F1-Score'], ascending=False)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



clf_feature_importance = RandomForestClassifier(random_state=0)

model_feature_importance = clf_feature_importance.fit(X_train,y_train)

importances = model_feature_importance.feature_importances_



df_feature_importance = pd.DataFrame()

df_feature_importance['features'] = X.columns

df_feature_importance['importances'] = importances

df_feature_importance = df_feature_importance.sort_values(by=['importances'], ascending=False)

features = list(df_feature_importance['features'].values)



list_n_features = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60]



results_features = []

for i in list_n_features:

    start = time()

    X_selected_features = X[features[0:i]]

    scores_f1 = cross_val_score(LogisticRegression(random_state=0), X_selected_features, y, cv=5, scoring = 'f1')

    scores_ = cross_val_score(LogisticRegression(random_state=0), X_selected_features, y, cv=5)

    cummulative_importance = sum(df_feature_importance.importances[0:i])

    end = time()

    train_time = end  - start

    results_features.append([i, cummulative_importance, np.mean(scores_f1), np.mean(scores_), train_time])



df_results_features = pd.DataFrame(np.array(results_features))

df_results_features.columns = ['Number of features', 'Cumulative Importance','F1-Score', 'Accuracy', 'Train Time']

df_results_features.sort_values(by=['F1-Score'], ascending=False)
print ('Most important features:\n')

for i in features[0:10]:

    print(i)
from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import fbeta_score, make_scorer

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



clf_grid = LogisticRegression(random_state=0, n_jobs = -1)



parameters = {'C':np.logspace(0, 4, 10),

             'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}



scorer = make_scorer(fbeta_score, beta=0.5)



grid_obj = GridSearchCV(clf_grid, parameters, scoring=scorer, n_jobs = -1)



grid_fit = grid_obj.fit(X_train, y_train)



best_clf = grid_fit.best_estimator_



print ("Logistic Regression\n")

print ("Unoptimized model\n------")

print ("Accuracy score on testing data: {:.4f}".format(np.mean(cross_val_score(clf_grid, X, y, cv=5))))

print ("F-score on testing data: {:.4f}".format(np.mean(cross_val_score(clf_grid, X, y, cv=5, scoring = 'f1'))))

print ("\nOptimized Model\n------")

print ("Accuracy score on testing data: {:.4f}".format(np.mean(cross_val_score(best_clf, X, y, cv=5))))

print ("F-score on testing data: {:.4f}".format(np.mean(cross_val_score(best_clf, X, y, cv=5, scoring = 'f1'))))

print ('\n')

print (best_clf)
from sklearn.metrics import confusion_matrix



final_model = LogisticRegression()

final_model.fit(X_train,y_train)

y_pred = final_model.predict(X_test)

confusion_matrix(y_test, y_pred)
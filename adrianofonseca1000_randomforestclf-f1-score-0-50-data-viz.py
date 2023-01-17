#Bibliotecas para criação e manipulação de DATAFRAMES e Algebra 

import pandas as pd

import numpy as np



#Bibliotecas para geração de gráficos

import matplotlib.pyplot as plt 

import seaborn as sns



#Bibliotecas para execução das metricas e modelo

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix, precision_score, accuracy_score, recall_score

from sklearn.metrics import precision_recall_curve, roc_curve , roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.preprocessing import scale

import missingno as msno
data = pd.read_csv('../input/titanic_data.csv')

print ('This one dataset it has %s rows e %s columns' % (data.shape[0], data.shape[1]))
data.info()
# Compute the correlation matrix

corr = data.loc[:, ['Pclass', 'Age', 'SibSp','Parch', 'Fare', 'Survived']].corr(method='spearman')



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmin= -1.0, vmax=1.0, annot=True)
potentialFeatures = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
# check how the features are correlated with the overall ratings



for f in potentialFeatures:

    related = data['Survived'].corr(data[f], method='spearman')

    print("%s: %f" % (f,related))
cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
# create a list containing Pearson's correlation between 'overall_rating' with each column in cols

correlations = [ data['Survived'].corr(data[f], method='spearman') for f in cols ]
len(cols), len(correlations)
# create a function for plotting a dataframe with string columns and numeric values



def plot_dataframe(data, y_label):  

    color='coral'

    fig = plt.gcf()

    fig.set_size_inches(9, 7)

    plt.ylabel(y_label)



    ax = data.correlation.plot(linewidth=3.3, color=color)

    ax.set_xticks(data.index)

    ax.set_xticklabels(data.attributes, rotation=75); #Notice the ; (remove it and see what happens !)

    plt.show()
# create a dataframe using cols and correlations



data2 = pd.DataFrame({'attributes': cols, 'correlation': correlations}) 
# let's plot above dataframe using the function we created

    

plot_dataframe(data2, 'Surviving Overall Rating')
#is any row NULL ?

data.isnull().any().any(), data.shape
msno.matrix(data.loc[:, ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 

                         'Embarked']],figsize=(11,9))
print(data.loc[:, ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']].count())
data.Age.fillna(data['Age'].mean(), inplace=True)
sns.countplot(data['Survived'], label="Count")
data['Age'].describe()
data.Age.hist (bins=50, figsize=(5,3), color = "blue")

plt.show
data['SibSp'].describe()
data.SibSp.hist (bins=50, figsize=(5,3), color = "blue")

plt.show
data['Parch'].describe()
data.Parch.hist (bins=50, figsize=(5,3), color = "blue")

plt.show
data['Fare'].describe()
data.Fare.hist (bins=50, figsize=(5,3), color = "blue")

plt.show
data.drop(['PassengerId', 'Pclass', 'Name', 'Cabin', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
data.head()
data.tail()
#def X and Y

Y = np.array(data.Survived.tolist())

predictors = data.drop('Survived', axis=1)

X = np.array(predictors.as_matrix())

seed=42
Y.shape
X.shape
print(X)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)



for train_index, test_index in skf.split(X, Y):

    

    X_train, X_test = X[train_index], X[test_index]

    Y_train, Y_test = Y[train_index], Y[test_index]
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
# ROC curve

def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
# precision-recall curve

def plot_precision_recall():

    plt.step(recall, precision, color = 'b', alpha = 0.2,

             where = 'post')

    plt.fill_between(recall, precision, step ='post', alpha = 0.2,

                 color = 'b')



    plt.plot(recall, precision, linewidth=2)

    plt.xlim([0.0,1])

    plt.ylim([0.0,1.05])

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.title('Precision Recall Curve')

    plt.show();
#feature importance plot

def plot_feature_importance(model):

    tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': model.feature_importances_})

    tmp = tmp.sort_values(by='Feature importance',ascending=False)

    plt.figure(figsize = (7,5))

    plt.title('Features importance',fontsize=14)

    s = sns.barplot(y='Feature',x='Feature importance',data=tmp)

    s.set_yticklabels(s.get_yticklabels(),rotation=360)

    plt.show()
# Ajustar o modelo usando X como dados de treinamento e y como valores de destino

rf_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)

                                

rf_clf = rf_clf.fit(X_train, Y_train)



# Modelo prevendo os valores para o conjunto de teste

Y_pred = rf_clf.predict(X_test)



# Prever as probabilidades de classe para o conjunto de teste

Y_score_rf = rf_clf.predict_proba(X_test)[:,1]



# Matriz de confusão

print("Confusion matrix:")

print(confusion_matrix(Y_test, Y_pred))



# Calculo Accuracy 

print("Accuracy:", accuracy_score(Y_test, Y_pred))



# Reportar para outras medidas de classificação

print("Classification report:")

print(classification_report(Y_test, Y_pred))



# ROC curve

fpr_rfc, tpr_rfc, thresholds = roc_curve(Y_test, Y_score_rf) #Test and probability

plot_roc_curve(fpr_rfc, tpr_rfc)



auc = roc_auc_score(Y_test, Y_score_rf)

print('AUC: %.2f' % auc)



# Precision-recall curve

print('Plot the Precision-Recall curve')

precision, recall, thresholds = precision_recall_curve(Y_test, Y_score_rf) #Test and probability

plot_precision_recall()



plot_feature_importance(rf_clf)
param_grid = {

            'n_estimators': [50, 100, 200, 300, 500, 800, 1000, 2000],

            'max_features': [2, 3, 4],

            'min_samples_leaf': [1, 2, 4],

            'min_samples_split': [2, 5, 10, 100]

            }
gs_rf = GridSearchCV(estimator = rf_clf, param_grid = param_grid, scoring = 'f1', verbose = 10, n_jobs=-1)

gs_rf.fit(X_train, Y_train)



best_parameters = gs_rf.best_params_

print("The best parameters for using this model is", best_parameters)
# Ajustar o modelo usando X como dados de treinamento e y como valores de destino

rf_clf = RandomForestClassifier(max_features=2, min_samples_leaf=1, min_samples_split=10, 

                                n_estimators=1000, n_jobs=-1)

                                

rf_clf = rf_clf.fit(X_train, Y_train)



# Modelo prevendo os valores para o conjunto de teste

Y_pred = rf_clf.predict(X_test)



# Prever as probabilidades de classe para o conjunto de teste

Y_score_rf = rf_clf.predict_proba(X_test)[:,1]



# Matriz de confusão

print("Confusion matrix:")

print(confusion_matrix(Y_test, Y_pred))



# Calculo Accuracy 

print("Accuracy:", accuracy_score(Y_test, Y_pred))



# Reportar para outras medidas de classificação

print("Classification report:")

print(classification_report(Y_test, Y_pred))



# ROC curve

fpr_rfc, tpr_rfc, thresholds = roc_curve(Y_test, Y_score_rf) #Test and probability

plot_roc_curve(fpr_rfc, tpr_rfc)



auc = roc_auc_score(Y_test, Y_score_rf)

print('AUC: %.2f' % auc)



# Precision-recall curve

print('Plot the Precision-Recall curve')

precision, recall, thresholds = precision_recall_curve(Y_test, Y_score_rf) #Test and probability

plot_precision_recall()



plot_feature_importance(rf_clf)
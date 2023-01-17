import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
'''Plotly visualization with cufflinks and native style.'''
import plotly
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import cufflinks as cf
cf.go_offline() # Required to use plotly offline with cufflinks.
py.init_notebook_mode() # Graphs charts inline (jupyter notebook).
init_notebook_mode() # Required to use plotly offline in jupyter notebook
import missingno as mn
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
'''Take offline the warnings'''
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
# warnings.simplefilter('error', SettingWithCopyWarning)
def load_dataset():
    train_data = pd.read_csv(os.path.join('../input', 'train.csv'))
    test_data = pd.read_csv(os.path.join('../input', 'test.csv'))
    return pd.concat([train_data, test_data]), train_data, test_data

def get_frequency(var):
    value_frequency = round(var.value_counts(normalize=True)*100, 2)
    return pd.DataFrame({'value_frequency': value_frequency})

def get_model_accuracy(model, x, y):
    model.fit(x, y)
    model_accuracy = model.score(x, y)
    return np.round(model_accuracy*100, 2)

def get_model_accuracy_cross_val(model, x, y):
    x_val_score = cross_val_score(model, x, y, cv = 10, scoring = 'accuracy').mean()
    return np.round(x_val_score*100, 2)

def tune_hyperparemeters(model, params):
    grid = GridSearchCV(model, params, verbose = 2, cv = 10, scoring = 'accuracy', n_jobs = -1)
    grid.fit(X_train, y_train)
    return grid.best_params_, np.round(grid.best_score_*100, 2)

def plot_feature_importance(model, title):
    importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': np.round(model.feature_importances_,5)})
    importance = importance.sort_values(by='Importance', ascending=False).set_index('Feature')
    trace = go.Scatter(x=importance.index, y=importance.Importance, mode='markers',
                       marker=dict(color=np.random.randn(500), size=20, showscale=True, colorscale='Rainbow'))
    layout = go.Layout(hovermode='closest', title=title, yaxis=dict(title='Importance'))
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
all_data, train_data, test_data = load_dataset()
all_data.head()
all_data.dtypes
get_frequency(all_data['Survived'])
get_frequency(all_data['Sex'])
get_frequency(all_data['Pclass'])
get_frequency(all_data['Embarked'])
all_data.Cabin.head(10)
all_data.Name.head()
all_data.Ticket.head()
get_frequency(all_data['SibSp'])
get_frequency(all_data['Parch'])
all_data.Cabin.fillna(value='X', inplace=True)
all_data.Cabin = all_data.Cabin.apply(lambda x: x[0])
get_frequency(all_data['Cabin'])
all_data['Title'] = all_data.Name.apply(lambda x: re.search('([A-z]+)\.', x).group().replace('.', ''))
all_data.Title.replace(to_replace={'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, inplace=True)
all_data.Title.replace(to_replace=['Dona', 'Jonkheer', 'Countess', 'Sir', 'Lady', 'Don'], value='Aristocrat', inplace=True)
all_data.Title.replace(to_replace = ['Dr', 'Rev', 'Col', 'Major', 'Capt'], value = 'Officer', inplace = True)
get_frequency(all_data['Title'])
all_data['Family_size'] = all_data.SibSp + all_data.Parch + 1
all_data.Family_size.unique()
all_data.Family_size.replace(to_replace=[1], value='single', inplace=True)
all_data.Family_size.replace(to_replace=[2, 3], value='small', inplace=True)
all_data.Family_size.replace(to_replace=[4, 5], value='medium', inplace=True)
all_data.Family_size.replace(to_replace=[6, 7, 8, 11], value='large', inplace=True)
get_frequency(all_data['Family_size'])
all_data.Ticket.head()
all_data.Ticket = all_data.Ticket.apply(lambda x: 'N' if x.isdigit() else x.replace('.','').replace('/','').strip().split(' ')[0][0])
get_frequency(all_data['Ticket'])
'''Create a function to count total outliers. And plot variables with and without outliers.'''
def outliers(variable):
    global filtered
    # Calculate 1st, 3rd quartiles and iqr.
    q1, q3 = variable.quantile(0.25), variable.quantile(0.75)
    iqr = q3 - q1
    
    # Calculate lower fence and upper fence for outliers
    l_fence, u_fence = q1 - 1.5*iqr , q3 + 1.5*iqr   # Any values less than l_fence and greater than u_fence are outliers.
    
    # Observations that are outliers
    outliers = variable[(variable<l_fence) | (variable>u_fence)]
    print('Total Outliers of', variable.name,':', outliers.count())
    
    # Drop obsevations that are outliers
    filtered = variable.drop(outliers.index, axis = 0)

    # Create subplots
    out_variables = [variable, filtered]
    out_titles = [' Distribution with Outliers', ' Distribution Without Outliers']
    title_size = 25
    font_size = 18
    plt.figure(figsize = (25, 15))
    for ax, outlier, title in zip(range(1,3), out_variables, out_titles):
        plt.subplot(2, 1, ax)
        sns.boxplot(outlier).set_title('%s' %outlier.name + title, fontsize = title_size)
        plt.xticks(fontsize = font_size)
        plt.xlabel('%s' %outlier.name, fontsize = font_size)
outliers(all_data.Age)
outliers(all_data.Fare)
mn.matrix(all_data)
all_data.isnull().sum()
all_data.Embarked.fillna(value = 'S', inplace = True)
all_data.Fare.fillna(value = all_data.Fare.median(), inplace = True)
correlation = all_data.loc[:, ['Sex', 'Pclass', 'Embarked', 'Title', 'Family_size', 'Parch', 'SibSp', 'Cabin', 'Ticket']]
correlation = correlation.agg(LabelEncoder().fit_transform)
correlation['Age'] = all_data.Age # Inserting Age in variable correlation.
correlation = correlation.set_index('Age').reset_index() # Move Age at index 0.
'''Now create the heatmap correlation.'''
plt.figure(figsize = (20,7))
sns.heatmap(correlation.corr(), cmap ='BrBG', annot = True)
plt.title('Variables Correlated with Age', fontsize = 18)
plt.show()
all_data.Age = all_data.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
all_data.isnull().sum()
all_data.Age.unique()
label_names = ['infant','child','teenager','young_adult','adult','aged']
cut_points = [0,5,12,18,35,60,81]
all_data['Age_binned'] = pd.cut(all_data.Age, cut_points, labels = label_names)
all_data[['Age', 'Age_binned']].head()
groups = ['low','medium','high','very_high']
cut_points = [-1, 130, 260, 390, 520]
all_data['Fare_binned'] = pd.cut(all_data.Fare, cut_points, labels = groups)
all_data[['Fare', 'Fare_binned']].head(2)
all_data.drop(['Name', 'Age', 'Fare'], inplace=True, axis=1)
dummies = pd.get_dummies(all_data["Pclass"], prefix="Pclass", dummy_na=False)
all_data = pd.concat([all_data, dummies], axis=1)
all_data = pd.get_dummies(all_data)
df_train = all_data.iloc[:891, :]
df_test = all_data.iloc[891:, :]
df_train = df_train.drop(['PassengerId'], axis = 1)
df_test = df_test.drop(['Survived'], axis = 1)
X_train = df_train.drop(['Survived', 'Pclass'], axis = 1)
y_train = df_train['Survived']
X_test = df_test.drop(['PassengerId', 'Pclass'], axis=1).copy()
print(y_train.shape)
print(X_train.shape)
print(X_test.shape)
lr = LogisticRegression()
dtc = DecisionTreeClassifier()
knn = KNeighborsClassifier()
sgd = SGDClassifier()
rfc = RandomForestClassifier()
gbc = GradientBoostingClassifier()
cbc = CatBoostClassifier()

train_list = [('LR', lr), ('DTC', dtc), ('KNN', knn), ('SGD', sgd), ('RFC', rfc), ('GBC', gbc), ('Catboost', cbc)]

train_accuracy = pd.DataFrame({'Train_accuracy(%)':[get_model_accuracy(lr, X_train, y_train), get_model_accuracy(dtc, X_train, y_train), get_model_accuracy(knn, X_train, y_train), get_model_accuracy(sgd, X_train, y_train), get_model_accuracy(rfc, X_train, y_train), get_model_accuracy(gbc, X_train, y_train), get_model_accuracy(cbc, X_train, y_train)]})
train_accuracy.index = ['LR', 'DTC', 'KNN', 'SGD', 'RFC', 'GBC', 'Catboost']
sorted_train_accuracy = train_accuracy.sort_values(by = 'Train_accuracy(%)', ascending = False)
print("Without cross-validation")
sorted_train_accuracy
train_accuracy = pd.DataFrame({'Train_accuracy(%)':[get_model_accuracy_cross_val(lr, X_train, y_train), get_model_accuracy_cross_val(dtc, X_train, y_train), get_model_accuracy_cross_val(knn, X_train, y_train), get_model_accuracy_cross_val(sgd, X_train, y_train), get_model_accuracy_cross_val(rfc, X_train, y_train), get_model_accuracy_cross_val(gbc, X_train, y_train), get_model_accuracy_cross_val(cbc, X_train, y_train)]})
train_accuracy.index = ['LR', 'DTC', 'KNN', 'SGD', 'RFC', 'GBC', 'Catboost']
sorted_train_accuracy = train_accuracy.sort_values(by = 'Train_accuracy(%)', ascending = False)
print("With cross-validation")
sorted_train_accuracy
'''Define hyperparameters the logistic regression will be tuned with. For LR, the following hyperparameters are usually tunned.'''
lr_params = {'penalty':['l1', 'l2'],
             'C': np.logspace(0, 4, 10)}

'''For GBC, the following hyperparameters are usually tunned.'''
gbc_params = {'learning_rate': [0.01, 0.02, 0.05, 0.01],
              'max_depth': [4, 6, 8],
              'max_features': [1.0, 0.3, 0.1], 
              'min_samples_split': [ 2, 3, 4],
              'random_state':[43]}

'''For DT, the following hyperparameters are usually tunned.'''
dt_params = {'max_features': ['auto', 'sqrt', 'log2'],
             'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
             'min_samples_leaf':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
             'random_state':[43]}

'''For RF, the following hyperparameters are usually tunned.'''
rf_params = {'criterion':['gini','entropy'],
             'n_estimators':[10, 15, 20, 25, 30],
             'min_samples_leaf':[1, 2, 3],
             'min_samples_split':[3, 4, 5, 6, 7], 
             'max_features':['sqrt', 'auto', 'log2'],
             'random_state':[43]}

'''For KNN, the following hyperparameters are usually tunned.'''
knn_params = {'n_neighbors':[3, 4, 5, 6, 7, 8],
              'leaf_size':[1, 2, 3, 5],
              'weights':['uniform', 'distance'],
              'algorithm':['auto', 'ball_tree','kd_tree','brute']}

'''For CatBoost, the following hyperparameters are ussually tunned'''
cbc_params = {'loss_function': ['Logloss'],
              'iterations': [500, 600, 650],
              'depth': list(range(1, 11)),
              'l2_leaf_reg': list(range(1, 10)),
              'random_seed': [30, 40, 50, 60]}
lr_best_params, lr_best_score = tune_hyperparemeters(lr, params = lr_params)
# cbc_best_params, cbc_best_score = tune_hyperparemeters(cbc, params = cbc_params)
print('CatBoost Best params')
# print(cbc_best_params)
# {'l2_leaf_reg': 4, 'loss_function': 'Logloss', 'iterations': 500, 'depth': 2, 'random_seed': 30}
gbc_best_params, gbc_best_score = tune_hyperparemeters(gbc, params = gbc_params)
print('GBC Best params')
print(gbc_best_params)
# {'random_state': 43, 'max_depth': 4, 'min_samples_split': 2, 'max_features': 0.1, 'learning_rate': 0.05}

dt_best_params, dt_best_score = tune_hyperparemeters(dtc, params = dt_params)

knn_best_params, knn_best_score = tune_hyperparemeters(knn, params = knn_params)

rfc_best_params, rfc_best_score = tune_hyperparemeters(rfc, params = rf_params)

tunned_scores = pd.DataFrame({'Tunned_accuracy(%)': [lr_best_score, gbc_best_score, dt_best_score, knn_best_score, rfc_best_score]})
tunned_scores.index = ['LR', 'GBC', 'DT', 'KNN', 'RFC']
sorted_tunned_scores = tunned_scores.sort_values(by = 'Tunned_accuracy(%)', ascending = False)

sorted_tunned_scores

cbc_best_params = {'l2_leaf_reg': 4, 'loss_function': 'Logloss', 'iterations': 500, 'depth': 2, 'random_seed': 30}
gbc_best_params = {'random_state': 43, 'max_depth': 4, 'min_samples_split': 2, 'max_features': 0.1, 'learning_rate': 0.05}

'''Make prediction using all the trained models'''
gbc = GradientBoostingClassifier(**gbc_best_params)
cbc = CatBoostClassifier(**cbc_best_params)
lr = LogisticRegression(**lr_best_params)
# dtc = DecisionTreeClassifier(**dt_best_params)
# knn = KNeighborsClassifier(**knn_best_params)
rfc = RandomForestClassifier(**rfc_best_params)

'''Scores for GBC'''
gbc.fit(X_train, y_train)
scores = cross_val_score(gbc, X_train, y_train, cv = 10, scoring = 'accuracy')*100
print('Mean Accuracy: %0.4f (+/- %0.4f) [%s]'  % (scores.mean(), scores.std(), 'GBC'))

'''Scores for CatBoost'''
cbc.fit(X_train, y_train)
scores = cross_val_score(cbc, X_train, y_train, cv = 10, scoring = 'accuracy')*100
print('Mean Accuracy: %0.4f (+/- %0.4f) [%s]'  % (scores.mean(), scores.std(), 'CatBoost'))

'''Scores for Logistic Regression'''
lr.fit(X_train, y_train)
scores = cross_val_score(lr, X_train, y_train, cv = 10, scoring = 'accuracy')*100
print('Mean Accuracy: %0.4f (+/- %0.4f) [%s]'  % (scores.mean(), scores.std(), 'LR'))

'''Scores for Random Forest'''
rfc.fit(X_train, y_train)
scores = cross_val_score(rfc, X_train, y_train, cv = 10, scoring = 'accuracy')*100
print('Mean Accuracy: %0.4f (+/- %0.4f) [%s]'  % (scores.mean(), scores.std(), 'RFC'))
model_prediction = pd.DataFrame({'CatBoost': cbc.predict(X_test), 'GBC': gbc.predict(X_test), 'RFC': rfc.predict(X_test), 'LR': lr.predict(X_test)})
print(model_prediction.head())
(pd.Series(gbc.feature_importances_, index=X_test.columns).nlargest(10).plot(kind='barh'))
plot_feature_importance(gbc, 'GBC Feature Importance')
'''Create a function that returns learning curves for different classifiers.'''
def plot_learning_curve(model):
    from sklearn.model_selection import learning_curve
    # Create feature matrix and target vector
    X, y = X_train, y_train
    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv = 10,
                                                    scoring='accuracy', n_jobs = -1, 
                                                    train_sizes = np.linspace(0.01, 1.0, 17), random_state = 43)
                                                    # 17 different sizes of the training set

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis = 1)
    train_std = np.std(train_scores, axis = 1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis = 1)
    test_std = np.std(test_scores, axis = 1)

    # Draw lines
    plt.plot(train_sizes, train_mean, 'o-', color = 'red',  label = 'Training score')
    plt.plot(train_sizes, test_mean, 'o-', color = 'green', label = 'Cross-validation score')
    
    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha = 0.1, color = 'r') # Alpha controls band transparency.
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha = 0.1, color = 'g')

    # Create plot
    font_size = 15
    plt.xlabel('Training Set Size', fontsize = font_size)
    plt.ylabel('Accuracy Score', fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.legend(loc = 'best')
    plt.grid()
'''Now plot learning curves of the optimized models in subplots.'''
plt.figure(figsize = (25,25))
lc_models = [gbc, cbc, lr, rfc]
lc_labels = ['GBC', 'CatBoost', 'LR', 'RFC']

for ax, models, labels in zip (range(1,9), lc_models, lc_labels):
    plt.subplot(4,2,ax)
    plot_learning_curve(models)
    plt.title(labels, fontsize = 18)
plt.suptitle('Learning Curves of Optimized Models', fontsize = 28)
plt.tight_layout(rect = [0, 0.03, 1, 0.97])
'''Return prediction to use it in another function.'''
def x_val_predict(model):
    from sklearn.model_selection import cross_val_predict
    predicted = cross_val_predict(model, X_train, y_train, cv = 10)
    return predicted # Now we can use it in another function by assigning the function to its return value.

'''Function to return confusion matrix.'''
def confusion_matrix(model):
    predicted = x_val_predict(model)
    confusion_matrix = pd.crosstab(y_train, predicted, rownames = ['Actual'], colnames = ['Predicted/Classified'], margins = True) # We use pandas crosstab
    return display(confusion_matrix)
confusion_matrix(gbc)
confusion_matrix(cbc)
confusion_matrix(lr)
confusion_matrix(rfc)
'''Function to calculate precision score.'''
def precision_score(model):
    from sklearn.metrics import precision_score
    predicted = x_val_predict(model)
    precision_score = precision_score(y_train, predicted)
    return np.round(precision_score*100, 2)

'''Compute precision score.'''
print('RFC  Precision Score:', precision_score(rfc))
print('GBC Precision Score:', precision_score(gbc))
print('CatBoost Precision Score:', precision_score(cbc))
print('Logistic Regression Precision Score:', precision_score(lr))
'''Function to calculate recall score.'''
def recall_score(model):
    from sklearn.metrics import recall_score
    predicted = x_val_predict(model)
    recall_score = recall_score(y_train, predicted)
    return np.round(recall_score*100, 2)

'''Compute recall score.'''
print('RFC  Recall Score:', recall_score(rfc))
print('GBC Recall Score:', recall_score(gbc))
print('Logistic Regression Recall Score:', recall_score(lr))
print('CatBoost Recall Score:', recall_score(cbc))
'''Function for specificity score.'''
def specificity_score(model):
    from sklearn.metrics import confusion_matrix
    predicted = x_val_predict(model)
    tn, fp, fn, tp = confusion_matrix(y_train, predicted).ravel()
    specificity_score = tn / (tn + fp)
    return np.round(specificity_score*100, 2)

'''Calculate specificity score.'''
print('RF  Specificity Score:', specificity_score(rfc))
print('GBC Specificity Score:', specificity_score(gbc))
print('Logistic Regression Specificity Score:', specificity_score(lr))
print('CatBoost Specificity Score:', specificity_score(cbc))
'''Function to compute classification report.'''
def classification_report(model):
    from sklearn.metrics import classification_report
    predicted = x_val_predict(model)
    classification_report = classification_report(y_train, predicted)
    return print(classification_report)

'''Now calculate classification report.'''

print('RF classification report:')
classification_report(rfc)
print('GBC classification report:')
classification_report(gbc)
print('Logistic Regression classification report:')
classification_report(lr)
print('CatBoost classification report:')
classification_report(cbc)

'''#7Function for plotting precision-recall vs threshold curve.'''
def precision_recall_vs_threshold(model, title):
    from sklearn.metrics import precision_recall_curve
    probablity = model.predict_proba(X_train)[:, 1]
    plt.figure(figsize = (18, 5))
    precision, recall, threshold = precision_recall_curve(y_train, probablity)
    plt.plot(threshold, precision[:-1], 'b-', label = 'precision', lw = 3.7)
    plt.plot(threshold, recall[:-1], 'g', label = 'recall', lw = 3.7)
    plt.xlabel('Threshold')
    plt.legend(loc = 'best')
    plt.ylim([0, 1])
    plt.title(title)
    plt.show()

'''Now plot precision-recall vs threshold curve for rf and gbc.'''
precision_recall_vs_threshold(rfc, title = 'RF Precision-Recall vs Threshold Curve' )
precision_recall_vs_threshold(gbc, title = 'GBC Precision-Recall vs Threshold Curve')
'''Function to plot ROC curve with AUC score.'''
def plot_roc_and_auc_score(model, title):
    from sklearn.metrics import roc_curve, roc_auc_score
    probablity = model.predict_proba(X_train)[:, 1]
    plt.figure(figsize = (18, 5))
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_train, probablity)
    auc_score = roc_auc_score(y_train, probablity)
    plt.plot(false_positive_rate, true_positive_rate, label = "ROC CURVE, AREA = "+ str(auc_score))
    plt.plot([0, 1], [0, 1], 'red', lw = 3.7)
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.axis([0, 1, 0, 1])
    plt.legend(loc = 4)
    plt.title(title)
    plt.show()

'''Plot roc curve and auc score for rf and gbc.'''
plot_roc_and_auc_score(rfc, title = 'RF ROC Curve with AUC Score')
plot_roc_and_auc_score(gbc, title = 'GBC ROC Curve with AUC Score')
'''Submission with the most accurate random forest classifier.'''
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": rfc.predict(X_test)}, dtype=np.int32)
submission.to_csv('submission_rf.csv', index = False)
# submission.head()

'''Submission with the most accurate gradient boosting classifier.'''
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": gbc.predict(X_test)}, dtype=np.int32)
submission.to_csv('submission_gbc.csv', index = False)
# submission.head()

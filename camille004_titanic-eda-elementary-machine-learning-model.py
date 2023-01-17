import pandas as pd

import plotly.graph_objects as go

import plotly.express as px

import plotly.figure_factory as ff

import numpy as np



from plotly.subplots import make_subplots

from tqdm import tqdm



from sklearn.preprocessing import StandardScaler



from sklearn.decomposition import PCA



from sklearn.feature_selection import VarianceThreshold



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



train_df = pd.read_csv("../input/titanic/train.csv")

train = train_df.copy()

family_column = train['SibSp'] + train['Parch']

train['Family'] = family_column

train = train[['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Family', 'Embarked', 'Fare']]



# Account for missingness

train['Age'] = train['Age'].interpolate()

train['Fare'] = train['Fare'].interpolate()



train.head(5)
train.describe()
print(str(round(np.mean(train['Survived']) * 100)) + "% of the passengers on the RMS Titanic survived.\n")

print(str(round((sum((train[train['Sex'] == 'female'])['Survived']) / sum(train['Survived'])) * 100)) + "% of the survivors were female.\n")

print(str(round((sum((train[train['Pclass'] == 1])['Survived']) / sum(train['Survived'])) * 100)) + "% of the survivors were first class.")

print(str(round((sum((train[train['Pclass'] == 2])['Survived']) / sum(train['Survived'])) * 100)) + "% of the survivors were second class.")

print(str(round((sum((train[train['Pclass'] == 3])['Survived']) / sum(train['Survived'])) * 100)) + "% of the survivors were third class.\n")

print(str(round((sum((train[train['Age'] <= 20])['Survived']) / sum(train['Survived'])) * 100)) + "% of the survivors were 20 or younger.")

print(str(round((sum((train[(train['Age'] > 20) & (train['Age'] < 50)])['Survived']) / sum(train['Survived'])) * 100)) + "% of the survivors were between 20 and 50.")

print(str(round((sum((train[train['Age'] >= 50])['Survived']) / sum(train['Survived'])) * 100)) + "% of the survivors were 50 or older.\n")

print(str(round((sum((train[train['Family'] == 0])['Survived']) / sum(train['Survived'])) * 100)) + "% of the survivors had no family members aboard.")

print(str(round((sum((train[train['Family'] >= 3])['Survived']) / sum(train['Survived'])) * 100)) + "% of the survivors had three or more family members aboard.\n")

print(str(round((sum((train[train['Embarked'] == 'S'])['Survived']) / sum(train['Survived'])) * 100)) + "% of the survivors embarked from Southampton.")

print(str(round((sum((train[train['Embarked'] == 'C'])['Survived']) / sum(train['Survived'])) * 100)) + "% of the survivors embarked from Cherbourg.")

print(str(round((sum((train[train['Embarked'] == 'Q'])['Survived']) / sum(train['Survived'])) * 100)) + "% of the survivors embarked from Queenstown.")
def get_title(name):

    if '.' in name:

        title = name.split(',')[1].split('.')[0].strip()

    else:

        title = 'None'

        

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        title = 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady']:

        title = 'Mrs'

    elif title in ['Mlle', 'Ms']:

        title = 'Miss'

    elif title == 'Dr':

        row = train.loc[train['Name'] == name].index[0]

        if train.iloc[row]['Sex'] == 'male':

            title = 'Mr'

        else:

            title = 'Mrs'

    return title



titles = train['Name'].map(lambda x: get_title(x))

train['Title'] = titles

train.head(5)
survivors = train[train['Survived'] == 1]

female_survivors = survivors[survivors['Sex'] == 'female']

male_survivors = survivors[survivors['Sex'] == 'male']

classes = ['First Class', 'Second Class', 'Third Class']

female_classes = female_survivors['Pclass'].value_counts(sort=False, normalize=True).to_list()

male_classes = male_survivors['Pclass'].value_counts(sort=False, normalize=True).to_list()

fig = go.Figure(data=[

    go.Bar(name='Female', x=classes, y=female_classes),

    go.Bar(name='Male', x=classes, y=male_classes)])

fig.update_layout(barmode='stack', width=400, height=400, title="Class and Sex of Survivors Ratios")

fig.show()
s_port = survivors[survivors['Embarked'] == 'S']

c_port = survivors[survivors['Embarked'] == 'C']

q_port = survivors[survivors['Embarked'] == 'Q']



s_classes = s_port['Pclass'].value_counts(sort=False, normalize=True).to_list()

c_classes = c_port['Pclass'].value_counts(sort=False, normalize=True).to_list()

q_classes = q_port['Pclass'].value_counts(sort=False, normalize=True).to_list()



fig = go.Figure(data=[

    go.Bar(name='Southampton', x=classes, y=s_classes),

    go.Bar(name='Cherbourg', x=classes, y=c_classes),

    go.Bar(name='Queenstown', x=classes, y=q_classes)])

fig.update_layout(barmode='stack', width=450, height=400, title="Class and Embarking Port of Survivors Ratios")

fig.show()
title_counts_survived = train[train['Survived'] == 1]['Title'].value_counts()

title_counts_dead = train[train['Survived'] == 0]['Title'].value_counts()

titles = list(title_counts_survived.index)



del train['Name']



fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]])



fig.add_trace(

    go.Pie(labels=titles, values=title_counts_survived, title='Survivor Titles'),

    row=1, col=1,

)

fig.add_trace(

    go.Pie(labels=titles, values=title_counts_dead, title='Non-Survivor Titles'),

    row=1, col=2

)



fig.update_layout(width=600, height=400, title_text="Titles of Survivors vs. Non-Survivors")

fig.show()

fig = px.histogram(train, x='Age', y='Survived', color='Survived', marginal='box', opacity=0.75, 

                   hover_data=train.columns, title='Ages of Survived and Dead Groups')

fig.update_layout(width=700, height=400)

fig.show()
fig = px.histogram(train, x='Survived', y='Family', color='Survived', marginal='box', opacity=0.75, 

                   hover_data=train.columns, orientation='h', title='Number of Family Members Aboard for Survived and Dead Groups')

fig.update_layout(width=700, height=400)

fig.show()
fig = px.histogram(train, x='Fare', y='Survived', color='Survived', marginal='box', opacity=0.75,

                  hover_data=train.columns, title='Fare Distribution Among Survivors and Non-Survivors')

fig.update_layout(width=700, height=400)

fig.show()
titanic_dummies = pd.get_dummies(train, columns=['Pclass', 'Sex', 'Embarked', 'Title'], prefix=['Class', 'Sex', 'Port', 'Title'])

titanic_dummies.head(5)
titanic_dummies[['Age', 'Family', 'Fare']] = titanic_dummies[['Age', 'Family', 'Fare']].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

titanic_dummies
sel = VarianceThreshold(threshold=0.8 * (1 - 0.8))

sel.fit_transform(titanic_dummies)

fitted = titanic_dummies[titanic_dummies.columns[sel.get_support(indices=True)]]

fitted.head(5)
print('Original DF shape vs feature-selected DF shape: ' + str(titanic_dummies.shape) + ', ' + str(fitted.shape))
SVC_classifier = SVC(kernel='linear')

features = fitted[fitted.columns[1:]]

label = fitted[fitted.columns[0]]



X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size=0.2)

SVC_classifier.fit(X_train, Y_train)
y_pred = SVC_classifier.predict(X_test)

y_pred
def cross_val(model, X_test, Y_test, cv):

    cross_val_scores = cross_val_score(model, X_test, Y_test, cv=cv)

    print("10-Fold Cross Validation Scores: " + str(list(cross_val_scores)))

    print("Accuracy: %0.2f (+/- %0.2f)" % (cross_val_scores.mean(), cross_val_scores.std() * 2))

    

cross_val(SVC_classifier, X_test, Y_test, 10)
tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()

print((tn, fp, fn, tp))



def plot_confusion_matrix(Y_true, Y_pred):

    cm = list(confusion_matrix(Y_true, Y_pred))

    x = ['Pred. Not Survived', 'Pred. Survived']

    y = ['Not Survived', 'Survived']

    cm_text = [['TN', 'FP'], ['FN', 'TP']]

    fig = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=cm_text, colorscale="aggrnyl")

    fig.update_layout(title="Confusion Matrix", width=400, height=400)

    fig.show()



plot_confusion_matrix(Y_test, y_pred)
print("Model accuracy score from Conf. Matrix: " + str((tp + tn) / float(tp + tn + fp + fn)))

print("True accuracy score: " + str(accuracy_score(Y_test, y_pred)))
sensitivity = tp / float(fn + tp) # These are all positive.



print("Recall score: " + str(sensitivity))
specificity = tn / float(tn + fp) # These are all negative.



print("Specificty score: " + str(specificity))
optimal_k = int(round(np.sqrt(len(X_train))))



neigh = KNeighborsClassifier(n_neighbors=optimal_k)

neigh.fit(X_train, Y_train)



y_pred_knn = neigh.predict(X_test)

y_pred_knn
cross_val(neigh, X_test, Y_test, 10)
plot_confusion_matrix(Y_test, y_pred_knn)
def get_confusion_metrics(Y_true, Y_pred):

    tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred).ravel()

    print("Model accuracy score from Conf. Matrix: " + str((tp + tn) / float(tp + tn + fp + fn)))

    print("True accuracy score: " + str(accuracy_score(Y_true, Y_pred)))

    

    sensitivity = tp / float(fn + tp)

    print("Recall score: " + str(sensitivity))

    

    specificity = tn / float(tn + fp)

    print("Specificty score: " + str(specificity))

    

get_confusion_metrics(Y_test, y_pred_knn)
neigh3 = KNeighborsClassifier(n_neighbors=3)

neigh3.fit(X_train, Y_train)



y_pred_knn3 = neigh3.predict(X_test)

y_pred_knn3
cross_val(neigh3, X_test, Y_test, 10)
get_confusion_metrics(Y_test, y_pred_knn3)
features_list = list(fitted.columns[1:])

vals = fitted.loc[:, features_list].values

vals = StandardScaler().fit_transform(vals)



# Now each value in the feature dataset is standardized!

vals
standardized_fitted = pd.DataFrame(vals, columns=features_list)

standardized_fitted.head(5)
pca = PCA().fit(standardized_fitted)



xi = np.arange(1, 9, step=1)

y = np.cumsum(pca.explained_variance_ratio_)



fig = px.line(x=xi, y=y,)

fig.add_trace(

    go.Scatter(

        mode='markers',

        x=[5],

        y=[0.985374],

        marker=dict(

            color='red',

            size=10,

            opacity=0.5

        ),

        showlegend=False

    )

)

fig.update_layout(width=600, height=400, xaxis_title='# Components', yaxis_title='PC Variance for Whole Dataset', title='PCA Explained Variance Ratio for Fitted Data')

fig.show()
pca_titanic = PCA(n_components=7)

principal_components_titanic = pca_titanic.fit_transform(standardized_fitted)



pc_columns = ['PC' + str(i + 1) for i in range(7)]



principal_components_df = pd.DataFrame(data=principal_components_titanic,

                                      columns=pc_columns)

principal_components_df.insert(0, 'Survived', label)

principal_components_df
print('Explained variation per principal component: {}'.format(pca_titanic.explained_variance_ratio_))
X_train_pca = pca_titanic.fit_transform(X_train)

X_test_pca = pca_titanic.fit_transform(X_test)



neigh_pca = KNeighborsClassifier(n_neighbors=optimal_k)

neigh_pca.fit(X_train_pca, Y_train)



y_pred_knn_pca = neigh_pca.predict(X_test_pca)

y_pred_knn_pca

cross_val(neigh_pca, X_test_pca, Y_test, 10)
get_confusion_metrics(Y_test, y_pred_knn_pca)
param_grid = [{'kernel': ['poly'], 'C': [0.01, 0.1, 1, 10, 100, 1000], 'degree': [2, 3, 4]},

             {'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [1e-3, 1e-4, 1e-6, 1e-8]},

             {'kernel': ['sigmoid'], 'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [1e-3, 1e-4]}]



clf = GridSearchCV(SVC(), param_grid, scoring='recall')

clf.fit(X_train, Y_train)

print(clf.best_params_)
# These were the best params from our grid search.

svc_nl = SVC(kernel='poly', degree=3, C=0.01)

svc_nl.fit(X_train, Y_train)
y_pred_svc_nl = svc_nl.predict(X_test)

y_pred_svc_nl
plot_confusion_matrix(Y_test, y_pred_svc_nl)
get_confusion_metrics(Y_test, y_pred_svc_nl)
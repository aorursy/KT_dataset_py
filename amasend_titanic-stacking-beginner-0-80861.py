import warnings
warnings.filterwarnings('ignore')
import numpy
from matplotlib import pyplot
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
import pandas as pd
import sklearn as sk
from xgboost import XGBClassifier
from sklearn.preprocessing.imputation import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
%matplotlib inline
dataset = pd.read_csv('../input/train.csv')
validation_dataset = pd.read_csv('../input/test.csv')
dataset.head(5)
# helper function to map one-hot encoding
def try_iter(x):
    try:
        return sum([ord(z) for z in x])
    except TypeError:
        return float('NaN')
# feature engineering
# extract person status (Mr., Ms., Dr.... etc.)
status = [name.split(',')[1].split('.')[0].split()[-1] for name in dataset['Name'].values.tolist()]
# extract person Cabin Letter (only first class ticket has separate cabin)
room = []
for letter in dataset['Cabin']:
    if isinstance(letter, str):
        room.append(letter[0])
    else:
        room.append('Unknown')
# count family members
family_members = dataset['SibSp'] + dataset['Parch']

# drop columns that cannot be processed
data = dataset.drop(['Name', 'Ticket', 'Cabin'], axis=1)

data['Status'] = status
data['Room'] = room
data['Family_members'] = family_members

# change data in object columns
# two methods
# Map string values to equivalent integer values
for col in data.columns.values.tolist():
    if data[col].dtypes == 'O':
        data[col] = list(map(lambda x: try_iter(x), data[col].values.tolist()))
columns = data.columns[2:].tolist()
data = data[columns + ['Survived']]
        
# Create additional columns containing true or false (1 or 0) in rows where that value was
# one-hot encoding (cathegorical data)
## data = pd.get_dummies(data)

# split training data into training and validation datasets
# this step is important, avoid fitting whole training dataset anywhere and then splitting into train and valid
array = data.values
X = array[:, 0:-1].astype(float)
Y = array[:, -1]
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
X_train = pd.DataFrame(data=X_train, columns=columns)
X_validation = pd.DataFrame(data=X_validation, columns=columns)

# handling missing values (NaN, Null)
# creates additonal new columns based on calumns where missing data was (fill those columns with 1 and 0)
# True where missing value was, False where not (1 or 0)
missing_columns = [col for col in X_train.columns if X_train[col].isnull().any()]
for col in missing_columns:
    X_train[col + '_missing_data'] = X_train[col].isnull()
original_data = X_train
# fill missing values with mean values
imputer = Imputer()
X_train = pd.DataFrame(data=imputer.fit_transform(X_train))
X_train.columns = original_data.columns
# make one column indicating where wasmissing point, drop missing_columns
X_train['missing_values'] = numpy.zeros((len(X_train),1))
for col in missing_columns:
    X_train['missing_values'] += X_train[col + '_missing_data']
    X_train = X_train.drop([col + '_missing_data'], axis=1)
X_train['Age'] = X_train['Age'].values.round()
X_train = X_train.values

# validation dataset
missing_columns = [col for col in X_validation.columns if X_validation[col].isnull().any()]
for col in missing_columns:
    X_validation[col + '_missing_data'] = X_validation[col].isnull()
original_data = X_validation
# fill missing values with mean values
imputer = Imputer()
X_validation = pd.DataFrame(data=imputer.fit_transform(X_validation))
X_validation.columns = original_data.columns
# make one column indicating where wasmissing point, drop missing_columns
X_validation['missing_values'] = numpy.zeros((len(X_validation),1))
for col in missing_columns:
    X_validation['missing_values'] += X_validation[col + '_missing_data']
    X_validation = X_validation.drop([col + '_missing_data'], axis=1)
X_validation['Age'] = X_validation['Age'].values.round()
X_validation = X_validation.values
# feature engineering
# extract person status (Mr., Ms., Dr.... etc.)
status = [name.split(',')[1].split('.')[0].split()[-1] for name in validation_dataset['Name'].values.tolist()]
# extract person Cabin Letter (only first class ticket has separate cabin)
room = []
for letter in validation_dataset['Cabin']:
    if isinstance(letter, str):
        room.append(letter[0])
    else:
        room.append('Unknown')
# count family members
family_members = validation_dataset['SibSp'] + validation_dataset['Parch'] 

test_data = validation_dataset.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_data['Status'] = status
test_data['Room'] = room
test_data['Family_members'] = family_members

for col in test_data.columns.values.tolist():
    if test_data[col].dtypes == 'O':
        test_data[col] = list(map(lambda x: try_iter(x), test_data[col].values.tolist()))
## test_data = pd.get_dummies(test_data)

# only for further result saving
pass_id = test_data[['PassengerId']]

original_data = test_data
missing_columns = [col for col in test_data.columns if test_data[col].isnull().any()]
for col in missing_columns:
    test_data[col + '_missing_data'] = test_data[col].isnull()
imputer = Imputer()
test_data = pd.DataFrame(imputer.fit_transform(test_data))
test_data.columns = original_data.columns
test_data['missing_values'] = numpy.zeros((len(test_data),1))
for col in missing_columns:
    test_data['missing_values'] += test_data[col + '_missing_data']
    test_data = test_data.drop([col + '_missing_data'], axis=1)
columns = test_data.columns[1::].tolist()
test_data = test_data[columns]
test_data['Age'] = test_data['Age'].values.round()
test_data.head(10)
test_X = test_data.values[:, :].astype(float)
num_fold = 10
seed = 7
scoring = 'accuracy'
pipelines = []
# you can choose which type of algorithms you want to use,
# better to check correlation between their predictions later and modify your choice
# if we are using all algorithms we can score 79% to 80% accuracy
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()), ('SVM', SVC())])))
pipelines.append(('AB', AdaBoostClassifier()))
pipelines.append(('GB', GradientBoostingClassifier()))
pipelines.append(('RF', RandomForestClassifier()))
pipelines.append(('ET', ExtraTreesClassifier()))
pipelines.append(('XGB', XGBClassifier()))
results = []
names = []
# k-fld test, produce extimated score per algorithm based on training and validation datasets
for name, model in pipelines:
    kfold = KFold(n_splits=num_fold, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "{}: {} ({})".format(name, cv_results.mean(), cv_results.std())
    print(msg)
# usage of 25,50,75 percentile and min/max values from each distribution of k-fold result per algorithm
fig = pyplot.figure(figsize=(15,6))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
importance = []
plots = []
new_train_data = pd.DataFrame(data=Y_validation, columns=['Survived'])
new_test_data = pd.DataFrame(data=test_data['Age'], columns=['Age'])
# scaling is only needed when we are using based algorithms
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
for name, model in pipelines:
    if 'Scaled' not in name:
        model.fit(X_train, Y_train)
        # stores feature importances per model
        importance.append((name, model.feature_importances_))
        # plot partial dependencies for GradientBoosting (in this version of sklearn GB only available)
        # we can look at particular feature dependencies
        if name == 'GB':
            plot_partial_dependence(model, X=X_train, features=[2], feature_names=data.columns.values.tolist(), grid_resolution=10)
        predictions = model.predict(X_validation)
        new_train_data[name] = predictions

        predictions = model.predict(test_X)
        new_test_data[name] = predictions
    else:
        model.fit(rescaledX, Y_train)
        # stores feature importances per model
        try:
            importance.append((name, model.steps[1][1].feature_importances_))
        except:
            pass
        rescaledValidX = scaler.transform(X_validation)
        predictions = model.predict(rescaledValidX)
        new_train_data[name] = predictions

        rescaledTestX = scaler.transform(test_X)
        predictions = model.predict(rescaledTestX)
        new_test_data[name] = predictions
new_test_data = new_test_data.drop(['Age'], axis=1)
cols = test_data.columns.values
# Create a dataframe with features
algs = {'features': cols}
for i, imp in enumerate(importance):
    algs['{} feature importances'.format(imp[0])] = imp[1]
feature_dataframe = pd.DataFrame(algs)
for model in importance:
    # Scatter plot 
    trace = go.Scatter(
        y = feature_dataframe['{} feature importances'.format(model[0])].values,
        x = feature_dataframe['features'].values,
        mode='markers',
        marker=dict(
            sizemode = 'diameter',
            sizeref = 1,
            size = 25,
            color = feature_dataframe['{} feature importances'.format(model[0])].values,
            colorscale='Portland',
            showscale=True
        ),
        text = feature_dataframe['features'].values
    )
    data = [trace]

    layout= go.Layout(
        autosize= True,
        title= '{} feature importances'.format(model[0]),
        hovermode= 'closest',
        yaxis=dict(
            title= 'Feature Importance',
            ticklen= 5,
            gridwidth= 2
        ),
        showlegend= False
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig,filename='scatter2010')
# calculate mean value of each particular feature importance
feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise
feature_dataframe.head(3)
y = feature_dataframe['mean'].values
x = feature_dataframe['features'].values
data = [go.Bar(
            x= x,
             y= y,
            width = 0.5,
            marker=dict(
               color = feature_dataframe['mean'].values,
            colorscale='Portland',
            showscale=True,
            reversescale = False
            ),
            opacity=0.6
        )]

layout= go.Layout(
    autosize= True,
    title= 'Barplots of Mean Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='bar-direct-labels')
data = [
    go.Heatmap(
        z= new_train_data[new_train_data.columns.values.tolist()[1:]].astype(float).corr().values ,
        x= new_train_data.columns.values.tolist()[1:],
        y= new_train_data.columns.values.tolist()[1:],
          colorscale='Viridis',
            showscale=True,
            reversescale = True
    )
]
py.iplot(data, filename='labelled-heatmap')
array = new_train_data.values
X = array[:, 0:-1].astype(float)
Y = array[:, -1]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
model = RandomForestClassifier()
model.fit(rescaledX, Y)
val = model.feature_importances_
t_X = new_test_data.values[:, :].astype(float)
rescaledTestX = scaler.transform(t_X)
predictions = model.predict(rescaledTestX)
cols = new_train_data.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
                                   'Random Forest feature importances': val[1]
                                  })
# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Random Forest feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest feature importances',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')
temp_p = []
for val in predictions:
    temp_p.append(int(val))
predictionsDF = pd.DataFrame({'Survived': temp_p})
temp = []
for val in pass_id.values:
    temp.append(val[0])
pass_idDF = pd.DataFrame({'PassengerId': temp})
result = pd.concat([pass_idDF, predictionsDF], axis=1)
print(result)
result.to_csv('result.csv', index=False)
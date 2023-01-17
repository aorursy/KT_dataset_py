# for data manipulation
import pandas as pd
import numpy as np

# for plotting/visualising the distibution of data
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly import tools

import random
import re

# for pre-processing of the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import warnings
# load the data
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df.head()
# Get the distribution of the data
train_df.describe(include='all')
train_df['Survived'].astype(int).plot.hist();
# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
# Missing values statistics
missing_values = missing_values_table(train_df)
missing_values
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
train_df['Title'] = train_df['Name'].apply(get_title)
test_df['Title'] = test_df['Name'].apply(get_title)
train_df['Title'].value_counts()
test_df['Title'].value_counts()
dict1 = {'Dr':'Others', 'Rev':'Others', 'Col':'Others', 'Mlle':'Others', 'Major':'Others', 'Capt':'Others', 'Ms':'Others', 
         'Don':'Others', 'Lady':'Others', 'Countess':'Others', 'Jonkheer':'Others', 'Mme':'Others', 'Sir':'Others'}
train_df['Title'] = train_df['Title'].replace(dict1)

dict2 = {'Col':'Others', 'Rev':'Others', 'Ms':'Others', 'Dona':'Others', 'Dr':'Others'}
test_df['Title'] = test_df['Title'].replace(dict2)
train_df['Title'].value_counts()
# Checking the incorrect entry when age is less than 13 for male and the title is Mr.
df = train_df.loc[train_df['Title']=='Mr']
df = df.loc[df['Age']<13]
df.head()
# Correcting the entry
train_df.loc[[731],['Title']] = 'Master'
# Function to plot the classes of the variables
def random_color_generator(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color

def get_percent(df, temp_col, width=800, height=500):
    cnt_srs = df[[temp_col, 'Survived']].groupby([temp_col], as_index=False).mean().sort_values(by=temp_col)

    trace = go.Bar(
        x = cnt_srs[temp_col].values[::-1],
        y = cnt_srs['Survived'].values[::-1],
        text = cnt_srs.values[::-1],
        textposition = 'auto',
        name = "Percent",
        textfont = dict(
            size=12,
            color='rgb(0, 0, 0)'
        ),
        orientation = 'v',
            marker = dict(
                color = random_color_generator(100),
                line=dict(color='rgb(8,48,107)',
                  width=1.5,)
            ),
            opacity = 0.7,
    )    
    return trace

def get_count(df, temp_col, width=800, height=500):
    cnt_srs = df[temp_col].value_counts().sort_index()

    trace = go.Bar(
        x = cnt_srs.index[::-1],
        y = cnt_srs.values[::-1],
        text = cnt_srs.values[::-1],
        textposition = 'auto',
        textfont = dict(
            size=12,
            color='rgb(0, 0, 0)'
        ),
        name = 'Count',
        orientation = 'v',
            marker = dict(
                color = random_color_generator(100),
                line=dict(color='rgb(8,48,107)',
                  width=1.5,)
            ),
            opacity = 0.7,
    )    
    return trace

def plot_count_percent_for_object(df, temp_col, height=400):
    trace1 = get_count(df, temp_col)
    trace2 = get_percent(df, temp_col)

    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Count', 'Percent'), print_grid=False)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)

    fig['layout']['yaxis1'].update(title='Count')
    fig['layout']['yaxis2'].update(range=[0, 1], title='% Survived')
    fig['layout'].update(title = temp_col, margin=dict(l=100), width=800, height=height, showlegend=False)

    py.iplot(fig)
# observe the distribution of title
warnings.simplefilter('ignore')
temp_col = train_df.columns.values[12]
plot_count_percent_for_object(train_df, temp_col)
# observe the distribution of Sex
temp_col = train_df.columns.values[4]
plot_count_percent_for_object(train_df, temp_col)
# Making a new variable/feature 'family'
train_df['family'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

test_df['family'] = test_df['SibSp'] + test_df['Parch'] + 1
test_df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
# observe the distribution of family
temp_col = train_df.columns.values[11]
plot_count_percent_for_object(train_df, temp_col)
# Making a new variable/feature 'family_status' from the variable 'family' 
train_df['family_status'] = train_df['family']
test_df['family_status'] = test_df['family']
dict2 = {1:'Alone', 2:'NotAlone', 3:'NotAlone', 4:'NotAlone', 5:'NotAlone', 6:'NotAlone', 7:'NotAlone', 8:'NotAlone', 
         9:'NotAlone', 10:'NotAlone', 11:'NotAlone'}
train_df['family_status'] = train_df['family_status'].replace(dict2)
test_df['family_status'] = test_df['family_status'].replace(dict2)
train_df['family_status'].dtype
# observe the distribution of family_status
temp_col = train_df.columns.values[12]
plot_count_percent_for_object(train_df, temp_col)
plt.figure(figsize = (10, 8))

df = train_df[['Survived', 'Age']]
df = df.dropna()

# KDE plot of passengers who did not survive 
sns.kdeplot(df.loc[df['Survived'] == 0, 'Age'], label = 'survived == 0')

# KDE plot of passengers who survived
sns.kdeplot(df.loc[df['Survived'] == 1, 'Age'], label = 'survived == 1')

# Labeling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Age');
# dividing the age variable into different classes
train_df['Age'] = train_df['Age'].fillna(200) # this is just indicating the missing values
train_df['Age'] = pd.cut(train_df['Age'], bins=[0,12,40,80, 250], labels = ['Child', 'Young', 'Old', 'Missing'])
train_df['Age'] = train_df['Age'].astype('O')
temp_col = train_df.columns.values[5]
plot_count_percent_for_object(train_df, temp_col)
age_train = pd.read_csv('../input/train.csv')
age_test = pd.read_csv('../input/test.csv')
train_df['Age'] = age_train['Age']
test_df['Age'] = age_test['Age']

# mean of the age variable
age_avg_train = train_df['Age'].mean()
age_avg_test = test_df['Age'].mean()
# standard deviation of the age variable 
age_std_train = train_df['Age'].std()
age_std_test = test_df['Age'].std()

age_null_count_train = train_df['Age'].isnull().sum()
age_null_count_test = test_df['Age'].isnull().sum()

# list of the random age values to be filled based on the distribution of the original age variable  
age_null_random_list_train = np.random.randint(age_avg_train - age_std_train, age_avg_train + age_std_train, size=age_null_count_train)
age_null_random_list_test = np.random.randint(age_avg_test - age_std_test, age_avg_test + age_std_test, size=age_null_count_test)

train_df['Age'][np.isnan(train_df['Age'])] = age_null_random_list_train
test_df['Age'][np.isnan(test_df['Age'])] = age_null_random_list_test
# dividing the age variable into different classes
train_df['Age'] = pd.cut(train_df['Age'], bins=[0,12,40,80], labels = ['Child', 'Young', 'Old'])
test_df['Age'] = pd.cut(test_df['Age'], bins=[0,12,40,80], labels = ['Child', 'Young', 'Old'])
train_df['Age'] = train_df['Age'].astype('O')
temp_col = train_df.columns.values[5]
plot_count_percent_for_object(train_df, temp_col)
train_df.head()
drop = ['Name', 'PassengerId', 'Ticket', 'Cabin']
train_df = train_df.drop(drop, axis=1)
test_df = test_df.drop(drop, axis=1)
# Create a label encoder object
encoder = LabelEncoder()
encoder_count = 0

# Iterate through the columns
for col in train_df:
    if train_df[col].dtype == 'object':
        # If 2 unique classes
        if len(list(train_df[col].unique())) <= 2:
            encoder.fit(train_df[col])
            train_df[col] = encoder.transform(train_df[col])
            test_df[col] = encoder.transform(test_df[col])
            # Keep track of how many columns were label encoded
            encoder_count += 1
            
print('%d columns were label encoded.' % encoder_count)
dict1 = {'Child':1, 'Young':2, 'Old':3}
train_df['Age'] = train_df['Age'].replace(dict1)
test_df['Age'] = test_df['Age'].replace(dict1)
# one-hot encoding of categorical variables
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)
train_df.head()
test_df.head()
train = train_df
test = test_df

print(train.shape)
print(test.shape)
y = train['Survived']
x = train.drop('Survived', axis=1)
from sklearn.model_selection import cross_validate, ShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn import ensemble, naive_bayes, svm, tree, discriminant_analysis, neighbors, feature_selection
MLA = [    
        # Generalized Linear Models
        LogisticRegressionCV(),
    
        # SVM
        svm.SVC(probability = True),
        svm.LinearSVC(),
    
        # KNN
        neighbors.KNeighborsClassifier(weights='distance'),
    
        #Discriminant Analysis
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),
     
        # Naive Bayes
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),
    
        #Trees    
        tree.DecisionTreeClassifier(),
    
        # Ensemble Methods
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier()
     
    ]

cv_split = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .7, random_state = 0)
MLA_columns = ['MLA Name','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean','MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

row_index = 0
for alg in MLA:
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    cv_results = cross_validate(alg, x,y, cv  = cv_split, return_train_score=True)
    
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
        
    row_index+=1
   

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

MLA_compare
# grid search for svm
classifier = svm.SVC()
base_results = cross_validate(classifier, x, y, cv  = cv_split, return_train_score=True)
classifier.fit(x, y)

epoch=0
for train_score,test_score in zip(base_results['train_score'], base_results['test_score']):
        epoch +=1       
        print("epoch:",epoch,"train_score:",train_score, "test_score:",test_score)
print('-'*10)

print('BEFORE Tuning Parameters: ', classifier.get_params())
print("BEFORE Tuning Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 
print("BEFORE Tuning Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))
print('-'*10)

param_grid = {'C':[0.5,1.0,2.0, 3.0],  # penalty parameter C of the error term
              'kernel':['linear', 'rbf'], # specifies the kernel type to be used in the algorithm  
              'gamma':[0.02, 0.08,0.2,1.0] # kernel coefficient for 'rbf'
             }

# Grid Search
tune_model = GridSearchCV(svm.SVC(), param_grid=param_grid, scoring = 'accuracy', cv = cv_split, return_train_score=True)
tune_model.fit(x, y)

for i in range(10):
    print("epoch:",i,"train_score:",tune_model.cv_results_['split'+str(i)+'_train_score'][tune_model.best_index_],
    "test_score:",tune_model.cv_results_['split'+str(i)+'_test_score'][tune_model.best_index_])

print('-'*10)    

print('AFTER Tuning Parameters: ', tune_model.best_params_)
print("AFTER Tuning Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print("AFTER Tuning Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('-'*10)

# grid search for decision trees
dtree = tree.DecisionTreeClassifier(random_state = 0)
base_results_dtree = cross_validate(dtree, x, y, cv  = cv_split, return_train_score=True)
dtree.fit(x, y)

epoch=0
for train_score,test_score in zip(base_results_dtree['train_score'], base_results_dtree['test_score']):
        epoch +=1       
        print("epoch:",epoch,"train_score:",train_score, "test_score:",test_score)
print('-'*10)

print('BEFORE Tuning Parameters: ', dtree.get_params())
print("BEFORE Tuning Training w/bin score mean: {:.2f}". format(base_results_dtree['train_score'].mean()*100)) 
print("BEFORE Tuning Test w/bin score mean: {:.2f}". format(base_results_dtree['test_score'].mean()*100))
print('-'*10)

param_grid = {'criterion': ['gini','entropy'], 
              'splitter': ['best', 'random'], 
              'max_depth': [2,4,6,8,10,None], 
              #'min_samples_split': [2,5,7,10,12], 
              #'min_samples_leaf': [1,3,5,7, 10], 
              'random_state': [0] 
             }


tune_model = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'accuracy', cv = cv_split, return_train_score=True)
tune_model.fit(x, y)

for i in range(10):
    print("epoch:",i,"train_score:",tune_model.cv_results_['split'+str(i)+'_train_score'][tune_model.best_index_],
    "test_score:",tune_model.cv_results_['split'+str(i)+'_test_score'][tune_model.best_index_])

print('-'*10)    

print('AFTER Tuning Parameters: ', tune_model.best_params_)
print("AFTER Tuning Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print("AFTER Tuning Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('-'*10)
# train the model using tuned decision tree parameters
dtree = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth= 4, random_state= 0, splitter= 'random')
base_results = cross_validate(dtree, x, y, cv  = cv_split, return_train_score=True)
dtree.fit(x, y)
def plot_feature_importances(df):
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    return df
importance = dtree.feature_importances_
feature = x.columns
fi = pd.DataFrame()
fi['importance'] = importance
fi['feature'] = feature
fi_sorted = plot_feature_importances(fi)
# grid search for bagging classifier
classifier = ensemble.BaggingClassifier()
base_results = cross_validate(classifier, x, y, cv  = cv_split, return_train_score=True)
classifier.fit(x, y)

cl1 = LogisticRegressionCV()
cl2 = tree.DecisionTreeClassifier()
cl3 = svm.LinearSVC()
cl4 = discriminant_analysis.LinearDiscriminantAnalysis()
cl5 = discriminant_analysis.QuadraticDiscriminantAnalysis()
param_grid = {'base_estimator':[cl1, cl2, cl3, cl4, cl5],
              'n_estimators':[10,13,17],
              #'warm_start':[False, True]
             }


tune_model = GridSearchCV(ensemble.BaggingClassifier(), param_grid=param_grid, scoring = 'accuracy', cv = cv_split, return_train_score=True)
tune_model.fit(x, y)


# printing the results of bagging before and after tuning
epoch=0
for train_score,test_score in zip(base_results['train_score'], base_results['test_score']):
        epoch +=1       
        print("epoch:",epoch,"train_score:",train_score, "test_score:",test_score)
print('-'*10)

print('BEFORE Tuning Parameters: ', classifier.get_params())
print("BEFORE Tuning Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 
print("BEFORE Tuning Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))
print('-'*10)

for i in range(10):
    print("epoch:",i,"train_score:",tune_model.cv_results_['split'+str(i)+'_train_score'][tune_model.best_index_],
    "test_score:",tune_model.cv_results_['split'+str(i)+'_test_score'][tune_model.best_index_])

print('-'*10)    


print('AFTER Tuning Parameters: ', tune_model.best_params_)
print("AFTER Tuning Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print("AFTER Tuning Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('-'*10)
classifier = ensemble.AdaBoostClassifier()
base_results = cross_validate(classifier, x, y, cv  = cv_split, return_train_score=True)
classifier.fit(x, y)

epoch=0
for train_score,test_score in zip(base_results['train_score'], base_results['test_score']):
        epoch +=1       
        print("epoch:",epoch,"train_score:",train_score, "test_score:",test_score)
print('-'*10)

print('BEFORE Tuning Parameters: ', classifier.get_params())
print("BEFORE Tuning Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 
print("BEFORE Tuning Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))
print('-'*10)

cl1 = LogisticRegressionCV()
cl2 = tree.DecisionTreeClassifier()
cl3 = naive_bayes.GaussianNB()
param_grid = {'base_estimator':[cl1, cl2, cl3]
             }


tune_model = GridSearchCV(ensemble.AdaBoostClassifier(), param_grid=param_grid, scoring = 'accuracy', cv = cv_split, return_train_score=True)
tune_model.fit(x, y)

for i in range(10):
    print("epoch:",i,"train_score:",tune_model.cv_results_['split'+str(i)+'_train_score'][tune_model.best_index_],
    "test_score:",tune_model.cv_results_['split'+str(i)+'_test_score'][tune_model.best_index_])

print('-'*10)    


print('AFTER Tuning Parameters: ', tune_model.best_params_)
print("AFTER Tuning Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print("AFTER Tuning Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('-'*10)
classifier = ensemble.RandomForestClassifier()
base_results = cross_validate(classifier, x, y, cv  = cv_split, return_train_score=True)
classifier.fit(x, y)

epoch=0
for train_score,test_score in zip(base_results['train_score'], base_results['test_score']):
        epoch +=1       
        print("epoch:",epoch,"train_score:",train_score, "test_score:",test_score)
print('-'*10)

print('BEFORE Tuning Parameters: ', classifier.get_params())
print("BEFORE Tuning Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 
print("BEFORE Tuning Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))
print('-'*10)

param_grid = {'n_estimators': [15,25,30,35],
              'criterion': ['gini','entropy'],  #scoring methodology; two supported formulas for calculating information gain - default is gini
              'max_depth': [2,4,6,None], #max depth tree can grow; default is none
              'min_samples_split': [2,5,7,10,12], #minimum subset size BEFORE new split (fraction is % of total); default is 2
              #'min_samples_leaf': [1,3,5], #minimum subset size AFTER new split split (fraction is % of total); default is 1
              'max_features': [2,3,'auto'], #max features to consider when performing split; default none or all
              'random_state': [0] #seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
             }


tune_model = GridSearchCV(ensemble.RandomForestClassifier(), param_grid=param_grid, scoring = 'accuracy', cv = cv_split, return_train_score=True)
tune_model.fit(x, y)

for i in range(10):
    print("epoch:",i,"train_score:",tune_model.cv_results_['split'+str(i)+'_train_score'][tune_model.best_index_],
    "test_score:",tune_model.cv_results_['split'+str(i)+'_test_score'][tune_model.best_index_])

print('-'*10)    


print('AFTER Tuning Parameters: ', tune_model.best_params_)
print("AFTER Tuning Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print("AFTER Tuning Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('-'*10)
# train the model using tuned random forest parameters
random_forest = ensemble.RandomForestClassifier(criterion= 'entropy', max_depth= None, random_state= 0, min_samples_split= 10, n_estimators=25)
base_results = cross_validate(random_forest, x, y, cv  = cv_split, return_train_score=True)
random_forest.fit(x, y)
importance = random_forest.feature_importances_
feature = x.columns
fi = pd.DataFrame()
fi['importance'] = importance
fi['feature'] = feature
fi_sorted = plot_feature_importances(fi)
MLA = [    
        # Generalized Linear Models
        LogisticRegressionCV(),
    
        # SVM
        svm.SVC(probability=True, C=1.0, gamma=0.02, kernel='linear'),
        svm.LinearSVC(),
    
        # KNN
        neighbors.KNeighborsClassifier(weights='distance'),
    
        #Discriminant Analysis
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),
     
        # Naive Bayes
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),
    
        #Trees    
        tree.DecisionTreeClassifier(criterion= 'entropy', max_depth= 4, random_state= 0, splitter= 'random'),
    
        # Ensemble Methods
        ensemble.AdaBoostClassifier(base_estimator = LogisticRegressionCV()),
        ensemble.BaggingClassifier(base_estimator=LogisticRegressionCV(), n_estimators=10),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(criterion='entropy', min_samples_split=10, n_estimators=25, random_state=0, max_features=3)
     
    ]

cv_split = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .7, random_state = 0)
MLA_columns = ['MLA Name','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean','MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

row_index = 0
for alg in MLA:
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    cv_results = cross_validate(alg, x,y, cv  = cv_split, return_train_score=True)
    
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
         
    row_index+=1
   

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

MLA_compare
print('BEFORE RFE Training Shape Old: ', x.shape) 
print('BEFORE RFE Training Columns Old: ', x.columns.values)

print("BEFORE RFE Training w/bin score mean: {:.2f}". format(base_results_dtree['train_score'].mean()*100)) 
print("BEFORE RFE Test w/bin score mean: {:.2f}". format(base_results_dtree['test_score'].mean()*100))
print('-'*10)

#feature selection
dtree_rfe = feature_selection.RFECV(tree.DecisionTreeClassifier(), step = 1, scoring = 'accuracy', cv = cv_split)
dtree_rfe.fit(x, y)

#transform x&y to reduced features and fit new model
X_rfe = x.columns.values[dtree_rfe.get_support()]
rfe_results = cross_validate(dtree, x[X_rfe], y, cv  = cv_split)

print('AFTER RFE Training Shape New: ', x[X_rfe].shape) 
print('AFTER RFE Training Columns New: ', X_rfe)

print("AFTER RFE Training w/bin score mean: {:.2f}". format(rfe_results['train_score'].mean()*100)) 
print("AFTER RFE Test w/bin score mean: {:.2f}". format(rfe_results['test_score'].mean()*100))
print('-'*10)

param_grid = {'criterion': ['gini','entropy'], 
              'splitter': ['best', 'random'], 
              'max_depth': [2,4,6,8,10,None], 
              #'min_samples_split': [2,5,7,10,12], 
              #'min_samples_leaf': [1,3,5,7, 10], 
              'random_state': [0] 
             }

#tune rfe model
rfe_tune_model = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'accuracy', cv = cv_split)
rfe_tune_model.fit(x[X_rfe], y)

print('AFTER RFE Tuned Parameters: ', rfe_tune_model.best_params_)
print("AFTER RFE Tuned Training w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 
print("AFTER RFE Tuned Test w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('-'*10)

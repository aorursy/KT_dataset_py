# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%pip install autoviz
from autoviz.AutoViz_Class import AutoViz_Class
import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from category_encoders import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy import stats
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, make_scorer
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv("/kaggle/input/titanic/train.csv")
dataset.head()
dataset.info()
dataset.isna().sum()
#Data Analysis
import pandas_profiling
report = pandas_profiling.ProfileReport(dataset)
display(report)
import warnings
warnings.filterwarnings("ignore")
AV=AutoViz_Class()
report2=AV.AutoViz("/kaggle/input/titanic/train.csv")
%pip install sweetviz
import sweetviz as sv
advert_report = sv.analyze(dataset)
advert_report.show_html('Advertising.html')
dataset2=dataset.drop(columns=['Name','Cabin'])

dataset2['Age'].fillna(dataset2['Age'].mean(), inplace=True)

permutation = np.random.permutation(dataset2['Embarked'])
empty_is = np.where(permutation == "")
permutation = np.delete(permutation, empty_is)
end = len(permutation)
dataset2['Embarked'] = dataset2['Embarked'].apply(lambda x: permutation[np.random.randint(end)] if pd.isnull(x) else x)
dataset2.isna().sum()
dataset2.describe()
dataset2.info()
dataset2["Family_Count"]=dataset2.SibSp+dataset2.Parch
dataset2 = dataset2.drop(columns=['SibSp','Parch'])
#Buidling dashboard

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_style("dark",{"axes.facecolor":"black"})

f,axes = plt.subplots(3,2,figsize=(15,15))
k1=sns.violinplot(data=dataset2,x="Sex",y="Pclass",ax=axes[0,0])
k2=sns.violinplot(data=dataset2,x="Embarked",y="Pclass",ax=axes[0,1])
k3=sns.violinplot(data=dataset2,x="Pclass",y="Survived",ax=axes[1,0])
#k4=sns.violinplot(data=dataset,x="Pclass",y="Age",ax=axes[1,1],palatte="YlorRd")
axes[1,1].hist(dataset2.Age,bins=15)
k4=sns.violinplot(data=dataset2,x="Pclass",y="Age",ax=axes[2,0])
k5=sns.violinplot(data=dataset2,x="Family_Count",y="Pclass",ax=axes[2,1])
#k1.set(xlim=(0,85))
plt.show()
#One Hot encoding
dataset2 = pd.get_dummies(dataset2,columns=['Pclass','Sex','Embarked'])
dataset2.columns.values
dataset2.info()

import matplotlib.pyplot as plt
%matplotlib inline

plt.title("Correlation of data with Survived",fontsize=35,color='DarkBlue',fontname='DejaVu Sans')
dataset2.corrwith(dataset2.Survived).plot.bar(figsize=(15,10),fontsize=20,rot=75,grid=True)
plt.show()
#Heatmap: data or feature correlation with each other
sns.set(style='white')

#Compute the correlational matrix
corr = dataset2.corr()

#Generate mask for the upper triange
mask=np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)]=True

#generate custom diverging colormap
cmap=sns.diverging_palette(220,10,as_cmap=True)

# Setup the matplotlib figure
f, ax = plt.subplots(figsize=(18,15))

sns.heatmap(corr,cmap=cmap,mask=mask,square=True,center=0,linewidths=0.5,fmt='g',vmax=0.3,cbar_kws={'shrink':0.5})
plt.title("Correlation of data with each other",fontsize=35,color='DarkBlue',fontname='DejaVu Sans')
plt.show()
# Removing extra columns and target
#dataset["Family_Count"]=dataset.SibSp+dataset.Parch
target = dataset['Survived']
Passenger = dataset['PassengerId']
dataset= dataset.drop(columns=['Survived','PassengerId','Ticket','Name','Cabin'])
dataset.head()
def defineBestModelPipeline(df, target, categorical_columns, numeric_columns):
    # Splitting into Train and Test Set
    x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.10, random_state=42)

# 1st -> Numeric Transformers
    numeric_transformer_1 = Pipeline(steps=[('imp', IterativeImputer(max_iter=30, random_state=42)),
                                            ('scaler', MinMaxScaler())])
    
    numeric_transformer_2 = Pipeline(steps=[('imp', IterativeImputer(max_iter=20, random_state=42)),
                                            ('scaler', StandardScaler())])
    
    numeric_transformer_3 = Pipeline(steps=[('imp', SimpleImputer(strategy='mean')),
                                            ('scaler', MinMaxScaler())])
    
    numeric_transformer_4 = Pipeline(steps=[('imp', SimpleImputer(strategy='median')),
                                            ('scaler', StandardScaler())])

# 2nd -> Categorical Transformer
    categorical_transformer = Pipeline(steps=[('frequent', SimpleImputer(strategy='most_frequent')),
                                              ('onehot', OneHotEncoder(use_cat_names=True))])
    

# 3rd -> Combining both numerical and categorical pipelines
    data_transformations_1 = ColumnTransformer(transformers=[('num', numeric_transformer_1, numeric_columns),
                                                             ('cat', categorical_transformer, categorical_columns)])
    
    data_transformations_2 = ColumnTransformer(transformers=[('num', numeric_transformer_2, numeric_columns),
                                                             ('cat', categorical_transformer, categorical_columns)])
    
    data_transformations_3 = ColumnTransformer(transformers=[('num', numeric_transformer_3, numeric_columns),
                                                             ('cat', categorical_transformer, categorical_columns)])
    
    data_transformations_4 = ColumnTransformer(transformers=[('num', numeric_transformer_4, numeric_columns),
                                                             ('cat', categorical_transformer, categorical_columns)])
    
    # And finally, we are going to apply these different data transformations to RandomSearchCV,
# trying to find the best imputing strategy, the best feature engineering strategy
    # and the best model with it's respective parameters.
    # Below, we just need to initialize a Pipeline object with any transformations we want, on each of the steps.
    pipe = Pipeline(steps=[('data_transformations', data_transformations_1),('feature_eng', PCA()),('clf', SVC())])

    params_grid = [
        {'data_transformations': [data_transformations_1, data_transformations_2, data_transformations_3, data_transformations_4],
                     'feature_eng': [None, 
                                     PCA(n_components=round(x_train.shape[1]*0.9)),
                                     PCA(n_components=round(x_train.shape[1]*0.8)),
                                     PCA(n_components=round(x_train.shape[1]*0.7)),
                                     PolynomialFeatures(degree=1), PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)],
                     'clf': [KNeighborsClassifier()],
                     'clf__n_neighbors': stats.randint(1, 50),
                     'clf__metric': ['minkowski', 'euclidean']},

    {'data_transformations': [data_transformations_1, data_transformations_2, data_transformations_3, data_transformations_4],
                     'feature_eng': [None, 
                                     PCA(n_components=round(x_train.shape[1]*0.9)),
                                     PCA(n_components=round(x_train.shape[1]*0.8)),
                                     PCA(n_components=round(x_train.shape[1]*0.7)),
                                     PolynomialFeatures(degree=1), PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)],
                     'clf': [LogisticRegression()],
                     'clf__penalty': ['l1', 'l2'],
                     'clf__C': stats.uniform(0.01, 10)},


    {'data_transformations': [data_transformations_1, data_transformations_2, data_transformations_3, data_transformations_4],
                     'feature_eng': [None, 
                                     PCA(n_components=round(x_train.shape[1]*0.9)),
                                     PCA(n_components=round(x_train.shape[1]*0.8)),
                                     PCA(n_components=round(x_train.shape[1]*0.7)),
                                     PolynomialFeatures(degree=1), PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)],
                     'clf': [SVC()],
                     'clf__C': stats.uniform(0.01, 1),
                     'clf__gamma': stats.uniform(0.01, 1),
                     'clf__kernel':['linear','rbf']},

    {'data_transformations': [data_transformations_1, data_transformations_2, data_transformations_3, data_transformations_4],
                     'feature_eng': [None, 
                                     PCA(n_components=round(x_train.shape[1]*0.9)),
                                     PCA(n_components=round(x_train.shape[1]*0.8)),
                                     PCA(n_components=round(x_train.shape[1]*0.7)),
                                     PolynomialFeatures(degree=1), PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)],
                     'clf': [RandomForestClassifier()],
                     'clf__n_estimators': stats.randint(10, 175),
                     'clf__max_features': [None, "auto", "log2"],
                     'clf__max_depth': [None, stats.randint(1, 5)],
                     'clf__random_state': stats.randint(1, 49)},

    {'data_transformations': [data_transformations_1, data_transformations_2, data_transformations_3, data_transformations_4],
                     'feature_eng': [None, 
                                     PCA(n_components=round(x_train.shape[1]*0.9)),
                                     PCA(n_components=round(x_train.shape[1]*0.8)),
                                     PCA(n_components=round(x_train.shape[1]*0.7)),
                                     PolynomialFeatures(degree=1), PolynomialFeatures(degree=2), PolynomialFeatures(degree=3)],
                     'clf': [GradientBoostingClassifier()],
                     'clf__n_estimators': stats.randint(10, 100),
                     'clf__learning_rate': stats.uniform(0.01, 0.7),
                     'clf__max_depth': [None, stats.randint(1, 6)]}

    ]
# Now, we fit a RandomSearchCV to search over the grid of parameters defined above
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    best_model_pipeline = RandomizedSearchCV(pipe, params_grid, n_iter=500, 
                                             scoring=metrics, refit='accuracy', 
                                             n_jobs=-1, cv=5, random_state=21)

    best_model_pipeline.fit(x_train, y_train)

    print("\n\n#---------------- Best Data Pipeline found in RandomSearchCV  ----------------#\n\n", best_model_pipeline.best_estimator_[0])
    print("\n\n#---------------- Best Feature Engineering technique found in RandomSearchCV  ----------------#\n\n", best_model_pipeline.best_estimator_[1])
    print("\n\n#---------------- Best Classifier found in RandomSearchCV  ----------------#\n\n", best_model_pipeline.best_estimator_[2])
    print("\n\n#---------------- Best Estimator's average Accuracy Score on CV (validation set) ----------------#\n\n", best_model_pipeline.best_score_)
    
    return x_train, x_test, y_train, y_test, best_model_pipeline
categorical_columns=['Pclass','Sex','Embarked']
numeric_columns=['Age', 'Fare', 'SibSp', 'Parch']
# Calling the function above, returing train/test data and best model's pipeline
x_train, x_test, y_train, y_test, best_model_pipeline = defineBestModelPipeline(dataset, target, categorical_columns, numeric_columns)
# Function responsible for checking our model's performance on the test data
def testSetResultsClassifier(classifier, x_test, y_test):
    predictions = classifier.predict(x_test)
    
    results = []
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    
    results.append(f1)
    results.append(precision)
    results.append(recall)
    results.append(accuracy)
    
    print("\n\n#---------------- Test set results (Best Classifier) ----------------#\n")
    print("F1 score, Precision, Recall, Accuracy:")
    print(results)
    
    return results
# Checking best model's performance on test data
test_set_results = testSetResultsClassifier(best_model_pipeline, x_test, y_test)

# Visualizing all results and metrics, from all models, obtained by the RandomSearchCV steps
df_results = pd.DataFrame(best_model_pipeline.cv_results_)

display(df_results)
# Based on above results , we received best accuracy from LogisticRegression (C=3.148239614927082) model. Lets use this model and check it on Test data
# and get predictions on test data
import os
validation_set = pd.read_csv("/kaggle/input/titanic/test.csv")
validation_set.head()
validation_set.isna().sum()
# Removing extra columns
Passenger_Test = validation_set['PassengerId']
validation_set["Family_Count"]=validation_set.SibSp+validation_set.Parch
validation_set = validation_set.drop(columns=['PassengerId','Ticket','Name','Cabin'])
validation_set.head()
# Applying best_model_pipeline
# Step 1 -> Transforming data the same way we did in the training set;
# Step 2 -> making predictions using the best model obtained by RandomSearchCV.
test_predictions = best_model_pipeline.predict(validation_set)
print(test_predictions)
test_predictions=pd.DataFrame({"Survived":test_predictions})
test_predictions
Final_Result = pd.concat([Passenger_Test,test_predictions],axis=1).dropna()
Final_Result = Final_Result.sort_values(by='PassengerId', ascending=True)
Final_Result
Final_Result.drop(Final_Result.columns.difference(['PassengerId', 'Survived']), axis=1, inplace=True) # Selecting only needed columns
Final_Result.head(10)
Final_Result.to_csv("Predictions for Titanic Project with Pipeline_Survival.csv",index=False)
Final_Result.count()

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

%matplotlib inline 

import warnings

warnings.filterwarnings("ignore")
data_dirty = pd.read_csv('../input/data.csv')

data_dirty
def count_missing(dataframe):   

    total = dataframe.isnull().sum().sort_values(ascending=False)

    percent = (dataframe.isnull().sum()/dataframe.isnull().count()*100).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data
count_missing(data_dirty)
data_dirty = data_dirty.dropna(subset=['Fare','Embarked'])
count_missing(data_dirty)
features = data_dirty.iloc[:,[2,4,5,6,7,11]].values

goal = data_dirty.iloc[:, 1]
# labeling the Data

from sklearn.preprocessing import LabelEncoder

encode = LabelEncoder()

features[:, 1] = encode.fit_transform(features[:, 1])

features[:,-1] = encode.fit_transform(features[:,-1].astype(str))
# imputing missing data 

from sklearn.preprocessing import Imputer 

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

features[:,:-1] = imputer.fit_transform(features[:,:-1])
from sklearn.preprocessing import OneHotEncoder

hotencoder = OneHotEncoder(categorical_features=[-1])

features = hotencoder.fit_transform(features).toarray()

pd.DataFrame(features)
# standardizing the data 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

features = scaler.fit_transform(features)

features
from sklearn.model_selection import train_test_split

train_set, test_set, goal_train, goal_test = train_test_split(features, goal, test_size= 0.3, random_state=0)
# Importing Modules 

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Estimators Parameters 

SVM_params = {'C':[0.001, 0.1, 10, 100], 'kernel':['rbf' ,'linear', 'poly', 'sigmoid']}

LR_params = {'C':[0.001, 0.1, 1, 10, 100]}

LDA_params = {'n_components':[None, 1,2,3], 'solver':['svd'], 'shrinkage':[None]}

KNN_params = {'n_neighbors':[1,5,10,20, 50], 'p':[2], 'metric':['minkowski']}

RF_params = {'n_estimators':[10,50,100]}

DTC_params = {'criterion':['entropy'], 'max_depth':[10,50,100]}
# The Estimator Function

models_opt = []



models_opt.append(('LR', LogisticRegression(), LR_params))

models_opt.append(('LDA', LinearDiscriminantAnalysis(), LDA_params))

models_opt.append(('KNN', KNeighborsClassifier(),KNN_params))

models_opt.append(('SVM', SVC(), SVM_params))

models_opt.append(('Decision Tree Classifier',DecisionTreeClassifier(), DTC_params))

models_opt.append(('Random Forest', RandomForestClassifier(), RF_params))
from sklearn.model_selection import KFold, cross_val_score

from sklearn.model_selection import learning_curve, GridSearchCV

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
results = []

names = []





def estimator_function(parameter_dictionary, scoring = 'accuracy'):

    

    

    for name, model, params in models_opt:

    

        kfold = KFold( n_splits=5 ,  random_state=2, shuffle=True)



        model_grid = GridSearchCV(model, params)



        cv_results = cross_val_score(model_grid, train_set, goal_train, cv = kfold, scoring=scoring)



        results.append(cv_results)



        names.append(name)



        msg = "Cross Validation Accuracy %s: Accarcy: %f SD: %f" % (name, cv_results.mean(), cv_results.std())



        print(msg)
estimator_function(models_opt, scoring = 'accuracy')
# Instantiate model

GNB =  GaussianNB()



# Define kfold - this was done above but not as a global variable 

kfold = KFold( n_splits=5, random_state=2, shuffle=True)



# Run cross validation

cv_results_GNB= cross_val_score(GNB, train_set, goal_train, cv = kfold )

print(cv_results_GNB.mean())

# Append results and names lists

results.append(cv_results_GNB)

names.append('GNB')
# Ensemble Voting



from sklearn.ensemble import VotingClassifier



# Create list for estimatators

estimators = []



# Create estimator object

model1 = LogisticRegression()



# Append list with estimator name and object

estimators.append(("logistic", model1))

model2 = DecisionTreeClassifier()

estimators.append(("cart", model2))

model3 = SVC()

estimators.append(("svm", model3))

model4 = KNeighborsClassifier()

estimators.append(("KNN", model4))

model5 = RandomForestClassifier()

estimators.append(("RFC", model5))

model6 = GaussianNB()

estimators.append(("GB", model6))

model7 = LinearDiscriminantAnalysis()

estimators.append(("LDA", model7))





voting = VotingClassifier(estimators)





results_voting = cross_val_score(voting, train_set, goal_train, cv=kfold)



results.append(results_voting)

names.append('Voting')



print('Accuracy: {} SD: {}'.format(results_voting.mean(), results_voting.std()))
import seaborn as sns 
# Visualize model accuracies for comparision - boxplots will be appropriate to visualize 

# data variation



plt.boxplot(results, labels = names)

plt.title('Breast Cancer Diagnosis Accuracy using Various Machine Learning Models')

plt.ylabel('Model Accuracy %')

sns.set_style("whitegrid")



plt.show()
from sklearn.decomposition import PCA



# Instantiate PCA

pca_var = PCA()



# Fit PCA to training data

pca_var.fit(train_set)



# Visualize explained variance with an increasing number of components

plt.plot(pca_var.explained_variance_, 'bo-', markersize=8)

plt.title("Elbow Curve for PCA Dimension of Breast Cancer Diagnosis Data")

plt.ylabel('Explained Variance')

plt.xlabel('Component Number')

sns.set_style("whitegrid")

plt.show()

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.cm as cm

from matplotlib import rcParams

rcParams['xtick.major.pad'] = 1

rcParams['ytick.major.pad'] = 1



#Instantiate new PCA object

pca = PCA(n_components=3)



# Fit and transform training data with PCA using 3 components

pca.fit(train_set)

X_train_norm_pca = pca.transform(train_set)



# Create a dataframe of 3 PCA

pca_df = pd.DataFrame(X_train_norm_pca, columns = ['PCA1', 'PCA2', 'PCA3'])



# Append diagnosis data into PCA dataframe

pca_df['Survived'] = goal_train



# Visualize PCA in a 3D plot - color points by diagnsosis to see if a visuale stratification occurs

pca_fig = plt.figure().gca(projection = '3d')

pca_fig.scatter(pca_df['PCA1'], pca_df['PCA2'], pca_df['PCA3'], c = pca_df['Survived'], cmap=cm.coolwarm)

pca_fig.set_xlabel('PCA1')

pca_fig.set_ylabel('PCA2')

pca_fig.set_zlabel('PCA3')

pca_fig.set_title('Data Visualized After 3-Component PCA')



sns.set_style("whitegrid")

plt.tight_layout()

plt.show()
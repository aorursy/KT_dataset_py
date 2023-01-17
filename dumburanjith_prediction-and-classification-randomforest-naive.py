# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import Libraries



# Basic Libraries

import numpy as np

import pandas as pd

# Visualization Libraries

from matplotlib import pyplot as plt

import matplotlib.colors as colors

import seaborn as sns

import itertools

from scipy.stats import norm

import scipy.stats



# Classification and Regression Algorithm Libraries

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import explained_variance_score

from sklearn.metrics import confusion_matrix





%matplotlib inline

sns.set()



# Importing the housing dataset

df = pd.read_csv('/kaggle/input/house-sales-prediction-and-classification/house_data.csv')



# Identify Schema and datatypes

df.info()
# Data Analysis

df.head(10)

df.tail(10)
# Field by Field Distinct Values

print("Number of Bedrooms avaialble:")

df.bedrooms.unique()

print("Number of Bathrooms avaialble:")

df.bathrooms.unique()

print("Number of FLoors avaialble:")

df.floors.unique()

print("Types of Views avaialble:")

df.view.unique()

print("Condition of Houses avaialble:")

df.condition.unique()

print("Grade of Houses avaialble:")

df.grade.unique()

print("Houses are built in the year range:")

df.yr_built.unique()

print("Houses are renovated in the year range:")

df.yr_renovated.unique()
# Number of houses avaialble avaiable by TYpe



df["bedrooms"].value_counts()

df["bathrooms"].value_counts()

df["floors"].value_counts()

df["view"].value_counts()

df["condition"].value_counts()

df["grade"].value_counts()

df["yr_built"].value_counts()

df["yr_renovated"].value_counts() 
# Field by Field Min and Max Distinct Values



print("Min Bedroom avaiable:")

min(df['bedrooms'])

print("Max Bedroom avaiable:")

max(df['bedrooms'])

print("Min Price for house:")

min(df['price'])

print("Max price avaiable:")

max(df['price'])

print("Min Number of floors available:")

min(df['floors'])

print("Min Number of floors available::")

max(df['floors'])

print("Oldest house built:")

min(df['yr_built'])

print("Newest  house built:")

max(df['yr_built'])
# Data Pre Procecssing and Cleansing



# Remove Null Values

df = df.dropna()



# Remove Duplicates

df.drop_duplicates(inplace = True)



# Remove Houses which have bedrooms greater than 10 for plotting purpose

df1 = df.query('bedrooms <= 10')



df1 = df1.drop(['id','date'],axis=1)



# Generate New Column Rating based on House Grading

df1['Rating'] = ['Good' if x < 7 else 'Excellent' for x in df1['grade']] 



df1.info()
# Data Visualization



# Seabron Visualization using pairplot

sns.pairplot(df1[['price','bedrooms','bathrooms','floors','view','condition','grade','yr_built','yr_renovated','sqft_lot']])



# Heat map showing co-relationship between variables



plt.figure(figsize=(15,10))

columns =['price','bedrooms','bathrooms','sqft_living','floors','grade','yr_built','condition','sqft_lot']

sns.heatmap(df1[columns].corr(),annot=True)
# Linear regression between Price and Sqft





# Rating(Good vs Excellent) and their price, # of Bedrooms and their price

fig, ax = plt.subplots(1,2, figsize=(14,8))

sns.boxplot(x = 'Rating',y='price', data = df1, ax=ax[0])

ax[0].set_title('Rating vs price')

sns.boxplot(x = 'bedrooms',y='price', data = df1, ax=ax[1])

ax[1].set_title('Number of Bedrooms')
# Indivudal variable relationship with Price



# Linear regression between sqft and price

sns.jointplot(x='sqft_lot',y='price',data=df1,kind='reg',size=4)



# Linear regression between Year Built and price

sns.jointplot(x='yr_built',y='price',data=df1,kind='reg',size=4)



# Linear regression between Year renovated and price

sns.jointplot(x='yr_renovated',y='price',data=df1,kind='reg',size=4)



# Linear regression between Number of Bedrooms and price

sns.jointplot(x='bedrooms',y='price',data=df1,kind='reg',size=4)



# Linear regression between Number of Bathrooms and price

sns.jointplot(x='bathrooms',y='price',data=df1,kind='reg',size=4)



# Linear regression between condition and price

sns.jointplot(x='condition',y='price',data=df1,kind='reg',size=4)



# Linear regression between grade and price

sns.jointplot(x='grade',y='price',data=df1,kind='reg',size=4)
# Building our model using different regression models



# Data Preparation for fitting data into model

df1.info()

df1.head(10)



df2 = df1.drop(['Rating'],axis=1)



df2['bathrooms'] = df2['bathrooms'].apply(np.int64)

df2['price'] = df2['price'].apply(np.int64)



df2.info()



# X(Independent variables) and y(target variables) 

X = df2.iloc[:,1:].values

y = df2.iloc[:,0].values



#Splitting the data into train,test data 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
# Multiple Linear Regression Model



mlr = LinearRegression()

mlr.fit(X_train,y_train)

mlr_score = mlr.score(X_test,y_test)

pred_mlr = mlr.predict(X_test)

expl_mlr = explained_variance_score(pred_mlr,y_test)





# Decision Tree Regressional Model



tr_regressor = DecisionTreeRegressor(random_state=0)

tr_regressor.fit(X_train,y_train)

tr_regressor.score(X_test,y_test)

pred_tr = tr_regressor.predict(X_test)

decision_score=tr_regressor.score(X_test,y_test)

expl_tr = explained_variance_score(pred_tr,y_test)



# Random Forest Regressional Model



rf_regressor = RandomForestRegressor(n_estimators=28,random_state=0)

rf_regressor.fit(X_train,y_train)

rf_regressor.score(X_test,y_test)

rf_pred =rf_regressor.predict(X_test)

rf_score=rf_regressor.score(X_test,y_test)

expl_rf = explained_variance_score(rf_pred,y_test)
# Calculate Model Score for all Regressional Models



print("Multiple Linear Regression Model Score is ",round(mlr.score(X_test,y_test)*100))

print("Decision tree  Regression Model Score is ",round(tr_regressor.score(X_test,y_test)*100))

print("Random Forest Regression Model Score is ",round(rf_regressor.score(X_test,y_test)*100))



#Let's have a tabular pandas data frame, for a clear comparison



models_score =pd.DataFrame({'Model':['Multiple Linear Regression','Decision Tree','Random forest Regression'],

                            'Score':[mlr_score,decision_score,rf_score],

                            'Explained Variance Score':[expl_mlr,expl_tr,expl_rf]

                           })

models_score.sort_values(by='Score',ascending=False)
# Apply Naïve Bayes Classifier to classify Rating of houses with the decision boundaries



def predict_NB_gaussian_class(X,mu_list,std_list,pi_list): 

    #Returns the class for which the Gaussian Naive Bayes objective function has greatest value

    scores_list = []

    classes = len(mu_list)

    

    for p in range(classes):

        score = (norm.pdf(x = X[0], loc = mu_list[p][0][0], scale = std_list[p][0][0] )  

                * norm.pdf(x = X[1], loc = mu_list[p][0][1], scale = std_list[p][0][1] ) 

                * pi_list[p])

        scores_list.append(score)

             

    return np.argmax(scores_list)



def predict_Bayes_class(X,mu_list,sigma_list): 

    #Returns the predicted class from an optimal bayes classifier - distributions must be known

    scores_list = []

    classes = len(mu_list)

    

    for p in range(classes):

        score = scipy.stats.multivariate_normal.pdf(X, mean=mu_list[p], cov=sigma_list[p])

        scores_list.append(score)

             

    return np.argmax(scores_list)



#Estimating the parameters

mu_list = np.split(df1.groupby('Rating').mean().values,[1,2])

std_list = np.split(df1.groupby('Rating').std().values,[1,2], axis = 0)



pi_list = df1.iloc[:,2].value_counts().values / len(df1)





# Our 2-dimensional distribution will be over variables X and Y

N = 100

X = np.linspace(4, 8, N)

Y = np.linspace(1.5, 5, N)

X, Y = np.meshgrid(X, Y)



color_list = ['b','r']



my_norm = colors.Normalize(vmin=-1.,vmax=1.)



g = sns.FacetGrid(df1, hue="Rating", size=10, palette = 'colorblind') .map(plt.scatter, "yr_built", "bedrooms",)  .add_legend()

my_ax = g.ax

# Apply logistic regression to classify Rating  with the decision boundaries



df1.info()



#split dataset in features and target variable

feature_cols = ['bedrooms','yr_built','floors']

X = df1[feature_cols] # Features

y = df1.condition # Target variable



# split X and y into training and testing sets

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)



# import the class

from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)

logreg = LogisticRegression()

# fit the model with data

logreg.fit(X_train,y_train)



y_pred=logreg.predict(X_test)



# import the metrics class

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix



# import required modules

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
import sklearn.linear_model as skl_lm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import confusion_matrix, classification_report, precision_score

from sklearn import preprocessing

from sklearn import neighbors



regr = skl_lm.LogisticRegression()

regr.fit(X_train, y_train)



def plot_confusion_matrix(cm, classes, n_neighbors, title='Confusion matrix (Normalized)',

                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title('Normalized confusion matrix: KNN-{}'.format(n_neighbors))

    plt.colorbar()

    plt.xticks(np.arange(2), classes)

    plt.yticks(np.arange(2), classes)

    plt.tight_layout()

    plt.xlabel('True label',rotation='horizontal', ha='right')

    plt.ylabel('Predicted label')

    plt.show()



pred = regr.predict(X_test)

cm_df = pd.DataFrame(confusion_matrix(y_test, pred).T, index=regr.classes_,columns=regr.classes_)

cm_df.index.name = 'Predicted'

cm_df.columns.name = 'True'

print(cm_df)

print(classification_report(y_test, pred))

# Data Preparation for Random Forest Plot



df1.info()

df1.head(10)



df3 = df1[['bedrooms','yr_built','Rating']]



df3['Rating'] = ['1' if x == 'Excellent' else '0' for x in df3['Rating']] 



df3.info()



df3.head(10)



# Select Year bedrooms, Year built 

X1 = df3.iloc[:, [0, 1]].values



# select Rating

y1 = df3.iloc[:, 2].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.25, random_state = 0)





from sklearn.ensemble import RandomForestClassifier



classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



# Visualising the Training set results

from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Random Forest Classification (Training set)')

plt.xlabel('No of BedRooms')

plt.ylabel('Year Built')

plt.legend()

plt.show()
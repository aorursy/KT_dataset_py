# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Let's import the data into this Kaggle workspace first.
ExasensData = "/kaggle/input/Exasens.csv"
ExasensDF = pd.read_csv("../input/exasens-data-set/Exasens.csv")
ExasensDF.head(30)
# i) Let's just quickly check the data types of the different columns in the Dataframe after we drop the first two rows. Help source: 
# https://hackersandslackers.com/pandas-dataframe-drop/
ExasensDF = ExasensDF.drop([0,1], axis = 0)
ExasensDF.head(5)
# ii) Let's try casting the real and imaginary (permittivity reading) part columns into float64 data to avoid issues with the .abs() function
#later (error trying to convert string types into their absolute values). To do this, let's use pandas'.to_numeric() function, as demonstrated
#in this thread: https://stackoverflow.com/questions/42719749/pandas-convert-string-to-int.
ExasensDF['Imaginary Part'] = ExasensDF['Imaginary Part'].astype(int)
ExasensDF['Real Part'] = ExasensDF['Real Part'].astype(int)

ExasensDF.head(5)
ExasensDF.dtypes
# iii) Now, let's convert these two columns' elements into their absolute values for ease of scatterplot interpretation: 
ExasensDF['Real Part']=ExasensDF['Real Part'].abs()
ExasensDF['Imaginary Part']=ExasensDF['Imaginary Part'].abs()
ExasensDF.head(5)
# iv) Now, let's clean the data by removing any participants with NaN values from the dataframe. More specifically, let's go ahead and 
#drop any participants (rows from the Dataframe) with 'Real Part', 'Imaginary Part', or 'Diagnosis' column values of NaN.
#Let's also completely drop the "Unnamed: 9" and "Unnamed: 10" columns, as it doesn't 
#seem like the researchers utilized these for anything. **Good Pandas Data Wrangling Reference: https://chrisalbon.com/python/data_wrangling/pandas_dropping_column_and_rows/**

ExasensDF = ExasensDF[ExasensDF['Imaginary Part'].notna()]
ExasensDF = ExasensDF[ExasensDF['Real Part'].notna()]
ExasensDF = ExasensDF[ExasensDF['Diagnosis'].notna()]
ExasensDF.head(5)
import matplotlib.pyplot as plt

ExasensScatters = plt.figure(figsize=(40,10))

ExasensScatterplot = ExasensScatters.add_subplot(1,2,1)

#Since we are going to analyze the relationship between two continuous parameters and a multiclass target variable, let's use the 
#following StackOverflow thread on how to use Pandas to scatterplot this scenario: https://stackoverflow.com/questions/26139423/plot-different-color-for-different-categorical-levels-using-matplotlib.
#This method utilizes a dictionary of color-to-class codes and a lambda function within a "c" parameter in the pyplot .scatter() function.

colors = {'COPD':'red', 'HC':'green', 'Asthma':'blue', 'Infected':'black'}
ExasensScatter = ExasensScatterplot.scatter(ExasensDF['Real Part'], ExasensDF['Imaginary Part'], c=ExasensDF['Diagnosis'].apply(lambda x: colors[x]))
plt.title("Real and Imaginary Part Saliva Permittivity Readings vs. Respiratory Illness Diagnoses: Exasens Saliva Biosensor Study",fontsize=20)
plt.xlabel("Permittivity Real Part",fontsize=20)
plt.ylabel("Permittivity Imaginary Part",fontsize=20)

#7/18/2020: NOTE: Let's scale these axes to better fit their ranges and make a more compressed plot. This modification must occur on the
#subplot object itself. Let's also add a plot legend for readability.
ExasensScatterplot.set_xlim([425,560])
handles, labels = ExasensScatter.legend_elements(prop = 'colors')
labels = ['COPD','HC','Asthma','Infected']
plt.legend(handles, labels, title = 'Diagnoses By Color')

#Note: After all of the cleaning that took place beforehand, we are left with a dataset of ~59 study participants
#7/18/2020: Still need to get the legend outputting correctly. 
# 2i) First, let's convert the Pandas dataframe above into two numpy arrays for more ease of use with scikit learn functions (train/test splitting, etc.): 
ExasensFeatures = np.asarray(ExasensDF[['Real Part','Imaginary Part']])
ExasensTarget = np.asarray(ExasensDF['Diagnosis'])

#Next, let's split the whole dataset into training and testing sets for higher validity: 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(ExasensFeatures, ExasensTarget, test_size=0.2, random_state=4)
# 2ii) Now, I'll go ahead and build the KNN Model with Scikit Learn and evaluate it, plotting the accuracy results: 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics 

ExasensKNNScores = []
ExasensNeighborsAccuracyArray = np.zeros(19)
kTestIterationList = range(1,20)
for k in kTestIterationList:

    ExasensNeighbors = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
    DiagnosesPredictions=ExasensNeighbors.predict(X_test)
    ExasensNeighborsAccuracyArray[k-1]=metrics.accuracy_score(y_test,DiagnosesPredictions)
    ExasensKNNScores.append(metrics.accuracy_score(y_test, DiagnosesPredictions))
    
plt.plot(kTestIterationList, ExasensKNNScores)
plt.xlabel('Value of K for Respiratory Illness Diagnosis KNN')
plt.ylabel('KNN Testing Accuracy')
# 3i) Implementing a Scikit Learn Logistic Model on the Dataset: 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
ExasensLogistic = LogisticRegression(C=0.01, solver='newton-cg', multi_class='multinomial').fit(X_train,y_train)

#Let's make a few predictions using this model and the test set, as well as the probability of each of the class targets:
ExasensLogisticScorePreds = ExasensLogistic.predict(X_test) 
ExasensLogisticScorePreds
# 3ii) Let's quickly use a Jaccard Score to evaluate the performance of the multiclass LR model: 
from sklearn.metrics import jaccard_score
jaccard_score(y_test, ExasensLogisticScorePreds, average = 'weighted')
#Using the "weighted" parameter returns the higest Jaccard score, which is only ~0.12894.
# 3iii) I'm going to quickly double check the probability of these LR classifications: 
LogisticExasensScoreProbas = ExasensLogistic.predict_proba(X_test)
LogisticExasensScoreProbas
#So in addition to the MLR model's low accuracy, the probabilities of its predictions' accuracies were low. Let's try incorporating
#a few more parameters to increase the accuracies of one of/both of these models. 
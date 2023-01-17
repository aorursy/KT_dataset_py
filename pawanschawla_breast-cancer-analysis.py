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
# The goal is to accurately classify whether the given cancer cell is benign or malignant based on the data using supervised learning.



# Importing Libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plot

import seaborn as sb





# Importing the data



bc = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')





# Looking at the dataset



print('\n Dataset: \n', bc)





# Checking for missing data



print('\n Missing Value Count: \n', bc.isna().sum())





# We need to remove the last column



bc = bc.drop(['Unnamed: 32'], axis=1)





# Looking at the Target Variable diagnosis(M=Malignant, B=Benign)



print('\n Target Variable (Diagnosis): \n', bc['diagnosis'].value_counts())





# Visualizing the Target Variable



sb.set(style="darkgrid")

plot.figure("Target Variable")

sb.countplot(x="diagnosis", data=bc)

plot.title('Count of Malignant vs Benign Tumors')

plot.show()



# Let's look at the Datatypes



print('\n Datatypes: \n', bc.dtypes)





# Encoding the Target Variable because it's an object(binary variable)



from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

bc['diagnosis'] = LE.fit_transform(bc['diagnosis'])





# Malignant(M) = 1 & Benign(B) = 0



print('\n Malignant(M) = 1 & Benign(B) = 0 \n', bc['diagnosis'])





# Simple Univariate Histograms



fig, axs = plot.subplots(6,5,figsize=(20,20),tight_layout=True, num='Simple Univariate Plots')

pos=2

for i in range(6):

    for j in range(5):

        axs[i][j].hist(bc.iloc[:,pos])

        axs[i][j].set(xlabel=(bc.columns[pos]), ylabel='Count')

        pos=pos+1

plot.show()



# Advanced Univariate Plots to check for normal distribution



from scipy.stats import norm

fig, axs = plot.subplots(6,5,figsize=(17,17),tight_layout=True, num='Advanced Univariate Plots')

pos=2

for i in range(6):

    for j in range(5):

        sb.distplot(bc.iloc[:,pos],fit=norm, ax=axs[i,j])

        axs[i][j].set(xlabel=(bc.columns[pos]))

        pos=pos+1

plot.show()



# As we can see, some variables follow a normal distribution, some follow a skewnormal distribution and the remaining follow a right-skewed distribution.



# A few plots explaining the distribution of data



sb.pairplot(bc.iloc[:,1:10], hue='diagnosis',corner=True)

plot.suptitle("A few Pairplots")

plot.show()



# Let's see the correlations



plot.figure("Correlation Matrix", figsize=(30,30))

sb.heatmap(bc.iloc[:,1:32].corr(), annot=True, fmt='.0%')

plot.show()



# We see that there is multicollinearity in the variables



# Let's split the data into Independent(X) and Dependent(Y) Datasets



X = bc.iloc[:,2:31].values

Y = bc['diagnosis'].values





# X contains all the features which will be used to predict Y(diagnosis)



# Splitting the data into Training (70%) and Test (30%) 



from sklearn.model_selection import train_test_split

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.3, random_state = 0)





#This removes the scientific notation of the numbers in the array



np.set_printoptions(suppress=True)





# Since there is multicollinearity in the data, we can use Tree based models like Decision Tree and Random Forest Classifier



from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(random_state = 0)

DT.fit(X_Train, Y_Train)

print('\n Decision Tree Training Accuracy is: ', DT.score(X_Train,Y_Train)*100,'% \n')



from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state = 0)

RF.fit(X_Train, Y_Train)

print('\n Random Forest Training Accuracy is: ', RF.score(X_Train,Y_Train)*100,'% \n')



# As we can see above, the Training Accuracy for both Decision Tree model and Random Forest model is 100%



# Now lets test these models to predict on the Test data



from sklearn.metrics import confusion_matrix

ConfDT = confusion_matrix(Y_Test, DT.predict(X_Test))

print('\n Decision Tree Confusion Matrix: \n', ConfDT)





from sklearn.metrics import confusion_matrix

ConfRF = confusion_matrix(Y_Test, RF.predict(X_Test))

print('\n Random Forest Confusion Matrix: \n', ConfRF)



#Importing the reports for Precision, Recall, Accuracy metrics



from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



print(classification_report(Y_Test, DT.predict(X_Test)))

print('\n Decision Tree Accuracy on Test Data: ',accuracy_score(Y_Test, DT.predict(X_Test))*100,'% \n')





# Let's try to improve the accuracy of the Decision Tree



DT1 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

DT1.fit(X_Train, Y_Train)

print('\n New Decision Tree Training Accuracy is: ', DT1.score(X_Train,Y_Train)*100,'% \n')



ConfDT1 = confusion_matrix(Y_Test, DT1.predict(X_Test))

print('\n New Decision Tree Confusion Matrix: \n', ConfDT1)



print(classification_report(Y_Test, DT1.predict(X_Test)))

print('\n Decision Tree Accuracy on Test Data: ', accuracy_score(Y_Test, DT1.predict(X_Test))*100,'% \n')





# The Random Forest Model is able to predict with an accuracy of 96%



print(classification_report(Y_Test, RF.predict(X_Test)))

print('\n Random Forest Classifier Accuracy on Test Data: ',accuracy_score(Y_Test, RF.predict(X_Test))*100,'% \n')





# Thank you!







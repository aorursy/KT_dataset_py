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
#Import pandas,numpy for Dataset Manupilation and matplotlib and seaborn for Visualization

import pandas as pd  

import matplotlib.pyplot as plt 

import seaborn as sns 

import numpy as np
#Import functions for Model, Dataset Splitting and Evaluation

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn import metrics
df=pd.read_csv('/kaggle/input/glass/glass.csv') #Read the Dataset CSV File to a dataframe object
df.shape # To view the shape of our dataset (214 rows and 10 columns)
df.head()
df.info() #Information about the Dataframe
df.describe() # Further Statistical Information about the dataset
# Display datapoints of glass by plotting Silicon vs Calcium (Two main chemicals of glass composition)

df.plot(kind="scatter", x="Si", y="Ca") # Plot the data points (x-Sepal Length and y-Sepal Width)

plt.show()
sns.FacetGrid(df, hue="Type", height=5).map(plt.scatter, "Si", "Ca").add_legend() 

plt.show()

print("""-- 1 building_windows_float_processed

-- 2 building_windows_non_float_processed

-- 3 vehicle_windows_float_processed

-- 4 vehicle_windows_non_float_processed (none in this database)

-- 5 containers

-- 6 tableware

-- 7 headlamps""")
# Display distribution of data points of each class in each attribute

plt.figure(figsize=(15,10))

plt.subplot(3,3,1)

sns.stripplot(x = 'Type', y = 'RI', data = df, jitter = True)

plt.subplot(3,3,2)

sns.stripplot(x = 'Type', y = 'Na', data = df, jitter = True)

plt.subplot(3,3,3)

sns.stripplot(x = 'Type', y = 'Mg', data = df, jitter = True)

plt.subplot(3,3,4)

sns.stripplot(x = 'Type', y = 'Al', data = df, jitter = True)

plt.subplot(3,3,5)

sns.stripplot(x = 'Type', y = 'Si', data = df, jitter = True)

plt.subplot(3,3,6)

sns.stripplot(x = 'Type', y = 'K', data = df, jitter = True)

plt.subplot(3,3,7)

sns.stripplot(x = 'Type', y = 'Ca', data = df, jitter = True)

plt.subplot(3,3,8)

sns.stripplot(x = 'Type', y = 'Ba', data = df, jitter = True)

plt.subplot(3,3,9)

sns.stripplot(x = 'Type', y = 'Fe', data = df, jitter = True)
sns.pairplot(data=df,kind='scatter') #Shows relationships among all pairs of features
corr=df.corr() #Correlation Matrix
# Display the correlation matrix using a heatmap

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
# Exactly –1. A perfect downhill (negative) linear relationship



# –0.70. A strong downhill (negative) linear relationship



# –0.50. A moderate downhill (negative) relationship



# –0.25. A weak downhill (negative) linear relationship



# 0. No linear relationship





# +0.25. A weak uphill (positive) linear relationship



# +0.50. A moderate uphill (positive) relationship



# +0.70. A strong uphill (positive) linear relationship



# Exactly +1. A perfect uphill (positive) linear relationship
X=df[['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']].values

y=df['Type'].values
clf=SVC(kernel='linear')
# Create the training and test sets using 0.2 as test size (i.e 80% of data for training rest 25% for model testing)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#Create the SVC Object and fit the training set to the model object

clf.fit(X_train, y_train)
clf.get_params()
y_pred = clf.predict(X_test)
# Model Accuracy: how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Get the confusion Matrix of the Model

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix
# Plot the Confusion Matrix as a HeatMap

class_names=[0,1,2,3,4,5] # Name  of classes

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
print(metrics.classification_report(y, clf.predict(X),zero_division=1)) # Displays a comprehensive Report of the SVC Model
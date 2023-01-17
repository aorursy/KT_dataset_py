# import dependencies

# data cleaning and manipulation 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL



# data visualization

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph. I like it most for plot



# machine learning

from sklearn.linear_model import LogisticRegression # to apply the Logistic regression

from sklearn.model_selection import train_test_split # to split the data into two parts

from sklearn import metrics #



from sklearn.preprocessing import StandardScaler



import sklearn.linear_model as skl_lm

from sklearn import preprocessing

from sklearn import neighbors

from sklearn.metrics import confusion_matrix, classification_report, precision_score





import statsmodels.api as sm

import statsmodels.formula.api as smf





# initialize some package settings

sns.set(style="whitegrid", color_codes=True, font_scale=1.3)





%matplotlib inline
# read in the data and check the first 5 rows

df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv', index_col=0) # here header 0 means the 0 th row is our coloumn 

                                                                                # header in data

df.head()
#look what kind of data we have

df.info()
# remove unwanted column

df = df.drop('Unnamed: 32', axis=1)

#df.drop("id",axis=1,inplace=True)

#axis=1 means drop column

#df.dtypes

df.columns
sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap='viridis') #visualise missing data if any
# visualize distribution of classes 

plt.figure(figsize=(8, 4))

sns.countplot(df['diagnosis'], palette='RdBu')



# count number of obvs in each class

benign, malignant = df['diagnosis'].value_counts()

print('Number of cells labeled Benign: ', benign)

print('Number of cells labeled Malignant : ', malignant)

print('')

print('% of cells labeled Benign', round(benign / len(df) * 100, 2), '%')

print('% of cells labeled Malignant', round(malignant / len(df) * 100, 2), '%')


features_mean= list(df.columns[1:11]) # mean

features_se= list(df.columns[11:20]) # standard errors

features_worst=list(df.columns[21:31]) # mean of the 3 largest value

print(features_mean)

print("-----------------------------------")

print(features_se)

print("------------------------------------")

print(features_worst)
correlation = df.corr() # .corr is used for find corelation

plt.figure(figsize=(27,27))

sns.heatmap(correlation, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},

           xticklabels= df.columns[1:31], yticklabels= df.columns[1:31],

           cmap= 'coolwarm') 

# for more on heatmap you can visit Link(http://seaborn.pydata.org/generated/seaborn.heatmap.html)
cols = ['radius_worst', 

        'texture_worst', 

        'perimeter_worst', 

        'area_worst', 

        'smoothness_worst', 

        'compactness_worst', 

        'concavity_worst',

        'concave points_worst', 

        'symmetry_worst', 

        'fractal_dimension_worst']



df = df.drop(cols, axis=1)



# then, drop all columns related to the "perimeter" and "area" attributes

cols = ['perimeter_mean',

        'perimeter_se', 

        'area_mean', 

        'area_se']

df = df.drop(cols, axis=1)



# lastly, drop all columns related to the "concavity" and "concave points" attributes

cols = ['concavity_mean',

        'concavity_se', 

        'concave points_mean', 

        'concave points_se']

df = df.drop(cols, axis=1)



# verify remaining columns

df.columns
X = df

y = df['diagnosis']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
# Create a string for the formula

cols = df.columns.drop('diagnosis')

formula = 'diagnosis ~ ' + ' + '.join(cols)

print(formula, '\n')
# Run the model and report the results

model = smf.glm(formula=formula, data=X_train, family=sm.families.Binomial())

logistic_fit = model.fit()



print(logistic_fit.summary())
# predict the test data and show the first 5 predictions

predictions = logistic_fit.predict(X_test)

predictions[1:6]
# Note how the values are numerical. 

# Convert these probabilities into nominal values and check the first 5 predictions again.

predictions_nominal = [ "M" if x < 0.5 else "B" for x in predictions]

predictions_nominal[1:6]
print(classification_report(y_test, predictions_nominal, digits=3))



cfm = confusion_matrix(y_test, predictions_nominal)



df_cm = pd.DataFrame(cfm,index = ['TN','TP'],columns=['FN','FP'])

sns.heatmap(df_cm,cbar=True,cmap='viridis', annot=True, fmt='',annot_kws={'size': 20})



true_negative = cfm[0][0]

false_positive = cfm[0][1]

false_negative = cfm[1][0]

true_positive = cfm[1][1]



print('Confusion Matrix: \n', cfm, '\n')



print('True Negative:', true_negative)

print('False Positive:', false_positive)

print('False Negative:', false_negative)

print('True Positive:', true_positive)

print('Correct Predictions', 

      round((true_negative + true_positive) / len(predictions_nominal) * 100, 1), '%')
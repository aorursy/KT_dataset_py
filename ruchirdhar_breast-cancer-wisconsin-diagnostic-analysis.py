#Lets import required libraries before we proceed further
import numpy as np 
import pandas as pd
import matplotlib
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
#Load data into dataframe
dfBCancer = pd.read_csv("../input/data.csv")
dfBCancer.info()
dfBCancer.describe()
dfBCancer.head(5)
#Drop Unnamed: 32 as it only contains Nan
# axis=1: represents column
#inplace = True : represents whether we want to delete column from this dataframe instance, in inplace=False is specified it will return a new
#dataframe having column "Unnamed: 32" deleted but will not change the original datafram
dfBCancer.drop(["Unnamed: 32"], axis=1, inplace=True)
fig, axs = plt.subplots(6, 5, figsize=(16,20))
df=dfBCancer.drop(["id"], axis=1)
g = sns.FacetGrid(df)
k=1
for i in range(6):
    for j in range(5):
        axs[i][j].set_title (df.columns[k])
        g.map(sns.boxplot, "diagnosis",  df.columns[k],  ax=axs[i][j])
        k=k+1
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)
plt.show()
# As we mentioned earlier we can see that above data frame contains block of features representing mean, worst and se
# In order to analyse these features separately we shall be dividing dfBCancer dataframe into 3 dataframes. 
# This will help us in performing further analysis on this dataset

dfCancer_mean = dfBCancer.drop(["id", "diagnosis"], axis=1).iloc[:, 0:10]
dfCancer_se = dfBCancer.drop(["id", "diagnosis"], axis=1).iloc[:, 10:20]
dfCancer_worst = dfBCancer.drop(["id", "diagnosis"], axis=1).iloc[:, 20:30]

print(dfCancer_mean.columns)
print("----------------")
print(dfCancer_se.columns)
print("----------------")
print(dfCancer_worst.columns)
#Lets draw histogram
fig, axs = plt.subplots(2, 5, figsize=(12, 8))

g = sns.FacetGrid(dfCancer_mean)
k=0
for i in range(2):
    for j in range(5):
        g.map(sns.distplot, dfCancer_mean.columns[k], ax=axs[i][j])
        k=k+1
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)
plt.show()
# Compute the correlation matrix
corr = dfCancer_mean.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

 # Set up the matplotlib figure
f, ax = plt.subplots(figsize=(14, 14))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, annot= True, cbar_kws={"shrink": .5})
plt.show()
#Lets draw histogram
fig, axs = plt.subplots(2, 5, figsize=(12, 8))

g = sns.FacetGrid(dfCancer_se)
k=0
for i in range(2):
    for j in range(5):
        g.map(sns.distplot, dfCancer_se.columns[k], ax=axs[i][j])
        k=k+1
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)
plt.show()
# Compute the correlation matrix
corr = dfCancer_se.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

 # Set up the matplotlib figure
f, ax = plt.subplots(figsize=(14, 14))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, annot= True, cbar_kws={"shrink": .5})
plt.show()
#Lets draw histogram
fig, axs = plt.subplots(2, 5, figsize=(12, 8))

g = sns.FacetGrid(dfCancer_worst)
k=0
for i in range(2):
    for j in range(5):
        g.map(sns.distplot, dfCancer_worst.columns[k], ax=axs[i][j])
        k=k+1
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)
plt.show()
# Compute the correlation matrix
corr = dfCancer_worst.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

 # Set up the matplotlib figure
f, ax = plt.subplots(figsize=(14, 14))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 50, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, annot= True, cbar_kws={"shrink": .5})
plt.show()
# I shall be combining all these dataframes back into one result dataframe after droppoing these columns, this time I shall not be using 
#inplace =True, as i dont want original dataframe's values to be lost lets concatenate it back
result = pd.concat([dfCancer_worst.drop(["area_worst", "perimeter_worst", "concave points_worst" , "compactness_worst"], axis=1),
                   dfCancer_se.drop(["area_se", "perimeter_se", "concave points_se" , "compactness_se"], axis=1),
                   dfCancer_mean.drop(["area_mean", "perimeter_mean", "concave points_mean" , "concavity_mean"], axis=1)],
                   axis=1)
# check if resulting dataframe as all the dataset combined except the ones which we didnt want
result.columns
# First convert categorical label into qunatitative values for prediction
factor = pd.factorize( dfBCancer.diagnosis)
diagnosis = factor[0]
definitions = factor[1]
# Split dataset into test and train data
trainX, testX, trainY, testY = train_test_split(result, diagnosis, test_size=0.35, random_state=42)
from sklearn.preprocessing import StandardScaler
stdScalar= StandardScaler().fit(trainX)
trainX = stdScalar.transform(trainX)
testX= stdScalar.transform(testX)
print("Mean of trainX: ", trainX.mean(axis=0), " and standard deviation of trainX: ", trainX.std(axis=0))
print("Mean of testX: ", testX.mean(axis=0), " and standard deviation of testX: ", testX.std(axis=0))
regression = LogisticRegression()
regression.fit(trainX, trainY)

predtrainY = regression.predict(trainX)

print('Accuracy {:.2f}%'.format(accuracy_score(trainY, predtrainY) * 100))
print(classification_report(trainY, predtrainY))
#Create a Confusion matrix

#Reverse factorize
reversefactor = dict(zip(range(len(definitions)),definitions))
y_test = np.vectorize(reversefactor.get)(trainY)
y_pred = np.vectorize(reversefactor.get)(predtrainY)
cm = confusion_matrix(y_test, y_pred)

# plot
fig, ax = plt.subplots()
ax.matshow(cm, cmap=plt.cm.Greens, alpha=0.3)
ax.grid(False)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, s=cm[i,j], va='center', ha='center', fontsize=9)

plt.xlabel('True Predictions')
plt.ylabel('False Predictions')
plt.xticks(range(len(definitions)), definitions.values, rotation=90, fontsize=8)
plt.yticks(range(len(definitions)), definitions.values, fontsize=8)

plt.show()
# use this on the test dataset
predtestY = regression.predict(testX)
print('Accuracy {:.2f}%'.format(accuracy_score(testY, predtestY) * 100))
print(classification_report(testY, predtestY))
#Create a Confusion matrix

#Reverse factorize
reversefactor = dict(zip(range(len(definitions)),definitions))
y_test = np.vectorize(reversefactor.get)(testY)
y_pred = np.vectorize(reversefactor.get)(predtestY)
cm = confusion_matrix(y_test, y_pred)

# plot
fig, ax = plt.subplots()
ax.matshow(cm, cmap=plt.cm.Greens, alpha=0.3)
ax.grid(False)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, s=cm[i,j], va='center', ha='center', fontsize=9)

plt.xlabel('True Predictions')
plt.ylabel('False Predictions')
plt.xticks(range(len(definitions)), definitions.values, rotation=90, fontsize=8)
plt.yticks(range(len(definitions)), definitions.values, fontsize=8)

plt.show()
randclassifier = RandomForestClassifier(max_depth=13,max_features ='sqrt', n_estimators=50,class_weight="balanced", random_state=42)

randclassifier.fit(trainX,trainY)
predtrainY = randclassifier.predict(trainX)

print('Accuracy {:.2f}%'.format(accuracy_score(trainY, predtrainY) * 100))
print(classification_report(trainY, predtrainY))
#use this on the test dataset

predtestY = randclassifier.predict(testX)

print('Accuracy {:.2f}%'.format(accuracy_score(testY, predtestY) * 100))
print(classification_report(testY, predtestY))
#Lets draw validation curve now for Random Forest Model

plt.figure(figsize=(10,8))

param_range = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

train_scores, test_scores = validation_curve(
    randclassifier, result, diagnosis, param_name="n_estimators", param_range=param_range,
    cv=10, scoring="accuracy", n_jobs=1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

lw = 2
# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(param_range, test_scores_mean, 'o-', color="g", label="Cross-validation score")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,alpha=0.1,
                     color="r")
plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

# Create plot
plt.xlabel("Number Of Trees")
plt.ylabel("Accuracy Score")

plt.tight_layout()
plt.legend(loc="best")
plt.ylim(0.8, 1.0)
plt.grid(True)
plt.show()   
svmClassifier = svm.SVC()
svmClassifier.fit(trainX,trainY)
predtrainY=svmClassifier.predict(trainX)

print('Accuracy {:.2f}%'.format(accuracy_score(trainY, predtrainY) * 100))
print(classification_report(trainY, predtrainY))
# use this on the test dataset
predtestY = svmClassifier.predict(testX)

print('Accuracy {:.2f}%'.format(accuracy_score(testY, predtestY) * 100))
print(classification_report(testY, predtestY))

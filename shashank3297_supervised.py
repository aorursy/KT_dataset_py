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

import matplotlib.pyplot as plot

import seaborn as sns

%matplotlib inline

sns.set(style="ticks")
data=pd.read_csv("/kaggle/input/bank-personal/Bank_Personal_Loan_Modelling.csv")
data.head()
data.columns
data.shape
data.info()
# No columns have null data in the file

data.isnull().sum()
# Eye balling the data

data.describe().transpose()
#finding unique data

data.apply(lambda x: len(x.unique()))
sns.pairplot(data.iloc[:,1:])
# there are 52 records with negative experience. Before proceeding any further we need to clean the same

data[data['Experience'] < 0]['Experience'].count()
#clean the negative variable

dfExp = data.loc[data['Experience'] >0]

negExp = data.Experience < 0

mylist = data.loc[negExp]['ID'].tolist() # getting the customer ID who has negative experience
# there are 52 records with negative experience

negExp.value_counts()
for id in mylist:

    age = data.loc[np.where(data['ID']==id)]["Age"].tolist()[0]

    education = data.loc[np.where(data['ID']==id)]["Education"].tolist()[0]

    df_filtered = dfExp[(dfExp.Age == age) & (dfExp.Education == education)]

    if(len(df_filtered)==0):

        #Match with education if there is no match with age and education.

        df_filtered = dfExp[(dfExp.Education == education)]

    if(len(df_filtered)==0):

        df_filtered = dfExp[(dfExp.Age == age)]

    if(len(df_filtered)==0):#Replace with median if there is no match found.

        df_filtered = dfExp[(dfExp.Experience == dfExp.Experience.median())]

    exp = df_filtered['Experience'].median()

    data.loc[data.loc[np.where(data['ID']==id)].index, 'Experience'] = exp
# checking if there are records with negative experience

data[data['Experience'] < 0]['Experience'].count()
data.describe().transpose()
# Making the ZIP Code to object data type as it is a categorical variable. 

data["ZIP Code"]=data["ZIP Code"].astype("object")
sns.boxplot(x='Education',y='Income',hue='Personal Loan',data=data)
sns.boxplot(x="Education", y='Mortgage', hue="Personal Loan", data=data,color='yellow')
sns.boxplot(x=data.Family,y=data.Income,hue=data["Personal Loan"])
sns.countplot(x="Securities Account", data=data,hue="Personal Loan")
sns.countplot(x='Family',data=data,hue='Personal Loan',palette='Set1')
sns.countplot(x='CD Account',data=data,hue='Personal Loan')
sns.distplot( data[data["Personal Loan"] == 0]['CCAvg'], color = 'r')

sns.distplot( data[data["Personal Loan"] == 1]['CCAvg'], color = 'g')
print('Credit card spending of Non-Loan customers: ',data[data["Personal Loan"] == 0]['CCAvg'].median()*1000)

print('Credit card spending of Loan customers    : ', data[data["Personal Loan"] == 1]['CCAvg'].median()*1000)
fig, ax = plot.subplots()

colors = {1:'red',2:'yellow',3:'green'}

ax.scatter(data['Experience'],data['Age'],c=data['Education'].apply(lambda x:colors[x]))

plot.xlabel('Experience')

plot.ylabel('Age')
# Correlation with heat map

import matplotlib.pyplot as plt

import seaborn as sns

corr = data.corr()

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(13,7))

# create a mask so we only see the correlation values once

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask, 1)] = True

sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')
#lets add dummies to education attribute

dummies=pd.get_dummies(data.Education , prefix='Education')

data=data.drop("Education",axis=1)

data=data.join(dummies)
loanY=len(data[data["Personal Loan"] == 1])

loanN=len(data[data["Personal Loan"] == 0])

print('Percentage of people who took loan is ',(loanY/(loanY+loanN))*100,'%')

print('Percentage of people who didn\'t took loan is ',(loanN/(loanY+loanN))*100,'%')
from sklearn.model_selection import train_test_split



X = data.drop(['ID','Personal Loan'],axis=1)

Y = data['Personal Loan']   # Predicted class (1=True, 0=False) (1 X m)



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# 1 is just any random seed number



x_train.head()
print("Original  True Values    : {0} ({1:0.2f}%)".format(len(data.loc[data['Personal Loan'] == 1]), (len(data.loc[data['Personal Loan'] == 1])/len(data.index)) * 100))

print("Original  False Values   : {0} ({1:0.2f}%)".format(len(data.loc[data['Personal Loan'] == 0]), (len(data.loc[data['Personal Loan'] == 0])/len(data.index)) * 100))

print("")

print("Training  True Values    : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train)) * 100))

print("Training  False Values   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train)) * 100))

print("")

print("Test  True Values        : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test)) * 100))

print("Test  False Values       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test)) * 100))

print("")
from sklearn import metrics



from sklearn.linear_model import LogisticRegression



# Fit the model on train

model = LogisticRegression(solver="liblinear")

model.fit(x_train, y_train)

#predict on test

y_predict = model.predict(x_test)





coef_df = pd.DataFrame(model.coef_)

coef_df['intercept'] = model.intercept_

print(coef_df)
model_score = model.score(x_test, y_test)

print(model_score)
cm=metrics.confusion_matrix(y_test, y_predict, labels=[1, 0])

print(cm)

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True)
from sklearn.neighbors import KNeighborsClassifier

NNH = KNeighborsClassifier(n_neighbors= 9 , metric='euclidean')
# Call Nearest Neighbour algorithm



NNH.fit(x_train, y_train)
# For every test data point, predict it's label based on 9 nearest neighbours in this model. The majority class will 

# be assigned to the test data point



predicted_labels = NNH.predict(x_test)

NNH.score(x_test, y_test)
# calculate accuracy measures and confusion matrix

from sklearn import metrics



print("Confusion Matrix")

cm=metrics.confusion_matrix(y_test, predicted_labels, labels=[1, 0])

print(cm)

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True)
from sklearn.naive_bayes import GaussianNB # using Gaussian algorithm from Naive Bayes



# create the model

diab_model = GaussianNB()



diab_model.fit(x_train, y_train.ravel())
diab_train_predict = diab_model.predict(x_train)



from sklearn import metrics



print("Model Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, diab_train_predict)))
diab_test_predict = diab_model.predict(x_test)



from sklearn import metrics



print("Model Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, diab_test_predict)))

print()
print("Confusion Matrix")

cm=metrics.confusion_matrix(y_test, diab_test_predict, labels=[1, 0])

print(cm)

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True)
print("Classification Report")

print(metrics.classification_report(y_test, diab_test_predict, labels=[1, 0]))
X_comp=data.drop(['Personal Loan','ID'],axis=1)

Y_comp=data['Personal Loan']
from sklearn import model_selection

models = []

models.append(('LR', LogisticRegression()))

models.append(('KNN', KNeighborsClassifier(n_neighbors= 9 , metric='euclidean')))

models.append(('NB', GaussianNB()))

# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:

	kfold = model_selection.KFold(n_splits=10, random_state=12345)

	cv_results = model_selection.cross_val_score(model, X_comp, Y_comp, cv=kfold, scoring=scoring)

	results.append(cv_results)

	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)

# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
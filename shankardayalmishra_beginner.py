import pandas as pd

import numpy as np



heart_failure_data = pd.read_csv("../input/heart-failure-data/datasets_727551_1263738_heart_failure_clinical_records_dataset.csv")
heart_failure_data

data = pd.DataFrame(heart_failure_data)
data
#We do not require time column for making predictions so dropping it

new_data = data.drop('time',axis=1)

new_data
#Checking if there is any null value

new_data.isna().sum()
# Importing RandomForestClassifier 

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier





#Setup random seed

np.random.seed(42)



#MAke the data

X = new_data.drop('DEATH_EVENT',axis=1)

y = new_data['DEATH_EVENT']



#Split the data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)



#Instantiate the estimator RFC

clf =RandomForestClassifier(n_estimators=200)

clf.fit(X_train,y_train)



#Evaluate the estimator RandomForestClassifier

clf.score(X_test,y_test)
#Plotting heatmap between different features in the data

import seaborn as sns

import matplotlib.pyplot as plt



sns.heatmap(X_train.corr())

plt.show()
#Counting number of deaths and plotting as countplot

sns.countplot(new_data['DEATH_EVENT'])

plt.show()
#making predictions

y_preds = clf.predict(X_test)

y_preds
#checking y test for comparison between y predicted and y test

np.array(y_test)
#comparing y_test and y_preds

from sklearn.metrics import accuracy_score

accuracy_score(y_preds,y_test)
#predicting death(1) and no death(0) and getting top 5 results

clf.predict_proba(X_test[:5])
#DEATH vs Age plot( categorical_plot)

sns.catplot(x="DEATH_EVENT", y="age", data=new_data);
#plot between age and death considering sex 

sns.catplot(x="DEATH_EVENT", y="age", hue="sex",

            kind="violin", inner="stick", split=True,

            palette="pastel", data=new_data);
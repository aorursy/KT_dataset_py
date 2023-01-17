import numpy as np 

import pandas as pd 



from sklearn import preprocessing

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

import seaborn as sns

sns.set(style="white") #white background style for seaborn plots

sns.set(style="whitegrid", color_codes=True)



#sklearn imports source: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# get titanic & test csv files as a DataFrame



#developmental data (train)

titanic_df = pd.read_csv("../input/train.csv")



#cross validation data (hold-out testing)

test_df    = pd.read_csv("../input/test.csv")



# preview developmental data

titanic_df.head(5)

test_df.head(5)
# check missing values in train dataset

titanic_df.isnull().sum()
sum(pd.isnull(titanic_df['Age']))
# proportion of "Age" missing

round(177/(len(titanic_df["PassengerId"])),4)
ax = titanic_df["Age"].hist(bins=15, color='teal', alpha=0.8)

ax.set(xlabel='Age', ylabel='Count')

plt.show()
# median age is 28 (as compared to mean which is ~30)

titanic_df["Age"].median(skipna=True)
# proportion of "cabin" missing

round(687/len(titanic_df["PassengerId"]),4)
# proportion of "Embarked" missing

round(2/len(titanic_df["PassengerId"]),4)
sns.countplot(x='Embarked',data=titanic_df,palette='Set2')

plt.show()
train_data = titanic_df

train_data["Age"].fillna(28, inplace=True)

train_data["Embarked"].fillna("S", inplace=True)

train_data.drop('Cabin', axis=1, inplace=True)
## Create categorical variable for traveling alone



train_data['TravelBuds']=train_data["SibSp"]+train_data["Parch"]

train_data['TravelAlone']=np.where(train_data['TravelBuds']>0, 0, 1)
train_data.drop('SibSp', axis=1, inplace=True)

train_data.drop('Parch', axis=1, inplace=True)

train_data.drop('TravelBuds', axis=1, inplace=True)
#create categorical variable for Pclass



train2 = pd.get_dummies(train_data, columns=["Pclass"])
train3 = pd.get_dummies(train2, columns=["Embarked"])
train4=pd.get_dummies(train3, columns=["Sex"])

train4.drop('Sex_female', axis=1, inplace=True)
train4.drop('PassengerId', axis=1, inplace=True)

train4.drop('Name', axis=1, inplace=True)

train4.drop('Ticket', axis=1, inplace=True)

train4.head(5)
df_final = train4
test_df["Age"].fillna(28, inplace=True)

test_df["Fare"].fillna(14.45, inplace=True)

test_df.drop('Cabin', axis=1, inplace=True)
test_df['TravelBuds']=test_df["SibSp"]+test_df["Parch"]

test_df['TravelAlone']=np.where(test_df['TravelBuds']>0, 0, 1)



test_df.drop('SibSp', axis=1, inplace=True)

test_df.drop('Parch', axis=1, inplace=True)

test_df.drop('TravelBuds', axis=1, inplace=True)



test2 = pd.get_dummies(test_df, columns=["Pclass"])

test3 = pd.get_dummies(test2, columns=["Embarked"])



test4=pd.get_dummies(test3, columns=["Sex"])

test4.drop('Sex_female', axis=1, inplace=True)



test4.drop('PassengerId', axis=1, inplace=True)

test4.drop('Name', axis=1, inplace=True)

test4.drop('Ticket', axis=1, inplace=True)

final_test = test4
final_test.head(5)
plt.figure(figsize=(15,8))

sns.kdeplot(titanic_df["Age"][df_final.Survived == 1], color="darkturquoise", shade=True)

sns.kdeplot(titanic_df["Age"][df_final.Survived == 0], color="lightcoral", shade=True)

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Age for Surviving Population and Deceased Population')

plt.show()

plt.figure(figsize=(20,8))

avg_survival_byage = df_final[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="LightSeaGreen")

df_final['IsMinor']=np.where(train_data['Age']<=16, 1, 0)
final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)
plt.figure(figsize=(15,8))

sns.kdeplot(df_final["Fare"][titanic_df.Survived == 1], color="darkturquoise", shade=True)

sns.kdeplot(df_final["Fare"][titanic_df.Survived == 0], color="lightcoral", shade=True)

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Fare for Surviving Population and Deceased Population')

# limit x axis to zoom on most information. there are a few outliers in fare. 

plt.xlim(-20,200)

plt.show()
sns.barplot('Pclass', 'Survived', data=titanic_df, color="darkturquoise")

plt.show()
sns.barplot('Embarked', 'Survived', data=titanic_df, color="teal")

plt.show()
sns.barplot('TravelAlone', 'Survived', data=df_final, color="mediumturquoise")

plt.show()
sns.barplot('Sex', 'Survived', data=titanic_df, color="aquamarine")

plt.show()
df_final.head(10)
cols=["Age", "Fare", "TravelAlone", "Pclass_1", "Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 

X=df_final[cols]

Y=df_final['Survived']
import statsmodels.api as sm

from scipy import stats

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

logit_model=sm.Logit(Y,X)

result=logit_model.fit()

print(result.summary())
cols2=["Age", "Pclass_1", "Pclass_2","Embarked_C","Embarked_S","Sex_male"]  

X2=df_final[cols2]

Y=df_final['Survived']



logit_model=sm.Logit(Y,X2)

result=logit_model.fit()



print(result.summary())
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X2, Y)



logreg.score(X2, Y)
#from sklearn.linear_model import LogisticRegression

#from sklearn import metrics

#logreg = LogisticRegression()

#logreg.fit(X2, Y)



#X_test = final_test[cols2]

#y_test = final_test['Survived']



#y_pred = logreg.predict(X_test)

#print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.model_selection import train_test_split

train, test = train_test_split(df_final, test_size=0.2)
#re-fit logistic regression on new train sample



cols2=["Age", "Pclass_1", "Pclass_2","Embarked_C","Embarked_S","Sex_male"] 

X3=train[cols2]

Y3=train['Survived']

logit_model3=sm.Logit(Y3,X3)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



logreg = LogisticRegression()

logreg.fit(X3, Y3)

logreg.score(X3, Y3)
from sklearn import metrics

logreg.fit(X3, Y3)



X3_test = test[cols2]

Y3_test = test['Survived']



Y3test_pred = logreg.predict(X3_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X3_test, Y3_test)))
# Model's in sample AUC



from sklearn.metrics import roc_auc_score

logreg.fit(X3, Y3)

Y3_pred = logreg.predict(X3)



y_true = Y3

y_scores = Y3_pred

roc_auc_score(y_true, y_scores)
#Visualizing the model's ROC curve (**source for graph code given below the plot)

from sklearn.metrics import roc_curve, auc

logreg.fit(X3, Y3)



y_test = Y3_test

X_test = X3_test

 

# Determine the false positive and true positive rates

FPR, TPR, _ = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

 

# Calculate the AUC



roc_auc = auc(FPR, TPR)

print ('ROC AUC: %0.3f' % roc_auc )

 

# Plot of a ROC curve

plt.figure(figsize=(10,10))

plt.plot(FPR, TPR, label='ROC curve (area = %0.3f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([-0.05, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve (Test Sample Performance)')

plt.legend(loc="lower right")

plt.show()
from sklearn.ensemble import RandomForestClassifier



cols=["Age", "Fare", "TravelAlone", "Pclass_1", "Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 

X=df_final[cols]

Y=df_final['Survived']



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X, Y)

random_forest.score(X, Y)
final_test_RF=final_test[cols]

Y_pred_RF = random_forest.predict(final_test_RF)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred_RF

    })

submission.to_csv('titanic.csv', index=False)
from sklearn import tree

import graphviz

tree1 = tree.DecisionTreeClassifier(criterion='gini', splitter='best',max_depth=3, min_samples_leaf=20)
cols=["Age", "Fare", "TravelAlone", "Pclass_1", "Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 

X_DT=df_final[cols]

Y_DT=df_final['Survived']



tree1.fit(X_DT, Y_DT)
import graphviz 

tree1_view = tree.export_graphviz(tree1, out_file=None, feature_names = X_DT.columns.values, rotate=True) 

tree1viz = graphviz.Source(tree1_view)

tree1viz
final_test_DT=final_test[cols]
Y_pred_DT = tree1.predict(final_test_DT)
# submission = pd.DataFrame({

#        "PassengerId": test_df["PassengerId"],

#        "Survived": Y_pred_DT

#    })

#submission.to_csv('titanic.csv', index=False)
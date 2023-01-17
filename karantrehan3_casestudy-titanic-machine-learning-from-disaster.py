# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
#Data analysis

import pandas as pd

import numpy as np



#Statistical Libraries

from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm

import scipy.stats as st

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.feature_selection import RFE 

from sklearn.svm import SVR



from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



#Machine Learning

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier



#Data Visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set_palette('bright')

sns.set_style("darkgrid")
#Reading the Datasets into two separate DataFrames 'train_df' and 'test_df'

train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
#Taking a look at the first 5 rows of training data.

train_df.head()
#Taking a look at the first 5 rows of testing data.

test_df.head()
#Checking information about the Training DataFrame

train_df.info()
#Checking information about the Testing DataFrame

test_df.info()
#Viewing the Statistical Measures/Details of the Training DataFrame

train_df.describe()
#Viewing the Statistical Measures/Details of the Testing DataFrame

test_df.describe()
#Checking for duplicacy in both the DataFrames using '.duplicated()' method and then checking the number of rows using

# '.shape[0]'

print("Number of Duplicate Rows in Training DataFrame:" , train_df[train_df.duplicated()].shape[0])

print("Number of Duplicate Rows in Testing DataFrame:" , test_df[test_df.duplicated()].shape[0])
#Checking the Percentage of Columns having Missing Values in both the DataFrames

print('-+-'*10)

print('Training Data')

print(round(train_df.isnull().sum()/len(train_df)*100,2))

print('-+-'*10)

print('Testing Data')

print(round(test_df.isnull().sum()/len(test_df)*100,2))

print('-+-'*10)
#Dropping 'cabin' variable because it has 77.10% and 78.23% missing values in 'train_df' and 'test_df' respectively

train_df.drop(columns=['Cabin'],axis=1,inplace=True)

test_df.drop(columns=['Cabin'],axis=1,inplace=True)
#Grouping the DataFrames according to their 'Pclass' and their 'Sex'.

#After Grouping, calculating the median 'Age' based on the above mentioned features.

print('-+-'*10)

print('Training Data')

print(train_df.groupby(['Pclass', 'Sex']).median()['Age'])

print('-+-'*10)

print('Testing Data')

print(test_df.groupby(['Pclass', 'Sex']).median()['Age'])

print('-+-'*10)
#Filling the Missing Values in 'Age' column based on the median values calculated in the above cell.

train_df['Age'] = train_df.groupby(['Pclass', 'Sex'])['Age'].apply(lambda a:a.fillna(a.median()))

test_df['Age'] = test_df.groupby(['Pclass', 'Sex'])['Age'].apply(lambda a:a.fillna(a.median()))



#Filling the Missing Values in 'Embarked' column based on the mode value.

mode = train_df['Embarked'].mode()

train_df['Embarked'] = train_df['Embarked'].fillna(str(mode[0]))



#Filling the Missing Values in 'Fare' column based on the median value.

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
#Again Checking the Percentage of Columns having Missing Values in case all the values have not been imputed.

print('-+-'*10)

print('Training Data')

print(round(train_df.isnull().sum()/len(train_df)*100,2))

print('-+-'*10)

print('Testing Data')

print(round(test_df.isnull().sum()/len(test_df)*100,2))

print('-+-'*10)
#Dropping 'Ticket' and 'Name' columns.

train_df.drop(columns=['Ticket','Name'],axis=1,inplace=True)

test_df.drop(columns=['Ticket','Name'],axis=1,inplace=True)
#Combining the 'SibSp' , 'Parch' and the person's own count to create a new 'Family' variable.

train_df['Family'] = train_df['SibSp'] + train_df['Parch'] + 1

test_df['Family'] = test_df['SibSp'] + test_df['Parch'] + 1
#Calculating Fare Per Person

train_df['FarePP'] = train_df['Fare']/train_df['Family']

test_df['FarePP'] = test_df['Fare']/test_df['Family']
#Dropping 'SibSp','Parch' and 'Fare' columns as new columns have been created.

train_df.drop(columns=['SibSp','Parch','Fare'],axis=1,inplace=True)

test_df.drop(columns=['SibSp','Parch','Fare'],axis=1,inplace=True)
#Plotting PairPlot to check the relations among the Variables

sns.pairplot(data=train_df,hue='Survived')

plt.show()
#Creating a Dataframe based on the number of people who survived and who did not.

sur = train_df.groupby('Survived')['Survived'].count().to_frame()

sur.rename(columns={'Survived' : 'No. of Passengers'} , inplace = True)

sns.barplot(data=sur, x=sur.index , y = 'No. of Passengers')

plt.show()
#Plotting 'Pclass' vs 'Survived' Graph to see how many people survived from each class.

fig, axs = plt.subplots(1,2,figsize = (15,5))

sns.barplot(data=train_df,x='Pclass',y='Survived',ax=axs[0])

sns.barplot(data=train_df,x='Pclass',y='Survived',hue='Sex',ax=axs[1])

plt.show()
#Confirming the above plots Statistically

psur = round(train_df[['Survived','Pclass']].groupby('Pclass').mean()*100 , 2)

psur = pd.DataFrame(psur)

psur
psurse = round(train_df[['Survived','Pclass','Sex']].groupby(['Sex','Pclass']).mean()*100 , 2)

psurse = pd.DataFrame(psurse)

psurse
#Plotting 'Sex' vs 'Survived' Graph to see how many people survived based on Gender.

sns.barplot(data=train_df,x='Sex',y='Survived')

plt.show()
#Confirming the above plot Statistically

agesur = round(train_df[['Survived','Sex']].groupby('Sex').mean()*100 , 2)

agesur = pd.DataFrame(agesur)

agesur
#Checking the Distribution Plot of 'Age' to see ranges of people present on the ship.

sns.distplot(train_df['Age'],color='k',bins=10,kde_kws=dict(linewidth=4))

plt.show()
#Creating Binned Age and Plotting the graph of 'Age' vs 'Survived'

binnedAge = pd.cut(train_df['Age'], bins=list(range(0,90,5)), include_lowest=True)

labels = [str(i) + ' to ' + str(j) for i , j in zip(range(0,90,5) , range(5,95,5))]





fig, axs = plt.subplots(2,1,figsize = (20,15))

sns.barplot(data=train_df,x=binnedAge,y='Survived',ax=axs[0])

sns.barplot(data=train_df,x=binnedAge,y='Survived',hue='Sex',ax=axs[1])

axs[0].set_xticklabels(labels,rotation=30)

axs[1].set_xticklabels(labels,rotation=30)

plt.show()
#Plotting 'Embarked' vs 'Survived' Graph to see how many people had the chance to survive based on the Port they boarded from.

fig, axs = plt.subplots(1,2,figsize = (15,5))

sns.barplot(data=train_df,x='Embarked',y='Survived',ax=axs[0])

sns.barplot(data=train_df,x='Embarked',y='Survived',hue='Sex',ax=axs[1])

plt.show()
#Plotting 'Family' vs 'Survived' Graph to see how many people survived based on Family Count.

fig, axs = plt.subplots(1,2,figsize = (15,5))

sns.barplot(data=train_df,x='Family',y='Survived',ax=axs[0])

sns.barplot(data=train_df,x='Family',y='Survived',hue='Sex',ax=axs[1])

plt.show()
#Creating separate Lists for Numerical and Categorical Features

num_features=[i for i in train_df.columns if train_df[i].dtypes!='O']

cat_features=[i for i in train_df.columns if train_df[i].dtypes=='O']
#Checking Skewness of the Continuous Variables 'Age' and 'FarePP'

fig, axs = plt.subplots(1,2,figsize = (15,5))

sns.distplot(train_df['Age'],color='#FF6050',kde_kws=dict(linewidth=4),ax=axs[0])

sns.distplot(train_df['FarePP'],color='orange',kde_kws=dict(linewidth=4),ax=axs[1])

plt.show()
#Calculating Skewness using the Skew function from the scipy.stats library

print('Age = ' , st.skew(train_df['Age']) , ' FarePP = ',st.skew(train_df['FarePP']))
#Removing Skewness from the Variables and then checking there distribution plots.

age_train = np.sqrt(train_df['Age'])

farepp_train = np.power(train_df['FarePP'],0.20)



fig, axs = plt.subplots(1,2,figsize = (15,5))

sns.distplot(age_train,color='#FF6050',kde_kws=dict(linewidth=4),ax=axs[0])

sns.distplot(farepp_train,color='orange',kde_kws=dict(linewidth=4),ax=axs[1])

plt.show()

print('Age = ' , st.skew(age_train) , ' FarePP = ',st.skew(farepp_train))
#Mapping 'Sex' to 0 and 1

# Male=0 and Female=1

train_df['Sex'] = train_df['Sex'].map({'male':1, 'female':0})

test_df['Sex'] = test_df['Sex'].map({'male':1, 'female':0})
#Mapping 'Embarked' to 0 and 1

# Q=0 , S=1 and C=0

train_df['Embarked'] = train_df['Embarked'].map({'Q':2, 'S':1, 'C':0})

test_df['Embarked'] = test_df['Embarked'].map({'Q':2, 'S':1, 'C':0})
#Dropping PassengerId because it contains unique discrete values, like an index.

train_df.drop(columns=['PassengerId'] , axis=1 , inplace = True)

train_df
#Removing 'PassengerId' from and storing it in test_pid because we will use it later

test_pid = test_df.pop('PassengerId')

test_df
#Removing the features mentioned below because 1 has been removed and 2 are Ordinal

num_features.remove('PassengerId')

num_features.remove('Pclass')

num_features.remove('Survived')
#Scaling the Features between 0 - 1, for easier and efficient performance by the Model

scaler = MinMaxScaler()

train_df[num_features] = scaler.fit_transform(train_df[num_features])

test_df[num_features] = scaler.fit_transform(test_df[num_features])



print('-+-'*25)

print('Training Data')

print(train_df.describe())

print('-+-'*25)

print('Testing Data')

print(test_df.describe())

print('-+-'*25)
#Splitting into Training and Testing Variables. Separating the Dependent Variable

y_train = train_df.pop('Survived')

X_train = train_df.copy()



X_test  = test_df.copy()



#Checking Shape of the Variables

X_train.shape, y_train.shape, X_test.shape
#Adding a constant manually because GLM otherwise fits the line through the origin

X_train_sm = sm.add_constant(X_train)



#Create a first fitted model

logglm = sm.GLM(y_train,X_train_sm,family=sm.families.Binomial()).fit()



#Viewing Summary

print(logglm.summary())
#Defining a function which will calculate the VIF values and store them in a DataFrame

#High VIF Means High Multicollinearity

def calculateVIF(X_train_lm):

    vif = pd.DataFrame()

    vif['Features'] = X_train_lm.columns

    vif['VIF'] = [variance_inflation_factor(X_train_lm.values, i) for i in range(X_train_lm.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False).reset_index()

    vif = vif.drop(columns = ['index'],axis = 1)

    return vif
#Calculating VIF

vif = calculateVIF(X_train_sm.drop(columns=['const'],axis=1))

vif
X_train_sm1 = sm.add_constant(X_train[X_train_sm.drop(columns=['const','Embarked','FarePP'],axis=1).columns])

logglm1 = sm.GLM(y_train,X_train_sm1,family=sm.families.Binomial()).fit()

print(logglm1.summary())
#Fitting the Training Data to RFE model

estimator = SVR(kernel="linear")

rfe = RFE(estimator,3, step=1)

rfe = rfe.fit(X_train, y_train)
#checking the listing of the features that got selected

list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]

col #Printing the list of Columns that are selected by RFE
#Again building a GLM Model

X_train_rfe = X_train[col]

X_train_rfe = sm.add_constant(X_train_rfe)

logglmrfe = sm.GLM(y_train,X_train_rfe,family=sm.families.Binomial()).fit()

print(logglmrfe.summary())
#Defining a function print_matrix that will print the Confusion Matrix

import sklearn.metrics as sklm #For Performance Measures

def print_matrix(labels, scores):

    metrics = sklm.precision_recall_fscore_support(labels,scores)

    conf = sklm.confusion_matrix(labels,scores)

    fig, (ax1,ax2) = plt.subplots(figsize=(15,7), ncols=2, nrows=1)

    

    left   =  0.125  # the left side of the subplots of the figure

    right  =  0.9    # the right side of the subplots of the figure

    bottom =  0.1    # the bottom of the subplots of the figure

    top    =  0.9    # the top of the subplots of the figure

    wspace =  .5     # the amount of width reserved for blank space between subplots

    hspace =  1.1    # the amount of height reserved for white space between subplots



    # This function actually adjusts the sub plots using the above paramters

    plt.subplots_adjust(

            left    =  left, 

            bottom  =  bottom, 

            right   =  right, 

            top     =  top, 

            wspace  =  wspace, 

            hspace  =  hspace

    )

    

    sns.set(font_scale=1.4)#for label size

    

    ax1.set_title('Confusion Matrix\n')

    axes = ['Positive' , 'Negative']

    g1 = sns.heatmap(conf, cmap="Greens", annot=True,annot_kws={"size": 16} , xticklabels = axes , yticklabels = axes , ax=ax1)# font size

    g1.set_ylabel('Actual Label')

    g1.set_xlabel('Predicted Label')

    

    print('\nAccuracy :  %0.2f' % sklm.accuracy_score(labels, scores) , '\n') #Printing Accuracy Of the Model



    ax2.set_title('Performance Measure\n')

    xaxes = ['Positive' , 'Negative']

    yaxes = ['Precision' , 'Recall' , 'F-Score' , 'NumCase']

    g2 = sns.heatmap(metrics, cmap="Greens", annot=True,annot_kws={"size": 16} , xticklabels = xaxes , yticklabels = yaxes , ax=ax2)

   

    plt.yticks(rotation=0) 

    plt.show()
#Creating an instance of LogisticRegression

logreg = LogisticRegression()



#Fitting the data.

logreg.fit(X_train, y_train)

logreg_Y_pred=logreg.predict(X_test)



#Storing the values predicted on training Data

logreg_Y_pred_train=logreg.predict(X_train)



#Checking the Accuracy

logreg_accuracy=logreg.score(X_train,y_train)

logreg_accuracy
#Plotting the ROC Curve



# generate a no skill prediction (majority class)

ns_probs = [0 for _ in range(len(y_train))]

lr_probs = logreg.predict_proba(X_train)

# keep probabilities for the positive outcome only

lr_probs = lr_probs[:, 1]

# calculate scores

ns_auc = roc_auc_score(y_train, ns_probs)

lr_auc = roc_auc_score(y_train, lr_probs)

# summarize scores

print('No Skill: ROC AUC=%.3f' % (ns_auc))

print('Logistic: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves

ns_fpr, ns_tpr, _ = roc_curve(y_train, ns_probs)

lr_fpr, lr_tpr, _ = roc_curve(y_train, lr_probs)

# plot the roc curve for the model

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()
print_matrix(y_train,logreg_Y_pred_train) #displaying the produced analysis in confusion matrix
#Creating an instance of DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()



#Fitting the data.

decision_tree.fit(X_train, y_train)

decision_tree_Y_pred = decision_tree.predict(X_test)



#Storing the values predicted on training Data

decision_tree_Y_pred_train = decision_tree.predict(X_train)



#Checking the accuracy

decision_tree_accuracy = decision_tree.score(X_train, y_train)

decision_tree_accuracy
print_matrix(y_train,decision_tree_Y_pred_train) #displaying the produced analysis in confusion matrix
#Creating an instance of RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)



#Fitting the data.

random_forest.fit(X_train, y_train)

random_forest_Y_pred = random_forest.predict(X_test)



#Storing the values predicted on training Data

random_forest_Y_pred_train = random_forest.predict(X_train)



#Checking the accuracy

random_forest.score(X_train, y_train)

random_forest_accuracy = random_forest.score(X_train, y_train)

print(random_forest_accuracy)



#Feature Importance

random_forest.feature_importances_
print_matrix(y_train,random_forest_Y_pred_train) #displaying the produced analysis in confusion matrix
#Creating the Model DataFrame based on accuracy

models = pd.DataFrame({

    'Model': ['Logistic Regression','Decision Tree','Random Forest'],

    'Score': [logreg_accuracy,decision_tree_accuracy, random_forest_accuracy]})

models.sort_values(by='Score', ascending=False)
#Creating submission file from each model

logreg_submission = pd.DataFrame({"PassengerId": test_pid, "Survived": logreg_Y_pred})

logreg_submission.to_csv('logreg_submission.csv', index=False)



decision_tree_submission = pd.DataFrame({"PassengerId": test_pid, "Survived": decision_tree_Y_pred})

decision_tree_submission.to_csv('decision_tree_submission.csv', index=False)



random_forest_submission = pd.DataFrame({"PassengerId": test_pid, "Survived": random_forest_Y_pred})

random_forest_submission.to_csv('random_forest_submission.csv', index=False)
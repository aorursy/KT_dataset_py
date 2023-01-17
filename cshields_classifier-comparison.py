# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#The train data has features (independent variables) and targets (dependent variable).
#Feature examples are Name, Age, or Fare. The target is if the passenger survived.
train=pd.read_csv('../input/train.csv')
#We print the first 5 rows of the train dataset to show what this looks like:
ntrain = train.shape[0] #this gets the number rows in the traning dataset
print("Training data (",ntrain,"rows)")

#Display the data in Pandas: .head(n_rows) shows the first n rows of the DataFrame
display(train.head(10))

#The test dataset is used to test how well the classifier performs
#Test data only has features, the targets are empty and must be predicted
test=pd.read_csv('../input/test.csv')

#Lets looks at the test data...
ntest = test.shape[0]
print("Test data (",ntest,"rows), notice that the survived column (target) is missing!")
display(test.head(10))

df_all=pd.concat([train,test],axis=0)
p_id=df_all['PassengerId']
print(df_all.info())
#First we get value to impute on the missing entries
age_med=df_all['Age'].median()
fare_med=df_all['Fare'].median()
emb_mode=df_all['Embarked'].mode()

#Impute values
df_all['Age']=df_all['Age'].fillna(age_med)
df_all['Fare']=df_all['Fare'].fillna(fare_med)
df_all['Embarked']=df_all['Embarked'].fillna(emb_mode)

#Create new features that might be helpful
#Family size is the number of parents/children plus siblings/spouses
df_all['Family Size']=df_all['Parch']+df_all['SibSp'] 

#Convert cabin to a dummy variable, 0 if null and 1 if it has value
df_all['Cabin']=df_all['Cabin'].notnull().astype('int')

#Drop other columns
df_all=df_all.drop(['Name','PassengerId','Ticket'],axis=1)

#Convert all categorical varaibles to dummy variables
#This is called one-hot encoding
df_all=pd.get_dummies(df_all,drop_first=True)

display(df_all.head(10))
#Split the data
train=df_all[:ntrain]

#Look at the correlations in the dataset
corr=train.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#Based on correlations drop SibSp, Family Size, Embarked_S, Embarked_Q, Age
# df_all=df_all.drop(['SibSp','Family Size','Embarked_Q',],axis=1)
# df_all=df_all.drop(['SibSp','Family Size','Embarked_S','Embarked_Q','Fare','Parch','Cabin'],axis=1)
print (df_all.columns)
train=df_all[:ntrain]
#Time to test the classifiers
#load them:
from matplotlib.colors import ListedColormap
from sklearn.model_selection import StratifiedShuffleSplit,KFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression 

#Visualize the classifier results

names = ["Nearest Neighbors","Decision Tree", 'Logistic Regression',
         "Linear SVM", "RBF SVM","Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=3),
    LogisticRegression(),
    SVC(gamma=2, C=1),
    SVC(kernel="linear", C=0.025),
    GaussianNB(),
    ]

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
i=1
figure = plt.figure(figsize=(21, 6))
for sex in train['Sex_male'].unique():
    h=0.2
    x_min, x_max = train['Fare'].loc[train['Sex_male']==sex].min() - .5, train['Fare'].loc[train['Sex_male']==sex].max() + .5
    y_min, y_max = train['Age'].loc[train['Sex_male']==sex].min() - .5, train['Age'].loc[train['Sex_male']==sex].max() + .5
#     x_min, x_max = train['Fare'].min() - .5, train['Fare'].max() + .5
#     y_min, y_max = train['Age'].min() - .5, train['Age'].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))    

    ax = plt.subplot(2, len(classifiers) + 1, i)

    ax.scatter(train['Fare'].loc[(train['Sex_male']==sex) & (train['Survived']==1.0)],
               train['Age'].loc[(train['Sex_male']==sex) & (train['Survived']==1.0)],
               c='#0000FF',edgecolors='k')
    
    ax.scatter(train['Fare'].loc[(train['Sex_male']==sex) & (train['Survived']==0.0)],
               train['Age'].loc[(train['Sex_male']==sex) & (train['Survived']==0.0)],
               c='#FF0000',edgecolors='k')
    
    # ax = plt.subplot(2, len(classifiers) + 1, 7)
    # ax.scatter(train['Fare'].loc[train['Sex_male']==0],train['Age'].loc[train['Sex_male']==0],c=train['Survived'].loc[train['Sex_male']==0],cmap=cm_bright)
    if i == 1:
        ax.set_title("Input Data")
        ax.set_ylabel('Males, Age')
        plt.legend(['Survived','Died'])
#         ax = plt.gca()
#         legend = ax.get_legend()
#         legend.legendHandles[0].set_color(cm_bright(0.0))
#         legend.legendHandles[1].set_color(cm_bright(1.0))
        
    if i==8:
        ax.set_ylabel('Females, Age')
        ax.set_xlabel('Fare')
    i+=1
    
    for name,clf in zip(names,classifiers):
        ax = plt.subplot(2, len(classifiers) + 1, i)
        clf.fit(train[['Fare','Age',]].loc[train['Sex_male']==sex], train['Survived'].loc[train['Sex_male']==sex])
        score = clf.score(train[['Fare','Age',]].loc[train['Sex_male']==sex], train['Survived'].loc[train['Sex_male']==sex])
        

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(train['Fare'].loc[train['Sex_male']==sex],train['Age'].loc[train['Sex_male']==sex], c=train['Survived'].loc[train['Sex_male']==sex],cmap=cm_bright,
                   edgecolors='k')
        # and testing points
#         ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
#                    edgecolors='k', alpha=0.6)

#         ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
#                 size=15, horizontalalignment='right')
       
        if sex == 1:
            ax.set_title(name) 
        if sex==0:
            ax.set_xlabel('Fare')
        
        i+=1

plt.tight_layout()

#Now test the models in with cross-validation
feature_columns=['Pclass','Age','Sex_male','Parch','Fare','Embarked_S','Cabin']#list(train.columns) 'Cabin'
# feature_columns.remove('Survived')
y_train=train['Survived'].values
x_train=train[feature_columns].values

log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", 
         "Decision Tree", "Naive Bayes", 'Logistic Regression']

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=3),
    GaussianNB(),
    LogisticRegression()]

#Split the data into test and training sets
#Use these split to train the classifier and test it mutliple times
#Helps us estimate the performance of the classifiers without "cheating" by scoring the classifier on data we've already seen
# sss = StratifiedShuffleSplit(n_splits=10, test_size=0.33, random_state=0)
split=KFold(n_splits=10, )

X = x_train
y = y_train

acc_dict = {}
for train_index, test_index in split.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    for name,clf in zip(names,classifiers):
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc

for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
#Decision Tree looks the best
#Train it on the full dataset and predict the test dataste
clf=DecisionTreeClassifier(max_depth=3)
# clf=SVC(kernel="linear", C=0.025)
# clf=GaussianNB()
# clf=LogisticRegression()
# clf=KNeighborsClassifier(3)

y_train=train['Survived'].values
x_train=train[feature_columns].values

clf.fit(X=x_train,y=y_train)

test_df=df_all[ntrain:]
x_test=test_df[feature_columns].values

#Now predict our results
results=clf.predict(x_test)

#Convert the results to int datatypes (real numbers)
results=[int(i) for i in results]

#Get passenger id's from test set with the .iloc command
results_id=p_id.iloc[ntrain:].values

#Create a dataframe for submission
submission=pd.DataFrame({'PassengerId':results_id,'Survived':results})

#Check what the submission looks like
display(submission.head(10))

#Save the dataFrame as a .csv (save to Kaggle)
submission.to_csv('submisison.csv',index=False)

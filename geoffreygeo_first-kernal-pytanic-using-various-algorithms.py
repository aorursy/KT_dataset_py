# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

#Machine learning models from sklearn
from sklearn.naive_bayes import GaussianNB,MultinomialNB  # Gaussian naive Bayes classifier
from sklearn.preprocessing import LabelEncoder
from IPython.display import display
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from IPython.display import display
import os
print(os.listdir("../input"))

#loading the dataset using pandas 
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
gen_sub=pd.read_csv('../input/gender_submission.csv')

train = train.drop(['Name','Ticket','PassengerId'],axis = 1)
test = test.drop(['Name','Ticket'],axis = 1)
print("The shape of Train data is {}\nThe shape of test data is {}\nThe shape of general submission is {}".format(train.shape,test.shape,gen_sub.shape))
# Any results you write to the current directory are saved as output.
display(train.head())
display(test.head())
display(train.describe())
#checking out missing values
print(train.isnull().sum(axis=0))
#train = train.replace([np.inf, -np.inf], np.nan)
#test = test.replace([np.inf, -np.inf], np.nan)
#train = train.dropna()
#test = test.dropna()
train = train.interpolate()
test = test.interpolate()

print("The result after interpolation \n{}".format(train.isnull().sum()))
display(train[train['Embarked'].isnull()])
fig, (ax1, ax2,ax3) = plt.subplots(ncols=3, sharey=True,figsize=(16,10))
sns.swarmplot(x="Embarked", y="Fare", hue="Pclass", data=train,ax=ax1);
sns.stripplot(x="Embarked", y="Fare", hue="Pclass", data=train,ax=ax2);
sns.pointplot(x="Embarked", y="Fare", hue="Pclass", data=train,ax=ax3);
train["Embarked"] = train["Embarked"].fillna('S')
cabin = train.Cabin.value_counts().index.tolist()
#cabin_NA =train[train['Cabin'] == 'NaN']
cabin_NA = train[train.isnull().any(axis=1)]
#null_data = train[train.isnull().any(axis=1)]
display(cabin_NA.shape)
def plot_correlation_map( df ):
    corr = titanic.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )
def plot_corr(df):
    corr=df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
#print(test['Age'].isnull().sum())
train =pd.get_dummies(train, columns=["Sex", "Embarked"], prefix=["sex", "embarked"])
test = pd.get_dummies(test,columns=['Sex','Embarked'],prefix=['sex','embarked'])
# CtoN={"Sex":{"male":0,"female":1},
#             "Embarked":{"S":0,"C":1,"Q":2}}
# train.replace(CtoN, inplace=True)
# test.replace(CtoN,inplace=True)
#using pandas.cat.codes to replaces categorical data to numerical 
# train["Sex"] = train["Sex"].astype('category')
# train["Embarked"] = train["Embarked"].astype('category')
# test["Sex"] = train["Sex"].astype('category')
# test["Embarked"] = test["Embarked"].astype('category')
# train["Sex"] = train["Sex"].cat.codes
# test["Sex"] = test["Sex"].cat.codes
# train['Embarked']=train['Embarked'].cat.codes
# test['Embarked']=test['Embarked'].cat.codes
#train.head()
#print(test['Age'].isnull().sum())
display(train.head())
#plotting  the correlation matrix
plot_corr(train)
g = sns.FacetGrid(train, col="sex_male", row="Survived", margin_titles=True)
g.map(plt.hist, "Age",color="skyblue");
g = sns.FacetGrid(train, col="Pclass", row="Survived", margin_titles=True)
g.map(plt.hist, "Pclass",color="lightgreen");
#considering features with high co-relation
traindata = pd.DataFrame(train[['Pclass','Age','SibSp','Parch','Fare']]).values
print("The type of train data set is {}\nThe shape of the dataframe is {}".format(type(traindata),traindata.shape))

#format the data and consider features which has high co-relation 
trainlabel = pd.DataFrame(train[['Survived']]).values.ravel()
traindata =  pd.DataFrame(train[['Pclass','Age','SibSp','Parch','Fare','sex_female','sex_male','embarked_C','embarked_Q','embarked_S']]).values
testdata =  pd.DataFrame(test[['Pclass','Age','SibSp','Parch','Fare','sex_female','sex_male','embarked_C','embarked_Q','embarked_S']]).values
testlabel = pd.DataFrame(gen_sub[['Survived']]).values.ravel()
print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape)
#creating a model 
clf = GaussianNB()
clf1 = MultinomialNB()
clf.fit(traindata,trainlabel)
clf1.fit(traindata,trainlabel)

#predicting values 
predicted_values = clf.predict(testdata)
accuracy = accuracy_score(testlabel,predicted_values)
print("The accuracy of the model {}".format(accuracy*100))

#predicting values 
predicted_values_MNB = clf1.predict(testdata)
accuracy_MNB = accuracy_score(testlabel,predicted_values_MNB)
print("The accuracy of the model {}".format(accuracy_MNB*100))


confusion_matrix = (confusion_matrix(testlabel, predicted_values))

print("The confusion matrix of the model is \n{}".format(confusion_matrix))
import matplotlib.pyplot as plt
import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(testlabel, predicted_values, normalize=True)
plt.show()
from sklearn.linear_model import LinearRegression
#creating models
#Building the model
#Naive bayes,Random-forest,SVM,DecisionTree,NeuralNetwork,KNN
LR = LinearRegression().fit(traindata,trainlabel)
NB_M=MultinomialNB().fit(traindata,trainlabel)
GB = GaussianNB().fit(traindata,trainlabel)
KNN=KNeighborsClassifier().fit(traindata,trainlabel)
DT=DecisionTreeClassifier().fit(traindata,trainlabel)
SVM = svm.SVC().fit(traindata,trainlabel)
RF = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456).fit(traindata,trainlabel)
MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1).fit(traindata,trainlabel)

#predicting with various Models and testing their Accuracy
predicted_values_LR = LR.predict(testdata)
predicted_values_NBM = NB_M.predict(testdata)
predicted_values_GB=GB.predict(testdata)
predicted_values_KNN = KNN.predict(testdata)
predicted_values_DT = DT.predict(testdata)
predicted_values_SVM = SVM.predict(testdata)
predicted_values_RF = RF.predict(testdata)
predicted_values_MLP = MLP.predict(testdata)

#Testing the accuracy of each model 
predictions=dict()
predicted_values_LR[predicted_values_LR > .5] = 1
predicted_values_LR[predicted_values_LR <=.5] = 0
accuracy_LR=accuracy_score(testlabel,predicted_values_LR)
predictions['Linear Regression']=accuracy_LR*100
acurracy_NB_M = accuracy_score(testlabel,predicted_values_NBM)
predictions['NaiveBayes_Multinomial']=acurracy_NB_M*100
acurracy_KNN = accuracy_score(testlabel,predicted_values_KNN)
predictions['KNearest']=acurracy_KNN*100
accuracy_NB_GM=accuracy_score(testlabel,predicted_values_GB)
predictions['NaiveBayes_Gaussian']=accuracy_NB_GM*100
acurracy_DT = accuracy_score(testlabel,predicted_values_DT)
predictions['DecisionTree']=acurracy_DT*100
acurracy_SVM = accuracy_score(testlabel,predicted_values_SVM)
predictions['Support Vector Machine']=acurracy_SVM*100
acurracy_RF = accuracy_score(testlabel,predicted_values_RF)
predictions['RandomForest']=acurracy_RF*100
acurracy_MLP = accuracy_score(testlabel,predicted_values_MLP)
predictions['MultiNeuralNetwork']=acurracy_MLP*100
fig, (ax1) = plt.subplots(ncols=1, sharey=True,figsize=(15,5))
df=pd.DataFrame(list(predictions.items()),columns=['Algorithms','Percentage'])
display(df)
sns.pointplot(x="Algorithms", y="Percentage", data=df,ax=ax1);
plt.show()
#generation submission file 
# submission['Survived']=predicted_values_LR
predicted_values = predicted_values_LR.astype(int)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predicted_values
    })
submission.to_csv('titanic_submission.csv',index=False)
display(submission.tail())
display(gen_sub.tail())
print("The shape of the sumbission file is {}".format(submission.shape))
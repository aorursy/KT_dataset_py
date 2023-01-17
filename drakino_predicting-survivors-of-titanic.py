import pandas as pd 

trainFilePath = '../input/train.csv'
testFilePath  = '../input/test.csv'
trainDf = pd.read_csv(trainFilePath)
from IPython.display import display, Markdown # For displaying texts more elegantly
pd.options.display.float_format = '{:,.1f}'.format #For showing a single decimal after point
display(trainDf.head(10))
def check_duplicates(df, colName):
    temp = df.duplicated(colName).sum()
    if temp == 0:
        return "No duplicate values found in the {} column of the dataframe. ".format(colName)
    else:
        return "There are {} duplicates in the {} column of the dataframe. ".format(temp, colName)
# Check if there are any duplicates in the PassengerId column of the train DataFrame
dummyText = check_duplicates(trainDf, 'PassengerId')
trainDf = trainDf.set_index('PassengerId')
dummyText += check_duplicates(trainDf, 'Name')
dummyText += check_duplicates(trainDf, 'Ticket')
display(Markdown(dummyText))
trainDf[trainDf.duplicated("Ticket")].head(5)
display(trainDf[(trainDf["Ticket"]=="349909") | (trainDf["Ticket"]=="CA 2144")])
def check_null(df, colName):
    temp = df[colName].isnull().values.sum()
    if temp == 0:
        return "No null element found in the {} column of the dataframe. ".format(colName)
    else:
        return "There are {} null elements in the {} column of the dataframe. ".format(temp, colName)

dummyText = check_null(trainDf, 'Survived')
dummyText += check_null(trainDf, 'Pclass')
dummyText += check_null(trainDf, 'Sex')
dummyText += check_null(trainDf, 'Embarked')
dummyText += check_null(trainDf, 'SibSp')
dummyText += check_null(trainDf, 'Parch')
dummyText += check_null(trainDf, 'Age')
dummyText += check_null(trainDf, 'Fare')
dummyText += check_null(trainDf, 'Cabin')
display(Markdown(dummyText))
display(trainDf[trainDf['Embarked'].isnull()==True])
dummyText = ""
if trainDf[trainDf['Cabin']=='B28'].index.tolist() == trainDf[trainDf['Embarked'].isnull()==True].index.tolist():
    dummyText += 'No person with cabin number B28 other than PassengerId 62 and 830. '
if trainDf[trainDf['Ticket']=='113572'].index.tolist() == trainDf[trainDf['Embarked'].isnull()==True].index.tolist():
    dummyText += 'No person with ticket number 113572 other than PassengerId 62 and 830'
display(Markdown(dummyText))
trainDf[(trainDf["Age"].isnull()) & ((trainDf["SibSp"]>0) | (trainDf["Parch"]>0))]["Name"].count()
trainDf = trainDf.drop(columns = ["Cabin"])
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True, palette='deep')

def subplot_pie(df, colName, colLabel, _title, _ax):
    temp = df.groupby(colName).count()
    temp = temp.rename(columns={'Name': colLabel})
    temp.plot(kind='pie', y=colLabel, ax=_ax, autopct='%1.0f%%', 
              startangle=90, shadow=True, #explode=[0.01, 0.01, 0.01],
             fontsize=11, legend=False, title=_title)

plt.figure(figsize=(18,12))
ax1 = plt.subplot(231, aspect='equal')
subplot_pie(trainDf, 'Pclass', 'Class', 'Class Distributions', ax1)
ax2 = plt.subplot(232, aspect='equal')
subplot_pie(trainDf, 'Survived', 'Survived', 'Survival Status', ax2)
ax3 = plt.subplot(233, aspect='equal')
subplot_pie(trainDf, 'Sex', 'Sex', 'Sex Distributions', ax3)
ax4 = plt.subplot(234, aspect='equal')
subplot_pie(trainDf, 'Embarked', 'Embarked', 'Port of Embarkation distributions', ax4)
ax4 = plt.subplot(235, aspect='equal')
subplot_pie(trainDf, 'SibSp', 'SibSp', 'Siblings', ax4)
ax4 = plt.subplot(236, aspect='equal')
subplot_pie(trainDf, 'Parch', 'Parch', 'Parents', ax4)
def subplot_hist(df, colName, _title, _ax):
    bins_ = 20
    df[colName].plot(kind="hist", alpha=0.8, bins= bins_, title=_title, ax=_ax)

plt.figure(figsize=(10,5))
ax1 = plt.subplot(121)
subplot_hist(trainDf, 'Age', 'Age Distributions', ax1)
ax2 = plt.subplot(122)
subplot_hist(trainDf, 'Fare', 'Fare Distributions', ax2)
display(trainDf[["Age", "Fare"]].describe())
def create_twowaytable_counts(df, explanatory, response):
    xIndices = trainDf.groupby(explanatory).count().index.values
    yIndices = trainDf.groupby(response).count().index.values
    resultDf = pd.DataFrame(index=xIndices)
    for y in yIndices:
        tempDf = df[df[response]==y].groupby(explanatory).count()
        tempDf = tempDf.rename(columns={"Name": y})
        resultDf = pd.concat([resultDf, tempDf[y]], axis=1)
    resultDf['total'] = resultDf.sum(axis=1)
    resultDf.loc['total'] = resultDf.sum(axis=0)
    return resultDf
def create_twowaytable_percentages(df, explanatory, response):
    xIndices = trainDf.groupby(explanatory).count().index.values
    yIndices = trainDf.groupby(response).count().index.values
    resultDf = pd.DataFrame(index=xIndices)
    for y in yIndices:
        tempDf = df[df[response]==y].groupby(explanatory).count()
        tempDf = tempDf.rename(columns={"Name": y})
        resultDf = pd.concat([resultDf, tempDf[y]], axis=1)
    resultDf['total'] = resultDf.sum(axis=1)
    resultDf = resultDf.div(resultDf.max(axis=1), axis=0)*100
    return resultDf
def create_doubleBarChart(df, explanatory, response):
    xIndices = trainDf.groupby(explanatory).count().index.values
    yIndices = trainDf.groupby(response).count().index.values
    resultDf = pd.DataFrame(index=yIndices)
    for x in xIndices:
        tempDf = df[df[explanatory]==x].groupby(response).count()
        tempDf = tempDf.rename(columns={"Name": x})
        resultDf = pd.concat([resultDf, tempDf[x]], axis=1)
    resultDf.loc['total'] = resultDf.sum(axis=0)
    resultDf = resultDf.div(resultDf.max(axis=0), axis=1)*100
    resultDf = resultDf.drop("total")
    resultDf.plot(kind="bar")
display(create_twowaytable_counts(trainDf, 'Sex', 'Survived'))
display(create_twowaytable_percentages(trainDf, 'Sex', 'Survived'))
create_doubleBarChart(trainDf, 'Sex', 'Survived')
display(create_twowaytable_counts(trainDf, 'Pclass', 'Survived'))
display(create_twowaytable_percentages(trainDf, 'Pclass', 'Survived'))
create_doubleBarChart(trainDf, 'Pclass', 'Survived')
display(create_twowaytable_counts(trainDf, 'Embarked', 'Survived'))
display(create_twowaytable_percentages(trainDf, 'Embarked', 'Survived'))
create_doubleBarChart(trainDf, 'Embarked', 'Survived')
display(create_twowaytable_counts(trainDf, 'SibSp', 'Survived'))
display(create_twowaytable_percentages(trainDf, 'SibSp', 'Survived'))
create_doubleBarChart(trainDf, 'SibSp', 'Survived')
display(create_twowaytable_counts(trainDf, 'Parch', 'Survived'))
display(create_twowaytable_percentages(trainDf, 'Parch', 'Survived'))
create_doubleBarChart(trainDf, 'Parch', 'Survived')
import numpy as np
from sklearn import linear_model

tempDf = trainDf[['Age', 'Survived']].dropna(axis=0, how='any')
clf = linear_model.LogisticRegression()
clf.fit(tempDf['Age'].values.reshape(-1, 1), tempDf['Survived'])
x = np.linspace(tempDf['Age'].min(), tempDf['Age'].max(), 100)
y = 1 / (1 + np.exp(-(x*clf.coef_[0][0] + clf.intercept_[0])))
plt.scatter(tempDf['Age'], tempDf['Survived'])
plt.plot(x, y, color="red")
#display(clf.score(tempDf['Age'].values.reshape(-1, 1), tempDf['Survived']))
ax = plt.gca()
ax.set_ylabel("Survived",fontsize=12)
ax.set_xlabel("Age",fontsize=12)
ax.legend(['train data', 'logistic regression'])
tempDf = trainDf[['Fare', 'Survived']].dropna(axis=0, how='any')
clf = linear_model.LogisticRegression()
clf.fit(tempDf['Fare'].values.reshape(-1, 1), tempDf['Survived'])
x = np.linspace(tempDf['Fare'].min(), tempDf['Fare'].max(), 100)
y = 1 / (1 + np.exp(-(x*clf.coef_[0][0] + clf.intercept_[0])))
plt.scatter(tempDf['Fare'], tempDf['Survived'])
plt.plot(x, y, color="red")
#display(clf.score(tempDf['Fare'].values.reshape(-1, 1), tempDf['Survived']))
ax = plt.gca()
ax.set_ylabel("Survived",fontsize=12)
ax.set_xlabel("Fare",fontsize=12)
ax.legend(['train data', 'logistic regression'], loc="center right")
import scipy.stats as stats

def calc_chi2prob(dataFrame):
    counts = dataFrame.as_matrix()
    [nr, nc] = counts.shape
    nr -= 1
    nc -= 1
    expected = np.zeros((nr, nc))
    for i in range(nr):
        for j in range(nc):
            expected[i,j] = counts[i,nc]*counts[nr,j]/counts[nr,nc]
    chi2stat = (((counts[:nr,:nc]-expected)**2/expected)).sum()
    df_ = (nr-1)*(nc-1)
    return (1 - stats.chi2.cdf(x=chi2stat, df=df_))<0.05
    
observedCounts = create_twowaytable_counts(trainDf, 'Sex', 'Survived')
dummyText = "Null hypothesis: {} and {} variables are independent. ".format("Survived", "Sex")
if calc_chi2prob(observedCounts):
    dummyText += "We can reject the null hypothesis since {} depend on {} ".format("Survived", "Sex")
else:
    dummyText += "We cannot reject the null hypothesis"
display(Markdown(dummyText))
observedCounts = create_twowaytable_counts(trainDf, 'Pclass', 'Survived')
dummyText =  "Null hypothesis: {} and {} variables are independent. ".format("Survived", "Pclass")
if calc_chi2prob(observedCounts):
    dummyText += "We can reject the null hypothesis since {} depend on {} ".format("Survived", "Pclass")
else:
    dummyText += "We cannot reject the null hypothesis"
display(Markdown(dummyText))
observedCounts = create_twowaytable_counts(trainDf, 'Embarked', 'Survived')
dummyText = "Null hypothesis: {} and {} variables are independent. ".format("Survived", "Embarked")
if calc_chi2prob(observedCounts):
    dummyText += "We can reject the null hypothesis since {} depend on {} ".format("Survived", "Embarked")
else:
    dummyText += "We cannot reject the null hypothesis"
display(Markdown(dummyText))
observedCounts = create_twowaytable_counts(trainDf, 'Parch', 'Survived')
dummyText = "Null hypothesis: {} and {} variables are independent. ".format("Survived", "Parch")
if calc_chi2prob(observedCounts):
    dummyText += "We can reject the null hypothesis since {} depend on {} ".format("Survived", "Parch")
else:
    dummyText += "We cannot reject the null hypothesis"
display(Markdown(dummyText))
observedCounts = create_twowaytable_counts(trainDf, 'SibSp', 'Survived')
dummyText = "Null hypothesis: {} and {} variables are independent. ".format("Survived", "SibSp")
if calc_chi2prob(observedCounts):
    dummyText += "We can reject the null hypothesis since {} depend on {} ".format("Survived", "SibSp")
else:
    dummyText += "We cannot reject the null hypothesis"
display(Markdown(dummyText))
import scipy.stats as stats
def calc_waldForLogit(dataFrame, ivName, dvName):
    tempDf = dataFrame[[ivName,dvName]].dropna(axis=0, how='any')
    xTrain = tempDf[ivName].values.reshape(-1, 1)
    y = tempDf[dvName]
    clf = linear_model.LogisticRegression()
    clf.fit(xTrain, y)
    predProbs = np.matrix(clf.predict_proba(xTrain))
    xDesign = np.hstack((np.ones(shape = (xTrain.shape[0],1)), xTrain))
    V = np.matrix(np.zeros(shape = (xDesign.shape[0], xDesign.shape[0])))
    np.fill_diagonal(V, np.multiply(predProbs[:,0], predProbs[:,1]).A1)
    covLogit = np.linalg.inv(xDesign.T*V*xDesign)
    stdErrors = np.sqrt(np.diag(covLogit))
    zScoreCoef = clf.coef_[0][0]/stdErrors[0]
    zScoreInt  = clf.intercept_[0]/stdErrors[1]
    return (stats.norm.sf(abs(zScoreCoef))*2.)<0.05 and (stats.norm.sf(abs(zScoreInt))*2.)<0.05 
    
dummyText = "Null hypothesis: The probability of a particular value of {} is not associated with the value of {}. ".format("Survived", "Age")
if calc_waldForLogit(trainDf, "Age", "Survived"):
    dummyText += "We can reject the null hypothesis since {} depend on {} ".format("Survived", "Age")
else:
    dummyText += "We cannot reject the null hypothesis"
display(Markdown(dummyText))
dummyText = "Null hypothesis: The probability of a particular value of {} is not associated with the value of {}. ".format("Survived", "Fare")
if calc_waldForLogit(trainDf, "Fare", "Survived"):
    dummyText += "We can reject the null hypothesis since {} depend on {} ".format("Survived", "Fare")
else:
    dummyText += "We cannot reject the null hypothesis"
display(Markdown(dummyText))
trainDf.loc[trainDf['Sex'] == "male", "Sex"] = 0
trainDf.loc[trainDf['Sex'] == "female", "Sex"] = 1
trainDf.loc[trainDf['Embarked'] == "S", "Embarked"] = 0
trainDf.loc[trainDf['Embarked'] == "C", "Embarked"] = 1
trainDf.loc[trainDf['Embarked'] == "Q", "Embarked"] = 2
features = trainDf[['Sex', 'Pclass', 'Embarked']]
output   = trainDf.Survived
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
featuresValues = features.values
features.loc[:, :] = imputer.fit_transform(featuresValues)
from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(features, output, random_state = 0)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
modelRF = RandomForestClassifier()
modelRF.fit(XTrain, yTrain)
yPredicted = modelRF.predict(XTest)
display(accuracy_score(yTest, yPredicted))
from sklearn import svm
modelSVM = svm.SVC()
modelSVM.fit(XTrain, yTrain)
yPredicted = modelSVM.predict(XTest)
display(accuracy_score(yTest, yPredicted))
from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier(n_neighbors=4)
modelKNN.fit(XTrain, yTrain)
yPredicted = modelKNN.predict(XTest)
display(accuracy_score(yTest, yPredicted))
testDf = pd.read_csv(testFilePath)
modelRF = RandomForestClassifier()
modelRF.fit(features, output)
xTest = testDf[["Sex", "Pclass", "Embarked"]]
xTest.loc[xTest['Sex'] == "male", "Sex"] = 0
xTest.loc[xTest['Sex'] == "female", "Sex"] = 1
xTest.loc[xTest['Embarked'] == "S", "Embarked"] = 0
xTest.loc[xTest['Embarked'] == "C", "Embarked"] = 1
xTest.loc[xTest['Embarked'] == "Q", "Embarked"] = 2
predictions = modelRF.predict(xTest)
passengerId = testDf["PassengerId"]
subDf = pd.DataFrame({ 'PassengerId': passengerId,
                            'Survived': predictions })
subDf.to_csv("subDf.csv", index=False)
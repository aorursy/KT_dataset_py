import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
traindata = pd.read_csv('../input/train.csv')

testdata = pd.read_csv('../input/test.csv')

mergeData = [traindata, testdata]
import re



def title(Name):

    getTitle = re.search('([A-Za-z]+)\.',Name)

    

    if getTitle:

        return getTitle.group(1)

    return ""



for data in mergeData:

    data['Titles'] = data['Name'].apply(title)



crosstabTrain = pd.crosstab(traindata['Titles'], traindata['Sex'])

crosstabTest = pd.crosstab(testdata['Titles'], testdata['Sex'])
### check if there two sex in one title

crosstabTrain[(crosstabTrain.female != 0) & (crosstabTrain.male !=0)]
# Dr are female & male

crosstabTest[(crosstabTest.female != 0) & (crosstabTest.male !=0)]

# Just in Train data
crosstabTest.loc[crosstabTest.male != 0, 'male'].sort_values(ascending=False)
#titles in data train have represent titles in data test

male = crosstabTrain.loc[crosstabTrain.male != 0, 'male'].sort_values(ascending=False)

male
# Age distibution for titles: Mr. and Master

ManOld = traindata[['Titles','Age']][traindata.Titles.isin(['Mr','Master'])]



f,axes = plt.subplots(1,2, figsize=(12,6))

visManOld = sns.violinplot(data=ManOld, x='Titles', y='Age',ax=axes[0], palette="husl")





plt.rcParams['figure.figsize'] = 10,8

visMO1 = sns.distplot(ManOld[ManOld.Titles=='Mr'].Age, hist=False, kde_kws={"shade": True}, color="r", label ='Mr')

visMO1 = sns.distplot(ManOld[ManOld.Titles=='Master'].Age, hist=False, kde_kws={"shade": True}, color="g", label='Master')

visMO1.legend()



plt.show()
mr = ['Rev','Dr','Major', 'Col', 'Sir', 'Jonkheer', 'Don', 'Capt']

traindata[(traindata.Titles.isin(mr)) & (traindata.Age<=20)][['Titles','Age']]

#all of them greater than 20 y.o

# testdata[(testdata.Titles.isin(maleL)) & (testdata.Age<=20) ][['Titles','Age']]
for data in mergeData:

    data.loc[data.Titles.isin(mr) & (data.Sex == 'male'),'Titles'] = 'Mr'
#female Dr

traindata[traindata.Titles == 'Dr']
crosstabTest.loc[crosstabTest.female != 0, 'female'].sort_values(ascending=False)
crosstabTrain.loc[crosstabTrain.female != 0, 'female'].sort_values(ascending=False)
woman = traindata[['Titles','Age']][traindata.Titles.isin(['Miss','Mrs'])]



f,axes = plt.subplots(1,2, figsize=(12,6))

viswoman = sns.violinplot(data=woman, x='Titles', y='Age',ax=axes[0], palette="husl")





plt.rcParams['figure.figsize'] = 10,8

visMO2 = sns.distplot(woman[woman.Titles=='Miss'].Age, hist=False, kde_kws={"shade": True}, color="r", label ='Miss')

visMO2 = sns.distplot(woman[woman.Titles=='Mrs'].Age, hist=False, kde_kws={"shade": True}, color="g", label='Mrs')

visMO2.legend()



plt.show()
mrs = ['Mlle','Ms','Mme','Lady','Dr','Countess', 'Dona']

traindata[traindata.Titles.isin(mrs) & (traindata.Age <=20)][['Titles','Age']]

#All of them greater than 20 y.o

#testdata[testdata.Titles.isin(mrs) & (testdata.Age <=20)][['Titles','Age']]
for data in mergeData:

    data.loc[(data.Titles.isin(mrs))&(data.Sex == 'female'),'Titles'] = 'Mrs'
titles = ['Mr','Master','Mrs','Miss']

for title in titles:

    for data in mergeData:

        dt = data[data.Titles==title]



        ageAvg = dt.Age.mean()

        ageStd = dt.Age.std()

        ageMin = ageAvg-ageStd

        ageMax = ageAvg+ageStd

        numNull = data.Age.isnull().sum()



        ageRand = np.random.randint(ageMin,ageMax,size=numNull)

        data['Age'][np.isnan(data['Age'])] = ageRand

        #data['Age'] = data['Age'].astype('int')
for data in mergeData:

    data.loc[(data.SibSp == 0) & (data.Parch == 0), 'Alone'] = 1

    data.loc[data.Alone.isnull(), 'Alone'] = 0 

    data.Alone = data.Alone.astype('int')
sns.heatmap(traindata.corr(),annot=True)

plt.show()
Pclass = list(traindata.Pclass.unique())

for pc in Pclass:

    for data in mergeData:

        df = data[data.Pclass == pc]

        

        fareAvg = df.Fare.mean()

        fareStd = df.Fare.std()

        fareMin = fareAvg-fareStd

        fareMax = fareAvg+fareStd

        numNull = data.Fare.isnull().sum()

        

        fareRand = np.random.randint(fareMin,fareMax,size=numNull)

        data['Fare'][np.isnan(data['Fare'])] = fareRand
for data in mergeData:

    data.loc[data.Embarked.isnull(),'Embarked'] = 'S'
dropColumns = ['PassengerId','Name','SibSp','Parch','Ticket','Cabin']

for data in mergeData:

    data.drop(dropColumns, axis=1,inplace=True)
# Age Interval

splitAge = traindata.copy()

splitAge['Age'] = pd.cut(traindata['Age'], 5)

splitAge.groupby('Age').count()
# Fare Interval

splitFare = traindata.copy()

splitFare['Fare'] = pd.qcut(traindata['Fare'], 6)

splitFare.groupby('Fare').count()
for data in mergeData:

    #AGE

    data.loc[data.Age <= 16.3, 'Age'] = 0

    data.loc[(data.Age >16.3) & (data.Age <= 32.3), 'Age'] = 1

    data.loc[(data.Age >32.3) & (data.Age <= 48.2), 'Age'] = 2

    data.loc[(data.Age >48.2) & (data.Age <= 64.1), 'Age'] = 3

    data.loc[data.Age > 64.1, 'Age'] = 4   

    

    #FARE

    data.loc[data.Fare <= 7.8, 'Fare'] = 0

    data.loc[(data.Fare > 7.8)&(data.Fare <= 8.7), 'Fare'] = 1

    data.loc[(data.Fare > 8.7)&(data.Fare <= 14.5), 'Fare'] = 2

    data.loc[(data.Fare > 14.5)&(data.Fare <= 26.0), 'Fare'] = 3

    data.loc[(data.Fare > 26.0)&(data.Fare <= 52.4), 'Fare'] = 4

    data.loc[(data.Fare > 52.4), 'Fare'] = 5

    

    data['Sex'] = data['Sex'].map({'female':0,'male':1}).astype('int')

    data['Titles'] = data['Titles'].map({'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3})

    data['Embarked'] = data['Embarked'].map({'Q':0, 'C':1, 'S':2})

    

    data['Age'] = data['Age'].astype(int)

    data['Fare'] = data['Fare'].astype(int)

    data['Embarked'] = data['Embarked'].astype(int)
trainData = traindata[['Embarked','Titles','Pclass','Sex','Age','Fare','Alone','Survived']]

testData= testdata[['Embarked','Titles','Pclass','Sex','Age','Fare','Alone']]
X = trainData.iloc[:,:-1].values

y = trainData.iloc[:,7:8].values
X.shape
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



# EMBARKED

onehotencoderE = OneHotEncoder(categorical_features = [0])

X = onehotencoderE.fit_transform(X).toarray()

#avoiding dummy var trap

X = X[:,1:]



#TITLES

onehotencoderT = OneHotEncoder(categorical_features = [2])

X = onehotencoderT.fit_transform(X).toarray()

#avoiding dummy var trap

X = X[:,1:]
import statsmodels.formula.api as sm

# ADD BIAS FIRST

X = np.append(arr = np.ones((891,1)).astype(int), values=X, axis=1)
X_opt = X[:,[0,1,2,3,5,6,7,8,10]] #remove index that have Pvalues > Significant Level

cekOLS = sm.OLS(endog = y, exog=X_opt).fit()

cekOLS.summary()



#X -- bias=0 #Embarked=1 #Titles=2 #Pclass=3 #Sex=4 #Age=5 ##Fare=6 ##Alone=7

#constant has index 0
X = X[:,[0,1,2,3,5,6,7,8,10]]

y = y
from sklearn.model_selection import train_test_split

Xtrain,Xval,ytrain,yval = train_test_split(X,y,stratify=y, test_size=0.3)
def sigmoid(z):

    return(1 / (1 + np.exp(-z)))



def sigmoidGradient(z):

    return(sigmoid(z)*(1-sigmoid(z)))
def getOptTheta(nn_params, input_layer_size, hidden_layer_size, num_labels, features, classes, reg,alpha,num_iter):

    

    theta1 = nn_params[0:(hidden_layer_size*(input_layer_size+1))].reshape(hidden_layer_size,(input_layer_size+1))

    theta2 = nn_params[(hidden_layer_size*(input_layer_size+1)):].reshape(num_labels,(hidden_layer_size+1))



    m = features.shape[0]

    if num_labels == 1:

        y_matrix = classes

    else:

        y_matrix = pd.get_dummies(classes.ravel()).as_matrix() 

        

    for i in np.arange(num_iter):

        

        # Cost

        a1 = features 



        z2 = theta1.dot(a1.T) 

        a2 = np.c_[np.ones((features.shape[0],1)),sigmoid(z2.T)] 



        z3 = theta2.dot(a2.T) 

        a3 = sigmoid(z3)



        # Gradients

        d3 = a3.T - y_matrix 

        d2 = theta2[:,1:].T.dot(d3.T)*sigmoidGradient(z2) 



        delta1 = d2.dot(a1) 

        delta2 = d3.T.dot(a2)



        theta1_ = np.c_[np.ones((theta1.shape[0],1)),theta1[:,1:]]

        theta2_ = np.c_[np.ones((theta2.shape[0],1)),theta2[:,1:]]



        theta1_grad = delta1/m + (theta1_*reg)/m

        theta2_grad = delta2/m + (theta2_*reg)/m



        theta1 = theta1 - (alpha*theta1_grad)

        theta2 = theta2 - (alpha*theta2_grad)

        

    return(theta1,theta2)
def prediction(X,thetaOpt1,thetaOpt2):

    [m,n] = X.shape

    

    a1 = X 

    a2 = sigmoid(X.dot(thetaOpt1.T) ) 

    hypo = sigmoid(np.c_[np.ones((a2.shape[0],1)),a2].dot(thetaOpt2.T)) 

    

    if thetaOpt2.shape[0] == 1:

        for i in range(hypo.shape[0]):

            if hypo[i] >= 0.5:

                hypo[i] = 1

            else:

                hypo[i] = 0

        opt = hypo

        

    else:

        opt = (np.argmax(hypo, axis=1))

        opt = opt.reshape(-1,1)

    

    return(opt)  
list_hidden_layer_size = [6,7,8,9,10,12,16,21,31]



reg=0.01

alpha= 0.1

num_iter=10000



input_layer_size = 8

num_labels = 1

#--------------------------------



AThl = []

AVhl = []



for hidden_layer_size in list_hidden_layer_size:

    

    #create initial random params

    eps = 0.12

    initialTheta1 = np.random.rand(hidden_layer_size,input_layer_size+1)*2*eps-eps

    initialTheta2 = np.random.rand(num_labels,hidden_layer_size+1)*2*eps-eps

    initial_nn_params = np.r_[initialTheta1.ravel(), initialTheta2.ravel()]

    

    [tOpt1,tOpt2] = getOptTheta(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, Xtrain, ytrain, reg,alpha,num_iter)

        

     

    predTrain = prediction(Xtrain,tOpt1,tOpt2)

    accTrain = np.mean(predTrain==ytrain)*100

    

    predVal = prediction(Xval,tOpt1,tOpt2)

    accVal = np.mean(predVal==yval)*100

    

    AThl.append(accTrain)

    AVhl.append(accVal)
plt.plot(list_hidden_layer_size,AThl, label = 'Accuracy Train' )

plt.plot(list_hidden_layer_size,AVhl, label='Accuracy Val')

plt.title('accurate for each hidden layer size')

plt.xlabel('number of hidden layer size')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
gethl = pd.DataFrame({'hidden_layer_size':list_hidden_layer_size,'Train Accuracy':AThl,'Val Accuracy':AVhl})

gethl.sort_values('Val Accuracy', ascending=False)
list_reg=[0, 0.01, 0.1, 1, 10, 100]



alpha= 0.1

num_iter=10000



hidden_layer_size = 7

input_layer_size = 8

num_labels = 1

#--------------------------------



ATlr = []

AVlr = []



for reg in list_reg:

    

    #CREATE INITIAL RANDOM PARAMS

    eps = 0.12

    initialTheta1 = np.random.rand(hidden_layer_size,input_layer_size+1)*2*eps-eps

    initialTheta2 = np.random.rand(num_labels,hidden_layer_size+1)*2*eps-eps

    initial_nn_params = np.r_[initialTheta1.ravel(), initialTheta2.ravel()]

    

    [tOpt1,tOpt2] = getOptTheta(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, Xtrain, ytrain, reg,alpha,num_iter)

        

     

    predTrain = prediction(Xtrain,tOpt1,tOpt2)

    accTrain = np.mean(predTrain==ytrain)*100

    

    predVal = prediction(Xval,tOpt1,tOpt2)

    accVal = np.mean(predVal==yval)*100

    

    ATlr.append(accTrain)

    AVlr.append(accVal)
plt.plot(list_reg,ATlr, label = 'Accuracy Train' )

plt.plot(list_reg,AVlr, label='Accuracy Val')

plt.title('Model selection : lambda')

plt.xlabel('lambda')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
getlr = pd.DataFrame({'Lambda - reg':list_reg,'Train Accuracy':ATlr,'Val Accuracy':AVlr})

getlr.sort_values('Val Accuracy', ascending=False)
numIters=[1000,2000,3000,4000,5000,10000,15000,20000,25000,30000]



reg=0.01

alpha= 0.1



hidden_layer_size = 7

input_layer_size = 8

num_labels = 1

#--------------------------------



ATni = []

AVni = []



for num_iter in numIters:

    

    #CREATE INITIAL RANDOM PARAMS

    eps = 0.12

    initialTheta1 = np.random.rand(hidden_layer_size,input_layer_size+1)*2*eps-eps

    initialTheta2 = np.random.rand(num_labels,hidden_layer_size+1)*2*eps-eps

    initial_nn_params = np.r_[initialTheta1.ravel(), initialTheta2.ravel()]

    

    [tOpt1,tOpt2] = getOptTheta(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, Xtrain, ytrain, reg,alpha,num_iter)

        

     

    predTrain = prediction(Xtrain,tOpt1,tOpt2)

    accTrain = np.mean(predTrain==ytrain)*100

    

    predVal = prediction(Xval,tOpt1,tOpt2)

    accVal = np.mean(predVal==yval)*100

    

    ATni.append(accTrain)

    AVni.append(accVal)
plt.plot(numIters,ATni, label = 'Accuracy Train' )

plt.plot(numIters,AVni, label='Accuracy Val')

plt.title('Model selection : Num Iter for GradientDescent')

plt.xlabel('Num iter')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
getni = pd.DataFrame({'Num Iters':numIters,'Train Accuracy':ATni,'Val Accuracy':AVni})

getni.sort_values('Val Accuracy', ascending=False)
alphaL= [0.01,0.03,0.1,0.3,1,3]



reg=0.01

num_iter=20000



hidden_layer_size = 7

input_layer_size = 8

num_labels = 1

#--------------------------------



ATa = []

AVa = []



for alpha in alphaL:

    

    #CREATE INITIAL RANDOM PARAMS

    eps = 0.12

    initialTheta1 = np.random.rand(hidden_layer_size,input_layer_size+1)*2*eps-eps

    initialTheta2 = np.random.rand(num_labels,hidden_layer_size+1)*2*eps-eps

    initial_nn_params = np.r_[initialTheta1.ravel(), initialTheta2.ravel()]

    

    [tOpt1,tOpt2] = getOptTheta(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, Xtrain, ytrain, reg,alpha,num_iter)

        

     

    predTrain = prediction(Xtrain,tOpt1,tOpt2)

    accTrain = np.mean(predTrain==ytrain)*100

    

    predVal = prediction(Xval,tOpt1,tOpt2)

    accVal = np.mean(predVal==yval)*100

    

    ATa.append(accTrain)

    AVa.append(accVal)

    
plt.plot(alphaL,ATa, label = 'Accuracy Train' )

plt.plot(alphaL,AVa, label='Accuracy Val')

plt.title('Model selection : Num Iter for GradientDescent')

plt.xlabel('Num iter')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
geta = pd.DataFrame({'Alpha - learning rate':alphaL,'Train Accuracy':ATa,'Val Accuracy':AVa})

geta.sort_values('Val Accuracy', ascending=False)
accuracyTrainLr = []

accuracyValLr = []

tOptl1 = []

tOptl2 = []



trying = np.arange(1,11)

for iters in trying:



    reg=0.01

    alpha= 0.1

    num_iter=20000



    hidden_layer_size = 7

    input_layer_size = 8

    num_labels = 1

    #--------------------------------



    #CREATE INITIAL RANDOM PARAMS

    eps = 0.12

    initialTheta1 = np.random.rand(hidden_layer_size,input_layer_size+1)*2*eps-eps

    initialTheta2 = np.random.rand(num_labels,hidden_layer_size+1)*2*eps-eps

    initial_nn_params = np.r_[initialTheta1.ravel(), initialTheta2.ravel()]



    [tOpt1,tOpt2] = getOptTheta(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, Xtrain, ytrain, reg,alpha,num_iter)





    predTrain = prediction(Xtrain,tOpt1,tOpt2)

    accTrain = np.mean(predTrain==ytrain)*100



    predVal = prediction(Xval,tOpt1,tOpt2)

    accVal = np.mean(predVal==yval)*100



    accuracyTrainLr.append(accTrain)

    accuracyValLr.append(accVal)

    tOptl1.append(tOpt1)

    tOptl2.append(tOpt2)
plt.plot(trying,accuracyTrainLr, label = 'Train Accuracy' )

plt.plot(trying,accuracyValLr, label='Val Accuracy')

plt.title('run n times for different param and accuracy')

plt.xlabel('running n times')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
tri = accuracyTrainLr.index(np.max(accuracyTrainLr))

trv = np.max(accuracyTrainLr)

tvi = accuracyValLr.index(np.max(accuracyValLr))

tvv = np.max(accuracyValLr)

print('the highest accuracy in data train in index:',tri,'with accuracy =',trv)

print('the highest accuracy in data CROSSVAL in index:',tvi,'with accuracy =',tvv)
imax = accuracyValLr.index(np.max(accuracyValLr))
Theta1Optimum = tOptl1[imax]

Theta2Optimum = tOptl2[imax]
predTrain = prediction(Xtrain,Theta1Optimum,Theta2Optimum)

accTrain = np.mean(predTrain==ytrain)*100

accTrain
predVal = prediction(Xval,Theta1Optimum,Theta2Optimum)

accVal = np.mean(predVal==yval)*100

accVal
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    sns.set_style('white')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    else:

        1#print('Confusion matrix, without normalization')



    #print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
cm = confusion_matrix(yval,predVal)



#bukan make index untuk akses confusion matrix. 

print("Recall =", round(cm[1,1]/(cm[1,0]+cm[1,1]),2))

print("Precision =",round(cm[1,1]/(cm[0,1]+cm[1,1]),2))

print("Accuracy=",(cm[1,1]+cm[0,0])/(cm[1,1]+cm[0,0]+cm[1,0]+cm[0,1]) )

# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cm

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
print(classification_report(yval,predVal))
fpr, tpr, thresholds = roc_curve(yval,predVal)

roc_auc = auc(fpr,tpr)



print('ROC AUC=',round((roc_auc*100),2),'%')

# Plot ROC

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
testData.iloc[:,:-2].head(5)
testData.head(3)
Xtest = testData.iloc[:,:].values

Xtest.shape
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



#EMBARKED

onehotencoderXtestE = OneHotEncoder(categorical_features = [0])

Xtest = onehotencoderXtestE.fit_transform(Xtest).toarray()

#avoiding dummy variable trap

Xtest= Xtest[:,1:]



#TITLES

onehotencoderXtestT = OneHotEncoder(categorical_features = [2])

Xtest = onehotencoderXtestT.fit_transform(Xtest).toarray()

#avoiding dummy variable trap

Xtest= Xtest[:,1:]

Xtest = np.c_[np.ones((Xtest.shape[0])),Xtest]

Xtest.shape
Xtest = Xtest[:,[0,1,2,3,5,6,7,8,10]]
Xtest.shape
predTest = prediction(Xtest,Theta1Optimum,Theta2Optimum)
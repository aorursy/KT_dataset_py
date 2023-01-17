import pandas as pd ##pandas dataframes are often used for statistical analysis,

import numpy as np ##calculate mean and standard deviation

import seaborn as sb ##includes convenient heatmaps and boxplots

import sklearn.linear_model as sklm ##Includes Logistic Regression, which will be tested for predictive capability

import sklearn.decomposition as skdc ##Includes Principal Component Analysis, a method of dimensionality reduction

import sklearn.pipeline as skpl ##Convenient module for calculating PCs and using them in logistic regression

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/data.csv') ##store csv in Pandas DataFrame
data.head() ##print first five rows of dataframe as a table
data = data.drop('id', 1) ##id column provides no information on data

data = data.drop('Unnamed: 32', 1) ##column of missing values
print(pd.get_dummies(data['diagnosis']).head(1))#dummy variable to represent categorical data as numeric

data['diagnosis_dummies'] = pd.get_dummies(data['diagnosis']).iloc[:,1]#extract Malignant dummy category to dataframe

###only inclulde one dummy category to avoid multicollinearity, either one could be chosen
datacorr = data.corr() #correlation matrix, showing correlation between each variable and all the others

data.corr().head()
sb.heatmap(datacorr, cmap = 'bwr') #heatmap of correlation matrix

###darker colors represent higher correlation, several pairs of variables are highly correlated.
sb.boxplot(x=data['concave points_mean'], y=data['diagnosis'], data=data, linewidth=2.5) #boxplot

###boxplot shows the distribution by classification of concave points_mean, chosen because 

###it had the highest correlation with the diagnosis dummy variable. There is a clear difference in distributions.
sb.boxplot(x=data['texture_se'], y=data['diagnosis'], data=data, linewidth=2.5)

###For comparison, this is the same boxplot construct with a variable that is minimally correlated with diagnosis
def standardization(x): #Define function to standardize the data, since all variables are not in the same units

    xmean = np.mean(x) ##calculate mean

    sd = np.std(x) ##calculate standard deviation 

    x_z = (x - xmean) / sd ##calculate standardized value to return

    return(x_z)
data_stnd = data.drop(['diagnosis','diagnosis_dummies'], 1).apply(standardization,broadcast = True) 

##drop response variable and standardize predictor variables

data_stnd.head()
X = data_stnd #store predictor variables

y = data['diagnosis_dummies'] #store response variable

pca = skdc.PCA() #empty model space
pcafit = pca.fit_transform(X,y) ##apply dimensionality reduction to X
var_explained = pca.explained_variance_ratio_ #ratio of variance each PC explains

print(pd.Series(var_explained))

###Since 29 components aren't necessary, the last 20 PCs will be disregarded 

###since they explain less than.01 of the variance

print(sum(var_explained[0:10]))

##indeed,the first 10 PCs explain 95% of the variance
pca = skdc.PCA(n_components = 10) #only include first 10 components

logreg = sklm.LogisticRegression()#empty model space

pipeline = skpl.Pipeline([('pca', pca), ('logistic', logreg)]) #create pipeline from pca to logregression space
predMalignantRight = 0 #create count variables

predMalignantWrong = 0

predBenignRight = 0

predBenignWrong = 0
for i in range(0,569): #run through each row in data set

    trainX = X.drop(i, 0) #train model with predictor dataframe, remove single row

    trainy = y.drop(i,0) #train model with response array, remove single row

    testX = X.iloc[i,:].values.reshape(1,30) #Removed row will be test predictor (Got error message before using values.reshape)

    testy = y[i] #Removed value will be test response

    fit = pipeline.fit(trainX, trainy) #fit model

    prediction = pipeline.predict(testX) #test model with left out value

    if prediction == 1 and testy == 1:

        predMalignantRight += 1

    elif prediction == 1 and testy == 0:

        predMalignantWrong += 1

    elif prediction == 0 and testy == 1:

        predBenignWrong += 1

    else:

        predBenignRight += 1
print(predMalignantRight,predMalignantWrong,predBenignRight,predBenignWrong)
###Time to create a nice confusion matrix to visualize

c = {'Predicted Benign' : pd.Series([predBenignRight, predBenignWrong],index=['Actual Benign', 'Actual Malignant']),

    'Predicted Malignant': pd.Series([predMalignantWrong, predMalignantRight], index=['Actual Benign','Actual Malignant'])}

confusionmat = pd.DataFrame(c)

confusionmat

###nearly 98% of the values lie on the correct diagonal
###Now sensitivity and specificity will be calculated

mr,mw = float(predMalignantRight), float(predMalignantWrong)

bw,br = float(predBenignWrong), float(predBenignRight)

sens = mr/(mr+mw) #calculate sensitivity, or rate of correctly predicting disease

spec = br/(br+bw) #calculate specificity, or rate of correctly predicting no disease

acc = (sens + spec)/2 #calculate balanced accuracy, or average of sensitivty and specificity

mis = (mw+bw)/(mw+bw+mr+br) #calculate misclassification rate
###create series of values, then convert to dataframe to print as table

outputseries = pd.Series([sens,spec,acc,mis],index=['Sensitivity','Specificity','Balanced Accuracy','Misclassification rate'])

outputdf = pd.DataFrame(outputseries)

outputdf.columns = [''] #blank header name

outputdf.head()
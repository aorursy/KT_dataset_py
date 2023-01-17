import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
## load the data 
diabetesDF = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
diabetesDF.head()
diabetesDF.info()
diabetesDF.describe()
bad_g = diabetesDF["Glucose"] == 0
diabetesDF.loc[bad_g, "Glucose"] = None
sum(diabetesDF['Glucose']==0)
bad_g = diabetesDF['BloodPressure'] == 0
diabetesDF.loc[bad_g, 'BloodPressure'] = None
sum(diabetesDF['BloodPressure']==0)
bad_s= diabetesDF['Insulin'] == 0
diabetesDF.loc[bad_s, 'Insulin'] = None
sum(diabetesDF['Insulin']==0)
bad_s= diabetesDF['SkinThickness'] == 0
diabetesDF.loc[bad_s, 'SkinThickness'] = None
sum(diabetesDF['SkinThickness']==0)
bad_s= diabetesDF['BMI'] == 0
diabetesDF.loc[bad_s,'BMI'] = None
sum(diabetesDF['BMI']==0)
ad_s= diabetesDF['DiabetesPedigreeFunction'] == 0
diabetesDF.loc[bad_s,'DiabetesPedigreeFunction'] = None
sum(diabetesDF['DiabetesPedigreeFunction']==0)
diabetesDF.describe()
diabetesDF.shape
diabetesDF.fillna(diabetesDF.mean(), inplace= True)
print(diabetesDF.isnull().sum())
sns.countplot(diabetesDF['Outcome'],label="Count")
diabetesDF.hist(figsize=(15,15))
# split into train and test 
dfTrain =diabetesDF[:700]
dfTest = diabetesDF[700:750]
dfCheck=diabetesDF[750:]
# Séparation de l'étiquette et les fonctionnalités, pour les ensembles d'apprentissages et de test. 
# Conversion en tableau numpy cer les données seront gérés par l'algorithme d'apprentissage au format tableau numpy
trainLabel= np.asarray(dfTrain['Outcome'])
trainData=  np.asarray(dfTrain.drop('Outcome',1)) 
testLabel= np.asarray(dfTest['Outcome'])
testData=  np.asarray(dfTest.drop('Outcome',1))
# Normalisation des données
# Séparation de l'étiquette et les fonctionnalités, pour les ensembles d'apprentissages et de test. 
# Conversion en tableau numpy car les données seront gérées par l'algorithme d'apprentissage au format tableau numpy

means= np.mean(trainData, axis=0)
stds= np.std(trainData, axis=0)

trainData= (trainData - means)/stds
testData= (testData - means)/stds
diabetesCheck = LogisticRegression()
diabetesCheck.fit(trainData, trainLabel)
accuracy = diabetesCheck.score(testData, testLabel)
print("accuracy = ", accuracy * 100, "%")
coeff = list(diabetesCheck.coef_[0])
labels = list(dfTrain.drop('Outcome',1).columns)
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')
joblib.dump([diabetesCheck, means, stds], 'diabeteseModel.pkl')
diabetesLoadedModel, means, stds = joblib.load('diabeteseModel.pkl')
accuracyModel = diabetesLoadedModel.score(testData, testLabel)
print("accuracy = ",accuracyModel * 100,"%")
sampleData = dfCheck[:1]
 
# prepare sample  
sampleDataFeatures = np.asarray(sampleData.drop('Outcome',1))
sampleDataFeatures = (sampleDataFeatures - means)/stds
 
# predict 
predictionProbability = diabetesLoadedModel.predict_proba(sampleDataFeatures)
prediction = diabetesLoadedModel.predict(sampleDataFeatures)
print('Probability:', predictionProbability)
print('prediction:', prediction)
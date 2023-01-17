import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
#Read data and show head for an overview ------- Replace paths to run code from different machine
FullTrainData=pd.read_csv("../input/TrainData.csv",sep=';',encoding='latin-1')
FullTestData=pd.read_csv("../input/TestData.csv",sep=';',encoding='latin-1')
Template=pd.read_csv("../input/Lsungstemplate.csv")
FullTrainDataOrg=FullTrainData.copy()


FullTestData.info()

display(FullTrainData.info())
FullTrainData['Stammnummer'].unique().size
FullTrainData.columns
#Calculate fraction of positive vs. negative results
display(FullTrainData[(FullTrainData['Zielvariable']=='ja')].size/FullTrainData.size)
#Explore 'letzte Kampagne'
resultsLast=FullTrainData['Ergebnis letzte Kampagne'].unique()
display(resultsLast)
a=[]
j=0
for i in resultsLast:
    a.append(FullTrainData[(FullTrainData['Ergebnis letzte Kampagne']==i)].size)
    j=j+1
print(a)

#'Ergebnis letzte Kampagne' is in relation with 'Tage seit letzter Kampagne', strongest correlation is if there was a letzte Kampagne; the exact days are not so important.
# Therefore we drop the not compleatly filled #'Ergebnis letzte Kampagne'
FullTrainData.drop(columns=['Tage seit letzter Kampagne','Monat'],inplace=True)


#Start converting the data frame to numerics -> Use get_dummies to create digital categories(one-hot encoding)
NumData=pd.get_dummies(FullTrainData)
#drop double columns from one-hot encoding
NumData2=NumData.drop(columns=['Zielvariable_nein','Anruf-ID','Haus_nein','Kredit_nein','Geschlecht_m','Ausfall Kredit_nein']) 
#Calculate list of correlations
cors=pd.DataFrame(columns=['Kor','Name'])
for i in NumData2.columns:
    cors=cors.append({'Kor': NumData2[i].corr(NumData2['Zielvariable_ja']),'Name':i},ignore_index=True)

display(cors.sort_values('Kor'))

#cors.sort_values('Kor').to_csv("/Users/Therlos/Documents/DataScience/APOBank/Korrelationen.csv")
#Visualize how 'Kontostand' is distributed
# It may be good to introduce discret categories to protect against small fluctuations in 'Kontostand'
display(FullTrainData[(FullTrainData['Kontostand']<40000)].hist('Kontostand',bins=10))
display(FullTrainData[(FullTrainData['Kontostand']<40000) & (FullTrainData['Zielvariable']=='ja')].hist('Kontostand',bins=10))
display(FullTrainData[(FullTrainData['Kontostand']>-5000)& (FullTrainData['Kontostand']<40000) & (FullTrainData['Zielvariable']=='nein')].hist('Kontostand',bins=10))
#Indeed categories for 'Kontostand' improve the correlations between 'Kontostand' and 'Zielvariable'
display(NumData2['Kontostand'].corr(NumData2['Zielvariable_ja']))
NumData2['Kontostand']=pd.cut(NumData['Kontostand'],[-np.inf,0,2000,40000,90000,np.inf],labels=[0,1,2,3,4])
display(NumData2['Kontostand'].corr(NumData2['Zielvariable_ja']))
NumData2['Kontostand']=NumData2['Kontostand'].astype('int64',inplace=True)
#Visualize 'Monat' -> Comparing the histogram with the correlations table shows that especially the months
#with small number of events are strongly correlated to 'Zielvariable'. This could be an artefakt from the distribution of events.
#Therfore we take 'Monat' out of the data for the models (already done further above)
display(FullTrainDataOrg['Monat'].value_counts().plot(kind='bar'))


#Exploring the numer of contact events. Mostly only a single contact event. 
#The important feature 'Dauer' can only be used in realistic scenarios 
#to predict future converions after at least a single contact. 
#The ratio of conversions with number of contact events>1 to 
#total number of contact events is better compared to only a single contact.
#The models can thus be used to predict if further contacts are useful and a lot more customers can lead to a conversion.
#However tne number of contact events correlates negatively with the conversion. This results from the fact that the 
#distribution of contact events is strongly shifteted towards a single contact.
#This does not necessarily mean, that a single contact is better in reality; it is only better to predict the test set
display(NumData['Anzahl der Ansprachen'].unique().size)
display(NumData.hist('Anzahl der Ansprachen',bins=45))
display(NumData[(NumData['Zielvariable_ja']==1)].hist('Anzahl der Ansprachen',bins=45))
#Visualize 'Stammnummer' -> The Histograms show a strong correlation between 'Stammnummer' and 'Zielvariable'. Even though first unexpected, this could be traced back 
#to the fact that persons with a larger number are probably newer customers and are thus more interested in new products, compared to persons who are already setteled down
# with their needs 
display(FullTrainDataOrg[(FullTrainDataOrg['Zielvariable']=='ja')].hist('Stammnummer',bins=3))

#Visualize 'Dauer' -> Strong correlation. In principle this should only be available in reality after there was already contact to a person. Can be used to predict if a customer is likely to convert with the next contact. 
#The same is true for 'Anzahl Kontakte' 
NumData[(NumData['Zielvariable_ja']==1)].hist('Dauer',bins=20)
#Create Train and Test set
X2=NumData2.drop(columns=['Zielvariable_ja'])
y2=NumData2['Zielvariable_ja']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.33, random_state=42)


#Creat Data Manipulation function for the upcoming test data
def manipulate(Data):
    DataPuffer=Data.drop(columns=['Tage seit letzter Kampagne','Monat']).copy()
    NumDataValidate=pd.get_dummies(DataPuffer)
    NumDataValidate2=NumDataValidate.drop(columns=['Zielvariable','Anruf-ID','Haus_nein','Kredit_nein','Geschlecht_m','Ausfall Kredit_nein'])
    NumDataValidate2['Kontostand']=pd.cut(NumData['Kontostand'],[-np.inf,0,2000,40000,90000,np.inf],labels=[0,1,2,3,4])
    NumDataValidate2['Kontostand'].astype('int',inplace=True);
    return NumDataValidate2
    
#Plot performance dependence to get an estimate of sample leafs for grid search 
LeafScore=[]
for i in range(40,170):
    clf2 = DecisionTreeClassifier(min_samples_leaf=i,criterion='gini',random_state=42)
    clf2 = clf2.fit(X_train2,y_train2)
    LeafScore.append(metrics.roc_auc_score(y_test2,clf2.predict_proba(X_test2)[:,1]))
plt.plot(range(40,170),LeafScore)   
#Grid Search for DecisionTree
param_grid_tree={'min_samples_leaf':range(100,140),'max_features':range(15,35)}
gridsearchTree = GridSearchCV(clf2, param_grid_tree,n_jobs=-1,cv=3)
gridsearchTree.fit(X_train2, y_train2);
gridsearchTree.best_params_
#Define final decision tree
clf2 = DecisionTreeClassifier(min_samples_leaf=110,max_features=28,criterion='entropy',class_weight={1:1,0:1},random_state=42)
clf2 = clf2.fit(X_train2,y_train2)
clf2.score(X_test2,y_test2)

#Calculate some metrics auc score , cross validated auc score for the decision tree
display(metrics.roc_auc_score(y_test2,clf2.predict_proba(X_test2)[:,1]))
display(cross_val_score(clf2,X_test2,y_test2,cv=5,scoring='roc_auc').mean())
display(cross_val_score(clf2,X_test2,y_test2,cv=5,scoring='roc_auc').std())
confmat2=metrics.confusion_matrix(y_test2,(clf2.predict_proba(X_test2)[:,1] >= 0.5).astype(bool))
display(confmat2[0][1]/(confmat2[1][1]+confmat2[0][1]))
#Plot performance dependence to get an estimate of sample leafs for grid search (Random Forest)
LeafScore=[]
for i in range(1,30):
    clfForest = RandomForestClassifier(min_samples_leaf=i,n_estimators=10,random_state=42)
    clfForest = clfForest.fit(X_train2,y_train2)
    LeafScore.append(metrics.roc_auc_score(y_test2,clfForest.predict_proba(X_test2)[:,1]))
plt.plot(LeafScore)
#Grid Search for Forest
param_grid_forest={'min_samples_leaf':range(1,30,3),'max_features':range(10,35,5)}
gridsearchforest = GridSearchCV(clfForest, param_grid_forest,n_jobs=-1)
gridsearchforest.fit(X_train2, y_train2);
gridsearchforest.best_params_
gridsearchforest.best_score_
#Define Random Forest and calcuate some metrics
clfForest = RandomForestClassifier(min_samples_leaf=19,n_estimators=500,max_features=20)
clfForest = clfForest.fit(X_train2,y_train2)
confmatForest=metrics.confusion_matrix(y_test2,clfForest.predict(X_test2))
display(confmatForest)
display(metrics.roc_auc_score(y_test2,(clfForest.predict_proba(X_test2)[:,1]+(clf2.predict_proba(X_test2)[:,1])/2)))
display(confmatForest[0][1]/(confmatForest[1][1]+confmatForest[0][1]))
display(clfForest.score(X_test2,y_test2))
cross_val_score(clfForest,X_test2,y_test2,cv=5,scoring='roc_auc').mean()
#Define Logistic Regression
scalerRegressionx = StandardScaler()
X_train2Scale=scalerRegressionx.fit_transform(X_train2)
X_test2Scale=scalerRegressionx.transform(X_test2)
clfRegression=LogisticRegression(solver='liblinear',C=0.1,random_state=42,penalty='l2',class_weight={1:1,0:1})
clfRegression.fit(X_train2Scale,y_train2)
confmatRegression=metrics.confusion_matrix(y_test2,clfRegression.predict(X_test2Scale))
display(confmatRegression)
display(metrics.roc_auc_score(y_test2,clfRegression.predict_proba(X_test2Scale)[:,1]))
display(confmatRegression[1][1]/(confmatRegression[1][0]+confmatRegression[1][1]))
display(confmatRegression[0][1]/(confmatRegression[0][1]+confmatRegression[0][0]))
cross_val_score(clfRegression,X_test2Scale,y_test2,cv=5,scoring='roc_auc').mean()
#Do grid search for Logistic Regression 
param_grid = {'C': [0.001,0.002,0.003, 0.1,0.3, 1, 10, 100], 'penalty': ['l1', 'l2']}

gridsearch = GridSearchCV(clfRegression, param_grid)

gridsearch.fit(X_train2Scale, y_train2);
gridsearch.best_params_
#Best auc score for Random Forest -> use this model for predictions
testData=manipulate(FullTestData)
SolDec=(clfForest.predict_proba(testData)[:,1])
#Save predictions to file
SolMap={'ID':testData['Stammnummer'],'Expected':SolDec}
preds=pd.DataFrame(data=SolMap)
preds.set_index('ID',inplace=True)
#preds.to_csv("/Users/Therlos/Documents/DataScience/APOBank/Predictions.csv")

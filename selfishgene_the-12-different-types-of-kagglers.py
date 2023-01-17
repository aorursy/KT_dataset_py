import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib

import numpy as np

import pandas as pd

from sklearn import cluster, decomposition, preprocessing
#%% load data

multipleChoiceResponsesDF = pd.read_csv('../input/multipleChoiceResponses.csv', encoding='ISO-8859-1', low_memory=False)

processedDF = multipleChoiceResponsesDF.copy()

processedDF = processedDF.fillna(value='NaN')

allQuestions = processedDF.columns.tolist()
#%% create annual salary in US dollars feature

processedDF['CompensationAmount'] = processedDF['CompensationAmount'].str.replace(',','')

processedDF['CompensationAmount'] = processedDF['CompensationAmount'].str.replace('-','')

processedDF.loc[processedDF['CompensationAmount'] == 'NaN','CompensationAmount'] = '0'

processedDF.loc[processedDF['CompensationAmount'] == '','CompensationAmount'] = '0'

processedDF['CompensationAmount'] = processedDF['CompensationAmount'].astype(float)



conversionRates = pd.read_csv('../input/conversionRates.csv', encoding='ISO-8859-1').set_index('originCountry')

conversionRates = pd.read_csv('../input/conversionRates.csv').set_index('originCountry')

conversionRates.loc['USD']['exchangeRate']



exchangeRate = []

for row in range(processedDF.shape[0]):

    if processedDF.loc[row,'CompensationCurrency'] not in conversionRates.index.tolist():

        exchangeRate.append(1.0)

    else:

        exchangeRate.append(conversionRates.loc[processedDF.loc[row,'CompensationCurrency']]['exchangeRate'])

        

processedDF['exchangeRate'] = exchangeRate

processedDF['annualSalary_USD'] = processedDF['CompensationAmount']*processedDF['exchangeRate']



processedDF.loc[processedDF['annualSalary_USD'] > 300000, 'annualSalary_USD'] = 300000

processedDF['annualSalary_USD'] = processedDF['annualSalary_USD']/1000.0
#%% collect all basic features ('age','education level','seniority', 'salary', ...)

def GetDictValueForKey(x):

    return answerToNumericalDict[x]



basicFeatures = ['Country','GenderSelect','Age','FormalEducation','Tenure','annualSalary_USD','MajorSelect','EmploymentStatus','CurrentJobTitleSelect','LanguageRecommendationSelect','TimeSpentStudying']

basicSubsetDF = processedDF[basicFeatures]



additionalFeatures = ['FirstTrainingSelect','ProveKnowledgeSelect','AlgorithmUnderstandingLevel','MLMethodNextYearSelect','MLToolNextYearSelect','HardwarePersonalProjectsSelect','JobSearchResource','EmployerSearchMethod']

additionalSubsetDF = processedDF[additionalFeatures]



# add impatience variables that counts the number of NaNs in a given row for the basic and additional subsets

def CountNaNs(row):

    return (row == 'NaN').sum()



basicSubsetDF['impatience_basic'] = basicSubsetDF.apply(CountNaNs,axis=1)

basicSubsetDF['impatience_additional'] = additionalSubsetDF.apply(CountNaNs,axis=1)

basicSubsetDF['impatience'] = basicSubsetDF['impatience_basic'] + basicSubsetDF['impatience_additional']



# cap age to be in [15,85] range

basicSubsetDF.loc[basicSubsetDF['Age'] == 'NaN','Age'] = basicSubsetDF.loc[basicSubsetDF['Age'] != 'NaN','Age'].mean()

basicSubsetDF.loc[basicSubsetDF['Age'] <= 15,'Age'] = 15

basicSubsetDF.loc[basicSubsetDF['Age'] >= 85,'Age'] = 85



basicSubsetNumericDF = pd.DataFrame()

basicSubsetNumericDF['Age'] = basicSubsetDF['Age']



# transform formal education into an ordinal variable

answerToNumericalDict = {'I prefer not to answer': 10.0,

                         'NaN': 11.0,

                         'I did not complete any formal education past high school': 12.0,

                         'Professional degree': 14.0,

                         "Some college/university study without earning a bachelor's degree": 14.5,

                         "Bachelor's degree": 15.5,

                         "Master's degree": 18.0,

                         "Doctoral degree": 22.0}



basicSubsetNumericDF['Education_Years'] = basicSubsetDF['FormalEducation'].apply(GetDictValueForKey)



# transform tenure into an ordinal variable

answerToNumericalDict = {"I don't write code to analyze data": -0.5,

                         'NaN': 0.0,

                         'Less than a year': 0.5,

                         '1 to 2 years': 1.5,

                         '3 to 5 years': 4.0,

                         '6 to 10 years': 8.0,

                         'More than 10 years': 12.0}



basicSubsetNumericDF['Experience_Years'] = basicSubsetDF['Tenure'].apply(GetDictValueForKey)



# anual salary

basicSubsetNumericDF['annualSalary_USD'] = basicSubsetDF['annualSalary_USD']



# gender to numerical 

answerToNumericalDict = {'Male': -1.0,

                         'NaN': 0.0,

                         'A different identity': 0.0,

                         'Non-binary, genderqueer, or gender non-conforming': 0.0,

                         'Female': 1.0}



basicSubsetNumericDF['Gender'] = basicSubsetDF['GenderSelect'].apply(GetDictValueForKey)



# transform time spent studying to ordinal

answerToNumericalDict = {'NaN': 0.0,

                         '0 - 1 hour': 0.5,

                         '2 - 10 hours': 6.0,

                         '11 - 39 hours': 25.0,

                         '40+': 45.0}



basicSubsetNumericDF['Study_Hours'] = basicSubsetDF['TimeSpentStudying'].apply(GetDictValueForKey)



# add impatience field

basicSubsetNumericDF['impatience'] = basicSubsetDF['impatience']
basicSubsetNumericDF.head(15)
#%% show correlations between the most basic feature

basicSubsetNoisyNumericDF = basicSubsetNumericDF.copy()

basicSubsetNoisyNumericDF['Age'] = basicSubsetNoisyNumericDF['Age'].astype(float)

plt.figure(figsize=(12,10)); plt.title('Basic Features - Correlation Matrix', fontsize=22)

sns.heatmap(basicSubsetNoisyNumericDF.corr(), vmin=-1, vmax=1, fmt='.2f', annot=True, cmap='jet'); 

plt.yticks(rotation=0); plt.xticks(rotation=15);
for col in ['impatience','Gender','Education_Years','Experience_Years','Study_Hours']:

    basicSubsetNoisyNumericDF[col] *= (1.0 + 0.05*np.random.randn(basicSubsetNoisyNumericDF.shape[0]))

    basicSubsetNoisyNumericDF[col] += 0.3*np.random.randn(basicSubsetNoisyNumericDF.shape[0])



g = sns.pairplot(basicSubsetNoisyNumericDF, diag_kind="kde", plot_kws=dict(s=2, edgecolor="r", alpha=0.1), diag_kws=dict(shade=True));

g.fig.subplots_adjust(top=0.95);

g.fig.suptitle('Basic Features - Pair Scatter Plots', fontsize=30);
#%% apply whitening on each of the basic numeric features we've seen so far

scaledBasicSubset = preprocessing.StandardScaler().fit_transform(basicSubsetNumericDF.values);

numericDF = pd.DataFrame(scaledBasicSubset,columns=basicSubsetNumericDF.columns);



#%% apply one hot encoding to all other features and add it to our numeric dataframe

listOfColsToOneHotEncode = ['Country','MajorSelect','EmploymentStatus','CurrentJobTitleSelect',

                            'LanguageRecommendationSelect','FirstTrainingSelect','ProveKnowledgeSelect',

                            'AlgorithmUnderstandingLevel','MLMethodNextYearSelect','MLToolNextYearSelect',

                            'HardwarePersonalProjectsSelect','JobSearchResource','EmployerSearchMethod']



for col in listOfColsToOneHotEncode:

    labelEncoder = preprocessing.LabelEncoder()

    labelTfromed = labelEncoder.fit_transform(processedDF[col])

    oneHotEncoder = preprocessing.OneHotEncoder()

    oneHotTformed = oneHotEncoder.fit_transform(labelTfromed.reshape(-1,1))

    currOneHotDF = pd.DataFrame(oneHotTformed.todense(), columns = [col+'_OneHot_'+str(x) for x in range(len(labelEncoder.classes_))])

    numericDF = pd.concat((numericDF,currOneHotDF),axis=1)
#%% add learning platform usefulness features to our numeric dataframe

def GetDictValueForKey(x):

    return answerToNumericalDict[x]



allLearningPlatformColumns = [q for q in allQuestions if q.find('LearningPlatformUsefulness') >= 0]

answerToNumericalDict = {'Not Useful':-1.0,'NaN':0.0,'Somewhat useful':1.0,'Very useful':2.0}



learningUsefulnessOrigDF = processedDF.loc[:,allLearningPlatformColumns]

learningUsefulnessOrigDF = learningUsefulnessOrigDF.applymap(GetDictValueForKey)



# compress cols to eliminate outliers and apply whitening using PCA

numComponents = 12

learningUsefulnessPCAModel = decomposition.PCA(n_components=numComponents,whiten=True)

learningUsefulnessFeatures = learningUsefulnessPCAModel.fit_transform(learningUsefulnessOrigDF)



explainedVarVec = learningUsefulnessPCAModel.explained_variance_ratio_

print('Total explained percent by PCA model with %d components is %.1f%s' %(numComponents, 100*explainedVarVec.sum(),'%'))



newColNames = ['learning_PCA_%d'%(x+1) for x in range(numComponents)]

learningUsefulnessDF = pd.DataFrame(data=learningUsefulnessFeatures, columns=newColNames)



importanceWeight = 0.5

numericDF = pd.concat((numericDF, (importanceWeight/numComponents)*learningUsefulnessDF),axis=1)
#%% add job skill imporance features to our numeric dataframe

allJobSkillColumns = [q for q in allQuestions if q.find('JobSkillImportance') >= 0] 

answerToNumericalDict = {'Unnecessary':-1.0,'NaN':0.0,'Nice to have':1.0,'Necessary':2.0}



jobSkillOrigDF = processedDF.loc[:,allJobSkillColumns]

jobSkillOrigDF = jobSkillOrigDF.applymap(GetDictValueForKey)



# compress cols to eliminate outliers and apply whitening using PCA

numComponents = 7

jobSkillPCAModel = decomposition.PCA(n_components=numComponents,whiten=True)

jobSkillFeatures = jobSkillPCAModel.fit_transform(jobSkillOrigDF)



explainedVarVec = jobSkillPCAModel.explained_variance_ratio_

print('Total explained percent by PCA model with %d components is %.1f%s' %(numComponents, 100*explainedVarVec.sum(),'%'))



newColNames = ['jobSkill_PCA_%d'%(x+1) for x in range(numComponents)]

jobSkillDF = pd.DataFrame(data=jobSkillFeatures, columns=newColNames)



importanceWeight = 0.5

numericDF = pd.concat((numericDF, (importanceWeight/numComponents)*jobSkillDF),axis=1)
#%% add work tools and methods frequency features to our dataframe

allWorkToolsColumns = [q for q in allQuestions if q.find('WorkToolsFrequency') >= 0] 

allWorkMethodsColumns = [q for q in allQuestions if q.find('WorkMethodsFrequency') >= 0] 

answerToNumericalDict = {'NaN':0.0,'Rarely':1.0,'Sometimes':2.0,'Often':3.0,'Most of the time':4.0}



workToolsOrigDF = processedDF.loc[:,allWorkToolsColumns+allWorkMethodsColumns]

workToolsOrigDF = workToolsOrigDF.applymap(GetDictValueForKey)



# compress cols to eliminate outliers and apply whitening using PCA

numComponents = 38

workToolsPCAModel = decomposition.PCA(n_components=numComponents,whiten=True)

workToolsFeatures = workToolsPCAModel.fit_transform(workToolsOrigDF)



explainedVarVec = workToolsPCAModel.explained_variance_ratio_

print('Total explained percent by PCA model with %d components is %.1f%s' %(numComponents, 100*explainedVarVec.sum(),'%'))



newColNames = ['workTools_PCA_%d'%(x+1) for x in range(numComponents)]

workToolsDF = pd.DataFrame(data=workToolsFeatures, columns=newColNames)



importanceWeight = 0.5

numericDF = pd.concat((numericDF, (importanceWeight/numComponents)*workToolsDF),axis=1)
#%% add work challanges features to our dataframe

allWorkChallengesColumns = [q for q in allQuestions if q.find('WorkChallengeFrequency') >= 0]

answerToNumericalDict = {'NaN':0.0,'Rarely':1.0,'Sometimes':2.0,'Often':3.0,'Most of the time':4.0}



workChallangesOrigDF = processedDF.loc[:,allWorkChallengesColumns]

workChallangesOrigDF = workChallangesOrigDF.applymap(GetDictValueForKey)



# compress cols to eliminate outliers and apply whitening using PCA

numComponents = 16

workChallengesPCAModel = decomposition.PCA(n_components=numComponents,whiten=True)

workChallengesFeatures = workChallengesPCAModel.fit_transform(workChallangesOrigDF)



explainedVarVec = workChallengesPCAModel.explained_variance_ratio_

print('Total explained percent by PCA model with %d components is %.1f%s' %(numComponents, 100*explainedVarVec.sum(),'%'))



newColNames = ['workChallenges_PCA_%d'%(x+1) for x in range(numComponents)]

workChallengesDF = pd.DataFrame(data=workChallengesFeatures, columns=newColNames)



importanceWeight = 0.5

numericDF = pd.concat((numericDF, (importanceWeight/numComponents)*workChallengesDF),axis=1)
#%% add job selection factors features to our dataframe

allJobFactorsColumns = [q for q in allQuestions if q.find('JobFactor') >= 0] 

answerToNumericalDict = {'Not important':-1.0,'NaN':0.0,'Somewhat important':1.0,'Very Important':2.0}



jobPreferenceOrigDF = processedDF.loc[:,allJobFactorsColumns]

jobPreferenceOrigDF = jobPreferenceOrigDF.applymap(GetDictValueForKey)



# compress cols to eliminate outliers and apply whitening using PCA

numComponents = 10

jobPreferencePCAModel = decomposition.PCA(n_components=numComponents,whiten=True)

jobPreferenceFeatures = jobPreferencePCAModel.fit_transform(jobPreferenceOrigDF)



explainedVarVec = jobPreferencePCAModel.explained_variance_ratio_

print('Total explained percent by PCA model with %d components is %.1f%s' %(numComponents, 100*explainedVarVec.sum(),'%'))



newColNames = ['jobPreference_PCA_%d'%(x+1) for x in range(numComponents)]

jobPreferenceDF = pd.DataFrame(data=jobPreferenceFeatures, columns=newColNames)



importanceWeight = 0.5

numericDF = pd.concat((numericDF, (importanceWeight/numComponents)*jobPreferenceDF),axis=1)
#%% add time allocation distribution features to our dataframe

def ReplaceOnlyNaNs(x):

    if x == 'NaN':

        return 0.0

    else:

        return x



allTimeAllocationColumns = ['TimeGatheringData', 'TimeModelBuilding', 'TimeProduction', 'TimeVisualizing', 'TimeFindingInsights', 'TimeOtherSelect']

timeAllocationOrigDF = processedDF.loc[:,allTimeAllocationColumns]

timeAllocationOrigDF = timeAllocationOrigDF.applymap(ReplaceOnlyNaNs)



numComponents = 4

timeAllocationPCAModel = decomposition.PCA(n_components=numComponents,whiten=True)

timeAllocationFeatures = timeAllocationPCAModel.fit_transform(timeAllocationOrigDF)



explainedVarVec = timeAllocationPCAModel.explained_variance_ratio_

print('Total explained percent by PCA model with %d components is %.1f%s' %(numComponents, 100*explainedVarVec.sum(),'%'))



newColNames = ['timeAllocation_PCA_%d'%(x+1) for x in range(numComponents)]

timeAllocationeDF = pd.DataFrame(data=timeAllocationFeatures, columns=newColNames)



importanceWeight = 0.5

numericDF = pd.concat((numericDF, (importanceWeight/numComponents)*jobPreferenceDF),axis=1)
numericDF.shape
#%% we now finally have a numeric representation of the dataset and we are ready to cluster the users

listOfNumClusters = [1,2,4,6,9,12,16,32,64,128,256]

listOfInertia = []

for numClusters in listOfNumClusters:

    KMeansModel = cluster.MiniBatchKMeans(n_clusters=numClusters, batch_size=2100, n_init=5, random_state=1)

    KMeansModel.fit(numericDF)

    listOfInertia.append(KMeansModel.inertia_)

explainedPercent = 100*(1-(np.array(listOfInertia)/listOfInertia[0]))



# plot the explained percent as a function of number of clusters

percentExplainedTarget = 40



numDesiredClusterInd = np.nonzero(explainedPercent > percentExplainedTarget)[0][0]

numDesiredClusters = listOfNumClusters[numDesiredClusterInd]



explainedPercentReached = explainedPercent[numDesiredClusterInd]

plt.figure(figsize=(14,6)); plt.plot(listOfNumClusters,explainedPercent,c='b')

plt.scatter(numDesiredClusters,explainedPercentReached,s=150,c='r')

plt.xlabel('Number of Clusters', fontsize=20); plt.ylabel('Explained Percent', fontsize=20)

plt.title('Desired Number of Clusters = %d, Explained Percent = %.2f%s' %(numDesiredClusters,explainedPercentReached,'%'),fontsize=22);

plt.xlim(-1,listOfNumClusters[-1]+1); plt.ylim(0,60);
#%% for the selected number of clusters, redo the Kmeans and sort the clusters by frequency

KMeansModel = cluster.KMeans(n_clusters=numDesiredClusters, n_init=15, random_state=10)

KMeansModel.fit(numericDF)



clusterInds = KMeansModel.predict(numericDF)



clusterFrequency = []

for clusterInd in range(numDesiredClusters):

    clusterFrequency.append((clusterInds == clusterInd).sum()/float(len(clusterInds)))

clusterFrequency = np.array(clusterFrequency)

sortedClusterFrequency = np.flipud(np.sort(np.array(clusterFrequency)))

sortedClustersByFrequency = np.flipud(np.argsort(clusterFrequency))
#%% show the attribures of most frequent kaggler

def GetMstCommonElement(a_list):

    return max(set(a_list), key=a_list.count)



# select cluster

k = 0

selectedCluster = sortedClustersByFrequency[k]



# find nearest neighbors

numNeighbors = 15

distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]

distFromCluster[clusterInds != selectedCluster] = np.inf

nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]
basicSubsetDF.loc[nearestNeighborInds,:]
additionalSubsetDF.loc[nearestNeighborInds,:]
#show original data for neighbors

print('-'*40)

print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))

print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))

print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))

print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))

print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))

print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))

print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))

print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))

print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))

print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))

print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))

print('-'*40)
#%% show the attribures of most frequent kaggler

def GetMstCommonElement(a_list):

    return max(set(a_list), key=a_list.count)



# select cluster

k = 1

selectedCluster = sortedClustersByFrequency[k]



# find nearest neighbors

numNeighbors = 15

distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]

distFromCluster[clusterInds != selectedCluster] = np.inf

nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]
basicSubsetDF.loc[nearestNeighborInds,:]
additionalSubsetDF.loc[nearestNeighborInds,:]
#show original data for neighbors

print('-'*40)

print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))

print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))

print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))

print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))

print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))

print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))

print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))

print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))

print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))

print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))

print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))

print('-'*40)
#%% show the attribures of most frequent kaggler

def GetMstCommonElement(a_list):

    return max(set(a_list), key=a_list.count)



# select cluster

k = 2

selectedCluster = sortedClustersByFrequency[k]



# find nearest neighbors

numNeighbors = 15

distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]

distFromCluster[clusterInds != selectedCluster] = np.inf

nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]
basicSubsetDF.loc[nearestNeighborInds,:]
additionalSubsetDF.loc[nearestNeighborInds,:]
#show original data for neighbors

print('-'*40)

print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))

print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))

print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))

print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))

print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))

print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))

print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))

print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))

print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))

print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))

print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))

print('-'*40)
#%% show the attribures of most frequent kaggler

def GetMstCommonElement(a_list):

    return max(set(a_list), key=a_list.count)



# select cluster

k = 3

selectedCluster = sortedClustersByFrequency[k]



# find nearest neighbors

numNeighbors = 15

distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]

distFromCluster[clusterInds != selectedCluster] = np.inf

nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]
basicSubsetDF.loc[nearestNeighborInds,:]
additionalSubsetDF.loc[nearestNeighborInds,:]
#show original data for neighbors

print('-'*40)

print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))

print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))

print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))

print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))

print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))

print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))

print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))

print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))

print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))

print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))

print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))

print('-'*40)
#%% show the attribures of most frequent kaggler

def GetMstCommonElement(a_list):

    return max(set(a_list), key=a_list.count)



# select cluster

k = 4

selectedCluster = sortedClustersByFrequency[k]



# find nearest neighbors

numNeighbors = 15

distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]

distFromCluster[clusterInds != selectedCluster] = np.inf

nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]
basicSubsetDF.loc[nearestNeighborInds,:]
additionalSubsetDF.loc[nearestNeighborInds,:]
#show original data for neighbors

print('-'*40)

print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))

print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))

print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))

print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))

print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))

print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))

print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))

print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))

print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))

print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))

print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))

print('-'*40)
#%% show the attribures of most frequent kaggler

def GetMstCommonElement(a_list):

    return max(set(a_list), key=a_list.count)



# select cluster

k = 5

selectedCluster = sortedClustersByFrequency[k]



# find nearest neighbors

numNeighbors = 15

distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]

distFromCluster[clusterInds != selectedCluster] = np.inf

nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]
basicSubsetDF.loc[nearestNeighborInds,:]
additionalSubsetDF.loc[nearestNeighborInds,:]
#show original data for neighbors

print('-'*40)

print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))

print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))

print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))

print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))

print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))

print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))

print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))

print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))

print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))

print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))

print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))

print('-'*40)
#%% show the attribures of most frequent kaggler

def GetMstCommonElement(a_list):

    return max(set(a_list), key=a_list.count)



# select cluster

k = 6

selectedCluster = sortedClustersByFrequency[k]



# find nearest neighbors

numNeighbors = 15

distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]

distFromCluster[clusterInds != selectedCluster] = np.inf

nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]
basicSubsetDF.loc[nearestNeighborInds,:]
additionalSubsetDF.loc[nearestNeighborInds,:]
#show original data for neighbors

print('-'*40)

print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))

print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))

print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))

print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))

print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))

print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))

print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))

print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))

print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))

print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))

print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))

print('-'*40)
#%% show the attribures of most frequent kaggler

def GetMstCommonElement(a_list):

    return max(set(a_list), key=a_list.count)



# select cluster

k = 7

selectedCluster = sortedClustersByFrequency[k]



# find nearest neighbors

numNeighbors = 15

distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]

distFromCluster[clusterInds != selectedCluster] = np.inf

nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]
basicSubsetDF.loc[nearestNeighborInds,:]
additionalSubsetDF.loc[nearestNeighborInds,:]
#show original data for neighbors

print('-'*40)

print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))

print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))

print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))

print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))

print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))

print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))

print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))

print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))

print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))

print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))

print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))

print('-'*40)
#%% show the attribures of most frequent kaggler

def GetMstCommonElement(a_list):

    return max(set(a_list), key=a_list.count)



# select cluster

k = 8

selectedCluster = sortedClustersByFrequency[k]



# find nearest neighbors

numNeighbors = 15

distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]

distFromCluster[clusterInds != selectedCluster] = np.inf

nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]
basicSubsetDF.loc[nearestNeighborInds,:]
additionalSubsetDF.loc[nearestNeighborInds,:]
#show original data for neighbors

print('-'*40)

print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))

print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))

print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))

print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))

print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))

print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))

print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))

print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))

print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))

print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))

print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))

print('-'*40)
#%% show the attribures of most frequent kaggler

def GetMstCommonElement(a_list):

    return max(set(a_list), key=a_list.count)



# select cluster

k = 9

selectedCluster = sortedClustersByFrequency[k]



# find nearest neighbors

numNeighbors = 15

distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]

distFromCluster[clusterInds != selectedCluster] = np.inf

nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]
basicSubsetDF.loc[nearestNeighborInds,:]
additionalSubsetDF.loc[nearestNeighborInds,:]
#show original data for neighbors

print('-'*40)

print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))

print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))

print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))

print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))

print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))

print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))

print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))

print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))

print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))

print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))

print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))

print('-'*40)
#%% show the attribures of most frequent kaggler

def GetMstCommonElement(a_list):

    return max(set(a_list), key=a_list.count)



# select cluster

k = 10

selectedCluster = sortedClustersByFrequency[k]



# find nearest neighbors

numNeighbors = 15

distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]

distFromCluster[clusterInds != selectedCluster] = np.inf

nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]
basicSubsetDF.loc[nearestNeighborInds,:]
additionalSubsetDF.loc[nearestNeighborInds,:]
#show original data for neighbors

print('-'*40)

print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))

print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))

print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))

print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))

print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))

print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))

print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))

print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))

print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))

print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))

print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))

print('-'*40)
#%% show the attribures of most frequent kaggler

def GetMstCommonElement(a_list):

    return max(set(a_list), key=a_list.count)



# select cluster

k = 11

selectedCluster = sortedClustersByFrequency[k]



# find nearest neighbors

numNeighbors = 15

distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]

distFromCluster[clusterInds != selectedCluster] = np.inf

nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]
basicSubsetDF.loc[nearestNeighborInds,:]
additionalSubsetDF.loc[nearestNeighborInds,:]
#show original data for neighbors

print('-'*40)

print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))

print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))

print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))

print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))

print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))

print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))

print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))

print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))

print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))

print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))

print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))

print('-'*40)
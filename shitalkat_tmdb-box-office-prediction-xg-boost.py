import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import ast

pd.options.display.max_columns=100

pd.options.display.max_rows=100
trainDF=pd.read_csv('../input/train.csv')

unseenTestDF=pd.read_csv('../input/test.csv')
def GetCSVFromListOfDict(keyNameToFetch,column,columnName):

    column=column.copy()

    column=column.fillna('[{}]')

    columnList=[]

    for index,row in column.iteritems():

        columnStr=''

        listofDict=ast.literal_eval(row)

        for dic in listofDict:



            if(keyNameToFetch in dic.keys()):

                columnStr=columnStr+';'+str(dic[keyNameToFetch]) 

        columnStr=columnStr.strip(';') # trim leading ;

        columnList.append(columnStr)



    tempDF=pd.DataFrame(columnList,columns=[columnName])

    return tempDF[columnName]





#GetCSVFromListOfDict('iso_639_1',trainDF.spoken_languages,'spoken_languages')
trainDF['belongs_to_collection']=GetCSVFromListOfDict('name',trainDF.belongs_to_collection,'belongs_to_collection')

trainDF['genres']=GetCSVFromListOfDict('name',trainDF.genres,'genres')

trainDF['production_companies']=GetCSVFromListOfDict('name',trainDF.production_companies,'production_companies')

trainDF['production_countries']=GetCSVFromListOfDict('name',trainDF.production_countries,'production_countries')

trainDF['spoken_languages']=GetCSVFromListOfDict('iso_639_1',trainDF.spoken_languages,'spoken_languages')

trainDF['Keywords']=GetCSVFromListOfDict('name',trainDF.Keywords,'Keywords')

trainDF['Crew_Dept']=GetCSVFromListOfDict('department',trainDF.crew,'crew')

trainDF['Crew_Job']=GetCSVFromListOfDict('job',trainDF.crew,'crew')

trainDF['Crew_Name']=GetCSVFromListOfDict('name',trainDF.crew,'crew')

trainDF['Crew_Gender']=GetCSVFromListOfDict('gender',trainDF.crew,'crew')





unseenTestDF['belongs_to_collection']=GetCSVFromListOfDict('name',unseenTestDF.belongs_to_collection,'belongs_to_collection')

unseenTestDF['genres']=GetCSVFromListOfDict('name',unseenTestDF.genres,'genres')

unseenTestDF['production_companies']=GetCSVFromListOfDict('name',unseenTestDF.production_companies,'production_companies')

unseenTestDF['production_countries']=GetCSVFromListOfDict('name',unseenTestDF.production_countries,'production_countries')

unseenTestDF['spoken_languages']=GetCSVFromListOfDict('iso_639_1',unseenTestDF.spoken_languages,'spoken_languages')

unseenTestDF['Keywords']=GetCSVFromListOfDict('name',unseenTestDF.Keywords,'Keywords')

unseenTestDF['Crew_Dept']=GetCSVFromListOfDict('department',unseenTestDF.crew,'crew')

unseenTestDF['Crew_Job']=GetCSVFromListOfDict('job',unseenTestDF.crew,'crew')

unseenTestDF['Crew_Name']=GetCSVFromListOfDict('name',unseenTestDF.crew,'crew')

unseenTestDF['Crew_Gender']=GetCSVFromListOfDict('gender',unseenTestDF.crew,'crew')









display(trainDF.head(1))

display(unseenTestDF.head(1))


print(len(trainDF.belongs_to_collection))

trainDF.belongs_to_collection.value_counts()

# Out of 3000 total 2396 missing values. i.e. 79% missing values.

# Lets check whether missing value vs. present value has effect on revenue?
trainDF['belongs_to_collection_ISMISSING']=(trainDF.belongs_to_collection.str.strip()=='').astype(int)

unseenTestDF['belongs_to_collection_ISMISSING']=(unseenTestDF.belongs_to_collection.str.strip()=='').astype(int)

trainDF[['belongs_to_collection_ISMISSING','revenue']].corr()

trainDF.drop(columns=['belongs_to_collection'],inplace=True)

unseenTestDF.drop(columns=['belongs_to_collection'],inplace=True)
print(len(trainDF.genres))

print(trainDF.genres.isna().sum())

trainDF.genres.value_counts().head()

# No missing values. Good
trainDF['genres']=trainDF.genres.str.replace(' ','_') # so bigrams will act as unigram, and it wont become 2 columns

trainDF['genres']=trainDF.genres.str.replace(';',' ')





from sklearn.feature_extraction.text import CountVectorizer



vectFeatures = CountVectorizer(max_features=10)

vectFeatures.fit(trainDF['genres'])



featuresTrainSplit=vectFeatures.transform(trainDF['genres'])

featuresUnseenTestSplit=vectFeatures.transform(unseenTestDF['genres'])



featuresTrainDF=pd.DataFrame(featuresTrainSplit.toarray(),columns=vectFeatures.get_feature_names())

featuresUnseenTestDF=pd.DataFrame(featuresUnseenTestSplit.toarray(),columns=vectFeatures.get_feature_names())

featuresTrainDF.columns='genres_'+featuresTrainDF.columns

featuresUnseenTestDF.columns='genres_'+featuresUnseenTestDF.columns
trainDF=pd.concat([trainDF,featuresTrainDF],axis=1)

unseenTestDF=pd.concat([unseenTestDF,featuresUnseenTestDF],axis=1)
trainDF.drop(columns=['genres'],inplace=True)

unseenTestDF.drop(columns=['genres'],inplace=True)
print(len(trainDF.production_companies))

trainDF.production_companies.value_counts().head(20)

# 156 missing values out of 3000
print(len(trainDF.production_countries))

trainDF.production_countries.value_counts().head(20)

# 55 Missing values
trainDF['production_countries']=trainDF.production_countries.str.replace(' ','_') # so bigrams will act as unigram, and it wont become 2 columns

trainDF['production_countries']=trainDF.production_countries.str.replace(';',' ')





unseenTestDF['production_countries']=unseenTestDF.production_countries.str.replace(' ','_') # so bigrams will act as unigram, and it wont become 2 columns

unseenTestDF['production_countries']=unseenTestDF.production_countries.str.replace(';',' ')

trainDF['IsProductionFromUSA']=(trainDF['production_countries']=='united_states_of_america').astype(int)

unseenTestDF['IsProductionFromUSA']=(unseenTestDF['production_countries']=='united_states_of_america').astype(int)
trainDF.drop(columns=['production_countries'],inplace=True)

unseenTestDF.drop(columns=['production_countries'],inplace=True)
trainDF['IsEnglishLanguage']=(

                    (trainDF['spoken_languages'].str.contains('en'))

                    & 

                    (trainDF['original_language']=='en')).astype(int)







unseenTestDF['IsEnglishLanguage']=(

                    (unseenTestDF['spoken_languages'].str.contains('en'))

                    &

                    (unseenTestDF['original_language']=='en')).astype(int)
trainDF[['IsEnglishLanguage','revenue']].corr()
trainDF.drop(columns=['spoken_languages','original_language'],inplace=True)

unseenTestDF.drop(columns=['spoken_languages','original_language'],inplace=True)
trainDF['Keywords']=trainDF.Keywords.str.replace(' ','_') # so bigrams will act as unigram, and it wont become 2 columns

trainDF['Keywords']=trainDF.Keywords.str.replace(';',' ')

trainDF['Keywords']=trainDF['Keywords'].str.lower()





unseenTestDF['Keywords']=unseenTestDF.Keywords.str.replace(' ','_') # so bigrams will act as unigram, and it wont become 2 columns

unseenTestDF['Keywords']=unseenTestDF.Keywords.str.replace(';',' ')

unseenTestDF['Keywords']=unseenTestDF['Keywords'].str.lower()

from sklearn.feature_extraction.text import CountVectorizer



vectFeatures = CountVectorizer(max_features=20)

vectFeatures.fit(trainDF['Keywords'].str.lower())



featuresTrainSplit=vectFeatures.transform(trainDF['Keywords'])

featuresUnseenTestSplit=vectFeatures.transform(unseenTestDF['Keywords'])







featuresTrainDF=pd.DataFrame(featuresTrainSplit.toarray(),columns=vectFeatures.get_feature_names())

featuresUnseenTestDF=pd.DataFrame(featuresUnseenTestSplit.toarray(),columns=vectFeatures.get_feature_names())





featuresTrainDF.columns='Keywords'+featuresTrainDF.columns

featuresUnseenTestDF.columns='Keywords'+featuresUnseenTestDF.columns





trainDF=pd.concat([trainDF,featuresTrainDF],axis=1)

unseenTestDF=pd.concat([unseenTestDF,featuresUnseenTestDF],axis=1)



trainDF.drop(columns=['Keywords'],inplace=True)

unseenTestDF.drop(columns=['Keywords'],inplace=True)

trainDF.homepage.isna().sum()
trainDF.homepage
trainDF['IsHomePageAvailable']=(trainDF.homepage.isna()==False).astype(int)

unseenTestDF['IsHomePageAvailable']=(unseenTestDF.homepage.isna()==False).astype(int)
trainDF[['IsHomePageAvailable','revenue']].corr()
dateSplit=trainDF.release_date.str.extract('([0-9]+)/([0-9]+)/([0-9]+)')

dateSplit.columns=['ReleaseMonth','ReleaseDate','ReleaseYear']



dateSplit.loc[dateSplit.ReleaseYear.astype(int)>20,'ReleaseYear']='19'+dateSplit.loc[dateSplit.ReleaseYear.astype(int)>20,'ReleaseYear']

dateSplit.loc[dateSplit.ReleaseYear.astype(int)<=20,'ReleaseYear']='20'+dateSplit.loc[dateSplit.ReleaseYear.astype(int)<=20,'ReleaseYear']



trainDF.drop(columns=['release_date'],inplace=True)

trainDF=pd.concat([trainDF,dateSplit.astype(int)],axis=1)
print(unseenTestDF.release_date.mode())

unseenTestDF['release_date'].fillna('9/9/11',inplace=True)
unseenTestDF['release_date'].isna().sum()
dateSplit=unseenTestDF.release_date.str.extract('([0-9]+)/([0-9]+)/([0-9]+)')

dateSplit.columns=['ReleaseMonth','ReleaseDate','ReleaseYear']





dateSplit.loc[dateSplit.ReleaseYear.astype(int)>20,'ReleaseYear']='19'+dateSplit.loc[dateSplit.ReleaseYear.astype(int)>20,'ReleaseYear']

dateSplit.loc[dateSplit.ReleaseYear.astype(int)<=20,'ReleaseYear']='20'+dateSplit.loc[dateSplit.ReleaseYear.astype(int)<=20,'ReleaseYear']





unseenTestDF.drop(columns=['release_date'],inplace=True)

unseenTestDF=pd.concat([unseenTestDF,dateSplit.astype(int)],axis=1)

## Month -- > SeasonEnd feature engg
pd.concat([pd.get_dummies(trainDF['ReleaseMonth'].astype(str)),trainDF.revenue],axis=1).corr()['revenue']
trainDF.groupby(by='ReleaseMonth')['revenue'].mean()




pd.concat([((trainDF.ReleaseMonth==6) |

            (trainDF.ReleaseMonth==12)|

           (trainDF.ReleaseMonth==7)

           ).astype(int),trainDF.revenue],axis=1).corr()['revenue']









trainDF['IsReleaseMonthSeasonEnd']=((trainDF.ReleaseMonth==6) |

            (trainDF.ReleaseMonth==12)|

           (trainDF.ReleaseMonth==7)

           ).astype(int)



unseenTestDF['IsReleaseMonthSeasonEnd']=((unseenTestDF.ReleaseMonth==6) |

            (unseenTestDF.ReleaseMonth==12)|

           (unseenTestDF.ReleaseMonth==7)

           ).astype(int)





trainDF.drop(columns=['ReleaseMonth'],inplace=True)

unseenTestDF.drop(columns=['ReleaseMonth'],inplace=True)
trainDF.drop(columns=['ReleaseDate'],inplace=True)

unseenTestDF.drop(columns=['ReleaseDate'],inplace=True)
trainDF['revenue']=np.log1p(trainDF.revenue)



trainDF['budget']=np.log1p(trainDF.budget)

unseenTestDF['budget']=np.log1p(unseenTestDF.budget)





trainDF['popularity']=np.log1p(trainDF.popularity)

unseenTestDF['popularity']=np.log1p(unseenTestDF.popularity)


from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

trainDFNum=trainDF.select_dtypes(include=numerics)

unseenTestDFNum=unseenTestDF.select_dtypes(include=numerics)

trainDFNum.drop(columns=['id'],inplace=True)

unseenTestDFNum.drop(columns=['id'],inplace=True)
trainDFNum=trainDFNum.fillna(trainDFNum.median())

unseenTestDFNum=unseenTestDFNum.fillna(trainDFNum.median())
from sklearn import model_selection # for splitting into train and test

import sklearn

# Split-out validation dataset

X = trainDFNum.drop(columns=['revenue'])

Y = trainDFNum['revenue']



validation_size = 0.2

seed = 100

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
import xgboost

model_XG = xgboost.XGBRegressor() 

model_XG.fit(X_train, Y_train)


# make predictions for test data



trainResult_XG = model_XG.predict(X_train)

testResult_XG = model_XG.predict(X_test)

unseenTestResult_XG=model_XG.predict(unseenTestDFNum)


    



########## TRAIN DATA RESULT ##########



print('---------- TRAIN DATA RESULT ----------')

# The mean squared error

print("Mean squared error: %.5f"%np.sqrt( mean_squared_error(Y_train, trainResult_XG)))

# Explained variance score: 1 is perfect prediction

print('Variance score: %.4f' % r2_score(Y_train, trainResult_XG))









########## TEST DATA RESULT ##########



print('---------- TEST DATA RESULT ----------')

# The mean squared error

print("Mean squared error: %.5f"% np.sqrt(mean_squared_error(Y_test, testResult_XG)))

# Explained variance score: 1 is perfect prediction

print('Variance score: %.4f' % r2_score(Y_test, testResult_XG))









unseenTestResult_XG=np.expm1(unseenTestResult_XG)
submission=pd.DataFrame([unseenTestDF.id,unseenTestResult_XG]).T



submission.columns=['id','revenue']



submission.id=submission.id.astype(int)



submission.to_csv('submission.csv',index=False)

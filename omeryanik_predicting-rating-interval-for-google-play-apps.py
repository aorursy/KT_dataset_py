import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
data=pd.read_csv('../input/googleplaystore.csv')

data
data.Category.unique()
data.isnull().sum()
dataframeArtDesign=data.loc[data["Category"]=="ART_AND_DESIGN"]

avrArtDesign=dataframeArtDesign["Rating"].mean()

print(avrArtDesign)

dataframeAutoVehicles=data.loc[data["Category"]=="AUTO_AND_VEHICLES"]

avrAutoVehicles=dataframeAutoVehicles["Rating"].mean()

print(avrAutoVehicles)

dataframeBeauty=data.loc[data["Category"]=="BEAUTY"]

avrBeauty=dataframeBeauty["Rating"].mean()

print(avrBeauty)



dataframeBooksReference=data.loc[data["Category"]=="BOOKS_AND_REFERENCE"]

avrBooksReference=dataframeBooksReference["Rating"].mean()

print(avrBooksReference)

dataframeBusiness=data.loc[data["Category"]=="BUSINESS"]

avrBusiness=dataframeBusiness["Rating"].mean()

print(avrBusiness)

dataframeComics=data.loc[data["Category"]=="COMICS"]

avrComics=dataframeComics["Rating"].mean()

print(avrComics)

dataframeCommunication=data.loc[data["Category"]=="COMMUNICATION"]

avrCommunication=dataframeCommunication["Rating"].mean()

print(avrCommunication)

dataframeDating=data.loc[data["Category"]=="DATING"]

avrDating=dataframeDating["Rating"].mean()

print(avrDating)

dataframeEducation=data.loc[data["Category"]=="EDUCATION"]

avrEducation=dataframeEducation["Rating"].mean()

print(avrEducation)

dataframeEntertainment=data.loc[data["Category"]=="ENTERTAINMENT"]

avrEntertainment=dataframeEntertainment["Rating"].mean()

print(avrEntertainment)

dataframeEvents=data.loc[data["Category"]=="EVENTS"]

avrEvents=dataframeEvents["Rating"].mean()

print(avrEvents)

dataframeFinance=data.loc[data["Category"]=="FINANCE"]

avrFinance=dataframeFinance["Rating"].mean()

print(avrFinance)

dataframeFoodDrink=data.loc[data["Category"]=="FOOD_AND_DRINK"]

avrFoodDrink=dataframeFoodDrink["Rating"].mean()

print(avrFoodDrink)

dataframeHealthFitness=data.loc[data["Category"]=="HEALTH_AND_FITNESS"]

avrHealthFitness=dataframeHealthFitness["Rating"].mean()

print(avrHealthFitness)

dataframeHouseHome=data.loc[data["Category"]=="HOUSE_AND_HOME"]

avrHouseHome=dataframeHouseHome["Rating"].mean()

print(avrHouseHome)

dataframeLibrariesDemo=data.loc[data["Category"]=="LIBRARIES_AND_DEMO"]

avrLibrariesDemo=dataframeLibrariesDemo["Rating"].mean()

print(avrLibrariesDemo)

dataframeLifestyle=data.loc[data["Category"]=="LIFESTYLE"]

avrLifestyle=dataframeLifestyle["Rating"].mean()

print(avrLifestyle)

dataframeGame=data.loc[data["Category"]=="GAME"]

avrGame=dataframeGame["Rating"].mean()

print(avrGame)

dataframeFamily=data.loc[data["Category"]=="FAMILY"]

avrFamily=dataframeFamily["Rating"].mean()

print(avrFamily)

dataframeMedical=data.loc[data["Category"]=="MEDICAL"]

avrMedical=dataframeMedical["Rating"].mean()

print(avrMedical)

dataframeSocial=data.loc[data["Category"]=="SOCIAL"]

avrSocial=dataframeSocial["Rating"].mean()

print(avrSocial)

dataframeShopping=data.loc[data["Category"]=="SHOPPING"]

avrShopping=dataframeShopping["Rating"].mean()

print(avrShopping)

dataframePhotography=data.loc[data["Category"]=="PHOTOGRAPHY"]

avrPhotography=dataframePhotography["Rating"].mean()

print(avrPhotography)

dataframeSports=data.loc[data["Category"]=="SPORTS"]

avrSports=dataframeSports["Rating"].mean()

print(avrSports)

dataframeTravelLocal=data.loc[data["Category"]=="TRAVEL_AND_LOCAL"]

avrTravelLocal=dataframeTravelLocal["Rating"].mean()

print(avrTravelLocal)

dataframeTools=data.loc[data["Category"]=="TOOLS"]

avrTools=dataframeTools["Rating"].mean()

print(avrTools)

dataframePersonalization=data.loc[data["Category"]=="PERSONALIZATION"]

avrPersonalization=dataframePersonalization["Rating"].mean()

print(avrPersonalization)

dataframeProductivity=data.loc[data["Category"]=="PRODUCTIVITY"]

avrProductivity=dataframeProductivity["Rating"].mean()

print(avrProductivity)

dataframeParenting=data.loc[data["Category"]=="PARENTING"]

avrParenting=dataframeParenting["Rating"].mean()

print(avrParenting)

dataframeWeather=data.loc[data["Category"]=="WEATHER"]

avrWeather=dataframeWeather["Rating"].mean()

print(avrWeather)

dataframeVideoPlayers=data.loc[data["Category"]=="VIDEO_PLAYERS"]

avrVideoPlayers=dataframeVideoPlayers["Rating"].mean()

print(avrVideoPlayers)

dataframeNewsMagazines=data.loc[data["Category"]=="NEWS_AND_MAGAZINES"]

avrNewsMagazines=dataframeNewsMagazines["Rating"].mean()

print(avrNewsMagazines)

dataframeMapsNavigation=data.loc[data["Category"]=="MAPS_AND_NAVIGATION"]

avrMapsNavigation=dataframeMapsNavigation["Rating"].mean()

print(avrMapsNavigation)
data.dtypes
dataframeforType=data['Type']

indices=np.where(dataframeforType.isna())

print(indices)

data=data.drop(data.index[9148])

data.isnull().sum()
data.dtypes

data['Reviews']=data["Reviews"].convert_objects(convert_numeric=True)

data.dtypes
data=data[data["Reviews"]>50]

data.isnull().sum()
data.index = range(len(data))
booleanfornull=data["Rating"].isnull()

for counter in range(len(data)):

    if booleanfornull[counter]:

        if data["Category"].iloc[counter]=="ART_AND_DESIGN":

            data.Rating.fillna(avrArtDesign,inplace=True)

        elif data["Category"].iloc[counter]=="AUTO_AND_VEHICLES":

            data.Rating.fillna(avrAutoVehicles,inplace=True)

        elif data["Category"].iloc[counter]=="BEAUTY":

            data.Rating.fillna(avrBeauty,inplace=True)

        elif data["Category"].iloc[counter]=="BOOKS_AND_REFERENCE":

            data.Rating.fillna(avrBooksReference,inplace=True)

        elif data["Category"].iloc[counter]=="BUSINESS":

            data.Rating.fillna(avrBusiness,inplace=True)

        elif data["Category"].iloc[counter]=="COMICS":

            data.Rating.fillna(avrComics,inplace=True)

        elif data["Category"].iloc[counter]=="COMMUNICATION":

            data.Rating.fillna(avrCommunication,inplace=True)

        elif data["Category"].iloc[counter]=="DATING":

            data.Rating.fillna(avrDating,inplace=True)

        elif data["Category"].iloc[counter]=="EDUCATION":

            data.Rating.fillna(avrEducation,inplace=True)

        elif data["Category"].iloc[counter]=="ENTERTAINMENT":

            data.Rating.fillna(avrEntertainment,inplace=True)

        elif data["Category"].iloc[counter]=="EVENTS":

            data.Rating.fillna(avrEvents,inplace=True)

        elif data["Category"].iloc[counter]=="FINANCE":

            data.Rating.fillna(avrFinance,inplace=True)

        elif data["Category"].iloc[counter]=="FOOD_AND_DRINK":

            data.Rating.fillna(avrFoodDrink,inplace=True)

        elif data["Category"].iloc[counter]=="HEALTH_AND_FITNESS":

            data.Rating.fillna(avrHealthFitness,inplace=True)

        elif data["Category"].iloc[counter]=="HOUSE_AND_HOME":

            data.Rating.fillna(avrHouseHome,inplace=True)

        elif data["Category"].iloc[counter]=="LIBRARIES_AND_DEMO":

            data.Rating.fillna(avrLibrariesDemo,inplace=True)

        elif data["Category"].iloc[counter]=="LIFESTYLE":

            data.Rating.fillna(avrLifestyle,inplace=True)

        elif data["Category"].iloc[counter]=="GAME":

            data.Rating.fillna(avrGame,inplace=True)

        elif data["Category"].iloc[counter]=="FAMILY":

            data.Rating.fillna(avrFamily,inplace=True)

        elif data["Category"].iloc[counter]=="MEDICAL":

            data.Rating.fillna(avrMedical,inplace=True)

        elif data["Category"].iloc[counter]=="SOCIAL":

            data.Rating.fillna(avrSocial,inplace=True)

        elif data["Category"].iloc[counter]=="SHOPPING":

            data.Rating.fillna(avrShopping,inplace=True)

        elif data["Category"].iloc[counter]=="PHOTOGRAPHY":

            data.Rating.fillna(avrPhotography,inplace=True)

        elif data["Category"].iloc[counter]=="SPORTS":

            data.Rating.fillna(avrSports,inplace=True)

        elif data["Category"].iloc[counter]=="TRAVEL_AND_LOCAL":

            data.Rating.fillna(avrTravelLocal,inplace=True)

        elif data["Category"].iloc[counter]=="TOOLS":

            data.Rating.fillna(avrTools,inplace=True)

        elif data["Category"].iloc[counter]=="PERSONALIZATION":

            data.Rating.fillna(avrPersonalization,inplace=True)

        elif data["Category"].iloc[counter]=="PRODUCTIVITY":

            data.Rating.fillna(avrProductivity,inplace=True)

        elif data["Category"].iloc[counter]=="PARENTING":

            data.Rating.fillna(avrParenting,inplace=True)

        elif data["Category"].iloc[counter]=="WEATHER":

            data.Rating.fillna(avrWeather,inplace=True)

        elif data["Category"].iloc[counter]=="VIDEO_PLAYERS":

            data.Rating.fillna(avrVideoPlayers,inplace=True)

        elif data["Category"].iloc[counter]=="NEWS_AND_MAGAZINES":

            data.Rating.fillna(avrNewsMagazines,inplace=True)

        elif data["Category"].iloc[counter]=="MAPS_AND_NAVIGATION":

            data.Rating.fillna(avrMapsNavigation,inplace=True)

        
data.isnull().sum()
dataframeforC_Ver=data["Current Ver"]

indices = np.where(dataframeforC_Ver.isna())

print(indices)

dataframeforC_Ver=dataframeforC_Ver.fillna(dataframeforC_Ver.mode().iloc[0])



dataframeforAnd_Ver=data['Android Ver']

indices=np.where(dataframeforAnd_Ver.isna())

print(indices)

print(dataframeforAnd_Ver.values[4490])

dataframeforAnd_Ver=dataframeforAnd_Ver.fillna(dataframeforAnd_Ver.mode().iloc[0])

dataframeforAnd_Ver.isnull().sum()
processeddata=data.iloc[:,:11]

finaldataframe=pd.concat([processeddata,dataframeforC_Ver,dataframeforAnd_Ver],axis=1).reset_index()

finaldataframe.isnull().sum()

for t in range(len(finaldataframe)):

    if (finaldataframe["Rating"][t]>4.5) and (finaldataframe["Rating"][t]<=5):

        finaldataframe["Rating"][t]="Between 5 and 4.5"

    elif (finaldataframe["Rating"][t]>4) and (finaldataframe["Rating"][t]<=4.5):

        finaldataframe["Rating"][t]="Between 4.5 and 4"

    elif (finaldataframe["Rating"][t]>3.5) and (finaldataframe["Rating"][t]<=4):

        finaldataframe["Rating"][t]="Between 4 and 3.5"

    elif (finaldataframe["Rating"][t]>3) and (finaldataframe["Rating"][t]<=3.5):

        finaldataframe["Rating"][t]="Between 3.5 and 3"

    elif (finaldataframe["Rating"][t]>2.5) and (finaldataframe["Rating"][t]<=3):

        finaldataframe["Rating"][t]="Between 3 and 2.5"

    elif (finaldataframe["Rating"][t]>2) and (finaldataframe["Rating"][t]<=2.5):

        finaldataframe["Rating"][t]="Between 2.5 and 2"

    else:

        finaldataframe["Rating"][t]="Lower than 2"
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

le=LabelEncoder()

ohe=OneHotEncoder(categorical_features='all')
categoriesofApps=finaldataframe.iloc[:,2:3].values

categoriesofApps[:,0]=le.fit_transform(categoriesofApps[:,0])

categoriesofApps=ohe.fit_transform(categoriesofApps).toarray()
sizesofApps=finaldataframe.iloc[:,5:6].values

sizesofApps[:,0]=le.fit_transform(sizesofApps)
installsofApps=finaldataframe.iloc[:,6:7].values

installsofApps[:,0]=le.fit_transform(installsofApps[:,0])

np.unique(installsofApps)

installsofApps=ohe.fit_transform(installsofApps).toarray()
paysofApps=finaldataframe.iloc[:,7:9]

for counter in range(len(paysofApps)):

    if(paysofApps["Type"][counter]=="Free"):

        paysofApps["Price"][counter]="Free"

paymentsofApps=paysofApps.iloc[:,1:2].values

paymentsofApps[:,0]=le.fit_transform(paymentsofApps[:,0])
contentRatingsofApps=finaldataframe.iloc[:,9:10].values

contentRatingsofApps[:,0]=le.fit_transform(contentRatingsofApps[:,0])

np.unique(contentRatingsofApps)

contentRatingsofApps=ohe.fit_transform(contentRatingsofApps).toarray()
lastUpdatesofApps=finaldataframe.iloc[:,10:11].values

lastUpdatesofApps[:,0]=le.fit_transform(lastUpdatesofApps[:,0])
androidVersionsofApps=finaldataframe.iloc[:,12:13].values

androidVersionsofApps[:,0]=le.fit_transform(androidVersionsofApps[:,0])



currentVersionsofApps=finaldataframe.iloc[:,11:12].values

currentVersionsofApps[:,0]=le.fit_transform(currentVersionsofApps[:,0])



finaldataframe.dtypes





df_categories=pd.DataFrame(data=categoriesofApps,index=range(len(finaldataframe)))

df_sizes=pd.DataFrame(data=sizesofApps,index=range(len(finaldataframe)))

df_installs=pd.DataFrame(data=installsofApps,index=range(len(finaldataframe)))

df_payments=pd.DataFrame(data=paymentsofApps,index=range(len(finaldataframe)))

df_contentRatings=pd.DataFrame(data=contentRatingsofApps,index=range(len(finaldataframe)))

df_lastUpdates=pd.DataFrame(data=lastUpdatesofApps,index=range(len(finaldataframe)))

df_androidVersions=pd.DataFrame(data=androidVersionsofApps,index=range(len(finaldataframe)))

df_currentVersions=pd.DataFrame(data=currentVersionsofApps,index=range(len(finaldataframe)))

encoded_df=pd.concat([df_categories,df_sizes,df_installs,df_payments,df_contentRatings,df_lastUpdates,df_androidVersions,df_currentVersions],axis=1)
ratingsofApps=finaldataframe["Rating"]

labels=pd.DataFrame(data=ratingsofApps,index=range(len(finaldataframe)))





from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

encoded_df=sc.fit_transform(encoded_df)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(encoded_df,labels,test_size=0.33,random_state=42)

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')

rfc.fit(x_train,y_train)

predictions=rfc.predict(x_test)

cm=confusion_matrix(y_test,predictions)

print(cm)



print("Accuracy of this algorithm is: " ,accuracy_score(y_test, predictions, normalize=True, sample_weight=None))

y_numpy=finaldataframe.iloc[:,3].values





import collections

collections.Counter(y_numpy)
# smote



from imblearn.over_sampling import SMOTE

sm=SMOTE(random_state=2)

encoded_df, y_numpy=sm.fit_sample(encoded_df,y_numpy.ravel())
import collections

collections.Counter(y_numpy)

# Besides, there are too many features after encoding. It is hard to learn for machine

# PCA scales features



from sklearn.decomposition import PCA

pca = PCA(n_components=43)

pca.fit(encoded_df)



encoded_df = pca.transform(encoded_df)

print(pca.explained_variance_ratio_.cumsum())

#That will return a vector x such that x[i] returns

                                                

#the cumulative variance explained by the first i+1 dimensions.
#split train and test sets





#from collections import Counter

#from sklearn.datasets import make_classification



from sklearn.model_selection import train_test_split

smoted_x_train,smoted_x_test,smoted_y_train,smoted_y_test=train_test_split(encoded_df,y_numpy,test_size=0.3,random_state=0)



from sklearn.neighbors import KNeighborsClassifier



knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski')

knn.fit(smoted_x_train,smoted_y_train)

predictions=knn.predict(smoted_x_test)

cm=confusion_matrix(smoted_y_test,predictions)

print(cm)





print("Accuracy of this algorithm is: " ,accuracy_score(smoted_y_test, predictions, normalize=True, sample_weight=None))

from sklearn.svm import SVC 

svc=SVC(kernel='linear')

svc.fit(smoted_x_train,smoted_y_train)

predictions=svc.predict(smoted_x_test)

cm=confusion_matrix(smoted_y_test,predictions)

print(cm)



print("Accuracy of this algorithm is: " ,accuracy_score(smoted_y_test, predictions, normalize=True, sample_weight=None))



print("-----------------------------------------------")



svc=SVC(kernel='rbf')

svc.fit(smoted_x_train,smoted_y_train)

predictions=svc.predict(smoted_x_test)

cm=confusion_matrix(smoted_y_test,predictions)

print(cm)



print("Accuracy of this algorithm is: " ,accuracy_score(smoted_y_test, predictions, normalize=True, sample_weight=None))



print("-----------------------------------------------")





svc=SVC(kernel='poly')

svc.fit(smoted_x_train,smoted_y_train)

predictions=svc.predict(smoted_x_test)

cm=confusion_matrix(smoted_y_test,predictions)

print(cm)



print("Accuracy of this algorithm is: " ,accuracy_score(smoted_y_test, predictions, normalize=True, sample_weight=None))



print("-----------------------------------------------")



from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(criterion='entropy')

dtc.fit(smoted_x_train,smoted_y_train)

predictions=dtc.predict(smoted_x_test)

cm=confusion_matrix(smoted_y_test,predictions)

print(cm)



print("Accuracy of this algorithm is: " ,accuracy_score(smoted_y_test, predictions, normalize=True, sample_weight=None))

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')

rfc.fit(smoted_x_train,smoted_y_train)

predictions=rfc.predict(smoted_x_test)

cm=confusion_matrix(smoted_y_test,predictions)

print(cm)

print("Accuracy of this algorithm is: " ,accuracy_score(smoted_y_test, predictions, normalize=True, sample_weight=None))





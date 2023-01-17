import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")



#Reading Demographics CSV File
demographics=pd.read_csv('../input/county_facts.csv')
demographics = demographics[['fips','area_name','state_abbreviation','PST045214','AGE775214','RHI225214','RHI725214','RHI825214','EDU635213','EDU685213','INC110213','PVY020213','POP060210']]
demographics.rename(columns={'PST045214': 'Population', 'AGE775214': 'Age > 65','RHI225214':'Black','RHI725214':'Latino','RHI825214':'White','EDU635213':'HighSchool','EDU685213':'Bachelors','INC110213':'Median Household','PVY020213':'< Powerty level','POP060210':'Population PSM'}, inplace=True)

#Reading Results CSV File
results=pd.read_csv('../input/primary_results.csv')
results = results[results.party == "Democrat"]
results = results[(results.state != "Maine") & (results.state != "Massachusetts") & (results.state != "Vermont") & (results.state != "Illinois") ]
results = results[(results.candidate != ' Uncommitted') & (results.candidate != 'No Preference')]
results = results[(results.candidate == "Hillary Clinton") |(results.candidate == "Bernie Sanders") ]
Dem=results


#Calculating statewise total votes and fraction votes
votesByState = [[candidate, state, party] for candidate in Dem.candidate.unique() for state in Dem.state.unique() for party in Dem.party.unique()]
for i in votesByState:
	i.append(Dem[(Dem.candidate == i[0]) & (Dem.state == i[1])].votes.sum())
	i.append(i[3]*1.0/Dem[Dem.state == i[1]].votes.sum())
vbs = pd.DataFrame(votesByState, columns = ['candidate', 'state', 'party', 'votes','partyFrac'])
print(vbs)
#Merging demographics and results	
allData = pd.merge(vbs, demographics, how="inner", left_on = 'state',right_on = 'area_name')
allData.drop('state_abbreviation',axis=1, inplace=True)

#Segregate data candidate wise
HRC = allData[(allData.candidate == "Hillary Clinton")]
HRC=HRC.reset_index();
HRC.drop('index',axis=1, inplace=True)
print (HRC)

#Select X for Prediction
feature_cols = ['Population', 'Age > 65','Black','Latino','White','HighSchool','Bachelors','Median Household','< Powerty level','Population PSM']
X = HRC[feature_cols]

#Select y for Prediction
y = HRC.partyFrac

#Initializing Liner Regression and Random Forest
rf1 = RandomForestRegressor(n_estimators=1000)

#Train Model
rf1.fit(X,y)
#Select States with result no avaialable in primary_results.csv
demographics1=demographics[(demographics.area_name=='Pennsylvania')]
demographics2=demographics[(demographics.area_name=='Connecticut')]
demographics3=demographics[(demographics.area_name=='Maryland')]
demographics4=demographics[(demographics.area_name=='Delaware')]
demographics5=demographics[(demographics.area_name=='New York')]

#Test Model
X_test1=demographics1[feature_cols]
y_pred4=rf1.predict(X_test1)
xx=np.array(y_pred4);

X_test1=demographics2[feature_cols]
y_pred4=rf1.predict(X_test1)
xx=np.append(xx, y_pred4[0])


X_test1=demographics3[feature_cols]
y_pred4=rf1.predict(X_test1)
xx=np.append(xx,y_pred4[0])


X_test1=demographics4[feature_cols]
y_pred4=rf1.predict(X_test1)
xx=np.append(xx, y_pred4[0])

X_test1=demographics5[feature_cols]
y_pred4=rf1.predict(X_test1)
xx=np.append(xx, y_pred4[0])


#Predicted Results
a=pd.Series(xx, index=['Pennsylvania','Connecticut','Maryland','Delaware','New York'])
#Actual Results
x = np.array([.55,.51,.63,.60,.58])
b=pd.Series(x, index=['Pennsylvania','Connecticut','Maryland','Delaware','New York'])

#Calculate RMS Error
error3 = np.sqrt(mean_squared_error(a,b))

d = {'Predicted' : a,'Real' : b}
final=pd.DataFrame(d)
print (final)
print("Error=",end='')
print(error3)


print ("Future Pedictions - Hillary Clinton")

#Select Future States 
#Change name of state for other states
demographics6=demographics[(demographics.area_name=='California')]
demographics7=demographics[(demographics.area_name=='Indiana')]


X_test1=demographics6[feature_cols]
y_pred4=rf1.predict(X_test1)
print("California = ",end='')
print (y_pred4[0])

X_test1=demographics7[feature_cols]
y_pred4=rf1.predict(X_test1)
print("Indiana = ",end='')
print (y_pred4[0])



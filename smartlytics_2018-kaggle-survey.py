import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
genderdata = pd.read_csv('../input/multipleChoiceResponses.csv',usecols=['Q1'])

genderdata.drop(0)

genderdata=genderdata[genderdata.Q1 != 'What is your gender? - Selected Choice']

genderdata=pd.get_dummies(genderdata.Q1)

genders = list(genderdata)
plt.figure(figsize=(10,5),facecolor='whitesmoke')

for i in genders:

    gendersum = np.sum(genderdata[i].values)

    plt.bar(i,gendersum,label=str(round(gendersum*100/len(genderdata),1))+'% '+i)

plt.legend()

plt.ylabel('Number of respondents')

plt.title('Gender')

plt.show()
agedata = pd.read_csv('../input/multipleChoiceResponses.csv',usecols=['Q2'])

agedata.drop(0)

agedata = agedata[agedata.Q2 != 'What is your age (# years)?']

agedata=pd.get_dummies(agedata.Q2)

ages=list(agedata)
plt.figure(figsize=(10,5),facecolor='whitesmoke')

for i in ages:

    agesum = np.sum(agedata[i].values)

    plt.bar(i,agesum,label=str(round(agesum*100/len(agedata),1))+'% '+i)



plt.legend()

plt.ylabel('Number of respondents')

plt.title('Age')

plt.show()
locdata = pd.read_csv('../input/kagglesurveycontinent.csv')

locdata=pd.get_dummies(locdata.Continent)

locations=list(locdata)
plt.figure(figsize=(10,5),facecolor='whitesmoke')

for i in locations:

    locsum = np.sum(locdata[i].values)

    plt.barh(i,locsum,label=str(round(locsum*100/len(locdata),1))+'% '+i)



plt.legend()

plt.xlabel('Number of respondents')

plt.title('Continents')

plt.show()
cols = sum([list(range(29,42)),list(range(65,81)),list(range(88,105)),list(range(110,120))],[])

data = pd.read_csv('../input/multipleChoiceResponses.csv',usecols=cols)



string = ''

for i in range(0,len(list(data))):

    string = string + ' ' + ' '.join(data.iloc[:,i].dropna().values[1:])
from wordcloud import WordCloud, STOPWORDS

wordcloudsurvey = WordCloud(background_color='white',collocations = False,stopwords=STOPWORDS,width=1600, height=800).generate(string)

plt.figure(figsize=(20,10))

plt.imshow(wordcloudsurvey)

plt.axis('off')

plt.show()
# Libraries.

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import StandardScaler
# Data - Preprocessed Data. Salary and age are taken as the average of their respective ranges.

data = pd.read_csv('../input//kagglesurvey.csv')
# Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression



plt.figure(facecolor='whitesmoke',figsize=(10,5))

for gender in ['Male','Female']:

    #Dependent and Independent variables

    X = data[data.Gender==gender].Age.values.reshape(-1,1)

    y = data[data.Gender==gender].Salary.values.reshape(-1,1)

    

    #Polnomial Features

    X_p = PolynomialFeatures(degree=4)

    X = X_p.fit_transform(X)

    

    #Model fitting

    poly_model = LinearRegression()

    poly_model.fit(X,y)

    

    #Model Prediction

    x = np.arange(18,70,0.1).reshape(-1,1)

    y_pred = poly_model.predict(X_p.fit_transform(x))



    #Model Visualisation

    plt.plot(x,y_pred,label=gender)



plt.legend()

plt.grid()

plt.title('Kaggle Survey - Polynomial Regression')

plt.xlabel('Age (Years)')

plt.ylabel('Salary')

plt.xlim([x.min(),x.max()])

plt.ylim(bottom=0)

ticks = np.arange(0, 140000, step=20000)

plt.yticks(ticks,('$'+format(i,',') for i in ticks))

plt.show()
# Random Forest regression

from sklearn.ensemble import RandomForestRegressor



plt.figure(facecolor='whitesmoke',figsize=(10,5))

for gender in ['Male','Female']:

    #Dependent and Independent variables

    X = data[data.Gender==gender].Age.values.reshape(-1,1)

    y = data[data.Gender==gender].Salary.values



    #Model fitting

    forest_model = RandomForestRegressor(n_estimators=100)

    forest_model.fit(X,y)

    

    #Model Prediction

    x = np.arange(18,80,0.1).reshape(-1,1)

    y_pred = forest_model.predict(x)



    #Model Visualisation

    plt.plot(x,y_pred,label=gender)



plt.legend()

plt.grid()

plt.title('Kaggle Survey - Random Forest Regression')

plt.xlabel('Age (Years)')

plt.ylabel('Salary')

plt.xlim([18,70])

plt.ylim(bottom=0)

ticks = np.arange(0, 140000, step=20000)

plt.yticks(ticks,('$'+format(i,',') for i in ticks))

plt.show()
# Gaussian process regression

from sklearn.gaussian_process import GaussianProcessRegressor



plt.figure(facecolor='whitesmoke',figsize=(10,5))

for continent in ['Asia', 'North America', 'South America', 'Europe', 'Africa','Australia']:

    #Dependent and Independent Variables

    X = data[data.Continent==continent].Age.values.reshape(-1,1)

    y = data[data.Continent==continent].Salary.values.reshape(-1,1)

    

    #Feature Scaling

    from sklearn.preprocessing import StandardScaler

    sc_X = StandardScaler()

    X = sc_X.fit_transform(X)

    sc_y = StandardScaler()

    y = sc_y.fit_transform(y)



    #Model Fitting

    gp_model = GaussianProcessRegressor(alpha=0.2,normalize_y=True)

    gp_model.fit(X,y)

    

    #Model Prediction

    x = np.arange(18,80,0.1).reshape(-1,1)

    y_pred = sc_y.inverse_transform(gp_model.predict(sc_X.transform(x)))



    #Model Visualisation

    plt.plot(x,y_pred,label=continent)



plt.legend()

plt.grid()

plt.title('Kaggle Survey - Gaussian Process Regression')

plt.xlabel('Age (Years)')

plt.ylabel('Salary')

plt.xlim([18,70])

plt.ylim(bottom=0)

ticks = np.arange(0, 180000, step=20000)

plt.yticks(ticks,('$'+format(i,',') for i in ticks))

plt.show()
features = ['Continent','Degree']

for j in features:

    f = data[j].unique()

    salaries = [data[data[j]==i].Salary.values for i in f]

    fig, ax = plt.subplots(facecolor='whitesmoke',figsize=(10, 5))

    bp = ax.boxplot(salaries,patch_artist=True,labels=f,showfliers=False,vert=False)  # will be used to label x-ticks

    ax.set_title('Salary by '+j)

    for patch, color in zip(bp['boxes'], ['mediumturquoise' for i in range(0,len(f))]):

        patch.set_facecolor(color)

    ticks = np.arange(0, 250000, step=50000)

    plt.xticks(ticks,('$'+format(i,',') for i in ticks))

    plt.show()
SalaryOrdering = data.SalaryBands.unique()[np.argsort(data.Salary.unique())]

salsum = len(data)

salperc = np.round(data['Salary'].value_counts().sort_index().values*100/salsum,1)
plt.figure(figsize=(12,6),facecolor='whitesmoke')

for i in range(0,len(salperc)):

    plt.barh(SalaryOrdering[i],salperc[i],label = str(salperc[i])+'% $'+SalaryOrdering[i])

plt.legend()

plt.gca().invert_yaxis()

plt.title('Overall Salaries')

plt.gca().set_xticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_xticks()]) 

plt.ylabel('Salary Band ($)')

plt.xlabel('Percentage')

plt.show()
mean_python = np.mean(data.Python)
# Python usage by Salary. Salary is taken as the average of the salary ranges,

plt.figure(figsize=(8,10),facecolor='whitesmoke')

features = [['Salary',1,'($)'],['Age',2,'(Years)']]



for feature in features:

    x = np.sort(data[feature[0]].unique())

    y = np.array([np.mean(data[data[feature[0]] == f].Python) for f in x])



    plt.subplot(2,1,feature[1])

    plt.plot(x,y,color='black',linewidth=2.0)

    plt.plot(x,np.full(np.shape(x),mean_python),'--',color='orangered',label ='Mean usage: '+str(round(100*mean_python,1))+'%')

    plt.legend(loc=3)

    plt.fill_between(x,0,y,facecolor='lightsteelblue')

    plt.ylim([0,1])

    plt.xlim([x.min(),x.max()])

    plt.title('Python usage and '+feature[0])

    plt.xlabel(feature[0]+' '+feature[2])

    plt.ylabel('Used Python in the last 5 years (%)')

    

plt.show()
# Python usage by Location.

x2 = data.Continent.unique()

y2 = [np.mean(data[data.Continent == c].Python) for c in x2]



plt.figure(figsize=(10,5),facecolor='whitesmoke')

plt.bar(x2,y2,color='peachpuff',edgecolor='k',linewidth=1.3)

plt.ylim([0,1])

plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])

plt.ylabel('Used Python in the last 5 years (%)')

plt.xlabel('Continent')

plt.title('Python usage and Location')

plt.show()
#Python usage by Gender

x3 = data.Gender.unique()

y3 = [np.mean(data[data.Gender == g].Python) for g in x3]



plt.figure(facecolor='whitesmoke')

plt.barh(x3,y3,color='deepskyblue',edgecolor='k',linewidth=1.3)

plt.xlim([0,1])

plt.gca().set_xticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_xticks()]) 

plt.xlabel('Used Python in the last 5 years (%)')

plt.ylabel('Gender')

plt.title('Python usage and Gender')

plt.show()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

pd.set_option("display.max_rows", None)



import datetime

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

%matplotlib inline



from scipy.optimize import curve_fit

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error



import warnings

warnings.filterwarnings('ignore')



fileNameTrain = '/kaggle/input/covid19-global-forecasting-week-4/train.csv'

fileNameTest  = '/kaggle/input/covid19-global-forecasting-week-4/test.csv'

#fileNameTrain = 'train.csv'

#fileNameTest  = 'test.csv'



dfTrain = pd.read_csv(fileNameTrain)

dfTest  = pd.read_csv(fileNameTest)



dfTrain = dfTrain.fillna('NONE')

dfTest = dfTest.fillna('NONE')



dateInit = dfTrain['Date'][0]

dateInit = datetime.datetime.strptime(dateInit, "%Y-%m-%d").date()

dayCount = [(datetime.datetime.strptime(d, "%Y-%m-%d").date()-dateInit).days for d in dfTrain['Date']]

dfTrain['dayCount'] = dayCount

dayCount = [(datetime.datetime.strptime(d, "%Y-%m-%d").date()-dateInit).days for d in dfTest['Date']]

dfTest['dayCount'] = dayCount



dfCtyPrs = dfTrain[['Country_Region','Province_State']].drop_duplicates().reset_index()



xPoints  = np.linspace(0,114,115)
popDict={'Afghanistan_NONE':34656032,

'Albania_NONE':2876101,

'Algeria_NONE':40606052,

'Andorra_NONE':77281,

'Angola_NONE':28813463,

'Antigua and Barbuda_NONE':100963,

'Argentina_NONE':43847430,

'Armenia_NONE':2924816,

'Australia_Australian Capital Territory':24127159,

'Australia_New South Wales':24127159,

'Australia_Northern Territory':24127159,

'Australia_Queensland':24127159,

'Australia_South Australia':24127159,

'Australia_Tasmania':24127159,

'Australia_Victoria':24127159,

'Australia_Western Australia':24127159,

'Austria_NONE':8747358,

'Azerbaijan_NONE':9762274,

'Bahamas_NONE':391232,

'Bahrain_NONE':1425171,

'Bangladesh_NONE':162951560,

'Barbados_NONE':284996,

'Belarus_NONE':9507120,

'Belgium_NONE':11348159,

'Belize_NONE':201674,

'Benin_NONE':10872298,

'Bhutan_NONE':797765,

'Bolivia_NONE':10887882,

'Bosnia and Herzegovina_NONE':3516816,

'Botswana_NONE':2250000,

'Brazil_NONE':207652865,

'Brunei_NONE':423196,

'Bulgaria_NONE':7127822,

'Burkina Faso_NONE':18646433,

'Burma_NONE':54000000,

'Burundi_NONE':11200000,

'Cabo Verde_NONE':539560,

'Cambodia_NONE':15762370,

'Cameroon_NONE':23439189,

'Canada_Alberta':4413146,

'Canada_British Columbia':5110917,

'Canada_Manitoba':1377517,

'Canada_New Brunswick':779993,

'Canada_Newfoundland and Labrador':521365,

'Canada_Northwest Territories':45000,

'Canada_Nova Scotia':977457,

'Canada_Ontario':14711827,

'Canada_Prince Edward Island':158158,

'Canada_Quebec':8537674,

'Canada_Saskatchewan':1181666,

'Canada_Yukon':40000,

'Central African Republic_NONE':4594621,

'Chad_NONE':14452543,

'Chile_NONE':17909754,

'China_Anhui':62200728,

'China_Beijing':25215113,

'China_Chongqing':31791850,

'China_Fujian':39789666,

'China_Gansu':26406335,

'China_Guangdong':13302000,

'China_Guangxi':50027447,

'China_Guizhou':35669112,

'China_Hainan':9645206,

'China_Hebei':76878832,

'China_Heilongjiang':38434821,

'China_Henan':94375586,

'China_Hong Kong':7478895,

'China_Hubei':59783978,

'China_Hunan':69804905,

'China_Inner Mongolia':25617169,

'China_Jiangsu':81134214,

'China_Jiangxi':46784338,

'China_Jilin':27653275,

'China_Liaoning':44260682,

'China_Macau':649335,

'China_Ningxia':7139502,

'China_Qinghai':6146631,

'China_Shaanxi':38379669,

'China_Shandong':101027107,

'China_Shanghai':27014899,

'China_Shanxi':37702752,

'China_Sichuan':82613096,

'China_Tianjin':19898557,

'China_Tibet':3414708,

'China_Xinjiang':24697132,

'China_Yunnan':49033220,

'China_Zhejiang':56292546,

'Colombia_NONE':48653419,

'Congo (Brazzaville)_NONE':5125821,

'Congo (Kinshasa)_NONE':78736153,

'Costa Rica_NONE':4857274,

"Cote d'Ivoire_NONE":23695919,

'Croatia_NONE':4170600,

'Cuba_NONE':11475982,

'Cyprus_NONE':1170125,

'Czechia_NONE':10561633,

'Denmark_Faroe Islands':49117,

'Denmark_Greenland':56186,

'Denmark_NONE':5731118,

'Diamond Princess_NONE':4000,

'Djibouti_NONE':942333,

'Dominica_NONE':73543,

'Dominican Republic_NONE':10648791,

'Ecuador_NONE':16385068,

'Egypt_NONE':95688681,

'El Salvador_NONE':6344722,

'Equatorial Guinea_NONE':1221490,

'Eritrea_NONE':3452786,

'Estonia_NONE':1316481,

'Eswatini_NONE':1136281,

'Ethiopia_NONE':102403196,

'Fiji_NONE':898760,

'Finland_NONE':5495096,

'France_French Guiana':275713,

'France_French Polynesia':202016,

'France_Guadeloupe':390704,

'France_Martinique':371246,

'France_Mayotte':259154,

'France_New Caledonia':278000,

'France_Reunion':865826,

'France_Saint Barthelemy':9625,

'France_Saint Pierre and Miquelon':6000,

'France_St Martin':31949,

'France_NONE':66896109,

'Gabon_NONE':1979786,

'Gambia_NONE':2038501,

'Georgia_NONE':3719300,

'Germany_NONE':82667685,

'Ghana_NONE':28206728,

'Greece_NONE':10746740,

'Grenada_NONE':107317,

'Guatemala_NONE':16582469,

'Guinea_NONE':12395924,

'Guinea-Bissau_NONE':1815698,

'Guyana_NONE':773303,

'Haiti_NONE':10847334,

'Holy See_NONE':800,

'Honduras_NONE':9112867,

'Hungary_NONE':9817958,

'Iceland_NONE':334252,

'India_NONE':1324171354,

'Indonesia_NONE':261115456,

'Iran_NONE':80277428,

'Iraq_NONE':37202572,

'Ireland_NONE':4773095,

'Israel_NONE':8547100,

'Italy_NONE':60600590,

'Jamaica_NONE':2881355,

'Japan_NONE':126994511,

'Jordan_NONE':9455802,

'Kazakhstan_NONE':17797032,

'Kenya_NONE':48461567,

'Korea, South_NONE':51245707,

'Kosovo_NONE':1800000,

'Kuwait_NONE':4052584,

'Kyrgyzstan_NONE':6082700,

'Laos_NONE':6758353,

'Latvia_NONE':1960424,

'Lebanon_NONE':6006668,

'Liberia_NONE':4613823,

'Libya_NONE':6293253,

'Liechtenstein_NONE':37666,

'Lithuania_NONE':2872298,

'Luxembourg_NONE':582972,

'Madagascar_NONE':24894551,

'Malaysia_NONE':31187265,

'Maldives_NONE':417492,

'Malawi_NONE':18400000,

'Mali_NONE':17994837,

'Malta_NONE':436947,

'Mauritania_NONE':4301018,

'Mauritius_NONE':1263473,

'Mexico_NONE':127540423,

'Moldova_NONE':3552000,

'Monaco_NONE':38499,

'Mongolia_NONE':3027398,

'Montenegro_NONE':622781,

'Morocco_NONE':35276786,

'Mozambique_NONE':28829476,

'MS Zaandam_NONE':4000,

'Namibia_NONE':2479713,

'Nepal_NONE':28982771,

'Netherlands_Aruba':17018408,

'Netherlands_Bonaire, Sint Eustatius and Saba':25000,

'Netherlands_Curacao':17018408,

'Netherlands_Sint Maarten':17018408,

'Netherlands_NONE':17018408,

'New Zealand_NONE':4692700,

'Nicaragua_NONE':6149928,

'Niger_NONE':20672987,

'Nigeria_NONE':185989640,

'North Macedonia_NONE':2081206,

'Norway_NONE':5232929,

'Oman_NONE':4424762,

'Pakistan_NONE':193203476,

'Panama_NONE':4034119,

'Papua New Guinea_NONE':8084991,

'Paraguay_NONE':6725308,

'Peru_NONE':31773839,

'Philippines_NONE':103320222,

'Poland_NONE':37948016,

'Portugal_NONE':10324611,

'Qatar_NONE':2569804,

'Romania_NONE':19705301,

'Russia_NONE':143201676,

'Rwanda_NONE':11917508,

'Saint Kitts and Nevis_NONE':54821,

'Saint Lucia_NONE':178015,

'Saint Vincent and the Grenadines_NONE':109643,

'San Marino_NONE':33203,

'Sao Tome and Principe_NONE':211000,

'Saudi Arabia_NONE':32275687,

'Senegal_NONE':15411614,

'Serbia_NONE':7057412,

'Seychelles_NONE':94677,

'Sierra Leone_NONE':7000000,

'Singapore_NONE':5607283,

'Slovakia_NONE':5428704,

'Slovenia_NONE':2064845,

'Somalia_NONE':14317996,

'South Africa_NONE':55908865,

'South Sudan_NONE':11000000,

'Spain_NONE':46443959,

'Sri Lanka_NONE':21203000,

'Sudan_NONE':39578828,

'Suriname_NONE':558368,

'Sweden_NONE':9903122,

'Switzerland_NONE':8372098,

'Syria_NONE':18430453,

'Taiwan*_NONE':23780452,

'Tanzania_NONE':55572201,

'Thailand_NONE':68863514,

'Timor-Leste_NONE':1268671,

'Togo_NONE':7606374,

'Trinidad and Tobago_NONE':1364962,

'Tunisia_NONE':11403248,

'Turkey_NONE':79512426,

'US_Alabama':4888949,

'US_Alaska':738068,

'US_Arizona':7123898,

'US_Arkansas':3020327,

'US_California':39776830,

'US_Colorado':5684203,

'US_Connecticut':3588683,

'US_Delaware':971180,

'US_District of Columbia':703608,

'US_Florida':21312211,

'US_Georgia':10545138,

'US_Guam':162896,

'US_Hawaii':1426393,

'US_Idaho':1753860,

'US_Illinois':12768320,

'US_Indiana':6699629,

'US_Iowa':3160553,

'US_Kansas':2918515,

'US_Kentucky':4472265,

'US_Louisiana':4682509,

'US_Maine':1341582,

'US_Maryland':6079602,

'US_Massachusetts':6895917,

'US_Michigan':9991177,

'US_Minnesota':5628162,

'US_Mississippi':2982785,

'US_Missouri':6135888,

'US_Montana':1062330,

'US_Nebraska':1932549,

'US_Nevada':3056824,

'US_New Hampshire':1350575,

'US_New Jersey':9032872,

'US_New Mexico':2090708,

'US_New York':19862512,

'US_North Carolina':10390149,

'US_North Dakota':755238,

'US_Ohio':11694664,

'US_Oklahoma':3940521,

'US_Oregon':4199563,

'US_Pennsylvania':12823989,

'US_Puerto Rico':3411307,

'US_Rhode Island':1061712,

'US_South Carolina':5088916,

'US_South Dakota':877790,

'US_Tennessee':6782564,

'US_Texas':28704330,

'US_Utah':3159345,

'US_Vermont':623960,

'US_Virgin Islands':102951,

'US_Virginia':8525660,

'US_Washington':7530552,

'US_West Virginia':1803077,

'US_Wisconsin':5818049,

'US_Wyoming':573720,

'Uganda_NONE':41487965,

'Ukraine_NONE':45004645,

'United Arab Emirates_NONE':9269612,

'United Kingdom_Anguilla':15000,

'United Kingdom_Bermuda':65331,

'United Kingdom_British Virgin Islands':32000,

'United Kingdom_Cayman Islands':60765,

'United Kingdom_Channel Islands':164541,

'United Kingdom_Falkland Islands (Malvinas)':3500,

'United Kingdom_Gibraltar':34408,

'United Kingdom_Isle of Man':83737,

'United Kingdom_Montserrat':5000,

'United Kingdom_NONE':65637239,

'United Kingdom_Turks and Caicos Islands':32000,

'Uruguay_NONE':3201607,

'Uzbekistan_NONE':31848200,

'Venezuela_NONE':31568179,

'Vietnam_NONE':92701100,

'West Bank and Gaza_NONE':5000000,

'Western Sahara_NONE':560000,

'Zambia_NONE':16591390,

'Zimbabwe_NONE':16150362}
def func0(x, a, b):    

    return a*x+b



def func1(x, a,b,c):

    return a * np.exp(b * (x-c))



def func2(x, a2, b2, c2):

    return a2*1/(1+np.exp(-b2*(x-c2)))



def getDataForLocation(df,k):

    cty = dfCtyPrs.iloc[k]['Country_Region']

    prs = dfCtyPrs.iloc[k]['Province_State']

    pop = popDict[cty+'_'+prs]



    dfSelectTrain = dfTrain[(dfTrain['Country_Region'] == cty) & (dfTrain['Province_State'] == prs)]

    dfSelectTrain = dfSelectTrain[['dayCount','ConfirmedCases','Fatalities']]



    dfSelectTest = dfTest[(dfTrain['Country_Region'] == cty) & (dfTrain['Province_State'] == prs)]

    dfSelectTest = dfSelectTest[['dayCount']]



    initPoint = 0

    selectX  = dfSelectTrain['dayCount'][initPoint:]

    selectY1 = dfSelectTrain['ConfirmedCases'][initPoint:]

    selectY2 = dfSelectTrain['Fatalities'][initPoint:]

    #xDataFitCC, yDataFitCC, yDataFitF = fitData(selectX,selectY1,selectY2)

    xDataFitCC, yDataFitCC, yDataFitF = None, None, None

    return cty, prs, pop, selectX, selectY1, selectY2, xDataFitCC, yDataFitCC, yDataFitF
def getFit(k):

    cty, prs, pop, selectX, selectY1, selectY2, xDataFit, yDataFitCC, yDataFitF = getDataForLocation(dfTrain,k)



    selectY   = selectY1

    bestRmse1 = np.inf

    bestPopt1 = None

    bestFit1  = None

    numberOfLeadingZeros = np.where(selectY<np.max(selectY+1)/100)[0].shape[0]+1

    for skipStart1 in range(numberOfLeadingZeros):

        if cty == 'China':

            popt, pcov = curve_fit(func2,selectX[skipStart1:],selectY[skipStart1:],p0=[selectY.to_numpy()[-1],0.3,0],bounds=[[selectY.to_numpy()[-1]-1,0,0],[selectY.to_numpy()[-1]+1,0.4,100]])   #,p0=poptBest

            yDataFit01 = func2(xPoints, popt[0], popt[1], popt[2])

        elif cty == 'Diamond Princess': 

            popt, pcov = curve_fit(func2,selectX[skipStart1:],selectY[skipStart1:],p0=[selectY.to_numpy()[-1],0.4,25],bounds=[[selectY.to_numpy()[-1]-1,0.3,20],[selectY.to_numpy()[-1]+1,0.5,30]])   #,p0=poptBest

            yDataFit01 = func2(xPoints, popt[0], popt[1], popt[2])

        else:

            popt, pcov = curve_fit(func2,selectX[skipStart1:],selectY[skipStart1:],p0=[0.01*pop,0.3,50],bounds=[[0.01*pop-1,0,0],[0.01*pop+1,0.4,100]])   #,p0=poptBest

            yDataFit01 = func2(xPoints, popt[0], popt[1], popt[2])



        rmse = np.sqrt(mean_squared_error(yDataFit01[:len(selectY)],selectY))

        if rmse < bestRmse1:

            bestPopt1 = popt

            bestRmse1 = rmse

            bestFit1  = yDataFit01



    selectY  = selectY2

    bestRmse2 = np.inf

    bestPopt2 = None

    bestFit2  = None

    numberOfLeadingZeros = np.where(selectY<np.max(selectY+1)/100)[0].shape[0]

    for skipStart2 in range(numberOfLeadingZeros):

        mortality = 0.05

        if cty == 'China':

            popt, pcov = curve_fit(func2,selectX[skipStart2:],selectY[skipStart2:],p0=[selectY.to_numpy()[-1],0.3,10],bounds=[[selectY.to_numpy()[-1]-1,0,0],[selectY.to_numpy()[-1]+1,0.4,100]])   #,p0=poptBest

            yDataFit02 = func2(xPoints, popt[0], popt[1], popt[2])

        elif cty == 'Diamond Princess':

            popt, pcov = curve_fit(func2,selectX[skipStart2:],selectY[skipStart2:],p0=[selectY.to_numpy()[-1],0.1,45],bounds=[[selectY.to_numpy()[-1]-1,0,40],[selectY.to_numpy()[-1]+1,0.4,50]])   #,p0=poptBest

            yDataFit02 = func2(xPoints, popt[0], popt[1], popt[2])

        else:

            popt, pcov = curve_fit(func2,selectX[skipStart2:],selectY[skipStart2:],p0=[mortality*bestFit1[-1],0.3,50],bounds=[[mortality*bestFit1[-1]-1,0,0],[mortality*bestFit1[-1]+1,0.4,100]])   #,p0=poptBest

            yDataFit02 = func2(xPoints, popt[0], popt[1], popt[2])



        rmse = np.sqrt(mean_squared_error(yDataFit02[:len(selectY)],selectY))

        if rmse < bestRmse2:

            bestPopt2 = popt

            bestRmse2 = rmse

            bestFit2  = yDataFit02

            

    print(k, cty, prs, bestPopt1,bestRmse1)

    print(k, cty, prs, bestPopt2,bestRmse2)

    if 1==0:

        fig, ax = plt.subplots(1,4, figsize=(20,3))

        ax[0].plot(selectX,selectY1,marker='+',linewidth=0,color='red')

        ax[0].plot(xPoints,yDataFit01,linewidth=0.5,color='red')

        ax[0].text(0,0,str(k)+' '+cty+'/'+prs, fontsize=12)

        ax[0].grid(True)

        ax[0].set_ylim(0,2*np.max(selectY1));



        ax[1].plot(selectX,selectY1,marker='+',linewidth=0,color='red')

        ax[1].plot(xPoints,yDataFit01,linewidth=0.5,color='red')

        ax[1].text(0,0,pop, fontsize=12)

        ax[1].grid(True)



        ax[2].plot(selectX,selectY2,marker='+',linewidth=0,color='blue')

        ax[2].plot(xPoints,yDataFit02,linewidth=0.5,color='blue')

        ax[2].grid(True)

        ax[2].set_ylim(0,2*np.max(selectY2));



        ax[3].plot(selectX,selectY2,marker='+',linewidth=0,color='blue')

        ax[3].plot(xPoints,yDataFit02,linewidth=0.5,color='blue')

        ax[3].grid(True)

    return cty, prs, pop, selectX, selectY1, selectY2, xPoints, yDataFit01, yDataFit02
outCC=np.array([])

outF=np.array([])

dataStorage = []



n=dfCtyPrs.shape[0]

for k in range(n):

    cty, prs, pop, selectX, selectY1, selectY2, xDataFit, yDataFitCC, yDataFitF = getFit(k)



    outCC = np.concatenate((outCC,np.around(selectY1[72:82]).astype(int)))

    outCC = np.concatenate((outCC,np.around(yDataFitCC[82:115]).astype(int)))

    

    outF  = np.concatenate((outF, np.around(selectY2[72:82]).astype(int)))

    outF  = np.concatenate((outF, np.around(yDataFitF[82:115]).astype(int)))



    #outCC = np.concatenate((outCC,np.around(yDataFitCC[57:]).astype(int)))

    #outF  = np.concatenate((outF, np.around(yDataFitF[57:]).astype(int)))



    dataStorage.append([selectX,selectY1,selectY2,xDataFit,yDataFitCC,yDataFitF])



dfOut = pd.DataFrame()

#dfOut['dayCount'] = dfTest['dayCount']

dfOut['ForecastId'] = dfTest['ForecastId']

dfOut['ConfirmedCases']= outCC

dfOut['Fatalities']= outF

dfOut.to_csv('submission.csv', index=False)
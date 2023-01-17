# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import xgboost as xgb

from sklearn.metrics import mean_squared_log_error as MSLE



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1")



#We convert the salaries to numerical values and keep salaries between 1000 and 1.000.000 Local currency

data['CompensationAmount'] = data['CompensationAmount'].fillna(0)

data['CompensationAmount'] = data.CompensationAmount.apply(lambda x: 0 if (pd.isnull(x) or (x=='-') or (x==0))

                                                       else float(x.replace(',','')))

df = data[(data['CompensationAmount']>1000) & (data['CompensationAmount']<2000000)]





#We only keep the countries with more than 80 respondents to get significant results later on

s_temp = df['Country'].value_counts()

s_temp = s_temp[s_temp>80]

countries=list(s_temp.index)

countries.remove('Other')

df=df[df.Country.isin(countries)]

df['CompensationCurrency'] =df.groupby('Country')['CompensationCurrency'].apply(lambda x: x.fillna(x.value_counts().idxmax()))
#The PPP rates

rates_ppp={'Countries':['United States','India','United Kingdom','Germany','France','Brazil','Canada','Spain','Australia','Russia','Italy',"People 's Republic of China",'Netherlands'],

           'Currency':['USD','INR','GBP','EUR','EUR','BRL','CAD','EUR','AUD','RUB','EUR','CNY','EUR'],

           'PPP':[1.00,17.7,0.7,0.78,0.81,2.05,1.21,0.66,1.46,25.13,0.74,3.51,0.8]}



rates_ppp = pd.DataFrame(data=rates_ppp)

rates_ppp

#we load the exchange rates that were given by Kaggle a

rates_mer=pd.read_csv('../input/conversionRates.csv', encoding="ISO-8859-1")

rates_mer.drop('Unnamed: 0',inplace=True,axis=1)



rates=rates_ppp.merge(rates_mer,left_on='Currency',right_on='originCountry',how='left')

rates['PPP/MER']=rates['PPP']*rates['exchangeRate']



#keep the PPP/MER rates plus the 'Countries' column that will be used for the merge

rates=rates[['Countries','PPP','PPP/MER']]

rates
df=df.merge(rates_mer,left_on='CompensationCurrency',right_on='originCountry',how='left')

df=df.merge(rates,left_on='Country',right_on='Countries',how='left')



df['AdjustedSalary']=df['CompensationAmount']*df['exchangeRate']/df['PPP/MER']



d_salary = {}

for country in df['Country'].value_counts().index :

    d_salary[country]=df[df['Country']==country]['AdjustedSalary'].median()

    

median_wages = pd.DataFrame.from_dict(data=d_salary, orient='index').round(2)

median_wages.sort_values(by=list(median_wages),axis=0, ascending=True, inplace=True)

ax = median_wages.plot(kind='barh',figsize=(15,8),width=0.7,align='center')

ax.legend_.remove()

ax.set_title("Adjusted incomes over the world",fontsize=16)

ax.set_xlabel("Amount", fontsize=14)

ax.set_ylabel("Country", fontsize=14)

for tick in ax.get_xticklabels():

    tick.set_rotation(0)

    tick.set_fontsize(10)

plt.tight_layout()



plt.show();
inflations={'Countries':['United States','India','United Kingdom','Germany','France','Brazil','Canada','Spain','Australia','Russia','Italy',"People 's Republic of China",'Netherlands'],

           'CPI_2013':[106.83,131.98,110.15,105.68,105.01,119.37,105.45,107.21,107.70,121.64,107.20,111.16,107.48],

           'CPI_2017':[113.10,162.01,116.51,109.6,107.1,156.73,112.39,109.13,113.48,168.50,108.61,119.75,111.55],

           'medians_2013':[15480,615,12399,14098,12445,2247,15181,7284,15026,4129,6874,1786,14450]}



rates_inflations = pd.DataFrame(inflations)

rates_inflations['adjusted_medians']=(rates_inflations['medians_2013']*rates_inflations['CPI_2017']/rates_inflations['CPI_2013']).round(2)

rates_inflations
tmp=median_wages.reset_index()

tmp = tmp.rename(columns={'index': 'Country', 0: 'median_income'})



rates_inflations=rates_inflations.merge(tmp,left_on='Countries',right_on='Country',how='left')

rates_inflations['ratio_incomes']=(rates_inflations['median_income']/rates_inflations['adjusted_medians']).round(2)



tmp2=rates_inflations[['Country','ratio_incomes']]

tmp2.sort_values(by='ratio_incomes',axis=0, ascending=True, inplace=True)
tmp2.plot.barh(x='Country',figsize=(12,8))

plt.show();
datasets = {'USA' : df[df['Country']=='United States'] , 

            'Eur+Ca' :df[df.Country.isin(['Australia','Germany','Canada','United Kingdom','Netherlands'])],

            'Eur2+Bra+Chi' : df[df.Country.isin(['Spain','France','Brazil',"People 's Republic of China",'Italy'])],

            'India/Russia' : df[df.Country.isin(['India','Russia'])]}
methods=['WorkMethodsFrequencyBayesian','WorkMethodsFrequencyNaiveBayes','WorkMethodsFrequencyLogisticRegression',

       'WorkMethodsFrequencyDecisionTrees','WorkMethodsFrequencyRandomForests',

       'WorkMethodsFrequencyEnsembleMethods','WorkMethodsFrequencyDataVisualization','WorkMethodsFrequencyPCA',

       'WorkMethodsFrequencyNLP','WorkMethodsFrequencyNeuralNetworks',

       'WorkMethodsFrequencyTextAnalysis',

       'WorkMethodsFrequencyRecommenderSystems','WorkMethodsFrequencyKNN','WorkMethodsFrequencySVMs',

       'WorkMethodsFrequencyTimeSeriesAnalysis']





d_method_countries={} 

for key, value in datasets.items():

    d_method_countries[key]={}

    for col in methods : 

        method = col.split('WorkMethodsFrequency')[1]

        d_method_countries[key][method]=value[value[col].isin(['Most of the time','Often'])]['AdjustedSalary'].median()

        

positions=[(0,0),(1,0),(0,1),(1,1)]

f,ax=plt.subplots(nrows=2, ncols=2,figsize=(15,8))

for ((key, value), pos) in zip(d_method_countries.items() , positions):

    methods = pd.DataFrame.from_dict(data=value, orient='index').round(2)

    methods.sort_values(by=list(methods),axis=0, ascending=True, inplace=True)

    methods.plot(kind='barh',figsize=(12,8),width=0.7,align='center',ax=ax[pos[0],pos[1]])

    ax[pos[0],pos[1]].set_title(key,fontsize=14)

    ax[pos[0],pos[1]].legend_.remove()

    



plt.tight_layout()

plt.show();

    
tools=['WorkToolsFrequencyC','WorkToolsFrequencyJava','WorkToolsFrequencyMATLAB',

       'WorkToolsFrequencyPython','WorkToolsFrequencyR','WorkToolsFrequencyTensorFlow',

       'WorkToolsFrequencyHadoop','WorkToolsFrequencySpark','WorkToolsFrequencySQL',

       'WorkToolsFrequencyNoSQL','WorkToolsFrequencyExcel','WorkToolsFrequencyTableau',

       'WorkToolsFrequencyJupyter','WorkToolsFrequencyAWS',

       'WorkToolsFrequencySASBase','WorkToolsFrequencyUnix']



d_tools_countries={} 

for key, value in datasets.items():

    d_tools_countries[key]={}

    for col in tools : 

        tool = col.split('WorkToolsFrequency')[1]

        d_tools_countries[key][tool]=value[value[col].isin(['Most of the time','Often'])]['AdjustedSalary'].median()

        

positions=[(0,0),(1,0),(0,1),(1,1)]

f,ax=plt.subplots(nrows=2, ncols=2,figsize=(15,8))

for ((key, value), pos) in zip(d_tools_countries.items() , positions):

    tools = pd.DataFrame.from_dict(data=value, orient='index').round(2)

    tools.sort_values(by=list(methods),axis=0, ascending=True, inplace=True)

    tools.plot(kind='barh',figsize=(12,8),width=0.7,align='center',ax=ax[pos[0],pos[1]])

    ax[pos[0],pos[1]].set_title(key,fontsize=14)

    ax[pos[0],pos[1]].legend_.remove()

    



plt.tight_layout()

plt.show();

        
titles=list(df['CurrentJobTitleSelect'].value_counts().index)

d_titles_countries={} 

for key, value in datasets.items():

    d_titles_countries[key]={}

    for title in titles : 

        d_titles_countries[key][title]=value[value['CurrentJobTitleSelect']==title]['AdjustedSalary'].median()

        

positions=[(0,0),(1,0),(0,1),(1,1)]

f,ax=plt.subplots(nrows=2, ncols=2,figsize=(15,8))

for ((key, value), pos) in zip(d_titles_countries.items() , positions):

    tools = pd.DataFrame.from_dict(data=value, orient='index').round(2)

    tools.sort_values(by=list(methods),axis=0, ascending=True, inplace=True)

    tools.plot(kind='barh',figsize=(12,8),width=0.7,align='center',ax=ax[pos[0],pos[1]])

    ax[pos[0],pos[1]].set_title(key,fontsize=14)

    ax[pos[0],pos[1]].legend_.remove()

    



plt.tight_layout()

plt.show();
func = list(df['JobFunctionSelect'].value_counts().index)

tmp = df

tmp=tmp.replace(to_replace=func, value=['Analyze data','Build a ML service','Build prototypes',

                                        'Build the Data Infrastructure','Other','Research'])



datasets_tmp = {'USA' : tmp[tmp['Country']=='United States'] , 

            'Eur+Ca' :tmp[tmp.Country.isin(['Australia','Germany','Canada','United Kingdom','Netherlands'])],

            'Eur2+Bra+Chi' : tmp[tmp.Country.isin(['Spain','France','Brazil',"People 's Republic of China",'Italy'])],

            'India/Russia' : tmp[tmp.Country.isin(['India','Russia'])]}



functions=list(tmp['JobFunctionSelect'].value_counts().index)

d_functions_countries={} 

for key, value in datasets_tmp.items():

    d_functions_countries[key]={}

    for function in functions : 

        d_functions_countries[key][function]=value[value['JobFunctionSelect']==function]['AdjustedSalary'].median()

        

positions=[(0,0),(1,0),(0,1),(1,1)]

f,ax=plt.subplots(nrows=2, ncols=2,figsize=(15,8))

for ((key, value), pos) in zip(d_functions_countries.items() , positions):

    tools = pd.DataFrame.from_dict(data=value, orient='index').round(2)

    tools.sort_values(by=list(methods),axis=0, ascending=True, inplace=True)

    tools.plot(kind='barh',figsize=(15,8),width=0.7,align='center',ax=ax[pos[0],pos[1]])

    ax[pos[0],pos[1]].set_title(key,fontsize=14)

    ax[pos[0],pos[1]].legend_.remove()

    

plt.tight_layout()

plt.show();
df['MATLABUsers']=[1 if df['WorkToolsFrequencyMATLAB'].iloc[i] in ['Most of the time','Often'] else 0 for i in range(df.shape[0])]

df['AWSUsers']=[1 if df['WorkToolsFrequencyAWS'].iloc[i] in ['Most of the time','Often'] else 0 for i in range(df.shape[0])]

df['HadoopUsers']=[1 if df['WorkToolsFrequencyHadoop'].iloc[i] in ['Most of the time','Often'] else 0 for i in range(df.shape[0])]

df['SparkUsers']=[1 if df['WorkToolsFrequencySpark'].iloc[i] in ['Most of the time','Often'] else 0 for i in range(df.shape[0])]



df['NaiveBayesUsers']=[1 if df['WorkMethodsFrequencyNaiveBayes'].iloc[i] in ['Most of the time','Often'] else 0 for i in range(df.shape[0])]

df['RecommenderSystemsUsers']=[1 if df['WorkMethodsFrequencyRecommenderSystems'].iloc[i] in ['Most of the time','Often'] else 0 for i in range(df.shape[0])]

df['DataVisualizationUsers']=[1 if df['WorkMethodsFrequencyDataVisualization'].iloc[i] in ['Most of the time','Often'] else 0 for i in range(df.shape[0])]



features= ['GenderSelect','Country','Age','FormalEducation','MajorSelect','ParentsEducation',

           'EmploymentStatus','StudentStatus','DataScienceIdentitySelect','CodeWriter',

           'CurrentEmployerType','SalaryChange','RemoteWork','WorkMLTeamSeatSelect',

           'Tenure','EmployerIndustry','EmployerSize','CurrentJobTitleSelect','JobFunctionSelect',

           'MATLABUsers','AWSUsers','HadoopUsers','SparkUsers',

           'NaiveBayesUsers','RecommenderSystemsUsers','DataVisualizationUsers',

           'TimeGatheringData','TimeModelBuilding','TimeProduction','TimeVisualizing','TimeFindingInsights',

           'AdjustedSalary']





df_us = df[df['Country']=='United States'][features]

df_eur = df[df.Country.isin(['Spain','France','Germany','Canada','United Kingdom','Netherlands','Italy','Australia','Canada'])][features]

df_bric = df[df.Country.isin(['India','Russia','Brazil',"People 's Republic of China"])][features]
for (dataset,zone) in zip([df_us,df_bric,df_eur],['USA','BRIC','Europe + Canada and Australia']) : 

    

    dataset=pd.get_dummies(dataset,columns=['GenderSelect','Country','FormalEducation','MajorSelect','ParentsEducation',

           'EmploymentStatus','StudentStatus','DataScienceIdentitySelect','CodeWriter',

           'CurrentEmployerType','SalaryChange','RemoteWork','WorkMLTeamSeatSelect',

           'Tenure','EmployerIndustry','EmployerSize','CurrentJobTitleSelect','JobFunctionSelect'])

    for col in ['Age','TimeGatheringData','TimeModelBuilding','TimeProduction','TimeVisualizing','TimeFindingInsights']:

        dataset[col] = dataset[col].fillna(value=dataset[col].median())

    dataset.dropna(axis=0,inplace=True)



    np.random.seed(42)

    perm = np.random.permutation(dataset.shape[0])

    train = dataset.iloc[perm[0:round(0.85*len(perm))]]

    test = dataset.iloc[perm[round(0.85*len(perm))::]]

    y_train , y_test = train['AdjustedSalary'] , test['AdjustedSalary']

    X_train , X_test = train.drop('AdjustedSalary',axis=1) , test.drop('AdjustedSalary',axis=1)



    clf=xgb.XGBRegressor(learning_rate=0.05, n_estimators=500, objective='reg:linear',reg_lambda=0.5, 

                         random_state=42)

    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)

    

    print('Prediction for : %s'%zone)

    print('The RMSLE score is : {:0.4f}'.format(np.sqrt(MSLE(y_test,y_pred)) /np.sqrt(len(y_pred)) ))

    print('-------------------------------------------------')
f,ax=plt.subplots(nrows=1, ncols=3,figsize=(15,8))

df_bric['AdjustedSalary'].plot.hist(bins=50,ax=ax[0],figsize=(15,8),title='Salary distribution in BRIC countries')

df_eur['AdjustedSalary'].plot.hist(bins=50,ax=ax[1],figsize=(15,8),title='Salary distribution in Europe + Ca+ Aus')

df_us['AdjustedSalary'].plot.hist(bins=50,ax=ax[2],figsize=(15,8),title='Salary distribution in the the US')

plt.show();
f,ax=plt.subplots(nrows=1, ncols=3,figsize=(15,8))

sns.boxplot(y=df_bric['AdjustedSalary'],data=df_bric,ax=ax[0]).set_title('Quartiles and outliers in BRIC')

sns.boxplot(y=df_eur['AdjustedSalary'],data=df_eur,ax=ax[1]).set_title('Quartiles and outliers in EUR')

sns.boxplot(y=df_us['AdjustedSalary'],data=df_us,ax=ax[2]).set_title('Quartiles and outliers in USA')

plt.show();
# Importing the necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Importing the Indicators dataset
indicators = pd.read_csv('../input/Indicators.csv')

# Countries of the European Union
eu_countries = ['European Union', 'Austria','Belgium', 'Bulgaria', 'Croatia', 'Cyprus',
                'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece',
                'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta',
                'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain',
                'Sweden', 'United Kingdom']

# Defining the dataset containg only the countries from European Union
eu = indicators[indicators['CountryName'].isin(eu_countries)]

print(eu['CountryName'].unique())
eu.head()

# Separating the indicators and putting them into a list
indicadores = eu[['IndicatorName','IndicatorCode']].drop_duplicates().values
indicadores
# Reorganizing the indicators to better access Reorganizando os indicadores para melhor acessá-los e transformando em um DataFrame
new_indicators =[]
indicators_code =[]

for ind in indicadores:
    indicador = ind[0]
    code = ind[1].strip()
    if code not in indicators_code:
        # Deleting the caracters ,() from the indicators and converting all the caracters to lower case.
        modified_indicator = re.sub('[,()]',"",indicador).lower()
        # Changing - for "to"
        modified_indicator = re.sub('-'," to ",modified_indicator).lower()
        new_indicators.append([modified_indicator,code])
        indicators_code.append(code)
                
new_indicadores = pd.DataFrame(new_indicators, columns=['IndicatorName','IndicatorCode']).drop_duplicates()
print('We have %s features in this dataset!' % new_indicadores.shape[0])
indncode = []
for element in new_indicators:
    if ('fertility' in element[0]) or ('GDP' in element[1]):
        indncode.append(element)
indncode
# Creating two filters for the two indicators
chosen_indicators = ['SP.DYN.TFRT.IN', 'NY.GDP.PCAP.CD']
# Creating the dataframe to work with
dftowork = eu[eu['IndicatorCode'].isin(chosen_indicators)]
print(dftowork.shape)
dftowork.head()
# Importing the plotly library, which allow us to plot iterative graphics
import plotly 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

init_notebook_mode(connected=True)
# List that contain the countries names and their codes
countries = eu[['CountryName','CountryCode']].drop_duplicates().values

# Function that creates a dataset, "df_stage", containing a filter per indicator and all the countries that have 
# some data from a minimum date and a commum maximum date. 

def stage_prep(indicator):
    
    # creating a dictionary of dataframes
    dfs = {'df_'+str(countries[i][1]): dftowork[(dftowork['CountryCode'] == countries[i][1]) &
                                       (dftowork['IndicatorCode'] == indicator)] for i in range(len(countries))}
    
    min_list = [dfs['df_'+str(countries[i][1])].Year.min() for i in range(len(countries))]
    max_among_all_min_years = max(min_list)
            
    max_list = [dfs['df_'+str(countries[i][1])].Year.max() for i in range(len(countries))]
    min_among_all_max_years = min(max_list)
        
    if((len(set(min_list)) == 1) & (len(set(max_list)) == 1)):
        
        df_stage = dftowork[dftowork['IndicatorCode'] == indicator]
        
        return df_stage
            
    else:
        
        year_and_indicator_filter = ((dftowork['Year'] >= max_among_all_min_years) & 
                                             (dftowork['Year'] <= min_among_all_max_years) &
                                             (dftowork['IndicatorCode'] == indicator))
        df_stage = dftowork[year_and_indicator_filter] 
        return df_stage
# Crating a function to plot the graphics 
def plot_line(df_stages):
    
    # Creating the figure
    figure ={
    'data':[],
    'layout':{}
    }
    # Dictionary of datasets
    df_c = {i: df_stages[df_stages['CountryCode'] == countries[i][1]] for i in range(len(countries))}
    
    # Creating a graphic for each country
    for i in range(len(countries)):
        traces = {i: go.Scatter({
            'x': list(df_c[i]['Year']),
            'y': list(df_c[i]['Value']),
            'connectgaps': True,
            'text': list(df_c[i]['Value']),
            'name': countries[i][0]
        }) }
        
        figure['data'].append(traces[i])
        title = df_stages['IndicatorName'].iloc[0]
        
        figure['layout']['title'] = title
        figure['layout']['xaxis'] = {'title': 'Years'}
        figure['layout']['yaxis'] = {'title': 'Value'}
        figure['layout']['hovermode'] = 'compare'
    
    iplot(figure, validate=False)
plot_line(stage_prep(chosen_indicators[0]))
onlyeu = dftowork[(dftowork['IndicatorCode'] == chosen_indicators[0]) & (dftowork['CountryCode'] == 'EUU')]
plot_line(onlyeu)
plot_line(stage_prep(chosen_indicators[1]))
nofilter = dftowork[(dftowork['IndicatorCode'] == chosen_indicators[1])]
plot_line(nofilter)
justeu = dftowork[(dftowork['IndicatorCode'] == chosen_indicators[1]) & (dftowork['CountryCode'] == 'EUU')]
plot_line(justeu)
# Codes
gdp = dftowork[(dftowork['IndicatorCode'] == chosen_indicators[1]) & (dftowork['CountryCode'] == 'EUU')]['Value']
fert = dftowork[(dftowork['IndicatorCode'] == chosen_indicators[0]) & (dftowork['CountryCode'] == 'EUU')]['Value']
corre = pd.DataFrame()
corre['GDP'] = gdp.reset_index(drop=True)
corre['Fertility'] = fert.reset_index(drop=True)
cor = corre.corr()
cor
sns.heatmap(cor, annot=True)
key_indicators = ['NE.CON.PETC.KD', 'NY.GDP.MKTP.KD']
data = eu[eu['IndicatorCode'].isin(key_indicators)]
data.head()
hhdata = data[data['IndicatorCode'] == key_indicators[0]].reset_index(drop=True)
gdpdata = data[data['IndicatorCode'] == key_indicators[1]].reset_index(drop=True)
gdpdata.shape, hhdata.shape
# Correcting the dates so that the variables have the same size
def correctyears(countrycode):
    
    # creating a list of years for the country comparing with the variables
    hhyears = list(set(hhdata[(hhdata['CountryCode'] == countrycode) &
                                       (hhdata['IndicatorCode'] == key_indicators[0])]['Year']))
    gdpyears = list(set(gdpdata[(gdpdata['CountryCode'] == countrycode) &
                                       (gdpdata['IndicatorCode'] == key_indicators[1])]['Year']))
    
    years = list(set(hhyears) & set(gdpyears))
    
    return years
   
# Defining a dictionary to colect the two variables for each country of the Eropean Union
variables = {}

for i in range(len(countries)):
    years = correctyears(countries[i][1])
    
    variables['C_'+str(countries[i][1])] = [hhdata[(hhdata['CountryCode'] == countries[i][1]) & 
                                                   (hhdata['Year'] >= min(years)) & 
                                             (hhdata['Year'] <= max(years))]['Value'].reset_index(drop=True)]
    variables['Y_'+str(countries[i][1])] = [gdpdata[(gdpdata['CountryCode'] == countries[i][1]) & 
                                                   (gdpdata['Year'] >= min(years)) & 
                                             (gdpdata['Year'] <= max(years))]['Value'].reset_index(drop=True)]
# A graphich analysis of the linear beahaviour between the variables with indicator "EUU"
plt.scatter(variables['C_EUU'], variables['Y_EUU'])
plt.show
# Now, for each country we implement a linear regression using statsmodels.api

import statsmodels.api as sm

def OLSmodel(C,Y):

    y = sm.add_constant(Y)
    model = sm.OLS(C,y)
    results = model.fit()
    
    return results

# Putting the results in a dictionary
results = {'model_'+str(countries[i][1]): OLSmodel(variables['C_'+str(countries[i][1])], 
                                                   variables['Y_'+str(countries[i][1])]) for i in range(len(countries))}
# As an example, we present the parameters from Belgium
results['model_BEL'].params
# Dataset to work with
codigos = ['SP.DYN.AMRT.FE', 'SH.MMR.DTHS', 'SP.DYN.TFRT.IN', 'NY.GDP.PCAP.CD', 'SP.DYN.IMRT.IN']
lastdata = eu[(eu['IndicatorCode'].isin(codigos)) & (eu['CountryCode']== 'EUU')].reset_index(drop=True)
lastdata.head()
# Creating the dataset
dicfeat = {}
for indicator in codigos:
    dicfeat[indicator] = lastdata[lastdata['IndicatorCode'] == indicator]['Value'].reset_index(drop=True)
lastdf= pd.DataFrame(dicfeat)
lastdf.head()
# Renaming the columns
columns = ['GDP_per_capita', 'Maternal_deaths', 'Mortality_rate_female', 'Mortality_rate_infant', 'fertility_rate']
lastdf.columns = columns
lastdf.head(n=100)
# Droping the last two lines of the dataframe and dealing with NaN's

lastdf = lastdf.drop([lastdf.index[54], lastdf.index[55]])
lastdf.fillna(lastdf.mean(), inplace=True)
# Analysing the correlation between the variables
corr_mean = lastdf.corr()
plt.figure(figsize=(14, 14))
sns.heatmap(corr_mean, cbar=True, square=True, annot=True, fmt='.2f', cmap='PiYG')
# Fertility vs Mortality_rate_infant
plt.figure(figsize=(14, 14))
plt.scatter(lastdf['fertility_rate'], lastdf['Mortality_rate_infant'], color='green')
plt.title('Fertility Vs Mortality Infant', fontsize=14)
plt.xlabel('Fertility Rate', fontsize=14)
plt.ylabel('Mortality Infant Rate', fontsize=14)
plt.grid(True)
plt.show()
# Fertility vs Maternal Deaths
plt.figure(figsize=(14, 14))
plt.scatter(lastdf['fertility_rate'], lastdf['Maternal_deaths'], color='green')
plt.title('Fertility Vs Maternal_deaths', fontsize=14)
plt.xlabel('Fertility Rate', fontsize=14)
plt.ylabel('MAternal Deaths', fontsize=14)
plt.grid(True)
plt.show()
# Fertility vs Mortality_rate_female
plt.figure(figsize=(14, 14))
plt.scatter(lastdf['fertility_rate'], lastdf['Mortality_rate_female'], color='green')
plt.title('Fertility Vs Mortality_rate_female', fontsize=14)
plt.xlabel('Fertility Rate', fontsize=14)
plt.ylabel('Mortality_Rate_Female', fontsize=14)
plt.grid(True)
plt.show()
# Fertility vs GDP_per_capita
plt.figure(figsize=(14, 14))
plt.scatter(lastdf['fertility_rate'], lastdf['GDP_per_capita'], color='green')
plt.title('Fertility Vs GDP_per_capita', fontsize=14)
plt.xlabel('Fertility Rate', fontsize=14)
plt.ylabel('GDP_per_capita', fontsize=14)
plt.grid(True)
plt.show()
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics 

# Isolating the target variable
X = lastdf.drop('fertility_rate', axis=1)
y = lastdf.fertility_rate
# Cross Validation and the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
# R^2 score
print('Variance score: {}'.format(lm.score(X_test,y_test)))

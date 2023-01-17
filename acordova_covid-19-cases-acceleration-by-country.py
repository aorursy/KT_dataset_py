import pandas as pd

import plotly.express as px

import plotly.graph_objects as go



chartTemplate = 'plotly_dark'



# get up to date data

cases = pd.read_csv('https://data.humdata.org/hxlproxy/data/download/time_series_covid19_confirmed_global_narrow.csv?dest=data_edit&filter01=explode&explode-header-att01=date&explode-value-att01=value&filter02=rename&rename-oldtag02=%23affected%2Bdate&rename-newtag02=%23date&rename-header02=Date&filter03=rename&rename-oldtag03=%23affected%2Bvalue&rename-newtag03=%23affected%2Binfected%2Bvalue%2Bnum&rename-header03=Value&filter04=clean&clean-date-tags04=%23date&filter05=sort&sort-tags05=%23date&sort-reverse05=on&filter06=sort&sort-tags06=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv')



# massage the data

cases = cases.drop(0)

cases['Datetime'] = pd.to_datetime(cases['Date'], infer_datetime_format=True)

cases['Cases'] = cases['Value'].astype(int)



countryCases = cases.groupby(['Country/Region','Datetime']).agg('sum')

countryCases.reset_index(inplace=True)



locationByTime = countryCases.pivot(index='Datetime', columns='Country/Region', values='Cases')

locationByTime.reset_index(inplace=True)



# calculate case velocity

newCaseCounts = locationByTime.loc[:, locationByTime.columns != 'Datetime'].diff()

dateTimes = locationByTime['Datetime']

dateTimes.drop(dateTimes.index[0])

caseVelocity = pd.concat([dateTimes, newCaseCounts], axis=1)



# # calculate case acceleration

caseVelocityChanges = caseVelocity.loc[:, caseVelocity.columns != 'Datetime'].diff()

velocityDateTimes = caseVelocity['Datetime']

velocityDateTimes.drop(velocityDateTimes.index[0])

caseAcceleration = pd.concat([velocityDateTimes, caseVelocityChanges], axis=1)



# calculate means

accelMeans5 = caseAcceleration.loc[:, caseAcceleration.columns != 'Datetime'].ewm(span=5).mean()

accelDateTimes = caseAcceleration['Datetime']

caseAccelEWM5 = pd.concat([accelDateTimes, accelMeans5], axis=1)



accelMeans10 = caseAcceleration.loc[:, caseAcceleration.columns != 'Datetime'].ewm(span=10).mean()

caseAccelEWM10 = pd.concat([accelDateTimes, accelMeans10], axis=1)



# Note - this isn't currently used in charts

accelWin5 = caseAcceleration.loc[:, caseAcceleration.columns != 'Datetime'].rolling(5).mean()

caseAccelWin5 = pd.concat([accelDateTimes, accelWin5], axis=1)



# generate charts



# main map

import pycountry



countries = {}

for country in pycountry.countries:

    countries[country.name] = country.alpha_3

    

countryAccel = caseAccelEWM10.iloc[-1].to_dict()



# fix a few known bad names

def rename(a, b):

    countryAccel[b] = countryAccel[a]

    del countryAccel[a]



rename('US', 'United States')

rename('Korea, South', 'South Korea')

rename('Taiwan*', 'Taiwan')



accelMap = []

for country,cases in countryAccel.items():

    if country == 'Datetime':

        continue

    code = countries.get(country, 'UNK')

    if country == 'Iran':

        code = 'IRN'

    if country == 'Russia':

        code = 'RUS'

    accelMap.append([code, country, cases])





accelMapDf = pd.DataFrame(accelMap, columns=['iso-alpha', 'country', 'caseAccel'])



mapFig = px.choropleth(accelMapDf, locations="iso-alpha",

                    color="caseAccel", # lifeExp is a column of gapminder

                    hover_name="country", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma,

                      template=chartTemplate,

                      title="World Case Acceleration (exponential weighted 10 day mean)")



# country charts



def generateCharts(country):

    casesTitle = country + ' Cases'

    casesFig = px.line(locationByTime, x="Datetime", y=country, title=casesTitle, template=chartTemplate)

    

    velocityTitle = country + ' Case Velocity (new cases per day)'

    caseVelocityFig = px.line(caseVelocity, x="Datetime", y=country, title=velocityTitle, template=chartTemplate)



    accelTitle = country + " Case Acceleration (Change in new cases per day a.k.a change in velocity)"

    caseAccelerationFig = go.Figure(layout=go.Layout(title=go.layout.Title(text=accelTitle)))



    caseAccelerationFig.add_trace(go.Scatter(x=caseAcceleration['Datetime'], y=caseAcceleration[country], name='Daily change'))

    caseAccelerationFig.add_trace(go.Scatter(x=caseAccelEWM5['Datetime'], y=caseAccelEWM5[country], name='5 day average'))

    caseAccelerationFig.add_trace(go.Scatter(x=caseAccelEWM10['Datetime'], y=caseAccelEWM10[country], name='10 day average'))



    caseAccelerationFig.update_layout(template=chartTemplate)

    

    return (casesFig, caseVelocityFig, caseAccelerationFig)



ukCasesFig, ukCaseVelocityFig, ukCaseAccelerationFig = generateCharts('United Kingdom')



usCasesFig, usCaseVelocityFig, usCaseAccelerationFig = generateCharts('US')



cnCasesFig, cnCaseVelocityFig, cnCaseAccelerationFig = generateCharts('China')



itCasesFig, itCaseVelocityFig, itCaseAccelerationFig = generateCharts('Italy')



spCasesFig, spCaseVelocityFig, spCaseAccelerationFig = generateCharts('Spain')



irnCasesFig, irnCaseVelocityFig, irnCaseAccelerationFig = generateCharts('Iran')



frCasesFig, frCaseVelocityFig, frCaseAccelerationFig = generateCharts('France')
mapFig.show()
caseAccelEWM10.tail(1)[['US','United Kingdom', 'China', 'Italy', 'Spain', 'Iran', 'France', 'Ukraine']]
accelCountries = []

accelTail = caseAccelEWM10.tail(1)

for o in accelTail:

    accelCountries.append((accelTail[o], o))



accelCountries[1]
usCasesFig.show()
usCaseVelocityFig.show()
usCaseAccelerationFig.show()
ukCasesFig.show()
ukCaseVelocityFig.show()
ukCaseAccelerationFig.show()
cnCasesFig.show()
cnCaseVelocityFig.show()
cnCaseAccelerationFig.show()
itCasesFig.show()
itCaseVelocityFig.show()
itCaseAccelerationFig.show()
spCasesFig.show()
spCaseVelocityFig.show()
spCaseAccelerationFig.show()
irnCasesFig.show()
irnCaseVelocityFig.show()
irnCaseAccelerationFig.show()
frCasesFig.show()
frCaseVelocityFig.show()
frCaseAccelerationFig.show()
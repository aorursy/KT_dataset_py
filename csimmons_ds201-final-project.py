import seaborn as sns

import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

from bokeh.plotting import figure, output_file, show

from bokeh.io import output_notebook

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn import tree

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from operator import itemgetter

import math
df = pd.read_csv('https://raw.githubusercontent.com/christian-simmons/DS201-Final/master/data-full.csv')

df.shape
# I did not do this by hand, I just made an Excel function to do it:

# (=A1&"_data = df[df[" & CHAR(34) & "name" & CHAR(34)  & "]==" & CHAR(34) & A1 & CHAR(34) &"]")

# With A1 being the country name in cell A1



Albania_data = df[df["name"]=="Albania"]

Algeria_data = df[df["name"]=="Algeria"]

Argentina_data = df[df["name"]=="Argentina"]

Armenia_data = df[df["name"]=="Armenia"]

Australia_data = df[df["name"]=="Australia"]

Austria_data = df[df["name"]=="Austria"]

Azerbaijan_data = df[df["name"]=="Azerbaijan"]

Bahrain_data = df[df["name"]=="Bahrain"]

Bangladesh__data = df[df["name"]=="Bangladesh_"]

Barbados_data = df[df["name"]=="Barbados"]

Belarus_data = df[df["name"]=="Belarus"]

Belgium_data = df[df["name"]=="Belgium"]

Belize_data = df[df["name"]=="Belize"]

Benin_data = df[df["name"]=="Benin"]

Bolivia_data = df[df["name"]=="Bolivia"]

Bosnia_and_Herzegovina_data = df[df["name"]=="Bosnia_and_Herzegovina"]

Botswana_data = df[df["name"]=="Botswana"]

Brazil_data = df[df["name"]=="Brazil"]

Brunei_Darussalam_data = df[df["name"]=="Brunei_Darussalam"]

Bulgaria_data = df[df["name"]=="Bulgaria"]

Burkina_Faso_data = df[df["name"]=="Burkina_Faso"]

Burma_data = df[df["name"]=="Burma"]

Cabo_Verde_data = df[df["name"]=="Cabo_Verde"]

Cambodia_data = df[df["name"]=="Cambodia"]

Cameroon_data = df[df["name"]=="Cameroon"]

Canada_data = df[df["name"]=="Canada"]

Chad_data = df[df["name"]=="Chad"]

Chile_data = df[df["name"]=="Chile"]

China_data = df[df["name"]=="China"]

Colombia_data = df[df["name"]=="Colombia"]

Costa_Rica__data = df[df["name"]=="Costa_Rica_"]

Cote_d_Ivoire__data = df[df["name"]=="Cote_d'Ivoire_"]

Croatia_data = df[df["name"]=="Croatia"]

Cuba_data = df[df["name"]=="Cuba"]

Cyprus_data = df[df["name"]=="Cyprus"]

Czech_Republic_data = df[df["name"]=="Czech_Republic"]

Denmark_data = df[df["name"]=="Denmark"]

Djibouti_data = df[df["name"]=="Djibouti"]

Dominican_Republic_data = df[df["name"]=="Dominican_Republic"]

Ecuador_data = df[df["name"]=="Ecuador"]

Egypt_data = df[df["name"]=="Egypt"]

El_Salvador__data = df[df["name"]=="El_Salvador_"]

Equatorial_Guinea_data = df[df["name"]=="Equatorial_Guinea"]

Estonia_data = df[df["name"]=="Estonia"]

Eswatini_data = df[df["name"]=="Eswatini"]

Ethiopia_data = df[df["name"]=="Ethiopia"]

Fiji_data = df[df["name"]=="Fiji"]

Finland_data = df[df["name"]=="Finland"]

France_data = df[df["name"]=="France"]

Gabon_data = df[df["name"]=="Gabon"]

Georgia_data = df[df["name"]=="Georgia"]

Germany_data = df[df["name"]=="Germany"]

Ghana_data = df[df["name"]=="Ghana"]

Greece_data = df[df["name"]=="Greece"]

Guatemala__data = df[df["name"]=="Guatemala_"]

Guinea_data = df[df["name"]=="Guinea"]

Guyana_data = df[df["name"]=="Guyana"]

Haiti_data = df[df["name"]=="Haiti"]

Honduras__data = df[df["name"]=="Honduras_"]

Hong_Kong_data = df[df["name"]=="Hong_Kong"]

Hungary__data = df[df["name"]=="Hungary_"]

Iceland_data = df[df["name"]=="Iceland"]

India_data = df[df["name"]=="India"]

Indonesia_data = df[df["name"]=="Indonesia"]

Iran_data = df[df["name"]=="Iran"]

Ireland_data = df[df["name"]=="Ireland"]

Israel_data = df[df["name"]=="Israel"]

Italy_data = df[df["name"]=="Italy"]

Jamaica__data = df[df["name"]=="Jamaica_"]

Japan_data = df[df["name"]=="Japan"]

Jordan_data = df[df["name"]=="Jordan"]

Kazakhstan_data = df[df["name"]=="Kazakhstan"]

Kenya_data = df[df["name"]=="Kenya"]

Kuwait_data = df[df["name"]=="Kuwait"]

Kyrgyz_Republic__data = df[df["name"]=="Kyrgyz_Republic_"]

Laos_data = df[df["name"]=="Laos"]

Latvia_data = df[df["name"]=="Latvia"]

Lebanon_data = df[df["name"]=="Lebanon"]

Lesotho_data = df[df["name"]=="Lesotho"]

Lithuania_data = df[df["name"]=="Lithuania"]

Luxembourg_data = df[df["name"]=="Luxembourg"]

Madagascar_data = df[df["name"]=="Madagascar"]

Malawi_data = df[df["name"]=="Malawi"]

Malaysia__data = df[df["name"]=="Malaysia_"]

Mali_data = df[df["name"]=="Mali"]

Malta_data = df[df["name"]=="Malta"]

Mauritania_data = df[df["name"]=="Mauritania"]

Mexico_data = df[df["name"]=="Mexico"]

Moldova_data = df[df["name"]=="Moldova"]

Mongolia_data = df[df["name"]=="Mongolia"]

Morocco_data = df[df["name"]=="Morocco"]

Mozambique__data = df[df["name"]=="Mozambique_"]

Namibia_data = df[df["name"]=="Namibia"]

Nepal_data = df[df["name"]=="Nepal"]

Netherlands_data = df[df["name"]=="Netherlands"]

New_Zealand_data = df[df["name"]=="New_Zealand"]

Nicaragua__data = df[df["name"]=="Nicaragua_"]

Niger_data = df[df["name"]=="Niger"]

Nigeria_data = df[df["name"]=="Nigeria"]

North_Korea_data = df[df["name"]=="North_Korea"]

Norway_data = df[df["name"]=="Norway"]

Oman_data = df[df["name"]=="Oman"]

Pakistan__data = df[df["name"]=="Pakistan_"]

Panama__data = df[df["name"]=="Panama_"]

Paraguay__data = df[df["name"]=="Paraguay_"]

Peru_data = df[df["name"]=="Peru"]

Philippines_data = df[df["name"]=="Philippines"]

Poland_data = df[df["name"]=="Poland"]

Portugal_data = df[df["name"]=="Portugal"]

Republic_of_Congo__data = df[df["name"]=="Republic_of_Congo_"]

Romania_data = df[df["name"]=="Romania"]

Russia_data = df[df["name"]=="Russia"]

Rwanda_data = df[df["name"]=="Rwanda"]

Saudi_Arabia_data = df[df["name"]=="Saudi_Arabia"]

Senegal_data = df[df["name"]=="Senegal"]

Sierra_Leone_data = df[df["name"]=="Sierra_Leone"]

Singapore_data = df[df["name"]=="Singapore"]

Slovakia_data = df[df["name"]=="Slovakia"]

Slovenia_data = df[df["name"]=="Slovenia"]

South_Africa_data = df[df["name"]=="South_Africa"]

South_Korea_data = df[df["name"]=="South_Korea"]

Spain_data = df[df["name"]=="Spain"]

Sri_Lanka_data = df[df["name"]=="Sri_Lanka"]

Suriname_data = df[df["name"]=="Suriname"]

Sweden_data = df[df["name"]=="Sweden"]

Switzerland_data = df[df["name"]=="Switzerland"]

Taiwan_data = df[df["name"]=="Taiwan"]

Tajikistan_data = df[df["name"]=="Tajikistan"]

Tanzania_data = df[df["name"]=="Tanzania"]

Thailand__data = df[df["name"]=="Thailand_"]

The_Bahamas_data = df[df["name"]=="The_Bahamas"]

The_Gambia_data = df[df["name"]=="The_Gambia"]

Trinidad_and_Tobago_data = df[df["name"]=="Trinidad_and_Tobago"]

Tunisia_data = df[df["name"]=="Tunisia"]

Turkey_data = df[df["name"]=="Turkey"]

Turkmenistan_data = df[df["name"]=="Turkmenistan"]

Uganda_data = df[df["name"]=="Uganda"]

Ukraine_data = df[df["name"]=="Ukraine"]

United_Arab_Emirates_data = df[df["name"]=="United_Arab_Emirates"]

United_Kingdom_data = df[df["name"]=="United_Kingdom"]

United_States_data = df[df["name"]=="United_States"]

Uruguay__data = df[df["name"]=="Uruguay_"]

Uzbekistan_data = df[df["name"]=="Uzbekistan"]

Venezuela__data = df[df["name"]=="Venezuela_"]

Vietnam_data = df[df["name"]=="Vietnam"]

Zambia_data = df[df["name"]=="Zambia"]

Zimbabwe_data = df[df["name"]=="Zimbabwe"]
country_data = [

   Albania_data,

Algeria_data,

Argentina_data,

Armenia_data,

Australia_data,

Austria_data,

Azerbaijan_data,

Bahrain_data,

Bangladesh__data,

Barbados_data,

Belarus_data,

Belgium_data,

Belize_data,

Benin_data,

Bolivia_data,

Bosnia_and_Herzegovina_data,

Botswana_data,

Brazil_data,

Brunei_Darussalam_data,

Bulgaria_data,

Burkina_Faso_data,

Burma_data,

Cabo_Verde_data,

Cambodia_data,

Cameroon_data,

Canada_data,

Chad_data,

Chile_data,

China_data,

Colombia_data,

Costa_Rica__data,

Cote_d_Ivoire__data,

Croatia_data,

Cuba_data,

Cyprus_data,

Czech_Republic_data,

Denmark_data,

Djibouti_data,

Dominican_Republic_data,

Ecuador_data,

Egypt_data,

El_Salvador__data,

Equatorial_Guinea_data,

Estonia_data,

Eswatini_data,

Ethiopia_data,

Fiji_data,

Finland_data,

France_data,

Gabon_data,

Georgia_data,

Germany_data,

Ghana_data,

Greece_data,

Guatemala__data,

Guinea_data,

Guyana_data,

Haiti_data,

Honduras__data,

Hong_Kong_data,

Hungary__data,

Iceland_data,

India_data,

Indonesia_data,

Iran_data,

Ireland_data,

Israel_data,

Italy_data,

Jamaica__data,

Japan_data,

Jordan_data,

Kazakhstan_data,

Kenya_data,

Kuwait_data,

Kyrgyz_Republic__data,

Laos_data,

Latvia_data,

Lebanon_data,

Lesotho_data,

Lithuania_data,

Luxembourg_data,

Madagascar_data,

Malawi_data,

Malaysia__data,

Mali_data,

Malta_data,

Mauritania_data,

Mexico_data,

Moldova_data,

Mongolia_data,

Morocco_data,

Mozambique__data,

Namibia_data,

Nepal_data,

Netherlands_data,

New_Zealand_data,

Nicaragua__data,

Niger_data,

Nigeria_data,

North_Korea_data,

Norway_data,

Oman_data,

Pakistan__data,

Panama__data,

Paraguay__data,

Peru_data,

Philippines_data,

Poland_data,

Portugal_data,

Republic_of_Congo__data,

Romania_data,

Russia_data,

Rwanda_data,

Saudi_Arabia_data,

Senegal_data,

Sierra_Leone_data,

Singapore_data,

Slovakia_data,

Slovenia_data,

South_Africa_data,

South_Korea_data,

Spain_data,

Sri_Lanka_data,

Suriname_data,

Sweden_data,

Switzerland_data,

Taiwan_data,

Tajikistan_data,

Tanzania_data,

Thailand__data,

The_Bahamas_data,

The_Gambia_data,

Trinidad_and_Tobago_data,

Tunisia_data,

Turkey_data,

Turkmenistan_data,

Uganda_data,

Ukraine_data,

United_Arab_Emirates_data,

United_Kingdom_data,

United_States_data,

Uruguay__data,

Uzbekistan_data,

Venezuela__data,

Vietnam_data,

Zambia_data,

Zimbabwe_data

]
df.isnull()

df.isnull().sum()
df_noyear = df.drop(['index_country', 'index_year','overall_score'], axis = 1) 

corr = df_noyear.corr()



corr.style.background_gradient(cmap='coolwarm')
top = []

for x in country_data:

  top.append([[x.iloc[-1]['adjusted_score']],[x.iloc[0]['name']]])

  

top.sort(key=itemgetter(0), reverse=True)

top[:5]
output_notebook()



v = figure(title = 'Countries Overall Score By Year')

v.title.align = 'center'

v.xaxis.axis_label = 'Year'

v.xaxis.axis_label_text_font_size = "12pt"

v.yaxis.axis_label = 'Overall Score'

v.yaxis.axis_label_text_font_size = "12pt"







v.line(x = Hong_Kong_data['index_year'], y= Hong_Kong_data['adjusted_score'], color = 'red', legend = 'Hong Kong')

v.circle(x = Hong_Kong_data['index_year'], y= Hong_Kong_data['adjusted_score'], color = 'red', legend = 'Hong Kong')



v.line(x = New_Zealand_data['index_year'], y = New_Zealand_data['adjusted_score'], color = 'purple', legend = 'New Zealand')

v.circle(x = New_Zealand_data['index_year'], y = New_Zealand_data['adjusted_score'], color = 'purple', legend = 'New Zealand')



v.line(x = Singapore_data['index_year'], y= Singapore_data['adjusted_score'], color = 'green', legend = 'Singapore')

v.circle(x = Singapore_data['index_year'], y= Singapore_data['adjusted_score'], color = 'green', legend = 'Singapore')



v.line(x = Switzerland_data['index_year'], y= Switzerland_data['adjusted_score'], color = 'black', legend = 'Switzerland')

v.circle(x = Switzerland_data['index_year'], y= Switzerland_data['adjusted_score'], color = 'black', legend = 'Switzerland')



v.line(x = Ireland_data['index_year'], y= Ireland_data['adjusted_score'], color = 'blue', legend = 'Ireland')

v.circle(x = Ireland_data['index_year'], y= Ireland_data['adjusted_score'], color = 'blue', legend = 'Ireland')



v.legend.location = 'bottom_right'

v.legend.background_fill_color = "gray"

v.legend.background_fill_alpha = .1



show(v)
regression_model = LinearRegression()



x = Australia_data[["index_year"]]

y = Australia_data[["overall_score"]]

#print(x)

#print(y)



regression_model.fit(x, y) 



y_predicted = regression_model.predict(x)



# model evaluation

rmse = mean_squared_error(y, y_predicted)

r2 = r2_score(y, y_predicted)



# printing values

print('Slope:' ,regression_model.coef_)

print('Intercept:', regression_model.intercept_)

print('Root mean squared error: ', rmse)

print('R2 score: ', r2)





plt.scatter(x, y)

plt.xlabel('x')

plt.ylabel('y')





plt.plot(x, y_predicted, color='r')

plt.show()
r2list = []

rmselist =[]

scores = []



for x in country_data:

  z = x[["index_year"]]

  y = x[["adjusted_score"]]

  #print(x)

  #print(y)

  regression_model.fit(z, y) 

  y_predicted = regression_model.predict(z)

  scores.append(y_predicted)

  # model evaluation

  rmse = mean_squared_error(y, y_predicted)

  r2 = r2_score(y, y_predicted)

  r2list.append(r2) 

  rmselist.append(rmse)

  

print("The average rmse is:",sum(rmselist) / len(rmselist))

print("The average r2 is:",sum(r2list) / len(r2list))

print("The highest r2 value is:",max(r2list),"from ",country_data[r2list.index(max(r2list))].name[0])

print("The lowest r2 value is:",min(r2list),"from ",country_data[r2list.index(min(r2list))].name[3366])
c=Albania_data

regression_model = LinearRegression()

yPrediction = []

predicted_scores = []

averages = []

summ = 0



for col, row in c.iteritems():

  if col in ["property_rights",	"government_integrity",	"tax_burden",

             "government_spending",	"business_freedom",	"monetary_freedom",	

             "trade_freedom",	"investment_freedom",	"financial_freedom"]:

    x=c[['index_country']]

    y = c[[col]]

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)

    regression_model.fit(xTrain, yTrain) 

    yPrediction = regression_model.predict(xTest)

    predicted_scores.append(yPrediction[0])

for x in predicted_scores:

  summ = summ + x

Australia_data

print(summ / 9)

    

c=Albania_data

regression_model = LinearRegression()

yPrediction = []

dft = pd.DataFrame({}) 

predicted_scores = []

dftest = pd.DataFrame({})

#testing variable

count = 0

years = 5

while count < years:

  for col, row in c.iteritems():

    if col in ["property_rights",	"government_integrity",	"tax_burden",

               "government_spending",	"business_freedom",	"monetary_freedom",	

               "trade_freedom",	"investment_freedom",	"financial_freedom"]:

      x = c[['adjusted_score']]

      y = c[[col]]

      xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)

      regression_model.fit(xTrain, yTrain) 

      yPrediction = regression_model.predict(xTest)

      predicted_scores.append(yPrediction[0])

      count = count + 1

  summ=0

  for x in predicted_scores:

    summ = summ + x

  avg = summ / 9



  dft = pd.DataFrame({"name":[Albania_data.iloc[0]['name']],

                        "index_country":[Albania_data.iloc[0]['index_country']],

                        "index_year":[Albania_data.iloc[-1]['index_year'] + 1],

                        "adjusted_score":[avg[0]],

                        "overall_score":[0],

                        "property_rights":[predicted_scores[0][0]], 

                        "government_integrity":[predicted_scores[1][0]],  

                        "tax_burden":[predicted_scores[2][0]],

                        "government_spending":[predicted_scores[3][0]],

                        "business_freedom":[predicted_scores[4][0]],

                        "monetary_freedom":[predicted_scores[5][0]],

                        "trade_freedom":[predicted_scores[6][0]],

                        "investment_freedom":[predicted_scores[7][0]],

                        "financial_freedom":[predicted_scores[8][0]],

                       }) 



  dftest = Albania_data

  dftest = dftest.append(dft,ignore_index=True)

dftest
Chile_data
def predict(c,years):

  regression_model = LinearRegression()

  count = 1

  while count < years + 1:

    #declare variable to clear them

    yPrediction = []

    dft = pd.DataFrame({}) 

    predicted_scores = []

    summ=0

  

    for col, row in c.iteritems():

      if col in ["property_rights",	"government_integrity",	"tax_burden",

                 "government_spending",	"business_freedom",	"monetary_freedom",	

                 "trade_freedom",	"investment_freedom",	"financial_freedom"]:

        x = c[['index_country']]

        y = c[[col]]

        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/6, random_state = 0)

        regression_model.fit(xTrain, yTrain) 

        yPrediction = regression_model.predict(xTest)

        predicted_scores.append(yPrediction)



    for x in predicted_scores:

      summ = summ + x

    avg = summ / 9



    dft = pd.DataFrame({"name":[c.iloc[0]['name']],

                        "index_country":[c.iloc[0]['index_country']],

                        "index_year":[c.iloc[-1]['index_year'] + 1],

                        "adjusted_score":avg[0],

                        "overall_score":[0],

                        "property_rights":predicted_scores[0][0], 

                        "government_integrity":predicted_scores[1][0],  

                        "tax_burden":predicted_scores[2][0],

                        "government_spending":predicted_scores[3][0],

                        "business_freedom":predicted_scores[4][0],

                        "monetary_freedom":predicted_scores[5][0],

                        "trade_freedom":predicted_scores[6][0],

                        "investment_freedom":predicted_scores[7][0],

                        "financial_freedom":predicted_scores[8][0],

                       })

    c=c.append(dft)

    count = count + 1

  return c

predict(Chile_data,1)
predict(Chile_data,6)
top5_2020 = []

for x in range(146):

  dftop = predict(country_data[x],1)

  top5_2020.append([[dftop.iloc[-1]['adjusted_score']],[dftop.iloc[0]['name']]])

  

top5_2020.sort(key=itemgetter(0), reverse=True)

top5_2020[:5]
top5_2025 = []

for x in range(146):

  dftop = predict(country_data[x],6)

  top5_2025.append([[dftop.iloc[-1]['adjusted_score']],[dftop.iloc[0]['name']]])

  

top5_2025.sort(key=itemgetter(0), reverse=True)

top5_2025[:5]
top5_2050 = []

for x in range(146):

  dftop = predict(country_data[x],31)

  top5_2050.append([[dftop.iloc[-1]['adjusted_score']],[dftop.iloc[0]['name']]])

  

top5_2050.sort(key=itemgetter(0), reverse=True)

top5_2050[:5]
output_notebook()



v = figure(title = 'Countries Overall Score By Year')

v.title.align = 'center'

v.xaxis.axis_label = 'Year'

v.xaxis.axis_label_text_font_size = "12pt"

v.yaxis.axis_label = 'Overall Score'

v.yaxis.axis_label_text_font_size = "12pt"









v.line(x = predict(Hong_Kong_data, 6)['index_year'], y= predict(Hong_Kong_data, 6)['adjusted_score'], color = 'red', legend = 'Hong Kong')

v.circle(x = predict(Hong_Kong_data, 6)['index_year'], y= predict(Hong_Kong_data, 6)['adjusted_score'], color = 'red', legend = 'Hong Kong')



v.line(x = predict(Singapore_data, 6)['index_year'], y= predict(Singapore_data, 6)['adjusted_score'], color = 'green', legend = 'Singapore')

v.circle(x = predict(Singapore_data, 6)['index_year'], y= predict(Singapore_data, 6)['adjusted_score'], color = 'green', legend = 'Singapore')



v.line(x = predict(New_Zealand_data, 6)['index_year'], y = predict(New_Zealand_data, 6)['adjusted_score'], color = 'purple', legend = 'New Zealand')

v.circle(x = predict(New_Zealand_data, 6)['index_year'], y = predict(New_Zealand_data, 6)['adjusted_score'], color = 'purple', legend = 'New Zealand')



v.line(x = predict(Switzerland_data, 6)['index_year'], y= predict(Switzerland_data, 6)['adjusted_score'], color = 'black', legend = 'Switzerland')

v.circle(x = predict(Switzerland_data, 6)['index_year'], y= predict(Switzerland_data, 6)['adjusted_score'], color = 'black', legend = 'Switzerland')



v.line(x = predict(Ireland_data, 6)['index_year'], y=predict(Ireland_data, 6)['adjusted_score'], color = 'blue', legend = 'Ireland')

v.circle(x = predict(Ireland_data, 6)['index_year'], y= predict(Ireland_data, 6)['adjusted_score'], color = 'blue', legend = 'Ireland')





v.legend.location = 'bottom_right'

v.legend.background_fill_color = "gray"

v.legend.background_fill_alpha = .6



show(v)
list = []

x=Albania_data

classifiers = [

    linear_model.SGDRegressor(),

    linear_model.BayesianRidge(),

    linear_model.LassoLars(),

    linear_model.ARDRegression(),

    linear_model.PassiveAggressiveRegressor(),

    linear_model.TheilSenRegressor()]

for item in classifiers:

  z = x[["index_year"]]

  y = x[["adjusted_score"]]

  clf = item

  for col, row in c.iteritems():

    if col in ["property_rights",	"government_integrity",	"tax_burden",

             "government_spending",	"business_freedom",	"monetary_freedom",	

             "trade_freedom",	"investment_freedom",	"financial_freedom"]:

      x=c[['index_country']]

      y = c[[col]]

      xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)

      clf.fit(xTrain, yTrain)

      list.append(clf.predict(y))

      #eval

      rmse = mean_squared_error(y, y_predicted)

      r2 = r2_score(y, y_predicted)

      print("rmse",rmse,"r2",r2)
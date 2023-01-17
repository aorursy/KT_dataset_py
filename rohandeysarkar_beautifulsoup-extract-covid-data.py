from bs4 import BeautifulSoup

import requests

from urllib.request import urlretrieve

import pandas as pd

from datetime import date, timedelta
url = 'https://www.worldometers.info/coronavirus/#countries'



r = requests.get(url)



html_doc = r.text



soup = BeautifulSoup(html_doc, "html5lib")



table = soup.find(lambda tag: tag.name == 'table' and tag.has_attr('id') and tag['id'] == "main_table_countries_yesterday")

# print(table)
output_rows = []

for table_row in table.findAll('tr'):

    columns = table_row.findAll('td')

    output_row = []

    for column in columns:

        output_row.append(column.text)

    output_rows.append(output_row)



# print(output_rows)
country = []

totalCases = []

newCases = []

totalDeaths = []

newDeaths = []

totalRecovered = []

activeCases = []

seriousCritical = []

totalCasePerMil = []

deathsPerMil = []

totalTests = []

testsPerMil = []

population = []



for i in range(9, len(output_rows) - 8):

    row = output_rows[i]

    country.append(row[1])

    totalCases.append(row[2])

    newCases.append(row[3])

    totalDeaths.append(row[4])

    newDeaths.append(row[5])

    totalRecovered.append(row[6])

    activeCases.append(row[8])

    seriousCritical.append(row[9])

    totalCasePerMil.append(row[10])

    deathsPerMil.append(row[11])

    totalTests.append(row[12])

    testsPerMil.append(row[13])

    population.append(row[14])
data = pd.DataFrame({'Country': country,

                     'TotalCases': totalCases,

                     'NewCases': newCases,

                     'TotalDeaths': totalDeaths,

                     'NewDeaths': newDeaths,

                     'TotalRecovered': totalRecovered,

                     'ActiveCases': activeCases,

                     'SeriousCritical': seriousCritical,

                     'TotalCasePerMil': totalCasePerMil,

                     'DeathsPerMil': deathsPerMil,

                     'TotalTests': totalTests,

                     'TestsPerMil': testsPerMil,

                     'Country_population': population

                    })



data.head()
yesterday = 'covid19_' + str(date.today() - timedelta(days=1)) 



data.to_csv(yesterday+'.csv', index=False)
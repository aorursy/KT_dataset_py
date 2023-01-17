# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



from csv import DictReader

from collections import Counter

from datetime import datetime
# This isn't an exhaustive list

MANUFACTURERS = set([

    'Aerospatiale', 'Aerostar', 'Agusta',

    'Airbus', 'Airspeed', 'Armstrong-Whitworth', 

    'Antonov', 'Arado', 'Arava', 

    'Armstrong Whitworth', 'Avro', 'BAC', 

    'Beech', 'Beechcraft', 'Bell', 

    'Breguet', 'Boeing', 'Bristol', 

    'British Aerospace', 'Britten Norman', 'CASA', 

    'Canadair', 'Cessna', 'Consolidated', 'Convair',

    'Curtiss', 'Curtiss-Wright', 'Dassault', 'De Havilland', 

    'Dornier', 'Douglas', 'Embraer', 'Eurocopter',

    'Farman', 'Fairchild', 'Fokker', 'Ford', 'Grumman', 'Handley Page', 

    'Hawker Siddeley', 'Junkers', 'Ilyushin', 'Latecoere', 'Learjet',

    'Lockheed', 'McDonnell Douglas', 'Piper', 'Rockwell', 

    'Savoia Marchetti', 'Short', 'Sikorsky', 'Sud Aviation', 

    'Swearingen', 'Tupolev', 'Vickers', 'Yakovlev', 'Zeppelin'])
def cleanRow(row):

    

    row['Date'] = datetime.strptime(row['Date'], '%m/%d/%Y')

    

    if row['Aboard'] == '':

        row['Aboard'] = 0

    else:

        row['Aboard'] = int(row['Aboard'])

    

    row['Manufacturer'] = None    

    for mfr in MANUFACTURERS:

        if row['Type'].lower().startswith(mfr.lower()):

            row['Manufacturer'] = mfr

            row['Type'] = row['Type'][len(mfr):].strip()

            break



    operator = row['Operator']

    if operator.startswith('Military - '):

        row['military'] = True

        row['Operator'] = operator.replace('Military - ', '')

    else:

        row['military'] = False    

    

    return row
def cleansedRows(iterable):

    i = 0

    for row in iterable:

        i += 1

        row['row'] = i

        

        if '/' in row['Type'] and 'collision' in row['Summary'].lower():

            

            # Split rows for collisions into separate records

            rowA = {'collisionWith': i+1}

            rowB = {'collisionWith': i}

            i += 1

            

            for key in row:

                value = row[key]

                if str(value).count('/') == 1:

                    pieces = value.split('/')

                    rowA[key] = pieces[0]

                    rowB[key] = pieces[1]

                else:

                    rowA[key] = value

                    rowB[key] = value

            

            yield cleanRow(rowA)

            yield cleanRow(rowB)

        else:

            row['collisionWith'] = None 

            yield cleanRow(row)
inFilePath = '../input/Airplane_Crashes_and_Fatalities_Since_1908.csv'

DATA = None

with open(inFilePath, 'r') as inFile:

    reader = DictReader(inFile)

    DATA = [record for record in cleansedRows(reader)]

#print(DATA[:3])        
# Determine which aircraft type has had the most crashes.  DC-3/C-47 tops the list



def manufacturerAndType(record):

    mfr = record.get('Manufacturer', '')

    if mfr is None:

        mfr = ''

    type = record.get('Type', '')

    if type is None:

        type = ''

    return ' '.join([mfr, type]).strip()

    

crashesByType = Counter(map(manufacturerAndType, DATA))

mostCrashesByType = sorted(crashesByType.items(), key=lambda x: x[1], reverse=True)

for type, crashes in mostCrashesByType[:25]:

    print("%s %d" % (type.ljust(45), crashes))
# determine crashes per year

crashesByYear = Counter([record['Date'].year for record in DATA])



years = sorted(crashesByYear.keys())

crashes = [crashesByYear[y] for y in years]

print(crashes)



plt.plot(years, crashes)

plt.show()
# determine total number of people aboard crashed aircraft by year

personCrashesPerYear = [(record['Date'].year, record['Aboard']) for record in DATA]

yearlySums = {}

for year, aboard in personCrashesPerYear:

    if year in yearlySums:

        yearlySums[year] += aboard

    else:

        yearlySums[year] = aboard

years = sorted(yearlySums.keys())

aboardByYear = [yearlySums[year] for year in years]



plt.plot(years, aboardByYear)

plt.show()
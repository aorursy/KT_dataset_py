import pandas as pd

import numpy as np

import seaborn as sns

pd.set_option('display.max_columns', None)

pd.options.mode.chained_assignment = None

accidents_short_default = pd.read_csv('../input/Accident_Information.csv', low_memory=False, index_col=None, nrows=10)

accidents_short_default
accidents_original_headers = pd.read_csv('../input/Accident_Information.csv', low_memory=False, index_col=None, nrows=1)

list(accidents_original_headers)
accidents_chunk = pd.read_csv('../input/Accident_Information.csv', low_memory=False, index_col=None, chunksize=200000)

chunk_list = []

for chunk in accidents_chunk:

    chunk_filter = chunk[

        (chunk.Year.astype(int) >= 2010) &

        (chunk.Year.astype(int) <= 2014) &

        (chunk['Road_Type'] != "Unknown") &

        (chunk['Junction_Detail'] != "Data missing or out of range") &

        (chunk['Road_Surface_Conditions'] != "Data missing or out of range") &

        (chunk['Weather_Conditions'] != "Data missing or out of range") &

        (chunk['Latitude'].notnull()) &

        (chunk['Longitude'].notnull()) &

        (chunk['Special_Conditions_at_Site'] == "None")

    ]

    chunk_list.append(chunk_filter)

accidents = pd.concat(chunk_list)
sns.countplot(accidents['Number_of_Vehicles'])
accidents = accidents[accidents.Number_of_Vehicles.astype(int) <= 5]
def label_serious_or_fatal_accident(row):

    if row['Accident_Severity'] == 'Fatal':

        return 'Fatal_Serious'

    elif row['Accident_Severity'] == 'Serious':

        return 'Fatal_Serious'

    else:

        return row['Accident_Severity']

accidents['Accident_Seriousness'] = accidents.apply (lambda row: label_serious_or_fatal_accident(row), axis=1)

accidents.drop(['Accident_Severity'], inplace=True, axis=1)

accidents.rename(columns={'Accident_Seriousness': 'Accident_Severity'}, inplace=True)
accidents['Datetime'] = pd.to_datetime(accidents['Date'] + ' ' + accidents['Time'])

accidents.drop(['Date', 'Time'], inplace=True, axis=1)

accidents = accidents[(accidents['Datetime'].notnull())]

accidents.rename(columns={'Year': 'Year_Number'}, inplace=True)



def label_season(row):

    if '2009-12-20' <= row['Datetime'].strftime('%Y-%m-%d') < '2010-03-20':

        return 4

    elif '2010-03-20' <= row['Datetime'].strftime('%Y-%m-%d') < '2010-06-21':

        return 1

    elif '2010-06-21' <= row['Datetime'].strftime('%Y-%m-%d') < '2010-09-23':

        return 2

    elif '2010-09-23' <= row['Datetime'].strftime('%Y-%m-%d') < '2010-12-21':

        return 3

    elif '2010-12-21' <= row['Datetime'].strftime('%Y-%m-%d') < '2011-03-20':

        return 4

    elif '2011-03-20' <= row['Datetime'].strftime('%Y-%m-%d') < '2011-06-21':

        return 1

    elif '2011-06-21' <= row['Datetime'].strftime('%Y-%m-%d') < '2011-09-22':

        return 2

    elif '2011-09-22' <= row['Datetime'].strftime('%Y-%m-%d') < '2011-12-22':

        return 3

    elif '2011-12-22' <= row['Datetime'].strftime('%Y-%m-%d') < '2012-03-20':

        return 4

    elif '2012-03-20' <= row['Datetime'].strftime('%Y-%m-%d') < '2012-06-21':

        return 1

    elif '2012-06-21' <= row['Datetime'].strftime('%Y-%m-%d') < '2012-09-22':

        return 2

    elif '2012-09-22' <= row['Datetime'].strftime('%Y-%m-%d') < '2012-12-22':

        return 3

    elif '2012-12-22' <= row['Datetime'].strftime('%Y-%m-%d') < '2013-03-20':

        return 4

    elif '2013-03-20' <= row['Datetime'].strftime('%Y-%m-%d') < '2013-06-20':

        return 1

    elif '2013-06-20' <= row['Datetime'].strftime('%Y-%m-%d') < '2013-09-22':

        return 2

    elif '2013-09-22' <= row['Datetime'].strftime('%Y-%m-%d') < '2013-12-21':

        return 3

    elif '2013-12-21' <= row['Datetime'].strftime('%Y-%m-%d') < '2014-03-20':

        return 4

    elif '2014-03-20' <= row['Datetime'].strftime('%Y-%m-%d') < '2014-06-21':

        return 1

    elif '2014-06-21' <= row['Datetime'].strftime('%Y-%m-%d') < '2014-09-23':

        return 2

    elif '2014-09-23' <= row['Datetime'].strftime('%Y-%m-%d') < '2014-12-21':

        return 3

    elif '2014-12-21' <= row['Datetime'].strftime('%Y-%m-%d') < '2015-03-20':

        return 4

    else:

        return row['Datetime'].strftime('%Y-%m-%d')

accidents['Season'] = accidents.apply (lambda row: label_season(row), axis=1)



def label_month_of_year(row):

    return row['Datetime'].strftime('%m').lstrip("0").replace(" 0", "")

accidents['Month_of_Year'] = accidents.apply (lambda row: label_month_of_year(row), axis=1)



def label_day_of_month(row):

    return row['Datetime'].strftime('%d').lstrip("0").replace(" 0", "")

accidents['Day_of_Month'] = accidents.apply (lambda row: label_day_of_month(row), axis=1)



accidents = accidents.rename(columns = {'Day_of_Week': 'Day'})

def label_day_of_week(row):

    if row['Day'] == 'Monday':

        return 1

    elif row['Day'] == 'Tuesday':

        return 2

    elif row['Day'] == 'Wednesday':

        return 3

    elif row['Day'] == 'Thursday':

        return 4

    elif row['Day'] == 'Friday':

        return 5

    elif row['Day'] == 'Saturday':

        return 6

    elif row['Day'] == 'Sunday':

        return 7

    else:

        return row['Day']

accidents['Day_of_Week'] = accidents.apply (lambda row: label_day_of_week(row), axis=1)

accidents.drop(['Day'], inplace=True, axis=1)



def label_hour_of_day(row):

    return round(int((int(row['Datetime'].strftime('%H'))*60) + (int(row['Datetime'].strftime('%M'))))/1440, 3)

accidents['Hour_of_Day'] = accidents.apply (lambda row: label_hour_of_day(row), axis=1)



def label_junction(row):

    if row['Junction_Detail'] == 'Mini-roundabout':

        return 'Roundabout'

    else:

        return row['Junction_Detail']

accidents['Junction'] = accidents.apply (lambda row: label_junction(row), axis=1)

accidents.drop(['Junction_Detail'], inplace=True, axis=1)

accidents.rename(columns={'Junction': 'Junction_Detail'}, inplace=True)



def label_weather(row):

    if row['Weather_Conditions'] == 'Fine + high winds':

        return 'Fine'

    elif row['Weather_Conditions'] == 'Fine no high winds':

        return 'Fine'

    elif row['Weather_Conditions'] == 'Raining + high winds':

        return 'Raining'

    elif row['Weather_Conditions'] == 'Raining no high winds':

        return 'Raining'

    elif row['Weather_Conditions'] == 'Snowing + high winds':

        return 'Snowing'

    elif row['Weather_Conditions'] == 'Snowing no high winds':

        return 'Snowing'

    else:

        return row['Weather_Conditions']

accidents['Weather'] = accidents.apply (lambda row: label_weather(row), axis=1)



def label_high_wind(row):

    if row['Weather_Conditions'] == 'Fine + high winds':

        return 'Yes'

    elif row['Weather_Conditions'] == 'Fine no high winds':

        return 'No'

    elif row['Weather_Conditions'] == 'Raining + high winds':

        return 'Yes'

    elif row['Weather_Conditions'] == 'Raining no high winds':

        return 'No'

    elif row['Weather_Conditions'] == 'Snowing + high winds':

        return 'Yes'

    elif row['Weather_Conditions'] == 'Snowing no high winds':

        return 'No'

    else:

        return 'No'

accidents['High_Wind'] = accidents.apply (lambda row: label_high_wind(row), axis=1)

accidents.drop(['Weather_Conditions'], inplace=True, axis=1)



def label_lights(row):

    if row['Light_Conditions'] == 'Darkness - lights lit':

        return 'Darkness - lights'

    elif row['Light_Conditions'] == 'Darkness - lights unlit':

        return 'Darkness - no lights'

    elif row['Light_Conditions'] == 'Darkness - no lighting':

        return 'Darkness - no lights'

    else:

        return row['Light_Conditions']

accidents['Lights'] = accidents.apply (lambda row: label_lights(row), axis=1)

accidents.drop(['Light_Conditions'], inplace=True, axis=1)
east_midlands = [

    'Amber Valley', 'Ashfield', 'Bassetlaw', 'Blaby', 'Bolsover', 'Boston', 'Broxtowe', 'Charnwood', 'Chesterfield',

    'Corby', 'Daventry', 'Derby', 'Derbyshire Dales', 'East Lindsey', 'East Northamptonshire', 'Erewash', 'Gedling',

    'Harborough', 'High Peak', 'Hinckley and Bosworth', 'Kettering', 'Leicester', 'Lincoln', 'Mansfield', 'Melton',

    'Newark and Sherwood', 'North East Derbyshire', 'North Kesteven', 'North West Leicestershire', 'Northampton',

    'Nottingham', 'Oadby and Wigston', 'Rushcliffe', 'Rutland', 'South Derbyshire', 'South Holland', 'South Kesteven',

    'South Northamptonshire', 'Wellingborough', 'West Lindsey'

]

east_england = [

    'Babergh', 'Basildon', 'Bedford', 'Braintree', 'Breckland', 'Brentwood', 'Broadland', 'Broxbourne', 'Cambridge',

    'Castle Point', 'Central Bedfordshire', 'Chelmsford', 'Colchester', 'Dacorum', 'East Cambridgeshire',

    'East Hertfordshire', 'Epping Forest', 'Fenland', 'Forest Heath', 'Great Yarmouth', 'Harlow', 'Hertsmere',

    'Huntingdonshire', 'Ipswich', "King's Lynn and West Norfolk", 'Luton', 'Maldon', 'Mid Suffolk', 'North Hertfordshire',

    'North Norfolk', 'Norwich', 'Peterborough', 'Rochford', 'South Cambridgeshire', 'South Norfolk', 'Southend-on-Sea',

    'St. Albans', 'St. Edmundsbury', 'Stevenage', 'Suffolk Coastal', 'Tendring', 'Three Rivers', 'Thurrock', 'Uttlesford',

    'Watford', 'Waveney', 'Welwyn Hatfield'

]

london = [

    'Barking and Dagenham', 'Barnet', 'Bexley', 'Brent', 'Bromley', 'Camden', 'City of London', 'Croydon', 'Ealing',

    'Enfield', 'Greenwich', 'Hackney', 'Hammersmith and Fulham', 'Haringey', 'Harrow', 'Havering', 'Hillingdon', 'Hounslow',

    'Islington', 'Kensington and Chelsea', 'Kingston upon Thames', 'Lambeth', 'Lewisham', 'London Airport (Heathrow)', 'Merton',

    'Newham', 'Redbridge', 'Richmond upon Thames', 'Southwark', 'Sutton', 'Tower Hamlets', 'Waltham Forest', 'Wandsworth', 'Westminster'

]

north_east_england = [

    'County Durham', 'Darlington', 'Gateshead', 'Hartlepool', 'Middlesbrough', 'Newcastle upon Tyne', 'North Tyneside',

    'Northumberland', 'Redcar and Cleveland', 'South Tyneside', 'Stockton-on-Tees', 'Sunderland'

]

north_west_england = [

    'Allerdale', 'Barrow-in-Furness', 'Blackburn with Darwen', 'Blackpool', 'Bolton', 'Burnley', 'Bury', 'Carlisle',

    'Cheshire East', 'Cheshire West and Chester', 'Chorley', 'Copeland', 'Eden', 'Fylde', 'Halton', 'Hyndburn',

    'Knowsley', 'Lancaster', 'Liverpool', 'Manchester', 'Oldham', 'Pendle', 'Preston', 'Ribble Valley', 'Rochdale',

    'Rossendale', 'Salford', 'Sefton', 'South Lakeland', 'South Ribble', 'St. Helens', 'Stockport', 'Tameside',

    'Trafford', 'Warrington', 'West Lancashire', 'Wigan', 'Wirral', 'Wyre'

]

scotland = [

    'Aberdeen City', 'Aberdeenshire', 'Angus', 'Argyll and Bute', 'Clackmannanshire', 'Dumfries and Galloway',

    'Dundee City', 'East Ayrshire', 'East Dunbartonshire', 'East Lothian', 'East Renfrewshire', 'Edinburgh, City of',

    'Falkirk', 'Fife', 'Glasgow City', 'Highland', 'Inverclyde', 'Midlothian', 'Moray', 'North Ayrshire',

    'North Lanarkshire', 'Orkney Islands', 'Perth and Kinross', 'Renfrewshire', 'Scottish Borders', 'Shetland Islands',

    'South Ayrshire', 'South Lanarkshire', 'Stirling', 'West Dunbartonshire', 'West Lothian', 'Western Isles'

]

south_east_england = [

    'Adur', 'Arun', 'Ashford', 'Aylesbury Vale', 'Basingstoke and Deane', 'Bracknell Forest', 'Brighton and Hove',

    'Canterbury', 'Cherwell', 'Chichester', 'Chiltern', 'Crawley', 'Dartford', 'Dover', 'East Hampshire', 'Eastbourne',

    'Eastleigh', 'Elmbridge', 'Epsom and Ewell', 'Fareham', 'Gosport', 'Gravesham', 'Guildford', 'Hart', 'Hastings',

    'Havant', 'Horsham', 'Isle of Wight', 'Lewes', 'Maidstone', 'Medway', 'Mid Sussex', 'Milton Keynes', 'Mole Valley',

    'New Forest', 'Oxford', 'Portsmouth', 'Reading', 'Reigate and Banstead', 'Rother', 'Runnymede', 'Rushmoor',

    'Sevenoaks', 'Shepway', 'Slough', 'South Bucks', 'South Oxfordshire', 'Southampton', 'Spelthorne', 'Surrey Heath',

    'Swale', 'Tandridge', 'Test Valley', 'Thanet', 'Tonbridge and Malling', 'Tunbridge Wells', 'Vale of White Horse',

    'Waverley', 'Wealden', 'West Berkshire', 'West Oxfordshire', 'Winchester', 'Windsor and Maidenhead', 'Woking',

    'Wokingham', 'Worthing', 'Wycombe'

]

south_west_england =[

    'Bath and North East Somerset', 'Bournemouth', 'Bristol, City of', 'Cheltenham', 'Christchurch', 'Cornwall',

    'Cotswold', 'East Devon', 'East Dorset', 'Exeter', 'Forest of Dean', 'Gloucester', 'Mendip', 'Mid Devon',

    'North Devon', 'North Dorset', 'North Somerset', 'Plymouth', 'Poole', 'Purbeck', 'Sedgemoor', 'South Gloucestershire',

    'South Hams', 'South Somerset', 'Stroud', 'Swindon', 'Taunton Deane', 'Teignbridge', 'Tewkesbury', 'Torbay',

    'Torridge', 'West Devon', 'West Dorset', 'West Somerset', 'Weymouth and Portland', 'Wiltshire'

]

wales = [

    'Blaenau Gwent', 'Bridgend', 'Caerphilly', 'Cardiff', 'Carmarthenshire', 'Ceredigion', 'Conwy', 'Denbighshire',

    'Flintshire', 'Gwynedd', 'Isle of Anglesey', 'Merthyr Tydfil', 'Monmouthshire', 'Neath Port Talbot', 'Newport',

    'Pembrokeshire', 'Powys', 'Rhondda, Cynon, Taff', 'Swansea', 'The Vale of Glamorgan', 'Torfaen', 'Wrexham'

]

west_midlands = [

    'Birmingham', 'Bromsgrove', 'Cannock Chase', 'Coventry', 'Dudley', 'East Staffordshire', 'Herefordshire, County of',

    'Lichfield', 'Malvern Hills', 'Newcastle-under-Lyme', 'North Warwickshire', 'Nuneaton and Bedworth', 'Redditch',

    'Rugby', 'Sandwell', 'Shropshire', 'Solihull', 'South Staffordshire', 'Stafford', 'Staffordshire Moorlands',

    'Stoke-on-Trent', 'Stratford-upon-Avon', 'Tamworth', 'Telford and Wrekin', 'Walsall', 'Warwick', 'Wolverhampton',

    'Worcester', 'Wychavon', 'Wyre Forest'

]

yorkshire_and_the_humber = [

    'Barnsley', 'Bradford', 'Calderdale', 'Craven', 'Doncaster', 'East Riding of Yorkshire', 'Hambleton', 'Harrogate',

    'Kingston upon Hull, City of', 'Kirklees', 'Leeds', 'North East Lincolnshire', 'North Lincolnshire', 'Richmondshire',

    'Rotherham', 'Ryedale', 'Scarborough', 'Selby', 'Sheffield', 'Wakefield', 'York'

]



def label_region(row):

    if row['Local_Authority_(District)'] in east_midlands:

        return 'East Midlands'

    elif row['Local_Authority_(District)'] in east_england:

        return 'East England'

    elif row['Local_Authority_(District)'] in london:

        return 'London'

    elif row['Local_Authority_(District)'] in north_east_england:

        return 'North East England'

    elif row['Local_Authority_(District)'] in north_west_england:

        return 'North West England'

    elif row['Local_Authority_(District)'] in scotland:

        return 'Scotland'

    elif row['Local_Authority_(District)'] in south_east_england:

        return 'South East England'

    elif row['Local_Authority_(District)'] in south_west_england:

        return 'South West England'

    elif row['Local_Authority_(District)'] in wales:

        return 'Wales'

    elif row['Local_Authority_(District)'] in west_midlands:

        return 'Wast Midlands'

    elif row['Local_Authority_(District)'] in yorkshire_and_the_humber:

        return 'Yorkshire and the Humber'

accidents['Region'] = accidents.apply (lambda row: label_region(row), axis=1)

accidents.drop(['Local_Authority_(District)'], inplace=True, axis=1)
vehicles_short_default = pd.read_csv('../input/Vehicle_Information.csv', low_memory=False, index_col=None, nrows=10)

vehicles_short_default
vehicles_original_headers = pd.read_csv('../input/Vehicle_Information.csv', low_memory=False, index_col=None, nrows=1)

list(vehicles_original_headers)
vehicles_chunk = pd.read_csv('../input/Vehicle_Information.csv', low_memory=False, index_col=None, chunksize=200000,

                             encoding = "ISO-8859-1")

chunk_list = []

for chunk in vehicles_chunk:

    chunk_list.append(chunk)

vehicles = pd.concat(chunk_list)

accidents_interesting_list = accidents['Accident_Index'].tolist()

vehicles = vehicles[vehicles['Accident_Index'].isin(accidents_interesting_list)]
vehicles_uninteresting = vehicles[

    (vehicles['Age_Band_of_Driver'] == 'Data missing or out of range') |

    (vehicles['Age_Band_of_Driver'] == '0 - 5') |

    (vehicles['Age_Band_of_Driver'] == '6 - 10') |

    (vehicles['Age_Band_of_Driver'] == '11 - 15') |

    (vehicles['Age_of_Vehicle'].isnull()) |

    (vehicles['Driver_IMD_Decile'].isnull()) |

    (vehicles['Engine_Capacity_.CC.'].isnull()) |

    (vehicles['Propulsion_Code'] == 'Electric diesel') |

    (vehicles['Propulsion_Code'] == 'Fuel cells') |

    (vehicles['Propulsion_Code'] == 'Gas') |

    (vehicles['Propulsion_Code'] == 'Gas diesel') |

    (vehicles['Propulsion_Code'] == 'Gas/Bi-fuel') |

    (vehicles['Propulsion_Code'] == 'Hybrid electric') |

    (vehicles['Propulsion_Code'] == 'New fuel technology') |

    (vehicles['Propulsion_Code'] == 'Petrol/Gas (LPG)') |

    (vehicles['make'].isnull()) |

    (vehicles['Driver_Home_Area_Type'] == 'Data missing or out of range') |

    (vehicles['Sex_of_Driver'] == 'Not known') |

    (vehicles['Sex_of_Driver'] == 'Data missing or out of range') |

    (vehicles['X1st_Point_of_Impact'] == 'Data missing or out of range') |

    (vehicles['Vehicle_Manoeuvre'] == 'Data missing or out of range') |

    (vehicles['Junction_Location'] == 'Data missing or out of range')

    ]

accidents_uninteresting_list = vehicles_uninteresting['Accident_Index'].tolist()

accidents = accidents[~accidents['Accident_Index'].isin(accidents_uninteresting_list)]
sns.countplot(vehicles['Age_of_Vehicle'])
sns.countplot(vehicles['Engine_Capacity_.CC.'])
accidents_interesting_list = accidents['Accident_Index'].tolist()

vehicles = vehicles[vehicles['Accident_Index'].isin(accidents_interesting_list)]

vehicles_uninteresting = vehicles[

    (vehicles.Age_of_Vehicle.astype(int) > 30) |

    (vehicles.Driver_IMD_Decile.astype(int) > 10) |

    (vehicles['Engine_Capacity_.CC.'].astype(int) > 3500)

    ]

accidents_uninteresting_list = vehicles_uninteresting['Accident_Index'].tolist()

accidents = accidents[~accidents['Accident_Index'].isin(accidents_uninteresting_list)]
def label_driver_journey_purpose(row):

    if row['Journey_Purpose_of_Driver'] == 'Other/Not known (2005-10)':

        return 'Other/Not known'

    elif row['Journey_Purpose_of_Driver'] == 'Not known':

        return 'Other/Not known'

    elif row['Journey_Purpose_of_Driver'] == 'Other':

        return 'Other/Not known'

    elif row['Journey_Purpose_of_Driver'] == 'Data missing or out of range':

        return 'Other/Not known'

    else:

        return row['Journey_Purpose_of_Driver']

vehicles['Driver_Journey_Purpose'] = vehicles.apply (lambda row: label_driver_journey_purpose(row), axis=1)

vehicles.drop(['Journey_Purpose_of_Driver'], inplace=True, axis=1)



def label_age_of_driver(row):

    if row['Age_Band_of_Driver'] == '16 - 20':

        return 1

    elif row['Age_Band_of_Driver'] == '21 - 25':

        return 2

    elif row['Age_Band_of_Driver'] == '26 - 35':

        return 3

    elif row['Age_Band_of_Driver'] == '36 - 45':

        return 4

    elif row['Age_Band_of_Driver'] == '46 - 55':

        return 5

    elif row['Age_Band_of_Driver'] == '56 - 65':

        return 6

    elif row['Age_Band_of_Driver'] == '66 - 75':

        return 7

    elif row['Age_Band_of_Driver'] == 'Over 75':

        return 8

vehicles['Age_of_Driver'] = vehicles.apply (lambda row: label_age_of_driver(row), axis=1)

vehicles.drop(['Age_Band_of_Driver'], inplace=True, axis=1)



vehicles.rename(columns={'Engine_Capacity_.CC.': 'Engine_CC'}, inplace=True)



def label_make(row):

    if row['make'] == 'FORD':

        return 'Ford'

    elif row['make'] == 'VAUXHALL':

        return 'Vauxhall'

    elif row['make'] == 'PEUGEOT':

        return 'Peugeot'

    elif row['make'] == 'VOLKSWAGEN':

        return 'Volkswagen'

    elif row['make'] == 'RENAULT':

        return 'Renault'

    elif row['make'] == 'HONDA':

        return 'Honda'

    elif row['make'] == 'TOYOTA':

        return 'Toyota'

    elif row['make'] == 'MERCEDES':

        return 'Mercedes'

    elif row['make'] == 'CITROEN':

        return 'Citroen'

    elif row['make'] == 'BMW':

        return 'BMW'

    elif row['make'] == 'NISSAN':

        return 'Nissan'

    elif row['make'] == 'AUDI':

        return 'Audi'

    elif row['make'] == 'FIAT':

        return 'Fiat'

    elif row['make'] == 'VOLVO':

        return 'Volvo'

    elif row['make'] == 'SUZUKI':

        return 'Suzuki'

    elif row['make'] == 'SKODA':

        return 'Skoda'

    elif row['make'] == 'YAMAHA':

        return 'Yamaha'

    elif row['make'] == 'LAND ROVER':

        return 'Land Rover'

    elif row['make'] == 'SEAT':

        return 'Seat'

    elif row['make'] == 'HYUNDAI':

        return 'Hyundai'

    elif row['make'] == 'MAZDA':

        return 'Mazda'

    elif row['make'] == 'ROVER':

        return 'Rover'

    elif row['make'] == 'MINI':

        return 'Mini'

    elif row['make'] == 'KIA':

        return 'Kia'

    elif row['make'] == '':

        return ''

    else:

        return 'Other'

vehicles['Vehicle_Make'] = vehicles.apply (lambda row: label_make(row), axis=1)

vehicles.drop(['make'], inplace=True, axis=1)



def label_vehicle_category(row):

    if row['Vehicle_Type'] == 'Bus or coach (17 or more pass seats)':

        return 'Bus/minibus'

    elif row['Vehicle_Type'] == 'Minibus (8 - 16 passenger seats)':

        return 'Bus/minibus'

    elif row['Vehicle_Type'] == 'Taxi/Private hire car':

        return 'Taxi'

    elif row['Vehicle_Type'] == 'Van / Goods 3.5 tonnes mgw or under':

        return 'Van'

    elif row['Vehicle_Type'] == 'Motorcycle 125cc and under':

        return 'Motorcycle'

    elif row['Vehicle_Type'] == 'Motorcycle 50cc and under':

        return 'Motorcycle'

    elif row['Vehicle_Type'] == 'Motorcycle over 125cc and up to 500cc':

        return 'Motorcycle'

    elif row['Vehicle_Type'] == 'Motorcycle over 500cc':

        return 'Motorcycle'

    elif row['Vehicle_Type'] == 'Motorcycle - unknown cc':

        return 'Motorcycle'

    elif row['Vehicle_Type'] == 'Agricultural vehicle':

        return 'Other'

    elif row['Vehicle_Type'] == 'Electric motorcycle':

        return 'Other'

    elif row['Vehicle_Type'] == 'Goods 7.5 tonnes mgw and over':

        return 'Other'

    elif row['Vehicle_Type'] == 'Goods over 3.5t. and under 7.5t':

        return 'Other'

    elif row['Vehicle_Type'] == 'Goods vehicle - unknown weight':

        return 'Other'

    elif row['Vehicle_Type'] == 'Other vehicle':

        return 'Other'

    elif row['Vehicle_Type'] == 'Pedal cycle':

        return 'Other'

    elif row['Vehicle_Type'] == 'Ridden horse':

        return 'Other'

    elif row['Vehicle_Type'] == 'Tram':

        return 'Other'

    elif row['Vehicle_Type'] == 'Data missing or out of range':

        return 'Other'

    elif row['Vehicle_Type'] == 'Mobility scooter':

        return 'Other'

    else:

        return row['Vehicle_Type']

vehicles['Vehicle_Category'] = vehicles.apply (lambda row: label_vehicle_category(row), axis=1)

vehicles.drop(['Vehicle_Type'], inplace=True, axis=1)



def label_vehicle_maneuver(row):

    if row['Vehicle_Manoeuvre'] == 'Changing lane to left':

        return 'Changing lane'

    elif row['Vehicle_Manoeuvre'] == 'Changing lane to right':

        return 'Changing lane'

    elif row['Vehicle_Manoeuvre'] == 'Going ahead left-hand bend':

        return 'Going ahead'

    elif row['Vehicle_Manoeuvre'] == 'Going ahead other':

        return 'Going ahead'

    elif row['Vehicle_Manoeuvre'] == 'Going ahead right-hand bend':

        return 'Going ahead'

    elif row['Vehicle_Manoeuvre'] == 'Overtaking - nearside':

        return 'Overtaking'

    elif row['Vehicle_Manoeuvre'] == 'Overtaking moving vehicle - offside':

        return 'Overtaking'

    elif row['Vehicle_Manoeuvre'] == 'Overtaking static vehicle - offside':

        return 'Overtaking'

    elif row['Vehicle_Manoeuvre'] == 'Waiting to go - held up':

        return 'Waiting to go'

    elif row['Vehicle_Manoeuvre'] == 'Waiting to turn left':

        return 'Waiting to go'

    elif row['Vehicle_Manoeuvre'] == 'Waiting to turn right':

        return 'Waiting to go'

    else:

        return row['Vehicle_Manoeuvre']

vehicles['Vehicle_Maneuver'] = vehicles.apply (lambda row: label_vehicle_maneuver(row), axis=1)

vehicles.drop(['Vehicle_Manoeuvre'], inplace=True, axis=1)

vehicles.rename(columns={'Vehicle_Maneuver': 'Vehicle_Manoeuvre'}, inplace=True)
accidents = accidents.set_index('Accident_Index')

vehicles = vehicles.set_index('Accident_Index')

accidents_categorical = pd.merge(accidents, vehicles, on='Accident_Index')

accidents_categorical.head(10)
columns_to_drop = ['1st_Road_Number', '2nd_Road_Class', '2nd_Road_Number', 'Carriageway_Hazards',

                   'Did_Police_Officer_Attend_Scene_of_Accident', 'Junction_Control',

                   'Local_Authority_(Highway)', 'Location_Easting_OSGR', 'Location_Northing_OSGR',

                   'LSOA_of_Accident_Location', 'Pedestrian_Crossing-Human_Control',

                   'Pedestrian_Crossing-Physical_Facilities', 'Police_Force', 'Number_of_Casualties',

                   'Special_Conditions_at_Site', 'InScotland', 'Hit_Object_in_Carriageway',

                   'Hit_Object_off_Carriageway', 'model', 'Skidding_and_Overturning', 

                   'Towing_and_Articulation', 'Vehicle_Leaving_Carriageway', 'Vehicle_Location.Restricted_Lane',

                   'Vehicle_Reference', 'Was_Vehicle_Left_Hand_Drive', 'Year']

accidents_categorical = accidents_categorical.drop(columns_to_drop, axis='columns')

accidents_categorical['Speed_limit'] = accidents_categorical['Speed_limit'].astype(int)

accidents_categorical['Age_of_Vehicle'] = accidents_categorical['Age_of_Vehicle'].astype(int)

accidents_categorical['Engine_CC'] = accidents_categorical['Engine_CC'].astype(int)

accidents_categorical['Driver_IMD_Decile'] = accidents_categorical['Driver_IMD_Decile'].astype(int)

accidents_categorical.rename(columns={'Year_Number': 'Year'}, inplace=True)

accidents_categorical['Accident_Index'] = accidents_categorical.index

cols = [

    'Accident_Index',

    'Latitude',

    'Longitude',

    'Region',

    'Urban_or_Rural_Area',

    '1st_Road_Class',

    'Driver_IMD_Decile',

    'Speed_limit',

    'Road_Type',

    'Road_Surface_Conditions',

    'Weather',

    'High_Wind',

    'Lights',

    'Datetime',

    'Year',

    'Season',

    'Month_of_Year',

    'Day_of_Month',

    'Day_of_Week',

    'Hour_of_Day',

    'Number_of_Vehicles',

    'Age_of_Driver',

    'Age_of_Vehicle',

    'Junction_Detail',

    'Junction_Location',

    'X1st_Point_of_Impact',

    'Driver_Journey_Purpose',

    'Engine_CC',

    'Propulsion_Code',

    'Vehicle_Make',

    'Vehicle_Category',

    'Vehicle_Manoeuvre',

    'Accident_Severity'

]

accidents_categorical = accidents_categorical[cols]

accidents_categorical.head(10)
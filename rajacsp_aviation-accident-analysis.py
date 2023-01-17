import warnings

import pandas as pd;

import numpy as np;

import matplotlib.pyplot as plt;

import operator; # for dictionary sorting by value



def fxn():

    warnings.warn("deprecated", DeprecationWarning)



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    fxn()





# Import .csv file (use forward slash - windows)

airplanedata = pd.read_csv('../input/AviationDataUP.csv',low_memory = False);
print(airplanedata.head(2));
'''

fill data

Total.Fatal.Injuries 0

'''

airplanedata["Total.Fatal.Injuries"] = airplanedata["Total.Fatal.Injuries"].fillna(0);

airplanedata["Make"] = airplanedata["Make"].fillna('UNKNOWN');

airplanedata["Total.Fatal.Injuries"] = airplanedata["Total.Fatal.Injuries"].fillna(0).astype(int);

airplanedata["Country"] = airplanedata["Country"].fillna('UNKNOWN');

airplanedata["Event.Date"] = airplanedata["Event.Date"].fillna('UNKNOWN');

airplanedata.Make = airplanedata.Make.str.lower();

airplanedata.Country = airplanedata.Country.str.lower();

airplanedata["Event.Date"] = airplanedata["Event.Date"].str.lower();



sorted_airplane_data = airplanedata.sort_values(['Total.Fatal.Injuries'], ascending = False);



# get total accidents

total_accidents = len(airplanedata);



# remove duplicates

sorted_airplane_data = sorted_airplane_data.drop_duplicates("Event.Id");
'''

2016-11-14

'''

def get_year(event_date):

    

    if(len(event_date.strip()) == 0):

        return -1;

    

    dob_stripped = event_date.split('-');

    

    if(len(dob_stripped) != 3):

        return -2;

    

    return int(dob_stripped[0]);





'''

Get Continent of specified country

'''

def get_continent(country, return_unknown_country = True):

    

    if(country == 'unknown' and return_unknown_country == True):

        return 'unknown_country';

    

    if (

        country in "Burundi, Comoros, Djibouti, Eritrea, Ethiopia, Kenya, Madagascar, Malawi, Mauritius, Mayotte, Mozambique, Réunion, Rwanda, Seychelles, Somalia, South Sudan, Uganda".lower()

        or country in "United Republic of Tanzania, Zambia, Zimbabwe, Angola, Cameroon, Central African Republic, Chad, Congo, Democratic Republic of the Congo, Equatorial Guinea, Gabon".lower()

        or country in "Sao Tome and Principe, Algeria, Egypt, Libya, Morocco, Sudan, Tunisia, Western Sahara, Botswana, Lesotho, Namibia, South Africa, Swaziland, Benin, Burkina Faso, Cabo Verde".lower()

        or country in "Cote d'Ivoire, Gambia, Ghana, Guinea,Guinea-Bissau Liberia, Mali,Mauritania Niger, Nigeria,Saint Helena Senegal, Sierra Leone,Togo, Cape Verde".lower()

        or country in "Congo, Dem Rep, reunion, ivory coast".lower()

    ):

        return "africa";

    

    if (

        country in "Anguilla, Antigua and Barbuda, Aruba, Bahamas, Barbados, Bonaire, Sint Eustatius and Saba, British Virgin Islands, Cayman Islands, Cuba,".lower()

        or country in "Curaçao, Dominica, Dominican Republic, Grenada, Guadeloupe, Haiti, Jamaica, Martinique, Montserrat, Puerto Rico, Saint-Barthélemy, Saint Kitts and Nevis,".lower()

        or country in "Saint Lucia, Saint Martin (French part), Saint Vincent and the Grenadines, Sint Maarten (Dutch part), Trinidad and Tobago, Turks and Caicos Islands,".lower()

        or country in "United States Virgin Islands, Belize, Costa Rica, El Salvador, Guatemala, Honduras, Mexico, Nicaragua, Panama, Bermuda, Canada, Greenland,".lower()

        or country in "Saint Pierre and Miquelon, United States of America, United States, Bahamas, st vincent and the grenadines, ".lower()

        or country in "st kitts and nevis, st kitts and nevis, west indies, st lucia".lower()

        

    ):

        return "north america";     

    

  

    if (

        country in "Argentina, Bolivia (Plurinational State of), Brazil, Chile, Colombia, Ecuador, Falkland Islands (Malvinas), French Guiana, Guyana,".lower()

        or country in "Paraguay, Peru, Suriname, Uruguay, Venezuela (Bolivarian Republic of)".lower()

    ):

        return "south america";

    

    if (

        country in "Kazakhstan, Kyrgyzstan, Tajikistan, Turkmenistan, Uzbekistan,China, China, Hong Kong Special Administrative Region, China, Macao Special Administrative Region,".lower()

        or country in "Democratic People's Republic of Korea, Japan, Mongolia, Republic of Korea, Afghanistan, Bangladesh, Bhutan, India, Iran (Islamic Republic of), Maldives,".lower()

        or country in "Nepal, Pakistan, Sri Lanka, Brunei Darussalam, Cambodia, Indonesia, Lao People's Democratic Republic, Malaysia, Myanmar, Philippines, Singapore, Thailand,".lower()

        or country in "Timor-Leste, Viet Nam, Armenia, Azerbaijan, Bahrain, Cyprus, Georgia, Iraq, Israel, Jordan, Kuwait, Lebanon, Oman, Qatar, Saudi Arabia, State of Palestine,".lower()

        or country in "Syrian Arab Republic, Turkey, United Arab Emirates, Yemen, ".lower()

        or country in "East Timor (Timor-Leste), Korea, North, Korea, South,  korea, republic of".lower()

        or country in "Laos, Burma, Palestine, Occupied Territories, Taiwan, Vietnam".lower()

    ):

        return "asia";

    

    if (

        country in "Belarus, Bulgaria, Czechia, Hungary, Poland, Republic of Moldova, Romania, Russian Federation, Slovakia, Ukraine, Åland Islands, Channel Islands,".lower()

        or country in "Denmark, Estonia,  Faeroe Islands, Finland, Guernsey, Iceland, Ireland, Isle of Man, Jersey, Latvia, Lithuania, Norway, Sark, Svalbard and Jan Mayen Islands,".lower()

        or country in "Sweden, United Kingdom of Great Britain and Northern Ireland, Albania, Andorra, Bosnia and Herzegovina, Croatia, Gibraltar,Greece, Holy See, Italy, Malta,".lower()

        or country in "Montenegro, Portugal, San Marino, Serbia, Slovenia, Spain, The former Yugoslav Republic of Macedonia, Austria, Belgium, France, Germany,".lower()

        or country in "Liechtenstein, Luxembourg, Monaco, Netherlands, Switzerland, Netherlands Antilles*, Czech Republic".lower()        

    ):

        return "europe";

    

    if (

        country in "Australia, New Zealand, Norfolk Island, Fiji, New Caledonia, Papua New Guinea,Solomon Islands, Vanuatu, Guam, Kiribati, Marshall Islands, Micronesia (Federated States of),".lower()

        or country in "Nauru, Northern Mariana Islands, Palau, American Samoa, Cook Islands, French Polynesia, Niue, Pitcairn, Samoa,Tokelau,".lower()

        or country in "Tonga, Tuvalu, Wallis and Futuna Islands, federated states of micronesia, ".lower()

    ):

        return "oceania";

    

    if (

        country in "antarctica".lower()

    ):    

        return "antarctica";

     

    return "unknown";





row_count = 0;

accidents_make = {};

accidents_country = {};

accidents_continent_yearly = {};

fatal_injuries_continent_yearly = {};

accidents_yearly = {};

fatal_injuries_yearly = {};

for index, row in sorted_airplane_data.iterrows():

    

    row_count = row_count + 1;

    

    country = row['Country'];   

    event_id = row['Event.Id'];

    make = row['Make'];

    event_year = get_year(row['Event.Date']);

    continent = get_continent(country, True);

    

    

    total_fatal_injuries = row['Total.Fatal.Injuries'];    

    

    if(make in accidents_make):

        accidents_make[make] = int(accidents_make[make]) + 1;

    else:

        accidents_make[make] = 1;

        

        

    if(country in accidents_country):

        accidents_country[country] = int(accidents_country[country]) + 1;

    else:

        accidents_country[country] = 1;

        

    if(event_year in accidents_yearly):

        accidents_yearly[event_year] = accidents_yearly[event_year] +  1;

    else:

        accidents_yearly[event_year] = 1;

        

        

    if(continent in accidents_continent_yearly):

        if(event_year in accidents_continent_yearly[continent]):

            accidents_continent_yearly[continent][event_year] = accidents_continent_yearly[continent][event_year] +  1;

        else:

            accidents_continent_yearly[continent][event_year] = 1;

    else:

        accidents_continent_yearly[continent] = {};

        accidents_continent_yearly[continent][event_year] = 1;        

    

    

    if(event_year in fatal_injuries_yearly):

        fatal_injuries_yearly[event_year] = fatal_injuries_yearly[event_year] +  total_fatal_injuries;

    else:

        fatal_injuries_yearly[event_year] = total_fatal_injuries;

        

        

    if(continent in fatal_injuries_continent_yearly):

        if(event_year in fatal_injuries_continent_yearly[continent]):

            fatal_injuries_continent_yearly[continent][event_year] = fatal_injuries_continent_yearly[continent][event_year] +  total_fatal_injuries;

        else:

            fatal_injuries_continent_yearly[continent][event_year] = total_fatal_injuries;

    else:

        fatal_injuries_continent_yearly[continent] = {};

        fatal_injuries_continent_yearly[continent][event_year] = total_fatal_injuries;    

                

    

''' 

get 'make' and 'country'

'''

def get_item_list_from_dict(item_dict = accidents_make, count = 10, reverse = False):



    item_sorted = sorted(item_dict.items(), key = operator.itemgetter(1), reverse = reverse);    



    # get only top 20 make

    item_count = 0;

    item_list_names = [];

    item_list = [];

    others_count = 0;

    top_item_list = [];

    top_item_list_percentage = [];

    top_item_list_and_percentage = [];

    for k, v in item_sorted:

        

        if(k == 'unknown'):

            continue;

        

        item_count = item_count + 1; 

        

        item_list_names.append(k);    

        item_list.append( v / total_accidents);

        

        top_item_list.append([k, v]);

        top_item_list_percentage.append([k, (v / total_accidents) * 100]);

        

        top_item_list_and_percentage.append([k, v, ((v / total_accidents) * 100)]);

        

        if(item_count > count):

            break;    

    

    

    row_format ="{:>25}" * (3)

    print (row_format.format("Country", "Total", "Percentage"))



    for row in top_item_list_and_percentage:

        print (row_format.format(row[0], row[1], ("%.2f" % row[2])));    







    

def show_graph(x_values, y_values, x_label = '', y_label = '', label = '', marker = '', title = ''):    

    

    plt.plot(x_values, y_values, lw = 2, marker = marker, label = label)

    

    

    # avoid exponential

    ax = plt.gca()

    ax.get_xaxis().get_major_formatter().set_useOffset(False)

    plt.xlabel(x_label);

    plt.ylabel(y_label);

    plt.title(title);

    plt.legend(loc = 'upper left');

    

    plt.show();



    

    



'''

graph of accidents every year

'''    

def show_accidents_per_year():

    

    x_values = [];

    y_values = [];

    

    accidents_yearly_sorted = sorted(accidents_yearly.items(), key = operator.itemgetter(0));

    

    for k, v in accidents_yearly_sorted:

        # ignore before 1982 as they don't have anything significant

        if(k < 1982):

            continue;

        x_values.append(k);

        y_values.append(v);

    

    show_graph(x_values, y_values, x_label = 'Year', y_label = 'No of Accidents', title = 'Accidents per Year', marker = 'o', label = 'accidents');

    

    

   

    

'''

graph of fatal injuries every year

'''    

def show_fatal_injuries_per_year():

    

    x_values = [];

    y_values = [];

    

    fatal_injuries_yearly_sorted = sorted(fatal_injuries_yearly.items(), key = operator.itemgetter(0));    

    

    for k, v in fatal_injuries_yearly_sorted:

        # ignore before 1982 as they don't have anything significant

        if(k < 1982):

            continue;

        x_values.append(k);

        y_values.append(v);    

    

    show_graph(x_values, y_values, x_label = 'Year', y_label = 'Fatal Injuries', marker = 'o', title = 'Fatal Injuries since 1982' , label = 'fatal injuries');    











'''

Compare North America and Europe Accidents

'''

def show_asia_and_europe_accidents():

    

    x_values_europe = [];

    y_values_europe = [];

    x_values_asia = [];

    y_values_asia = [];

    

    accidents_europe_yearly_sorted = sorted(accidents_continent_yearly['europe'].items(), key = operator.itemgetter(0));

    accidents_asia_yearly_sorted = sorted(accidents_continent_yearly['asia'].items(), key = operator.itemgetter(0));

    

    for k, v in accidents_europe_yearly_sorted:

        # ignore before 1982 as they don't have anything significant

        if(k < 1982):

            continue;

        x_values_europe.append(k);

        y_values_europe.append(v);

        

    for k, v in accidents_asia_yearly_sorted:

        # ignore before 1982 as they don't have anything significant

        if(k < 1982):

            continue;

        x_values_asia.append(k);

        y_values_asia.append(v);    

    

    

    plt.plot(x_values_europe, y_values_europe, label = 'Europe', lw = 2, marker = 'o');

    plt.plot(x_values_asia, y_values_asia, label = 'Asia', lw = 2, marker = 's')

            

    # avoid exponential

    ax = plt.gca()

    ax.get_xaxis().get_major_formatter().set_useOffset(False)

    plt.xlabel('Year');

    plt.ylabel('No. of Accidents');

    plt.legend(loc = 'upper left');

    plt.title('Asia vs Europe Accidents since 1980')

    

    plt.show()

    











'''

Compare North America and Europe Fatal Injuries

'''

def show_asia_and_europe_fatal_injuries():

    

    x_values_europe = [];

    y_values_europe = [];

    x_values_asia = [];

    y_values_asia = [];

    

    fatal_injuries_europe_yearly_sorted = sorted(fatal_injuries_continent_yearly['europe'].items(), key = operator.itemgetter(0));

    fatal_injuries_asia_yearly_sorted = sorted(fatal_injuries_continent_yearly['asia'].items(), key = operator.itemgetter(0));

    

    for k, v in fatal_injuries_europe_yearly_sorted:

        # ignore before 1982 as they don't have anything significant

        if(k < 1982):

            continue;

        x_values_europe.append(k);

        y_values_europe.append(v);

        

    for k, v in fatal_injuries_asia_yearly_sorted:

        # ignore before 1982 as they don't have anything significant

        if(k < 1982):

            continue;

        x_values_asia.append(k);

        y_values_asia.append(v);    

    

    

    plt.plot(x_values_europe, y_values_europe, label = 'Europe', lw = 2, marker = 'o');

    plt.plot(x_values_asia, y_values_asia, label = 'Asia', lw = 2, marker = 's')

            

    # avoid exponential

    ax = plt.gca()

    ax.get_xaxis().get_major_formatter().set_useOffset(False)

    plt.xlabel('Year');

    plt.ylabel('Total Fatal Injuries');

    plt.legend(loc = 'upper left');

    plt.title('Asia vs Europe Fatal Injuries since 1980')

    

    plt.show();   







def get_worst_10_years():

    fatal_injuries_yearly_sorted_reverse = sorted(fatal_injuries_yearly.items(), key = operator.itemgetter(1), reverse = True);

    

    #print(fatal_injuries_yearly_sorted_reverse);

    

    print("Year - Total Fatal Injuries");

    row_count = 1;

    for k, v in fatal_injuries_yearly_sorted_reverse:

        

        print(str(k) + ' - ' + str(v));

        row_count = row_count + 1;

        

        if(row_count > 10):

            break;
# get accidents country [top 20]

get_item_list_from_dict(accidents_country, 20, True);
# show total fatal injuries every year

show_fatal_injuries_per_year();
# Show accidents per year in a graph

show_accidents_per_year();
# Show Asia vs Europe Accidents

show_asia_and_europe_accidents();
# Show Asia vs Europe Fatal Injuries

show_asia_and_europe_fatal_injuries();    
# print worst 10 years by total fatal injuries

get_worst_10_years();
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt;

import operator; # for dictionary sorting by value



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

%matplotlib inline
# Import .csv file (use forward slash - windows)

athletes = pd.read_csv('../input/athletes.csv',low_memory = False);
# lowercase all values for easy handling

athletes.name = athletes.name.str.lower();

athletes.nationality = athletes.nationality.str.lower();

athletes.sex = athletes.sex.str.lower();

athletes.sport = athletes.sport.str.lower();
#country code fix

'''

    Romania   ROU    ROM

    Serbia    SRB    SCG

    

'''

athletes.loc[athletes["nationality"] == "rou", "nationality"] = 'rom';

athletes.loc[athletes["nationality"] == "srb", "nationality"] = 'scg';
# countreis and codes

countries = pd.read_csv('../input/countries.csv',low_memory = False);

countries.country = countries.country.str.lower();

countries.code = countries.code.str.lower();
def get_country(country_code):    

    try:

        return countries[countries.code == country_code].country.values[0];

    except:

        return 'unknown';

    return '';
def get_continent(country):

    

    #print('inside continent : '+country);

    

    #remove if any star at last  

    if("*"in country and country.index('*') > 0):

        country = country[:country.index('*')];

    

    if (

        country in "Burundi, Comoros, Djibouti, Eritrea, Ethiopia, Kenya, Madagascar, Malawi, Mauritius, Mayotte, Mozambique, Réunion, Rwanda, Seychelles, Somalia, South Sudan, Uganda".lower()

        or country in "United Republic of Tanzania, Zambia, Zimbabwe, Angola, Cameroon, Central African Republic, Chad, Congo, Democratic Republic of the Congo, Equatorial Guinea, Gabon".lower()

        or country in "Sao Tome and Principe, Algeria, Egypt, Libya, Morocco, Sudan, Tunisia, Western Sahara, Botswana, Lesotho, Namibia, South Africa, Swaziland, Benin, Burkina Faso, Cabo Verde".lower()

        or country in "Cote d'Ivoire, Gambia, Ghana, Guinea,Guinea-Bissau Liberia, Mali,Mauritania Niger, Nigeria,Saint Helena Senegal, Sierra Leone,Togo, Cape Verde".lower()

        or country in "Congo, Dem Rep".lower()

    ):

        return "africa";

    

    if (

        country in "Anguilla, Antigua and Barbuda, Aruba, Bahamas, Barbados, Bonaire, Sint Eustatius and Saba, British Virgin Islands, Cayman Islands, Cuba,".lower()

        or country in "Curaçao, Dominica, Dominican Republic, Grenada, Guadeloupe, Haiti, Jamaica, Martinique, Montserrat, Puerto Rico, Saint-Barthélemy, Saint Kitts and Nevis,".lower()

        or country in "Saint Lucia, Saint Martin (French part), Saint Vincent and the Grenadines, Sint Maarten (Dutch part), Trinidad and Tobago, Turks and Caicos Islands,".lower()

        or country in "United States Virgin Islands, Belize, Costa Rica, El Salvador, Guatemala, Honduras, Mexico, Nicaragua, Panama, Bermuda, Canada, Greenland,".lower()

        or country in "Saint Pierre and Miquelon, United States of America".lower()

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

        or country in "East Timor (Timor-Leste), Korea, North, Korea, South".lower()

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

        or country in "Tonga, Tuvalu, Wallis and Futuna Islands".lower()

    ):

        return "oceania";

     

    return "unknown";
medal_continents = {};

medal_continents_male = {};

medal_continents_female = {};

def get_medal_counts():

    for index, row in athletes.iterrows():

        

        country_code = row['nationality'];

        country = get_country(country_code);

        continent = get_continent(country);

        

        has_gold = 0;

        has_silver = 0;

        has_bronze = 0;

        

        has_gold = int(athletes.iloc[index]['gold']);

        has_silver = int(athletes.iloc[index]['silver']);

        has_bronze = int(athletes.iloc[index]['bronze']);

        

        total_individual_medals = has_gold + has_silver + has_bronze;      

        

        if(total_individual_medals == 0 or continent == 'unknown'):

            continue;

                

        if(continent in medal_continents):

            medal_continents[continent] = int(medal_continents[continent]) + total_individual_medals;            

        else:

            medal_continents[continent] = total_individual_medals;

            

        if(row['sex'] == 'male'):    

            if(continent in medal_continents_male):            

                medal_continents_male[continent] = int(medal_continents_male[continent]) + total_individual_medals;            

            else:

                medal_continents_male[continent] = total_individual_medals;

            

        if(row['sex'] == 'female'):    

            if(continent in medal_continents_female):

                medal_continents_female[continent] = int(medal_continents_female[continent]) + total_individual_medals;            

            else:

                medal_continents_female[continent] = total_individual_medals;    
get_medal_counts();
continents_ranking = sorted(medal_continents.items(), key = operator.itemgetter(1), reverse=True);

continents_ranking_male = sorted(medal_continents_male.items(), key = operator.itemgetter(1), reverse=True);

continents_ranking_female = sorted(medal_continents_female.items(), key = operator.itemgetter(1), reverse=True);
def make_autopct(values):

    def my_autopct(pct):

        total = sum(values)

        val = int(round(pct * total/100.0))

        return '{p:.2f}%'.format(p=pct,v=val)

    return my_autopct
colors = ['yellowgreen', 'mediumpurple', 'lightskyblue', 'lightcoral','#FEBFDC', '#FED980'];
'''

Show graph with total medals for each continent

Note: This graph ignores unknown continent

'''

def show_total_graph(ranking, current_title):            

            

    medals = [];

    names = [];    

    

    for k, v in ranking:         

        #print(k + ' - '+str(v));

        names.append(k);

        medals.append(v);    

        

    # Plot pie with total medals

    plt.pie(medals, colors = colors, labels = names, autopct = make_autopct(medals), shadow = False);

    plt.axis('equal')

    plt.title(current_title);

    plt.show();

    

    
show_total_graph(continents_ranking, 'Total Medals % by Continents');
show_total_graph(continents_ranking_male, 'Male Medals % by Continents');
show_total_graph(continents_ranking_female, 'Female Medals % by Continents');
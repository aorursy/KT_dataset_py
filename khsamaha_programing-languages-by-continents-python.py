import pandas as pd

from matplotlib import pyplot as plt



kag = pd.read_csv('../input/multipleChoiceResponses.csv',

                  encoding = "ISO-8859-1",

                 low_memory=False)

kag.head()

kag.shape
vec_Countries = ["Algeria","Angola","Benin","Botswana","Burkina","Burundi","Cameroon","Cape Verde",

               "Central African Republic","Chad","Comoros","Congo","Congo, Democratic Republic of","Djibouti",

               "Egypt","Equatorial Guinea","Eritrea","Ethiopia","Gabon","Gambia","Ghana","Guinea","Guinea-Bissau","Ivory Coast",

               "Kenya","Lesotho","Liberia","Libya","Madagascar","Malawi","Mali","Mauritania","Mauritius","Morocco",

               "Mozambique","Namibia","Niger","Nigeria","Rwanda","Sao Tomeand Principe","Senegal","Seychelles",

               "SierraLeone","Somalia","South Africa","South Sudan","Sudan","Swaziland","Tanzania","Taiwan",

               "Togo","Tunisia","Uganda","Zambia","Zimbabwe","Afghanistan","Bahrain","Bangladesh","Bhutan","Brunei",

               "Burma(Myanmar)","Cambodia","People s Republic of China","Hong Kong","EastTimor","India","Indonesia","Iran",

               "Iraq","Israel","Japan","Jordan","Kazakhstan","NorthKorea","South Korea","Kuwait","Kyrgyzstan","Laos",

               "Lebanon","Malaysia","Maldives","Mongolia","Nepal","Oman","Pakistan","Philippines","Qatar","Russia",

               "Saudi Arabia","Singapore","SriLanka","Syria","Tajikistan","Thailand","Turkey","Turkmenistan","United Arab Emirates","Uzbekistan",

               "Vietnam","Yemen","Albania","Andorra","Armenia","Austria","Azerbaijan","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia",

               "Cyprus","Czech Republic","Denmark","Estonia","Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland",

               "Italy","Latvia","Liechtenstein","Lithuania","Luxembourg","Macedonia","Malta","Moldova","Monaco","Montenegro",

               "Netherlands","Norway","Poland","Portugal","Romania","SanMarino","Serbia","Slovakia","Slovenia","Spain",

               "Sweden","Switzerland","Ukraine","United Kingdom","Vatican City","Antiguaand Barbuda","Bahamas","Barbados",

               "Belize","Canada","CostaRica","Cuba","Dominica","Dominican Republic","ElSalvador","Grenada",

               "Guatemala","Haiti","Honduras","Jamaica","Mexico","Nicaragua","Panama","Saint Kittsand Nevis",

               "Saint Lucia","Saint Vincentand the Grenadines","Trinidadand Tobago","United States","Australia","Fiji",

               "Kiribati","Marshall Islands","Micronesia","Nauru","New Zealand","Palau","Papua New Guinea","Samoa",

               "Solomon Islands","Tonga","Tuvalu","Vanuatu","Argentina","Bolivia","Brazil","Chile","Colombia","Ecuador",

               "Guyana","Paraguay","Peru","Suriname","Uruguay","Venezuela","Other"]



vec_Continents = ["Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa",

                "Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa,","Africa",

                "Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa",

                "Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa","Africa",

                "Africa","Asia","Africa","Africa","Africa","Africa",

                "Africa","Asia","Asia","Asia","Asia","Asia","Asia","Asia","Asia","Asia","Asia","Asia",

                "Asia","Asia","Asia","Asia","Asia","Asia","Asia","Asia","Asia","Asia","Asia","Asia",

                "Asia","Asia","Asia","Asia","Asia","Asia","Asia","Asia","Asia","Asia","Asia","Asia",

                "Asia","Asia","Asia","Asia","Asia","Asia","Asia","Asia","Asia","Asia",

                "Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe",

                "Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe",

                "Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe",

                "Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe","Europe",

                "North America","North America","North America","North America","North America","North America","North America",

                "North America","North America","North America","North America","North America","North America","North America",

                "North America","North America","North America","North America","North America","North America","North America",

                "North America","North America",

                "Oceania","Oceania","Oceania","Oceania","Oceania","Oceania","Oceania","Oceania","Oceania","Oceania","Oceania",

                "Oceania","Oceania","Oceania",

                "South America","South America","South America","South America","South America","South America","South America",

                "South America","South America","South America","South America","South America","Other"]
new_kag = kag.loc[:,["GenderSelect","Age","Country","LanguageRecommendationSelect"] ]

new_kag["Country"] = new_kag["Country"].str.replace("People 's Republic of China", "People s Republic of China")

new_kag["Country"] = new_kag["Country"].str.replace("Republic of China", "People s Republic of China")

new_kag.head()
Countries_continents = pd.DataFrame({"Country": vec_Countries,"Continent": vec_Continents})

Countries_continents.head()
new_kag_filter = new_kag[(new_kag.Age.notnull()) & (new_kag.Country.notnull()) &

                         (new_kag.LanguageRecommendationSelect.notnull()) & 

                         (new_kag.Age >= 14) &

                         (new_kag.Age <80) &

                         (new_kag["LanguageRecommendationSelect"].isin(["R","Python","Java","SQL","C/C++/C#"]))]

new_kag_filter.head()

new_kag.insert(3, 'Continent', new_kag['Country'].map(Countries_continents.set_index('Country')['Continent']))

new_kag.LanguageRecommendationSelect.unique()

new_kag.head()
new_kag_filter = new_kag[(new_kag.Age.notnull()) & (new_kag.Country.notnull()) &

                         (new_kag.LanguageRecommendationSelect.notnull()) & 

                         (new_kag.Age >= 14) &

                         (new_kag.Age <80) &

                         (new_kag["LanguageRecommendationSelect"].isin(["R","Python","Java","SQL","C/C++/C#"]))]

new_kag_filter.head()



count_conti = new_kag_filter.groupby("LanguageRecommendationSelect").Age.mean()

count_conti.plot.bar()

plt.show()

count_conti = new_kag_filter.groupby(["Continent", "LanguageRecommendationSelect"]).LanguageRecommendationSelect.count()

count_conti.plot.bar()

plt.tick_params(axis='x', labelsize=9)

plt.rcParams["figure.figsize"] = (10,10)

plt.show()
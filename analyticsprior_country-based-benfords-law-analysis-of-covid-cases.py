import warnings
warnings.filterwarnings("ignore")
#!pip install --upgrade pip
!pip install benford_py
!pip install folium
import benford as bf
import pandas as pd
import folium
# European Centre for Disease Prevention and Control data on the geographic distribution of COVID-19 cases worldwide
covid_data_all = pd.read_csv("https://opendata.ecdc.europa.eu/covid19/casedistribution/csv", dayfirst=True, parse_dates=True)
# Here we change the display option of pandas. 
pd.options.display.max_rows = 200
pd.options.display.max_columns = None
covid_data_all
covid_data_all.rename(columns={'dateRep':'date','countriesAndTerritories':'country', 'countryterritoryCode':'country_code', 'popData2019':'population_2019', 'continentExp':'continent', 'Cumulative_number_for_14_days_of_COVID-19_cases_per_100000':'cum_case_perht'}, inplace=True)
covid_data_all.columns.sort_values()
covid_data_all.head()
covid_data_all.tail()
covid_data_all.describe()
covid_data_all.shape
covid_data_all.info()
# At least case data should be double digit
covid_cases=covid_data_all[covid_data_all.cases >=10].copy() 
#Replacing all underscores with a space (country names changed) 
covid_cases.replace(r'_', ' ', regex=True, inplace = True)
covid_cases
# Group data by country and add sum and count of covid-19 cases
covid_cases_country_count=covid_cases.groupby(['country'], as_index=False)['cases'].agg(['sum','count'])
covid_cases_country_count
# In the chi-square test, it is required that the expected frequency of each cell is at least 5. Thus, minimum n is 109
covid_cases_country_count=covid_cases_country_count[covid_cases_country_count['count']>=109].copy() 
covid_cases_country_count
# Make country names list
covid_cases_country_list=covid_cases_country_count.index.values.tolist()
covid_cases_country_list
def benford_first_digit(country):
    
    covid_cases_test=covid_cases[covid_cases['country']==country].copy()        
    data=list(covid_cases_test.cases)

    for i in range(len(data)):
        while data[i]>=10:
            data[i]=data[i]/10

    first_digits=[int(x) for x in sorted(data)]
    unique=(set(first_digits))
    data_count=[]
    digits = [1, 2, 3, 4, 5, 6, 7, 8, 9]
              
    for i in digits:
        count=first_digits.count(i)
        data_count.append(count)
           
    total_count=sum(data_count)
    data_percentage=[(i/total_count)*100 for i in data_count]
    benford = [30.103, 17.6091, 12.4939, 9.691, 7.91812, 6.69468, 5.79919, 5.11525, 4.57575]
    expected = [(i * total_count / 100) for i in benford]
    chi_square_index = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    chi_square = [((data_count[i]-expected[i])**2)/expected[i] for i in chi_square_index]
    chi_square_sum = sum(chi_square)
                
    return  chi_square_sum
benford_covid_cases_results = []
for i in covid_cases_country_list:
    if benford_first_digit(i)>=145:
        conformity = "(g) Severe Non-Conformity"
    elif benford_first_digit(i)>=117:
        conformity = "(f) Very High Non-Conformity"
    elif benford_first_digit(i)>=89:
        conformity = "(e) High Non-Conformity"    
    elif benford_first_digit(i)>=61:
        conformity = "(d) Moderate Non-Conformity" 
    elif benford_first_digit(i)>=33:
        conformity = "(c) Low Non-Conformity"
    elif benford_first_digit(i)>=15.507:
        conformity = "(b) Minor Non-Conformity" 
    else:
        conformity = "(a) Conformity"    
    benford_covid_cases_results.append((i, benford_first_digit(i),"15.507", conformity))                                    
benford_covid_cases_results=pd.DataFrame(benford_covid_cases_results, columns=('country', 'sum_of_chi_square', 'critical_chi_square', 'conformity'))
benford_covid_cases_results
# Here we add covid-19 case sum data.
benford_covid_cases_results=pd.merge(benford_covid_cases_results, covid_cases_country_count[['sum', 'count']], on='country', how='left')
# Here we change column names.
benford_covid_cases_results.rename(columns={'sum':'sum_of_covid_19_cases','count':'count_of_covid_19_reporting_dates'}, inplace=True)
# Sorting data by country name
benford_covid_cases_results
# Sorting data by total covid-19 cases
benford_covid_cases_results.sort_values(by='sum_of_covid_19_cases', ascending=False, ignore_index=True)
# Sorting data by sum of chi-squares
benford_covid_cases_results.sort_values(by='sum_of_chi_square', ascending=False, ignore_index=True)
# Sorting data by conformity and total covid-19 cases
benford_covid_cases_results.sort_values(['conformity','sum_of_covid_19_cases'], ascending=[True, False], ignore_index=True)
url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
world_geo = f'{url}/world-countries.json'

# create a plain world map
world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')
# generate choropleth map using the sum of chi-square of each country.
world_map.choropleth(
    geo_data=world_geo,
    data=benford_covid_cases_results,
    columns=['country', 'sum_of_chi_square'],
    key_on='feature.properties.name',
    fill_color='YlOrRd', 
    fill_opacity=0.9, 
    line_opacity=0.2,
    nan_fill_color='white',
    name="COUNTRY BASED BENFORDS FIRST DIGIT ANALYSIS OF COVID-19 CASES DATASET"
)

# display map
world_map
for i in covid_cases_country_list:
    
    covid_cases_test=covid_cases[covid_cases['country']==i].copy()
    
    # Country based Benford's Law first digit analysis and 
    # assessing conformity with Chi-Square, Kolmogorov-Smirnov, Mean Absolute Deviation (MAD) and Z Scores
    print(" #################  COUNTRY: ", i ," #################")
    print(" #################  BENFORD'S FIRST DIGIT ANALYSIS OF COVID-19 CASES DATASET AND CONFORMITY TESTS: #################")
    first_digit_cases = bf.first_digits(covid_cases_test.cases, decimals=8, digs=1, show_plot=True,  MAD=True, MSE=True, confidence=95, KS=True, chi_square=True)
    
    # Country based Benford's Law first two digits analysis and 
    # assessing conformity with Chi-Square, Kolmogorov-Smirnov, Mean Absolute Deviation (MAD) and Z Scores
    print("\n")
    print(" #################  COUNTRY: ", i ," #################")
    print(" #################  BENFORD'S FIRST TWO DIGIT ANALYSIS OF COVID-19 CASES DATASET AND CONFORMITY TESTS: #################")
    first_two_digit_cases = bf.first_digits(covid_cases_test.cases, decimals=8, digs=2, show_plot=True,  MAD=True, MSE=True, confidence=95, KS=True, chi_square=True)
    
    # Country based Benford's Law second digits analysis and 
    # assessing conformity with Chi-Square, Kolmogorov-Smirnov, Mean Absolute Deviation (MAD) and Z Scores
    print("\n")
    print(" #################  COUNTRY: ", i ," #################")
    print(" #################  BENFORD'S SECOND DIGIT ANALYSIS OF COVID-19 CASES DATASET AND CONFORMITY TESTS: #################")
    second_digit_cases = bf.second_digit(covid_cases_test.cases, decimals=8, show_plot=True,  MAD=True, MSE=True, confidence=95, KS=True, chi_square=True)
    
    print(" ____________________________________________________________________________________________________________________")
    print("\n","\n","\n")

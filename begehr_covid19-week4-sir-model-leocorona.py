from sklearn.metrics import mean_squared_log_error

import warnings

warnings.filterwarnings('ignore')

import pandas as pd

from sklearn import preprocessing

import numpy as np

from scipy import integrate, optimize

import math



predictions_total = []

actual_total = []

val_loss_dict = {}



val_info_dict = {}

predictions_dict = {}

actuals_dict = {}

colors_dict = {}

loss_dict = {}

train_start = 0

train_end = 0

val_start = 0

val_end = 0

test_start = 0

test_end = 0

modes = ["Confirmed Cases", "Fatalities"]

method = "SIR"

dynamic_start_day = False
test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv", parse_dates=["Date"])

train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

submission = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

all_data = train.copy()

# Create date columns

all_data['Date'] = pd.to_datetime(all_data['Date'])

le = preprocessing.LabelEncoder()

all_data['Day_num'] = le.fit_transform(all_data.Date)

all_data['Day'] = all_data['Date'].dt.day

all_data['Month'] = all_data['Date'].dt.month

all_data['Year'] = all_data['Date'].dt.year
# Load countries data file

world_population = pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv")



# Select desired columns and rename some of them

world_population = world_population[['Country (or dependency)', 'Population (2020)', 'Density (P/Km²)', 'Land Area (Km²)', 'Med. Age', 'Urban Pop %']]

world_population.columns = ['Country (or dependency)', 'Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']



# Replace United States by US

world_population.loc[world_population['Country (or dependency)']=='United States', 'Country (or dependency)'] = 'US'



# Remove the % character from Urban Pop values

world_population['Urban Pop'] = world_population['Urban Pop'].str.rstrip('%')



# Replace Urban Pop and Med Age "N.A" by their respective modes, then transform to int

world_population.loc[world_population['Urban Pop']=='N.A.', 'Urban Pop'] = int(world_population.loc[world_population['Urban Pop']!='N.A.', 'Urban Pop'].mode()[0])

world_population['Urban Pop'] = world_population['Urban Pop'].astype('int16')

world_population.loc[world_population['Med Age']=='N.A.', 'Med Age'] = int(world_population.loc[world_population['Med Age']!='N.A.', 'Med Age'].mode()[0])

world_population['Med Age'] = world_population['Med Age'].astype('int16')



print("Cleaned country details dataset")





# Now join the dataset to our previous DataFrame and clean missings (not match in left join)- label encode cities

print("Joined dataset")

all_data = all_data.merge(world_population, left_on='Country_Region', right_on='Country (or dependency)', how='left')

all_data[['Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']] = all_data[['Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']].fillna(0)





print("Encoded dataset")

# Label encode countries and provinces. Save dictionary for exploration purposes

all_data.drop('Country (or dependency)', inplace=True, axis=1)

all_data['Country_Region'] = le.fit_transform(all_data['Country_Region'])



number_c = all_data['Country_Region']

countries = le.inverse_transform(all_data['Country_Region'])

country_dict = dict(zip(countries, number_c)) 

all_data['Province_State'].fillna("None", inplace=True)

all_data['Province_State'] = le.fit_transform(all_data['Province_State'])

number_p = all_data['Province_State']

province = le.inverse_transform(all_data['Province_State'])

province_dict = dict(zip(province, number_p)) 
class SIR:

    def __init__(self, beta=0, gamma=0, fix_gamma=False):

        self.beta = beta

        self.gamma = gamma

        self.infected_t0 = 0

        self.fitted_on = np.array([])

        self.fix_gamma = fix_gamma

        self.fitted = False

        

    def ode(self, y, x, beta, gamma):

        '''Defines the ODE that governs the SIRs behaviour'''

        dSdt = -beta * y[0] * y[1]

        dRdt = gamma * y[1]

        dIdt = -(dSdt + dRdt)

        return dSdt, dIdt, dRdt

    

    def solve_ode(self, x, beta, gamma):

        '''Solves the resulting ODE to get predictions for each time step'''

        return np.cumsum(integrate.odeint(self.ode, (1-self.infected_t0, self.infected_t0, 0.0), x, args=(beta, gamma))[:,1])

    

    def solve_ode_fixed(self, x, beta):

        '''Solves the resulting ODE to get predictions for each time step'''

        return np.cumsum(integrate.odeint(self.ode, (1-self.infected_t0, self.infected_t0, 0.0), x, args=(beta, self.gamma))[:,1])

    

    def describe(self):

        assert self.fitted, "You need to fit the model before describing it!"

        print("Beta: ", self.beta)

        print("Gamma: ", self.gamma)

        print("At t=0: ", self.infected_t0)

        

        plt.plot(range(1,len(self.fitted_on)+1), self.fitted_on, "x", label='Actual')

        plt.plot(range(1,len(self.fitted_on)+1), self.predict(len(self.fitted_on)), label='Prediction')

        plt.title("Fit of SIR model to actual")

        plt.ylabel("% of Population")

        plt.xlabel("Days")

        plt.legend()

        plt.show()

    

    def evaluate(self, y_test):

        assert self.fitted, "You need to fit the model before evaluating it!"

        print("Beta: ", self.beta)

        print("Gamma: ", self.gamma)

        print("At t=0: ", self.infected_t0)

        

        y_train = self.fitted_on

        l_train = len(self.fitted_on)

        l_test = len(y_test)

        l_all = l_train + l_test

        

        plt.plot(range(1, l_train + 1), y_train, "x", label='Actual Train')

        plt.plot(range(1 + l_train, l_all + 1), y_test, "x", label='Actual Test')

        plt.plot(range(1, l_all + 1), self.predict(l_all), label='Prediction')

        plt.title("Fit of SIR model to actual")

        plt.ylabel("% of Population")

        plt.xlabel("Days")

        plt.legend()

        plt.show()

    

    def fit(self, y):

        '''Fits the parameters to the data, assuming the first data point is the start of the outbreak'''

        if len(y) == 1: y = np.array([0, y[0]]) # SIR needs at least 2 datapoints to fit

        self.infected_t0 = y[0]

        x = np.array(range(1,len(y)+1), dtype=float)

        self.fitted_on = y

        if(self.fix_gamma):

            popt, _ = optimize.curve_fit(self.solve_ode_fixed, x, y)

            self.beta = popt[0]

        else:

            popt, _ = optimize.curve_fit(self.solve_ode, x, y, maxfev=1000)

            self.beta = popt[0]

            self.gamma = popt[1]

        self.fitted = True

        

    def predict(self ,length):

        '''Returns the predicted cumulated cases at each time step, assuming outbreak starts at t=0'''

        #assert self.fitted, "You need to fit the model before predicting!"

        return self.solve_ode(range(1, length+1), self.beta, self.gamma)
unknown_countries = []

hardcoded_countries = {

    "Korea, South": 51269000,

    "Diamond Princess": 3711,

    "Taiwan*": 23800000,

    "Saint Vincent and the Grenadines": 109897,

    "Congo (Brazzaville)":5261000,

    "Congo (Kinshasa)":81340000,

    "Cote d'Ivoire":24300000,

    "Czechia": 10650000,

    "Saint Kitts and Nevis": 55345,

    "Burma": 53370000,

    "Kosovo": 1831000,

    "MS Zaandam": 1432, # cruise ship

    "West Bank and Gaza": 4685,

    "Sao Tome and Principe": 204327,

}

hardcoded_province = {

    "Saint Pierre and Miquelon": 5888,

    "Bonaire, Sint Eustatius and Saba": 25157,

    "Falkland Islands (Malvinas)": 2840,

}

state_populations= pd.read_csv("../input/covid19-forecasting-metadata/region_metadata.csv")



def get_population(country_name, province_name=None):

    if province_name:

        pop = state_populations[state_populations['Province_State']==province_name]['population']

        if len(pop)==0:

            if province_name in hardcoded_province:

                return hardcoded_province[province_name]

            else:

                print(f"Warning: We have no province population data at the moment. Instead of data for {province_name}, using data for {country_name}")

        else:

            return pop.iloc[0]

    

    if country_name in hardcoded_countries:

        return hardcoded_countries[country_name]

    

    pop = all_data[all_data["Country_Region"] == country_dict[country_name]].iloc[0]["Population (2020)"]

    if not pop:

        print(f"population of {country_name} unknown")

        pop = 100

        unknown_countries.append(country_name)

    

    return pop
country_name = 'China'

all_data[all_data["Country_Region"] == country_dict[country_name]].iloc[0]["Population (2020)"]
country_name = 'Hubei'

all_data[all_data["Province_State"] == province_dict[country_name]].iloc[0]["Population (2020)"]
def get_country_data(country_name, province_name=None, train_split_factor=1.0):

  if province_name:

    confirmed_total_date_country = train[(train['Country_Region']==country_name) & (train['Province_State']==province_name)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

    fatalities_total_date_country = train[(train['Country_Region']==country_name) & (train['Province_State']==province_name)].groupby(['Date']).agg({'Fatalities':['sum']})

    total_date_country = confirmed_total_date_country.join(fatalities_total_date_country)



    cases = total_date_country.ConfirmedCases['sum'].values

    cases_normalized = total_date_country.ConfirmedCases['sum'].values / get_population(country_name, province_name)

    fatalities_normalized = total_date_country.Fatalities['sum'].values / get_population(country_name, province_name)



    cases_final = cases_normalized[np.argmax(cases>0):]

    fatalities_final = fatalities_normalized[np.argmax(fatalities_normalized>0):]



    cases_length = len(cases_final)

    fat_length = len(fatalities_final)

    cases_split = math.floor(cases_length * train_split_factor)

    fat_split = math.floor(fat_length * train_split_factor)

  else:

    confirmed_total_date_country = train[train['Country_Region']==country_name].groupby(['Date']).agg({'ConfirmedCases':['sum']})

    fatalities_total_date_country = train[train['Country_Region']==country_name].groupby(['Date']).agg({'Fatalities':['sum']})

    total_date_country = confirmed_total_date_country.join(fatalities_total_date_country)



    cases = total_date_country.ConfirmedCases['sum'].values

    cases_normalized = cases / get_population(country_name, province_name)

    fatalities_normalized = total_date_country.Fatalities['sum'].values / get_population(country_name, province_name)



    cases_final = cases_normalized[np.argmax(cases>0):]

    fatalities_final = fatalities_normalized[np.argmax(fatalities_normalized>0):]



    cases_length = len(cases_final)

    fat_length = len(fatalities_final)

    cases_split = math.floor(cases_length * train_split_factor)

    fat_split = math.floor(fat_length * train_split_factor)

    

  return cases_final, fatalities_final, cases_split, fat_split, cases_length, fat_length
import matplotlib.pyplot as plt

def visualize(val_loss_dict, val_info_dict, start=0, end=150):

  fig = plt.figure(figsize=(10,2))

  ax = fig.add_axes([0,0,1,1])



  loss_sorted = sorted(val_loss_dict.items(), key=lambda x: x[1], reverse=True)

  print(loss_sorted[10:20])

  losses = [x[1] for x in loss_sorted[start:end]]

  countries = [x[0] for x in loss_sorted[start:end]]

  colors = [val_info_dict[x]["Color"] for x in countries]

  ax.bar(countries, losses, color=colors)
def visualize_country(country_name, val_info_dict=val_info_dict):

  info = val_info_dict[country_name]

  cases_actual = info["Cases Actual"]

  cases_predicted = info["Cases Predicted"]

  cases_split = info["Case Split"]

  fat_actual = info["Fatalities Actual"]

  fat_predicted = info["Fatalities Predicted"]

  fat_split = info["Fatality Split"]

  

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,15))



  ax1.plot(cases_actual, 'o')

  ax1.plot(cases_predicted)

  ax1.axvline(x=cases_split, color='gray', linestyle='--')

  ax1.set_title("Fit of SIR model to global infected cases")

    

  ax2.plot(fat_actual, 'o')

  ax2.plot(fat_predicted)

  ax2.axvline(x=fat_split, color='gray', linestyle='--')

  ax2.set_title("Fit of SIR model to global fatalities")

  

  plt.show()
def train_val_country(country_name, train_split_factor=1.0):

    cases, fatalities, case_split, fat_split, case_length, fat_length = get_country_data(country_name, train_split_factor=train_split_factor)

    cases_train = cases[0:case_split]

    cases_test = cases[case_split:]

    fat_train = fatalities[0:fat_split]

    fat_test = fatalities[fat_split:]

    

    case_model = SIR()

    case_model.fit(cases_train)

    fat_model = SIR()

    fat_model.fit(fat_train)

    

    cases_pred_all = case_model.predict(len(cases_train) + len(cases_test))

    cases_pred_train = cases_pred_all[:case_split]

    cases_pred_test = cases_pred_all[case_split:]

    fat_pred_all = fat_model.predict(len(fat_train) + len(fat_test))

    fat_pred_train = fat_pred_all[:fat_split]

    fat_pred_test = fat_pred_all[fat_split:]

    

    if(sum(cases_test) > sum(cases_pred_test)):

      color = "red"

    else:

      color = "blue"

    

    cases_train_val_loss = np.sqrt(mean_squared_log_error(cases_train, cases_pred_train)) if (len(cases_train) > 0) else 0

    fat_train_val_loss = np.sqrt(mean_squared_log_error(fat_train, fat_pred_train)) if (len(fat_train) > 0) else 0

    cases_test_val_loss = np.sqrt(mean_squared_log_error(cases_test, cases_pred_test)) if (len(cases_test) > 0) else 0

    fat_test_val_loss = np.sqrt(mean_squared_log_error(fat_test, fat_pred_test)) if (len(fat_test) > 0) else 0

    #print(f"Val Loss for {country_name}: {val_loss}")

    #print(f"Sum actual: {sum(cases_test)} Sum predicted: {sum(cases_pred_val)}")

    val_loss_dict[country_name] = cases_test_val_loss

    results_dict =  {

        "Country": country_name,

        "Province": float('nan'),

        "Case Model": case_model,

        "Fatality Model": fat_model,

        "Color": color,

        "Cases Predicted": cases_pred_all,

        "Cases Actual": cases,

        "Fatalities Predicted": fat_pred_all,

        "Fatalities Actual": fatalities,

        "Cases Loss Train": cases_train_val_loss,

        "Fatality Loss Train": fat_train_val_loss,

        "Cases Loss Test": cases_test_val_loss,

        "Fatality Loss Test": fat_test_val_loss,

        "Case Split": case_split,

        "Fatality Split": fat_split,

        "Case length": case_length,

        "Fatality length": fat_length

    }

    return results_dict



def train_val_province(country_name, province_name, train_split_factor=1.0):

    cases, fatalities, case_split, fat_split, case_length, fat_length = get_country_data(country_name, province_name, train_split_factor=train_split_factor)

    cases_train = cases[0:case_split]

    cases_test = cases[case_split:]

    fat_train = fatalities[0:fat_split]

    fat_test = fatalities[fat_split:]

    

    case_model = SIR()

    case_model.fit(cases_train)

    fat_model = SIR()

    fat_model.fit(fat_train)

      

    cases_pred_all = case_model.predict(len(cases_train) + len(cases_test))

    cases_pred_train = cases_pred_all[:case_split]

    cases_pred_test = cases_pred_all[case_split:]

    fat_pred_all = fat_model.predict(len(fat_train) + len(fat_test))

    fat_pred_train = fat_pred_all[:fat_split]

    fat_pred_test = fat_pred_all[fat_split:]

    

    if(sum(cases_test) > sum(cases_pred_test)):

      color = "red"

    else:

      color = "blue"



    cases_train_val_loss = np.sqrt(mean_squared_log_error(cases_train, cases_pred_train)) if (len(cases_train) > 0) else 0

    fat_train_val_loss = np.sqrt(mean_squared_log_error(fat_train, fat_pred_train)) if (len(fat_train) > 0) else 0

    cases_test_val_loss = np.sqrt(mean_squared_log_error(cases_test, cases_pred_test)) if (len(cases_test) > 0) else 0

    fat_test_val_loss = np.sqrt(mean_squared_log_error(fat_test, fat_pred_test)) if (len(fat_test) > 0) else 0

    #print(f"Val Loss for {country_name}: {val_loss}")

    #print(f"Sum actual: {sum(cases_test)} Sum predicted: {sum(cases_pred_val)}")

    val_loss_dict[province_name] = cases_test_val_loss

    results_dict = {

        "Country": country_name,

        "Province": province_name,

        "Case Model": case_model,

        "Fatality Model": fat_model,

        "Color": color,

        "Cases Predicted": cases_pred_all,

        "Cases Actual": cases,

        "Fatalities Predicted": fat_pred_all,

        "Fatalities Actual": fatalities,

        "Cases Loss Train": cases_train_val_loss,

        "Fatality Loss Train": fat_train_val_loss,

        "Cases Loss Test": cases_test_val_loss,

        "Fatality Loss Test": fat_test_val_loss,

        "Case Split": case_split,

        "Fatality Split": fat_split,

        "Case length": case_length,

        "Fatality length": fat_length

    }

    return results_dict
country_and_provinces = {}

only_provinces = {}

only_country = []

for country in test['Country_Region'].unique():

  provinces = test[test['Country_Region']==country]['Province_State'].unique()

  

  if len(provinces)>1:

    contains_nan = False

    for province in provinces:

      if type(province) == float:

        contains_nan = True

    if contains_nan:

      country_and_provinces[country] = provinces

    else:

      only_provinces[country] = provinces

  else:

    only_country.append(country)

from tqdm import tqdm



train_split_factor = 0.9



for country in tqdm(train['Country_Region'].unique()):

    #If we only need to predict for the provinces, not for the whole country

    if country in only_provinces:

        for province in only_provinces[country]:

            val_info_dict[province] = train_val_province(country, province, train_split_factor=train_split_factor)

    

    #If we need to predict for the provinces and for the whole country

    elif country in country_and_provinces:

        for province in country_and_provinces[country]:

            #For the 'nan' province value: Make predictions for the whole country

            if type(province) == float:

                val_info_dict[country] = train_val_country(country, train_split_factor=train_split_factor)

            else:

                val_info_dict[province] = train_val_province(country, province, train_split_factor=train_split_factor)

    

    #If we don't have any provinces for this country

    elif country in only_country:

        val_info_dict[country] = train_val_country(country, train_split_factor=train_split_factor)
val_loss_dict
# losses

visualize(val_loss_dict, val_info_dict, end=20)
def evaluate(name, val_info_dict=val_info_dict):

    info = val_info_dict[name]

    

    case_split = info["Case Split"]

    fat_split = info["Fatality Split"]

    cases_test = info["Cases Actual"][case_split:]

    fat_test = info["Fatalities Actual"][fat_split:]

    

    case_model = info["Case Model"]

    fat_model = info["Fatality Model"]

    

    print(name)

    print("Confirmed Cases:")

    print("  Loss Train: ", info["Cases Loss Train"])

    print("  Loss Test: ", info["Cases Loss Test"])

    display(case_model.evaluate(cases_test))

    print("Fatalities:")

    print("  Loss Train: ", info["Fatality Loss Train"])

    print("  Loss Test: ", info["Fatality Loss Test"])

    display(fat_model.evaluate(fat_test))
evaluate("Germany")
evaluate("Spain")
evaluate("Hubei")
evaluate("Italy")
evaluate("New York")
evaluate("India")
evaluate("France")
# submission date range: 02Apr20-14May20

pd_daterange_submission = pd.date_range("02Apr20", "14May20") #TODO get from test dataset: min/max of Date

length_submission = len(pd_daterange_submission)



def make_submission(val_info_dict=val_info_dict, name="submission"):

  # generate submission frames for all items in val_info_dict

  frames = []

  for attr, item in val_info_dict.items():

    country = item["Country"]

    province = item["Province"]

    case_length = item["Case length"]

    fat_length = item["Fatality length"]

    case_model = item["Case Model"]

    fat_model = item["Fatality Model"]



    if(type(province)==float):

        pop = get_population(country)

    else:

        pop = get_population(country, province)

        

    case_preds = pop * case_model.predict(case_length + length_submission)[case_length:]

    fat_preds = pop * fat_model.predict(fat_length + length_submission)[fat_length:]



    frames.append(pd.DataFrame({

        "Country_Region": country,

        "Province_State": province,

        "Date": pd_daterange_submission,

        "ConfirmedCases": case_preds,

        "Fatalities": fat_preds

        })

    )

  

  # concat sub frames and prepare for mergeing with test to get ForecastId

  submission_data = pd.concat(frames)

  submission = test.copy()



  index = ["id", "Date"]

  submission["id"] = submission["Country_Region"].astype(str) + "_" + submission["Province_State"].astype(str)

  submission = submission[["id", "Date", "ForecastId"]].set_index(index)



  submission_data["id"] = submission_data["Country_Region"].astype(str) + "_" + submission_data["Province_State"].astype(str)

  submission_data = submission_data[["id", "Date", "ConfirmedCases", "Fatalities"]].set_index(index)



  # merge w/ ForecastId and extract submission columns

  submission = submission.join(submission_data)

  submission = submission[["ForecastId", "ConfirmedCases", "Fatalities"]]



  # fillna (China)

  submission = submission.fillna(1)

    

  # write to csv

  submission.to_csv(name + ".csv", index=False)



  print("submission saved to csv.")

  

make_submission()
class SIRT:

    def __init__(self, gamma=0, a=0, b=0, c=0, d=0, fix_gamma=False):

        self.gamma = gamma

        self.a = a

        self.b = b

        self.c = c

        self.d = d

        self.infected_t0 = 0

        self.fitted_on = np.array([])

        self.fix_gamma = fix_gamma

        self.fitted = False

        

    def ode(self, y, timestep, c, d, gamma):

        '''Defines the ODE that governs the SIRs behaviour'''

        beta = c * timestep + d

        

        dSdt = -beta * y[0] * y[1]

        dRdt = gamma * y[1]

        dIdt = -(dSdt + dRdt)

        return dSdt, dIdt, dRdt

    

    def solve_ode(self, x, c, d, gamma):

        '''Solves the resulting ODE to get predictions for each time step'''

        return np.cumsum(integrate.odeint(self.ode, (1-self.infected_t0, self.infected_t0, 0.0), x, args=(c, d, gamma))[:,1])

    

    def solve_ode_fixed(self, x, beta):

        '''Solves the resulting ODE to get predictions for each time step'''

        return np.cumsum(integrate.odeint(self.ode, (1-self.infected_t0, self.infected_t0, 0.0), x, args=(beta, self.gamma))[:,1])

    

    def describe(self):

        assert self.fitted, "You need to fit the model before describing it!"

        print("c: ", self.c)

        print("d: ", self.d)

        print("Gamma: ", self.gamma)

        print("Infected at t=0: ", self.infected_t0)

        

        plt.plot(range(1,len(self.fitted_on)+1), self.fitted_on, "x", label='Actual')

        plt.plot(range(1,len(self.fitted_on)+1), self.predict(len(self.fitted_on)), label='Prediction')

        plt.title("Fit of SIR model to global infected cases")

        plt.ylabel("Population infected")

        plt.xlabel("Days")

        plt.legend()

        plt.show()

    

    def fit(self, y):

        '''Fits the parameters to the data, assuming the first data point is the start of the outbreak'''

        self.infected_t0 = y[0]

        x = np.array(range(1,len(y)+1), dtype=float)

        self.fitted_on = y

        if(self.fix_gamma):

            popt, _ = optimize.curve_fit(self.solve_ode_fixed, x, y)

            self.beta = popt[0]

        else:

            popt, _ = optimize.curve_fit(self.solve_ode, x, y)

            self.c = popt[0]

            self.d = popt[1]

            self.gamma = popt[2]

        self.fitted = True

        

    def predict(self ,length):

        '''Returns the predicted cumulated cases at each time step, assuming outbreak starts at t=0'''

        #assert self.fitted, "You need to fit the model before predicting!"

        return self.solve_ode(range(1, length+1), self.c, self.d, self.gamma)
measures = pd.read_csv("../input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv")

measures["Keywords"].fillna(value="-", inplace=True)

measures["Country"] = measures["Country"].str.replace('South Korea', 'Korea, South', regex=True)

measures["Country"] = measures["Country"].str.replace('US:Georgia', 'US', regex=True)

measures["Country"] = measures["Country"].str.replace('US: Illinois', 'US', regex=True)

measures["Country"] = measures["Country"].str.replace('US:Maryland', 'US', regex=True)



measures = measures[measures["Country"] != "Vatican City"]

measures = measures[measures["Country"] != "Hong Kong"]



def get_measures(measure_name):

    

    took_measure = measures[measures["Keywords"].str.contains("distancing")]

    output = pd.DataFrame(data=0,

                          columns=train['Country_Region'].unique(),

                          index=pd.date_range("02.01.2020", "03.01.2020"))

    

    print(took_measure)

    

    for index, row in took_measure.iterrows():

        output[row["Country"]][pd.to_datetime(row["Date Start"]):] = 1

    return output

                               

#get_measures("distancing")["Italy"]
model = SIRT()

c, _, case_split, _, case_length, _ = get_country_data("Spain")

model.fit(c[:case_split])

model.describe()
get_country_data("Spain")
for attr, item in val_info_dict.items():

    country = item["Country"]

    province = item["Province"]

    case_length = item["Case length"]

    fat_length = item["Fatality length"]

    case_model = item["Case Model"]

    fat_model = item["Fatality Model"]
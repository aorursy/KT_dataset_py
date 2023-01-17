import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.display import display

from collections import namedtuple





pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)



np.seterr(divide='ignore', invalid='ignore')  # :/



def to_key(row):

    return (row['Country/Region'], row['Province/State'])





class PopulationData:

    def __init__(self):

        pop_info = pd.read_csv('../input/covid19-population-data/population_data.csv')

        country_pop = pop_info.query('Type == "Country/Region"')

        province_pop = pop_info.query('Type == "Province/State"')

        self.country_lookup = dict(zip(country_pop['Name'], country_pop['Population']))

        self.province_lookup = dict(zip(province_pop['Name'], province_pop['Population']))



    def __call__(self, row):

        key = to_key(row)

        population = self.country_lookup.get(key[0])

        if population is None:

            population = self.province_lookup[key[1]]

        return population





class HotCityDensity:

    def __init__(self):

        # Use population density

        population_density = pd.read_csv('../input/covid19highestcitypopulationdensity/population_density.csv')

        key_cols = ["Country/Region", "Province/State"]

        self.density_map = {(i, j): k for i, j, k

                            in population_density[key_cols + ["density"]].values}

        self.dencity_map = {(i, j): k for i, j, k

                            in population_density[key_cols + ["mostdensecity"]].values}



    def density(self, row):

        return self.density_map[to_key(row)]



    def dencity(self, row):

        return self.dencity_map[to_key(row)]





def for_region(df, country, state=""):

    return df[(df['Country/Region'] == country) & (df['Province/State'] == state)]





test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")





pop = PopulationData()

hot = HotCityDensity()



for i in (train, test):

    # Also convert to `datetime` dates

    i['Date'] = pd.to_datetime(i['Date'])

    i["population"] = i.apply(pop, axis=1)

    i["density"] = i.apply(hot.density, axis=1)

    i["dencity"] = i.apply(hot.dencity, axis=1)

    i["Province/State"].fillna("", inplace=True)

    i.drop(["Lat", "Long"], axis=1, inplace=True)





def active(x):

    return ((x.values[-1] - x.values[-2]) > 3)





meta = (train.groupby(["Country/Region", "Province/State"])

        .agg(Id=('Id', 'first'),

             ConfirmedCases=('ConfirmedCases', 'max'),

             Fatalities=('Fatalities', 'max'),

             Active=('Fatalities', active),

             population=('population', 'first'),

             density=('density', 'first'),

             dencity=('dencity', 'first')))





meta.reset_index(inplace=True)
meta["mortality_rate_mil"] = 1_000_000 * meta["Fatalities"] / meta["population"]

meta.sort_values('mortality_rate_mil', ascending=False, inplace=True)



meta[meta["Active"]]
Candidate = namedtuple('Candidate', 'y, days, today_is_day, growth_rate, start, end')



def delta_logit(growth_rate):

    maxn = 30000

    logit = maxn / (1 + np.exp(-growth_rate * np.linspace(-500, 500, 200)))

    return np.round(logit[1:] - logit[:logit.shape[0] - 1])





def create_seed_kernels():

    args_for_days = {}

    for growth_rate in np.linspace(0.01, 0.5, 1000):

        x = delta_logit(growth_rate)

        active = np.where(x >= 1)[0]

        try:

            start = active[0]

            end = active[-1]

            days = end - start



            active_segment = x[start:(end + 1)]



            args_for_days[days] = (growth_rate, start, end)

        except IndexError:

            pass

    return args_for_days





def create_candidates():

    candidates = []

    kernels = create_seed_kernels()

    max_size = max(kernels.keys())

    for days, args in kernels.items():

        growth_rate, start, end = args



        full_y = delta_logit(growth_rate)[start:end]



        for today_is_day in range(1, days + 1, 1):

            y = full_y[:today_is_day].copy()

            y /= np.max(y)

            y = np.pad(y, (0, max_size - y.shape[0]), 'constant', constant_values=0)

            candidates.append(Candidate(y, days, today_is_day, growth_rate, start, end))



    return candidates





candidates = create_candidates()
def get_best(candidates, xm):

    # All candidates have the same size

    candidate_length = candidates[0].y.shape[0]

    assert(all(i.y.shape[0] == candidate_length for i in candidates))

    # All candidates are normalized to 1

    assert(all(((np.abs((i.y - 1.).max()) < 10e-9) for i in candidates)))



    normalize = xm.max()

    values_length = xm.values.shape[0]

    debug = None



    if values_length == 0:

        return None

    

    # Find the best fit curve for this country

    best_cor, best_candidate, best_cor_arg_max = None, None, None

    for candidate in candidates:



        slow = True

        if slow:

            for cor_arg_max in range(0, candidate_length - values_length):

                yslice = candidate.y[cor_arg_max:cor_arg_max+values_length].copy()

                yslice /= yslice.max() / normalize



                cor = np.sum(np.abs(xm.values - yslice))

                if best_cor is None or cor < best_cor:

                    best_cor = cor

                    best_candidate = candidate

                    best_cor_arg_max = cor_arg_max



            start_offset = best_cor_arg_max

        else:

            # Would be fast but doesn't work because `np.correlate` doesn't normalize

            # each time it "shifts" the correlation window.

            denorm_y = normalize * candidate.y

            cor = np.correlate(xm.values, denorm_y, mode='full')

            cor_arg_max = cor.argmax()

            cor = cor.max()



            if best_cor is None or cor > best_cor:

                best_cor = cor

                best_candidate = candidate

                best_cor_arg_max = cor_arg_max



            # Cross-correlation argmax is this weird offset from the end of the vector

            start_offset = candidate_length - best_cor_arg_max



    return best_candidate, start_offset, debug





def plot(xm, best):

    max_len = xm.index.shape[0]

    plt.plot(xm.index, xm.values)



    if best is not None:

        candidate, start_offset, debug = best  # Unpack

        unnormalized = candidate.y[start_offset:][:max_len]

        y = xm.max() * unnormalized / unnormalized.max()

        plt.plot(xm.index[:max_len], y)

        if debug:

            plt.plot(xm.index, debug[0])

            print(f'cost: {debug[1]}')





def best_to_string(best):

    if best is None:

        return f"N/A"

    candidate, start_offset, _ = best

    return (f'total duration: {candidate.days}, today is day: '

            f'{candidate.today_is_day}, start out of dataset: {start_offset}')





def to_smooth_delta(x):

    delta = (x["Fatalities"] - x["Fatalities"].shift(1))

    delta_mort_ma = delta.rolling(window=3).mean()

    xm = np.trim_zeros(delta_mort_ma[~np.isnan(delta_mort_ma)], 'f')

    trimmed = x.shape[0] - xm.shape[0]

    return xm, trimmed
%%time



def identify_timeline(country, state):

    x = for_region(train, country, state)

    xm, trimmed = to_smooth_delta(x)

    return get_best(candidates, xm)





def dev(country, state):

    x = for_region(train, country, state).set_index("Date")



    plt.figure(figsize=(20,7))



    plt.subplot(1, 3, 1)



    delta = (x["Fatalities"] - x["Fatalities"].shift(1))

    delta.plot()

    delta_mort_ma = delta.rolling(window=3).mean()

    delta_mort_ma.plot()



    plt.subplot(1, 3, 2)



    # Get rid of the Na's due to previous moving average window

    xm = np.trim_zeros(delta_mort_ma[~np.isnan(delta_mort_ma)], 'f')



    best = identify_timeline(country, state)

    if best is not None:

        plot(xm, best)



        plt.subplot(1, 3, 3)

        plt.plot(best[0].y)



    return f"Country: {country}, state: {state}, {best_to_string(best)}"





dev("Italy", "")
def precalculated(blind):

    if blind:

        return {('Republic of the Congo', ''): (None, None, None),

 ('Reunion', ''): (None, None, None),

 ('Saudi Arabia', ''): (None, None, None),

 ('Slovenia', ''): (None, None, None),

 ('Serbia', ''): (None, None, None),

 ('Trinidad and Tobago', ''): (None, None, None),

 ('Sudan', ''): (None, None, None),

 ('Seychelles', ''): (None, None, None),

 ('Pakistan', ''): (None, None, None),

 ('Nigeria', ''): (None, None, None),

 ('The Bahamas', ''): (None, None, None),

 ('Saint Vincent and the Grenadines', ''): (None, None, None),

 ('Tunisia', ''): (None, None, None),

 ('Somalia', ''): (None, None, None),

 ('Singapore', ''): (None, None, None),

 ('Sri Lanka', ''): (None, None, None),

 ('US', 'Indiana'): (None, None, None),

 ('Romania', ''): (None, None, None),

 ('Puerto Rico', ''): (None, None, None),

 ('Togo', ''): (None, None, None),

 ('US', 'Ohio'): (None, None, None),

 ('Qatar', ''): (None, None, None),

 ('Rwanda', ''): (None, None, None),

 ('South Africa', ''): (None, None, None),

 ('US', 'Virgin Islands'): (None, None, None),

 ('Afghanistan', ''): (None, None, None),

 ('Turkey', ''): (None, None, None),

 ('US', 'Alaska'): (None, None, None),

 ('Poland', ''): (None, None, None),

 ('Netherlands', 'Curacao'): (None, None, None),

 ('New Zealand', ''): (None, None, None),

 ('US', 'Oklahoma'): (None, None, None),

 ('Russia', ''): (None, None, None),

 ('US', 'South Carolina'): (None, None, None),

 ('Oman', ''): (None, None, None),

 ('Tanzania', ''): (None, None, None),

 ('US', 'Puerto Rico'): (None, None, None),

 ('US', 'Texas'): (None, None, None),

 ('US', 'North Carolina'): (None, None, None),

 ('Peru', ''): (None, None, None),

 ('US', 'United States Virgin Islands'): (None, None, None),

 ('US', 'Tennessee'): (None, None, None),

 ('US', 'Virginia'): (None, None, None),

 ('The Gambia', ''): (None, None, None),

 ('US', 'West Virginia'): (None, None, None),

 ('US', 'US'): (None, None, None),

 ('US', 'Rhode Island'): (None, None, None),

 ('US', 'Utah'): (None, None, None),

 ('US', 'Wyoming'): (None, None, None),

 ('US', 'Pennsylvania'): (None, None, None),

 ('Portugal', ''): (None, None, None),

 ('Senegal', ''): (None, None, None),

 ('Ukraine', ''): (None, None, None),

 ('Suriname', ''): (None, None, None),

 ('Venezuela', ''): (None, None, None),

 ('US', 'Wisconsin'): (None, None, None),

 ('Saint Lucia', ''): (None, None, None),

 ('Uzbekistan', ''): (None, None, None),

 ('US', 'Oregon'): (None, None, None),

 ('United Kingdom', 'Cayman Islands'): (None, None, None),

 ('US', 'Vermont'): (None, None, None),

 ('US', 'New York'): (None, None, None),

 ('United Arab Emirates', ''): (None, None, None),

 ('Uruguay', ''): (None, None, None),

 ('United Kingdom', 'Gibraltar'): (None, None, None),

 ('United Kingdom', 'Montserrat'): (None, None, None),

 ('Paraguay', ''): (None, None, None),

 ('Vietnam', ''): (None, None, None),

 ('US', 'Colorado'): (None, None, None),

 ('US', 'North Dakota'): (None, None, None),

 ('United Kingdom', 'Channel Islands'): (None, None, None),

 ('US', 'Iowa'): (None, None, None),

 ('US', 'District of Columbia'): (None, None, None),

 ('US', 'Delaware'): (None, None, None),

 ('US', 'Michigan'): (None, None, None),

 ('US', 'Arizona'): (None, None, None),

 ('US', 'Diamond Princess'): (None, None, None),

 ('US', 'Connecticut'): (None, None, None),

 ('US', 'Maine'): (None, None, None),

 ('US', 'Guam'): (None, None, None),

 ('US', 'Alabama'): (None, None, None),

 ('US', 'Kentucky'): (None, None, None),

 ('US', 'Illinois'): (None, None, None),

 ('US', 'Louisiana'): (None, None, None),

 ('US', 'Maryland'): (None, None, None),

 ('US', 'Massachusetts'): (None, None, None),

 ('US', 'Missouri'): (None, None, None),

 ('US', 'Minnesota'): (None, None, None),

 ('US', 'New Mexico'): (None, None, None),

 ('Nepal', ''): (None, None, None),

 ('Netherlands', 'Aruba'): (None, None, None),

 ('Norway', ''): (None, None, None),

 ('Canada', 'New Brunswick'): (None, None, None),

 ('US', 'Mississippi'): (None, None, None),

 ('US', 'Kansas'): (None, None, None),

 ('US', 'Georgia'): (None, None, None),

 ('US', 'Hawaii'): (None, None, None),

 ('Slovakia', ''): (None, None, None),

 ('Namibia', ''): (None, None, None),

 ('US', 'Grand Princess'): (None, None, None),

 ('US', 'New Hampshire'): (None, None, None),

 ('US', 'Idaho'): (None, None, None),

 ('US', 'Arkansas'): (None, None, None),

 ('US', 'Nebraska'): (None, None, None),

 ('Kuwait', ''): (None, None, None),

 ('US', 'Montana'): (None, None, None),

 ('US', 'Nevada'): (None, None, None),

 ('Canada', 'Nova Scotia'): (None, None, None),

 ('Canada', 'Grand Princess'): (None, None, None),

 ('China', 'Henan'): (None, None, None),

 ('Canada', 'Newfoundland and Labrador'): (None, None, None),

 ('Canada', 'Ontario'): (None, None, None),

 ('China', 'Macau'): (None, None, None),

 ('Central African Republic', ''): (None, None, None),

 ('China', 'Jiangsu'): (None, None, None),

 ('China', 'Shanxi'): (None, None, None),

 ('China', 'Qinghai'): (None, None, None),

 ('Colombia', ''): (None, None, None),

 ('Canada', 'Alberta'): (None, None, None),

 ('China', 'Shanghai'): (None, None, None),

 ('Canada', 'Manitoba'): (None, None, None),

 ('China', 'Heilongjiang'): (None, None, None),

 ('Canada', 'Prince Edward Island'): (None, None, None),

 ('China', 'Ningxia'): (None, None, None),

 ('Canada', 'Saskatchewan'): (None, None, None),

 ('China', 'Yunnan'): (None, None, None),

 ('Congo (Kinshasa)', ''): (None, None, None),

 ('Canada', 'Quebec'): (None, None, None),

 ('China', 'Tibet'): (None, None, None),

 ('Algeria', ''): (None, None, None),

 ('Armenia', ''): (None, None, None),

 ('Australia', 'Australian Capital Territory'): (None, None, None),

 ('Chile', ''): (None, None, None),

 ('Cruise Ship', 'Diamond Princess'): (None, None, None),

 ("Cote d'Ivoire", ''): (None, None, None),

 ('Bahrain', ''): (None, None, None),

 ('Congo (Brazzaville)', ''): (None, None, None),

 ('Aruba', ''): (None, None, None),

 ('Andorra', ''): (None, None, None),

 ('Croatia', ''): (None, None, None),

 ('Cyprus', ''): (None, None, None),

 ('Australia', 'Northern Territory'): (None, None, None),

 ('Austria', ''): (None, None, None),

 ('Australia', 'Queensland'): (None, None, None),

 ('Bangladesh', ''): (None, None, None),

 ('Barbados', ''): (None, None, None),

 ('Australia', 'South Australia'): (None, None, None),

 ('Antigua and Barbuda', ''): (None, None, None),

 ('Bosnia and Herzegovina', ''): (None, None, None),

 ('Costa Rica', ''): (None, None, None),

 ('Australia', 'From Diamond Princess'): (None, None, None),

 ('Australia', 'Victoria'): (None, None, None),

 ('Brazil', ''): (None, None, None),

 ('Cameroon', ''): (None, None, None),

 ('Australia', 'Tasmania'): (None, None, None),

 ('Brunei', ''): (None, None, None),

 ('Azerbaijan', ''): (None, None, None),

 ('Bolivia', ''): (None, None, None),

 ('Benin', ''): (None, None, None),

 ('Belarus', ''): (None, None, None),

 ('Israel', ''): (None, None, None),

 ('Luxembourg', ''): (None, None, None),

 ('Montenegro', ''): (None, None, None),

 ('Kosovo', ''): (None, None, None),

 ('Burkina Faso', ''): (None, None, None),

 ('Kyrgyzstan', ''): (None, None, None),

 ('Cuba', ''): (None, None, None),

 ('Liechtenstein', ''): (None, None, None),

 ('Hungary', ''): (None, None, None),

 ('Jersey', ''): (None, None, None),

 ('Czechia', ''): (None, None, None),

 ('Holy See', ''): (None, None, None),

 ('Cambodia', ''): (None, None, None),

 ('Denmark', 'Denmark'): (None, None, None),

 ('Jordan', ''): (None, None, None),

 ('Latvia', ''): (None, None, None),

 ('Iceland', ''): (None, None, None),

 ('Mayotte', ''): (None, None, None),

 ('Kenya', ''): (None, None, None),

 ('Bhutan', ''): (None, None, None),

 ('North Macedonia', ''): (None, None, None),

 ('Jamaica', ''): (None, None, None),

 ('Kazakhstan', ''): (None, None, None),

 ('France', 'Mayotte'): (None, None, None),

 ('Lithuania', ''): (None, None, None),

 ('Liberia', ''): (None, None, None),

 ('Mexico', ''): (None, None, None),

 ('Mauritania', ''): (None, None, None),

 ('Honduras', ''): (None, None, None),

 ('Monaco', ''): (None, None, None),

 ('Denmark', 'Faroe Islands'): (None, None, None),

 ('Malaysia', ''): (None, None, None),

 ('Malta', ''): (None, None, None),

 ('Mauritius', ''): (None, None, None),

 ('Guyana', ''): (None, None, None),

 ('Mongolia', ''): (None, None, None),

 ('Martinique', ''): (None, None, None),

 ('Finland', ''): (None, None, None),

 ('Moldova', ''): (None, None, None),

 ('Djibouti', ''): (None, None, None),

 ('Dominican Republic', ''): (None, None, None),

 ('Eswatini', ''): (None, None, None),

 ('France', 'Guadeloupe'): (None, None, None),

 ('France', 'Reunion'): (None, None, None),

 ('France', 'French Guiana'): (None, None, None),

 ('Ecuador', ''): (None, None, None),

 ('France', 'French Polynesia'): (None, None, None),

 ('French Guiana', ''): (None, None, None),

 ('Guadeloupe', ''): (None, None, None),

 ('Estonia', ''): (None, None, None),

 ('France', 'Saint Barthelemy'): (None, None, None),

 ('Ethiopia', ''): (None, None, None),

 ('Maldives', ''): (None, None, None),

 ('Georgia', ''): (None, None, None),

 ('Equatorial Guinea', ''): (None, None, None),

 ('Guernsey', ''): (None, None, None),

 ('Guinea', ''): (None, None, None),

 ('Gabon', ''): (None, None, None),

 ('Guam', ''): (None, None, None),

 ('Greenland', ''): (None, None, None),

 ('Zambia', ''): (None, None, None),

 ('Guatemala', ''): (None, None, None),

 ('Gambia, The', ''): (None, None, None),

 ('Ghana', ''): (None, None, None),

 ('France', 'St Martin'): (None, None, None),

 ('China', 'Sichuan'): (44, 21, 0),

 ('China', 'Shandong'): (148, 18, 0),

 ('China', 'Chongqing'): (26, 16, 0),

 ('China', 'Hainan'): (28, 19, 0),

 ('China', 'Hebei'): (74, 29, 0),

 ('China', 'Gansu'): (198, 14, 11),

 ('China', 'Guizhou'): (198, 3, 0),

 ('China', 'Guangdong'): (134, 13, 0),

 ('China', 'Hubei'): (94, 72, 25),

 ('China', 'Guangxi'): (198, 7, 0),

 ('China', 'Jiangxi'): (198, 3, 0),

 ('China', 'Xinjiang'): (198, 3, 0),

 ('Philippines', ''): (198, 3, 0),

 ('China', 'Tianjin'): (16, 11, 0),

 ('China', 'Jilin'): (198, 3, 0),

 ('China', 'Hong Kong'): (198, 3, 0),

 ('Iran', ''): (96, 24, 2),

 ('China', 'Shaanxi'): (198, 3, 0),

 ('Thailand', ''): (198, 3, 0),

 ('US', 'South Dakota'): (198, 1, 0),

 ('China', 'Liaoning'): (198, 3, 0),

 ('Taiwan*', ''): (198, 3, 0),

 ('Indonesia', ''): (198, 1, 0),

 ('China', 'Beijing'): (82, 35, 0),

 ('France', 'France'): (64, 26, 0),

 ('US', 'California'): (198, 19, 17),

 ('China', 'Hunan'): (198, 13, 0),

 ('China', 'Fujian'): (198, 3, 0),

 ('Belgium', ''): (198, 1, 0),

 ('Canada', 'British Columbia'): (198, 3, 0),

 ('Australia', 'New South Wales'): (198, 7, 0),

 ('China', 'Anhui'): (134, 127, 119),

 ('China', 'Inner Mongolia'): (198, 3, 0),

 ('Australia', 'Western Australia'): (198, 3, 0),

 ('Ireland', ''): (198, 1, 0),

 ('Sweden', ''): (198, 1, 0),

 ('Korea, South', ''): (190, 35, 1),

 ('Italy', ''): (62, 27, 6),

 ('Spain', ''): (36, 18, 9),

 ('Japan', ''): (68, 31, 0),

 ('China', 'Zhejiang'): (198, 3, 0),

 ('Iraq', ''): (198, 15, 8),

 ('Argentina', ''): (198, 3, 0),

 ('Morocco', ''): (198, 2, 0),

 ('San Marino', ''): (198, 3, 0),

 ('Switzerland', ''): (198, 14, 7),

 ('Netherlands', 'Netherlands'): (106, 11, 5),

 ('Egypt', ''): (198, 3, 0),

 ('US', 'Florida'): (198, 2, 0),

 ('India', ''): (198, 1, 0),

 ('Greece', ''): (198, 1, 0),

 ('Bulgaria', ''): (198, 1, 0),

 ('Germany', ''): (198, 19, 16),

 ('Panama', ''): (198, 1, 0),

 ('Lebanon', ''): (26, 3, 1),

 ('United Kingdom', 'United Kingdom'): (88, 31, 4),

 ('Albania', ''): (198, 1, 0),

 ('US', 'New Jersey'): (198, 2, 0),

 ('US', 'Washington'): (78, 25, 20)}

    else:

        return {('China', 'Hubei'): (98, 86, 26),

 ('Philippines', ''): (124, 54, 0),

 ('Taiwan*', ''): (198, 3, 0),

 ('Australia', 'Western Australia'): (198, 3, 0),

 ('Thailand', ''): (26, 3, 0),

 ('Japan', ''): (150, 69, 28),

 ('US', 'Florida'): (148, 18, 3),

 ('France', 'France'): (86, 38, 0),

 ('US', 'Indiana'): (66, 12, 0),

 ('US', 'Texas'): (198, 13, 5),

 ('US', 'Connecticut'): (72, 10, 4),

 ('US', 'Illinois'): (72, 12, 2),

 ('US', 'Massachusetts'): (24, 13, 8),

 ('Guatemala', ''): (198, 3, 0),

 ('Korea, South', ''): (198, 96, 60),

 ('Italy', ''): (86, 44, 10),

 ('Argentina', ''): (196, 17, 0),

 ('US', 'California'): (80, 39, 0),

 ('Netherlands', 'Curacao'): (198, 3, 0),

 ('Ukraine', ''): (198, 10, 0),

 ('Azerbaijan', ''): (198, 3, 0),

 ('US', 'New Jersey'): (52, 16, 0),

 ('Colombia', ''): (198, 20, 17),

 ('Australia', 'New South Wales'): (30, 19, 0),

 ('Ghana', ''): (12, 8, 4),

 ('Iran', ''): (86, 47, 12),

 ('Egypt', ''): (58, 28, 11),

 ('Cuba', ''): (198, 3, 0),

 ('US', 'Michigan'): (44, 23, 1),

 ('Morocco', ''): (148, 17, 2),

 ('US', 'Georgia'): (50, 25, 12),

 ('Iraq', ''): (162, 23, 0),

 ('US', 'Louisiana'): (62, 15, 3),

 ('Canada', 'Quebec'): (16, 10, 5),

 ('Canada', 'British Columbia'): (128, 16, 0),

 ('Canada', 'Ontario'): (82, 24, 2),

 ('US', 'Washington'): (186, 25, 3),

 ('Brazil', ''): (48, 22, 14),

 ('Indonesia', ''): (36, 22, 8),

 ('Poland', ''): (134, 46, 0),

 ('Finland', ''): (198, 3, 0),

 ('Kazakhstan', ''): (198, 1, 0),

 ('Slovakia', ''): (198, 3, 0),

 ('San Marino', ''): (124, 21, 0),

 ('US', 'New York'): (42, 22, 9),

 ('Chile', ''): (198, 14, 11),

 ('Algeria', ''): (36, 22, 9),

 ('Spain', ''): (70, 32, 8),

 ('Albania', ''): (48, 14, 0),

 ('United Kingdom', 'United Kingdom'): (50, 26, 6),

 ('Peru', ''): (198, 189, 184),

 ('Bulgaria', ''): (198, 11, 0),

 ('Bahrain', ''): (198, 13, 4),

 ('United Arab Emirates', ''): (198, 3, 0),

 ('Dominican Republic', ''): (80, 10, 1),

 ('Germany', ''): (66, 31, 14),

 ('Lebanon', ''): (18, 11, 7),

 ('Switzerland', ''): (50, 26, 6),

 ('Netherlands', 'Netherlands'): (58, 29, 9),

 ('Slovenia', ''): (26, 13, 2),

 ('Turkey', ''): (26, 14, 6),

 ('Ireland', ''): (122, 14, 0),

 ('Tunisia', ''): (106, 10, 4),

 ('US', 'West Virginia'): (None, None, None),

 ('US', 'Wyoming'): (None, None, None),

 ('Burkina Faso', ''): (62, 7, 0),

 ('US', 'Idaho'): (None, None, None),

 ('Malaysia', ''): (60, 8, 0),

 ('Martinique', ''): (198, 3, 0),

 ('Seychelles', ''): (None, None, None),

 ('Paraguay', ''): (198, 4, 0),

 ('Sri Lanka', ''): (None, None, None),

 ('United Kingdom', 'Channel Islands'): (None, None, None),

 ('US', 'Iowa'): (None, None, None),

 ('The Gambia', ''): (None, None, None),

 ('Somalia', ''): (None, None, None),

 ('South Africa', ''): (None, None, None),

 ('US', 'Oregon'): (None, None, None),

 ('Uruguay', ''): (None, None, None),

 ('Venezuela', ''): (None, None, None),

 ('United Kingdom', 'Montserrat'): (None, None, None),

 ('Uzbekistan', ''): (None, None, None),

 ('Vietnam', ''): (None, None, None),

 ('US', 'Montana'): (None, None, None),

 ('US', 'Virgin Islands'): (None, None, None),

 ('US', 'New Mexico'): (None, None, None),

 ('United Kingdom', 'Gibraltar'): (None, None, None),

 ('Tanzania', ''): (None, None, None),

 ('Togo', ''): (None, None, None),

 ('Jamaica', ''): (198, 3, 0),

 ('US', 'Diamond Princess'): (None, None, None),

 ('US', 'Delaware'): (None, None, None),

 ('Croatia', ''): (198, 3, 0),

 ('Trinidad and Tobago', ''): (None, None, None),

 ('The Bahamas', ''): (None, None, None),

 ('Greece', ''): (90, 33, 2),

 ('US', 'Rhode Island'): (None, None, None),

 ('US', 'Alaska'): (None, None, None),

 ('US', 'North Carolina'): (None, None, None),

 ('US', 'US'): (None, None, None),

 ('Ecuador', ''): (34, 17, 5),

 ('Hungary', ''): (104, 15, 4),

 ('US', 'Alabama'): (None, None, None),

 ('US', 'Nebraska'): (None, None, None),

 ('US', 'North Dakota'): (None, None, None),

 ('US', 'United States Virgin Islands'): (None, None, None),

 ('Senegal', ''): (None, None, None),

 ('US', 'Maine'): (None, None, None),

 ('Kuwait', ''): (None, None, None),

 ('Canada', 'Newfoundland and Labrador'): (None, None, None),

 ('Suriname', ''): (None, None, None),

 ('Israel', ''): (198, 13, 9),

 ('Canada', 'Prince Edward Island'): (None, None, None),

 ('China', 'Heilongjiang'): (None, None, None),

 ('China', 'Qinghai'): (None, None, None),

 ('China', 'Jiangsu'): (None, None, None),

 ('China', 'Macau'): (None, None, None),

 ('China', 'Shanghai'): (None, None, None),

 ('China', 'Ningxia'): (None, None, None),

 ('Moldova', ''): (198, 3, 0),

 ('China', 'Henan'): (None, None, None),

 ('Canada', 'Saskatchewan'): (None, None, None),

 ('Congo (Brazzaville)', ''): (None, None, None),

 ('Cruise Ship', 'Diamond Princess'): (None, None, None),

 ('Czechia', ''): (None, None, None),

 ('Central African Republic', ''): (None, None, None),

 ('China', 'Tibet'): (None, None, None),

 ("Cote d'Ivoire", ''): (None, None, None),

 ('Denmark', 'Faroe Islands'): (None, None, None),

 ('Canada', 'Nova Scotia'): (None, None, None),

 ('China', 'Yunnan'): (None, None, None),

 ('China', 'Shanxi'): (None, None, None),

 ('Djibouti', ''): (None, None, None),

 ('Canada', 'New Brunswick'): (None, None, None),

 ('Panama', ''): (32, 15, 0),

 ('Saint Vincent and the Grenadines', ''): (None, None, None),

 ('Armenia', ''): (None, None, None),

 ('Australia', 'Australian Capital Territory'): (None, None, None),

 ('Antigua and Barbuda', ''): (None, None, None),

 ('Canada', 'Manitoba'): (None, None, None),

 ('Guyana', ''): (198, 3, 0),

 ('Australia', 'Northern Territory'): (None, None, None),

 ('Australia', 'Queensland'): (None, None, None),

 ('Australia', 'Victoria'): (None, None, None),

 ('Barbados', ''): (None, None, None),

 ('Aruba', ''): (None, None, None),

 ('Costa Rica', ''): (24, 15, 10),

 ('Australia', 'Tasmania'): (None, None, None),

 ('Australia', 'South Australia'): (None, None, None),

 ('Brunei', ''): (None, None, None),

 ('Benin', ''): (None, None, None),

 ('Belarus', ''): (None, None, None),

 ('Bolivia', ''): (None, None, None),

 ('Cambodia', ''): (None, None, None),

 ('Canada', 'Grand Princess'): (None, None, None),

 ('Estonia', ''): (None, None, None),

 ('Equatorial Guinea', ''): (None, None, None),

 ('Ethiopia', ''): (None, None, None),

 ('Bosnia and Herzegovina', ''): (198, 13, 9),

 ('Eswatini', ''): (None, None, None),

 ('Liberia', ''): (None, None, None),

 ('Liechtenstein', ''): (None, None, None),

 ('Maldives', ''): (None, None, None),

 ('Australia', 'From Diamond Princess'): (None, None, None),

 ('Malta', ''): (None, None, None),

 ('Lithuania', ''): (198, 4, 0),

 ('Nepal', ''): (None, None, None),

 ('Namibia', ''): (None, None, None),

 ('Mayotte', ''): (None, None, None),

 ('Mauritania', ''): (None, None, None),

 ('Bhutan', ''): (None, None, None),

 ('Monaco', ''): (None, None, None),

 ('Cameroon', ''): (None, None, None),

 ('New Zealand', ''): (None, None, None),

 ('Iceland', ''): (198, 1, 0),

 ('Netherlands', 'Aruba'): (None, None, None),

 ('Republic of the Congo', ''): (None, None, None),

 ('Puerto Rico', ''): (None, None, None),

 ('Mongolia', ''): (None, None, None),

 ('Saint Lucia', ''): (None, None, None),

 ('Rwanda', ''): (None, None, None),

 ('Reunion', ''): (None, None, None),

 ('Kosovo', ''): (None, None, None),

 ('Kyrgyzstan', ''): (None, None, None),

 ('French Guiana', ''): (None, None, None),

 ('France', 'Saint Barthelemy'): (None, None, None),

 ('France', 'Reunion'): (None, None, None),

 ('France', 'French Guiana'): (None, None, None),

 ('Gambia, The', ''): (None, None, None),

 ('Greenland', ''): (None, None, None),

 ('Georgia', ''): (None, None, None),

 ('Guam', ''): (None, None, None),

 ('France', 'French Polynesia'): (None, None, None),

 ('Guinea', ''): (None, None, None),

 ('Guernsey', ''): (None, None, None),

 ('Qatar', ''): (None, None, None),

 ('Latvia', ''): (None, None, None),

 ('France', 'St Martin'): (None, None, None),

 ('Holy See', ''): (None, None, None),

 ('Oman', ''): (None, None, None),

 ('Kenya', ''): (None, None, None),

 ('Honduras', ''): (None, None, None),

 ('France', 'Mayotte'): (None, None, None),

 ('Gabon', ''): (198, 3, 0),

 ('Jersey', ''): (None, None, None),

 ('Jordan', ''): (None, None, None),

 ('Guadeloupe', ''): (None, None, None),

 ('Serbia', ''): (198, 15, 10),

 ('Zambia', ''): (None, None, None),

 ('Denmark', 'Denmark'): (64, 25, 1),

 ('Austria', ''): (60, 21, 0),

 ('Belgium', ''): (34, 19, 5),

 ('Singapore', ''): (198, 3, 0),

 ('North Macedonia', ''): (198, 14, 11),

 ('Norway', ''): (198, 90, 2),

 ('Cyprus', ''): (26, 3, 0),

 ('Romania', ''): (24, 13, 9),

 ('Montenegro', ''): (198, 2, 0),

 ('Sweden', ''): (106, 45, 4),

 ('Luxembourg', ''): (106, 12, 2),

 ('Portugal', ''): (38, 19, 11),

 ('Mauritius', ''): (12, 8, 4),

 ('Andorra', ''): (198, 3, 0),

 ('China', 'Beijing'): (82, 35, 0),

 ('China', 'Hebei'): (74, 29, 0),

 ('US', 'Virginia'): (74, 11, 0),

 ('Saudi Arabia', ''): (198, 1, 0),

 ('China', 'Hainan'): (28, 19, 0),

 ('Bangladesh', ''): (198, 14, 7),

 ('China', 'Shandong'): (148, 18, 0),

 ('US', 'Colorado'): (122, 14, 3),

 ('China', 'Sichuan'): (44, 21, 0),

 ('Canada', 'Alberta'): (198, 3, 0),

 ('China', 'Guangdong'): (134, 13, 0),

 ('China', 'Chongqing'): (26, 16, 0),

 ('Mexico', ''): (198, 14, 8),

 ('Sudan', ''): (198, 3, 0),

 ('China', 'Gansu'): (198, 14, 11),

 ('Pakistan', ''): (198, 23, 17),

 ('China', 'Hunan'): (198, 13, 0),

 ('US', 'Ohio'): (118, 17, 10),

 ('China', 'Tianjin'): (16, 11, 0),

 ('China', 'Jilin'): (198, 3, 0),

 ('China', 'Anhui'): (134, 127, 119),

 ('China', 'Guizhou'): (198, 3, 0),

 ('China', 'Guangxi'): (198, 7, 0),

 ('Afghanistan', ''): (198, 3, 0),

 ('Congo (Kinshasa)', ''): (198, 4, 0),

 ('US', 'Missouri'): (104, 14, 6),

 ('US', 'Vermont'): (54, 8, 2),

 ('United Kingdom', 'Cayman Islands'): (198, 3, 0),

 ('China', 'Hong Kong'): (198, 3, 0),

 ('China', 'Xinjiang'): (198, 3, 0),

 ('US', 'Arizona'): (52, 6, 2),

 ('Russia', ''): (198, 3, 0),

 ('US', 'Nevada'): (16, 10, 1),

 ('US', 'Kansas'): (198, 3, 0),

 ('US', 'Maryland'): (198, 16, 10),

 ('China', 'Jiangxi'): (198, 3, 0),

 ('US', 'Wisconsin'): (24, 16, 11),

 ('China', 'Liaoning'): (198, 3, 0),

 ('US', 'Oklahoma'): (198, 13, 7),

 ('US', 'District of Columbia'): (24, 15, 10),

 ('India', ''): (72, 17, 0),

 ('US', 'Arkansas'): (198, 1, 0),

 ('US', 'Kentucky'): (198, 13, 5),

 ('China', 'Fujian'): (198, 3, 0),

 ('US', 'Pennsylvania'): (52, 10, 1),

 ('China', 'Zhejiang'): (198, 3, 0),

 ('China', 'Shaanxi'): (198, 3, 0),

 ('US', 'South Dakota'): (198, 3, 0),

 ('China', 'Inner Mongolia'): (198, 3, 0),

 ('US', 'Mississippi'): (198, 3, 0),

 ('US', 'Tennessee'): (198, 14, 11),

 ('US', 'South Carolina'): (74, 25, 0),

 ('Nigeria', ''): (198, 2, 0),

 ('France', 'Guadeloupe'): (198, 2, 0),

 ('US', 'New Hampshire'): (198, 2, 0),

 ('US', 'Utah'): (198, 3, 0),

 ('US', 'Minnesota'): (198, 3, 0),

 ('US', 'Puerto Rico'): (12, 8, 4),

 ('US', 'Grand Princess'): (198, 3, 0),

 ('US', 'Hawaii'): (198, 1, 0),

 ('US', 'Guam'): (198, 3, 0)}
%%time

from multiprocessing import Pool



def parallel(args):

    try:

        country, state = args

        return False, country, state, identify_timeline(country, state)

    except:

        return True, country, state, None



many_cpus = False

if many_cpus:

    models = {}

    with Pool(96) as p:

        for err, country, state, best in p.imap_unordered(parallel,

                                                          meta[["Country/Region", "Province/State"]].values):

            if best is None:

                models[(country, state)] = None, None, None

            else:

                if not err:

                    candidate, start_offset, _ = best

                    models[(country, state)] = candidate.days, candidate.today_is_day, start_offset

                else:

                    print(f"Error in {country}, {state}")

                    break
models = precalculated(blind=False)



void = None, None

meta["total duration"] = meta.apply(lambda row: models.get(to_key(row), void)[0], axis=1)

meta["today is day"] = meta.apply(lambda row: models.get(to_key(row), void)[1], axis=1)



meta.dropna()
cleaner = meta[meta["total duration"] != meta["total duration"].max()].dropna()

sns.lmplot(x="density", y="total duration", data=cleaner[["density", "total duration"]], order=2, scatter_kws={"s": 15})
def replay_to_find_normalization(train, country, state, model):

    if any(i is None for i in model):

        return None, None, None



    total_duration, today_is_day, start_out_of_dataset = model



    selected = [candidate for candidate in candidates if ((candidate.days == total_duration) and

                                                          (candidate.today_is_day == today_is_day))]

    assert len(selected) == 1  # If error the candidate generation parameters have changed



    candidate = selected[0]



    # All this block shouldn't be necessary, if we store the 'trimmed' and the

    # 'n_div_factor' arguments when we choose the candidate (on top of 'start_out_of_dataset').

    xm, trimmed = to_smooth_delta(for_region(train, country, state))

    values_length = xm.values.shape[0]

    normalize = xm.max()

    yslice = candidate.y[start_out_of_dataset:start_out_of_dataset+values_length].copy()

    n_div_factor = yslice.max() / normalize

    return trimmed, n_div_factor, values_length





def project(train, test, models, country, state=""):



    model = models[(country, state)]



    trimmed, n_div_factor, values_length = replay_to_find_normalization(train, country, state, model)



    if trimmed is None:

        return None



    total_duration, today_is_day, start_out_of_dataset = model



    # Now we can go get the unbound candidate

    selected = [candidate for candidate in candidates if ((candidate.days == total_duration) and

                                                          (candidate.today_is_day == total_duration))]

    assert len(selected) == 1  # If error the candidate generation parameters have changed

    candidate = selected[0]



    # Normalize

    yslice = candidate.y[start_out_of_dataset:]

    my_norm = candidate.y[start_out_of_dataset:start_out_of_dataset+values_length].max()

    yslice = yslice / my_norm / n_div_factor



    # Add late start because of the moving average

    deltas = np.append(np.zeros(trimmed), yslice)



    # Go from delta to actual fatalities

    cumsums = np.cumsum(deltas)



    date_0 = for_region(train, country, state)["Date"].values[0]



    # Return all available data

    return {date_0 + pd.offsets.Day(i): v for i, v in enumerate(cumsums)}





def plot_for(context, country, state):

    train, test, models = context



    fatalities_at = project(train, test, models, country, state)



    if fatalities_at is None:

        print("No model available")

        return



    t_train = for_region(train, country, state).copy()



    t_train["FatalitiesProjected"] = t_train.apply(lambda row: fatalities_at[row['Date']], axis=1)



    plt.figure(figsize=(20,7))

    plt.subplot(1, 2, 1)

    t_train = t_train.set_index('Date')

    plt.plot(t_train["Fatalities"])

    plt.plot(t_train["FatalitiesProjected"])

    plt.title(f'present {country}-{state}')



    t_test = for_region(test, country, state).copy()

    t_test["Fatalities"] = t_test.apply(lambda row: fatalities_at[row['Date']], axis=1)



    plt.subplot(1, 2, 2)

    t_test = t_test.set_index('Date')

    plt.plot(t_test["Fatalities"])

    plt.title(f'future {country}-{state}')





context = train, test, models
plot_for(context, "US", "New York")
plot_for(context, "Korea, South", "")
plot_for(context, "China", "Hubei")
plot_for(context, "Italy", "")
plot_for(context, "Spain", "")
plot_for(context, "France", "France")
plot_for(context, "Greece", "")
plot_for(context, "Germany", "")
dates_overlap = ['2020-03-12','2020-03-13','2020-03-14','2020-03-15','2020-03-16','2020-03-17','2020-03-18',

                 '2020-03-19','2020-03-20','2020-03-21','2020-03-22','2020-03-23','2020-03-24','2020-03-25',

                 '2020-03-26']

train_blind = train.loc[~train['Date'].isin(dates_overlap)]



# Double check that there are no informed ConfirmedCases and Fatalities after 2020-03-11

train_blind.loc[train_blind['Date'] >= '2020-03-12', 'ConfirmedCases'] = np.nan

train_blind.loc[train_blind['Date'] >= '2020-03-12', 'Fatalities'] = np.nan





models = precalculated(blind=True)



context = train_blind, test, models
plot_for(context, "US", "New York")
plot_for(context, "China", "Hubei")
plot_for(context, "Italy", "")
region_fatalities_at = {}

for country, state in meta[["Country/Region", "Province/State"]].values:

    fatalities_at = project(train_blind, test, models, country, state)

    if fatalities_at is None:

        continue

    for date, forecast in fatalities_at.items():

        region_fatalities_at[(country, state, date)] = int(np.round(forecast))



test["Fatalities"] = test.apply(lambda row: region_fatalities_at.get((row["Country/Region"], row["Province/State"], row["Date"]), 1), axis=1)



last_confirmed = train_blind.groupby(["Country/Region", "Province/State"]).agg({"ConfirmedCases": "last"})

last_confirmed.reset_index(inplace=True)

last_confirmed_map = {(i,j): v for i, j, v in last_confirmed[["Country/Region", "Province/State", "ConfirmedCases"]].values}



test["ConfirmedCases"] = test.apply(lambda row: int(last_confirmed_map.get((row["Country/Region"], row["Province/State"]), 0)), axis=1)



test[:200]
submission = test[["ForecastId", "ConfirmedCases", "Fatalities"]]

submission.to_csv('submission.csv', index=False)
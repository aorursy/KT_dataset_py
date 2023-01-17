%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt



def get_csv_path(year): 

    return '../input/' + str(year) + '.csv'



def get_col_name(name, year): #Ühtlustab aasta 2017 tulpade nimed eelnevate aastate omadega

    return ''.join([c if c.isalnum() else '.' for c in name]) if year == 2017 else name



def get_all_countries(dfs, years):

    countries = set()

    for df, year in zip(dfs, years):

        for country in df[get_col_name('Country', year)].values:

            countries.add(country)

    return tuple(countries)



def df_has_country(df, year, country):

    return country in df[get_col_name('Country', year)].values



def get_excess_countries(dfs, years, all_countries):

    excess = []

    for country in all_countries:

        for df, year in zip(dfs, years):

            if not df_has_country(df, year, country):

                excess.append(country)

                break

    return excess



def rem_countries(dfs, years, countries):

    new_dfs = []

    for df, year in zip(dfs, years):

        keep = ~df[get_col_name('Country', year)].isin(countries)

        new_dfs.append(df[keep])

    return new_dfs



def sort_by_country(dfs, years):

    new_dfs = []

    for df, year in zip(dfs, years):

        new_dfs.append(df.sort_values(get_col_name('Country', year)))

        new_dfs[-1].reset_index(drop=True, inplace=True)

    return new_dfs



def get_clean_data(years):

    dfs = tuple(pd.read_csv(get_csv_path(year)) for year in years)

    all_countries = get_all_countries(dfs, years)

    excess = get_excess_countries(dfs, years, all_countries) # TODO: Hong Kong, Taiwan rename?

    dfs = rem_countries(dfs, years, excess)

    dfs = sort_by_country(dfs, years)

    all_countries = dfs[0][get_col_name('Country', years[0])].values # Sorted

    return dfs, all_countries



def get_gdp_name(year):

    return get_col_name('Economy (GDP per Capita)', year)



def get_gdp_values(df, year):

    return df[get_gdp_name(year)].values



def get_happiness_name(year):

    return get_col_name('Happiness Score', year)



def get_happiness_values(df, year):

    return df[get_happiness_name(year)].values



def get_gdp_happiness_deltas(dfs, years):

    dgs, dhs = [], []

    for i in range(len(dfs) - 1):

        cur_gdp = get_gdp_values(dfs[i], years[i])

        next_gdp = get_gdp_values(dfs[i + 1], years[i + 1])

        cur_happiness = get_happiness_values(dfs[i], years[i])

        next_happiness = get_happiness_values(dfs[i + 1], years[i + 1])

        dgs.append(next_gdp - cur_gdp)

        dhs.append(next_happiness - cur_happiness)

    return dgs, dhs



def draw_zeros(ax, xs, ys):

    x0, x1 = min(xs), max(xs)

    y0, y1 = min(ys), max(ys)

    ax.plot((x0, x1), (0, 0), '--') # x-axis

    ax.plot((0, 0), (y0, y1), '--')

    

def plot_gdp_happiness_delta(dg, dh, countries=None):

    fig, ax = plt.subplots(figsize=(17,19))

    ax.scatter(dg, dh)

    if countries is not None: # Draw country names next to points.

        for i in range(len(countries)):

            ax.annotate(countries[i], (dg[i], dh[i]))

    draw_zeros(ax, dg, dh)

    plt.xlabel('GDP per Capita') # TODO: Estonian labels?

    plt.ylabel('Happiness Score')



def plot_gdp_happiness_hist(df, year):

    fig, ax = plt.subplots()

    names = (get_gdp_name(year), get_happiness_name(year))

    gdp_vals = df[names[0]].values

    avg_gdp = sum(gdp_vals) / len(gdp_vals)

    poor, rich = df[df[names[0]] <= avg_gdp], df[df[names[0]] > avg_gdp]

    plt.hist((poor[names[1]].values, rich[names[1]].values), label=('poor', 'rich'))

    plt.xlabel('Happiness Score')

    plt.ylabel('Countries')

    ax.set_xlim(xmin=0)

    ax.set_ylim(ymin=0)



years = (2015, 2016, 2017)

dfs, all_countries = get_clean_data(years)

dfs[0].groupby(get_gdp_name(2015))[get_happiness_name(2015)].mean()
dgs, dhs = get_gdp_happiness_deltas(dfs, years)

plot_gdp_happiness_delta(dgs[0], dhs[0], all_countries) # 2016 - 2015
plot_gdp_happiness_delta(dgs[1], dhs[1]) # 2017 - 2016
for df, year in zip(dfs, years):

    plot_gdp_happiness_hist(df, year)
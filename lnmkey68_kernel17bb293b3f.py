import pandas as pd

import numpy as np



# Very big number to be used for a parameter values of some models

BIG_NUMBER = 1000000
train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv', na_filter=False)

test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv', na_filter=False)

submission_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
train_df
test_df
submission_df
train_df = train_df[["Province_State", "Country_Region", "Date", "ConfirmedCases", "Fatalities"]]

train_df
len(set(train_df["Country_Region"])) == len(set(test_df["Country_Region"]))
set(train_df["Country_Region"]) == set(test_df["Country_Region"])
countries = set(train_df["Country_Region"])

countries
# This function assumes that the number of confirmed cases/fatalities doubles every n days

# The task is to find the optimal n from curve fitting, separately for cases and fatalities

def func(x, b, n):

    return x * (b ** (1/n))
from scipy.optimize import curve_fit



# Iterate through countries sorted alphabetically from A to Z. 

# As some countries, like USA, China, Canada, UK, Australia, have provinces/states, decend to the province level 

# (i.e., each province within such countries gets its owm model)

for country in sorted(countries):

    print("Country: ", country)

    # Select information related to the current country

    c_df = train_df[train_df["Country_Region"] == country]

    # Get a list of country's provinces

    provinces = set(c_df["Province_State"])

    print("Provinces: ", provinces)

    # Iterate over provinces

    for province in sorted(provinces):

        # Create a compound name for each country when provinces are present

        if province != "":

            full_country = country + "-" + province

        else:

            full_country = country

        

        # From country information, select the current province information

        p_df = c_df[c_df["Province_State"] == province]

        

        # Prepare data for building a model

        X1 = p_df[p_df["ConfirmedCases"] > 0]["ConfirmedCases"].values[:-1]  # Omit the last value in order to properly form labels

        y1 = p_df[p_df["ConfirmedCases"] > 0]["ConfirmedCases"].values[1:]  # Notice that "labels" are in fact "data" shifted one position to the right

        X2 = p_df[p_df["Fatalities"] > 0]["Fatalities"].values[:-1]  # Omit the last value in order to properly form labels

        y2 = p_df[p_df["Fatalities"] > 0]["Fatalities"].values[1:] # Notice that "labels" are in fact "data" shifted one position to the right

        

        # For confirmed cases, find the optimal value of a model parameter and perform the curve fitting if possible 

        # Treat special cases when either X or y or both contains all zeroes or just one (last) non-zero value!

        if len(X1) > 1 and len(y1) > 1:  # Build a model only if there are two or more non-zero values

            popt, _ = curve_fit(func, X1, y1)

            popt_cases = popt  # there is just one parameter

        else:  

            # otherwise, just set the parameter to a very big number, implying that there would be almost no change in numbers

            popt_cases = 2, BIG_NUMBER

        # Treat the special case if it turned out that the parameter value is zero

        if popt_cases[1] == 0:

            # Set the parameter to a very large value m so that the quantity 2**(1/m) -> 1, which implies that

            # the numbers won't grow

            popt_cases = 2, BIG_NUMBER

        print("{}: Optimal parameter value for confirmed cases: {}".format(full_country, popt_cases))

        

        # For fatalities, find the optimal value of a model parameter and perform the curve fitting if possible 

        # Treat special cases when either X or y or both contains all zeroes or just one (last) non-zero value!

        if len(X2) > 1 and len(y2) > 1:

            popt, _ = curve_fit(func, X2, y2)

            popt_fatalities = popt  # there is just one parameter

        else:

            # otherwise, just set the parameter to a very big number, implying that there would be almost no change in numbers

            popt_fatalities = 2, BIG_NUMBER

        # Treat the special case if it turned out that the parameter value is zero

        if popt_fatalities[1] == 0:

            # Set the parameter to a very large value m so that the quantity 2**(1/m) -> 1, which implies that

            # the numbers won't grow

            popt_fatalities = 2, BIG_NUMBER

        print("{}: Optimal parameter value for fatalities: {}".format(full_country, popt_fatalities))

        

        # Select test data for a given country and its province if the latter is given

        condition = (test_df["Province_State"] == province) & (test_df["Country_Region"] == country)

        t_df = test_df[condition]

        

        # Get the initial values to be used for generating future values for confirmed cases and fatalities

        last_train_date = t_df["Date"].values[0]

        print(last_train_date)

        cases = p_df[p_df["Date"] == last_train_date]["ConfirmedCases"].values[0]

        print(cases)

        fatalities = p_df[p_df["Date"] == last_train_date]["Fatalities"].values[0]

        print(fatalities)

        

        # It's necessary to drop index in 't_df': otherwise, 't_df.loc[i, "ForecastId"]' would fail,

        # starting from the second country

        t_df.reset_index(inplace=True, drop=True)

        for i in range(t_df.shape[0]):

            # Get a row index to write to

            idx = t_df.loc[i, "ForecastId"] - 1

            # make predictions

            cases = round(cases * (popt_cases[0] ** (1/popt_cases[1])), 0)

            submission_df.loc[idx, "ConfirmedCases"] = cases

            fatalities = round(fatalities * (popt_fatalities[0] ** (1/popt_fatalities[1])), 0)

            submission_df.loc[idx, "Fatalities"] = fatalities

        

        print("*"*70)
submission_df.to_csv("submission.csv", index=False, header=True)
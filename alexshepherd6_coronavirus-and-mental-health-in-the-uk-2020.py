import re

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.rcParams['figure.figsize'] = (8, 6)

from scipy.stats import pearsonr



from tqdm import tqdm



import tensorflow as tf



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

os.chdir("/kaggle/input/uk-coronavirus-and-mental-health-2020/")
# (Mental) Health Profile of UK Population over time

hp = pd.read_csv("ALL_DATA.csv")

hp_values = hp.iloc[:, 2:].values

# Covid statistics in UK

covid = pd.read_csv("COVID_CASES.csv")

covid["Date"] = pd.to_datetime(covid["Date"], format = "%d/%m/%Y")



# Date ranges to group covid statistics

dr = pd.read_csv("DATE_RANGES.csv")

dr["Start Date"] = pd.to_datetime(dr["Start Date"], format = "%d/%m/%Y")

dr["End Date"] = pd.to_datetime(dr["End Date"], format = "%d/%m/%Y")

dr["Survey"] = np.arange(1,dr.shape[0] + 1)
# For the analysis, we have taken the following stastics from the survey.



hp["Concerns"].unique()
# Group Covid stats by date ranges dictated from the survey data



covid_clean = pd.DataFrame({k:0 for k in covid.columns[1:]}, index = [0])



for entry in tqdm(dr.iterrows()):

    s = entry[1]["Start Date"]

    e = entry[1]["End Date"]

    c_entry = covid[(covid["Date"] >= s) & (covid["Date"] <= e)]

    c_entry = c_entry.sum(axis = 0)

    

    # Date cleaning so it's the same as in hp

    s = s.strftime('%d/%m/%Y')

    e = e.strftime('%d/%m/%Y')

    



    #try:

    covid_clean.loc["{} - {}".format(s, e)] = c_entry.values

    #except Exception as e:

        #print(e)

        #print(c_entry.values)



covid_clean = pd.DataFrame(covid_clean[1:]).reset_index()

covid_clean.columns = ["Date"] + list(covid_clean.columns[1:])

covid_clean.loc[:, "Start Date"] = pd.to_datetime(covid_clean["Date"].str[:10], format = "%d/%m/%Y")



covid_clean = covid_clean.sort_values("Start Date")

for c, colour in zip(covid_clean.columns[1:], ("gold", "orange", "red")):

    try:

        sns.lineplot(x = covid_clean.index, y = covid_clean[c], label = c, color = colour)

    except:

        pass

plt.yscale("log")

plt.xticks(np.arange(covid_clean.shape[0]),covid_clean.Date, rotation = 90)

plt.ylabel("Log Frequency")

plt.xlabel("Survey Period")

plt.title("COVID-19 UK Statistics between \n 20/3/20 - 11/10/2020")

plt.show()
def hp_query(concern = "", demographic = "", mv_win = None):

    """

    Querying health profile dataframe.

    Annoyingly have to include a separate a start date column becuase 

    when plotting, it doesnt put the dates in the correct order.

    

    Parameters:

    

    - concern: (String) Type of concern, as stated in the dataframe

    - demographic: (String) Type of demographic, as stated in the dataframe

    - mv_win: (Integer) Window size for applying moving average

    

    Returns:

    

    df: (Pandas DataFrame) subset as a result from the query

    """

    

    try:

        df = hp[(hp["Concerns"].str.contains(concern)) &

                (hp["Demographics"].str.contains(demographic))]

        if "(" not in concern and "mean" not in concern.lower():

            df = df.drop("Concerns", axis = 1).set_index("Demographics")

        else:

            df = df.drop("Demographics", axis = 1).set_index("Concerns")

           

        df = df.T.dropna()

        

        df = df.reset_index()

        df["index"] = pd.to_datetime(df["index"].str[:10], format = "%d/%m/%Y")

        df.columns = ["Start Date"] + list(df.columns[1:])

        

        df = df.sort_values("Start Date")

        df = df.set_index("Start Date")

        

        if mv_win is not None:

            try:

                df = df.rolling(mv_win).mean()

            except Exception as e:

                print(e)

            

        return df.dropna()

    

    except Exception as e:

        print("Invalid query: {}".format(e))



        



def time_series_plot(concern = "", demographics = [], mv_win = None,

                    colours = ()):

    """

    Plots times series.

    

    Parameters:

    

    - concern: (String) Type of concern, as stated in the dataframe

    - demographic: (List of Strings) Type of demographic, as stated in the dataframe

    - mv_win: (Integer) Window size for applying moving average

    - colours: (Tuple of Strings) colours to give to each level in a concern.

    """

    

    

    if len(demographics) > 2:

        print("Parameter demographic can exceed over 2 elements...")

    elif len(demographics) == 1:

        # Single plot

        

        _ = hp_query(concern = concern, demographic = demographics[0])

            

        if len(colours) > 0:



            for c, colour in zip(_.columns, colours):

                try:

                    sns.lineplot(x = _.index, y = _[c], label = c, color = colour)

                except:

                    pass



        else:



            for c in _.columns:

                try:

                    sns.lineplot(x = _.index, y = _[c], label = c)

                except:

                    pass

        

        plt.xticks(rotation = 90)

        plt.ylabel("Proportion")

        

    else:

        # 2-axes plot

    

        f, (ax1, ax2) = plt.subplots(1,2, sharey = True, figsize = (15, 6))

        

        for d, ax in zip(demographics, (ax1, ax2)):



            _ = hp_query(concern = concern, demographic = d)

            

            if len(colours) > 0:



                for c, colour in zip(_.columns, colours):

                    try:

                        sns.lineplot(x = _.index, y = _[c], label = c, color = colour, ax = ax)

                    except Exception as e:

                        print(e)

            

            else:



                for c in _.columns:

                    

                    try:

                        sns.lineplot(x = _.index, y = _[c], label = c, ax = ax) 

                    except Exception as e:

                        print(e)

                        pass

            

            #plt.yscale("log")

            #ax.set_xticklabels(_.index, rotation = 90)

            ax.set_title(d)

            ax.set_ylabel("Proportion")
time_series_plot(concern = "Coronavirus", demographics = ["Males > 16yo", "Females > 16yo"], mv_win = 4,

                colours = ("darkred", "red", "orange", "yellowgreen", "darkgreen"))



plt.suptitle("COVID-19-related Concerns UK between \n 20/3/20 - 11/10/2020")

plt.show()
time_series_plot(concern = "(\d+)", demographics = ["All"], mv_win = 4)



plt.title("COVID-19 Wellbeing Concerns UK between \n 17/5/20 - 11/10/2020")

plt.show()
time_series_plot(concern = "(\d+)", demographics = ["Males", "Females"], mv_win = 4)

plt.suptitle("COVID-19-related Wellbeing Concerns UK between \n 14/5/20 - 11/10/2020")

plt.show()
time_series_plot(concern = "Mean", demographics = ["Males", "Females"], mv_win = 4)

plt.suptitle("COVID-19 Life Wellbeing Ratings in UK population between \n 3/2020 - 10/2020")

plt.show()
time_series_plot(concern = "loneliness", demographics = ["Males", "Females"], mv_win = 4,

                colours = ("darkred", "red", "orange", "yellowgreen", "darkgreen"))

plt.suptitle("Loneliness in UK population between \n 3/2020 - 10/2020")

plt.show()
# tau - time step interval between health feature and covid infection rate

"""covid infection rate time point will always be before health feature 

since we are assuming that we can predict health feature from covid infection rate.""" 



def signal_correlation(concern = "", demographic = "",

                      hp_feature = "", covid_feature = "",

                      plot = True):



    _ = hp_query(concern = concern, demographic = demographic)

    c = covid_clean[covid_clean["Start Date"].isin(list(_.index))][covid_feature].values

    try:

        _ = _[hp_feature].values

    except:

        print(_)

    

    pearson_coef = []

    prob = []



    for tau in tqdm(np.arange(21)):

        #print(tau)

        try:

            if tau > 0:

                p = pearsonr(c[:-tau],_[tau:])

            else:

                p = pearsonr(c,_)

            pearson_coef.append(p[0])

            prob.append(p[1])

        except Exception as e:

            print(e)

            break

    

    pearson_coef = np.array(pearson_coef)

    opt_idx = np.argmin(np.array(prob))

    

    tau_star = opt_idx

    pearson_r_star = pearson_coef[opt_idx] 

    

    print("Optimal tau: {}".format(tau_star))

    print("Optimal pearson coefficent: {}".format(pearson_r_star))

    

    if plot:

        

        if tau_star > 0:

            sns.regplot(x = c[:-tau_star], y = _[tau_star:], order = 1)

        else:

            sns.regplot(x = c, y = _, order = 1)



        plt.title("Relationship between {} and {} \n tau = {}, pearson r = {}".format(hp_feature, covid_feature,

                                                                                     tau_star, round(pearson_r_star,4)))

        plt.ylabel(hp_feature)

        plt.xlabel(covid_feature + " {} week(s) later".format(tau_star))

        plt.show()

    

    return pearson_coef, tau_star, pearson_r_star, min(prob)





def correlation_analysis(c_feature):

    """

    Runs signal correlation over all health features to output

    Pandas DataFrame containing all pearson r correlation coefficients and 

    optimal tau (time step difference) and probability associated with the 

    optimal tau.

    

    Parameters:

    

    c_feature: covid feature (e.g. cases, hospitalisations, deaths)

    """

    

    df = None

    

    for entry in tqdm(hp.iterrows()):

        

        cncrn = entry[1]["Concerns"]

        dmg = entry[1]["Demographics"]

        

        if "(" in cncrn:

            cncrn = re.findall(string = cncrn, pattern = "\(\d+\)")[0]

        

        if "-" in dmg:

            new_entry, tau_star , _, prob = signal_correlation(concern = cncrn,

                                                             demographic = dmg,

                                                             hp_feature = entry[1]["Demographics"],

                                                             covid_feature = c_feature,

                                                             plot = False) 

        else:

            new_entry, tau_star , _, prob = signal_correlation(concern = cncrn,

                                                             demographic = dmg,

                                                             hp_feature = entry[1]["Concerns"],

                                                             covid_feature = c_feature,

                                                             plot = False) 

        # reformat new entry for output (df)

        new_entry = np.append(np.append(new_entry, tau_star), prob)

        

        #add to df

        

        try:

            df = np.vstack([df,new_entry])

        except:

            df = new_entry

            

    df = pd.DataFrame(df, columns = list(np.arange(0,21)) + ["tau*", "prob*"])

    df.insert(0, "Concern", hp["Concerns"])

    df.insert(1, "Demographic", hp["Demographics"])

    

    return df
corr_df = correlation_analysis(c_feature = "New COVID Cases")
sns.distplot(corr_df["tau*"], bins = 10)

plt.title("Distribution of Optimal tau for correlation \n between health features and covid infection rates")

plt.ylabel("Frequency Density")

plt.xlabel("Optimal tau")

plt.show()
corr_df[corr_df["tau*"] <= 10]

# finding optimal correlations with pearson |r| > 0.5 (Strong correlation)

# Constrained tau* <= 10 since number of data points to instantiate linear model becomes small.



corr_df = corr_df[corr_df["tau*"] <= 10]



corr_df_clean = None



for entry in corr_df.iterrows():

    #print(entry[1]["Concern":"Demographic"].values)

    tau_star = entry[1]["tau*"]

    optimal_r = entry[1][int(tau_star)]

    if abs(optimal_r) >= 0.5:

        _ = pd.DataFrame(entry[1]["Concern":"Demographic"]).T

        _.loc[:, "Optimal pearson r"] = optimal_r

        _.loc[:, "tau*"] = entry[1]["tau*"]

        _.loc[:, "prob*"] = entry[1]["prob*"]

        try:

            corr_df_clean = pd.concat([corr_df_clean, _])

        except Exception as e:

            print(e)

            corr_df_clean = _
corr_df_clean
cross_corr_check = hp.iloc[corr_df_clean.index, :]

cross_corr_check.loc[:, "Feature"] = cross_corr_check.loc[:, "Concerns"] + " - \n" + cross_corr_check.loc[:, "Demographics"]
cross_corr_check = cross_corr_check.drop(["Concerns", "Demographics", "Unnamed: 28"], axis = 1).set_index("Feature").T
sns.pairplot(cross_corr_check)

plt.show()
cross_corr_check = cross_corr_check.corr()

sns.heatmap(cross_corr_check, 

        xticklabels=cross_corr_check.columns,

        yticklabels=cross_corr_check.columns,

           cmap = "bwr",

           annot = True)



plt.show()
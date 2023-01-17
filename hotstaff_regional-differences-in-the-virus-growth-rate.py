import datetime

import random



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



from scipy.optimize import curve_fit # curve_fit

import matplotlib.pyplot as plt # draw

import matplotlib.dates as mdates # draw



from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



from tqdm.notebook import tqdm
TRAIN = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")



# Change astype

TRAIN["Date"] = pd.to_datetime(TRAIN["Date"])

TRAIN["ConfirmedCases"] = TRAIN["ConfirmedCases"].fillna(0).astype('int')

TRAIN["Fatalities"] = TRAIN["Fatalities"].fillna(0).astype('int')



unique_date = TRAIN.Date.unique()

print(f"Train: {unique_date[0]} to  {unique_date[-1]}") 
csid_to_name = pd.concat([TRAIN[0: :len(unique_date)].Province_State,

                          TRAIN[0: :len(unique_date)].Country_Region]

                         , axis=1).reset_index(drop=True)

csid_to_name["CSId"] = csid_to_name.index + 1

csid_to_name = csid_to_name.set_index("CSId")

csid_to_name
def logistic(t, K, P0, r, a):

    return  (K / (1 + ((K - P0) / P0) 

                  * np.exp(-r * (t - a))))



def d_logistic(t, K, P0, r, a):

    return ((K * P0 * r * (K - P0) * np.exp(r * (t - a)))

            / (K + P0 * (np.exp(r * (t - a)) - 1))**2)



def f(t, K, r, a):

    # Fix P0 and do the fitting, but instead let the function estimate a,

    # a parameter that represents the time when P0 = 1.

    return logistic(t, K, 1, r, a)



def df(t, K, r, a):

    return d_logistic(t, K, 1, r, a)



valnames = list(f.__code__.co_varnames[:f.__code__.co_argcount][1:])
def prepare(column, csid):

    length = 114

    df = TRAIN[(TRAIN.Id >= (csid-1)*length) 

               & (TRAIN.Id <= csid*length)]

    # df -> list

    ret = {

        "Date": df.Date,

        "Days": list(range(0, len(df.Date.unique()))),

        "Values": df[column].values.tolist(),

        "Diff": df[column].diff().fillna(0).values.tolist()

    }



    return ret





def fitting(prepared, retry=0, r0=0.5):

    # return popt, pcov

    x_values = prepared["Days"]

    y_values = prepared["Values"]

    

    # initial values

#     p0 = [max(y_values), min(y_values), r0, 0]

    p0 = [max(y_values), r0, 0]

    # bounds

    bounds = ((0, 0, -100), (10**10, 2, 100))

    

    try:

        popt, pcov = curve_fit(f,

                               x_values,

                               y_values,

                               p0=p0,

                               bounds=bounds,

                               method="trf",

                               maxfev=1000000)

    except Exception as e:

        if retry > 0:

            return fitting(prepared, retry - 1, r0 = random.random())

        print(e)

        return None, None

    return popt, pcov







def draw(prepared, popt, nday, title, column="Confirmed"):

    x_values_date = prepared["Date"].reset_index(drop=True)

    origin_date = x_values_date[0]

    x_values = prepared["Days"]

    y_values = prepared["Values"]

    diff_values = prepared["Diff"]

    

    # init main graph

    fig = plt.figure(figsize=(8, 5))

    ax1 = fig.add_subplot(1, 1, 1)

    ax2 = ax1.twinx()



    # main graph captions

    plt.suptitle(title, fontweight="bold")

    plt.xlabel('Date')

    ax1.set_ylabel(f"{column}")

    ax2.set_ylabel(f"New {column}")



    # main fitting plot

    xx = np.linspace(0, x_values[-1] + nday, 100)

    xx_date = [origin_date + datetime.timedelta(days=x) for x in xx]

 

    yy = f(xx, *popt)

    dyy = df(xx, *popt)



    ax1.set_xlim(xx_date[0], xx_date[-1])

    ax1.set_ylim(0, yy[-1])



    ax1.plot(x_values_date, y_values, 'o', label=f'{column}')

    ax1.plot(xx_date, yy, label=f"{column} (Fitting)")



    ax2.plot(x_values_date, diff_values, label=f"New {column}")

    ax2.plot(xx_date, dyy, color="r", label=f"New {column} (Fitting)")



    handler1, label1 = ax1.get_legend_handles_labels()

    handler2, label2 = ax2.get_legend_handles_labels()

    plt.legend(handler1 + handler2, label1 + label2, loc=2)

 

    major_locator = mdates.WeekdayLocator(interval=3)

    major_formatter = mdates.AutoDateFormatter(major_locator)

    ax1.xaxis.set_major_locator(major_locator)

    ax1.xaxis.set_major_formatter(major_formatter)



    plt.show()





def fit(column, start=1, end=313, draw_nday=None):

    # return parameter dataframe

    popts = []

    index = []

    csids = range(start, end + 1)

    for csid in tqdm(csids, desc=f"fit progress: "):

        # data dict prepare

        prepared = prepare(column, csid)

        # curve fitting

        popt, pcov = fitting(prepared, retry=10)

        # standard deviation errors

        perr = np.sqrt(np.diag(pcov))

        # multi index

        key = csid_to_name.loc[csid].tolist()

        index.append(key)



        if popt is not None:

            # Success

            popts.append(np.concatenate((popt, perr)))

        else:

            # Failed

            popts.append(np.array([None] * 8))

            print(f"Failed fitting {column} {csid} {key} ")



    index = pd.MultiIndex.from_tuples(index, 

                                      names=['Province_State', 'Country_Region'])



    errnames = list(map(lambda x: f"S_{x}", valnames))

    columns = valnames + errnames



    popts_df = pd.DataFrame(popts, columns=columns, index=index)

    popts_df["CSId"] = csids



    return popts_df.set_index("CSId")





def predict(prepared, popt, nday):

    x_values_date = prepared["Date"].reset_index(drop=True)

    origin_date = x_values_date[0]

    x_values = prepared["Days"]



    xx = np.arange(0, x_values[-1] + nday + 1, 1)

    xx_date = [origin_date + datetime.timedelta(days=int(x)) for x in xx]



    return xx_date, f(xx, *popt)
TRAIN.loc[TRAIN["ConfirmedCases"] < 0, ["ConfirmedCases", "Fatalities"]] = 0

# TRAIN.loc[TRAIN["ConfirmedCases"] < 0]
popts_df_Confirmed = fit("ConfirmedCases")

popts_df_Confirmed
popts_df_Fatalities = fit("Fatalities")

popts_df_Fatalities
def outlier_detection(series, sigma):

    me = series.mean()

    sd = series.std()

    return (me - sigma * sd, me + sigma * sd)



def outlier(df, sigma=2):

    for i in range(len(df.columns)):

        col = df.iloc[:,i]

        o_min, o_max = outlier_detection(col, sigma)

        col[col < o_min] = np.nan

        col[col > o_max] = np.nan

    return df
popts_c_clean = pd.merge(outlier(popts_df_Confirmed[valnames]), csid_to_name, left_on="CSId", right_on="CSId")

popts_c_clean = popts_c_clean.dropna()

popts_f_clean = pd.merge(outlier(popts_df_Fatalities[valnames]), csid_to_name, left_on="CSId", right_on="CSId")

popts_f_clean = popts_f_clean.dropna()
g = sns.relplot(x="a", y="r", hue="Country_Region", size="K",

                sizes=(40, 400), alpha=.7, palette="muted",

                height=6, data=popts_c_clean)

g.fig.suptitle("ConfirmedCases")

plt.show()
g = sns.relplot(x="a", y="r", hue="Country_Region", size="K",

                sizes=(40, 400), alpha=.7, palette="muted",

                height=6, data=popts_f_clean)

g.fig.suptitle("Fatalities")

plt.show()
g = (sns.jointplot("a", "r",

                   data=popts_c_clean, color="k", ylim=(0, 1.0))

     .plot_joint(sns.kdeplot, zorder=0, n_levels=11))



g.fig.suptitle("ConfirmedCases")

plt.show()
g = (sns.jointplot("a", "r",

                   data=popts_f_clean, color="k", ylim=(0, 1.0))

     .plot_joint(sns.kdeplot, zorder=0, n_levels=11))

g.fig.suptitle("Fatalities")

plt.show()
mean_c = popts_c_clean.groupby("Country_Region").mean()

mean_c.sort_values("a")
mean_f = popts_f_clean.groupby("Country_Region").mean()

mean_f.sort_values("a")
PLOT = False

nday = 114 - len(TRAIN.Date.unique())



past = []

forecast = []

for csid in range(1, 313 + 1):

    # (start, end)

    forecastid = (1+(csid-1)*43, 43*(csid))

    

    # select popt

    popt_C = popts_df_Confirmed.loc[csid][valnames].values

    popt_F = popts_df_Fatalities.loc[csid][valnames].values

    

    # select values

    prepared_C = prepare("ConfirmedCases", csid)

    prepared_F = prepare("Fatalities", csid)

    

    # predict

    date_C, y_C = predict(prepared_C, popt_C, nday)

    date_F, y_F = predict(prepared_F, popt_F, nday)

    

    # plot

    if PLOT:

        (state, country) = csid_to_name.loc[csid].tolist()

        if type(state) != str and np.isnan(state):

            title = f"COVID-19 Logistic Function Fitting {csid} - {country}"

        else:

            title = f"COVID-19 Logistic Function Fitting {csid} - {state} - {country}"

        

        if not np.isnan(popt_C[0]):

            draw(prepared_C, popt_C, nday, title, column="ConfirmedCases")

        if not np.isnan(popt_F[0]):

            draw(prepared_F, popt_F, nday, title, column="Fatalities")

    

    forecast.append(pd.DataFrame({

        "ForecastId": range(forecastid[0], forecastid[1] + 1), 

        "ConfirmedCases": y_C[-43:],

        "Fatalities": y_F[-43:]})

    )



    # for debug

#     print(f"Write {csid} {forecastid} {csid_to_name.loc[csid].values}\n"

#           f" {date_C[-43].strftime('%Y-%m-%d')} {y_C[-43].round(0)} - {date_C[-1].strftime('%Y-%m-%d')} {y_C[-1].round(0)}\n"

#           f" {date_F[-43].strftime('%Y-%m-%d')} {y_F[-43].round(0)}- {date_F[-1].strftime('%Y-%m-%d')} {y_F[-1].round(0)}\n")



forecast = pd.concat(forecast, axis=0, sort=False).round(0).fillna(0).set_index("ForecastId")

forecast
TEST = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv").set_index("ForecastId")

TEST["Date"] = pd.to_datetime(TEST["Date"])

MERGED_TEST = pd.merge(TEST, forecast, left_on="ForecastId", right_on="ForecastId")



# check Newyork is 266

MERGED_TEST[43*(266-1)-1: 43*266+1]
LAST_UPDATE_STRING = pd.to_datetime(unique_date[-1]).strftime('%Y-%m-%d')

c_sum = MERGED_TEST.groupby(["Country_Region", "Date"]).sum()

plt.close("all")

shows = ["US","Germany", "Spain", "France", "Italy", "Iran", "Turkey"]

ax = c_sum.loc[(shows, )]["ConfirmedCases"].unstack(0).plot(figsize=(8, 5), logy=False,

                                                       title=f"Prediction ConfirmedCases as of {LAST_UPDATE_STRING}")

plt.show()
world_obs = TRAIN.groupby(["Date"]).sum().drop(columns=["Id"])

world_pred = MERGED_TEST.groupby(["Date"]).sum()



world_pred["CFR"] = world_pred["Fatalities"] * 100 / world_pred["ConfirmedCases"]

world_obs["CFR"] = world_obs["Fatalities"] * 100 / world_obs["ConfirmedCases"]



world = pd.merge(world_obs, world_pred, on="Date", how='outer', suffixes=['_observed', '_predicted'])

y1col = ["ConfirmedCases_observed", "Fatalities_observed","ConfirmedCases_predicted", "Fatalities_predicted"]

y2col = ["CFR_observed", "CFR_predicted"]

ax = world[y1col].plot.line(logy=True, style=['r.', 'b.', 'r--', 'b--'], legend=False)

ax2 = world[y2col].plot.line(logy=False, figsize=(8, 5), style=['g.', 'g--'],

                title = f"World Prediction as of {LAST_UPDATE_STRING}",

                secondary_y=['CFR_observed', 'CFR_predicted'], ax=ax, legend=True)

lines = ax.get_lines() + ax2.get_lines()

ax.set_ylabel("Cases")

ax2.set_ylabel("CFR")

ax2.legend(lines, [l.get_label() for l in lines])



plt.show()
# submission

forecast.to_csv("./submission.csv")
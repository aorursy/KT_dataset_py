import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os



import pylab as pl

import datetime



import matplotlib.pyplot as plt

import matplotlib.dates as mdates



from scipy.optimize import curve_fit
pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")
TRAIN = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")



# Change astype

TRAIN["Date"] = pd.to_datetime(TRAIN["Date"])

TRAIN["ConfirmedCases"] = TRAIN["ConfirmedCases"].fillna(0).astype('int')

TRAIN["Fatalities"] = TRAIN["Fatalities"].fillna(0).astype('int')

TRAIN
csid_to_name = pd.concat([TRAIN[0: :70].Province_State, TRAIN[0: :70].Country_Region], axis=1).reset_index(drop=True)

csid_to_name["CSId"] = csid_to_name.index + 1

csid_to_name = csid_to_name.set_index("CSId")

csid_to_name
# fitting functions

def f(t, K, P0, r, a):

    return  (K / (1 + ((K - P0) / P0) * np.exp(-r * (t - a))))



def df(t, K, P0, r, a):

    return (K * P0 * r * (K - P0) * np.exp(r * (t - a)))/(K + P0 * (np.exp(r * (t - a)) - 1))**2



def prepare(column, csid):

    df = TRAIN[(TRAIN.Id >= (csid-1)*100)  & (TRAIN.Id <= csid*100)]

    # df -> list

    ret = {

        "Date": df.Date,

        "Days": list(range(0, 70)),

        "Values": df[column].values.tolist(),

        "Diff": df[column].diff().fillna(0).values.tolist()

    }



    return ret
def fitting(prepared):    

    # Fitting

    # return popt, pcov

    x_values = prepared["Days"]

    y_values = prepared["Values"]

    try:

        popt, pcov = curve_fit(f, x_values, y_values,

                                 p0=[max(y_values), min(y_values), 0.5, 0],

                                 bounds=((0, 0, 0, -100), (10**10, 5000, 1, 100)),

                                 method="trf", maxfev=1000000)

    except Exception as e:

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

    fig = plt.figure(figsize=(12, 8))

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

 

    major_locator = mdates.WeekdayLocator(interval=2)

    major_formatter = mdates.AutoDateFormatter(major_locator)

    ax1.xaxis.set_major_locator(major_locator)

    ax1.xaxis.set_major_formatter(major_formatter)



    plt.show()
def fit(column, start=1, end=294, draw_nday=30):

    # return parameter dataframe

    popts = []

    index = []

    csids = range(start, end + 1)

    for csid in csids:

        # Data dict prepare

        prepared = prepare(column, csid)

        # Curve fitting

        popt, pcov = fitting(prepared)

        # Standard deviation errors

        perr = np.sqrt(np.diag(pcov))

        # Multi index

        key = csid_to_name.loc[csid].tolist()

        index.append(key)



        if popt is not None and popt[0] < 10**7:

            # Success

            popts.append(np.concatenate((popt, perr)))

            # Draw

            draw(prepared, popt,

                 draw_nday, f"COVID-19 Logistic Function Fitting {key[0]} - {key[1]}",

                 column="Confirmed")

        else:

            # Failed

            popts.append(np.array([None] * 8))

            print(f"Failed fitting {csid} {key} ")



    index = pd.MultiIndex.from_tuples(index, 

                                      names=['Province_State', 'Country_Region'])

    popts_df = pd.DataFrame(popts,

                            columns=["K", "P0", "r", "a", "S_K", "S_P0", "S_r", "S_a"],

                            index=index)

    popts_df["CSId"] = csids



    return popts_df.set_index("CSId")
def predict(prepared, popt, nday):

    x_values_date = prepared["Date"].reset_index(drop=True)

    origin_date = x_values_date[0]

    x_values = prepared["Days"]



    # main fitting plot

    xx = np.arange(0, x_values[-1] + nday + 1, 1)

    xx_date = [origin_date + datetime.timedelta(days=int(x)) for x in xx]



    return xx_date, f(xx, *popt)



popts_df_Confirmed = fit("ConfirmedCases")

popts_df_Fatalities = fit("Fatalities")
forecast = []

for csid in range(1, 294 + 1):

    # (start, end)

    forecastid = (1+(csid-1)*43, 43*(csid))

    

    # select popt

    popt_C = popts_df_Confirmed.loc[csid][["K", "P0", "r", "a"]].values

    popt_F = popts_df_Fatalities.loc[csid][["K", "P0", "r", "a"]].values

    

    # predict

    date_C, y_C = predict(prepare("ConfirmedCases",csid), popt_C, 30)

    date_F, y_F = predict(prepare("Fatalities",csid), popt_F, 30)

    

    forecast.append(pd.DataFrame({

        "ForecastId": range(forecastid[0], forecastid[1] + 1), 

        "ConfirmedCases": y_C[-43:],

        "Fatalities": y_F[-43:]})

    )

    # for debug

#     print(f"Write {csid} {forcastid} {csid_to_name.loc[csid].values}\n"

#           f" {date_C[-43].strftime('%Y-%m-%d')} {y_C[-43].round(0)} - {date_C[-1].strftime('%Y-%m-%d')} {y_C[-1].round(0)}\n"

#           f" {date_F[-43].strftime('%Y-%m-%d')} {y_F[-43].round(0)}- {date_F[-1].strftime('%Y-%m-%d')} {y_F[-1].round(0)}\n")



forecast = pd.concat(forecast, axis=0, sort=False).round(0).fillna(0).set_index("ForecastId")

forecast
TEST = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv").set_index("ForecastId")

MERGED_TEST = pd.merge(TEST, forecast, left_on="ForecastId", right_on="ForecastId")

MERGED_TEST
forecast.to_csv("submission.csv")
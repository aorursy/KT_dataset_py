import numpy as np

import pandas as pd

import warnings



# 警告表示の制御

warnings.simplefilter('ignore', FutureWarning)

# dataframeの全ての列を表示するように設定

pd.set_option('display.max_rows', None)

# ファイル読み込み

corona_df = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')

# 日付のフォーマットを統一

corona_df["Date"] = list(map(lambda x: pd.datetime.strptime(x, '%m/%d/%y'),corona_df["Date"]))

# カラム名の変更

corona_df["Country"] = corona_df["Country/Region"]

# 最新の日付の取得

latest_date = corona_df["Date"].max().strftime("%Y/%m/%d")



# 国と日付ごとにグループ化

grouped_corona_df = corona_df.groupby(["Date", "Country"], as_index=False)

# 地域と州の数を国と日付毎に数えます

prov_counted_df = grouped_corona_df["Province/State"].count()

# 地域と州の数が2つ以上の国のリストを作成します

countries_with_prov = prov_counted_df[prov_counted_df["Province/State"]>1]["Country"].unique()

# 地域と州の数が2つ以上の国を表示します

display(countries_with_prov)
# 最新の日付で、地域と州の数が2つ以上の国の情報のみをみてみます

corona_prov_df = corona_df[(corona_df["Date"] == latest_date) & (corona_df["Country"].isin(countries_with_prov))]

# 内容を表示します

display(corona_prov_df.head(25))
# 全ての国ごとに累計罹患者数が最大の行を取り出し、首都の緯度と経度を見つけます

capital_df = corona_df.loc[corona_df[corona_df["Date"] == latest_date].groupby(["Country"], as_index=False)["Confirmed"].idxmax(),:][["Country", "Lat", "Long"]]



# 内容を表示します

display(capital_df.head(25))
# グループ化したもののうち「累計罹患者数」「死者数」「回復者数」は合計を取り新たなcorona_dfとして置き換え

corona_df = grouped_corona_df[["Confirmed", "Deaths", "Recovered"]].sum()



# 「緯度」「経度」については先ほど作成したcapital_dfから国ごとに追加

corona_df = pd.merge(corona_df, capital_df, on="Country", how="left")



# データのある国名をリストで取得

countries = corona_df["Country"].unique()



# データにある国の数

num_countries = len(countries)



# 日付、累計罹患者数、国と地域でソートします

corona_df = corona_df.sort_values(by=["Date", "Confirmed", "Country"])



# 内容の表示

display(corona_df[["Country", "Confirmed", "Deaths", "Recovered", "Lat", "Long"]].tail(25)[::-1].style.background_gradient(cmap='Reds'))
# 現在の罹患者数カラムを追加

corona_df["Infected"] = corona_df["Confirmed"] - corona_df["Recovered"] - corona_df["Deaths"]



# データの情報を表示（国の数とデータの期間）

display("Number of COVID19 confirmed/deaths/recovered in total of " + str(num_countries) + " countries through " +

        corona_df["Date"].min().strftime("%Y/%m/%d") + " to " + latest_date)



# データの情報を表示（罹患者数、新規罹患者数、死亡者数、致死率）

display("Cumulated infected number = " + "{:,}".format(corona_df["Confirmed"].sum()) +  ", Current infected number = " + "{:,}".format(corona_df["Infected"].sum()) +

        ", total deaths = " + "{:,}".format(corona_df["Deaths"].sum()))



# 前日比と差分を追加

corona_df = corona_df.sort_values(by=["Country", "Date"])

corona_df["New_Confirmed"] = corona_df.groupby(["Country"])["Confirmed"].diff().fillna(0)

corona_df["New_Deaths"] = corona_df.groupby(["Country"])["Deaths"].diff().fillna(0)

corona_df["Fatality_Rate"] = corona_df["Deaths"]*100/corona_df["Confirmed"]

pd.set_option("display.max_rows", len(countries))



# 日付、罹患者数、国と地域でソートします

corona_df = corona_df.sort_values(by=["Date", "Confirmed", "Country"])



# 罹患者数が多い順に上位50ヵ国の最新の情報を表示します

display(corona_df[["Country", "Confirmed", "Infected", "New_Confirmed", "New_Deaths", "Fatality_Rate"]].tail(50)[::-1].style.background_gradient(cmap='Reds'))

countries50 = corona_df.tail(50)[::-1]["Country"].values
import folium

map_object = folium.Map(location=[0, 10], min_zoom=2, max_zoom=10, zoom_start=2)



for_map = corona_df.tail(num_countries)

for_map = for_map.dropna(subset=["Deaths"]).reset_index(drop=True)



for i in range(0, len(for_map)):

    folium.Circle(location=[for_map.iloc[i]["Lat"], for_map.iloc[i]["Long"]], color='crimson', fill=True,

        tooltip =   '<li><bold>Country : '+str(for_map.iloc[i]["Country"])+

                    '<li><bold>Deaths : '+str(for_map.iloc[i]["Deaths"]),

        radius=int(for_map.iloc[i]["Deaths"])*20).add_to(map_object)

map_object
# 不要な要素（緯度・経度）を削除

corona_df = corona_df.drop(columns=["Lat", "Long"])



country_df = corona_df[corona_df["Country"]=="Japan"].copy()

display(country_df.tail(30))
import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import math

%matplotlib inline



fig_rows = math.ceil((len(corona_df.columns) - 2)/2)

fig = plt.figure(figsize=(40,60),facecolor='white')

plt.rcParams["font.size"] = 18



for i, col in enumerate(corona_df.columns):

  temp_df = country_df.copy()

  if col == "Date" or col =="Country":

    continue

  plt.subplot(fig_rows,2,i-1)

  plt.title(col)

  plt.plot(temp_df["Date"], temp_df[col], label="Japan")

  plt.yscale("linear")

  if col == "Fatality_Rate":

    plt.ylim([0, 5])

  plt.grid(True)

plt.legend(loc="upper center", bbox_to_anchor=(0.5,-0.05),ncol=6)

plt.show()
fig_rows = math.ceil((len(corona_df.columns) - 2)/2)

fig = plt.figure(figsize=(40,60),facecolor='white')

plt.rcParams["font.size"] = 18



for i, col in enumerate(corona_df.columns):

  temp_df = country_df.copy()

  if col == "Date" or col =="Country":

    continue

  plt.subplot(fig_rows,2,i-1)

  plt.title(col)

  plt.plot(temp_df["Date"], temp_df[col], label="Japan")

  plt.yscale("log")

  if col == "Fatality_Rate":

    plt.yscale("linear")

    plt.ylim([0, 5])

  plt.grid(True)

plt.legend(loc="upper center", bbox_to_anchor=(0.5,-0.05),ncol=6)

plt.show()
fig_rows = math.ceil((len(corona_df.columns) - 2)/2)

fig = plt.figure(figsize=(40,60),facecolor='white')

plt.rcParams["font.size"] = 18

for i, col in enumerate(corona_df.columns):

  temp_df = country_df.copy()

  if col == "Date" or col =="Country":

    continue

  plt.subplot(fig_rows,2,i-1)

  plt.title(col)

  plt.plot(temp_df["Date"], temp_df[col], label="Raw")

  plt.plot(temp_df["Date"], temp_df[col].rolling(3).mean().shift(-1), label="MA3")

  plt.plot(temp_df["Date"], temp_df[col].rolling(5).mean().shift(-3), label="MA5")

  plt.yscale("log")

  if col == "Fatality_Rate":

    plt.yscale("linear")

    plt.ylim([0, 5])

  plt.grid(True)

plt.legend(loc="upper center", bbox_to_anchor=(0.5,-0.05),ncol=6)

plt.show()
fig_rows = math.ceil((len(corona_df.columns) - 2)/2)

fig = plt.figure(figsize=(40,60),facecolor='white')

plt.rcParams["font.size"] = 18

for i, col in enumerate(corona_df.columns):

  if col == "Date" or col =="Country":

    continue

  plt.subplot(fig_rows,2,i-1)

  plt.title(col + "_MA5-3")

  for country in corona_df["Country"].unique():

    country_df = corona_df[corona_df["Country"]==country].copy()

    plt.plot(country_df["Date"], country_df[col].rolling(5).mean().shift(-3), label=country)

    plt.yscale("log")

  if col == "Fatality_Rate":

    plt.yscale("linear")

    plt.ylim([0, 20])

  plt.grid(True)

plt.legend(loc="upper center", bbox_to_anchor=(0.5,-0.05),ncol=6)

plt.show()
day_one = corona_df[(corona_df["Confirmed"]>1) & (corona_df["Country"].isin(countries50))].groupby(["Country"])["Date"].min()

display(day_one.sort_values().head(20))
day_one_shift = day_one - day_one["Japan"]

display(day_one_shift.sort_values().head(20))
from datetime import timedelta



fig_rows = math.ceil((len(corona_df.columns) - 2)/2)

fig = plt.figure(figsize=(40,60),facecolor='white')

plt.rcParams["font.size"] = 18



for i, col in enumerate(corona_df.columns):

  if col == "Date" or col =="Country":

    continue

  plt.subplot(fig_rows,2,i-1)

  plt.title(col + "_MA5-3")

  for country in countries50:

    country_df = corona_df[corona_df["Country"]==country].copy()

    if country in day_one_shift.index:

      plt.plot(country_df["Date"] - timedelta(days=day_one_shift[country].days), country_df[col].rolling(5).mean().shift(-3), label=country)    

    plt.yscale("log")

  if col == "Fatality_Rate":

    plt.yscale("linear")

    plt.ylim([0, 20])

  plt.grid(True)

plt.legend(loc="upper center", bbox_to_anchor=(0.5,-0.05),ncol=6)

plt.show()
day_one = corona_df[(corona_df["Deaths"]>2) & (corona_df["Country"].isin(countries50))].groupby(["Country"])["Date"].min()

day_one_shift = day_one - day_one["Japan"]

fig_rows = math.ceil((len(corona_df.columns) - 2)/2)

fig = plt.figure(figsize=(40,60),facecolor='white')

plt.rcParams["font.size"] = 18



for i, col in enumerate(corona_df.columns):

  if col == "Date" or col =="Country":

    continue

  plt.subplot(fig_rows,2,i-1)

  plt.title(col + "_MA5-3")

  for country in countries50:

    country_df = corona_df[corona_df["Country"]==country].copy()

    if country in day_one_shift.index:

      plt.plot(country_df["Date"] - timedelta(days=day_one_shift[country].days), country_df[col].rolling(5).mean().shift(-3), label=country)  

    plt.yscale("log")

  if col == "Fatality_Rate":

    plt.yscale("linear")

    plt.ylim([0, 20])

  plt.grid(True)

plt.legend(loc="upper center", bbox_to_anchor=(0.5,-0.05),ncol=6)

plt.show()
df_pop = pd.read_csv('../input/population-by-country-2020/population_by_country_2020.csv')

N = df_pop[df_pop["Country (or dependency)"] == "Japan"]["Population (2020)"].values[0]

display(N)
from scipy.integrate import odeint

from matplotlib.ticker import ScalarFormatter



def logistic_R_0(t, R_0_start, k, x0, R_0_end):

    return (R_0_start-R_0_end) / (1 + np.exp(-k*(-t+x0))) + R_0_end



def func_SIR(initial, t, N, R_0_start, k, x0, R_0_end, gamma):

    S, I, R = initial

    beta = logistic_R_0(t, R_0_start, k, x0, R_0_end) * gamma

    dSdt = -beta * S * I/ N

    dIdt = beta * S * I / N - gamma * I

    dRdt = gamma * I

    return dSdt, dIdt, dRdt



full_days = 250

D = 13.5

gamma = 1.0 / D

t = np.linspace(0, full_days-1, full_days)

initial = N-1.0, 1.0, 0.0



def Model(days, N, R_0_start, k, x0, R_0_end):

    t = np.linspace(0, days-1, days)

    ret = odeint(func_SIR, initial, t, args=(N, R_0_start, k, x0, R_0_end, gamma))

    S, I, R = ret.T

    R_0_over_time = [logistic_R_0(i, R_0_start, k, x0, R_0_end) for i in range(len(t))]

    return t, S, I, R, R_0_over_time



outbreak_shift = 30

fatal_rate = corona_df[corona_df["Country"] == "Japan"]["Fatality_Rate"].values[[-1]]/100

print(fatal_rate)



# plt.gcf().subplots_adjust(bottom=0.15)



def plotter(t, S, I, R, R_0, x_ticks=None):

    f, ax = plt.subplots(1,1,figsize=(20,4))

    if x_ticks is None:

        ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')

        ax.plot(t, R*fatal_rate, 'b', alpha=0.7, linewidth=2, label='Fatalities')

    else:

        ax.plot(x_ticks, I, 'r', alpha=0.7, linewidth=2, label='Infected')

        ax.plot(x_ticks, R*fatal_rate, 'g', alpha=0.7, linewidth=2, label='Fatalities')



        ax.xaxis.set_major_locator(mdates.MonthLocator())

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        f.autofmt_xdate()



    ax.title.set_text('SIR-Model')



    ax.grid(b=True, which='major', c='w', lw=2, ls='-')

    legend = ax.legend()

    legend.get_frame().set_alpha(0.5)

    for spine in ('top', 'right', 'bottom', 'left'):

        ax.spines[spine].set_visible(False)



    plt.show();

    

    f = plt.figure(figsize=(20,4))



    ax1 = f.add_subplot(131)

    if x_ticks is None:

        ax1.plot(t, R_0, 'b--', alpha=0.7, linewidth=2, label='R_0')

    else:

        ax1.plot(x_ticks, R_0, 'b--', alpha=0.7, linewidth=2, label='R_0')

        ax1.xaxis.set_major_locator(mdates.MonthLocator())

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        f.autofmt_xdate()



 

    ax1.title.set_text('R_0 over time')

    ax1.grid(b=True, which='major', c='w', lw=2, ls='-')

    legend = ax1.legend()

    legend.get_frame().set_alpha(0.5)

    for spine in ('top', 'right', 'bottom', 'left'):

        ax.spines[spine].set_visible(False)



    plt.show();

    

first_date = np.datetime64(corona_df["Date"].min()) - np.timedelta64(outbreak_shift,'D')

x_ticks = pd.date_range(start=first_date, periods=full_days, freq="D")

    

plotter(*Model(full_days, N, 5.0, 1.0, 60, 1.0), x_ticks=x_ticks)



display(day_one["Japan"])

# def end99(t, day1, I, R):



# display(t[I<R.max()*0.01])

# display("Final R = " + "{:,}".format(int(R.max())))
# カーブフィッティングライブラリのインストール

!pip install lmfit
import lmfit



data = corona_df[corona_df["Country"] == "Japan"]["Infected"].values

days = outbreak_shift + len(data)

if outbreak_shift >= 0:

    y_data = np.concatenate((np.zeros(outbreak_shift), data))

else:

    y_data = y_data[-outbreak_shift:]



x_data = np.linspace(0, days - 1, days, dtype=int)  # x_data is just [0, 1, ..., max_days] array



def fitter(x, R_0_start, k, x0, R_0_end):

    ret = Model(days, N, R_0_start, k, x0, R_0_end)

    return ret[3][x]



mod = lmfit.Model(fitter)



params_init_min_max = {"R_0_start": (5.0, 2.0, 5.8), "k": (2.5, 0.01, 5.0), "x0": (90, 0, 120), "R_0_end": (0.9, 0.3, 3.5)}

# form: {parameter: (initial guess, minimum value, max value)}



for kwarg, (init, mini, maxi) in params_init_min_max.items():

    mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)



params = mod.make_params()

fit_method = "leastsq"



result = mod.fit(y_data, params, method="least_squares", x=x_data)

result.plot_fit(datafmt="-")



plotter(*Model(full_days, N, **result.best_values), x_ticks=x_ticks)



result.best_values
#7日分の傾きから罹患者数の対数成長速度を算出、



confirm_delay = 10

diff_days =7

sma = 3

adjust = math.floor(sma/2)

# smaスムージング有り

logged_inf_delayed_sma = corona_df['Infected'].shift(-confirm_delay).rolling(sma).mean().shift(-adjust).apply(np.log10).copy()

# smaスムージング無し

# logged_inf_delayed = corona_df['Infected'].shift(-confirm_delay).apply(np.log10).copy()

                    

corona_df["K_t"] = (logged_inf_delayed_sma-logged_inf_delayed_sma.shift(diff_days))/diff_days



#　$L=5.5$、$D=13.5$として$R(t)$を算出

L = 5.5

D = 13.5

corona_df["R_t"] = corona_df["K_t"].apply(lambda x: np.power(x, 2)*L*D + x*(L+D) + 1)



fig = plt.figure(figsize=(40,30),facecolor='white')

plt.rcParams["font.size"] = 18



country_df = corona_df[corona_df["Country"]=="Japan"].copy()



country_df["R_t"] = country_df["R_t"].replace([np.inf, -np.inf], np.nan)

country_df = country_df.dropna(subset=["R_t"])

display(country_df[["Date", "R_t"]].tail(70).style.background_gradient(cmap='Reds'))



plt.subplot(2,2,1)

plt.title("K(t)")

plt.plot(country_df["Date"], country_df["K_t"])

plt.ylim([0,0.2])

plt.grid(True)

plt.subplot(2,2,2)

plt.title("R(t)")

plt.plot(country_df["Date"], country_df["R_t"])

plt.ylim([0,6])

plt.grid(True)

plt.show()
country_df = corona_df[corona_df["Country"]=="Japan"].copy()



fig = plt.figure(figsize=(17,10),facecolor='white')

plt.rcParams["font.size"] = 18

plt.title("R(t)")

plt.ylim([0,6])

plt.plot(country_df["Date"], country_df["R_t"].rolling(3).mean().shift(-1))

plt.grid(True)

plt.show()
#7日分の傾きから罹患者数の対数成長速度を算出、



diff_days =7

sma = 3

adjust = math.floor(sma/2)

# smaスムージング有り

logged_new_conf_sma3 = corona_df['Infected'].rolling(sma).mean().shift(-adjust).apply(np.log10).copy()

                    

corona_df["K_t"] = (logged_new_conf_sma3-logged_new_conf_sma3.shift(diff_days))/diff_days



#　$L=5.5$、$D=13.5$として$R(t)$を算出

L = 5.5

D = 13.5

corona_df["R_t"] = corona_df["K_t"].apply(lambda x: np.power(x, 2)*L*D + x*(L+D) + 1)



fig = plt.figure(figsize=(17,10),facecolor='white')

plt.rcParams["font.size"] = 18



country_df = corona_df[corona_df["Country"]=="Japan"].copy()



country_df["R_t"] = country_df["R_t"].replace([np.inf, -np.inf], np.nan)

country_df = country_df.dropna(subset=["R_t"])

display(country_df[["Date", "R_t"]].style.background_gradient(cmap='Reds'))



plt.title("R(t)")

plt.plot(country_df["Date"], country_df["R_t"])

plt.ylim([0,6])

plt.grid(True)

plt.show()
country_df = corona_df[corona_df["Country"]=="Japan"].copy()



fig = plt.figure(figsize=(17,10),facecolor='white')

plt.rcParams["font.size"] = 18

plt.title("R(t)")

plt.ylim([0,6])

# plt.plot(country_df["Date"], country_df["R_t"])

plt.plot(country_df["Date"], country_df["R_t"].rolling(5).mean().shift(-3))

plt.grid(True)

plt.show()
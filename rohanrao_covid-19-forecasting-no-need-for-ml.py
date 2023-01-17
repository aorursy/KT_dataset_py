## importing packages

import numpy as np

import pandas as pd

from scipy import stats



from datetime import datetime, timedelta
## reading data

df_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

df_test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")

## preparing data

df_panel = df_train.merge(df_test,

                          on = ["Country_Region", "Province_State", "County", "Population", "Weight", "Date", "Target"],

                          how = "outer")



df_panel["geography"] = df_panel.Country_Region + "_" + df_panel.Province_State + "_" + df_panel.County

df_panel.loc[df_panel.County.isna(), "geography"] = df_panel[df_panel.County.isna()].Country_Region + "_" + df_panel[df_panel.County.isna()].Province_State

df_panel.loc[df_panel.Province_State.isna(), "geography"] = df_panel[df_panel.Province_State.isna()].Country_Region



df_panel.sort_values(["geography", "Date", "Target"], inplace = True)

## creating features for validation data

df_val_cc = pd.pivot_table(df_panel[(df_panel.Date >= "2020-04-20") & (df_panel.Target == "ConfirmedCases")], index = "geography", columns = "Date", values = "TargetValue").reset_index()

df_val_ft = pd.pivot_table(df_panel[(df_panel.Date >= "2020-04-20") & (df_panel.Target == "Fatalities")], index = "geography", columns = "Date", values = "TargetValue").reset_index()



df_val_cc["val_avg_cc_1_3"] = df_val_cc[["2020-04-26", "2020-04-25", "2020-04-24"]].mean(axis = 1)

df_val_ft["val_avg_ft_1_3"] = df_val_ft[["2020-04-26", "2020-04-25", "2020-04-24"]].mean(axis = 1)



df_val_cc["val_avg_cc_1_5"] = df_val_cc[["2020-04-26", "2020-04-25", "2020-04-24", "2020-04-23", "2020-04-22"]].mean(axis = 1)

df_val_ft["val_avg_ft_1_5"] = df_val_ft[["2020-04-26", "2020-04-25", "2020-04-24", "2020-04-23", "2020-04-22"]].mean(axis = 1)



df_val_cc["val_avg_cc_1_7"] = df_val_cc[["2020-04-26", "2020-04-25", "2020-04-24", "2020-04-23", "2020-04-22", "2020-04-21", "2020-04-20"]].mean(axis = 1)

df_val_ft["val_avg_ft_1_7"] = df_val_ft[["2020-04-26", "2020-04-25", "2020-04-24", "2020-04-23", "2020-04-22", "2020-04-21", "2020-04-20"]].mean(axis = 1)



df_val_cc["val_std_cc_1_3"] = df_val_cc[["2020-04-26", "2020-04-25", "2020-04-24"]].mean(axis = 1)

df_val_ft["val_std_ft_1_3"] = df_val_ft[["2020-04-26", "2020-04-25", "2020-04-24"]].mean(axis = 1)



df_val_cc["val_std_cc_1_5"] = df_val_cc[["2020-04-26", "2020-04-25", "2020-04-24", "2020-04-23", "2020-04-22"]].mean(axis = 1)

df_val_ft["val_std_ft_1_5"] = df_val_ft[["2020-04-26", "2020-04-25", "2020-04-24", "2020-04-23", "2020-04-22"]].mean(axis = 1)



df_val_cc["val_std_cc_1_7"] = df_val_cc[["2020-04-26", "2020-04-25", "2020-04-24", "2020-04-23", "2020-04-22", "2020-04-21", "2020-04-20"]].mean(axis = 1)

df_val_ft["val_std_ft_1_7"] = df_val_ft[["2020-04-26", "2020-04-25", "2020-04-24", "2020-04-23", "2020-04-22", "2020-04-21", "2020-04-20"]].mean(axis = 1)



df_val_cc["val_change_cc_1_7"] = df_val_cc.val_avg_cc_1_3 - df_val_cc.val_avg_cc_1_7

df_val_ft["val_change_ft_1_7"] = df_val_ft.val_avg_ft_1_3 - df_val_ft.val_avg_ft_1_7



df_val_cc.loc[df_val_cc["2020-04-20"] == 0, "val_change_cc_1_7"] = 0.0

df_val_ft.loc[df_val_ft["2020-04-20"] == 0, "val_change_ft_1_7"] = 0.0



df_panel = df_panel.merge(df_val_cc[["geography", "val_avg_cc_1_3", "val_avg_cc_1_5", "val_avg_cc_1_7",

                                     "val_std_cc_1_3", "val_std_cc_1_5", "val_std_cc_1_7", "val_change_cc_1_7"]], on = "geography")



df_panel = df_panel.merge(df_val_ft[["geography", "val_avg_ft_1_3", "val_avg_ft_1_5", "val_avg_ft_1_7",

                                     "val_std_ft_1_3", "val_std_ft_1_5", "val_std_ft_1_7", "val_change_ft_1_7"]], on = "geography")



df_panel["val_avg_1_3"] = df_panel.val_avg_cc_1_3

df_panel.loc[df_panel.Target == "Fatalities", "val_avg_1_3"] = df_panel[df_panel.Target == "Fatalities"].val_avg_ft_1_3



df_panel["val_std_1_7"] = df_panel.val_std_cc_1_7

df_panel.loc[df_panel.Target == "Fatalities", "val_std_1_7"] = df_panel[df_panel.Target == "Fatalities"].val_avg_ft_1_7

df_panel.loc[df_panel.val_std_1_7 <= 0, "val_std_1_7"] = 0.01



df_panel["val_change_1_7"] = df_panel.val_change_cc_1_7

df_panel.loc[df_panel.Target == "Fatalities", "val_change_1_7"] = df_panel[df_panel.Target == "Fatalities"].val_change_ft_1_7

## creating features for test data

df_test_cc = pd.pivot_table(df_panel[(df_panel.Date >= "2020-05-04") & (df_panel.Target == "ConfirmedCases")], index = "geography", columns = "Date", values = "TargetValue").reset_index()

df_test_ft = pd.pivot_table(df_panel[(df_panel.Date >= "2020-05-04") & (df_panel.Target == "Fatalities")], index = "geography", columns = "Date", values = "TargetValue").reset_index()



df_test_cc["test_avg_cc_1_3"] = df_test_cc[["2020-05-10", "2020-05-09", "2020-05-08"]].mean(axis = 1)

df_test_ft["test_avg_ft_1_3"] = df_test_ft[["2020-05-10", "2020-05-09", "2020-05-08"]].mean(axis = 1)



df_test_cc["test_avg_cc_1_5"] = df_test_cc[["2020-05-10", "2020-05-09", "2020-05-08", "2020-05-07", "2020-05-06"]].mean(axis = 1)

df_test_ft["test_avg_ft_1_5"] = df_test_ft[["2020-05-10", "2020-05-09", "2020-05-08", "2020-05-07", "2020-05-06"]].mean(axis = 1)



df_test_cc["test_avg_cc_1_7"] = df_test_cc[["2020-05-10", "2020-05-09", "2020-05-08", "2020-05-07", "2020-05-06", "2020-05-05", "2020-05-04"]].mean(axis = 1)

df_test_ft["test_avg_ft_1_7"] = df_test_ft[["2020-05-10", "2020-05-09", "2020-05-08", "2020-05-07", "2020-05-06", "2020-05-05", "2020-05-04"]].mean(axis = 1)



df_test_cc["test_std_cc_1_3"] = df_test_cc[["2020-05-10", "2020-05-09", "2020-05-08"]].std(axis = 1)

df_test_ft["test_std_ft_1_3"] = df_test_ft[["2020-05-10", "2020-05-09", "2020-05-08"]].std(axis = 1)



df_test_cc["test_std_cc_1_5"] = df_test_cc[["2020-05-10", "2020-05-09", "2020-05-08", "2020-05-07", "2020-05-06"]].std(axis = 1)

df_test_ft["test_std_ft_1_5"] = df_test_ft[["2020-05-10", "2020-05-09", "2020-05-08", "2020-05-07", "2020-05-06"]].std(axis = 1)



df_test_cc["test_std_cc_1_7"] = df_test_cc[["2020-05-10", "2020-05-09", "2020-05-08", "2020-05-07", "2020-05-06", "2020-05-05", "2020-05-04"]].std(axis = 1)

df_test_ft["test_std_ft_1_7"] = df_test_ft[["2020-05-10", "2020-05-09", "2020-05-08", "2020-05-07", "2020-05-06", "2020-05-05", "2020-05-04"]].std(axis = 1)



df_test_cc["test_change_cc_1_7"] = df_test_cc.test_avg_cc_1_3 - df_test_cc.test_avg_cc_1_7

df_test_ft["test_change_ft_1_7"] = df_test_ft.test_avg_ft_1_3 - df_test_ft.test_avg_ft_1_7



df_test_cc.loc[df_test_cc["2020-05-04"] == 0, "test_change_cc_1_7"] = 0.0

df_test_ft.loc[df_test_ft["2020-05-04"] == 0, "test_change_ft_1_7"] = 0.0



df_panel = df_panel.merge(df_test_cc[["geography", "test_avg_cc_1_3", "test_avg_cc_1_5", "test_avg_cc_1_7",

                                     "test_std_cc_1_3", "test_std_cc_1_5", "test_std_cc_1_7", "test_change_cc_1_7"]], on = "geography")



df_panel = df_panel.merge(df_test_ft[["geography", "test_avg_ft_1_3", "test_avg_ft_1_5", "test_avg_ft_1_7",

                                     "test_std_ft_1_3", "test_std_ft_1_5", "test_std_ft_1_7", "test_change_ft_1_7"]], on = "geography")



df_panel["test_avg_1_3"] = df_panel.test_avg_cc_1_3

df_panel.loc[df_panel.Target == "Fatalities", "test_avg_1_3"] = df_panel[df_panel.Target == "Fatalities"].test_avg_ft_1_3



df_panel["test_std_1_7"] = df_panel.test_std_cc_1_7

df_panel.loc[df_panel.Target == "Fatalities", "test_std_1_7"] = df_panel[df_panel.Target == "Fatalities"].test_avg_ft_1_7

df_panel.loc[df_panel.test_std_1_7 <= 0, "test_std_1_7"] = 0.01



df_panel["test_change_1_7"] = df_panel.test_change_cc_1_7

df_panel.loc[df_panel.Target == "Fatalities", "test_change_1_7"] = df_panel[df_panel.Target == "Fatalities"].test_change_ft_1_7

## generating predictions using statistical heuristics

df_panel["val_day_progression"] = 1 - ((datetime(2020, 5, 10) - pd.to_datetime(df_panel.Date)).dt.days / 13)

df_panel["test_day_progression"] = 1 - ((datetime(2020, 6, 10) - pd.to_datetime(df_panel.Date)).dt.days / 29)



df_panel["val_50"] = (1 - (df_panel.val_day_progression / 2)) * df_panel.val_avg_1_3

df_panel["test_50"] = (1 - (df_panel.test_day_progression / 2)) * df_panel.test_avg_1_3



df_panel.loc[df_panel.val_change_1_7 > 0, "val_50"] = df_panel.val_avg_1_3[df_panel.val_change_1_7 > 0] + 2 * df_panel.val_change_1_7[df_panel.val_change_1_7 > 0] * (1 - abs(df_panel.val_day_progression[df_panel.val_change_1_7 > 0] - 0.5) / 0.5)

df_panel.loc[df_panel.test_change_1_7 > 0, "test_50"] = df_panel.test_avg_1_3[df_panel.test_change_1_7 > 0] + 2 * df_panel.test_change_1_7[df_panel.test_change_1_7 > 0] * (1 - abs(df_panel.test_day_progression[df_panel.test_change_1_7 > 0] - 0.5) / 0.5)



df_panel["val_05"] = np.floor(df_panel.val_50)

df_panel["test_05"] = np.floor(df_panel.test_50)



df_panel.loc[(df_panel.Date >= "2020-04-27") & (df_panel.Date <= "2020-05-10"), "val_05"] = df_panel[(df_panel.Date >= "2020-04-27") & (df_panel.Date <= "2020-05-10")][["val_50", "val_std_1_7"]].apply(lambda x: stats.norm.ppf(0.05, x.val_50, x.val_std_1_7), axis = 1)

df_panel.loc[df_panel.Date > "2020-05-10", "test_05"] = df_panel[df_panel.Date > "2020-05-10"][["test_50", "test_std_1_7"]].apply(lambda x: stats.norm.ppf(0.05, x.test_50, x.test_std_1_7), axis = 1)



df_panel["val_05"] = np.maximum(np.floor(1 * df_panel.val_05 + 0 * df_panel.val_50), 0)

df_panel["test_05"] = np.maximum(np.floor(1 * df_panel.test_05 + 0 * df_panel.test_50), 0)



df_panel["val_95"] = np.ceil(df_panel.val_50)

df_panel["test_95"] = np.ceil(df_panel.test_50)



df_panel.loc[(df_panel.Date >= "2020-04-27") & (df_panel.Date <= "2020-05-10"), "val_95"] = df_panel[(df_panel.Date >= "2020-04-27") & (df_panel.Date <= "2020-05-10")][["val_50", "val_std_1_7"]].apply(lambda x: stats.norm.ppf(0.95, x.val_50, x.val_std_1_7), axis = 1)

df_panel.loc[df_panel.Date > "2020-05-10", "test_95"] = df_panel[df_panel.Date > "2020-05-10"][["test_50", "test_std_1_7"]].apply(lambda x: stats.norm.ppf(0.95, x.test_50, x.test_std_1_7), axis = 1)



df_panel["val_95"] = np.ceil(0.75 * df_panel.val_95 + 0.25 * df_panel.val_50)

df_panel["test_95"] = np.ceil(0.5 * df_panel.test_95 + 0.25 * df_panel.test_50)



df_panel.loc[df_panel.val_05 + df_panel.val_50 == 0, "val_95"] = 0

df_panel.loc[df_panel.test_05 + df_panel.test_50 == 0, "test_95"] = 0



df_panel.loc[(df_panel.Date < "2020-04-27") | (df_panel.Date > "2020-05-10"), "val_50"] = np.NaN

df_panel.loc[(df_panel.Date < "2020-04-27") | (df_panel.Date > "2020-05-10"), "val_05"] = np.NaN

df_panel.loc[(df_panel.Date < "2020-04-27") | (df_panel.Date > "2020-05-10"), "val_95"] = np.NaN



df_panel.loc[df_panel.Date <= "2020-05-10", "test_50"] = np.NaN

df_panel.loc[df_panel.Date <= "2020-05-10", "test_05"] = np.NaN

df_panel.loc[df_panel.Date <= "2020-05-10", "test_95"] = np.NaN

## validation score

df_val = df_panel[(df_panel.Date >= "2020-04-27") & (df_panel.Date <= "2020-05-10")]



val_loss_05 = np.maximum(0.05 * (df_val.TargetValue.values - df_val.val_05.values),

                         -0.95 * (df_val.TargetValue.values - df_val.val_05.values))

val_loss_50 = np.maximum(0.50 * (df_val.TargetValue.values - df_val.val_50.values),

                         -0.50 * (df_val.TargetValue.values - df_val.val_50.values))

val_loss_95 = np.maximum(0.95 * (df_val.TargetValue.values - df_val.val_95.values),

                         -0.05 * (df_val.TargetValue.values - df_val.val_95.values))



pinball_loss_lgb_05 = np.mean(df_val.Weight.values * val_loss_05)

pinball_loss_lgb_50 = np.mean(df_val.Weight.values * val_loss_50)

pinball_loss_lgb_95 = np.mean(df_val.Weight.values * val_loss_95)

pinball_loss_lgb = np.mean(df_val.Weight.values * (val_loss_05 + val_loss_50 + val_loss_95) / 3)



print("Pinball Loss Val at 0.05 quantile:", round(pinball_loss_lgb_05, 2))

print("Pinball Loss Val at 0.5 quantile:", round(pinball_loss_lgb_50, 2))

print("Pinball Loss Val at 0.95 quantile:", round(pinball_loss_lgb_95, 2))

print("Pinball Loss Val:", round(pinball_loss_lgb, 2))

## visualizing Fatalities

from bokeh.models import Panel, Tabs

from bokeh.io import output_notebook, show

from bokeh.plotting import figure



output_notebook()



tab_list = []

geographies = df_panel.groupby("geography")["Weight"].max().reset_index().sort_values("Weight", ascending = False).geography.values



for geography in geographies[3333:]:

    df_geography = df_panel[(df_panel.Target == "Fatalities") & (df_panel.geography == geography)]

    df_geography.Date = pd.to_datetime(df_geography.Date)

    v = figure(plot_width = 800, plot_height = 400, x_axis_type = "datetime", title = "Covid-19 ConfirmedCases over time")

    v.line(df_geography.Date, df_geography.TargetValue, color = "green", legend_label = "CC Train")

    v.line(df_geography.Date, df_geography.val_05, color = "blue", legend_label = "CC Val 0.05")

    v.line(df_geography.Date, df_geography.val_50, color = "purple", legend_label = "CC Val 0.50")

    v.line(df_geography.Date, df_geography.val_95, color = "blue", legend_label = "CC Val 0.95")

    v.line(df_geography.Date, df_geography.test_05, color = "red", legend_label = "CC Test 0.05")

    v.line(df_geography.Date, df_geography.test_50, color = "orange", legend_label = "CC Test 0.50")

    v.line(df_geography.Date, df_geography.test_95, color = "red", legend_label = "CC Test 0.95")

    v.legend.location = "top_left"

    tab = Panel(child = v, title = geography)

    tab_list.append(tab)



tabs = Tabs(tabs=tab_list)

show(tabs)

## visualizing ConfirmedCases

from bokeh.models import Panel, Tabs

from bokeh.io import output_notebook, show

from bokeh.plotting import figure



output_notebook()



tab_list = []

geographies = df_panel.groupby("geography")["Weight"].max().reset_index().sort_values("Weight", ascending = False).geography.values



for geography in geographies[3131:3232]:

    df_geography = df_panel[(df_panel.Target == "ConfirmedCases") & (df_panel.geography == geography)]

    df_geography.Date = pd.to_datetime(df_geography.Date)

    v = figure(plot_width = 800, plot_height = 400, x_axis_type = "datetime", title = "Covid-19 ConfirmedCases over time")

    v.line(df_geography.Date, df_geography.TargetValue, color = "green", legend_label = "CC Train")

    v.line(df_geography.Date, df_geography.val_05, color = "blue", legend_label = "CC Val 0.05")

    v.line(df_geography.Date, df_geography.val_50, color = "purple", legend_label = "CC Val 0.50")

    v.line(df_geography.Date, df_geography.val_95, color = "blue", legend_label = "CC Val 0.95")

    v.line(df_geography.Date, df_geography.test_05, color = "red", legend_label = "CC Test 0.05")

    v.line(df_geography.Date, df_geography.test_50, color = "orange", legend_label = "CC Test 0.50")

    v.line(df_geography.Date, df_geography.test_95, color = "red", legend_label = "CC Test 0.95")

    v.legend.location = "top_left"

    tab = Panel(child = v, title = geography)

    tab_list.append(tab)



tabs = Tabs(tabs=tab_list)

show(tabs)

## submission

df_panel["test_0.05"] = df_panel.test_05

df_panel["test_0.5"] = df_panel.test_50

df_panel["test_0.95"] = df_panel.test_95



df_panel.loc[(df_panel.Date >= "2020-04-27") & (df_panel.Date <= "2020-05-10"), "test_0.05"] = df_panel.val_05 * 41

df_panel.loc[(df_panel.Date >= "2020-04-27") & (df_panel.Date <= "2020-05-10"), "test_0.5"] = df_panel.val_50 * 41

df_panel.loc[(df_panel.Date >= "2020-04-27") & (df_panel.Date <= "2020-05-10"), "test_0.95"] = df_panel.val_95 * 41



submission = pd.melt(df_panel[~df_panel.ForecastId.isna()],

                     id_vars = "ForecastId",

                     value_vars = ["test_0.05", "test_0.5", "test_0.95"],

                     var_name = "Quantile",

                     value_name = "TargetValue")



submission["ForecastId_Quantile"] = submission.ForecastId.astype(int).astype(str) + "_" + submission.Quantile.str.replace("test_", "")



submission[["ForecastId_Quantile", "TargetValue"]].to_csv("submission.csv", index = False)

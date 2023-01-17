import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
%load_ext autoreload
%autoreload 2
from IPython.display import display, Markdown, Latex

class Util:

    @classmethod
    def print(cls, st, bold=False):
        if bold:
            st = "**{}**".format(st)
        return display(Markdown(st))

    @classmethod
    def print_data(cls, dic, lst, fmt=None, bold=False):
        st = cls.data_line(dic, lst, fmt)
        if bold:
            st = "**{}**".format(st)
        return display(Markdown(st))

    @classmethod
    def data_line(cls, dic, lst, fmt=None):
        if fmt is None:
            fmt = ""
        fmtstr = "{{0:}}: {{1:{}}}".format(fmt)
        return ", ".join([fmtstr.format(e, dic[e]) for e in lst])
import matplotlib.ticker as ticker

class InfoPlotter(object):

    @classmethod
    def plot_rate_components(cls, se):

        # data
        numer = ["AP_death", "AP_endow", "AP_alpha", "AP_gamma", "AP_beta"]
        denom = "AP"

        ratios = np.array([se[v] / se[denom] for v in numer])
        left = np.roll(np.cumsum(ratios), 1)
        left[0] = 0.0

        # plotting
        fig, ax = plt.subplots(1, 1, figsize=(8, 1))
        palette = sns.color_palette("Set1", len(numer))

        for i, v in enumerate(numer):
            ax.barh([0], [ratios[i]], left=left[i], color=palette[i], label=numer[i])

        ax.set_xlim(0.0, 1.0)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.legend(bbox_to_anchor=(1.01, 1))
        ax.set_title("Rate Components (of AP)")

        plt.show()

    @classmethod
    def plot_pv_components(cls, se):

        # data
        numer = ["PV_commission", "PV_death_benefit", "PV_medical_benefit", "PV_endow_benefit", "PV_annuity_benefit",
                 "PV_surrender_benefit", "PV_unpaid_cashout", "PV_init_expense", "PV_maint_expense"]
        denom = "PV_premium_income"
        ratios = np.array([se[v] / se[denom] for v in numer])
        left = np.roll(np.cumsum(ratios), 1)
        left[0] = 0.0
        mx = np.cumsum(ratios)[-1]

        # plotting
        fig, ax = plt.subplots(1, 1, figsize=(8, 2))
        palette = sns.color_palette("Set2", len(numer)+1)

        ax.barh([1, 0], [1.0, 0.0], color=palette[0], label=denom)
        for i, v in enumerate(numer):
            ax.barh([1, 0], [0, ratios[i]], left=[left[i]], color=palette[i + 1], label=numer[i])

        ax.set_xlim(0.0, max(mx, 1.0))
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.legend(bbox_to_anchor=(1.01, 1))
        ax.set_title("Benefit Components (of PV Premium)")

        plt.show()


    @classmethod
    def plot_cf(cls, df, length):

        # data
        module = "OutputBSPL"
        lst = ["premium_income", "commission", "death_benefit", "medical_benefit", "endow_benefit", "annuity_benefit",
               "surrender_benefit", "unpaid_cashout", "init_expense", "maint_expense"]
        value_signs = [1, -1, -1, -1, -1, -1,
                       -1, -1, -1, -1, -1]

        values = np.array([df.loc[(module, v)].values[:length] * value_signs[i] for i, v in enumerate(lst)])
        values_abs_pos = [np.maximum(values[i, :], 0.0) for i in range(len(lst))]
        values_abs_pos_bottom = np.roll(np.cumsum(values_abs_pos, axis=0), 1, axis=0)
        values_abs_pos_bottom[0, :] = 0.0

        values_abs_neg = [np.minimum(values[i, :], 0.0) for i in range(len(lst))]
        values_abs_neg_bottom = np.roll(np.cumsum(values_abs_neg, axis=0), 1, axis=0)
        values_abs_neg_bottom[0, :] = 0.0

        values_abs_mx = np.max([np.max(np.abs(np.cumsum(values_abs_pos, axis=0))),
                                np.max(np.abs(np.cumsum(values_abs_neg, axis=0)))])

        # plotting
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        palette = sns.color_palette("Set2", len(lst))
        ind = np.arange(length)  # the x locations for the groups
        width = 0.8  # the width of the bars: can also be len(x) sequence

        pos_bars = []
        for i in range(len(lst)):
            bar = ax.bar(ind, values_abs_pos[i], width, bottom=values_abs_pos_bottom[i, :],
                         color=palette[i], label=lst[i])
            pos_bars.append(bar)

        for i in range(len(lst)):
            ax.bar(ind, values_abs_neg[i], width, bottom=values_abs_neg_bottom[i - 1, :], color=palette[i])

        @ticker.FuncFormatter
        def major_formatter(x, pos):
            return '{:,.0f}'.format(x)

        ax.yaxis.set_major_formatter(major_formatter)
        ax.set_ylim((-values_abs_mx * 1.1, values_abs_mx * 1.1))
        ax.legend(handles=pos_bars, bbox_to_anchor=(1.01, 1))
        ax.set_title("CashFlow")

        plt.show()

    @classmethod
    def plot_reserve(cls, df, length):

        # data
        module = "OutputBSPL"
        lst = ["reserve"]

        values = np.array([df.loc[(module, v)].values[:length] for i, v in enumerate(lst)])
        values = [values[i, :] for i in range(len(lst))]
        values_mx = np.max(values)

        # plotting
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        palette = sns.color_palette("Set1", len(lst))

        ind = np.arange(length)  # the x locations for the groups
        width = 0.8  # the width of the bars: can also be len(x) sequence

        for i in range(len(lst)):
            ax.bar(ind, values[i], width, color=palette[i], label=lst[i])

        @ticker.FuncFormatter
        def major_formatter(x, pos):
            return '{:,.0f}'.format(x)

        ax.yaxis.set_major_formatter(major_formatter)
        ax.set_ylim((0.0, values_mx*1.1))
        ax.legend()
        ax.set_title("Reserve")

        plt.show()

    @classmethod
    def plot_survivor_and_rates(cls, df, length):

        # data
        module = "AnalysisArray"
        lst1_1 = ["lx_end"]
        lst1_2 = ["d_x", "dwx"]
        lst2 = ["qx", "qx_crude", "qwx", "qwx_crude"]

        vals1_1 = np.array([df.loc[(module, v)].values[:length] for i, v in enumerate(lst1_1)])
        vals1_2 = np.array([df.loc[(module, v)].values[:length] for i, v in enumerate(lst1_2)])
        vals2 = np.array([df.loc[(module, v)].values[:length] for i, v in enumerate(lst2)])
        vals1_1[:, 0] = np.nan
        vals1_2[:, 0] = np.nan
        vals2[:, 0] = np.nan

        # plotting
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ind = np.arange(length)
        palette = sns.color_palette("Set2", 8)
        width = 0.8

        # plot1
        for i in range(len(lst1_1)):
            axes[0].plot(vals1_1[i, :], label=lst1_1[i])
        for i in range(len(lst1_2)):
            axes[0].bar(ind, vals1_2[i, :], width, color=palette[i], label=lst1_2[i])

        axes[0].legend()
        axes[0].set_ylim((0.0, 1.0))
        axes[0].set_title("Survivorship")

        # plot2
        for i in range(len(lst2)):
            axes[1].plot(vals2[i, :], label=lst2[i])

        axes[1].legend()
        axes[1].set_ylim((0.0, 0.1))
        axes[1].set_title("Mortality/Lapse Rates")

        plt.show()

    @classmethod
    def plot_risks(cls, df, length):

        # data
        module = "OutputRisk"
        lst = ['risk', 'risk_mortup', 'risk_mortdown', 'risk_lapse', 'risk_morbup', 'risk_expup',]
        vals = np.array([df.loc[(module, v)].values[:length] for i, v in enumerate(lst)])

        # plotting
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        palette = sns.color_palette("Set2", len(lst))

        for i in range(len(lst)):
            ax.plot(vals[i, :], color=palette[i], label=lst[i])

        @ticker.FuncFormatter
        def major_formatter(x, pos):
            return '{:,.0f}'.format(x)

        ax.yaxis.set_major_formatter(major_formatter)
        ax.legend()
        ax.set_title("Risks")

        plt.show()

    @classmethod
    def plot_pvliab(cls, df, length):

        # data
        module = "OutputRisk"
        lst = ["PVLiab_adj_base"]

        values = np.array([df.loc[(module, v)].values[:length] for i, v in enumerate(lst)])
        values_mn, values_mx = np.min(values), np.max(values)

        # plotting
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        palette = sns.color_palette("Set1", len(lst))
        ind = np.arange(length)
        width = 0.8

        for i in range(len(lst)):
            ax.bar(ind, values[i], width, color=palette[i], label=lst[i])

        @ticker.FuncFormatter
        def major_formatter(x, pos):
            return '{:,.0f}'.format(x)

        ax.yaxis.set_major_formatter(major_formatter)
        ax.set_ylim((values_mn * 1.1, values_mx*1.1))
        ax.legend()
        ax.set_title("PV Liability")

        plt.show()

    @classmethod
    def plot_sensitivity(cls, se):

        # data
        lst = ["NBM_base", "NBM_mortup", "NBM_lapseup", "NBM_morbup", "NBM_expup"]
        labels = ["base", "mortup", "lapseup", "morbup", "expup"]
        values = np.array([se[v] for i, v in enumerate(lst)])

        # plotting
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ind = range(len(lst))
        width = 0.8

        ax.bar(ind, values, width)

        @ticker.FuncFormatter
        def major_formatter(x, pos):
            return '{:,.2%}'.format(x)

        ax.set_xticks(ticks=ind)
        ax.set_xticklabels(labels=labels)
        ax.yaxis.set_major_formatter(major_formatter)
        ax.set_title("NBM - Sensitivity")

        plt.show()
df_info_all = pd.read_csv("../input/mlsinput/insurance_info.csv")
df_sensitivity_all = pd.read_csv("../input/mlsinput/insurance_sensitivity.csv")
df_cashflow_all = pd.read_csv("../input/mlsinput/insurance_cashflow.csv")
inf_query = "x == 50 & n_plan == 80 & m_plan == 10 & sex == 'F'"

inf_id = int(df_info_all.query(inf_query)["inforce"])
series_info = df_info_all[df_info_all["inforce"] == inf_id].iloc[0,:]
series_sensitivity = df_sensitivity_all[df_sensitivity_all["inforce"] == inf_id].iloc[0,:]
df_cashflow = df_cashflow_all[df_cashflow_all["inforce"] == inf_id]
df_cashflow = df_cashflow.drop(["inforce"], axis=1).set_index(["module", "variable"])
length = series_info["n"] + 1
Util.print("#### [契約情報]")
Util.print_data(series_info, ["inforce", "plan"], bold=True)   
Util.print_data(series_info, ["x", "n_plan", "m_plan", "n", "m", "S",], bold=True)
Util.print_data(series_info, ["AP", "AP_death", "AP_endow", "AP_alpha", "AP_gamma", "AP_beta"], ",.0f", bold=True)

Util.print("#### [収益性]")
Util.print_data(series_info, ["MCEV", "PVFP", "TVOG", "CNHR"], ",.0f", bold=True)
Util.print_data(series_info, ["NBM", "PVFP%", "TVOG%", "CNHR%"], ",.2%", bold=True)
Util.print_data(series_info, ["profit_margin_before_tax", "profit_margin_after_tax"], ",.2%", bold=True)
InfoPlotter.plot_rate_components(series_info)
InfoPlotter.plot_pv_components(series_info)
InfoPlotter.plot_cf(df_cashflow, length)
InfoPlotter.plot_reserve(df_cashflow, length)
InfoPlotter.plot_survivor_and_rates(df_cashflow, length)
InfoPlotter.plot_risks(df_cashflow, length)
InfoPlotter.plot_pvliab(df_cashflow, length)
InfoPlotter.plot_sensitivity(series_sensitivity)
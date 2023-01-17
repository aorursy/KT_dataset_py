!pip install drdigit-brezniczky==0.0.12
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import drdigit as drd

import austria_2019_ep_preprocessing as pp



# the below line allows publishing the kernel, for experimenting

# just comment it out - saves time by caching some calculations

# across Python sessions

drd.set_option(physical_cache_path="")  



df, info = pp.get_preprocessed_data()
df.head()
info
len(df)
df["group"] = df[info.area_code].str.slice(0, 4)



np.mean(df.groupby("group").aggregate({"GKZ": len}))
tests = {}



for party in info.parties[0:3]:

    for filter_by in ["by row", "by municipality"]:

        filtered = drd.filter_df(df, group_column="group",

                                 value_columns=[party],

                                 entire_groups=filter_by == "by municipality",

                                 min_value=100)

        print("%s by %s: %d" % (party, filter_by, len(filtered)))

        tests[(party, filter_by)] = drd.LodigeTest(

            digits=filtered["ld_" + party],

            group_ids=filtered["group"],

            bottom_n=20,

            ll_iterations=5000,

            quiet=True  # this looks a lot better in these online notebooks

        )
# calculate and print the test p-values

# this takes a while, the calculations are carried out in a lazy fashion

# (sort of, when someone asks for the result)



for spec, test in tests.items():

    print(spec, test.p)
for party, filter_by in sorted(tests.keys()):

    test = tests[(party, filter_by)]

    drd.plot_entropy_distribution(

        actual_total_ent=test.likelihood,

        p=test.p,

        entropies=test.cdf.sample,

        title="%s filtered on the %s level" % (party, filter_by),

    )
df_susp = drd.filter_df(df, value_columns=["ÖVP"], min_value=100)



scores = drd.get_group_scores(df_susp["group"], df_susp["ld_ÖVP"], 

                              [df_susp[info.valid_votes], df_susp["ld_ÖVP"]], 

                              [df_susp["ld_FPÖ"], df_susp["ld_SPÖ"]],

                             quiet=True)
help(drd.filter_df)
ranking = pd.DataFrame(dict(group=scores.index, score=scores))

ranking.sort_values(["score"], inplace=True)

ranking.head()
len(ranking)
plt.hist(ranking.score, bins=20)

plt.show()
def get_top_group(index):

    group = ranking.iloc[index]["group"]

    return df[df.group==group][["group", "GKZ", info.valid_votes, "ÖVP", "SPÖ", "FPÖ"]]

    

get_top_group(0)
get_top_group(1)
get_top_group(2)
get_top_group(3)
old_figsize = plt.rcParams["figure.figsize"]

try:

    plt.rcParams["figure.figsize"] = [9, 7]

    drd.plot_explanatory_fingerprint_responses(fontsize=16)

finally:

    plt.rcParams["figure.figsize"] = old_figsize
len(df_susp)
def plot_party_fingerprint(party, start_perc, end_perc):

    l = len(ranking)

    groups = ranking.group[int(l * start_perc):int(l * end_perc)]

    df_act = df_susp[df_susp["group"].isin(groups)]

    drd.plot_fingerprint(df_act[party], df_act[info.valid_votes], df_act[info.registered_voters], 

                         "%s from %d %% to %d %%" % (party, start_perc * 100, end_perc * 100))





plot_party_fingerprint("ÖVP", 0, 0.5)

plot_party_fingerprint("ÖVP", 0.5, 1.0)
plot_party_fingerprint("SPÖ", 0, 0.50)

plot_party_fingerprint("SPÖ", 0.50, 1.00)
plot_party_fingerprint("FPÖ", 0, 0.50)

plot_party_fingerprint("FPÖ", 0.50, 1.00)
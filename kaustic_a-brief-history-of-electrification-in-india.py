# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 2001 and 2011 Census data

electrification_census = pd.read_csv("../input/statewise-household-electrification-2001-and-2011/statewise_electrified_households_percentage_census_2001_and_2011.csv")

households_2001 = pd.read_csv("../input/statewise-household-electrification-2001-and-2011/statewise_total_households_2001.csv")

households_2011 = pd.read_csv("../input/statewise-household-electrification-2001-and-2011/statewise_total_households_2011.csv")



# 2005-06 and 2015-16 Data (NFHS)

nfhs_raw = nfhs_raw = pd.read_csv("https://raw.githubusercontent.com/HindustanTimesLabs/nfhs-data/master/nfhs_state-wise.csv")



# 2017-19 Data (Saubhagya)

sbhg_data = pd.read_csv("../input/saubhagya-all-india-electrification/ALL-INDIA.csv")
nfhs_raw.sample(5)
nfhs_raw.head(8)
# Note that "Households with electricity" has indicator_id 6



nfhs_data = pd.DataFrame({

    "State": pd.Series([], dtype=object),

    "2006": pd.Series([], dtype=np.float64),  # Percentage electrification in 2005-06

    "2016": pd.Series([], dtype=np.float64),  # Percentage electrification in 2015-16

})



for _, row in nfhs_raw[nfhs_raw["indicator_id"] == "6"].iterrows():

    nfhs_data = nfhs_data.append(

        {

            "State": row["state"],

            "2006": np.float64(row["total_2005-06"]),

            "2016": np.float64(row["total"]),

        },

        ignore_index=True,

    )
nfhs_data.sample(5)
# Make e.g. "Jammu & Kashmir" to "Jammu and Kashmir" so that it's not confused later



for i in sbhg_data.index:

    sbhg_data.iloc[i, 0] = sbhg_data.iloc[i, 0].replace("&", "and")



for i in households_2001.index:

    households_2001.iloc[i, 0] = households_2001.iloc[i, 0].replace("&", "and")
sbhg_data.info()
households_2017 = pd.DataFrame({

    "State": pd.Series([], dtype=object),

    "Total Households": pd.Series([], dtype=np.int64),

})



for _, row in sbhg_data.iterrows():

    state, total_hh = row["State"], row["Total Households"]

    if state == "Total":

        pass

    else:

        households_2017 = households_2017.append({

            "State": state,

            "Total Households": total_hh,

        },

        ignore_index=True)

households_2017.sample(5)
electrification_census.info()
print(households_2001.info())

print(households_2011.info())

print(households_2017.info())
# Make Total Households of type "int64"



households_2017["Total Households"] = households_2017[

    "Total Households"

].apply(lambda x: np.int64(x))



print(households_2017.info())

households_2017.sample(5)
print(nfhs_data.info())

print(sbhg_data.info())
sbhg_data_refined = pd.DataFrame(

    {

        "State": pd.Series([], dtype=object),

        "2017": pd.Series([], dtype=np.float64),

        "2019": pd.Series([], dtype=np.float64),

    }

)



for _, row in sbhg_data.iterrows():

    state = row["State"]

    

    if state == "Total":

        continue

    

    p = np.float64(row.iloc[2].replace(",", ""))

    q = np.float64(row.iloc[1].replace(",", ""))



    sbhg_data_refined = sbhg_data_refined.append(

        {

            "State": state,

            "2017": p / q * 100,  # Percent Households electrified

            "2019": row.iloc[8],

        },

        ignore_index=True,

    )

sbhg_data_refined
import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use("seaborn-darkgrid")



def plt_init(figsize=(12, 8)):

    plt.figure(figsize=figsize)
plt_init((18, 10))



sorted_census_data = electrification_census.sort_values("2001", ascending=False)



g = sns.barplot(x="State", y="2001", data=sorted_census_data)



for xtick in g.get_xticklabels():

    xtick.set_rotation(75)

    xtick.set_fontsize(11)



for p in g.patches:

    h = p.get_height()

    g.annotate(

        f"{h}",

        xy=(p.get_x() + p.get_width() / 2, h),

        xytext=(0, 3),

        textcoords="offset points",

        ha="center", va="bottom"

    )



plt.ylabel("State-wise % of electrified households in 2001")

    

plt.legend()

plt.show()
plt_init((18, 10))



sorted_census_2011 = electrification_census.sort_values("2011", ascending=False)



g = sns.barplot(x="State", y="2011", data=sorted_census_2011)



for xtick in g.get_xticklabels():

    xtick.set_rotation(75)

    xtick.set_fontsize(11)



for p in g.patches:

    h = p.get_height()

    g.annotate(

        f"{h}",

        xy=(p.get_x() + p.get_width() / 2, h),

        xytext=(0, 3),

        textcoords="offset points",

        ha="center", va="bottom"

    )



plt.ylabel("State-wise % of electrified households in 2011")

    

plt.legend()

plt.show()
plt_init((10, 16))



sorted_census_data = electrification_census.sort_values("2011", ascending=False)



sns.set_color_codes("pastel")

g = sns.barplot(x="2011", y="State", color="b", 

                data=sorted_census_data, label="Electrification % as per 2011")



sns.set_color_codes("muted")

sns.barplot(x="2001", y="State", color="b",

                data=sorted_census_data, label="Electrification % as per 2001")



for p1, p2 in zip(g.patches[:35], g.patches[35:]):

    w1, w2 = p1.get_width(), p2.get_width()

    g.annotate(

        f"{(w1 - w2):.2f}",

        xy=((w1 + w2) / 2 - 2, p1.get_y() + p1.get_height() / 2),

        ha="left", va="center",

    )



plt.ylabel("States and Union Territories")

plt.xlabel("% of Electrified Households in 2001 and 2011 with difference")



plt.legend()



plt.show()
bihar_percent_electrification_2011 = electrification_census[electrification_census["State"] == "Bihar"]["2011"].values[0]

bihar_percent_electrification_2001 = electrification_census[electrification_census["State"] == "Bihar"]["2001"].values[0]



bihar_hh_2001 = households_2001[households_2001["State"] == "Bihar"]["Total Households"].values[0]

bihar_hh_2011 = households_2011[households_2011["State"] == "Bihar"]["Total Households"].values[0]



without_electricity_2001 = np.float64(bihar_hh_2001) * (1 - bihar_percent_electrification_2001 / 100)

still_without_electricity_2011 = np.float64(bihar_hh_2011) * (1 - bihar_percent_electrification_2011 / 100)



print(f"Households without electricity in 2001: {without_electricity_2001:.0f}")

print(f"Households without electricity as of 2011: {still_without_electricity_2011:.0f}")
village_electrification = pd.read_csv("../input/percentage-villages-electrified-source-livemint/data-8crSo.csv")

village_electrification.info()
plt_init((12, 6))



g = sns.barplot(x="X.1", y="Number of villages electrified", data=village_electrification)



for p in g.patches:

    h = p.get_height()

    g.annotate(

        str(h), xy=(p.get_x() + p.get_width() / 2, h),

        xytext=(0, 3),

        textcoords="offset points",

        ha="center", va="bottom",

    )



plt.xlabel("Year")

plt.show()
plt_init((12, 6))



g = sns.barplot(x="X.1", y="% of villages electrified", data=village_electrification)



for p in g.patches:

    h = p.get_height()

    g.annotate(

        str(h), xy=(p.get_x() + p.get_width() / 2, h),

        xytext=(0, 3),

        textcoords="offset points",

        ha="center", va="bottom",

    )



plt.xlabel("Year")

plt.show()
household_electrification_all_india = pd.read_csv("../input/households-electrification-whole-india/data-EQZVP.csv")
plt_init((10, 6))



g = plt.plot(

    household_electrification_all_india["X.1"],

    household_electrification_all_india["Rural"],

    "o--", label="Rural"

)



plt.plot(

    household_electrification_all_india["X.1"],

    household_electrification_all_india["Urban"],

    "o--", label="Urban"

)



plt.plot(

    household_electrification_all_india["X.1"],

    household_electrification_all_india["Total"],

    "o--", label="Total"

)



for _, row in household_electrification_all_india.iterrows():

    for h in row.iloc[1:]:

        plt.annotate(

            str(h),

            xy=(row["X.1"], h),

            xytext=(row["X.1"] - 0.5, h + 0.8)

        )



plt.xlabel("Year")

plt.ylabel("% of Households electrified")



plt.legend()

plt.show()
plt_init((18, 10))



sorted_nfhs_2016 = nfhs_data.sort_values("2016", ascending=False)



g = sns.barplot(x="State", y="2016", data=sorted_nfhs_2016)



for xtick in g.get_xticklabels():

    xtick.set_rotation(75)

    xtick.set_fontsize(11)



for p in g.patches:

    h = p.get_height()

    g.annotate(

        f"{h}",

        xy=(p.get_x() + p.get_width() / 2, h),

        xytext=(0, 3),

        textcoords="offset points",

        ha="center", va="bottom"

    )



plt.ylabel("State-wise % of electrified households in 2016")

    

plt.legend()

plt.show()
# Let's plot 2017 data



plt_init((20, 8))



g = sns.barplot(x="State", y="2017", data=sbhg_data_refined.sort_values("2017", ascending=False))



for xtick in g.get_xticklabels():

    xtick.set_rotation(75)



for p in g.patches:

    h = p.get_height()

    g.annotate(

        f"{h:.2f}",

        xy=(p.get_x() + p.get_width() / 2, h),

        xytext=(0, 3),

        textcoords="offset points",

        ha="center", va="bottom"

    )



plt.show()
plt_init((20, 8))



g = sns.barplot(x="State", y="2019", data=sbhg_data_refined)



for xtick in g.get_xticklabels():

    xtick.set_rotation(75)



for p in g.patches:

    h = p.get_height()

    g.annotate(

        f"{h}",

        xy=(p.get_x() + p.get_width() / 2, h),

        xytext=(0, 3),

        textcoords="offset points",

        ha="center", va="bottom"

    )



plt.show()
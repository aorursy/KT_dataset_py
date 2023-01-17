import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



import folium

from folium import plugins

sns.set()



import os

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



data_folder = '../input/crimes-in-boston'

# Any results you write to the current directory are saved as output.
CRIME_DATA_FILE = os.path.join(data_folder, "crime.csv")

print("Crime data:", CRIME_DATA_FILE)

crime_df = pd.read_csv(CRIME_DATA_FILE, encoding = 'ISO-8859-1')  # note the odd encoding...



print("Number of crimes:", len(crime_df))



# Peek a bit

crime_df.head()
# Do a bit of data processing

crime_df.UCR_PART = crime_df.UCR_PART.astype('category')

crime_df.OFFENSE_CODE_GROUP = crime_df.OFFENSE_CODE_GROUP.astype('category')

crime_df.SHOOTING.fillna("N", inplace=True)  # replace nan for shootings with 'N' for "no"

crime_df.OCCURRED_ON_DATE = pd.to_datetime(crime_df.OCCURRED_ON_DATE)  # use Pandas datetime

crime_df.Location = crime_df.Location.apply(eval)  # set location as tuple

crime_df.Lat.replace(-1.0, None, inplace=True)

crime_df.Long.replace(-1.0, None, inplace=True)

crime_df.dropna(subset=["Lat", "Long"], inplace=True)



rename_map = {

    "OFFENSE_CODE": "Code",

    "OFFENSE_CODE_GROUP": "Code_group",

    "OFFENSE_DESCRIPTION": "Description",

    "OCCURRED_ON_DATE": "Date",

    "DISTRICT": "District",

    "STREET": "Street",

    "YEAR": "Year",

    "MONTH": "Month",

    "HOUR": "Hour",

    "DAY_OF_WEEK": "Day_of_week",

    "SHOOTING": "Shooting"

}



crime_df.rename(columns=rename_map, inplace=True)

crime_df.sort_values(by="Date", inplace=True)
# Let's keep the major crimes

crime_df = crime_df[crime_df.UCR_PART == "Part One"]

# Remove the unused categories

crime_df.Code_group.cat.remove_unused_categories(inplace=True)
crime_df.head()
print(crime_df.Code_group.value_counts())

print()

print(crime_df.groupby("Year").Code_group.value_counts())
# NB: removing unused categories was helpful here because

# setting the code groups to 'categorical' before filtering Part One crimes

# meant Seaborn would pickup the empty Part (One, Two) categories

g = sns.catplot(y="Code_group", kind="count",

                data=crime_df,

                order=crime_df.Code_group.value_counts().index,

                aspect=1.6,

                palette="muted")

g.set_axis_labels("Number of occurrences", "Offense group")
# NB: removing unused categories was helpful here because

# setting the code groups to 'categorical' before filtering Part One crimes

# meant Seaborn would pickup the empty Part (One, Two) categories

g = sns.catplot(y="Code_group", col="Year", col_wrap=2, kind="count",

                data=crime_df,

                order=crime_df.Code_group.value_counts().index,

                aspect=1.5,

                height=3,

                palette="muted")

g.set_axis_labels(x_var="Number of occurrences", y_var="Offense group")
base_location = crime_df.Location.iloc[0]  # grab location of one offense

base_location
boston_map = folium.Map(location=base_location,

                        prefer_canvas=True,

                        zoom_start=12,

                        min_zoom=12)

plugins.ScrollZoomToggler().add_to(boston_map)



boston_map
# Now we add the homicides up to 2016

year = 2016

for row in crime_df[(crime_df.Code_group == "Homicide") & (crime_df.Year <= year)].itertuples(index=False):

    icon = folium.Icon(color='red', icon='times', prefix='fa')

    popup_txt = str(row.Date)

    if row.Shooting == 'Y':

        popup_txt = "Shooting " + popup_txt

    folium.Marker(row.Location, icon=icon, popup=popup_txt).add_to(boston_map)



boston_map
times_ = crime_df[crime_df.Code_group == "Homicide"].groupby(by="Year").Date



# Create counters for number of events

counts_ = []

for i, t in times_:

    counts_.append(np.arange(len(t)))
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

axes = axes.flatten()

for i, (year, t) in enumerate(times_):

    axes[i].step(t, counts_[i], where='post')

    axes[i].set_title(f"Number of homicides. Year = {year}")

    axes[i].set_xlabel("Date")

    axes[i].set_ylabel("Homicide count")

fig.tight_layout()
yearly_homic_rates = {}

for i, (y, ti) in enumerate(times_):

    dt_window = ti.iloc[-1] - ti.iloc[0]



    rate_mle = counts_[i][-1] / dt_window.days  # Poisson event rate in days

    yearly_homic_rates[y] = rate_mle

    print(f"Rate for year {y} (days^{{-1}}): {rate_mle:.3f}", f" i.e. every {1./rate_mle:.2f} days")
from scipy.spatial import distance as scdist

def ripley_K(data, t):

    """

    Args

        data (ndarray): point pattern data array

        t (float): interval radius

    """

    data = data.astype('datetime64[s]')  # ensure time data is in ms

    dist = scdist.pdist(data, 'euclidean')  # compute the array of pair-wise distances

    lbda = data.size / (data[-1] - data[0]).astype(float)

    ksum = 1./lbda * np.sum((dist <= t), axis=-1) / data.shape[0]

    return ksum
evt_time_data = crime_df[crime_df.Code_group == "Homicide"].Date[:, None].astype('datetime64[s]')



dist = scdist.pdist(evt_time_data.astype("datetime64[s]"), 'euclidean')

dist
lbda = evt_time_data.size / (evt_time_data[-1] - evt_time_data[0]).astype(float)

print(f"Event rate {lbda.item():.3e} events/s")

print(f"Event rate {86400*lbda.item():.3f} events/day")
# Let's make our testing time values run over 1 1/2 years

ripley_test_times = np.linspace(0, 86400 * 365 * 1.5, 150)  # time expressed in seconds



kstat = ripley_K(evt_time_data, ripley_test_times[:, None])
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

ax.plot(ripley_test_times, kstat, label="Empirical Ripley $\widehat{K}(t)$")

ax.plot(ripley_test_times, 2*ripley_test_times, label="Theoretical Ripley $K(t) = 2t$")

ax.legend()

ax.set_title("Ripley $K$-function")

ax.set_xlabel("Time $t$ (s)")

ax.set_ylabel("Statistic $\widehat{K}(t)$")
class LocalAverageKernel:

    def __init__(self, bandwidth):

        self.bandwidth = bandwidth

    

    def __call__(self, x):

        normalize_cst = float(2. * self.bandwidth)

        ks = (np.abs(x) <= self.bandwidth).astype(dtype='float')

        return ks / normalize_cst



class GaussianKernel:

    def __init__(self, bandwidth):

        self.bandwidth = bandwidth

    

    def __call__(self, x):

        x = x.astype(float)

        bw = float(self.bandwidth)  # in nanoseconds

        argument = - 0.5 * (x / bw) ** 2

        return np.exp(argument) / np.sqrt(2 * np.pi * bw ** 2)
dfmt = '%Y%m%d'
# Let's do a kernel estimation for 2017

evt_times_ = times_.get_group(2017)
plot_dates = pd.date_range('20170101', '20180101', freq='D')
print(plot_dates.shape)



print(evt_times_[None, :].shape)
np.shape(plot_dates[:, None] - evt_times_[None, :])
bandwidths = []

for bw in [7, 10, 15, 20]:

    bandwidth = pd.Timedelta('1D') * bw

    print("Bandwidth:", bandwidth)

    bandwidths.append(bandwidth)



# Let's get a few different kernels for our bandwidths

kernels_ = [GaussianKernel(bw.asm8) for bw in bandwidths]
dts = plot_dates[:, None] - evt_times_[None, :]
num_ns_day = 86400 * 1e9  # number of nanoseconds in a day
intensity_estims_ = [num_ns_day * kern(dts).sum(axis=-1) for kern in kernels_]
plt.figure(figsize=(14, 9))

for j, ins_estim in enumerate(intensity_estims_):

    plt.plot(plot_dates, ins_estim, linestyle='-', label=f"Bandwidth = {bandwidths[j]}")

plt.title("Intensity estimate for homicides, year 2017")

ylims = plt.ylim()

plt.vlines(evt_times_, *ylims, linestyles='--', lw=1.0, alpha=0.8)

plt.ylim(*ylims)

plt.legend()

plt.xlabel("Date")

plt.ylabel("Intensity (1/days)")
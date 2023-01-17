import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
!wget https://raw.githubusercontent.com/h4pZ/h4tils/master/h4tils/plotting/hibm.mplstyle
# Stuff :)
plt.style.use("hibm.mplstyle")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
data_path = "../input/all-space-missions-from-1957/Space_Corrected.csv"
space = pd.read_csv(data_path)
space.head()
space.info()
# Drop the first two columns because they aren't useful.
space = space.iloc[:, 2:]

# Fixing dtypes.
space[" Rocket"] = (space[" Rocket"].apply(lambda x: x.replace(",", "") if isinstance(x, str) else x)
                                  .astype("float"))
space["Datum"] = pd.to_datetime(space["Datum"], utc=True)

# Removing white spaces at the beginning and end of the column names.
space.columns = [col.strip() for col in space.columns]

# Adding features.
space["Year"] = (space["Datum"].dt
                               .year
                               .astype(np.int16))
space["Loc"] = space["Location"].apply(lambda x: x.split(",")[-1])
import missingno as msno
msno.bar(space, figsize=(13, 8), color="#A573E1")
plt.title("Feature Completeness");
top_n = 50
top_n_space = (space.sort_values(by="Rocket", ascending=False)
                    .iloc[:top_n, :])


fig, ax = plt.subplots(figsize=(10, top_n / 2))
sns.barplot(x="Rocket", y="Detail", data=top_n_space, palette="plasma")
ax.set_title(f"Top {top_n} most expensive rockets")
ax.set_xlabel("Cost of the rocket (in $ million)")
ax.set_ylabel("Rocket Name");
total_spent_companies = (space[["Company Name", "Rocket"]].groupby(by="Company Name")
                                                          .sum()
                                                          .reset_index()
                                                          .sort_values(by="Rocket", ascending=False)
                                                          .transform({"Company Name": lambda x: x,
                                                                      "Rocket": np.log10})
                                                          .replace(to_replace=-np.inf, value=np.nan)
                                                          .dropna())

n_companies = len(total_spent_companies)

fig, ax = plt.subplots(figsize=(10, n_companies / 2))
sns.barplot(x="Rocket", y="Company Name", data=total_spent_companies, palette="plasma")
ax.set_title(r"Total Money Spent by Companies")
ax.set_xlabel(r"Cost in $log_{10}$(million usd)");
location_race = (space[["Loc", "Year"]].groupby(by=["Loc", "Year"])
                                          .size()
                                          .reset_index(name="counts"))\

locations = location_race["Loc"].unique()
n_locations = len(locations)
palette = sns.husl_palette(n_locations, s=1.0)

fig, ax = plt.subplots(figsize=(13, 8))

for i, location in enumerate(locations):
    mask = location_race["Loc"] == location
    location_df = location_race[mask]
    ax.plot(location_df["Year"], location_df["counts"], label=location, color=palette[i])
    
ax.set_ylim(0)
ax.axvline(x=1969, linewidth=4, linestyle=":", label="Apollo 11 Moon Landing")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title("Space Race by Launch Location")
ax.set_ylabel("Number of Missions")
ax.set_xlabel("Year");

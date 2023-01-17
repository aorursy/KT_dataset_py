import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
data = pd.read_csv('../input/cumulative.csv')
data.head()
data.columns
ax = data.koi_kepmag.hist(bins=100, figsize=(12, 8))

ax.set_xlabel("Magnitude (lower is brighter)")

ax.set_title("All targets")
# only the Confirmed planets

ax = data[data.koi_disposition == 'CONFIRMED'].koi_kepmag.hist(bins=100, figsize=(12, 8))

ax.set_xlabel("Magnitude (lower is brighter)")

ax.set_title("Confirmed planet")
ax = data.koi_slogg.hist(bins=100, figsize=(12, 8))

ax.set_xlabel("$\log{g}$")

ax.set_title("Surface gravity")
confirmed = data[data.koi_disposition == 'CONFIRMED']



ra, dec = data.ra, data.dec

ra_c, dec_c = confirmed.ra, confirmed.dec
fig = plt.figure(figsize=(12, 12))



plt.scatter(ra, dec, s=3, label='Candidates')

plt.scatter(ra_c, dec_c, s=3, label="Confirmed")



plt.xlabel("Right Ascension")

plt.ylabel("Declination")



plt.legend()
ax = confirmed.koi_period.hist(bins=100, figsize=(12, 8))

ax.set_xlabel("Orbital Period (days)")
ax = confirmed.koi_duration.hist(bins=100, figsize=(12, 8))

ax.set_xlabel("Duration of transit (hours)")
ax = data.koi_steff.hist(bins=100, figsize=(12, 8), label="Star")

ax.set_xlabel("T$_{eff}$ K")

ax.set_title("Effective temperatures of stars and planets.")



data.koi_teq.hist(ax=ax, bins=100, label='Planet')

ax.legend()
# radius only for stars with a confirmed planet, the rest is not known.

ax = confirmed.koi_prad.hist(bins=100, label='Planet (earth radii)')

ax.set_title("Radii of stars and planets")



confirmed.koi_srad.hist(ax=ax, bins=100, figsize=(12, 8), label="Star (solar radii)")

ax.set_xlabel("Radius")

ax.legend()

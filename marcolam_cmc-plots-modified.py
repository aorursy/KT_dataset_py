import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data. Data are added with the "+ Add Data" button on the right hand panel.
# Look for the LSST competition
t = pd.read_csv("../input/training_set.csv")
m = pd.read_csv("../input/training_set_metadata.csv")
# Get the array of Object ID and length
objectid_list = np.unique(m['object_id'])
objectid_list_len = len(objectid_list)
# Loop 100 times
for j in range(100):

    # Draw 10 random Object IDs from the list
    objectid = objectid_list[np.int64(np.random.random() * objectid_list_len)]

    # Get the rows with the objectid drawn above
    lightcurve = t.loc[t['object_id']==objectid]

    # Data
    mjdstart = lightcurve['mjd'].iloc[0]
    u = lightcurve.loc[lightcurve['passband']==0]
    g = lightcurve.loc[lightcurve['passband']==1]
    r = lightcurve.loc[lightcurve['passband']==2]
    i = lightcurve.loc[lightcurve['passband']==3]
    z = lightcurve.loc[lightcurve['passband']==4]
    y = lightcurve.loc[lightcurve['passband']==5]

    # More data
    header = m.loc[m['object_id']==objectid]
    objclass = header['target'].iloc[0]
    specz = header['hostgal_specz'].iloc[0]
    photz = header['hostgal_photoz'].iloc[0]
    gallat = header['gal_b'].iloc[0]

    # Preset the title of the plot
    title = "Object number: " + str(objectid) + ", Object class: " + str(objclass) + "\n\n PhotZ = " + str(photz) + ", SpecZ = " + str(specz) + ", GalLat = " + str(gallat) + "deg"

    # Configure plot
    f = plt.figure(figsize=(12,12))
    plt.subplots_adjust(hspace=0)

    # Adding a subplot to the 6th(third argument) plotting environment of size 6 (first) down 1 (second) across
    ax1 = f.add_subplot(6,1,6)
    plt.ylabel("Y")
    plt.errorbar(y['mjd']-mjdstart, y['flux'], yerr=y['flux_err'], fmt='o',color='brown')
    plt.grid()

    ax2 = f.add_subplot(6,1,5,sharex=ax1)
    plt.ylabel("z")
    plt.errorbar(z['mjd']-mjdstart, z['flux'], yerr=z['flux_err'], fmt='o',color='red')
    plt.grid()

    ax3 = f.add_subplot(6,1,4,sharex=ax1)
    plt.ylabel("i")
    plt.errorbar(i['mjd']-mjdstart, i['flux'], yerr=i['flux_err'], fmt='o',color='orange')
    plt.grid()

    ax4 = f.add_subplot(6,1,3,sharex=ax1)
    plt.ylabel("r")
    plt.errorbar(r['mjd']-mjdstart, r['flux'], yerr=r['flux_err'], fmt='o',color='green')
    plt.grid()

    ax5 = f.add_subplot(6,1,2,sharex=ax1)
    plt.ylabel("g")
    plt.errorbar(g['mjd']-mjdstart, g['flux'], yerr=g['flux_err'], fmt='o',color='blue')
    plt.grid()

    ax6 = f.add_subplot(6,1,1,sharex=ax1)
    plt.ylabel("u")
    plt.errorbar(u['mjd']-mjdstart, u['flux'], yerr=u['flux_err'], fmt='o',color='purple')
    plt.grid()

    plt.title(title)
    plt.xlabel("MJD")

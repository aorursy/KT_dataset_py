"""
 plot training data in HealPix
"""

import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt


""" read data """
df_gal_lb=pd.read_csv("./data/training_set_metadata.csv")

""" set grid lines"""
hp.graticule()

""" draw scatter plot in HealPix """
object=hp.projscatter(df_gal_lb.gal_l, df_gal_lb.gal_b, s=5,
                      c=df_gal_lb.hostgal_specz, cmap="jet", lonlat=True, coord="G")

""" draw colorbar """
cbar=plt.colorbar(object, orientation="horizontal")
cbar.set_label("red shift",size=14)

""" show image """
plt.show()

#numeric
import numpy as np
import pandas as pd

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
import folium

plt.style.use('bmh')
%matplotlib inline

#system
import os
import re

#Pandas warnings
import warnings
warnings.filterwarnings('ignore')
loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
phil_loans = loans[loans.country == 'Philippines']
geonames_phil = pd.read_csv('../input/administrative-regions-in-the-philippines/ph_regions.csv')

from difflib import get_close_matches

def match_region(loc_string, match_entity = 'province', split = True):
    if split == True:
        region = loc_string.split(',')[-1]
    else:
        region = loc_string
    
    matches = get_close_matches(region, geonames_phil[match_entity].unique().tolist())
    
    if not matches:
        return 'no_match'
    else:
        return geonames_phil.region[geonames_phil[match_entity] == matches[0]].iloc[0]
    
phil_loans.region.fillna('', inplace = True)
phil_loans.rename(columns = {'region' : 'location'}, inplace = True)
phil_loans['region'] = [match_region(loc_string) for loc_string in phil_loans.location]

city_drop = re.compile(r'(.*)(city)', re.I)
phil_loans.location[phil_loans.region == 'no_match'] = [re.match(city_drop, l).group(1).lower()\
                                                        if re.match(city_drop, l)\
                                                        else l for l\
                                                        in phil_loans.location[phil_loans.region == 'no_match']]

phil_loans['region'][phil_loans.region == 'no_match'] = np.vectorize(match_region)(phil_loans['location'][phil_loans.region == 'no_match'], 'city', False)

phil_loans.region[phil_loans.location == 'Sogod Cebu'] = geonames_phil.region[geonames_phil.city == 'cebu'].iloc[0]

phil_loans_extract = phil_loans[(phil_loans.borrower_genders.notna()) & (phil_loans.region != 'no_match')]

phil_loans_extract['borrower_genders'] = phil_loans_extract['borrower_genders']\
.map({'female' : 1,\
      'male' : 0})

phil_loans_extract.rename(columns = {'borrower_genders' : 'house_head_sex_f'}, inplace = True)


phil_loans_extract.to_csv('kiva_loans_ph_transofrmed.csv')
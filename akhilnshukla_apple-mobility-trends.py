import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
apple = pd.read_csv('../input/apples-mobility-report/Copy of applemobilitytrends-2020-04-14.csv')
apple.head()
apple.shape
apple.info()
apple.describe()
print(apple.geo_type.unique())
print(apple.transportation_type.unique())
# drop country/ region as it is not needed in the analysis

apple = apple.drop(['geo_type'], axis = 1)
print(apple.region.nunique())
regions = list(apple.region.unique())
print(regions)
apple_total_mob_per_day = apple.groupby(['region']).sum().reset_index()
apple_total_mob_per_day
# defining a function for plotting the cumulative mobility across all transport types for a given region/country

def plot_mobility_total(region_country):
    apple_total_mob_per_day_region = apple_total_mob_per_day[apple_total_mob_per_day.region == region_country]
    apple_total_mob_per_day_region = apple_total_mob_per_day_region.drop(['region'], axis = 1)
    apple_total_mob_per_day_region = apple_total_mob_per_day_region.T
    apple_total_mob_per_day_region.columns = ['Cumulative Mobility across all transport']
    apple_total_mob_per_day_region.plot.line()
    
    
    
# defining function to take region name input from the user.

def input_region():
    print('\nWhich region/country mobility report would you like to see?')
    region_name = input()
    if(region_name in regions):
        print('\nBelow plotted is the mobility report for the region/country: ', region_name)
        plot_mobility_total(region_name)
    else:
        print('\nData on this region/country is not available')
        print('\nWanna see the mobility plots of some other region/country ? Enter `Y` or `N`')
        ques = input()
        if(ques == 'Y'):
            input_region()
        else:
            print('\n\n\nCOVID-19 has spread all across the world. Many countries have adopted lockdown measures in their regions, evident from the plots you saw. Stay Home Stay Safe.')
# Lets plot for Germany.
input_region()
apple.transportation_type.value_counts()
# defining a function for plotting the transportation_type wise mobility for a given region/country

def plot_mobility_region_trans(region_country_trans):
    apple_mob_trans_wise = apple[apple.region == region_country_trans]
    apple_mob_trans_wise = apple_mob_trans_wise.drop(['region'], axis = 1)
    apple_mob_trans_wise = apple_mob_trans_wise.T

    if len(apple_mob_trans_wise.columns) < 3:
        apple_mob_trans_wise.columns = ['driving', 'walking']
    elif len(apple_mob_trans_wise.columns) == 3:
        apple_mob_trans_wise.columns = ['driving', 'walking', 'transit']
    
    apple_mob_trans_wise = apple_mob_trans_wise[1:]
    apple_mob_trans_wise.plot.line()
    
    
# defining function to take region name input from the user.
    
def input_region_trans():
    print('Which region/country mobility report would you like to see?')
    region_name_trans = input()
    if(region_name_trans in regions):
        print('Below plotted is the mobility report for the region/country: ', region_name_trans)
        plot_mobility_region_trans(region_name_trans)
    else:
        print('Data on this country is not available')
        print('\nWanna see the mobility plots of some other region/country ? Enter `Y` or `N`')
        ques = input()
        if(ques == 'Y'):
            input_region_trans()
        else:
            print('\n\n\nCOVID-19 has spread all across the world. Many countries have adopted lockdown measures in their regions, evident from the plots you saw. Stay Home Stay Safe.')
# Lets plot for Detroit, USA
input_region_trans()
# Lets plot for Spain
input_region_trans()
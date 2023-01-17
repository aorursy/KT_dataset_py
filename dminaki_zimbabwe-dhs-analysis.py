import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import glob as glob
"""Zimbabwe Data"""
zimbabwe_clean = pd.read_csv('../input/zimbabwe-data/zimbabwe_full.csv').drop(columns=['Unnamed: 0'])
Zimbabwe_dhs_mpi_admin2_sjoin = pd.read_csv('../input/zimbabwe-preprocessed/Zimbabwe_dhs_mpi_admin2_sjoin.csv')
zimbabwe_dhs = pd.read_csv('../input/zimbabwe-preprocessed/zimbabwe_dhs_cluster.csv')
#Columns with Binary Data
binary_columns = ['financial_depriv', 'electricity_depriv', 'water_depriv', 'sanitation_depriv',\
                  'cooking_fuel_depriv', 'floor_depriv', 'information_asset', 'mobility_asset',\
                  'livelihood_asset', 'asset_depriv','child_under_5', 'woman_15_to_49',\
                  'child_not_in_school', 'school_attainment_depriv', 'school_attendance_depriv']

#Get Percent Deprivation by DHS by grouping dhs by mean.
binary_dhs_data = zimbabwe_clean.groupby('DHSID').mean()[binary_columns].reset_index()

#Merge Percentiles to DHS
zimbabwe_dhs = zimbabwe_dhs.merge(binary_dhs_data)
#Columns that need to be Averaged
average_columns = ['total_household_members','total_of_weighted_deprivations','headcount_poor','total_poverty_intensity']

#Average columns grouped by DHS
average_dhs_data = zimbabwe_clean.groupby('DHSID').mean()[average_columns].reset_index()

#Merge Averages to DHS
zimbabwe_dhs = zimbabwe_dhs.merge(average_dhs_data)
#Population Change by DHS
zimbabwe_dhs['population_change_2010_2015'] = zimbabwe_dhs['All_Population_Count_2015'] / zimbabwe_dhs['All_Population_Count_2010']

#Population Density Change by DHS
zimbabwe_dhs['pop_density_change_2010_2015'] = zimbabwe_dhs['All_Population_Density_2015'] / zimbabwe_dhs['All_Population_Density_2010']

#Vegetation Index Change by DHS
zimbabwe_dhs['vegetation_change_2010_2015'] = zimbabwe_dhs['Enhanced_Vegetation_Index_2015'] / zimbabwe_dhs['Enhanced_Vegetation_Index_2010']

#Insecticide-Treated Nets (ITN) Cover Change by DHS
zimbabwe_dhs['itn_coverage_change_2010_2015'] = zimbabwe_dhs['ITN_Coverage_2015'] / zimbabwe_dhs['ITN_Coverage_2010']

#Malaria Change by DHS
zimbabwe_dhs['malaria_change_2010_2015'] = zimbabwe_dhs['Malaria_2015'] / zimbabwe_dhs['Malaria_2010']

#Rainfall Change by DHS
zimbabwe_dhs['rainfall_change_2010_2015'] = zimbabwe_dhs['Rainfall_2015'] / zimbabwe_dhs['Rainfall_2010']
print('Total Population: ', np.sum(zimbabwe_dhs.All_Population_Count_2015))
_ = plt.figure(figsize=(14,12))
_ = plt.subplot(2,1,1)
_ = plt.hist(zimbabwe_dhs.All_Population_Count_2015, bins=40)
_ = plt.axvline(np.mean(zimbabwe_dhs.All_Population_Count_2015), color='red')
_ = plt.title('Population by DHS 2015')
_ = plt.xlabel('Population')

_ = plt.subplot(2,1,2)
_ = plt.hist(zimbabwe_dhs.All_Population_Density_2015, bins=40)
_ = plt.axvline(np.mean(zimbabwe_dhs.All_Population_Density_2015), color='red')
_ = plt.title('Population Density by DHS 2015')
_ = plt.xlabel('Population Density')
_ = plt.figure(figsize=(14,6))
_ = plt.hist(zimbabwe_dhs.population_change_2010_2015, bins=80)
_ = plt.axvline(np.mean(zimbabwe_dhs.population_change_2010_2015), color='red')
_ = plt.title('Population Growth Distribution by DHS')
_ = plt.xlabel('Percent Change of Population')
_ = plt.figure(figsize=(14,6))
_ = sns.kdeplot(zimbabwe_dhs.total_household_members, shade=True)
_ = plt.axvline(np.mean(zimbabwe_dhs.total_household_members), color='red')
_ = plt.title('Household Size Distribution by DHS')
_ = plt.xlabel('Total Household Distribution')
_ = plt.figure(figsize=(14,6))
_ = sns.kdeplot(zimbabwe_dhs.total_poverty_intensity, shade=True)
_ = plt.axvline(np.mean(zimbabwe_dhs.total_poverty_intensity), color='red')
_ = plt.title('Poverty Distribution by DHS')
_ = plt.xlabel('Total Poverty Intensity')
_ = plt.figure(figsize=(14,6))
_ = sns.kdeplot(zimbabwe_dhs.headcount_poor, shade=True)
_ = plt.axvline(np.mean(zimbabwe_dhs.headcount_poor), color='red')
_ = plt.title('Average Household Poor Headcount by DHS')
_ = plt.xlabel('Headcount per Household')
_ = plt.figure(figsize=(15,6))
_ = sns.kdeplot(zimbabwe_dhs['child_under_5'] * 100, shade=True)
_ = plt.axvline(np.mean(zimbabwe_dhs['child_under_5']) * 100, color='red')
_ = plt.title('Percent of Families with Child Under 5')
_ = plt.xlabel('Percent')
_ = plt.ylabel('Count')
deprivation = ['financial_depriv', 'electricity_depriv', 'water_depriv',\
               'sanitation_depriv', 'cooking_fuel_depriv', 'floor_depriv']

_ = plt.figure(figsize=(16,8))
for i in range(len(deprivation)):
    column = deprivation[i]
    _ = plt.subplot(2,3, i + 1)
    _ = plt.subplots_adjust(bottom=-0.2)
    _ = sns.kdeplot(zimbabwe_dhs[column] * 100, shade=True)
    _ = plt.axvline(np.mean(zimbabwe_dhs[column]) * 100, color='red')
    _ = plt.title('Percent with ' + column.capitalize())
    _ = plt.xlabel('Percent')
    _ = plt.ylabel('Count')
asset_columns = ['information_asset', 'mobility_asset', 'livelihood_asset', 'asset_depriv']

_ = plt.figure(figsize=(15,8))
for i in range(len(asset_columns)):
    column = asset_columns[i]
    _ = plt.subplot(2,2, i + 1)
    _ = plt.subplots_adjust(bottom=-0.1)
    _ = sns.kdeplot(zimbabwe_dhs[column] * 100, shade=True)
    _ = plt.axvline(np.mean(zimbabwe_dhs[column]) * 100, color='red')
    _ = plt.title('Percent with ' + column.capitalize())
    _ = plt.xlabel('Percent')
    _ = plt.ylabel('Count')
school_columns = ['child_not_in_school', 'school_attainment_depriv', 'school_attendance_depriv']
school_titles = ['Percent of Children Not In School','Percent Deprived of School Attainment','Percent Deprived of School Attendance']
_ = plt.figure(figsize=(15,8))
for i in range(len(school_columns)):
    column = school_columns[i]
    _ = plt.subplot(3,1, i + 1)
    _ = plt.subplots_adjust(bottom=-0.6)
    _ = plt.hist(zimbabwe_dhs[column] * 100, bins=20)
    _ = plt.axvline(np.mean(zimbabwe_dhs[column]) * 100, color='red')
    _ = plt.title(school_titles[i])
    _ = plt.xlabel('Percent')
    _ = plt.ylabel('Count')
geographic = ['Proximity_to_National_Borders','Proximity_to_Protected_Areas','Proximity_to_Water','Travel_Times']
geographic_titles = ['Proximity to National Borders','Proximity to Protected Areas','Proximity to Water','Travel Times']
geographic_xlabel = ['Kilometers (km)','Kilometers (km)','Kilometers (km)','Minutes']
_ = plt.figure(figsize=(15,8))
for i in range(len(geographic)):
    column = geographic[i]
    _ = plt.subplot(2,2, i + 1)
    _ = plt.subplots_adjust(bottom=-0.5)
    _ = sns.kdeplot(zimbabwe_dhs[column], shade=True)
    _ = plt.axvline(np.mean(zimbabwe_dhs[column]), color='red')
    _ = plt.title(geographic_titles[i])
    _ = plt.xlabel(geographic_xlabel[i])
    _ = plt.ylabel('Count')
water_columns = ['Rainfall_2010','Rainfall_2015','Enhanced_Vegetation_Index_2010','Enhanced_Vegetation_Index_2015',\
                 'Drought_Episodes', 'Aridity','Growing_Season_Length','Proximity_to_Water']

_ = plt.figure(figsize=(15,12))
for i in range(len(water_columns)):
    column = water_columns[i]
    _ = plt.subplot(4,2, i + 1)
    _ = plt.subplots_adjust(bottom=-0.6)
    _ = sns.kdeplot(zimbabwe_dhs[column], shade=True)
    _ = plt.axvline(np.mean(zimbabwe_dhs[column]), color='red')
    _ = plt.title(' '.join(water_columns[i].split('_')))
malaria = ['ITN_Coverage_2010', 'ITN_Coverage_2015','Malaria_2010', 'Malaria_2015']
malaria_title = ['ITN Coverage in 2010', 'ITN Coverage in 2015','Malaria Cases in 2010', 'Malaria Cases in 2015']
malaria_real = zimbabwe_dhs[zimbabwe_dhs.Malaria_2010 > 0]

for i in range(len(malaria)):
    print('Average ', malaria_title[i], ': ', np.mean(malaria_real[malaria[i]]))
_ = plt.figure(figsize=(15,8))
for i in range(len(malaria)):
    column = malaria[i]
    _ = plt.subplot(2,2, i + 1)
    _ = plt.subplots_adjust(bottom=-0.6)
    _ = sns.kdeplot(malaria_real[column] * 100, shade=True)
    _ = plt.axvline(np.mean(malaria_real[column]) * 100, color='red')
    _ = plt.title(malaria_title[i])
    _ = plt.xlabel('Percent')
    _ = plt.ylabel('Count')
_ = plt.figure(figsize=(14,6))
_ = sns.kdeplot(zimbabwe_dhs.Global_Human_Footprint, shade=True)
_ = plt.axvline(np.mean(zimbabwe_dhs.Global_Human_Footprint), color='red')
_ = plt.title('Global Human Footprint by DHS')
_ = plt.xlabel('Global Human Footprint')
_ = plt.figure(figsize=(14,6))
_ = sns.kdeplot(zimbabwe_dhs.Nightlights_Composite, shade=True)
_ = plt.axvline(np.mean(zimbabwe_dhs.Nightlights_Composite), color='red')
_ = plt.title('Nightlight Composite by DHS')
_ = plt.xlabel('Nightlight Composite')
slope, intercept = np.polyfit(zimbabwe_dhs.total_poverty_intensity, zimbabwe_dhs.total_household_members, 1)
x_range = range(4)
_ = plt.figure(figsize=(14,6))
_ = plt.plot(zimbabwe_dhs.total_poverty_intensity, zimbabwe_dhs.total_household_members, linestyle='none', marker='.')
_ = plt.plot(x_range, slope*x_range + intercept)
_ = plt.title('Relationship Between Poverty and Household Size')
_ = plt.xlabel('Total Poverty Intensity')
_ = plt.ylabel('Total Household Members')
print('Correlation: ', np.corrcoef(zimbabwe_dhs.total_poverty_intensity, zimbabwe_dhs.total_household_members)[0][1])
slope, intercept = np.polyfit(zimbabwe_dhs.total_poverty_intensity, zimbabwe_dhs.Global_Human_Footprint, 1)
x_range = range(4)
_ = plt.figure(figsize=(14,6))
_ = plt.plot(zimbabwe_dhs.total_poverty_intensity, zimbabwe_dhs.Global_Human_Footprint, linestyle='none', marker='.')
_ = plt.plot(x_range, slope*x_range + intercept)
_ = plt.title('Relationship Between Poverty and Human Footprint')
_ = plt.xlabel('Total Poverty Intensity')
_ = plt.ylabel('Global Human Footprint')
print('Correlation: ', np.corrcoef(zimbabwe_dhs.total_poverty_intensity, zimbabwe_dhs.Global_Human_Footprint)[0][1])
slope, intercept = np.polyfit(zimbabwe_dhs.total_poverty_intensity, zimbabwe_dhs.Nightlights_Composite, 1)
x_range = range(4)
_ = plt.figure(figsize=(14,6))
_ = plt.plot(zimbabwe_dhs.total_poverty_intensity, zimbabwe_dhs.Nightlights_Composite, linestyle='none', marker='.')
_ = plt.plot(x_range, slope*x_range + intercept)
_ = plt.title('Relationship Between Poverty and Nightlight Composite')
_ = plt.xlabel('Total Poverty Intensity')
_ = plt.ylabel('Nightlight Composites')
print('Correlation: ', np.corrcoef(zimbabwe_dhs.total_poverty_intensity, zimbabwe_dhs.Nightlights_Composite)[0][1])
slope, intercept = np.polyfit(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.All_Population_Density_2015, 1)
x_range = range(max(zimbabwe_dhs.All_Population_Count_2015.astype(int)))
_ = plt.figure(figsize=(14,6))
_ = plt.plot(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.All_Population_Density_2015, linestyle='none', marker='.')
_ = plt.plot(x_range, slope*x_range + intercept)
_ = plt.title('Relationship Between Population Size and population Density')
_ = plt.xlabel('Population Size')
_ = plt.ylabel('Population Density')
print('Correlation: ', np.corrcoef(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.All_Population_Density_2015)[0][1])
slope, intercept = np.polyfit(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.total_poverty_intensity, 1)
x_range = range(max(zimbabwe_dhs.All_Population_Count_2015.astype(int)))
_ = plt.figure(figsize=(14,6))
_ = plt.plot(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.total_poverty_intensity, linestyle='none', marker='.')
_ = plt.plot(x_range, slope*x_range + intercept)
_ = plt.title('Relationship Between Population Size and Poverty Intensity')
_ = plt.xlabel('Population Size')
_ = plt.ylabel('Poverty Intensity')
print('Correlation: ', np.corrcoef(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.total_poverty_intensity)[0][1])
slope, intercept = np.polyfit(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.total_of_weighted_deprivations, 1)
x_range = range(max(zimbabwe_dhs.All_Population_Count_2015.astype(int)))
_ = plt.figure(figsize=(14,6))
_ = plt.plot(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.total_of_weighted_deprivations, linestyle='none', marker='.')
_ = plt.plot(x_range, slope*x_range + intercept)
_ = plt.title('Relationship Between Population Size and Weighted Deprivation')
_ = plt.xlabel('Population Size')
_ = plt.ylabel('Weighted Deprivation')
print('Correlation: ', np.corrcoef(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.total_of_weighted_deprivations)[0][1])
slope, intercept = np.polyfit(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Proximity_to_National_Borders, 1)
x_range = range(max(zimbabwe_dhs.All_Population_Count_2015.astype(int)))
_ = plt.figure(figsize=(14,6))
_ = plt.plot(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Proximity_to_National_Borders, linestyle='none', marker='.')
_ = plt.plot(x_range, slope*x_range + intercept)
_ = plt.title('Relationship Between Population and Proximity to National Borders')
_ = plt.xlabel('Population Size')
_ = plt.ylabel('Proximity to National Borders')
print('Correlation: ', np.corrcoef(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Proximity_to_National_Borders)[0][1])
slope, intercept = np.polyfit(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Proximity_to_Protected_Areas, 1)
x_range = range(max(zimbabwe_dhs.All_Population_Count_2015.astype(int)))
_ = plt.figure(figsize=(14,6))
_ = plt.plot(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Proximity_to_Protected_Areas, linestyle='none', marker='.')
_ = plt.plot(x_range, slope*x_range + intercept)
_ = plt.title('Relationship Between Population and Proximity to Protected Areas')
_ = plt.xlabel('Population Size')
_ = plt.ylabel('Proximity to Protected Areas')
print('Correlation: ', np.corrcoef(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Proximity_to_Protected_Areas)[0][1])
slope, intercept = np.polyfit(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Proximity_to_Water, 1)
x_range = range(max(zimbabwe_dhs.All_Population_Count_2015.astype(int)))
_ = plt.figure(figsize=(14,6))
_ = plt.plot(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Proximity_to_Water, linestyle='none', marker='.')
_ = plt.plot(x_range, slope*x_range + intercept)
_ = plt.title('Relationship Between Population and Proximity to Water')
_ = plt.xlabel('Population Size')
_ = plt.ylabel('Proximity to Water')
print('Correlation: ', np.corrcoef(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Proximity_to_Water)[0][1])
slope, intercept = np.polyfit(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Drought_Episodes, 1)
x_range = range(max(zimbabwe_dhs.All_Population_Count_2015.astype(int)))
_ = plt.figure(figsize=(14,6))
_ = plt.plot(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Drought_Episodes, linestyle='none', marker='.')
_ = plt.plot(x_range, slope*x_range + intercept)
_ = plt.title('Relationship Between Population and Proximity to Water')
_ = plt.xlabel('Population Size')
_ = plt.ylabel('Proximity to Water')
print('Correlation: ', np.corrcoef(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Drought_Episodes)[0][1])
slope, intercept = np.polyfit(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Growing_Season_Length, 1)
x_range = range(max(zimbabwe_dhs.All_Population_Count_2015.astype(int)))
_ = plt.figure(figsize=(14,6))
_ = plt.plot(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Growing_Season_Length, linestyle='none', marker='.')
_ = plt.plot(x_range, slope*x_range + intercept)
_ = plt.title('Relationship Between Population and Global Human Footprint')
_ = plt.xlabel('Population Size')
_ = plt.ylabel('Global Human Footprint')
print('Correlation: ', np.corrcoef(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Growing_Season_Length)[0][1])
slope, intercept = np.polyfit(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Global_Human_Footprint, 1)
x_range = range(max(zimbabwe_dhs.All_Population_Count_2015.astype(int)))
_ = plt.figure(figsize=(14,6))
_ = plt.plot(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Global_Human_Footprint, linestyle='none', marker='.')
_ = plt.plot(x_range, slope*x_range + intercept)
_ = plt.title('Relationship Between Population and Global Human Footprint')
_ = plt.xlabel('Population Size')
_ = plt.ylabel('Global Human Footprint')
print('Correlation: ', np.corrcoef(zimbabwe_dhs.All_Population_Count_2015, zimbabwe_dhs.Global_Human_Footprint)[0][1])
slope, intercept = np.polyfit(zimbabwe_dhs.itn_coverage_change_2010_2015, zimbabwe_dhs.malaria_change_2010_2015, 1)
x_range = range(8)
_ = plt.figure(figsize=(14,6))
_ = plt.plot(zimbabwe_dhs.itn_coverage_change_2010_2015, zimbabwe_dhs.malaria_change_2010_2015, linestyle='none', marker='.')
_ = plt.plot(x_range, slope*x_range + intercept)
_ = plt.title('Relationship Between Change in ITN Coverage and Change in Malaria')
_ = plt.xlabel('Change in ITN Coverage')
_ = plt.ylabel('Change in Malaria')
print('Correlation: ', np.corrcoef(zimbabwe_dhs.itn_coverage_change_2010_2015, zimbabwe_dhs.malaria_change_2010_2015)[0][1])
slope, intercept = np.polyfit(malaria_real.Malaria_2010, malaria_real.itn_coverage_change_2010_2015, 1)
x_range = range(2)
_ = plt.figure(figsize=(14,6))
_ = plt.plot(malaria_real.Malaria_2010, malaria_real.itn_coverage_change_2010_2015, linestyle='none', marker='.')
_ = plt.plot(x_range, slope*x_range + intercept)
_ = plt.title('Relationship Between Change in ITN Coverage and Change in Malaria')
_ = plt.xlabel('Change in ITN Coverage')
_ = plt.ylabel('Change in Malaria')
print('Correlation: ', np.corrcoef(malaria_real.Malaria_2010, malaria_real.itn_coverage_change_2010_2015)[0][1])
slope, intercept = np.polyfit(zimbabwe_dhs.total_poverty_intensity, zimbabwe_dhs.total_of_weighted_deprivations, 1)
x_range = range(4)
_ = plt.figure(figsize=(14,6))
_ = plt.plot(zimbabwe_dhs.total_poverty_intensity, zimbabwe_dhs.total_of_weighted_deprivations, linestyle='none', marker='.')
_ = plt.plot(x_range, slope*x_range + intercept)
_ = plt.title('Relationship Between Total Poverty Intensity and Total of Weighted Deprivation')
_ = plt.xlabel('Total Poverty Intensity')
_ = plt.ylabel('Total of Weighted Deprivation')
print('Correlation: ', np.corrcoef(zimbabwe_dhs.total_poverty_intensity, zimbabwe_dhs.total_of_weighted_deprivations)[0][1])
deprivation = ['financial_depriv', 'electricity_depriv', 'water_depriv',\
               'sanitation_depriv', 'cooking_fuel_depriv', 'floor_depriv']

_ = plt.figure(figsize=(16,13))
position = 1
for x in range(len(deprivation)):
    row = deprivation[x]
    for y in range(len(deprivation)):
        column = deprivation[y]
        slope, intercept = np.polyfit(zimbabwe_dhs[row], zimbabwe_dhs[column], 1)
        x_range = range(8)
        _ = plt.subplot(6,6, position)
        _ = plt.subplots_adjust(bottom=-0.4)
        _ = plt.plot(zimbabwe_dhs[row], zimbabwe_dhs[column], linestyle='none', marker='.')
        _ = plt.plot(x_range, slope*x_range + intercept)
        _ = plt.title(row.split('_')[0] + ' vs ' + column.split('_')[0])
        position = position + 1
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y
#Bootstrap Data
def bootstrap_replicate_1d(data, func):
    """Returns Single Bootstrap Replicate"""
    return func(np.random.choice(data, size=len(data)))

#Repeat bootstrap x amount of times based on size
def draw_bs_reps(data, func, size=1):
    """Bootstrap Replicates for function func and size size"""
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)
        
    return bs_replicates
def diff_of_means(data1, data2):
    """Difference of Means"""
    return np.mean(data1) - np.mean(data2)

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

def draw_perm_reps(data1, data2, func, size=1):
    """Permutation Replicates for function func and size size"""
    perm_replicates = np.empty(size)
    
    for i in range(size):
        perm_sample_1, perm_sample_2 = permutation_sample(data1, data2)
        
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)
        
    return perm_replicates
zimbabwe_dhs = zimbabwe_dhs[zimbabwe_dhs.Malaria_2015 > 0]
_ = plt.figure(figsize=(14,6))
_ = sns.kdeplot(zimbabwe_dhs.electricity_depriv, shade=True)
_ = plt.axvline(np.mean(zimbabwe_dhs.electricity_depriv), color='red')
_ = plt.title('Electricity Deprivation by DHS')
_ = plt.xlabel('Percent Deprived')
high_electricity_depriv = zimbabwe_dhs[zimbabwe_dhs.electricity_depriv == 1]
no_electricity_depriv = zimbabwe_dhs[zimbabwe_dhs.electricity_depriv < 0.1]
high_electricity_depriv.shape
no_electricity_depriv.shape
def get_change(row):
    current = row.no_electric_ave
    previous = row.high_electric_ave
    if current == previous:
        return 100.0
    try:
        return (abs(current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return 0
#Average All Variables of DHSs with No Electricity Deprivation
no_electricity_depriv_averages = pd.DataFrame(no_electricity_depriv.mean()).reset_index()
no_electricity_depriv_averages.columns = ['variable','no_electric_ave']

#Average All Variables of DHSs with High Electricity Deprivation
high_electricity_depriv_averages = pd.DataFrame(high_electricity_depriv.mean()).reset_index()
high_electricity_depriv_averages.columns = ['variable','high_electric_ave']

#Merge Tables for comparison
electric_depriv_averages = no_electricity_depriv_averages.merge(high_electricity_depriv_averages)

#Calculate Difference of Variables
electric_depriv_averages['mean_differences'] = abs(electric_depriv_averages['no_electric_ave'] - electric_depriv_averages['high_electric_ave'])
electric_depriv_averages['perc_change'] = electric_depriv_averages.apply(get_change, axis=1)
electric_depriv_averages[electric_depriv_averages.variable.isin(deprivation) == False].sort_values('perc_change', ascending=False).head(15)
x_pop_high, y_pop_high = ecdf(high_electricity_depriv.All_Population_Density_2015)
x_pop_no, y_pop_no = ecdf(no_electricity_depriv.All_Population_Density_2015)

_ = plt.figure(figsize=(14,6))
_ = plt.plot(x_pop_high, y_pop_high, linestyle='none', marker='.')
_ = plt.plot(x_pop_no, y_pop_no, linestyle='none', marker='.')
_ = plt.title('Population Density by Electricity Deprivation')
_ = plt.legend(['High Electricity Deprevation','No Electricity Deprevation'])
_ = plt.xlabel('Population Density')
_ = plt.ylabel('ECDF')
#Bootstrap Density Distributions of Data
bs_reps_pop_high_electric_depriv = draw_bs_reps(high_electricity_depriv.All_Population_Density_2015, np.mean, 1000)
bs_reps_pop_low_electric_depriv = draw_bs_reps(no_electricity_depriv.All_Population_Density_2015, np.mean, 1000)
_ = plt.figure(figsize=(14,6))
_ = sns.kdeplot(bs_reps_pop_high_electric_depriv, shade=True)
_ = sns.kdeplot(bs_reps_pop_low_electric_depriv, shade=True)
_ = plt.title('Bootstrapped Population Density')
_ = plt.legend(['High Electricity Deprivation','Low Electricity Deprivation'])
print('High Electricity Deprivation Bootstrap Mean: ', np.mean(bs_reps_pop_high_electric_depriv))
print('High Electricity Deprivation (95% Conf. Int): ', np.percentile(bs_reps_pop_high_electric_depriv, [2.5,97.5]))
print('')
print('Low Electricity Deprivation Bootstrap Mean: ', np.mean(bs_reps_pop_low_electric_depriv))
print('Low Electricity Deprivation (95% Conf. Int): ', np.percentile(bs_reps_pop_low_electric_depriv, [2.5,97.5]))
true_pop_diff_of_means = np.mean(no_electricity_depriv.All_Population_Density_2015) - np.mean(high_electricity_depriv.All_Population_Density_2015)
print("True Difference Of Means: ", true_pop_diff_of_means)
pop_density_perms = draw_perm_reps(no_electricity_depriv.All_Population_Density_2015, high_electricity_depriv.All_Population_Density_2015, diff_of_means, 10000)
_ = plt.figure(figsize=(14,6))
_ = plt.hist(pop_density_perms, bins=50)
_ = plt.axvline(np.mean(pop_density_perms), color='red')
_ = plt.axvline(true_pop_diff_of_means, color='blue')
_ = plt.title('Difference in Mean Population Density Between High and Low Electricity Deprived DHSs')
_ = plt.legend(('Sample Diff of Means','True Diff of Means','Difference Of Means'))
_ = plt.xlabel('Difference of Means')
print('P-Value: ', np.sum(pop_density_perms > true_pop_diff_of_means) / len(pop_density_perms))
x_pov_high, y_pov_high = ecdf(high_electricity_depriv.total_poverty_intensity)
x_pov_no, y_pov_no = ecdf(no_electricity_depriv.total_poverty_intensity)

_ = plt.figure(figsize=(14,6))
_ = plt.plot(x_pov_high, y_pov_high, linestyle='none', marker='.')
_ = plt.plot(x_pov_no, y_pov_no, linestyle='none', marker='.')
_ = plt.title('Poverty by Electricity Deprivation')
_ = plt.legend(['High Deprivation','No Deprivation'])
_ = plt.xlabel('Poverty Intensity')
_ = plt.ylabel('ECDF')
#Bootstrap Density Distributions of Data
bs_reps_pov_high_electric_depriv = draw_bs_reps(high_electricity_depriv.total_poverty_intensity, np.mean, 1000)
bs_reps_pov_low_electric_depriv = draw_bs_reps(no_electricity_depriv.total_poverty_intensity, np.mean, 1000)
_ = plt.figure(figsize=(14,6))
_ = sns.kdeplot(bs_reps_pov_high_electric_depriv, shade=True)
_ = sns.kdeplot(bs_reps_pov_low_electric_depriv, shade=True)
_ = plt.title('Bootstrapped Poverty Distributions')
_ = plt.legend(['High Electricity Deprivation','Low Electricity Deprivation'])
print('High Electricity Deprivation Bootstrap Mean: ', np.mean(bs_reps_pov_high_electric_depriv))
print('High Electricity Deprivation (95% Conf. Int): ', np.percentile(bs_reps_pov_high_electric_depriv, [2.5,97.5]))
print('')
print('Low Electricity Deprivation Bootstrap Mean: ', np.mean(bs_reps_pov_low_electric_depriv))
print('Low Electricity Deprivation (95% Conf. Int): ', np.percentile(bs_reps_pov_low_electric_depriv, [2.5,97.5]))
true_pov_diff_of_means = np.mean(no_electricity_depriv.total_poverty_intensity) - np.mean(high_electricity_depriv.total_poverty_intensity)
print("True Difference Of Means: ", true_pov_diff_of_means)
pov_density_perms = draw_perm_reps(no_electricity_depriv.total_poverty_intensity, high_electricity_depriv.total_poverty_intensity, diff_of_means, 10000)
_ = plt.figure(figsize=(14,6))
_ = plt.hist(pov_density_perms, bins=50)
_ = plt.axvline(np.mean(pov_density_perms), color='red')
_ = plt.axvline(true_pov_diff_of_means, color='blue')
_ = plt.title('Difference in Mean Population Density Between High and Low Electricity Deprived DHSs')
_ = plt.legend(('Sample Diff of Means','True Diff of Means','Difference Of Means'))
_ = plt.xlabel('Difference of Means')
print('P-Value: ', np.sum(pov_density_perms < true_pov_diff_of_means) / len(pov_density_perms))
x_pop_size_high, y_pop_size_high = ecdf(high_electricity_depriv.All_Population_Count_2015)
x_pop_size_no, y_pop_size_no = ecdf(no_electricity_depriv.All_Population_Count_2015)

_ = plt.figure(figsize=(14,6))
_ = plt.plot(x_pop_size_high, y_pop_size_high, linestyle='none', marker='.')
_ = plt.plot(x_pop_size_no, y_pop_size_no, linestyle='none', marker='.')
_ = plt.title('Popultion Size by Electricity Deprivation')
_ = plt.legend(['High Deprivation','No Deprivation'])
_ = plt.xlabel('Population Size')
_ = plt.ylabel('ECDF')
#Bootstrap Density Distributions of Data
bs_reps_pop_size_high_electric_depriv = draw_bs_reps(high_electricity_depriv.All_Population_Count_2015, np.mean, 1000)
bs_reps_pop_size_low_electric_depriv = draw_bs_reps(no_electricity_depriv.All_Population_Count_2015, np.mean, 1000)
_ = plt.figure(figsize=(14,6))
_ = sns.kdeplot(bs_reps_pop_size_high_electric_depriv, shade=True)
_ = sns.kdeplot(bs_reps_pop_size_low_electric_depriv, shade=True)
_ = plt.title('Bootstrapped Population Size Distributions')
_ = plt.legend(['High Electricity Deprivation','Low Electricity Deprivation'])
print('High Electricity Deprivation Bootstrap Mean: ', np.mean(bs_reps_pop_size_high_electric_depriv))
print('High Electricity Deprivation (95% Conf. Int): ', np.percentile(bs_reps_pop_size_high_electric_depriv, [2.5,97.5]))
print('')
print('Low Electricity Deprivation Bootstrap Mean: ', np.mean(bs_reps_pop_size_low_electric_depriv))
print('Low Electricity Deprivation (95% Conf. Int): ', np.percentile(bs_reps_pop_size_low_electric_depriv, [2.5,97.5]))
true_pop_size_diff_of_means = np.mean(no_electricity_depriv.All_Population_Count_2015) - np.mean(high_electricity_depriv.All_Population_Count_2015)
print("True Difference Of Means: ", true_pop_size_diff_of_means)
pop_size_perms = draw_perm_reps(no_electricity_depriv.All_Population_Count_2015, high_electricity_depriv.All_Population_Count_2015, diff_of_means, 10000)
_ = plt.figure(figsize=(14,6))
_ = plt.hist(pop_size_perms, bins=50)
_ = plt.axvline(np.mean(pop_size_perms), color='red')
_ = plt.axvline(true_pop_size_diff_of_means, color='blue')
_ = plt.title('Difference in Mean Population Density Between High and Low Electricity Deprived DHSs')
_ = plt.legend(('Sample Diff of Means','True Diff of Means','Difference Of Means'))
_ = plt.xlabel('Difference of Means')
print('P-Value: ', np.sum(pop_size_perms < true_pop_size_diff_of_means) / len(pop_size_perms))
_ = plt.figure(figsize=(14,6))
_ = sns.kdeplot(zimbabwe_dhs.cooking_fuel_depriv, shade=True)
_ = plt.axvline(np.mean(zimbabwe_dhs.cooking_fuel_depriv), color='red')
_ = plt.title('Cooking Fuel Deprivation Distribution by DHS')
_ = plt.xlabel('Percent Deprived')
high_cooking_fuel_depriv = zimbabwe_dhs[zimbabwe_dhs.cooking_fuel_depriv == 1]
low_cooking_fuel_depriv = zimbabwe_dhs[zimbabwe_dhs.cooking_fuel_depriv < 0.2]
high_cooking_fuel_depriv.shape
low_cooking_fuel_depriv.shape
def get_change_cooking(row):
    current = row.low_cooking_ave
    previous = row.high_cooking_ave
    if current == previous:
        return 100.0
    try:
        return (abs(current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return 0
#Average All Variables of DHSs with No Electricity Deprivation
low_cooking_depriv_averages = pd.DataFrame(low_cooking_fuel_depriv.mean()).reset_index()
low_cooking_depriv_averages.columns = ['variable','low_cooking_ave']

#Average All Variables of DHSs with High Electricity Deprivation
high_cooking_depriv_averages = pd.DataFrame(high_cooking_fuel_depriv.mean()).reset_index()
high_cooking_depriv_averages.columns = ['variable','high_cooking_ave']

#Merge Tables for comparison
cooking_depriv_averages = low_cooking_depriv_averages.merge(high_cooking_depriv_averages)

#Calculate Difference of Variables
cooking_depriv_averages['mean_differences'] = abs(cooking_depriv_averages['low_cooking_ave'] - cooking_depriv_averages['high_cooking_ave'])
cooking_depriv_averages['perc_change'] = cooking_depriv_averages.apply(get_change_cooking, axis=1)
cooking_depriv_averages[cooking_depriv_averages.variable.isin(deprivation) == False].sort_values('perc_change', ascending=False).head(15)
x_pop_cooking_high, y_pop_cooking_high = ecdf(high_cooking_fuel_depriv.All_Population_Density_2015)
x_pop_cooking_no, y_pop_cooking_no = ecdf(low_cooking_fuel_depriv.All_Population_Density_2015)

_ = plt.figure(figsize=(14,6))
_ = plt.plot(x_pop_cooking_high, y_pop_cooking_high, linestyle='none', marker='.')
_ = plt.plot(x_pop_cooking_no, y_pop_cooking_no, linestyle='none', marker='.')
_ = plt.title('Population Density by Cooking Fuel Deprivation')
_ = plt.legend(['High Cooking Fuel Deprevation','Low Cooking Fuel Deprevation'])
_ = plt.xlabel('Population Density')
_ = plt.ylabel('ECDF')
#Bootstrap Density Distributions of Data
bs_reps_pop_high_cooking_depriv = draw_bs_reps(high_cooking_fuel_depriv.All_Population_Density_2015, np.mean, 1000)
bs_reps_pop_low_cooking_depriv = draw_bs_reps(low_cooking_fuel_depriv.All_Population_Density_2015, np.mean, 1000)
_ = plt.figure(figsize=(14,6))
_ = sns.kdeplot(bs_reps_pop_high_cooking_depriv, shade=True)
_ = sns.kdeplot(bs_reps_pop_low_cooking_depriv, shade=True)
_ = plt.title('Bootstrapped Population Density')
_ = plt.legend(['High Cooking Fuel Deprivation','Low Cooking Fuel Deprivation'])
print('High Cooking Fuel Deprivation Bootstrap Mean: ', np.mean(bs_reps_pop_high_cooking_depriv))
print('High Cooking Fuel Deprivation (95% Conf. Int): ', np.percentile(bs_reps_pop_high_cooking_depriv, [2.5,97.5]))
print('')
print('Low Cooking Fuel Deprivation Bootstrap Mean: ', np.mean(bs_reps_pop_low_cooking_depriv))
print('Low Cooking Fuel Deprivation (95% Conf. Int): ', np.percentile(bs_reps_pop_low_cooking_depriv, [2.5,97.5]))
true_pop_density_cooking_diff_of_means = np.mean(low_cooking_fuel_depriv.All_Population_Density_2015) - np.mean(high_cooking_fuel_depriv.All_Population_Density_2015)
print("True Difference Of Means: ", true_pop_density_cooking_diff_of_means)
pop_density_cooking_perms = draw_perm_reps(low_cooking_fuel_depriv.All_Population_Density_2015, high_cooking_fuel_depriv.All_Population_Density_2015, diff_of_means, 10000)
_ = plt.figure(figsize=(14,6))
_ = plt.hist(pop_density_cooking_perms, bins=50)
_ = plt.axvline(np.mean(pop_density_cooking_perms), color='red')
_ = plt.axvline(true_pop_density_cooking_diff_of_means, color='blue')
_ = plt.title('Difference in Mean Population Density Between High and Low Cooking Fuel Deprived DHSs')
_ = plt.legend(('Sample Diff of Means','True Diff of Means','Difference Of Means'))
_ = plt.xlabel('Difference of Means')
print('P-Value: ', np.sum(pop_density_cooking_perms > true_pop_density_cooking_diff_of_means) / len(pop_density_cooking_perms))
x_pov_cooking_high, y_pov_cooking_high = ecdf(high_cooking_fuel_depriv.total_poverty_intensity)
x_pov_cooking_no, y_pov_cooking_no = ecdf(low_cooking_fuel_depriv.total_poverty_intensity)

_ = plt.figure(figsize=(14,6))
_ = plt.plot(x_pov_cooking_high, y_pov_cooking_high, linestyle='none', marker='.')
_ = plt.plot(x_pov_cooking_no, y_pov_cooking_no, linestyle='none', marker='.')
_ = plt.title('Poverty by Cooking Fuel Deprivation')
_ = plt.legend(['High Deprivation','No Deprivation'])
_ = plt.xlabel('Poverty Intensity')
_ = plt.ylabel('ECDF')
#Bootstrap Density Distributions of Data
bs_reps_pov_high_cooking_depriv = draw_bs_reps(high_cooking_fuel_depriv.total_poverty_intensity, np.mean, 1000)
bs_reps_pov_low_cookingc_depriv = draw_bs_reps(low_cooking_fuel_depriv.total_poverty_intensity, np.mean, 1000)
_ = plt.figure(figsize=(14,6))
_ = sns.kdeplot(bs_reps_pov_high_cooking_depriv, shade=True)
_ = sns.kdeplot(bs_reps_pov_low_cookingc_depriv, shade=True)
_ = plt.title('Bootstrapped Poverty Distributions')
_ = plt.legend(['High Cooking Fuel Deprivation','Low Cooking Fuel Deprivation'])
print('High Cooking Fuel Deprivation Bootstrap Mean: ', np.mean(bs_reps_pov_high_cooking_depriv))
print('High Cooking Fuel Deprivation (95% Conf. Int): ', np.percentile(bs_reps_pov_high_cooking_depriv, [2.5,97.5]))
print('')
print('Low Cooking Fuel Deprivation Bootstrap Mean: ', np.mean(bs_reps_pov_low_cookingc_depriv))
print('Low Cooking Fuel Deprivation (95% Conf. Int): ', np.percentile(bs_reps_pov_low_cookingc_depriv, [2.5,97.5]))
true_pov_cooking_diff_of_means = np.mean(low_cooking_fuel_depriv.total_poverty_intensity) - np.mean(high_cooking_fuel_depriv.total_poverty_intensity)
print("True Difference Of Means: ", true_pov_cooking_diff_of_means)
pov_cooking_density_perms = draw_perm_reps(low_cooking_fuel_depriv.total_poverty_intensity, high_cooking_fuel_depriv.total_poverty_intensity, diff_of_means, 10000)
_ = plt.figure(figsize=(14,6))
_ = plt.hist(pov_cooking_density_perms, bins=50)
_ = plt.axvline(np.mean(pov_cooking_density_perms), color='red')
_ = plt.axvline(true_pov_cooking_diff_of_means, color='blue')
_ = plt.title('Difference in Mean Poverty Intensity Between High and Low Cooking Fuel Deprived DHSs')
_ = plt.legend(('Sample Diff of Means','True Diff of Means','Difference Of Means'))
_ = plt.xlabel('Difference of Means')
print('P-Value: ', np.sum(pov_cooking_density_perms < true_pov_cooking_diff_of_means) / len(pov_cooking_density_perms))
x_pop_size_cooking_high, y_pop_size_cooking_high = ecdf(high_cooking_fuel_depriv.All_Population_Count_2015)
x_pop_size_cooking_no, y_pop_size_cooking_no = ecdf(low_cooking_fuel_depriv.All_Population_Count_2015)

_ = plt.figure(figsize=(14,6))
_ = plt.plot(x_pop_size_cooking_high, y_pop_size_cooking_high, linestyle='none', marker='.')
_ = plt.plot(x_pop_size_cooking_no, y_pop_size_cooking_no, linestyle='none', marker='.')
_ = plt.title('Popultion Size by Cooking Fuel Deprivation')
_ = plt.legend(['High Deprivation','No Deprivation'])
_ = plt.xlabel('Population Size')
_ = plt.ylabel('ECDF')
#Bootstrap Density Distributions of Data
bs_reps_pop_size_high_cooking_depriv = draw_bs_reps(high_cooking_fuel_depriv.All_Population_Count_2015, np.mean, 1000)
bs_reps_pop_size_low_cooking_depriv = draw_bs_reps(low_cooking_fuel_depriv.All_Population_Count_2015, np.mean, 1000)
_ = plt.figure(figsize=(14,6))
_ = sns.kdeplot(bs_reps_pop_size_high_cooking_depriv, shade=True)
_ = sns.kdeplot(bs_reps_pop_size_low_cooking_depriv, shade=True)
_ = plt.title('Bootstrapped Population Size Distributions')
_ = plt.legend(('High Cooking Fuel Deprivation','Low Cooking Fuel Deprivation'))
print('High Electricity Deprivation Bootstrap Mean: ', np.mean(bs_reps_pop_size_high_cooking_depriv))
print('High Electricity Deprivation (95% Conf. Int): ', np.percentile(bs_reps_pop_size_high_cooking_depriv, [2.5,97.5]))
print('')
print('Low Electricity Deprivation Bootstrap Mean: ', np.mean(bs_reps_pop_size_low_cooking_depriv))
print('Low Electricity Deprivation (95% Conf. Int): ', np.percentile(bs_reps_pop_size_low_cooking_depriv, [2.5,97.5]))
true_pop_size_cooking_diff_of_means = np.mean(low_cooking_fuel_depriv.All_Population_Count_2015) - np.mean(high_cooking_fuel_depriv.All_Population_Count_2015)
print("True Difference Of Means: ", true_pop_size_cooking_diff_of_means)
pop_size_cooking_perms = draw_perm_reps(low_cooking_fuel_depriv.All_Population_Count_2015, high_cooking_fuel_depriv.All_Population_Count_2015, diff_of_means, 10000)
_ = plt.figure(figsize=(14,6))
_ = plt.hist(pop_size_cooking_perms, bins=50)
_ = plt.axvline(np.mean(pop_size_cooking_perms), color='red')
_ = plt.axvline(true_pop_size_cooking_diff_of_means, color='blue')
_ = plt.title('Difference in Mean Population Density Between High and Low Electricity Deprived DHSs')
_ = plt.legend(('Sample Diff of Means','True Diff of Means','Difference Of Means'))
_ = plt.xlabel('Difference of Means')
print('P-Value: ', np.sum(pop_size_cooking_perms > true_pop_size_cooking_diff_of_means) / len(pop_size_cooking_perms))




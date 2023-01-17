import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn
import matplotlib as mpl
import matplotlib.pylab as plt
plt.style.use('ggplot')
import os
print('Data Files....', os.listdir("../input/cpe-data/"))
%matplotlib inline
ACSdf = pd.read_csv('../input/cpe-data/ACS_variable_descriptions.csv')
print(ACSdf.shape)
for row in ACSdf.iterrows():
    print(row[1][0], row[1][1])
    if row[0] == 20:
        break
dept11_education = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_education-attainment-over-25/ACS_16_5YR_B15003_with_ann.csv')
pd.to_numeric(dept11_education['HD01_VD01'].drop(0)).plot(kind='hist', figsize=(15, 5), bins=40, title='Dept 11: Population Distribution per Census Tract')
dept11_education = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_education-attainment-over-25/ACS_16_5YR_B15003_with_ann.csv')
dept11_education_wth_labels = dept11_education.copy()
dept11_education = dept11_education_wth_labels.drop(0)
# Make Columns numeric
for col in dept11_education.columns:
    if col[:2] == 'HD':
        dept11_education[col] = pd.to_numeric(dept11_education[col])
dept11_education_wth_labels.head()
# Make Percentages
# TODO Combine into larger bins like Elementry school, Middle School... etc
dept11_education['PCT_No_schooling'] = dept11_education['HD01_VD02'] / dept11_education['HD01_VD01']
dept11_education['PCT_Nursery_school'] = dept11_education['HD01_VD03'] / dept11_education['HD01_VD01']
dept11_education['PCT_Kindergarten'] = dept11_education['HD01_VD04'] / dept11_education['HD01_VD01']
dept11_education['PCT_1st_grade'] = dept11_education['HD02_VD05'] / dept11_education['HD01_VD01']
dept11_education['PCT_2nd_grade'] = dept11_education['HD01_VD06'] / dept11_education['HD01_VD01']
dept11_education['PCT_3nd_grade'] = dept11_education['HD01_VD07'] / dept11_education['HD01_VD01']
dept11_education['PCT_4th_grade'] = dept11_education['HD01_VD08'] / dept11_education['HD01_VD01']
dept11_education['PCT_5th_grade'] = dept11_education['HD01_VD09'] / dept11_education['HD01_VD01']
dept11_education['PCT_6th_grade'] = dept11_education['HD01_VD10'] / dept11_education['HD01_VD01']
dept11_education['PCT_7th_grade'] = dept11_education['HD01_VD11'] / dept11_education['HD01_VD01']
dept11_education['PCT_8th_grade'] = dept11_education['HD01_VD12'] / dept11_education['HD01_VD01']
dept11_education['PCT_9th_grade'] = dept11_education['HD01_VD13'] / dept11_education['HD01_VD01']
dept11_education['PCT_10th_grade'] = dept11_education['HD01_VD14'] / dept11_education['HD01_VD01']
dept11_education['PCT_11th_grade'] = dept11_education['HD01_VD15'] / dept11_education['HD01_VD01']
dept11_education['PCT_12th_grade'] = dept11_education['HD01_VD16'] / dept11_education['HD01_VD01']
dept11_education['PCT_high_school_diploma'] = dept11_education['HD01_VD17'] / dept11_education['HD01_VD01']
dept11_education['PCT_GED or alternative credential'] = dept11_education['HD01_VD18'] / dept11_education['HD01_VD01']
dept11_education['PCT_college_less than 1 year'] = dept11_education['HD01_VD19'] / dept11_education['HD01_VD01']
dept11_education['PCT_college 1 or more years'] = dept11_education['HD01_VD20'] / dept11_education['HD01_VD01']
dept11_education['PCT_Associates degree'] = dept11_education['HD01_VD21'] / dept11_education['HD01_VD01']
dept11_education['PCT_Bachelors degree'] = dept11_education['HD01_VD22'] / dept11_education['HD01_VD01']
dept11_education['PCT_Masters degree'] = dept11_education['HD01_VD23'] / dept11_education['HD01_VD01']
dept11_education['PCT_Professional school degree'] = dept11_education['HD01_VD24'] / dept11_education['HD01_VD01']
dept11_education['PCT_Doctorate degree'] = dept11_education['HD01_VD25'] / dept11_education['HD01_VD01']
# Plot Distributions.
# TODO think of a more interesting way to display this data. group into levels. Cumulative?
for col in dept11_education.columns:
    color_loc = 0
    if col[:3] == 'PCT':
        dept11_education[col].plot(kind='hist',
                                   figsize=(15, 2),
                                   bins=int(round(dept11_education[col].max()*100)),
                                   title=col,
                                   xlim=(0,1))
        plt.show()
dept11_education['PCT_Some_Elementary'] = dept11_education['PCT_Kindergarten'] + \
                                          dept11_education['PCT_1st_grade'] + \
                                          dept11_education['PCT_2nd_grade'] + \
                                          dept11_education['PCT_3nd_grade'] + \
                                          dept11_education['PCT_4th_grade'] + \
                                          dept11_education['PCT_5th_grade']

dept11_education['PCT_Some_Middle_School'] = dept11_education['PCT_7th_grade'] + \
                                          dept11_education['PCT_8th_grade'] + \
                                          dept11_education['PCT_9th_grade'] + \
                                          dept11_education['PCT_10th_grade'] + \
                                          dept11_education['PCT_11th_grade']
dept11_education['PCT_High_School'] = dept11_education['PCT_12th_grade']
dept11_education['PCT_Some_College'] = dept11_education['PCT_college_less than 1 year'] + dept11_education['PCT_college 1 or more years']
dept11_education
dept11_education.set_index('GEO.display-label')[['PCT_No_schooling','PCT_Some_Elementary',
                  'PCT_Some_Middle_School',
                  'PCT_High_School',
                  'PCT_Some_College',
                  'PCT_Associates degree',
                  'PCT_Bachelors degree',
                  'PCT_Masters degree',
                  'PCT_Professional school degree',
                  'PCT_Doctorate degree']].plot(kind='barh', stacked=True, figsize=(15,45))
dept11_housing_metadata = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_owner-occupied-housing/ACS_16_5YR_S2502_metadata.csv')
dept11_housing = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_owner-occupied-housing/ACS_16_5YR_S2502_with_ann.csv')
dept11_housing_metadata
dept11_housing['HC01_EST_VC01'][0]
pd.to_numeric(dept11_housing['HC01_EST_VC01'].drop(0)).plot(kind='hist',
                                                            bins=25,
                                                            title='Occupied housing units; Estimate; Occupied housing units',
                                                            figsize=(15 ,5),
                                                            color='g')
dep11_poverty_metadata = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_poverty/ACS_16_5YR_S1701_metadata.csv')
dep11_poverty = pd.read_csv('../input/cpe-data/Dept_11-00091/11-00091_ACS_data/11-00091_ACS_poverty/ACS_16_5YR_S1701_with_ann.csv')
dep11_poverty_metadata.head()
dep11_poverty_metadata.head()
pd.to_numeric(dep11_poverty['HC01_EST_VC01'].drop(0)).plot(kind='hist',
                                                            bins=25,
                                                            title=dep11_poverty['HC01_EST_VC01'][0],
                                                            figsize=(15 ,5),
                                                            color='y')
pd.to_numeric(dep11_poverty['HC03_EST_VC01'].replace('-',0).drop(0)).plot(kind='hist',
                                                            bins=25,
                                                            title=dep11_poverty['HC03_EST_VC01'][0],
                                                            figsize=(15 ,5),
                                                            color='m')
pd.to_numeric(dep11_poverty['HC03_EST_VC01'].replace('-',0).drop(0)).max()
dep11_poverty.loc[dep11_poverty['HC03_EST_VC01'] == '58.3']['GEO.display-label'].values[0]

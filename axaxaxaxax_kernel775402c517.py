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
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import numpy as np

dtypes = {
 'Date First Observed': np.str,
 'Days Parking In Effect    ': np.str,
 'Double Parking Violation': np.str,
 'Feet From Curb': np.float32,
 'From Hours In Effect': np.str,
 'House Number': np.str,
 'Hydrant Violation': np.str,
 'Intersecting Street': np.str,
 'Issue Date': np.str,
 'Issuer Code': np.float32,
 'Issuer Command': np.str,
 'Issuer Precinct': np.float32,
 'Issuer Squad': np.str,
 'Issuing Agency': np.str,
 'Law Section': np.float32,
 'Meter Number': np.str,
 'No Standing or Stopping Violation': np.str,
 'Plate ID': np.str,
 'Plate Type': np.str,
 'Registration State': np.str,
 'Street Code1': np.uint32,
 'Street Code2': np.uint32,
 'Street Code3': np.uint32,
 'Street Name': np.str,
 'Sub Division': np.str,
 'Summons Number': np.uint32,
 'Time First Observed': np.str,
 'To Hours In Effect': np.str,
 'Unregistered Vehicle?': np.str,
 'Vehicle Body Type': np.str,
 'Vehicle Color': np.str,
 'Vehicle Expiration Date': np.str,
 'Vehicle Make': np.str,
 'Vehicle Year': np.float32,
 'Violation Code': np.uint16,
 'Violation County': np.str,
 'Violation Description': np.str,
 'Violation In Front Of Or Opposite': np.str,
 'Violation Legal Code': np.str,
 'Violation Location': np.str,
 'Violation Post Code': np.str,
 'Violation Precinct': np.float32,
 'Violation Time': np.str
}

nyc_data_raw = dd.read_csv('/kaggle/input/nyc-parking-tickets/*.csv', dtype=dtypes, usecols=dtypes.keys())

# Listing 5.2
with ProgressBar():
    display(nyc_data_raw['Plate ID'].head())

# Listing 5.3
with ProgressBar():
    display(nyc_data_raw[['Plate ID', 'Registration State']].head())
# Listing 5.4
columns_to_select = ['Plate ID', 'Registration State']

with ProgressBar():
    display(nyc_data_raw[columns_to_select].head())
# Listing 5.5
with ProgressBar():
    display(nyc_data_raw.drop('Violation Code', axis=1).head())
# Listing 5.6
violationColumnNames = list(filter(lambda columnName: 'Violation' in columnName, nyc_data_raw.columns))

with ProgressBar():
    display(nyc_data_raw.drop(violationColumnNames, axis=1).head())
# Listing 5.7
nyc_data_renamed = nyc_data_raw.rename(columns={'Plate ID':'License Plate'})
nyc_data_renamed
# Listing 5.8
with ProgressBar():
    display(nyc_data_raw.loc[56].head(1))
# Listing 5.9
with ProgressBar():
    display(nyc_data_raw.loc[100:200].head(100))
# Listing 5.10
with ProgressBar():
    some_rows = nyc_data_raw.loc[100:200].head(100)
some_rows.drop(range(100, 200, 2))
# Listing 5.11
missing_values = nyc_data_raw.isnull().sum()
with ProgressBar():
    percent_missing = ((missing_values / nyc_data_raw.index.size) * 100).compute()
percent_missing
# Listing 5.12
columns_to_drop = list(percent_missing[percent_missing >= 50].index)
nyc_data_clean_stage1 = nyc_data_raw.drop(columns_to_drop, axis=1)
# Listing 5.13
with ProgressBar():
    count_of_vehicle_colors = nyc_data_clean_stage1['Vehicle Color'].value_counts().compute()
most_common_color = count_of_vehicle_colors.sort_values(ascending=False).index[0]

# Fill missing vehicle color with the most common color
nyc_data_clean_stage2 = nyc_data_clean_stage1.fillna({'Vehicle Color': most_common_color})
# Listing 5.14

# Updated to compensate for bug identified in https://github.com/dask/dask/issues/5854

# Old code:
# rows_to_drop = list(percent_missing[(percent_missing > 0) & (percent_missing < 5)].index)
# nyc_data_clean_stage3 = nyc_data_clean_stage2.dropna(subset=rows_to_drop)

# New code splits the rows to drop into two separate lists and chains the dropna methods to drop all the columns we want
rows_to_drop1 =['Plate ID', 'Vehicle Body Type', 'Vehicle Make', 'Vehicle Expiration Date', 'Violation Precinct', 'Issuer Precinct', 'Issuer Code', 'Violation Time', 'Street Name']
rows_to_drop2 =['Date First Observed', 'Law Section', 'Sub Division', 'Vehicle Color', 'Vehicle Year', 'Feet From Curb']
nyc_data_clean_stage3 = nyc_data_clean_stage2.dropna(subset=rows_to_drop1).dropna(subset=rows_to_drop2)
# Listing 5.15
remaining_columns_to_clean = list(percent_missing[(percent_missing >= 5) & (percent_missing < 50)].index)
nyc_data_raw.dtypes[remaining_columns_to_clean]
# Listing 5.16
unknown_default_dict = dict(map(lambda columnName: (columnName, 'Unknown'), remaining_columns_to_clean))
# Listing 5.17
nyc_data_clean_stage4 = nyc_data_clean_stage3.fillna(unknown_default_dict)
# Listing 5.18
with ProgressBar():
    print(nyc_data_clean_stage4.isnull().sum().compute())
    nyc_data_clean_stage4.persist()
# Listing 5.19
with ProgressBar():
    license_plate_types = nyc_data_clean_stage4['Plate Type'].value_counts().compute()
license_plate_types
# Listing 5.20
condition = nyc_data_clean_stage4['Plate Type'].isin(['PAS', 'COM'])
plate_type_masked = nyc_data_clean_stage4['Plate Type'].where(condition, 'Other')
nyc_data_recode_stage1 = nyc_data_clean_stage4.drop('Plate Type', axis=1)
nyc_data_recode_stage2 = nyc_data_recode_stage1.assign(PlateType=plate_type_masked)
nyc_data_recode_stage3 = nyc_data_recode_stage2.rename(columns={'PlateType':'Plate Type'})
# Listing 5.21
with ProgressBar():
    display(nyc_data_recode_stage3['Plate Type'].value_counts().compute())
# Listing 5.22
single_color = list(count_of_vehicle_colors[count_of_vehicle_colors == 1].index)
condition = nyc_data_clean_stage4['Vehicle Color'].isin(single_color)
vehicle_color_masked = nyc_data_clean_stage4['Vehicle Color'].mask(condition, 'Other')
nyc_data_recode_stage4 = nyc_data_recode_stage3.drop('Vehicle Color', axis=1)
nyc_data_recode_stage5 = nyc_data_recode_stage4.assign(VehicleColor=vehicle_color_masked)
nyc_data_recode_stage6 = nyc_data_recode_stage5.rename(columns={'VehicleColor':'Vehicle Color'})
# Listing 5.23
from datetime import datetime
issue_date_parsed = nyc_data_recode_stage6['Issue Date'].apply(lambda x: datetime.strptime(x, "%m/%d/%Y"), meta=datetime)
nyc_data_derived_stage1 = nyc_data_recode_stage6.drop('Issue Date', axis=1)
nyc_data_derived_stage2 = nyc_data_derived_stage1.assign(IssueDate=issue_date_parsed)
nyc_data_derived_stage3 = nyc_data_derived_stage2.rename(columns={'IssueDate':'Issue Date'})
# Listing 5.24
with ProgressBar():
    display(nyc_data_derived_stage3['Issue Date'].head())
# Listing 5.25
issue_date_month_year = nyc_data_derived_stage3['Issue Date'].apply(lambda dt: dt.strftime("%Y%m"), meta=str)
nyc_data_derived_stage4 = nyc_data_derived_stage3.assign(IssueMonthYear=issue_date_month_year)
nyc_data_derived_stage5 = nyc_data_derived_stage4.rename(columns={'IssueMonthYear':'Citation Issued Month Year'})
# Listing 5.26
with ProgressBar():
    display(nyc_data_derived_stage5['Citation Issued Month Year'].head())
# Listing 5.27
months = ['201310','201410','201510','201610','201710']
condition = nyc_data_derived_stage5['Citation Issued Month Year'].isin(months)
october_citations = nyc_data_derived_stage5[condition]

with ProgressBar():
    display(october_citations.head())
# Listing 5.28
bound_date = '2016-4-25'
condition = nyc_data_derived_stage5['Issue Date'] > bound_date
citations_after_bound = nyc_data_derived_stage5[condition]

with ProgressBar():
    display(citations_after_bound.head())
# Listing 5.29
with ProgressBar():
    condition = (nyc_data_derived_stage5['Issue Date'] > '2014-01-01') & (nyc_data_derived_stage5['Issue Date'] <= '2017-12-31')
    nyc_data_filtered = nyc_data_derived_stage5[condition]
    nyc_data_new_index = nyc_data_filtered.set_index('Citation Issued Month Year')
# Listing 5.30
years = ['2014', '2015', '2016', '2017']
months = ['01','02','03','04','05','06','07','08','09','10','11','12']
divisions = [year + month for year in years for month in months]

with ProgressBar():
    nyc_data_new_index.repartition(divisions=divisions).to_parquet('nyc_data_date_index', compression='snappy')
    
nyc_data_new_index = dd.read_parquet('nyc_data_date_index')

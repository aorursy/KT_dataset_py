import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
india_raw_data    = pd.read_csv('../input/API_IND_DS2_en_csv_v2_10400058.csv', skiprows=3)
gbr_raw_data      = pd.read_csv('../input/API_GBR_DS2_en_csv_v2_10402095.csv', skiprows=3)
india_metadata    = pd.read_csv('../input/Metadata_Indicator_API_GBR_DS2_en_csv_v2_10402095.csv', skiprows=0)
gbr_metadata      = pd.read_csv('../input/Metadata_Indicator_API_GBR_DS2_en_csv_v2_10402095.csv', skiprows=0)
years             = india_raw_data.columns[5:-1] # Valid columns for years
india_indicators  = india_raw_data['Indicator Name']
gbr_indicators    = gbr_raw_data['Indicator Name']
keyword               = "high-technology exports"

lower_case_indicators = india_indicators.str.lower()
keyword_indicators    = india_indicators[lower_case_indicators.str.contains(keyword.lower(), regex=False)]
print("All Indicators matching keyword:\n\n", keyword_indicators, "\n")

desired_index       = int(input("Choose the desired index (number in left column above): "))
desired_indicator   = india_raw_data["Indicator Name"][desired_index]
print("\n Desired Indicator:\n", desired_indicator)

india_yearly_data   = [float(india_raw_data[year][desired_index]) for year in years]

gbr_desired_index   = gbr_indicators.str.contains(desired_indicator, regex=False)
gbr_yearly_data     = [float(gbr_raw_data[year][gbr_desired_index]) for year in years]

years_to_plot       = np.array([int(year) for year in years])
bar_width           = 0.4
plt.bar(years_to_plot, india_yearly_data, bar_width, label="India")
plt.bar(years_to_plot+bar_width, gbr_yearly_data, bar_width, label="GBR")
plt.title(desired_indicator)
plt.legend(loc="upper left")
plt.grid()

# Metadata
metadata_index      = india_metadata.INDICATOR_NAME.str.contains(desired_indicator, regex=False)
print("Source:\n", list(india_metadata["SOURCE_ORGANIZATION"][metadata_index]), "\n")
print("Note:\n", list(india_metadata["SOURCE_NOTE"][metadata_index]))
pd.options.display.max_rows = 2000
print(india_indicators)


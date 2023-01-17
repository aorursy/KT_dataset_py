import pandas as pd
# Specify the data path of the financials
# Change this if you are using the data on a local machine to the path of
# quarterly_financials.csv
financials_data_path = '/kaggle/input/quarterly_financials.csv'
financials_data = pd.read_csv(financials_data_path, index_col = 0) # Read CSV file of financials data
import matplotlib.pyplot as plt # for plotting graphs
financials_data['filing_date'] = pd.DatetimeIndex(financials_data['filing_date'])
# Convert all filing_dates to pd.Timestamp
AAPL_data = financials_data[financials_data['stock'] == 'AAPL'] # get filed data for Apple
AAPL_data = AAPL_data.sort_values(by = 'filing_date') # sort by filing date
plt.title('AAPL - Assets Current')
plt.plot(AAPL_data['filing_date'], AAPL_data['assetscurrent']) # Plot current assets for AAPL over time
plt.show()

import pandas as pd
# Read file

color_data = pd.read_csv('../input/colors.csv')
# Summarize file

color_data.describe()
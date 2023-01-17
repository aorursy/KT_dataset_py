# Constants 
INPUT_PATH = '/kaggle/input/netflix-shows/netflix_titles.csv'

# Libraries 
import pandas as pd 
import matplotlib.pyplot as plt

# Set default properties for plotting 
plt.rcParams['figure.figsize'] = [11, 4]
plt.rcParams['figure.dpi'] = 100 
# Read data and display 5 random entries 
raw_df = pd.read_csv(INPUT_PATH)

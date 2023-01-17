import pandas as pd
import seaborn as sns
%matplotlib inline
marvel_data = pd.read_csv("https://raw.githubusercontent.com/aniket-spidey/bitgrit-webinar/master/code/datasets/marvel-wikia-data.csv")
dc_data = pd.read_csv("https://raw.githubusercontent.com/aniket-spidey/bitgrit-webinar/master/code/datasets/dc-wikia-data.csv")
marvel_data.head()
marvel_clean = marvel_data.dropna(subset=['name', 'Year'])
dc_clean = dc_data.dropna(subset=['name', 'Year'])
def yearPlot(data):
    dictionary = {}
    # {1942: 2, 1945, 12}
    years = data['Year']
    
    for year in years:
        if year in dictionary:
            dictionary[year] = dictionary[year] + 1
        else:
            dictionary[year] = 1

    # Our dictionary now looks like this -> {1941: 12, 1942: 3, ..... , 2012: 9}
    X = []
    y = []
    
    for key, val in dictionary.items():
        X.append(key)
        y.append(val)
        
    sns.lineplot(X, y)
yearPlot(marvel_clean)
yearPlot(dc_clean)
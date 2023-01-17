# Preparation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Data Import & Cleaning
Distillery = pd.read_csv("../input/world-whisky-distilleries-brands-dataset/Distillery.csv")
Distillery = Distillery.iloc[:, 1:]

Country_Code = pd.read_csv("../input/country-code/country_code.csv")
Country_Code = Country_Code.iloc[:, 1:]

Brand = pd.read_csv("../input/world-whisky-distilleries-brands-dataset/Whisky_Brand.csv")
Brand = Brand.iloc[:, 1:7]
# Data Content
Distillery.head()
Distillery.info()
type(Distillery)
Distillery.shape
Distillery['Rating'] = pd.to_numeric(Distillery['Rating'], errors='coerce')
Distillery = Distillery.dropna(subset=['Rating'])
sns.distplot(Distillery['Rating'], kde = False, bins = 90)
Distillery = Distillery[(Distillery['Rating'] > 0)]
fig, ax =plt.subplots(1,2, figsize=(20,8))
sns.distplot(Distillery['Rating'], kde = True, bins = 90, ax = ax[1])
sns.boxplot(x = Distillery['Rating'], ax = ax[0])
plt.suptitle("Distribution of Rating Score", fontsize = 30)
plt.show()
Country = Distillery['Country'].value_counts()
Country = pd.DataFrame({'Country':Country.index,'Count':(Country.values)})
barplot1 = [go.Bar(
            x=Country['Country'],
            y=Country['Count']
    )]
layout = go.Layout(
    title = 'Distillery Number of Each Country'
)
fig = go.Figure(data = barplot1, layout = layout)
iplot(fig)
NW_Distillery = Distillery[(Distillery['Country'] != "Japan") & (Distillery['Country'] != "Scotland") & 
                                     (Distillery['Country'] != "United States") & (Distillery['Country'] != "Canada") & 
                                     (Distillery['Country'] != "Ireland")]
Country = NW_Distillery['Country'].value_counts()
Country = pd.DataFrame({'Country':Country.index,'Count':(Country.values)})

barplot2 = [go.Bar(
            x=Country['Country'],
            y=Country['Count']
    )]
layout = go.Layout(
    title = 'Distillery Number of Each New World Whisky Country'
)
fig = go.Figure(data = barplot2, layout = layout)
iplot(fig)
Country_Rating = NW_Distillery.groupby("Country").mean()["Rating"]
Country_Rating = Country_Rating.to_frame()
Country_Rating['Country'] = Country_Rating.index
Country_Rating = Country_Rating.sort_values(by = 'Rating', ascending=False)
fig = plt.figure(figsize=(10,10))
sns.barplot(
    data = Country_Rating,
    x = "Rating",
    y = "Country"
)
plt.axvline(60, color = 'r')
plt.title("Average Rating Scores of Whisky Disitilleries for each Country", fontsize = 20)
fig.show()
Top_NW_Country = Country_Rating[Country_Rating['Rating'] > 60]
Top_NW_Country = Top_NW_Country.reset_index(drop = True)
Top_NW_Country = pd.DataFrame(Top_NW_Country, columns = ['Country', 'Rating'])
Top_NW_Country_name = Top_NW_Country['Country']
Top_NW_Brand = Brand.loc[Brand['Country'].isin(['Taiwan','Israel', 'Mexico', 'United Kingdom', 
                                                'Australia', 'South Africa', 'Finland', 'Liechtenstein', 
                                                'Bhutan', 'Sweden', 'Sweden', 'Denmark', 'Switzerland', 
                                                'Czech Republic','Norway', 'Slovakia', 'France', 
                                                'Indonesia', 'Spain', 'Germany', 'Hungary', 'Italy',
                                                'India', 'Austria', 'Netherlands', 'Belgium',
                                                'Luxembourg', 'Iceland', 'Turkey', 'New Zealand', 
                                                'Poland', 'Netherlands Antilles', 'Serbia And Montenegro'])]
Top_NW_Brand.info()
Top_NW_Brand = Top_NW_Brand[np.isfinite(Top_NW_Brand['Rating'])]
Top_NW_Brand = Top_NW_Brand[Top_NW_Brand['Votes'] > 10]
fig = plt.figure(figsize = (10,6))
sns.boxplot(
    data = Top_NW_Brand,
    x = 'WB Ranking',
    y = 'Rating'
)
plt.axhline(80, color = 'red')
plt.suptitle("Rating Band", fontsize = 20)
fig.show()
Top_NW_Brand = Top_NW_Brand[Top_NW_Brand['Rating'] > 80].sort_values(by = ['Rating'], ascending = False)
fig = plt.figure(figsize = (10,10))
sns.barplot(x="Rating", y="Brand", hue="Country",
                 data=Top_NW_Brand, dodge=False)
plt.title("Top New World Whisky Brands", fontsize = 20)
Top_NW_Brand.head(5)
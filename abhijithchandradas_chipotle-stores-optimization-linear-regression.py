import pandas as pd 

import folium 

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



from sklearn.linear_model import LinearRegression



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Read Data

df = pd.read_csv('/kaggle/input/chipotle-locations/chipotle_stores.csv')

df_gdp=pd.read_csv('/kaggle/input/us_energy_census_gdp_10-14/Energy Census and Economic Data US 2010-2014.csv')

df.head()
usa_map = folium.Map([39.358, -98.118], zoom_start=4, tiles="Stamen toner")

for lat, lon,loc in zip(df.latitude, df.longitude,df.location):

    folium.CircleMarker([lat, lon],radius=10, color=None,

                        fill_color='red',fill_opacity=0.3,

                        tooltip="Location : "+str(loc)).add_to(usa_map)

usa_map
total=len(df.state.unique())

print(f'Total Number of states with chipotle outlet : {total}')
#Statewise distribution of Outlets

state_chip=df.state.value_counts()

df_state_chip=pd.DataFrame({'state':state_chip.index,

                           'outlets':state_chip.values})



plt.figure(figsize=(10,6))

plt.title("Number of Chipotle Stores by State")

sns.barplot('state', 'outlets', data=df_state_chip)

plt.xticks(rotation=90)

plt.show()
df_gdp.columns
#Selecting only the required columns

df_gdp=df_gdp[['State', 'GDP2014', 'POPESTIMATE2014']]

df_gdp.head()
#States in GDP data only

set(df_gdp.State)-set(df_state_chip.state)
#States in Chipotle data only

set(df_state_chip.state)-set(df_gdp.State)
#Removing states not required and renaming the columns

df_gdp=df_gdp[df_gdp.State.isin(['Alaska', 'Hawaii', 'United States'])==False]

df_gdp.State[df_gdp.State=='District of Columbia']='Washington DC'

df_gdp=df_gdp.rename(columns={'State':'state','GDP2014':'gdp','POPESTIMATE2014':'popl'})

df_gdp.head()
#Adding South Dakota to State list for Chipotle Stores

df_state_chip=df_state_chip.append(pd.Series({'state':'South Dakota','outlets':0}),ignore_index=True)
# Joining the stores and stats dataframes

df_merged=pd.merge(df_state_chip,df_gdp, on='state')

df_merged.head()
#Calculating GDP per capita

df_merged["gdp_pc"]=round(df_merged.gdp/df_merged.popl,2)

df_merged.head()
sns.heatmap(df_merged.corr(), annot=True, cmap='coolwarm')

plt.show()
plt.title("GDP vs Outlets")

sns.regplot('outlets','gdp', data=df_merged)

plt.show()
plt.title("Population vs Outlets")

sns.regplot('outlets','popl', data=df_merged)

plt.show()
plt.title("GDP per capita vs Outlets")

sns.regplot('outlets','gdp_pc', data=df_merged)

plt.show()
plt.title("GDP vs Population vs Outlets")

sns.scatterplot('popl','gdp',size='outlets', alpha=0.8, data=df_merged)

plt.show()
#Select Features and target

X=df_merged[['gdp','popl']]

y=df_merged.outlets
#Fitting Linear Regression

model=LinearRegression()

model.fit(X,y)

print(f"R2 score : {round(model.score(X,y)*100,2)}%")
#Predicting the number of Outlets

y_pred=model.predict(X)

y_pred=np.round(y_pred,0)

y_pred
#Adding Predicted Outlets to main dataframe

df_merged['outlets_pred']=y_pred

df_merged.outlets_pred=df_merged.outlets_pred.astype('int')

df_merged.head()
plt.figure(figsize=(10,8))

plt.title("Overserved and Underserved Markets")

plot=sns.scatterplot('outlets_pred','outlets',data=df_merged)

for i in range(0, df_merged.shape[0]):

    plot.text(df_merged.outlets_pred[i], df_merged.outlets[i], df_merged.state[i], alpha=0.8, fontsize=8 )

plt.plot([-50,500],[-50,500],'r--')

plt.xlim(-10,max(df_merged.outlets_pred)+20)

plt.ylim(-10,max(df_merged.outlets)+20)

plt.show()
#Calculating the number of outlets in excess/short of the predicted number of outlets

df_merged['scope']=df_merged.outlets_pred-df_merged.outlets

plt.figure(figsize=(4,8))

plt.title("Number of outlets in excess/short of the predicted number of outlets")

sns.barplot(data=df_merged.sort_values(by='scope', ascending=False),

           x='scope',y='state', orient='h')

plt.show()

#df_merged.sort_values(by='scope', ascending=False)
# Percentage increase of decrease in number of stores per state

df_merged['scope_perc']=round(df_merged.scope*100/df_merged.outlets,2)

df_merged.sort_values(by='scope_perc', ascending=False, inplace=True)



plt.figure(figsize=(6,10))

sns.barplot(y=df_merged.state, x=df_merged.scope_perc, orient='h')

plt.show()

#df_merged
# Taking only states where more than 5 stores should be opened or closed and scope perc greater than +/-20%
df_final=df_merged[(abs(df_merged.scope_perc)>50)& (abs(df_merged.scope)>4)].reset_index()

df_final.head(10)
df_final.tail(10)
df_final['perc_change']=abs(df_final.scope_perc)

df_final['density']=df_final.scope.apply(lambda x: "Underserved" if x>0 else "Overserved")

df_final
plt.figure(figsize=(10,6))

plt.title("Overserved and Underserved Markets")

plot=sns.scatterplot('outlets_pred','outlets',data=df_final,

                     size='perc_change', size_norm=(50,200),

                     hue='density')

for i in range(0, df_final.shape[0]):

    plot.text(df_final.outlets_pred[i], df_final.outlets[i], df_final.state[i], alpha=0.8, fontsize=8 )

plt.plot([-50,500],[-50,500],'r--')

plt.xlim(0,max(df_final.outlets_pred)+2)

plt.ylim(0,max(df_final.outlets)+2)

plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Any results you write to the current directory are saved as output.
# Import .csv file
foodFactData = pd.read_csv('../input/FoodFacts.csv',low_memory=False);
print(foodFactData.shape);
# Now plot the frequencies of the country in dataset
plt.figure(figsize=(12,8))
foodFactData.countries.value_counts(normalize=True).head(10).plot(kind='bar')
foodFactData.countries=foodFactData.countries.str.lower()
foodFactData.loc[foodFactData['countries'] == 'en:fr', 'countries'] = 'france'
foodFactData.loc[foodFactData['countries'] == 'en:es', 'countries'] = 'spain'
foodFactData.loc[foodFactData['countries'] == 'en:gb', 'countries']='united kingdom'
foodFactData.loc[foodFactData['countries'] == 'en:uk', 'countries']='united kingdom'
foodFactData.loc[foodFactData['countries'] == 'holland','countries']='netherlands'
foodFactData.loc[foodFactData['countries'] == 'españa','countries']='spain'
foodFactData.loc[foodFactData['countries'] == 'us','countries']='united states'
foodFactData.loc[foodFactData['countries'] == 'en:us','countries']='united states'
foodFactData.loc[foodFactData['countries'] == 'usa','countries']='united states'
foodFactData.loc[foodFactData['countries'] == 'en:cn','countries']='canada'
foodFactData.loc[foodFactData['countries'] == 'en:au','countries']='australia'
foodFactData.loc[foodFactData['countries'] == 'en:de','countries']='germany'
foodFactData.loc[foodFactData['countries'] == 'deutschland','countries']='germany'
foodFactData.loc[foodFactData['countries'] == 'en:cn','countries']='china'
foodFactData.loc[foodFactData['countries'] == 'en:be','countries']='belgium'
foodFactData.loc[foodFactData['countries'] == 'en:ch','countries']='switzerland'
# For better visualization purpose , import seaborn library
import seaborn as sns
fig=plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(1,1,1)
# Now plot possibly top five countries in this dataset
foodFactData.countries.value_counts(normalize=True).head().plot(kind='bar')
foodFactData[foodFactData['countries'] == 'france'].countries.value_counts()
top_countries = ['france','united kingdom','spain','germany','united states' ]
fruits_vegetables_nuts=[]
# Loop through all top five countries 
for country in top_countries:
    fruits_vegetables_nuts.append(getattr(foodFactData[foodFactData.countries==country], 'fruits_vegetables_nuts_100g').mean())
# We can combine two lists to form a Data Frame
# countriesDF=pd.DataFrame(dict(country=pd.Series(top_countries),mean=pd.Series(fruits_vegetables_nuts)))
fig=plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(1,1,1)
y_pos = np.arange(len(top_countries))
    
plt.bar(y_pos,fruits_vegetables_nuts, align='center')
plt.title('Average total fruits_vegetables_nuts content per 100g')
plt.xticks(y_pos, top_countries)
plt.ylabel('fruits_vegetables_nuts/100g') 
plt.show()
alcohol=[]
# Loop through all top five countries 
for country in top_countries:
    alcohol.append(getattr(foodFactData[foodFactData.countries==country], 'alcohol_100g').mean())
fig=plt.figure(figsize=(12,8))
plt.bar(y_pos,alcohol, align='center')
plt.title('Average total alcohol content per 100g')
plt.xticks(y_pos, top_countries)
plt.ylabel('alcohol/100g')    
plt.show()
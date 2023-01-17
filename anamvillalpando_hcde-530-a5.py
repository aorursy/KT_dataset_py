# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd # data processing
import urllib.error, urllib.parse, urllib.request # URL interation
import json # JSON handler
import matplotlib.pyplot as plt # Import the library that handles coloring
import numpy as np# We import the numpy library which will help us make multi-dimentional array operation (like the tables in pandas)

# This code retrieves our key from our Kaggle Secret Keys file
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("usdafd") # Import our USDA FoodData Central secret

# Get our Fruit and Vegetable list from our uploaded dataset(list of veggies and fruits text file)
fruitveggies_file = "/kaggle/input/a5data/fruitsveggies.txt"
# Our list of Fruit/Veggies
fruitveggies = []
with open(fruitveggies_file) as f:
    # Get each line timming the EOL characters
    fruitveggies = [line.rstrip() for line in f]
    # Print the result
    print(fruitveggies)

# API's Base URL https://fdc.nal.usda.gov/api-guide.html#bkmk-2
usdafd_base_url = "https://api.nal.usda.gov/fdc/v1/"

# Function to build the API URL using the provided Path and Query on top of the API's Base URL
def buildURL(path,query):
    return usdafd_base_url+path+"?"+query+"&api_key="+secret_value_0

# Function that calls a URL safely with error handling
def safeGet(url):
    try:
        # Open the URL
        return urllib.request.urlopen(url)
    # An error ocurred
    except urllib.error.URLError as e:
        # Server error
        if hasattr(e,"code"):
            print("The server couldn't fulfill the request.")
            print("Error code: ", e.code)
        # Unknown error
        elif hasattr(e,'reason'):
            print("We failed to reach a server")
            print("Reason: ", e.reason)
        return None
# Function to get the food data from the FoodData Central of the USDA
def getFoodData(food):
    # The API's path https://fdc.nal.usda.gov/api-spec/fdc_api.html#/
    path = "foods/search"
    # The API's query https://api.nal.usda.gov/fdc/v1/foods/search
    query = "dataType=Foundation,SR%20Legacy&query=%2Bdescription:%22"+food+"%22%20%2Bdescription:raw"
    # Build the full URL and print it
    full_url = buildURL(path,query)
    print(full_url)
    
    # Call our SafeGet request fuction
    result = safeGet(full_url)
    # If the result is valid return it as a JSON, otherwise, print a warning and return None
    if result:
        return json.load(result)
    else:
        print("Warning: Unable to retrieve information for: "+fruitveggie)
        return None

# Our Fruit dictionary, contains a "fv" key that will be a set of "Fruit/Veggies" found in the USDA's database
fruitveggie_dict = {"fv":[]}

# Iterate through all the fruits/Veggies in our list of Fruit/Veggies "fruitveggies"
for fruitveggie in fruitveggies:
    # If the name has a space, raplace is with "%20" which is the URL version of " "
    fruitveggie_fixed = fruitveggie.replace(" ", "%20")
    # Call our getFoodData() function to retrieve the USDA data for the current fruit/veggie
    result = getFoodData(fruitveggie_fixed)
    # Check if our result is no "None"(is valid)
    if result:
        # Get only the "foods" key of a valid result, we are only interested in the "foods"(food related) data, for the purpose of this Assignment, we don't care about all the other data.
        foods=result["foods"]
        # Each food can be present multiple times with different names, so iterate through all of them.
        for food in foods:
            # Assign each sub-dictionary("food") a key "name" that will correspond to the "searched food name"
            food["name"]=fruitveggie
        # Add the set of foods to our dictionary, extending the list in it.
        fruitveggie_dict["fv"].extend(foods)
# Print the JSON pritty form of it, currently commented out to not slow down my notebook
# print(json.dumps(fruitveggie_dict, indent=2))
# Normalize the JSON data of the "fruitveggie_dict['fv']" list using Pandas. We are interested in the "foodNutrients" which are nested in the JSON, so put it as required path.
# Use Description,Name ans Scientific Name as out metadata because we will be using them later.
df = pd.json_normalize(data=fruitveggie_dict['fv'], record_path='foodNutrients', meta=['description', 'scientificName','name'], errors='ignore')
# Check the result
df.head(40)
# Pivot our table to make the "nutrientName" with their 'unitName' columns. Using "name" as index and "median" as our aggregate function. The Value will still be the Value.
all_nutrients = df.pivot_table(values='value', index=['name'], columns=['nutrientName', 'unitName'], aggfunc='median')
# Verify the result
all_nutrients.head(50)
# We print the DataFrame wiht the Energy columns only
print(all_nutrients[['Energy']])
# We plot the values to see if we have any outliers
all_nutrients[['Energy']].plot(kind='bar', rot=90)
# Now, filter the "Nutrient Name" to only include "Energy" given that in this Assignment is what we are looking for.
energy = df[df.nutrientName == 'Energy']
# Pivot our table to make the "Unit Name" values(KCAL,kJ) columns. Use "name" and "Description" as our "index" so prevent aggregation on them and treat the combinations as different rows.
energy = energy.pivot_table(values='value', index=['name','description'], columns=['unitName'])
# Remove the name of the columns axis(which says "Unit Name") by setting it to NONE. 
energy = energy.rename_axis(columns = None)
print(energy)
# We group by name(which is what we used to search) and we get the median(the median function ignores the Null values so we are good)
energy = energy.groupby(['name']).median()
# Show part of the result
print(energy)
# We plot the values to see if we have any outliers
energy.plot(kind='bar', rot=90)
# We plot the all_nutrients values as a line chart to see if we have any outliers
energy.plot(kind='line', rot=90, grid=True)
# We get the Energy columns only.
mAL = all_nutrients[['Energy']].copy()
# Right now the energy columns are MultiIndex, so we make sure we remove the multiple indexes and just keep the Unit Name
mAL.columns = mAL.columns.droplevel() # Could also use: [col[1] for col in mAL.columns]
# We remove the Column axis name, we don't want 'unitName' to show up everywhere
mAL = mAL.rename_axis(columns = None)
# I will use numpy's function "where" to especify the conditional that will help us save either the round value(0 as decimal) or the X.5 value.
# If the the first decimal(N x10 mod 10) is not 5, we round, otherwise, we keep the 5 and we trim the other decimals(N x10 [as Int] /10).
# Finally insert the result to our mAL DataFrame as "KCALc"
mAL['KCALc'] = np.where( (mAL.KCAL * 10 % 10).astype(int) != 5, round(mAL.kJ / 4.184), (mAL.KCAL * 10).astype(int) / 10 )# mAL['KCALc'] = round(mAL.kJ / 4.184)
# Here I am selecting all the Rows for which teh KCAL value does not match out new KCALc value. Then ignoring all other columns and storing the result as discrepancies(dKCAL)
dKCAL = mAL[mAL.KCAL != mAL.KCALc][['KCALc','KCAL']]
# We now shot the result to see it
print(dKCAL)

# Use temporary styling because we want to be able to play with other styles and colors
with plt.style.context("ggplot"):
    # We now create a small sctter plot with the content of dKCAL
    dKCALplot = dKCAL.plot(kind='scatter',x='KCALc',y='KCAL')
    # We also draw a line in the center going from the MIN to the MAX values of KCALc to see how far from each other the values are visually
    # This line will represent the spot where the points of the scatter plot should be if "KCALc - KCAK = 0"
    dKCALplot.plot([dKCAL.KCALc.min(), dKCAL.KCALc.max()],[dKCAL.KCALc.min(), dKCAL.KCALc.max()], "r--", label='(KCALc - KCAL) = 0')
    # We show the legend forthe line ("Center")
    dKCALplot.legend()
# Show it
plt.show()


mAL['KCAL'].plot(kind='bar', rot=90, figsize=(12,12))
# Get the mean of the column "KCAL"
kcal_mean = mAL.KCAL.mean()
print("The Mean is: "+str(kcal_mean))
# Now use it to filter all the Produce that fall below it(keep the ones bigger or equal to the median)
mf = mAL[mAL.KCAL >= kcal_mean][['KCAL']]
mf.head(50)
# Plot the result
mf.plot(kind='bar', rot=90)
# Get the mean of the column "KCAL"
kcal_mean = mf.KCAL.mean()
print("The Mean is: "+str(kcal_mean))
# Now use it to filter all the Produce that fall below it(keep the ones bigger or equal to the median)
mf2=mf[mf.KCAL >= kcal_mean]
mf2.head(50)
# Plot the result icreasing the "figsize" to see it better
mf2.plot(kind='bar', grid=True, rot=90)
# Plot the result icreasing the "figsize" to see it better and chaging the line color
p = mf2.plot(kind='bar', figsize=(12,12),color="#9dcc5f")

# Change the background color to green just as an experiment
p.set_facecolor('black')

# Change the xlabel name and color
p.set_xlabel("Produce", color="#3366cc", fontsize='x-large')
# Change the x axis ticker colors
p.tick_params(axis='x', colors='#6ec3c1')

# Change the ylabel name and color
p.set_ylabel("Calories", color="#808000",fontsize='x-large')
# Change the y axis ticker colors
p.tick_params(axis='y', colors='#f86f15')

# Show the y grid only
p.grid(True, axis='y', color='#0d5f8a')

# Change the legend color
legend = plt.legend(facecolor='#1c1411', edgecolor='black', framealpha=1, fontsize='x-large')
# Change the legend font color
for text in legend.get_texts():
    text.set_color("#c197d2")
# Use temporary styling because we want to be able to play with other styles and colors
with plt.style.context("ggplot"):
    # We plot a Pie Chart(kind=pie) we explode the bigger value(using MAX), we increase the figure size and we include percentages.
    # shadow=True(include a shadow), startangle=90(start with a 90 angle), pctdistance=0.85(the percentages closer to the edge)
    # It will return a numpy.ndarray which contains 2 things, a pie chart and a table, we do 0 to get the Pie Chart axes.
    pie = mf2.plot(subplots=True, kind='pie', explode = (mf2.KCAL == max(mf2.KCAL)) * 0.1, figsize=(12,12), autopct='%1.1f%%', shadow=True, startangle=90, pctdistance=0.85)[0]
    # We Now set the title.
    pie.set_title('My Food Calorie Percentages', fontsize= 30, color='#3366cc')
    # We make sure we remove the ylabel
    pie.set_ylabel('')
    # We now chang the color of all the texts inside the chart(index and percentages)
    for text in pie.texts:
        text.set_color('black')       
    # Draw white circle circle from the center with 0.76 of radius
    center_circle = plt.Circle((0,0),0.76,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(center_circle)
plt.show()
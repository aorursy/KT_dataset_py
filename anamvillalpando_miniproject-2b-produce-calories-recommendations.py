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
    display(fruitveggies)

# API's Base URL https://fdc.nal.usda.gov/api-guide.html#bkmk-2
usdafd_base_url = "https://api.nal.usda.gov/fdc/v1/"

# Function to build the API URL using the provided Path and Query on top of the API's Base URL
def buildURL(path,dataType,query):
    return usdafd_base_url+path+"?"+dataType+"&"+query+"&api_key="+secret_value_0

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
def getFoodData(food, isProduce=True):
    # The API's path https://fdc.nal.usda.gov/api-spec/fdc_api.html#/
    path = "foods/search"
    # The API's data types(origin) to include.
    dataType = "dataType=SR%20Legacy"
    # The API's query https://api.nal.usda.gov/fdc/v1/foods/search
    query = "query=%2Bdescription:%22"+food+"%22"
    # Should we append the "raw" word?
    if (isProduce):
        dataType+="%2CFoundation"
        query+="%20%2Bdescription:raw"
    # Build the full URL and print it
    full_url = buildURL(path,dataType,query)
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
display(all_nutrients[['Energy']])
# We plot the values to see if we have any outliers
all_nutrients[['Energy']].plot(kind='bar', rot=90, figsize=(12,12))
# Now, filter the "Nutrient Name" to only include "Energy" given that in this Assignment is what we are looking for.
energy = df[df.nutrientName == 'Energy']
# Pivot our table to make the "Unit Name" values(KCAL,kJ) columns. Use "name" and "Description" as our "index" so prevent aggregation on them and treat the combinations as different rows.
energy = energy.pivot_table(values='value', index=['name','description'], columns=['unitName'])
# Remove the name of the columns axis(which says "Unit Name") by setting it to NONE. 
energy = energy.rename_axis(columns = None)
display(energy)
# We group by name(which is what we used to search) and we get the median(the median function ignores the Null values so we are good)
energy = energy.groupby(['name']).median()
# Show part of the result
display(energy)
# We plot the values to see if we have any outliers
energy.plot(kind='bar', rot=90, figsize=(12,12))
# We plot the all_nutrients values as a line chart to see if we have any outliers
energy.plot(kind='line', rot=90, grid=True, figsize=(12,12))
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
display(dKCAL)

# Use temporary styling because we want to be able to play with other styles and colors
with plt.style.context("ggplot"):
    # We now create a small sctter plot with the content of dKCAL
    dKCALplot = dKCAL.plot(kind='scatter',x='KCALc',y='KCAL', figsize=(12,12))
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
mf.plot(kind='bar', rot=90, figsize=(12,12))
# Get the mean of the column "KCAL"
kcal_mean = mf.KCAL.mean()
print("The Mean is: "+str(kcal_mean))
# Now use it to filter all the Produce that fall below it(keep the ones bigger or equal to the median)
mf2=mf[mf.KCAL >= kcal_mean]
mf2.head(50)
# Plot the result icreasing the "figsize" to see it better
mf2.plot(kind='bar', grid=True, rot=90, figsize=(12,12))
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
# Items to check for calorie counts
check_list = {"Water":"Water%20tap%20drinking", "Vitamins":"Vitamin%20water", "Minerals":"Salt%20table"}
# Our dictionary of water
check_dict = {"fv":[]}
for check_item in check_list:
    # Water API result
    check_result = getFoodData(check_list[check_item], False)
    # If the result is valid
    if check_result:
        # Get only the "foods" key of a valid result, we are only interested in the "foods"(food related) data, for the purpose of this Assignment, we don't care about all the other data.
        checks=check_result["foods"]
        # Each food can be present multiple times with different names, so iterate through all of them.
        for check in checks:
            # Assign each sub-dictionary("food") a key "name" that will correspond to the equivalent_name.
            check["name"]=check_item # check_item.split("%")[0]
        # Add the set of foods to our dictionary, extending the list in it.
        check_dict["fv"].extend(checks)

# Print the pretty JSON result
#### print(json.dumps(check_dict, indent=2))        
        
# Normalize the JSON data of the "fruitveggie_dict['fv']" list using Pandas. We are interested in the "foodNutrients" which are nested in the JSON, so put it as required path.
# Use Description,Name ans Scientific Name as out metadata because we will be using them later.
check_df = pd.json_normalize(data=check_dict['fv'], record_path='foodNutrients', meta=['name','description'], errors='ignore')
# Pivot our table to make the "nutrientName" with their 'unitName' columns. Using "name" as index and "median" as our aggregate function. The Value will still be the Value.
check_nutrients = check_df.pivot_table(values='value', index=['name','description'], columns=['nutrientName', 'unitName'], aggfunc='median')
# Remove everything but the energy.
check_nutrients = check_nutrients[['Energy']]
# Show the Head
check_nutrients.head()
# Columns to eliminate by groups already specified in the description above.
lipids_mono = ["Fatty acids, total monounsaturated","4:0","6:0","8:0","10:0","12:0","13:0","14:0","15:0","16:0","17:0","18:0","20:0","22:0","24:0"]
lipids_sat = ["Fatty acids, total saturated","14:1","14:1 c","15:1","16:1","16:1 c","16:1 t","17:1","17:1 c","18:1","18:1 c","18:1 t","20:1","20:1 c","22:1","22:1 c"]
lipids_poly = ["Fatty acids, total polyunsaturated","18:2","18:2 c","18:2 CLAs","18:2 n-6 c,c","18:2 t not further defined","18:3","18:3 c","18:3 n-3 c,c,c (ALA)","18:3 n-6 c,c,c","18:3i","18:4","20:2 c","20:2 n-6 c,c","20:3","20:3 c","20:3 n-3","20:3 n-6","20:4","20:4 c","20:4 n-6","20:5 c","20:5 n-3 (EPA)","21:5","22:1 t","22:2","22:4","22:5 n-3 (DPA)","22:6 c","22:6 n-3 (DHA)","24:1 c"]
lipids_trans = ["Fatty acids, total trans","Fatty acids, total trans-monoenoic","Fatty acids, total trans-polyenoic"]
sub_carbs = ["Fiber, total dietary","Total dietary fiber (AOAC 2011.25)","Fiber, soluble","Fiber, insoluble","Sugars, total including NLEA","Sugars, Total NLEA","Sucrose","Glucose (dextrose)","Fructose","Lactose","Maltose","Galactose","Starch"]
amino_acids = ["Tryptophan","Threonine","Isoleucine","Leucine","Lysine","Methionine","Cystine","Phenylalanine","Hydroxyproline","Tyrosine","Valine","Arginine","Histidine","Alanine","Aspartic acid","Glutamic acid","Glycine","Proline","Serine"]
vitamins = ["Vitamin A, RAE","Vitamin A, IU","Vitamin B-6","Vitamin B-12","Vitamin B-12, added","Vitamin C, total ascorbic acid","Vitamin D (D2 + D3)","Vitamin D2 (ergocalciferol)","Vitamin D3 (cholecalciferol)","Vitamin D (D2 + D3), International Units","Vitamin E (alpha-tocopherol)","Vitamin E, added","Vitamin K (phylloquinone)","Vitamin K (Menaquinone-4)","Vitamin K (Dihydrophylloquinone)","Biotin","Lutein","Citric acid","Cryptoxanthin, alpha","Thiamin","Riboflavin","Retinol","Niacin","Pantothenic acid","10-Formyl folic acid (10HCOFA)","Betaine","Malic acid","Folate, DFE","Folic acid","Folate, food","Folate, total","Phytoene","Phytofluene","Carotene, beta","Carotene, alpha","Cryptoxanthin, beta","Lycopene","Lutein + zeaxanthin","Zeaxanthin","cis-Lutein/Zeaxanthin","cis-Lycopene","cis-beta-Carotene","trans-Lycopene","trans-beta-Carotene","Tocopherol, beta","Tocopherol, delta","Tocopherol, gamma","Tocotrienol, alpha","Tocotrienol, beta","Tocotrienol, delta","Tocotrienol, gamma","5-methyl tetrahydrofolate (5-MTHF)","5-Formyltetrahydrofolic acid (5-HCOH4"]
minerals = ["Calcium, Ca","Iron, Fe","Magnesium, Mg","Phosphorus, P","Potassium, K","Sodium, Na","Zinc, Zn","Copper, Cu","Manganese, Mn","Selenium, Se","Fluoride, F","Iodine, I"]
other = ["Total fat (NLEA)", "Carbohydrate, by summation","Water"]
# We will concatenate all the lists into one and pass it to the Drop function.
joined_list = lipids_mono + lipids_sat + lipids_poly + lipids_trans + sub_carbs + amino_acids + vitamins + minerals + other
# Drop the columns corresponding to all the previous nutrient names. 
caloric_nutrients_df = all_nutrients.drop(columns=joined_list)
# Drop Energy in KiloJouls
caloric_nutrients_df.drop(('kJ'), axis = 1, level=1, inplace = True)
# Show Result
caloric_nutrients_df.head(50)
# Drop that do not have at least 80 non-0 or non-NaN values.
very_significant_df = caloric_nutrients_df.replace(0 , np.nan)
very_significant_df.dropna(axis=1, thresh=80, inplace = True)
# Show Result
very_significant_df.head(20)
# Columns to eliminate by groups already specified in the description above.
xanthines = ["Caffeine","Theobromine"] # Theophylline
phytosterol = ["Phytosterols","Campesterol","Beta-sitosterol","Stigmasterol"]
insignificant = ["Nitrogen","Ash","Cholesterol","Choline, free","Choline, from glycerophosphocholine","Choline, from phosphocholine","Choline, from phosphotidyl choline","Choline, from sphingomyelin","Choline, total"]
# We will concatenate all the lists into one and pass it to the Drop function.
joined_list2 = xanthines + phytosterol + insignificant
# Drop the columns corresponding to all the previous nutrient names. 
caloric_nutrients_df.drop(columns=joined_list2, inplace=True)
# Show Result
caloric_nutrients_df.head()
# We know that all the X units(Nutritional Values) are in Grams(G) and the Y unit is in KiloCalories(KCAL), so remove them.
caloric_nutrients_df = caloric_nutrients_df.droplevel('unitName', axis=1)
# Rename the columns.
caloric_nutrients_df.rename(columns={"Alcohol, ethyl": "Alcohol","Carbohydrate, by difference": "Carbs","Energy": "Calories","Total lipid (fat)":"Fat"}, inplace = True)
# Reorder the Columns
caloric_nutrients_df = caloric_nutrients_df[['Carbs', 'Protein', 'Fat', 'Alcohol', "Calories"]]
# Remove the axis names
caloric_nutrients_df.rename_axis(None, axis=0, inplace = True)
caloric_nutrients_df.rename_axis(None, axis=1, inplace = True)
# Show Result
caloric_nutrients_df.head()
# We first reset the index to avoid DataFrame warnings in case we end up with an empty one.
alcohol_df = caloric_nutrients_df.reset_index()
# Only get the Produce with non-0 and non-NaN Alcohol values.
alcohol_df = alcohol_df[alcohol_df.Alcohol.notnull() & alcohol_df.Alcohol != 0]
# Is the resulting DataFrame Empty?
print("-Are all Alcohol values either 0 or NaN?\n" + str(alcohol_df.empty))
# Show them
alcohol_df.head(20)
# Drop Alcohol inplace
caloric_nutrients_df.drop("Alcohol", axis=1, inplace=True)
# Show the result
caloric_nutrients_df.head()
# Create a scatter plot using 'ggplot' style to be able to see how close to the center the countries are.
with plt.style.context("ggplot"):
    # Create a subplot of 4x4
    fig, axs = plt.subplots(2, 2,figsize=(13, 13), gridspec_kw={'top': 0.92, 'hspace': 0.2, 'wspace': 0.1})
    # Set the super title
    fig.suptitle('Linearity Check', fontsize=20)
    # Create each of the scatter plots using the Nutritional Value against the Calories.
    # Carbs
    axs[0, 0].scatter(caloric_nutrients_df['Carbs'],caloric_nutrients_df['Calories'], c='orange', label='Carbs')
    axs[0, 0].set_title('Carbs')
    axs[0, 0].set_xlabel('Grams', fontsize=12)
    axs[0, 0].set_ylabel('Calories', fontsize=12)
    # Protein
    axs[0, 1].scatter(caloric_nutrients_df['Protein'],caloric_nutrients_df['Calories'], c='cyan', label='Protein')
    axs[0, 1].set_title('Protein')
    axs[0, 1].set_xlabel('Grams', fontsize=12)
    # Fat
    axs[1, 0].scatter(caloric_nutrients_df['Fat'],caloric_nutrients_df['Calories'], c='magenta', label='Fat')
    axs[1, 0].set_title('Fat')
    axs[1, 0].set_xlabel('Grams', fontsize=12)
    axs[1, 0].set_ylabel('Calories', fontsize=12)
    # Sum
    axs[1, 1].scatter(caloric_nutrients_df['Protein']+caloric_nutrients_df['Fat']+caloric_nutrients_df['Carbs'],caloric_nutrients_df['Calories'], c='Green', label='Sum')
    axs[1, 1].set_title('Sum')
    axs[1, 1].set_xlabel('Grams', fontsize=12)
# Show the Plot
plt.show()
# Create a scatter plot using 'ggplot' style to be able to see how close to the center the countries are.
with plt.style.context("ggplot"):    
    # Increase the figure size
    plt.figure(figsize=(13, 13))
    # Create the scatter using Carbs and Calories as x and y axis.
    plt.scatter(caloric_nutrients_df['Carbs'],caloric_nutrients_df['Calories'], c='orange', label='Carbs')
    # Create the scatter using CaProteinrbs and Calories as x and y axis.
    plt.scatter(caloric_nutrients_df['Protein'],caloric_nutrients_df['Calories'], c='cyan', label='Protein')
    # Create the scatter using Fat and Calories as x and y axis.
    plt.scatter(caloric_nutrients_df['Fat'],caloric_nutrients_df['Calories'], c='magenta', label='Fat')
    # Create the scatter using Sum of Nutrients and Calories as x and y axis.
    plt.scatter(caloric_nutrients_df['Protein']+caloric_nutrients_df['Fat']+caloric_nutrients_df['Carbs'],caloric_nutrients_df['Calories'], c='Green', label='Sum')
    # Show the legend
    plt.legend()
    # Set title
    plt.title('Linearity Check', fontsize=20)
    # Set the axis labels
    plt.xlabel('Nutritional Values', fontsize=14)
    plt.ylabel('Energy(Calories)', fontsize=14)
# Show the Plot
plt.show()
# Import sklearn
import sklearn.linear_model as lm

# Grab our X and Y values
X = caloric_nutrients_df[['Carbs','Protein','Fat']]
Y = caloric_nutrients_df['Calories']

# Create a Linear Regression model and train it(fit it) with out X and Y values
model_sk = lm.LinearRegression()
model_sk.fit(X, Y)

# Print the Intercept and the Coefficients. These will correspond to the constant (is the expected mean value of Y when all X=0), and the 
print('Intercept:\n', model_sk.intercept_)
print('Coefficients[Carbs,Protein,Fat]:\n', model_sk.coef_)

# Example prediction with our Sklearn model
Carbs = 10
Protein = 10
Fat = 10
print ('\nPredicted Calories:\n', model_sk.predict([[Carbs,Protein,Fat]]))
Y_pred = model_sk.predict(X)
# Create a scatter plot using 'ggplot' style to be able to see how close to the center the countries are.
with plt.style.context("ggplot"):
    # Increase the figure size
    plt.figure(figsize=(12, 12))
    # Plot the Prediction vs the Actual Value.
    plt.scatter(Y_pred, Y, color = 'magenta')#"r--"
    # Plot the equal value line
    plt.plot([Y.min(), Y.max()],[Y.min(), Y.max()], "r--", label='Equal Value', color='black')#
    # Show the legend
    plt.legend(loc=2)
    # Set title
    plt.title('Avtual Calories vs Predicted Calories', fontsize=14)
    # Set the axis labels
    plt.xlabel('Model Prediction(Calories)', fontsize=12)
    plt.ylabel('Actual Value(Calories)', fontsize=12)
# Show the Plot
plt.show()
# Import Statsmodels
import statsmodels.api as sm

# Add the Intercept(Constant)
X_const = sm.add_constant(X)
# We use Ordinary Least Squares to get our Multiple Linear Regresion Model(https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.OLS.html)
model1 = sm.OLS(Y, X_const).fit()
# We get our predictions
Y_model1 = model1.predict(X_const)

# Print the summary
print(model1.summary())
print("\n")
# We now the "hasconst" to False which means we are setting the Constant to 0 and get the result.
model2 = sm.OLS(Y, X, hasconst=False).fit()# hasconst=False
# We get our predictions
Y_model2 = model2.predict(X)

# Print the summary
print(model2.summary())
# Create a scatter plot using 'ggplot' style to be able to see how close to the center the countries are.
with plt.style.context("ggplot"):
    # Increase the figure size
    plt.figure(figsize=(12, 12))
    # Plot the normal prediction vs the actual value.
    plt.scatter(Y_model1, Y, color = '#3366ccaa')
    # Plot the no intercept prediction vs the actual value.
    plt.scatter(Y_model2, Y, color = "#9dcc5faa")
    # Plot the equal value line
    plt.plot([Y.min(), Y.max()],[Y.min(), Y.max()], "r--", label='Equal Value', color='red')
    # Show the legend
    plt.legend(loc=2)
    # Set title
    plt.title('Actual Calories vs Predicted Calories', fontsize=14)
    # Set the axis labels
    plt.xlabel('Model Prediction(Calories)', fontsize=12)
    plt.ylabel('Actual Value(Calories)', fontsize=12)
# Show the Plot
plt.show()
# Pivot our table to make the "nutrientName" with their 'unitName' columns. Using "description" as index and no aggregate function. The Value will still be the Value.
all_unaggregated = df.pivot_table(values='value', index=['description'], columns=['nutrientName', 'unitName'])
# We already know we only care about "Energy", "Protein", "Carbohydrate, by difference", "alcohol", and "Total lipid (fat)", so extract those only.
all_unaggregated = all_unaggregated[['Carbohydrate, by difference','Protein','Total lipid (fat)', 'Alcohol, ethyl','Energy']]
# Drop Energy in KiloJouls
all_unaggregated.drop(('kJ'), axis = 1, level=1, inplace = True)
# We know that all the X units(Nutritional Values) are in Grams(G) and the Y unit is in KiloCalories(KCAL), so remove them.
all_unaggregated = all_unaggregated.droplevel('unitName', axis=1)
# Rename the columns.
all_unaggregated.rename(columns={"Alcohol, ethyl": "Alcohol","Carbohydrate, by difference": "Carbs","Energy": "Calories","Total lipid (fat)":"Fat"}, inplace = True)
# Remove the axis names
all_unaggregated.rename_axis(None, axis=0, inplace = True)
all_unaggregated.rename_axis(None, axis=1, inplace = True)
# Replace all the NaN Alcohol values with 0
all_unaggregated.fillna(0, inplace = True)
# Remove all the produce with non-0 Alcohol values (if any).
all_unaggregated = all_unaggregated[all_unaggregated.Alcohol == 0]
# Drop The Alcohol column now that we don't need it.
all_unaggregated.drop('Alcohol', axis = 1, inplace = True)
# Finally, remove the produce names which are not needed in our model.
all_unaggregated.reset_index(drop=True, inplace=True)
# Show Result
all_unaggregated.head(10)
# Grab our X and Y values
new_X = all_unaggregated[['Carbs','Protein','Fat']]
new_Y = all_unaggregated['Calories']

# Add the Intercept(Constant)
new_X_const = sm.add_constant(new_X)

# We use Ordinary Least Squares to get our Multiple Linear Regresion Model(https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.OLS.html)
model3 = sm.OLS(new_Y, new_X_const).fit()
# We get our predictions
# Y_model3 = model3.predict(new_X_const)

# Print the model summary
print(model3.summary())
print("\n")
# We now the "hasconst" to False which means we are setting the Constant to 0 and get the result.
model4 = sm.OLS(new_Y, new_X, hasconst=False).fit()# hasconst=False
# We get our predictions
# Y_model4 = model2.predict(new_X)

# Print the summary
print(model4.summary())
# Our new Produce list.
new_produce = ['Cassava', 'Durian', 'Roselle', 'Soursop', 'Tamarinds']
# Our Fruit dictionary, contains a "fv" key that will be a set of "Fruit/Veggies" found in the USDA's database
new_procude_dict = {"fv":[]}

# Iterate through all the fruits/Veggies in our list of Fruit/Veggies "fruitveggies"
for produce in new_produce:
    # Call our getFoodData() function to retrieve the USDA data for the current fruit/veggie
    result = getFoodData(produce)
    # Check if our result is no "None"(is valid)
    if result:
        # Get only the "foods" key of a valid result, we are only interested in the "foods"(food related) data, for the purpose of this Assignment, we don't care about all the other data.
        foods = result["foods"]
        # Each food can be present multiple times with different names, so iterate through all of them.
        for food in foods:
            # Assign each sub-dictionary("food") a key "name" that will correspond to the "searched food name"
            food["name"]=produce
        # Add the set of foods to our dictionary, extending the list in it.
        new_procude_dict["fv"].extend(foods)

# Pretty print the results.
# print(json.dumps(new_procude_dict, indent=2))
        
# Normalize the JSON data of the "new_procude_dict['fv']" list using Pandas. We are interested in the "foodNutrients" which are nested in the JSON, so put it as required path.
# Use Description,Name ans Scientific Name as out metadata because we will be using them later.
new_produce_df = pd.json_normalize(data=new_procude_dict['fv'], record_path='foodNutrients', meta=['description', 'scientificName','name'], errors='ignore')
# Pivot our table to make the "nutrientName" with their 'unitName' columns. Using "description" as index and no aggregate function. The Value will still be the Value.
np_df = new_produce_df.pivot_table(values='value', index=['name'], columns=['nutrientName', 'unitName'], aggfunc='median')
# We already know we only care about "Energy", "Protein", "Carbohydrate, by difference", "alcohol", and "Total lipid (fat)", so extract those only.
np_df = np_df[['Carbohydrate, by difference','Protein','Total lipid (fat)', 'Alcohol, ethyl','Energy']]
# Drop Energy in KiloJouls
np_df.drop(('kJ'), axis = 1, level=1, inplace = True)
# We know that all the X units(Nutritional Values) are in Grams(G) and the Y unit is in KiloCalories(KCAL), so remove them.
np_df = np_df.droplevel('unitName', axis=1)
# Rename the columns.
np_df.rename(columns={"Alcohol, ethyl": "Alcohol","Carbohydrate, by difference": "Carbs","Energy": "Calories","Total lipid (fat)":"Fat"}, inplace = True)
# Remove the axis names
np_df.rename_axis(None, axis=0, inplace = True)
np_df.rename_axis(None, axis=1, inplace = True)
# Replace all the NaN Alcohol values with 0
np_df.fillna(0, inplace = True)
# Remove all the produce with non-0 Alcohol values (if any).
np_df = np_df[np_df.Alcohol == 0]
# Drop The Alcohol column now that we don't need it.
np_df.drop('Alcohol', axis = 1, inplace = True)

# We now get the independent variables (Nutritional Values)
X_proof = np_df[['Carbs','Protein','Fat']]
# Add the Intercept(Constant)
X_proof_const = sm.add_constant(X_proof)
# Now we use our 3 models to predict the calories.
Y_model1 = model1.predict(X_proof_const)
Y_model2 = model2.predict(X_proof)
Y_model3 = model3.predict(X_proof_const)
Y_model4 = model4.predict(X_proof)
# Add them to the DataFrame
np_df['1-Calories (Agg+Const)'] = Y_model1
np_df['2-Calories (Agg)'] = Y_model2
np_df['3-Calories (Const)'] = Y_model3
np_df['4-Calories (None)'] = Y_model4
# Show Result
np_df.head(10)
# Create a scatter plot using 'ggplot' style to be able to see how close to the center the countries are.
with plt.style.context("ggplot"):
    # Increase the figure size
    plt.figure(figsize=(12, 12))
    # Plot the agg+const prediction vs the actual value.
    plt.scatter(np_df['1-Calories (Agg+Const)'], np_df['Calories'], color = '#3366ccaa')
    # Plot the agg prediction vs the actual value.
    plt.scatter(np_df['2-Calories (Agg)'], np_df['Calories'], color = "#9dcc5faa")
    # Plot the const prediction vs the actual value.
    plt.scatter(np_df['3-Calories (Const)'], np_df['Calories'], color = "#6ec3c1aa")
    # Plot the none prediction vs the actual value.
    plt.scatter(np_df['4-Calories (None)'], np_df['Calories'], color = "#f86f15aa")
    # Plot the equal value line
    plt.plot([np_df['Calories'].min(), np_df['Calories'].max()],[np_df['Calories'].min(), np_df['Calories'].max()], "r--", label='Equal Value', color='red')
    # Show the legend
    plt.legend(loc=2)
    # Set title
    plt.title('Actual Calories vs Predicted Calories', fontsize=14)
    # Set the axis labels
    plt.xlabel('Model Prediction(Calories)', fontsize=12)
    plt.ylabel('Actual Value(Calories)', fontsize=12)
# Show the Plot
plt.show()
# Create a scatter plot using 'ggplot' style to be able to see how close to the center the countries are.
with plt.style.context("ggplot"):
    # Plot the agg+const prediction vs the actual value.
    np_df[['Calories','1-Calories (Agg+Const)','2-Calories (Agg)','3-Calories (Const)','4-Calories (None)']].plot(kind='bar', rot=90, figsize=(12, 12))
    # Show the legend
    plt.legend(loc=2)
    # Set title
    plt.title('Actual Calories vs Predicted Calories', fontsize=14)
    # Set the axis labels
    plt.xlabel('Model Prediction(Calories)', fontsize=12)
    plt.ylabel('Actual Value(Calories)', fontsize=12)
# Show the Plot
plt.show()
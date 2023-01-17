# Import the libraries needed

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

df = pd.read_csv('../input/nutrition-facts/menu.csv') # Read the dataset
df.head() #Get the first 5 rows of the dataset
df.describe(include="all") # Generate descriptive statistics
df.info() # Generate dataset info
df.isnull().values.any() # Check if there is any missing value
df.Category.unique() # Get the menu categories
import matplotlib.pyplot as plt



meals = df.head(110) # All meals (without drinks) are in first 110 rows of the dataset

meals_cal = pd.DataFrame({'Item': meals.Item, 'Calories': meals.Calories}) # Select only Items and Calories columns

meals_sort = meals_cal.sort_values('Calories', ascending=False) # Sort by calories

meals_sort.plot.barh(x='Item', y='Calories', figsize= (10,45)) # Plot horizontal bar

plt.show()
# Calories on each category



# Define dataframes for each category

brkf = df.loc[df.Category == 'Breakfast']

bnp = df.loc[df.Category == 'Beef & Pork']

cnf = df.loc[df.Category == 'Chicken & Fish']

sld = df.loc[df.Category == 'Salads']

snass = df.loc[df.Category == 'Snacks & Sides']

dess = df.loc[df.Category == 'Desserts'] 

bev = df.loc[df.Category == 'Beverages']

cnt = df.loc[df.Category == 'Coffee & Tea']

ss = df.loc[df.Category == 'Smoothies & Shakes']



import seaborn as sns

# Plot calorie distribution for each category

fig, axes = plt.subplots(3, 3, figsize=(15, 7), sharex=True)

sns.color_palette("tab10")

sns.distplot( brkf["Calories"] , color='red', ax=axes[0, 0], label = "Breakfast")

sns.distplot( bnp["Calories"] , color='orange',ax=axes[0, 1], label = "Beef & Pork")

sns.distplot( cnf["Calories"] , color='brown',ax=axes[0, 2], label = "Chicken & Fish")

sns.distplot( sld["Calories"] , color='lime',ax=axes[1, 0], label = "Salads")

sns.distplot( snass["Calories"] , color='green',ax=axes[1, 1], label = "Snacks & Sides")

sns.distplot( dess["Calories"] ,  color='teal',ax=axes[1, 2], label = "Desserts")

sns.distplot( bev["Calories"] ,  color='gold',ax=axes[2, 0], label = "Beverages")

sns.distplot( cnt["Calories"] ,  color='blue',ax=axes[2, 1], label = "Coffee & Tea")

sns.distplot( ss["Calories"] ,  color='violet',ax=axes[2, 2], label = "Smoothies & Shake")

fig.suptitle("Calories Distribution on Each Menu Category")

fig.legend()

plt.show()
# Average calories on each categories

avg_cat = [round(brkf['Calories'].mean(axis=0), 2), round(bnp['Calories'].mean(axis=0), 2), round(cnf['Calories'].mean(axis=0), 2),

          round(sld['Calories'].mean(axis=0), 2), round(snass['Calories'].mean(axis=0), 2), round(dess['Calories'].mean(axis=0), 2),

          round(bev['Calories'].mean(axis=0), 2), round(cnt['Calories'].mean(axis=0), 2), round(ss['Calories'].mean(axis=0), 2)]

index = ['Breakfast', 'Beef & Pork', 'Chicken & Fish', 'Salads', 'Snacks & Sides', 'Desserts', 'Beverages', 'Coffee & Tea', 'Smoothies & Shakes']

avg_calat= pd.DataFrame({'Avg Calories': avg_cat}, index=index)

ax = avg_calat.plot.bar(rot=0, color='gray', figsize=(13,8), title='Average Calories in Each Menu Category (in Cal)', legend=True)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

#ax.set_title("Average Calories in Each Menu Category (in Cal)")
print("Average calories of all McD's meals (include drinks) is ", round(df['Calories'].mean(axis=0), 2), "Cal.") #average calories of all meals (include drinks)

print("Average calories of all McD's meals (drinks excluded) is ", round(meals.Calories.mean(axis=0), 2), "Cal.") #average calories of meals (no drinks)
# Category: Beverages

#bev = df.loc[df.Category == 'Beverages']
# Get caloric intake contribution from Beverages category

cal_bev = pd.DataFrame({'Item': bev.Item, 'Calories': bev.Calories})

cal_bev['Men'] = cal_bev.Calories/2500

cal_bev['Women'] = cal_bev.Calories/2000

cal_bev
# Average Calories in 'Beverages' Category

avg_bev = bev['Calories'].mean(axis=0)

print("Average calories of 'Beverages' category is", round(avg_bev, 2))



# Generally, the recommended daily calorie intake is 2,000 calories a day for women and 2,500 for men.

avg_bev = pd.to_numeric(avg_bev) 

bev_men = avg_bev/2500

print("Averagely 'Beverages' contributes to men's calories intake for ", round(bev_men,2))



bev_women = avg_bev/2000

print("Averagely 'Beverages' contributes to women's calories intake for ", round(bev_women, 2))
# Category: Coffee & Tea

#cnt = df.loc[df.Category == 'Coffee & Tea']
# Get caloric intake contribution from Coffee & Tea category

cal_cnt = pd.DataFrame({'Item': cnt.Item, 'Calories': cnt.Calories})

cal_cnt['Men'] = cal_cnt.Calories/2500

cal_cnt['Women'] = cal_cnt.Calories/2000

cal_cnt
# Average Calories in 'Coffee & Tea' Category

avg_cnt = cnt['Calories'].mean(axis=0)

print("Average calories of 'Coffee & Tea' category is", round(avg_cnt, 2))



# Generally, the recommended daily calorie intake is 2,000 calories a day for women and 2,500 for men.

avg_cnt = pd.to_numeric(avg_cnt)

cnt_men = avg_cnt/2500

print("Averagely 'Coffee & Tea' contributes to men's calories intake for ", round(cnt_men, 2))



cnt_women = avg_cnt/2000

print("Averagely 'Coffee & Tea' contributes to women's calories intake for ", round(cnt_women, 2))
# Category: Smoothies & Shakes

#ss = df.loc[df.Category == 'Smoothies & Shakes']
# Get caloric intake contribution from Smoothies & Shakes category

cal_ss = pd.DataFrame({'Item': ss.Item, 'Calories': ss.Calories})

cal_ss['Men'] = cal_ss.Calories/2500

cal_ss['Women'] = cal_ss.Calories/2000

cal_ss
# Average Calories in 'Smoothies & Shakes' Category

avg_ss = ss['Calories'].mean(axis=0)

print("Average calories of 'Smoothies & Shakes' category is", round(avg_ss, 2))



# Generally, the recommended daily calorie intake is 2,000 calories a day for women and 2,500 for men.

avg_ss = pd.to_numeric(avg_ss)

ss_men = avg_ss/2500

print("Averagely 'Smoothies & Shakes' contributes to men's calories intake for ", round(ss_men, 2))



ss_women = avg_ss/2000

print("Averagely 'Coffee & Tea' contributes to women's calories intake for ", round(ss_women, 2))
drink = df.tail(150)

avg_drink = drink['Calories'].mean(axis=0)

print("Average calories of all drinks is", round(avg_drink, 2))



# Generally, the recommended daily calorie intake is 2,000 calories a day for women and 2,500 for men.

avg_drink = pd.to_numeric(avg_drink)

drink_men = avg_drink/2500

print("Averagely drink options contributes to men's calories intake for ", round(drink_men, 2))



drink_women = avg_drink/2000

print("Averagely drink options contributes to women's calories intake for ", round(drink_women, 2))
# Exploring the amount of calories on Crispy Chicken category

crispy = df[df['Item'].str.contains('Crispy Chicken')]

crispy_cal = pd.DataFrame({'Item': crispy.Item, 'Calories': crispy.Calories})

crispy_sort = crispy_cal.sort_values('Calories', ascending=False)

ax = crispy_sort.plot.barh(x='Item', y='Calories', figsize= (10,7))

ax.set_title("Total Calories in Crispy Chicken Item Category (in Cal)")
# Average calories on Crispy Chicken Category

avg_crispy_cal = crispy.Calories.mean(axis=0)

print("Average calories on Crispy Chicken category is ", avg_crispy_cal, "Cal.")
# Exploring the amount of calories on Grilled Chicken category

grilled = df[df['Item'].str.contains('Grilled Chicken')]

grilled_cal = pd.DataFrame({'Item': grilled.Item, 'Calories': grilled.Calories})

grilled_sort = grilled_cal.sort_values('Calories', ascending=False)

ax = grilled_sort.plot.barh(x='Item', y='Calories', figsize= (10,7))

ax.set_title("Total Calories in Grilled Chicken Item Category (in Cal)")
# Average calories on Grilled Chicken Category

avg_grilled_cal = round(grilled.Calories.mean(axis=0), 2)

print("Average calories on Grilled Chicken category is ", avg_grilled_cal, "Cal.")
# Get visual comparison of Crispy vs Grilled on calories

avg_cal = pd.DataFrame({'Categories':['Crispy Chicken', 'Grilled Chicken'], 'Avg Calories': [avg_crispy_cal, avg_grilled_cal]})

ax = avg_cal.plot.bar(x = 'Categories', y = 'Avg Calories', figsize=(7,5))

ax.set_title("Average Calories on Crispy Chicken and Grilled Chicken Menu Categories (in Cal)")

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
# Get the amount of cholesterol and sodium in Crispy Chicken category

crispy_chsod = pd.DataFrame({'Item': crispy.Item, 'Cholesterol': crispy.Cholesterol, 'Sodium': crispy.Sodium})

crispy_chsod_sort = crispy_chsod.sort_values('Sodium', ascending=False) #sort by sodium amount

ax = crispy_chsod_sort.plot.barh(x='Item', y=['Cholesterol','Sodium'], stacked=False, figsize= (10,15), color=['yellow', 'orange'])

ax.set_title("Total Cholesterol and Sodium in Crispy Chicken Menu Category (in mg)")

#for p in ax.patches:

#    ax.annotate(str(p.get_width()), (p.get_x() * 1.005, p.get_width() * 1.005))

#ax = crispy.plot.barh(stacked=True, figsize=(10, 12))

for p in ax.patches:

    left, bottom, width, height = p.get_bbox().bounds

    ax.annotate(str(width), xy=(left+width/2, bottom+height/2), 

                horizontalalignment='left', verticalalignment='center') # Give annotation to the chart
# Get the amount of cholesterol and sodium in Grilled Chicken category

grilled_chsod = pd.DataFrame({'Item': grilled.Item, 'Cholesterol': grilled.Cholesterol, 'Sodium': grilled.Sodium})

grilled_chsod_sort = grilled_chsod.sort_values('Sodium', ascending=False) #sort by sodium amount

ax = grilled_chsod_sort.plot.barh(x='Item', y=['Cholesterol','Sodium'], stacked=False, figsize= (10,15), color=['yellow', 'orange'])

ax.set_title("Total Cholesterol and Sodium in Grilled Chicken Menu Category (in mg)")

#ax = grilled.plot.barh(stacked=True, figsize=(10, 12))

for p in ax.patches:

    left, bottom, width, height = p.get_bbox().bounds

    ax.annotate(str(width), xy=(left+width/2, bottom+height/2), 

                horizontalalignment='left', verticalalignment='center') # Give annotation to the chart
# Get the average of sodium and cholesterol amount in Crispy Chicken category

avg_crispy_sod = round(crispy.Sodium.mean(axis=0), 2)

avg_crispy_ch = round(crispy.Cholesterol.mean(axis=0), 2)

print("The average amount of sodium in Crispy Chicken category is ", avg_crispy_sod, "mg, while the average amount of cholesterol is ", avg_crispy_ch, "mg.")



# Get the average of sodium and cholesterol amount in Grilled Chicken category

avg_grilled_sod = round(grilled.Sodium.mean(axis=0), 2)

avg_grilled_ch = round(grilled.Cholesterol.mean(axis=0), 2)

print("The average amount of sodium in Grilled Chicken category is ", avg_grilled_sod, "mg, while the average amount of cholesterol is ", avg_grilled_ch, "mg.")
# Plot a bar chart for visual comparison

avg_sod = [avg_crispy_sod, avg_grilled_sod]

avg_ch = [avg_crispy_ch, avg_grilled_ch]

index = ['Crispy Chicken', 'Grilled Chicken']

avg_sodch = pd.DataFrame({'Avg Sodium': avg_sod,

                   'Avg Cholesterol': avg_ch}, index=index)

ax = avg_sodch.plot.bar(rot=0, color=['brown', 'orange'], figsize=(13,5))

ax.set_title("Average Sodium and Cholesterol in Crispy Chicken and Grilled Chicken Categories (in mg)")

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

#plt.barh(['Avg Sodium on Crispy Chicken items', 'Avg Cholesterol on Crispy Chicken items', 'Avg Sodium on Grilled Chicken items', 'Avg Cholesterol on Grilled Chicken items'], [avg_crispy_sod, avg_crispy_ch, avg_grilled_sod, avg_grilled_ch])
# Get the amount of sugar and dietary fiber in Crispy Chicken category

crispy_sdf = pd.DataFrame({'Item': crispy.Item, 'Sugars': crispy.Sugars, 'Dietary Fiber': crispy['Dietary Fiber']})

crispy_sdf_sort = crispy_sdf.sort_values('Sugars', ascending=False) #sort by sugar amount

ax = crispy_sdf_sort.plot.barh(x='Item', y=['Sugars','Dietary Fiber'], stacked=False, figsize= (10,10), color=['lime', 'yellow'])

ax.set_title("Total Sugars and Dietary Fiber in Crispy Chicken Menu Category (in g)")

#ax = grilled.plot.barh(stacked=True, figsize=(10, 12))

for p in ax.patches:

    left, bottom, width, height = p.get_bbox().bounds

    ax.annotate(str(width), xy=(left+width/2, bottom+height/2), 

                horizontalalignment='left', verticalalignment='center') # Give annotation to the chart
# Get the amount of sugar and dietary fiber in Grilled Chicken category

grilled_sdf = pd.DataFrame({'Item': grilled.Item, 'Sugars': grilled.Sugars, 'Dietary Fiber': grilled['Dietary Fiber']})

grilled_sdf_sort = grilled_sdf.sort_values('Sugars', ascending=False) #sort by sugar amount

ax = grilled_sdf_sort.plot.barh(x='Item', y=['Sugars','Dietary Fiber'], stacked=False, figsize= (10,10), color=['lime', 'yellow'])

ax.set_title("Total Sugars and Dietary Fiber in Grilled Chicken Menu Category (in g)")

#ax = grilled.plot.barh(stacked=True, figsize=(10, 12))

for p in ax.patches:

    left, bottom, width, height = p.get_bbox().bounds

    ax.annotate(str(width), xy=(left+width/2, bottom+height/2), 

                horizontalalignment='left', verticalalignment='center') # Give annotation to the chart
# Get the average of sugars and dietary fiber amount in Crispy Chicken category

avg_crispy_sug = round(crispy.Sugars.mean(axis=0), 2)

avg_crispy_df = round(crispy['Dietary Fiber'].mean(axis=0), 2)

print("The average amount of sugars in Crispy Chicken category is ", avg_crispy_sug, "g, while the average amount of dietary fiber is ", avg_crispy_df, "g.")



# Get the average of sodium and cholesterol amount in Grilled Chicken category

avg_grilled_sug = round(grilled.Sugars.mean(axis=0), 2)

avg_grilled_df = round(grilled['Dietary Fiber'].mean(axis=0), 2)

print("The average amount of sugars in Grilled Chicken category is ", avg_grilled_sug, "g, while the average amount of dietary fiber is ", avg_grilled_df, "g.")
# Plot a bar chart for visual comparison

avg_sug = [avg_crispy_sug, avg_grilled_sug]

avg_df = [avg_crispy_df, avg_grilled_df]

index = ['Crispy Chicken', 'Grilled Chicken']

avg_sdf = pd.DataFrame({'Avg Sugars': avg_sug,

                   'Avg Dietary Fiber': avg_df}, index=index)

ax = avg_sdf.plot.bar(rot=0, color=['lime', 'green'], figsize=(10,5))

ax.set_title("Average Sodium and Cholesterol in Crispy Chicken and Grilled Chicken Categories (in mg)")

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
# Get the amount of vitamins and minerals in Crispy Chicken category

crispy_vm = pd.DataFrame({'Item': crispy.Item, 'Vit A': crispy['Vitamin A (% Daily Value)'], 'Vit C': crispy['Vitamin C (% Daily Value)'], 'Calcium': crispy['Calcium (% Daily Value)'], 'Iron': crispy['Iron (% Daily Value)']})

crispy_vm_sort = crispy_vm.sort_values('Vit A', ascending=False) #sort by vit A amount

ax = crispy_vm_sort.plot.barh(x='Item', y=['Vit A','Vit C', 'Calcium', 'Iron'], stacked=False, figsize= (8,15), colormap='spring')

ax.set_title("Total Vitamins and Minerals in Crispy Chicken Menu Category (in %DV)")

for p in ax.patches:

    left, bottom, width, height = p.get_bbox().bounds

    ax.annotate(str(width), xy=(left+width/2, bottom+height/2), 

                horizontalalignment='left', verticalalignment='center') # Give annotation to the chart
# Get the amount of vitamins and minerals in Grilled Chicken category

grilled_vm = pd.DataFrame({'Item': grilled.Item, 'Vit A': grilled['Vitamin A (% Daily Value)'], 'Vit C': grilled['Vitamin C (% Daily Value)'], 'Calcium': grilled['Calcium (% Daily Value)'], 'Iron': grilled['Iron (% Daily Value)']})

grilled_vm_sort = grilled_vm.sort_values('Vit A', ascending=False) #sort by vit A amount

ax = grilled_vm_sort.plot.barh(x='Item', y=['Vit A','Vit C', 'Calcium', 'Iron'], stacked=False, figsize= (8,15), colormap='spring')

ax.set_title("Total Vitamins and Minerals in Grilled Chicken Menu Category (in %DV)")

for p in ax.patches:

    left, bottom, width, height = p.get_bbox().bounds

    ax.annotate(str(width), xy=(left+width/2, bottom+height/2), 

                horizontalalignment='left', verticalalignment='center') # Give annotation to the chart
# Get the average of vitamins and minerals in Crispy Chicken category

avg_crispy_vita = round(crispy_vm['Vit A'].mean(axis=0), 2)

avg_crispy_vitc = round(crispy_vm['Vit C'].mean(axis=0), 2)

avg_crispy_calc = round(crispy_vm['Calcium'].mean(axis=0), 2)

avg_crispy_iron = round(crispy_vm['Iron'].mean(axis=0), 2)

print("The average amount of Vit A, Vit C, Calcium, and Iron respectively in Crispy Chicken category is ", avg_crispy_vita, "%DV,", avg_crispy_vitc, "%DV,", avg_crispy_calc, "%DV, and", avg_crispy_iron, "%DV.")



# Get the average of vitamins and minerals in Grilled Chicken category

avg_grilled_vita = round(grilled_vm['Vit A'].mean(axis=0), 2)

avg_grilled_vitc = round(grilled_vm['Vit C'].mean(axis=0), 2)

avg_grilled_calc = round(grilled_vm['Calcium'].mean(axis=0), 2)

avg_grilled_iron = round(grilled_vm['Iron'].mean(axis=0), 2)

print("The average amount of Vit A, Vit C, Calcium, and Iron respectively in Grilled Chicken category is ", avg_grilled_vita, "%DV,", avg_grilled_vitc, "%DV,", avg_grilled_calc, "%DV, and", avg_grilled_iron, "%DV.")
# Plot a bar chart for visual comparison

avg_vita = [avg_crispy_vita, avg_grilled_vita]

avg_vitc = [avg_crispy_vitc, avg_grilled_vitc]

avg_calc = [avg_crispy_calc, avg_grilled_calc]

avg_iron = [avg_crispy_iron, avg_grilled_iron]

index = ['Crispy Chicken', 'Grilled Chicken']

avg_vm = pd.DataFrame({'Avg Vit A': avg_vita,

                   'Avg Vit C': avg_vitc,

                   'Avg Calcium': avg_calc,

                   'Avg Iron': avg_iron}, index=index)

ax = avg_vm.plot.bar(rot=0, colormap='spring', figsize=(10,5))

ax.set_title("Average Vitamins and Minerals in Crispy Chicken and Grilled Chicken Categories (in %DV)")

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
# Exploring the amount of calories on Crispy Chicken category

whites = df[df['Item'].str.contains('Egg White')]

whites_cal = pd.DataFrame({'Item': whites.Item, 'Calories': whites.Calories})

whites_sort = whites_cal.sort_values('Calories', ascending=False)

whites_sort.plot.barh(x='Item', y='Calories', figsize= (10,7))
# Exploring the amount of calories on Whole Eggs category

whole = df[df['Item'].str.contains('Egg')] 

whole = whole[~whole['Item'].str.contains('White')]

whole_cal = pd.DataFrame({'Item': whole.Item, 'Calories': whole.Calories})

whole_sort = whole_cal.sort_values('Calories', ascending=False)

whole_sort.plot.barh(x='Item', y='Calories', figsize= (10,7))
# Average calories on Egg Whites Category

avg_whites_cal = whites.Calories.mean(axis=0)

print("Average calories on Egg Whites category is ", round(avg_whites_cal, 2), "Cal.")



# Average calories on Whole Egg Category

avg_whole_cal = whole.Calories.mean(axis=0)

print("Average calories on Whole Egg category is ", round(avg_whole_cal, 2), "Cal.")
# Plot a bar chart for visual comparison

avg_egg_cal = [avg_whites_cal, avg_whole_cal]

index = ['Egg Whites', 'Whole Eggs']

avg_egg_cal = pd.DataFrame({'Avg Cal': avg_egg_cal,}, index=index)

ax = avg_egg_cal.plot.bar(rot=0)

ax.set_title("Average Calories in Egg Whites and Whole Eggs Categories (in Cal)")

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
# Get the amount of cholesterol and sodium in Egg Whites category

whites_chsod = pd.DataFrame({'Item': whites.Item, 'Cholesterol': whites.Cholesterol, 'Sodium': whites.Sodium})

whites_chsod_sort = whites_chsod.sort_values('Sodium', ascending=False) #sort by sodium amount

ax = whites_chsod_sort.plot.barh(x='Item', y=['Cholesterol','Sodium'], stacked=False, figsize= (10,15), color=['yellow', 'orange'])

ax.set_title("Total Cholesterol and Sodium in Egg Whites Menu Category (in mg)")

#for p in ax.patches:

#    ax.annotate(str(p.get_width()), (p.get_x() * 1.005, p.get_width() * 1.005))

#ax = crispy.plot.barh(stacked=True, figsize=(10, 12))

for p in ax.patches:

    left, bottom, width, height = p.get_bbox().bounds

    ax.annotate(str(width), xy=(left+width/2, bottom+height/2), 

                horizontalalignment='left', verticalalignment='center') # Give annotation to the chart
# Get the amount of cholesterol and sodium in Whole Whites category

whole_chsod = pd.DataFrame({'Item': whole.Item, 'Cholesterol': whole.Cholesterol, 'Sodium': whole.Sodium})

whole_chsod_sort = whole_chsod.sort_values('Sodium', ascending=False) #sort by sodium amount

ax = whole_chsod_sort.plot.barh(x='Item', y=['Cholesterol','Sodium'], stacked=False, figsize= (10,15), color=['yellow', 'orange'])

ax.set_title("Total Cholesterol and Sodium in Whole Eggs Menu Category (in mg)")

#for p in ax.patches:

#    ax.annotate(str(p.get_width()), (p.get_x() * 1.005, p.get_width() * 1.005))

#ax = crispy.plot.barh(stacked=True, figsize=(10, 12))

for p in ax.patches:

    left, bottom, width, height = p.get_bbox().bounds

    ax.annotate(str(width), xy=(left+width/2, bottom+height/2), 

                horizontalalignment='left', verticalalignment='center') # Give annotation to the chart
# Get the average of sodium and cholesterol amount in Egg Whites category

avg_whites_sod = round(whites.Sodium.mean(axis=0), 2)

avg_whites_ch = round(whites.Cholesterol.mean(axis=0), 2)

print("The average amount of sodium in Egg Whites category is ", avg_whites_sod, "mg, while the average amount of cholesterol is ", avg_whites_ch, "mg.")



# Get the average of sodium and cholesterol amount in Whole Eggs category

avg_whole_sod = round(whole.Sodium.mean(axis=0), 2)

avg_whole_ch = round(whole.Cholesterol.mean(axis=0), 2)

print("The average amount of sodium in Whole Eggs category is ", avg_whole_sod, "mg, while the average amount of cholesterol is ", avg_whole_ch, "mg.")
# Plot a bar chart for visual comparison

avg_egg_sod = [avg_whites_sod, avg_whole_sod]

avg_egg_ch = [avg_whites_ch, avg_whole_ch]

index = ['Egg Whites', 'Whole Eggs']

avg_egg_sodch = pd.DataFrame({'Avg Sodium': avg_sod,

                   'Avg Cholesterol': avg_ch}, index=index)

ax = avg_egg_sodch.plot.bar(rot=0, color=['brown', 'orange'], figsize=(10,5))

ax.set_title("Average Sodium and Cholesterol in Egg Whites and Whole Eggs Categories (in mg)")

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

#plt.barh(['Avg Sodium on Crispy Chicken items', 'Avg Cholesterol on Crispy Chicken items', 'Avg Sodium on Grilled Chicken items', 'Avg Cholesterol on Grilled Chicken items'], [avg_crispy_sod, avg_crispy_ch, avg_grilled_sod, avg_grilled_ch])
# Get the amount of sugar and dietary fiber in Egg Whites category

whites_sdf = pd.DataFrame({'Item': whites.Item, 'Sugars': whites.Sugars, 'Dietary Fiber': whites['Dietary Fiber']})

whites_sdf_sort = whites_sdf.sort_values('Sugars', ascending=False) #sort by sugar amount

ax = whites_sdf_sort.plot.barh(x='Item', y=['Sugars','Dietary Fiber'], stacked=False, figsize= (10,10), color=['lime', 'yellow'])

ax.set_title("Total Sugars and Dietary Fiber in Egg Whites Menu Category (in g)")

#ax = grilled.plot.barh(stacked=True, figsize=(10, 12))

for p in ax.patches:

    left, bottom, width, height = p.get_bbox().bounds

    ax.annotate(str(width), xy=(left+width/2, bottom+height/2), 

                horizontalalignment='left', verticalalignment='center') # Give annotation to the chart
# Get the amount of sugar and dietary fiber in Whole Eggs category

whole_sdf = pd.DataFrame({'Item': whole.Item, 'Sugars': whole.Sugars, 'Dietary Fiber': whole['Dietary Fiber']})

whole_sdf_sort = whole_sdf.sort_values('Sugars', ascending=False) #sort by sugar amount

ax = whole_sdf_sort.plot.barh(x='Item', y=['Sugars','Dietary Fiber'], stacked=False, figsize= (10,10), color=['lime', 'yellow'])

ax.set_title("Total Sugars and Dietary Fiber in Whole Eggs Menu Category (in g)")

#ax = grilled.plot.barh(stacked=True, figsize=(10, 12))

for p in ax.patches:

    left, bottom, width, height = p.get_bbox().bounds

    ax.annotate(str(width), xy=(left+width/2, bottom+height/2), 

                horizontalalignment='left', verticalalignment='center') # Give annotation to the chart
# Get the average of sugars and dietary fiber amount in Egg Whites category

avg_whites_sug = round(whites.Sugars.mean(axis=0), 2)

avg_whites_df = round(whites['Dietary Fiber'].mean(axis=0), 2)

print("The average amount of sugars in Egg Whites category is ", avg_whites_sug, "g, while the average amount of dietary fiber is ", avg_whites_df, "g.")



# Get the average of sodium and cholesterol amount in Grilled Chicken category

avg_whole_sug = round(whole.Sugars.mean(axis=0), 2)

avg_whole_df = round(whole['Dietary Fiber'].mean(axis=0), 2)

print("The average amount of sugars in Whole Whites category is ", avg_whole_sug, "g, while the average amount of dietary fiber is ", avg_whole_df, "g.")
# Plot a bar chart for visual comparison

avg_egg_sug = [avg_whites_sug, avg_whole_sug]

avg_egg_df = [avg_whites_df, avg_whole_df]

index = ['Egg Whites', 'Whole Eggs']

avg_egg_sdf = pd.DataFrame({'Avg Sugars': avg_sug,

                   'Avg Dietary Fiber': avg_df}, index=index)

ax = avg_egg_sdf.plot.bar(rot=0, color=['lime', 'green'], figsize=(10,5))

ax.set_title("Average Sugars and Dietary Fiber in Egg Whites and Whole Eggs Categories (in g)")

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
# Get the amount of vitamins and minerals in Egg Whites category

whites_vm = pd.DataFrame({'Item': whites.Item, 'Vit A': whites['Vitamin A (% Daily Value)'], 'Vit C': whites['Vitamin C (% Daily Value)'], 'Calcium': whites['Calcium (% Daily Value)'], 'Iron': whites['Iron (% Daily Value)']})

whites_vm_sort = whites_vm.sort_values('Iron', ascending=False) #sort by iron amount

ax = whites_vm_sort.plot.barh(x='Item', y=['Vit A','Vit C', 'Calcium', 'Iron'], stacked=False, figsize= (8,15), colormap='spring')

ax.set_title("Total Vitamins and Minerals in Egg Whites Menu Category (in %DV)")

for p in ax.patches:

    left, bottom, width, height = p.get_bbox().bounds

    ax.annotate(str(width), xy=(left+width/2, bottom+height/2), 

                horizontalalignment='left', verticalalignment='center') # Give annotation to the chart
# Get the amount of vitamins and minerals in Whole Eggs category

whole_vm = pd.DataFrame({'Item': whole.Item, 'Vit A': whole['Vitamin A (% Daily Value)'], 'Vit C': whole['Vitamin C (% Daily Value)'], 'Calcium': whole['Calcium (% Daily Value)'], 'Iron': whole['Iron (% Daily Value)']})

whole_vm_sort = whole_vm.sort_values('Vit A', ascending=False) #sort by iron amount

ax = whole_vm_sort.plot.barh(x='Item', y=['Vit A','Vit C', 'Calcium', 'Iron'], stacked=False, figsize= (8,15), colormap='spring')

ax.set_title("Total Vitamins and Minerals in Whole Eggs Menu Category (in %DV)")

for p in ax.patches:

    left, bottom, width, height = p.get_bbox().bounds

    ax.annotate(str(width), xy=(left+width/2, bottom+height/2), 

                horizontalalignment='left', verticalalignment='center') # Give annotation to the chart
# Get the average of vitamins and minerals in Egg Whites category

avg_whites_vita = round(whites_vm['Vit A'].mean(axis=0), 2)

avg_whites_vitc = round(whites_vm['Vit C'].mean(axis=0), 2)

avg_whites_calc = round(whites_vm['Calcium'].mean(axis=0), 2)

avg_whites_iron = round(whites_vm['Iron'].mean(axis=0), 2)

print("The average amount of Vit A, Vit C, Calcium, and Iron respectively in Egg Whites category is ", avg_whites_vita, "%DV,", avg_whites_vitc, "%DV,", avg_whites_calc, "%DV, and", avg_whites_iron, "%DV.")



# Get the average of vitamins and minerals in Whole Eggs category

avg_whole_vita = round(whole_vm['Vit A'].mean(axis=0), 2)

avg_whole_vitc = round(whole_vm['Vit C'].mean(axis=0), 2)

avg_whole_calc = round(whole_vm['Calcium'].mean(axis=0), 2)

avg_whole_iron = round(whole_vm['Iron'].mean(axis=0), 2)

print("The average amount of Vit A, Vit C, Calcium, and Iron respectively in Whole Eggs category is ", avg_whole_vita, "%DV,", avg_whole_vitc, "%DV,", avg_whole_calc, "%DV, and", avg_whole_iron, "%DV.")
# Plot a bar chart for visual comparison

avg_egg_vita = [avg_whites_vita, avg_whole_vita]

avg_egg_vitc = [avg_whites_vitc, avg_whole_vitc]

avg_egg_calc = [avg_whites_calc, avg_whole_calc]

avg_egg_iron = [avg_whites_iron, avg_whole_iron]

index = ['Egg Whites', 'Whole Eggs']

avg_egg_vm = pd.DataFrame({'Avg Vit A': avg_egg_vita,

                   'Avg Vit C': avg_egg_vitc,

                   'Avg Calcium': avg_egg_calc,

                   'Avg Iron': avg_egg_iron}, index=index)

ax = avg_egg_vm.plot.bar(rot=0, colormap='spring', figsize=(10,5))

ax.set_title("Average Vitamins and Minerals in Crispy Chicken and Grilled Chicken Categories (in %DV)")

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
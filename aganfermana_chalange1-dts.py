# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Gracia Evelyn Setyaputri
#A. How Many Calories Does The Average McDonald's Value Meal Contain ?

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/nutrition-facts/menu.csv') # baca dataset
#Gracia Evelyn Setyaputri
df.head() #tampilkan 5 row pertama dari data set
#Gracia Evelyn Setyaputri
df.describe(include="all")
#Gracia Evelyn Setyaputri
df.info()
#Gracia Evelyn Setyaputri
df.isnull().values.any()
#Gracia Evelyn Setyaputri
df.Category.unique()
#Gracia Evelyn Setyaputri
meals = df.head(110) # All meals (without drinks) are in first 110 rows of the dataset
meals_cal = pd.DataFrame({'Item': meals.Item, 'Calories': meals.Calories}) # Select only Items and Calories columns
meals_sort = meals_cal.sort_values('Calories', ascending=False) # Sort by calories
meals_sort.plot.barh(x='Item', y='Calories', figsize= (10,45)) # Plot horizontal bar
plt.show()
#Gracia Evelyn Setyaputri
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
#Gracia Evelyn Setyaputri
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
#Gracia Evelyn Setyaputri
print("Average calories of all McD's meals (include drinks) is ", round(df['Calories'].mean(axis=0), 2), "Cal.") #average calories of all meals (include drinks)
print("Average calories of all McD's meals (drinks excluded) is ", round(meals.Calories.mean(axis=0), 2), "Cal.") #average calories of meals (no drinks)
#Jaka Satria Prayuda
# B. How Much Do Beverages, Like Soda or Coffee, Contribute To The Overall Caloricv Intake ?


#Jaka Satria Prayuda
# C. Does ordered grilled chicken instead of crispy increase a sandwich's nutritional value? 

# EXPLORASI JUMLAH KALORI PADA CRISPY CHICKEN
crispy = df[df['Item'].str.contains('Crispy Chicken')]
crispy_cal = pd.DataFrame({'Item': crispy.Item, 'Calories': crispy.Calories})

# KALORI PADA CRISPY CHICKEN - RATA-RATA
avg_criscal = crispy.Calories.mean(axis=0)
print("CALORIES ON CRISPY CHICKEN (AVG): ", avg_criscal, "Cal.")

# EXPLORASI JUMLAH KALORI PADA GRILLED CHICKEN
grilled = df[df['Item'].str.contains('Grilled Chicken')]
grilled_cal = pd.DataFrame({'Item': grilled.Item, 'Calories': grilled.Calories})

# KALORI PADA PADA GRILLED CHICKEN - RATA-RATA
avg_grilcal = round(grilled.Calories.mean(axis=0))
print("CALORIES ON GRILLED CHICKEN (AVG): ", avg_grilcal, "Cal.")
#Jaka Satria Prayuda
# VISUALISASI PADA BAR CHART

avg_cal = pd.DataFrame({'Categories':['Crispy Chicken', 'Grilled Chicken'], 'Avg Calories': [avg_criscal, avg_grilcal]})
ax = avg_cal.plot.bar(x = 'Categories', y = 'Avg Calories')
ax.set_title("AVG CALORIES ON : CRISPY V GRILLED")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
#Jaka Satria Prayuda
# PERBANDINGAN RATA-RATA VIT A, VIT C, CALCIUM, IRON  

# EXPLORASI VIT A, VIT C, CALCIUM, DAN IRON PADA CRISPY CHICKEN
crispy_vm = pd.DataFrame({'Item': crispy.Item, 'Vit A': crispy['Vitamin A (% Daily Value)'], 'Vit C': crispy['Vitamin C (% Daily Value)'], 'Calcium': crispy['Calcium (% Daily Value)'], 'Iron': crispy['Iron (% Daily Value)']})

# EXPLORASI VIT A, VIT C, CALCIUM, DAN IRON PADA CRISPY CHICKEN
grilled_vm = pd.DataFrame({'Item': grilled.Item, 'Vit A': grilled['Vitamin A (% Daily Value)'], 'Vit C': grilled['Vitamin C (% Daily Value)'], 'Calcium': grilled['Calcium (% Daily Value)'], 'Iron': grilled['Iron (% Daily Value)']})

#DV : DAILY VALUE 

# RATA-RATA VIT A, VIT C, CALCIUM, DAN IRON PADA CRISPY CHICKEN
avg_crispy_vita = round(crispy_vm['Vit A'].mean(axis=0), 2)
avg_crispy_vitc = round(crispy_vm['Vit C'].mean(axis=0), 2)
avg_crispy_calc = round(crispy_vm['Calcium'].mean(axis=0), 2)
avg_crispy_iron = round(crispy_vm['Iron'].mean(axis=0), 2)
print("AVG OF VIT A, VIT C, CALCIUM, AND IRON IN CRISPY  CHICKEN: ", 
      avg_crispy_vita, "%DV,", avg_crispy_vitc, "%DV,", avg_crispy_calc, "%DV, and", avg_crispy_iron, "%DV.")

# RATA-RATA VIT A, VIT C, CALCIUM, DAN IRON PADA CRISPY CHICKEN
avg_grilled_vita = round(grilled_vm['Vit A'].mean(axis=0), 2)
avg_grilled_vitc = round(grilled_vm['Vit C'].mean(axis=0), 2)
avg_grilled_calc = round(grilled_vm['Calcium'].mean(axis=0), 2)
avg_grilled_iron = round(grilled_vm['Iron'].mean(axis=0), 2)
print("AVG OF VIT A, VIT C, CALCIUM, AND IRON IN GRILLED  CHICKEN: ", avg_grilled_vita, "%DV,", avg_grilled_vitc, "%DV,", avg_grilled_calc, "%DV, and", avg_grilled_iron, "%DV.")
#Jaka Satria Prayuda
# VISUALISASI PADA BAR CHART
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
ax.set_title("AVG VIT A, VIT C, CALCIUM AND IRON IN CRISPY AND GRILLED CHICKEN")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
# AGAN ERSYAD FERMANA
# D. What about ordering egg whites instead of whole eggs ? 

# EXPLORASI JUMLAH KALORI PADA WHITES EGG
whites = df[df['Item'].str.contains('Egg White')]
whites_cal = pd.DataFrame({'Item': whites.Item, 'Calories': whites.Calories})

# KALORI PADA WHITES EGG  - RATA-RATA
avg_whites_cal = whites.Calories.mean(axis=0)
print("CALORIES ON WHITE EGGS (AVG): ", avg_whites_cal, "Cal.")

# EXPLORASI JUMLAH KALORI PADA WHOLE EGG
whole = df[df['Item'].str.contains('Egg')] 
whole = whole[~whole['Item'].str.contains('White')]
whole_cal = pd.DataFrame({'Item': whole.Item, 'Calories': whole.Calories})

# KALORI PADA PADA WHOLE EGG - RATA-RATA
avg_whole_cal = whole.Calories.mean(axis=0)
print("CALORIES ON WHOLE EGGS (AVG): ", avg_whole_cal, "Cal.")
# AGAN ERSYAD FERMANA
# VISUALISASI PADA BAR CHART

avg_egg_cal = pd.DataFrame({'Categories':['White Egg ', 'Whole Egg'], 'Avg Calories': [avg_whites_cal, avg_whole_cal]})
ax = avg_egg_cal.plot.bar(x = 'Categories', y = 'Avg Calories')
ax.set_title("AVG CALORIES ON : WHITE EGG V WHOLE EGG")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
# AGAN ERSYAD FERMANA
# PERBANDINGAN RATA-RATA VIT A, VIT C, CALCIUM, IRON WHITE EGG & WHOLE EGG

# EXPLORASI VIT A, VIT C, CALCIUM, DAN IRON PADA WHITE EGG
whites_vm = pd.DataFrame({'Item': whites.Item, 'Vit A': whites['Vitamin A (% Daily Value)'], 
                          'Vit C': whites['Vitamin C (% Daily Value)'], 
                          'Calcium': whites['Calcium (% Daily Value)'], 
                          'Iron': whites['Iron (% Daily Value)']})

# EXPLORASI VIT A, VIT C, CALCIUM, DAN IRON PADA WHOLE EGG
whole_vm = pd.DataFrame({'Item': whole.Item, 'Vit A': whole['Vitamin A (% Daily Value)'], 
                         'Vit C': whole['Vitamin C (% Daily Value)'], 
                         'Calcium': whole['Calcium (% Daily Value)'], 
                         'Iron': grilled['Iron (% Daily Value)']})

#DV : DAILY VALUE 

# RATA-RATA VIT A, VIT C, CALCIUM, DAN IRON PADA WHITE EGG
avg_whites_vita = round(whites_vm['Vit A'].mean(axis=0), 2)
avg_whites_vitc = round(whites_vm['Vit C'].mean(axis=0), 2)
avg_whites_calc = round(whites_vm['Calcium'].mean(axis=0), 2)
avg_whites_iron = round(whites_vm['Iron'].mean(axis=0), 2)
print("AVG OF VIT A, VIT C, CALCIUM, AND IRON IN WHITE EGG: ", 
      avg_whites_vita, "%,", avg_whites_vitc, "%,", avg_whites_calc, "%, and", avg_whites_iron, "%.")

# RATA-RATA VIT A, VIT C, CALCIUM, DAN IRON PADA WHOLE EGG
avg_whole_vita = round(whole_vm['Vit A'].mean(axis=0), 2)
avg_whole_vitc = round(whole_vm['Vit C'].mean(axis=0), 2)
avg_whole_calc = round(whole_vm['Calcium'].mean(axis=0), 2)
avg_whole_iron = round(whole_vm['Iron'].mean(axis=0), 2)
print("AVG OF VIT A, VIT C, CALCIUM, AND IRON IN WHOLE EGG: ", 
      avg_whole_vita, "%,", avg_whole_vitc, "%,", avg_whole_calc, "%, and", avg_whole_iron, "%.")
# AGAN ERSYAD FERMANA
# VISUALISASI PADA BAR CHART
avg_vita = [avg_whites_vita, avg_whole_vita]
avg_vitc = [avg_whites_vitc, avg_whole_vitc]
avg_calc = [avg_whites_calc, avg_whole_calc]
avg_iron = [avg_whites_iron, avg_whole_iron]
index = ['White Egg', 'Whole Egg']
avg_vm = pd.DataFrame({'Avg Vit A': avg_vita,
                   'Avg Vit C': avg_vitc,
                   'Avg Calcium': avg_calc,
                   'Avg Iron': avg_iron}, index=index)
ax = avg_vm.plot.bar(rot=0, colormap='spring', figsize=(10,5))
ax.set_title("AVG VIT A, VIT C, CALCIUM AND IRON IN WHITES EGG AND WHOLE EGG")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
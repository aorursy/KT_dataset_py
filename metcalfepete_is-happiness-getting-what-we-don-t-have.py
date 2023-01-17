import pandas as pd

import matplotlib.pyplot as plt

import warnings



warnings.filterwarnings('ignore')



df = pd.read_csv('../input/world-happiness-report-2019.csv')



# Print the top 20 countries that rate Freedom, then money key to happiness

print("Top 20 Countries that think Freedom IS key to Happiness\n")

print(df.nlargest(20,'Freedom').sort_values('Freedom',ascending=False)[["Freedom","Country (region)"]])



print("\n\nTop 20 Countries that think Money IS key to Happiness\n")

print(df.nlargest(20,'Log of GDP\nper capita').sort_values('Log of GDP\nper capita',ascending=False)[["Log of GDP\nper capita","Country (region)"]])
print("Top 20 Countries that think money IS NOT key to Happiness\n")

print(df.nsmallest(20,'Log of GDP\nper capita').sort_values('Log of GDP\nper capita',ascending=True)[["Log of GDP\nper capita","Country (region)"]])

# Show a scatter plot of Generosity vs. Happiness

ax = df.plot(kind='scatter', x='Ladder', y= 'Generosity', title ="Generosity and Happiness", figsize=(15, 10), legend=True, fontsize=12)

ax.set_xlabel("Happiness", fontsize=12)

ax.set_ylabel("Generosity", fontsize=12)

plt.show()
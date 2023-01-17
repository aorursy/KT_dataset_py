import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

ps = pd.read_csv('../input/videogames-sales-dataset/PS4_GamesSales.csv',encoding = 'windows-1252')
xbox = pd.read_csv('../input/videogames-sales-dataset/XboxOne_GameSales.csv',encoding = 'windows-1252')
xbox.head()
ps.head()
xbox = xbox.drop(['Pos', 'Genre', 'Year', 'North America', 'Europe', 'Japan', 'Rest of World'], axis=1)
xbox
ps = ps.drop(['Genre', 'Year', 'North America', 'Europe', 'Japan', 'Rest of World'], axis=1)
ps
xbox.info()
ps.info()
xbox['Publisher'].value_counts()
ps['Publisher'].value_counts()
# XBOX offered 613 games including 31 exclusives, which is equal to 5% exclusives
fig1, ax1 = plt.subplots()
ax1.pie([5, 95], explode=(0.2, 0), labels=['Microsoft Studios','Other studios'], autopct='%1.1f%%',
        shadow=True, startangle=180)
ax1.axis('equal')  
plt.title('DEVELOPERS OF GAMES SOLD ON XBOX BEFORE PURCHASING BETHESDA')

# PlayStation offered 1034 games including 47 + 25 exclusives, which is equal to 7% exclusives
fig2, ax2 = plt.subplots()
ax2.pie([7, 93], explode=(0.4, 0), labels=['Sony Studios','Other studios'], autopct='%1.1f%%',
        shadow=True, startangle=180)
ax1.axis('equal')  
plt.title('DEVELOPERS OF GAMES SOLD ON PLAYSTATION BEFORE THE LOSS OF BETHESDA GAMES')
# The total number of games sold on Xbox
sumXbox = pd.to_numeric(xbox['Global']).sum() 
print (sumXbox) 
xbox2 = xbox[xbox['Publisher'] == 'Microsoft Studios']
xbox2
# The number of sales of exclusive xbox games
sumXbox2 = pd.to_numeric(xbox2['Global']).sum() 
print (sumXbox2) 
# The total number of games sold on PlayStation
sumPs = pd.to_numeric(ps['Global']).sum() 
print (sumPs) 
ps1 = ps[ps['Publisher'] == 'Sony Interactive Entertainment']
ps1
ps2 = ps[ps['Publisher'] == 'Sony Computer Entertainment']
ps2
# The number of sales of exclusive PlayStation games by Sony Interactive Entertainment
sumPs1 = pd.to_numeric(ps1['Global']).sum() 
print (sumPs1) 
# The number of sales of exclusive PlayStation games by Sony Computer Entertainment
sumPs2 = pd.to_numeric(ps2['Global']).sum() 
print (sumPs2) 
# XBOX sold 269.03 million games including 44.61 million exclusives, which is equal to 16.6% of exclusives
fig1, ax1 = plt.subplots()
ax1.pie([16.6, 83.4], explode=(0.4, 0), labels=['Microsoft Studios','Other studios'], autopct='%1.1f%%',
        shadow=True, startangle=180)
ax1.axis('equal')  
plt.title('SALES OF XBOX GAMES BY STUDIO BEFORE PURCHASING BETHESDA')

# PlayStation sold 595.64 million games including 54.85 + 42.26 million exclusives, which is equal 16.3% of exclusives
fig2, ax2 = plt.subplots()
ax2.pie([16.3, 83.7], explode=(0.4, 0), labels=['Sony Studios','Other studios'], autopct='%1.1f%%',
        shadow=True, startangle=180)
ax2.axis('equal')  
plt.title('SALES OF PLAYSTATION BY STUDIO BEFORE THE LOSS OF BETHESDA GAMES')
N = 2
excluMeans = (44.61, 97.11)
nexcluMeans = (224.42, 498.53)

ind = np.arange(N)   
width = 0.7     

p1 = plt.bar(ind, excluMeans, width)
p2 = plt.bar(ind, nexcluMeans, width, bottom=excluMeans)

plt.ylabel('Sales in millions')
plt.title('Games sales by console (before Microsoft buy Bethesda)')
plt.xticks(ind, ('Xbox One', 'PS4'))
plt.yticks(np.arange(0, 651, 50))
plt.legend((p1[0], p2[0]), ('Exclusives', 'Available on both platforms'))

plt.show()
xboxb = xbox[xbox['Publisher'] == 'Bethesda Softworks']
xboxb
# The number of sales of Bethesda games on Xbox 
sumXboxb = pd.to_numeric(xboxb['Global']).sum() 
print (sumXboxb) 
psb = ps[ps['Publisher'] == 'Bethesda Softworks']
psb
# The number of sales of Bethesda games on PlayStation
sumPsb = pd.to_numeric(psb['Global']).sum() 
print (sumPsb) 
# Before buying Bethesda, XBOX offered 5% exclusive games
fig1, ax1 = plt.subplots()
ax1.pie([5, 95], explode=(0.2, 0), labels=['Microsoft Studios','Other studios'], autopct='%1.1f%%',
        shadow=True, startangle=180)
ax1.axis('equal')  
plt.title('DEVELOPERS OF GAMES SOLD ON XBOX BEFORE PURCHASING BETHESDA')

# Xbox is picking up 17 new exclusive games, increasing the exclusivity rate to 48/613 = 7.8% 
fig2, ax2 = plt.subplots()
ax2.pie([7.8, 92.2], explode=(0.4, 0), labels=['Microsoft Studios','Other studios'], autopct='%1.1f%%',
        shadow=True, startangle=180)
ax2.axis('equal')  
plt.title('DEVELOPERS OF GAMES SOLD ON XBOX AFTER PURCHASING BETHESDA')



# Before buying Bethesda, Xbox exclusive game sales accounted for 16.6% of total sales
fig1, ax1 = plt.subplots()
ax1.pie([16.6, 83.4], explode=(0.4, 0), labels=['Microsoft Studios','Other studios'], autopct='%1.1f%%',
        shadow=True, startangle=180)
ax1.axis('equal')  
plt.title('SALES OF XBOX GAMES BY STUDIO BEFORE PURCHASING BETHESDA')

# Xbox recovers 43.23 million exclusive sales, and the number of overall sales increases by 28.96 million. 
# Thus the number of sales of exclusive games represents 87.84 / 297.99 = 29.4% of total sales
fig2, ax2 = plt.subplots()
ax2.pie([29.4, 70.6], explode=(0.4, 0), labels=['Microsoft Studios','Other studios'], autopct='%1.1f%%',
        shadow=True, startangle=180)
ax1.axis('equal')  
plt.title('SALES OF XBOX GAMES BY STUDIO AFTER PURCHASING BETHESDA')
N = 2
excluMeans = (87.84, 97.11)
nexcluMeans = (210.15, 469.57)

ind = np.arange(N)   
width = 0.7      

p1 = plt.bar(ind, excluMeans, width)
p2 = plt.bar(ind, nexcluMeans, width, bottom=excluMeans)

plt.ylabel('Sales in millions')
plt.title('Games sales by console (after Microsoft bought Bethesda)')
plt.xticks(ind, ('Xbox One', 'PS4'))
plt.yticks(np.arange(0, 651, 50))
plt.legend((p1[0], p2[0]), ('Exclusives', 'Available on both platforms'))

plt.show()


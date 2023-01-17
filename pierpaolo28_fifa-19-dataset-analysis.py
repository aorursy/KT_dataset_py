# from IPython.display import HTML

# HTML('''

# <script>

#   function code_toggle() {

#     if (code_shown){

#       $('div.input').hide('500');

#       $('#toggleButton').val('Show Code')

#     } else {

#       $('div.input').show('500');

#       $('#toggleButton').val('Hide Code')

#     }

#     code_shown = !code_shown

#   }



#   $( document ).ready(function(){

#     code_shown=false;

#     $('div.input').hide()

#   });

# </script>

# <form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')



df = pd.read_csv("../input/data.csv")

df = df.drop('Unnamed: 0', axis=1)

df.head()
import matplotlib.pyplot as plt

plt.hist(df['Age'])

plt.xlabel('Players Age')

plt.ylabel('Number of players')

plt.show()

oldest = df.loc[df['Age'].idxmax()]

print("The oldest player in FIFA 19 is", df['Age'].max(), "years old. His name is", oldest['Name'], 

      'he is from',oldest['Nationality'],'and plays for',oldest['Club'],'.')

print('The median age of a player on FIFA 19 is', np.mean(df['Age']))

youngest = df.loc[df['Age'].idxmin()]

print('The youngest players is',df['Age'].min(), "years old. His name is", youngest['Name'], 

      'he is from',youngest['Nationality'],'and plays for',youngest['Club'],'.')
import matplotlib.pyplot as plt

plt.hist(df['Overall'])

plt.xlabel('Players Rating')

plt.ylabel('Number of players')

plt.show()

best = df.loc[df['Overall'].idxmax()]

print("The best player in FIFA 19 is", df['Overall'].max(), "overall. His name is", best['Name'], 

      'he is from',best['Nationality'],'and plays for',best['Club'],'.')

print('The median rating of a player on FIFA 19 is', np.mean(df['Overall']))

worst = df.loc[df['Overall'].idxmin()]

print('The worst players is',df['Overall'].min(), "overall. His name is", worst['Name'], 

      'he is from',worst['Nationality'],'and plays for',worst['Club'],'.')
import matplotlib.pyplot as plt

plt.hist(df['Potential'])

plt.xlabel('Players Potential')

plt.ylabel('Number of players')

plt.show()

bestp = df.loc[df['Potential'].idxmax()]

print("The best potential player in FIFA 19 is", df['Potential'].max(), "overall. His name is", bestp['Name'], 

      'he is from',bestp['Nationality'],'and plays for',bestp['Club'],'.')

print('The median potential rating of a player on FIFA 19 is', np.mean(df['Potential']))

worstp = df.loc[df['Potential'].idxmin()]

print('The worst potential player is',df['Potential'].min(), "overall. His name is", worstp['Name'], 

      'he is from',worstp['Nationality'],'and plays for',worstp['Club'],'.')
import seaborn as sns

plt.figure(figsize=(20, 15))

sns.regplot(df['Age'] , df['Overall'])

plt.title('Age vs Overall rating')

plt.show()
#Creating Dateframe with playars above 88 overall

df1 = df.query("Overall>=88")

df1.head()
plt.hist(df1.Age)

plt.xlabel('Age')

plt.ylabel('Number df players')

plt.title('Number of Players with Rating greater than 90')
import seaborn as sns

plt.figure(figsize=(10,8))

sns.heatmap(df1.corr(),linewidths=4)

plt.title('Dataset Heatmap')

plt.show()
#Getting rid of all the elements that makes difficult to convert the different columns datatypes

df1['Value'] = df1['Value'].str.replace('€', '')

#df1['Value'] = df1['Value'].str.replace('K', '000')

df1['Value'] = df1['Value'].str.replace('M', '')

df1['Wage'] = df1['Wage'].str.replace('€', '')

df1['Wage'] = df1['Wage'].str.replace('K', '')



# Changing the datatypes of the selected columns

df1.Value = df1.Value.astype('float')

df1.Wage = df1.Wage.astype('int')

df1.Name = df1.Name.astype('category')
plt.figure(figsize=(10, 10))

df2 = df1.sort_values(['Value'])

sns.barplot(x = "Name" , y  = 'Value', data = df2 ,order = df2['Name'], 

             palette = 'rocket')

plt.title("Value of players (In millions)")

plt.xticks(rotation = 90)
plt.figure(figsize=(10, 10))

df2 = df1.sort_values(['Wage'])

sns.barplot(x = "Name" , y  = 'Wage', data = df2 ,order = df2['Name'], 

             palette = 'rocket')

plt.title("Wage Of players (In K)")

plt.xticks(rotation = 90)
import matplotlib.pyplot as plt; plt.rcdefaults()



df_best_players = pd.DataFrame.copy(df.sort_values(by = 'Overall' , 

                                                   ascending = False ).head(22))



plt.figure(figsize=(35, 10))

plt.bar('Name' , 'Overall' , data = df_best_players, width=0.7)

plt.xlabel('Players names', fontsize=20) 

plt.xticks(rotation = 90,fontsize=20, fontname='monospace')

plt.ylabel('Overall Rating', fontsize=20)

plt.title('Top 22 players Overall Rating', fontsize=25)

plt.ylim(87 , 95)

plt.show()
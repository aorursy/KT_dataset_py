import pandas as pd



data = """

food,color,grams

apple,red,250

chicken,pink,500

kale,green,200

bread,brown,300

"""



df = pd.read_csv(pd.compat.StringIO(data))

print(df)
print("The food column:\n", df[['food']]) # get the "food" column

print("The food column part 2:\n",df.loc[:,'food']) # also get the food column

print("The food column using numbers:\n",df.iloc[:,0] ) # once again, food

print("Heavy foods:\n",df.loc[df['grams'] >= 300,:])# rows where grams >= 300

pokemon_df = pd.read_csv('../input/competitive-pokemon-dataset/pokemon-data.csv', sep = ';')

pokemon_df.head()

print(pokemon_df.loc[pokemon_df['Name'] == 'Charmander','Next Evolution(s)'].values[0])

print(type(pokemon_df.loc[pokemon_df['Name'] == 'Charmander','Next Evolution(s)'].values[0]))
black_friday = pd.read_csv("../input/black-friday/BlackFriday.csv")



black_friday.head()

grouped_data = black_friday.groupby(["User_ID"])

grouped_data[["Purchase"]].sum().head()

from IPython.display import Image

Image("../input/datasaurus/DataSaurus.gif")

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



# First let's get the total amount spent by each customer

black_friday["Total_Purchase"] = grouped_data['Purchase'].transform('sum')

plot_data = black_friday[

    ["User_ID",

     "Gender",

     "Age",

     "Occupation",

     "Total_Purchase"]]

plot_data = plot_data.drop_duplicates().reindex()

plot_data.head()

sns.set()



sns.swarmplot('Age', 'Total_Purchase', hue='Gender', dodge=True, data=plot_data.sample(frac = 0.25))

plt.show()



sns.swarmplot('Occupation', 'Total_Purchase', hue = 'Gender', dodge=True, data=plot_data.sample(frac = 0.25))

plt.show()

job_numbers = sorted(plot_data['Occupation'].unique()) #unique sorted job number

print(job_numbers)

job_names = [

    "Boxer", "Doctor", "Business-person", "Astronaut", "Youtuber", "Biologist",

    "Rocket scientist", "Teacher", "Baker", "Janitor", "Magician", "Author",

    "Statistician", "Engineer", "Diplomat", "Coach", "Jeweler", "Roboticist",

    "Spy", "Programmer", "Chemist"]



# create a mapping from numbers to jobs

mapping = {key: value for (key, value) in zip(job_numbers, job_names)}

print(mapping)



# create a new DataFrame with a Job column for our names

named_plot_data = plot_data.assign(

    Job = [mapping[plot_data['Occupation'].iloc[i]] for i in range(len(plot_data))])



named_plot_data.head()
sns.swarmplot('Job', 'Total_Purchase', hue = 'Gender', dodge=True, data=named_plot_data.sample(frac = 0.25))

plt.xticks(rotation=90)

plt.show()
named_plot_data.to_csv("../black_friday_plotting.csv")
import os

os.listdir('../')
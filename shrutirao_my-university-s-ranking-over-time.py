######### INSTRUCTIONS #########
#
# Fork this script and change the university name to see what rank it gets:
#
my_university_name = ["University of Waterloo"]
#
# Look at the log for a full list of universities you can choose from.
#
# If your university is listed under multiple names, you can combine as many names as you want like this:
# my_university_name = ["The Johns Hopkins University", "Johns Hopkins University"]
#
################################

# Import Packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
plt.rcParams['figure.figsize'] = 16, 12

# Import Data

timesData = pd.read_csv("../input/timesData.csv")
shanghaiData = pd.read_csv("../input/shanghaiData.csv")
cwurData = pd.read_csv("../input/cwurData.csv")

# Print off a list of universities

all_university_names = set(timesData.university_name).union(set(shanghaiData.university_name)).union(set(cwurData.institution))
all_university_names_list = [str(i) for i in (list(all_university_names))]

print("List of All Universities in Dataset")
print("-----------------------------------")
print ('\n'.join([ str(university) for university in sorted(all_university_names_list) ]))


times_plot_data = timesData[timesData.university_name.isin(my_university_name)][['world_rank','year']]
shanghai_plot_data = shanghaiData[shanghaiData.university_name.isin(my_university_name)][['world_rank','year']]
cwur_plot_data = cwurData[cwurData.institution.isin(my_university_name)][['world_rank','year']]

times_plot_data['source'] = 'Times'
shanghai_plot_data['source'] = 'Shanghai'
cwur_plot_data['source'] = 'CWUR'

# parse the first number in rank for data ranges
times_plot_data['world_rank'] = times_plot_data['world_rank'].str.split('-').str[0]
shanghai_plot_data['world_rank'] = shanghai_plot_data['world_rank'].str.split('-').str[0]

plot_data = times_plot_data.append(shanghai_plot_data).append(cwur_plot_data)
plot_data['world_rank'] = plot_data['world_rank'].astype(int)
ax = sns.pointplot(x='year',y='world_rank',hue='source',data=plot_data);

# Styling

plt.title(my_university_name[0] + " Ranking", fontsize=26)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)    
plt.ylabel("World Rank", fontsize=26)  
plt.xlabel("Year", fontsize=26) 
plt.tight_layout()
plt.legend(loc='upper left',fontsize=20)
ax.spines["top"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left()

# Save File
plt.savefig('university.png')
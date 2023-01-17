import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output

# Load the csv 
df = pd.DataFrame()
df = pd.read_csv('../input/developers-and-programming-languages/user-languages.csv')

# Remove names, to have only float values
try: 
    del(df['user_id'])
except Exception:
    print ("Error", Exception)

    
# Delete skills without users
df = df.loc[:, (df != 0).any(axis=0)]

# Small sample
(i  for i in df.columns )
#print(df.columns)
arr = df.columns
for c in arr : print( '"' + c + '" ,')
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

N_DEVELOPERS = 5000
df_reduced = df.tail(N_DEVELOPERS)

scores = []
for n_clusters in range(3,13):
    kmeans = KMeans(n_clusters = n_clusters, random_state = 0).fit(df_reduced)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(df_reduced, labels)
    print("For developers =", N_DEVELOPERS ," and n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    scores.append(silhouette_avg)
    
scores




df
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
score = [0.11414573480812767,
 0.13023532795215215,
 0.13807087484342054,
 0.14444646540548431,
 0.10572328418828708,
 0.1499842911925803,
 0.11669811474070241,
 0.12019922654379787,
 0.1231189528756723,
 0.083330824963460268]
all_scores = [None,None,None] + scores
plt.plot((all_scores))
plt.title('Silhouette scores, for number of clusters') 

plt.show()


import operator
import pandas as pd
from sklearn.cluster import KMeans

n_clusters = 8
kmeans = KMeans(n_clusters = n_clusters, random_state = 0).fit(df)
labels = kmeans.labels_#
#   
roles = pd.DataFrame()# Glue back to originaal data
#
df['clusters'] = labels
label_df = []
for cluster in range(n_clusters):
    sub_df = df[df['clusters'] == cluster]
    dict_tags = {}
    for column in sub_df.columns:
        if sub_df[column].sum() > 0: dict_tags[column] = sub_df[column].sum()#
    dict_tags.pop('clusters', None)
    sorted_dict_tags = sorted(dict_tags.items(), key = operator.itemgetter(1))
    my_type = pd.DataFrame.from_dict(sorted_dict_tags).tail(10)
    my_type.columns = ['Skill' , 'Weight' ]
    print("Type: ", cluster , " " ,sub_df.shape[0]/df.shape[0]*100 ," % of users" )#
    print(my_type)
    new_role_element = pd.DataFrame.from_dict(sorted_dict_tags).tail(10).T.iloc[0: 2]
    new_role_element.columns = pd.DataFrame.from_dict(sorted_dict_tags).tail(10).T.iloc[0]
    total =  float(sub_df.shape[0])
    new_role_element  = new_role_element.iloc[1: 2] / total
    roles = pd.concat((new_role_element, roles))
    
roles.fillna(0, inplace=True)

from math import pi
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML




def show_graph(cat, values, title,index):
    N = len(cat)
    
    COLORS=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    x_as = [n / float(N) * 2 * pi for n in range(N)]

    # Because our chart will be circular we need to append a copy of the first 
    # value of each list at the end of each list with data
    values += values[:1]
    x_as += x_as[:1]

    # Set color of axes
    plt.rc('axes', linewidth=0.5, edgecolor="#888888")

    # Create polar plot
    plt.figure(figsize=(15,7.5))
    ax = plt.subplot(111, polar=True)

    # Set clockwise rotation. That is:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Set position of y-labels
    ax.set_rlabel_position(0)

    # Set color and linestyle of grid
    ax.xaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)
    ax.yaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)

    # Set number of radial axes and remove labels
    plt.xticks(x_as[:-1], [])

    # Set yticks
    #plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"])

    # Plot data
    ax.plot(x_as, values, linewidth=0, linestyle='solid', zorder=3)

    # Fill area
    ax.fill(x_as, values, 'b', alpha=0.3,color=COLORS[index])

    # Set axes limits
    plt.ylim(0,max(values))
    plt.title("Skills of the developer type : " + title)

    # Draw ytick labels to make sure they fit properly
    for i in range(N):
        angle_rad = i / float(N) * 2 * pi

        if angle_rad == 0:
            ha, distance_ax = "center", 10
        elif 0 < angle_rad < pi:
            ha, distance_ax = "left", 1
        elif angle_rad == pi:
            ha, distance_ax = "center", 1
        else:
            ha, distance_ax = "right", 1
        ax.text(angle_rad, max(values) + distance_ax, cat[i], size=8, horizontalalignment=ha, verticalalignment="center")

    # Show polar plot
    plt.show()
    
def Get_Description(cat) :
    # Return developer description for a given skill set
    DEVELOPER_TYPES = [
        "Apple Developer",
        "Android with Java",
        "Multi language Jedi Developer",
        "Python with django ", 
        "React Angular" , 
        "PHP Developer",
        "Ruby on Rails", 
        "Static HTML Designer" ,
        "Unkown"]
    type_index = 8 # Default value
    if "c++"      in cat : type_index = 2
    if "android"  in cat : type_index = 1
    if "django"   in cat : type_index = 3
    if "react"    in cat : type_index = 4
    if "rails"    in cat : type_index = 6
    if "website"  in cat : type_index = 7
    if "swift"    in cat : type_index = 0
    if "wordpress" in cat : type_index = 5
    return DEVELOPER_TYPES[type_index]


role_index = []    
j = 0
for index, row in roles.iterrows():
    cat = []
    values = []
    for column in roles.columns: 
        if  row[column] > 0 :
            cat.append(column)
            values.append( row[column] / np.sum(row) * 100  )
    developer_description = Get_Description(cat[:10])
    skills = pd.DataFrame()
    skills['Skill']  = cat[:10]
    skills['Weight'] = values[:10]
    print ("Developer type : ",index , developer_description,)
    role_index.append(developer_description)
    display(skills.sort_values('Weight', ascending = False ))
                               
    show_graph(cat = cat[:10], values = values[:10], title = developer_description,index=j )
    j = j +1 
    

from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
Compute the correlation matrix
""" 
from scipy.spatial.distance import squareform, pdist
roles.index = role_index
display(roles)

res = pdist(roles, 'euclidean')
squareform(res)
roles_dist = pd.DataFrame(squareform(res), index=role_index, columns=role_index)
roles_dist

import seaborn as sns
import matplotlib.pyplot as plt

corr = roles_dist.corr()



"""
 Chart from seaborn documentation: https://seaborn.pydata.org/examples/many_pairwise_correlations.html
"""
sns.set(style="white")

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = False

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(13,11))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(240, 10)

plt.title("Developer type heatmap")

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,fmt= '.1f', annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

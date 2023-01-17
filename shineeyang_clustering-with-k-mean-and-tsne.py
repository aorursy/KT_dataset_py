import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

import matplotlib.patches as patches

from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})



import seaborn as sns

from functools import reduce

import pylab

import scipy.stats as scp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

%matplotlib inline

# Any results you write to the current directory are saved as output.
#define function as data-cleaning)

def data_cleaning(df, col_list):

    """

    Return a dataset with the col_list removed and without duplicates

    """

    df2 = df.drop(col_list, axis=1)

    df2.drop_duplicates(inplace=True)

    return df2
#Importing the datasets

df_pop = pd.read_csv("../input/population.csv")

df_geo = pd.read_csv("../input/name_geographic_information.csv", na_values='-') 

df_geo.rename(columns={'code_insee': 'CODGEO'}, inplace = True)

df_salary = pd.read_csv("../input/net_salary_per_town_categories.csv")

#df_ind = pd.read_csv("../input/base_etablissement_par_tranche_effectif.csv")



#Cleaning the data for easier merging. We don't need (for now) the different population category

df_pop = df_pop[df_pop["CODGEO"].apply(lambda x: str(x).isdigit())]

df_pop["CODGEO"] = df_pop["CODGEO"].astype(int)

df_pop_red = df_pop.loc[:,["CODGEO",'SEXE','NB']]

df_pop_grouped = df_pop_red.groupby(["CODGEO",'SEXE']).agg({'NB' : 'sum'})
#Main merge into the df dataset which will stay untouched.

#It is the "reference" dataset with as few modification as possible

drop_salary = ['LIBGEO','SNHMC14','SNHMP14','SNHME14','SNHMO14','SNHM1814', 'SNHM2614', 'SNHM5014']

df_salary_red = data_cleaning(df_salary, drop_salary)

df_salary_red = df_salary_red[df_salary_red["CODGEO"].apply(lambda x: str(x).isdigit())]

df_salary_red["CODGEO"] = df_salary_red["CODGEO"].astype(int)

df_salary_red.set_index('CODGEO', inplace=True, verify_integrity=True)



drop_geo = ['chef.lieu_région','préfecture','éloignement','numéro_circonscription','codes_postaux']

df_geo_red = data_cleaning(df_geo, drop_geo)

df_geo_red.drop([33873,36823], axis=0, inplace=True) #There are duplicates for Saint-Pierre-et-Miquelon and Laguépie

df_geo_red.set_index('CODGEO', inplace=True, verify_integrity=True)



df = df_salary_red.merge(df_geo_red, how="outer", left_index=True, right_index=True)

df = df.merge(df_pop_grouped, how = "outer", left_index=True, right_index=True)
#Creation of df_main, which is df with human-readable labels.

#Drop unnecessary columns

df_main = df.copy()

print(df_main.columns)



dict_columns = {

    'CODGEO': 'codgeo',

    'code_région':'region_code', 

    'nom_région':'region_name',

    'numéro_département':'department_code', 

    'nom_département':'department_name', 

    'nom_commune':'commune_name',

    'SNHM14' : 'mean_net_salary_hour',

    'SNHMF14': 'mean_net_salary_hour_female', 

    'SNHMFC14':'mean_net_salary_hour_female_executive', 

    'SNHMFP14':'mean_net_salary_hour_female_middle_manager', 

    'SNHMFE14':'mean_net_salary_hour_female_employee', 

    'SNHMFO14':'mean_net_salary_hour_female_worker',

    'SNHMH14' :'mean_net_salary_hour_male', 

    'SNHMHC14':'mean_net_salary_hour_male_executive', 

    'SNHMHP14':'mean_net_salary_hour_male_middle_manager', 

    'SNHMHE14':'mean_net_salary_hour_male_employee', 

    'SNHMHO14':'mean_net_salary_hour_male_worker', 

    'SNHMF1814':'mean_net_salary_hour_female_18_25',

    'SNHMF2614':'mean_net_salary_hour_female_26_50', 

    'SNHMF5014':'mean_net_salary_hour_female_51',

    'SNHMH1814':'mean_net_salary_hour_male_18_25',

    'SNHMH2614':'mean_net_salary_hour_male_26_50',

    'SNHMH5014':'mean_net_salary_hour_male_51'

}



df_main.reset_index(level=1, inplace=True)

df_main.rename(columns = dict_columns, inplace = True)

df_main.dropna(inplace=True, subset=['mean_net_salary_hour'])

df_main['longitude'] = pd.to_numeric(df_main['longitude'].str.replace(',', '.'))

#We print relevant information about the dataset

df_main.shape

df_main.info()

df_main.head(5)
# Perform the necessary imports

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from sklearn.cluster import KMeans

from sklearn.manifold import TSNE

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



# Prepare the data

list_departments = ["department_name", "NB",'mean_net_salary_hour_male', 'mean_net_salary_hour_male_executive', 'mean_net_salary_hour_male_middle_manager', 'mean_net_salary_hour_male_employee', 'mean_net_salary_hour_male_worker']

departments = df_main[list_departments].groupby("department_name").agg(

    {'NB':'sum',

     'mean_net_salary_hour_male':'mean',

     'mean_net_salary_hour_male_executive':'mean',

     'mean_net_salary_hour_male_middle_manager':'mean', 

     'mean_net_salary_hour_male_employee':'mean', 

     'mean_net_salary_hour_male_worker':'mean'

    }).sort_values("NB", ascending=True).reset_index()

departments.rename(columns={'index': 'department_rank',}, inplace=True)

departments_labels = departments["department_name"].tolist()



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,18))



departments_salary = departments.loc[:,['mean_net_salary_hour_male', 'mean_net_salary_hour_male_executive', 'mean_net_salary_hour_male_middle_manager', 

                                        'mean_net_salary_hour_male_employee', 'mean_net_salary_hour_male_worker']].values



# Calculate the linkage: mergings

mergings = linkage(departments_salary, method='ward')



# Plot the dendrogram, using varieties as labels

dend = dendrogram(mergings,

        labels=departments_labels,

        leaf_rotation=0,

        leaf_font_size=10,

        orientation = 'right',

        color_threshold = 4,

        ax = ax

)

ax.set_title = "Dendrogram of the French departments, groups by male salaries"

plt.show()
# Do the TSNE

tsne = TSNE(learning_rate=150)

departments_transformed = tsne.fit_transform(departments_salary)



ks = range(4, 20)

inertias = []



for k in ks:

    # Create a KMeans instance with k clusters: model

    model = KMeans(n_clusters = k)

    

    # Fit model to samples

    model.fit(departments_transformed)

    

    # Append the inertia to the list of inertias

    inertias.append(model.inertia_)

    

# Plot ks vs inertias

fig, ax = plt.subplots(nrows=1, ncols=1)

ax.plot(ks, inertias, '-o')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')

plt.title('Inertia by number of clusters')

plt.xticks(ks)

ax.add_patch(

    patches.Rectangle(

        (6.5, 40),   # (x,y)

        7,          # width

        110,          # height

        facecolor='none',

        edgecolor='r',

        linewidth=1

    )

)

plt.show()
# Create a KMeans model with 10 clusters: model

nclust = 10

model = KMeans(n_clusters = nclust)



# Fit model and obtain cluster labels and cmap them

classes = model.fit_predict(departments_transformed)

num_colors = nclust

cm = pylab.get_cmap('tab20')



# Create the visualization

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))



for i in np.unique(classes):

    ix = np.where(classes == i)

    ax = plt.scatter(departments_transformed[ix,0], departments_transformed[ix,1], 

                 c = cm(1*i/num_colors), label = i)

# Annotations

    for label, x, y in zip(departments_labels, departments_transformed[:, 0], departments_transformed[:, 1]):

        plt.annotate(

            label,

            xy=(x + x/100, y + y/100),

            fontsize=10, 

            alpha=0.6)

    plt.legend(loc = "upper right")



    # Print the cluster

departments['cluster'] = classes

for i in departments['cluster'].sort_values().unique():

    print("-----#####-----")

    print("Cluster " + str(i))

    print(departments[departments["cluster"] == i]["department_name"])
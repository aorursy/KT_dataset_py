import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



mpl.rcParams['figure.figsize'] = [6, 4]

mpl.rcParams['figure.dpi'] = 100

mpl.rcParams['font.size'] = 14



import plotly.graph_objects as go

from plotly.subplots import make_subplots

from scipy.stats import variation

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        
!pip install numpy-stl

from stl import mesh
df_ansur2_female = pd.read_csv("../input/ansur-ii/ANSUR II FEMALE Public.csv", 

                               encoding='latin-1') 

df_ansur2_male = pd.read_csv("../input/ansur-ii/ANSUR II MALE Public.csv",

                             encoding='latin-1') 

df_ansur2_female = df_ansur2_female.rename(

                columns = {"SubjectId":"subjectid"}) # Fixing a column name

df_ansur2_all = pd.concat([df_ansur2_female,df_ansur2_male])

print("Shapes of the dataframes (Female,Male,All): " + 

      str((df_ansur2_female.shape,df_ansur2_male.shape,df_ansur2_all.shape)))

df_ansur2_all.head()
for gender in ['Male','Female']:

    subset = df_ansur2_all[df_ansur2_all['Gender'] == gender]

    

    # Draw the density plot

    sns.distplot(subset['stature'], hist = False, kde = True,

                 kde_kws = {'shade': True, 'linewidth': 3}, 

                  label = gender)

    

# Plot formatting

plt.legend(prop={'size': 14}, title = 'Gender')

plt.title('Density Plot comparing Height Distributions')

plt.xlabel('Height (mm)')

plt.ylabel('Density')

plt.show()
for gender in ['Male','Female']:

    subset = df_ansur2_all[df_ansur2_all['Gender'] == gender]

    

    # Draw the density plot

    sns.distplot(subset['interpupillarybreadth'], hist = False, kde = True,

                 kde_kws = {'shade': True, 'linewidth': 3}, 

                  label = gender)

    

# Plot formatting

plt.legend(prop={'size': 14}, title = 'Gender')

plt.title('Density Plot comparing Inter-Pupilary Distance')

plt.xlabel('Inter-pupillary Breadth (mm)')

plt.ylabel('Density')

plt.show()
# Identify the numeric columns.

numeric_cols = list(df_ansur2_all.select_dtypes([np.number]).columns)

numeric_cols = ([ele for ele in numeric_cols if ele 

                 not in ['subjectid','SubjectNumericRace', 'DODRace', 'Age']]) 



# Function to compute coefficient of variation

# defined as ratio of standard deviation to mean

def cov(x):

    return(np.std(x) / np.mean(x))



# Generate a data frame with coeffecient variation for each column



df_cov = (df_ansur2_all[['Gender'] + numeric_cols]

.groupby('Gender')

.apply(cov)

.reset_index()

.melt(id_vars = ['Gender'], value_name = 'Coeff_of_Variation', 

      var_name = 'Measurement')

 .sort_values(by = ['Coeff_of_Variation'], ascending = False))
# Plot showing top-coefficients of variation

clrs = ['black' if ('earprotrusion' in x) else 'grey' for x in df_cov.head(24).Measurement ]

clrs = [clrs[i] for i in range(len(clrs)) if i % 2 != 0] 



g = sns.catplot(x="Coeff_of_Variation", y="Measurement",

                 col="Gender",

                data=df_cov.head(24), kind="bar", palette = clrs,

                height=6, aspect=1);





plt.subplots_adjust(top=0.85)

g.fig.suptitle('Coefficeint of Variation for Various Measurements')

plt.show()
for gender in ['Male','Female']:

    subset = df_ansur2_all[df_ansur2_all['Gender'] == gender]

    

    # Draw the density plot

    sns.distplot(subset['wristcircumference'], hist = False, kde = True,

                 kde_kws = {'shade': True, 'linewidth': 3}, 

                  label = gender)

    

# Plot formatting

plt.legend(prop={'size': 12}, title = 'Gender')

plt.title('Density Plot comparing Wrist Circumference')

plt.xlabel('Wrist Circumference (mm)')

plt.ylabel('Density')

plt.show()
def stl2mesh3d(stl_mesh):

    # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles (i.e. three 3d points) 

    # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d

    p, q, r = stl_mesh.vectors.shape #(p, 3, 3)

    # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;

    # extract unique vertices from all mesh triangles

    vertices, ixr = np.unique(stl_mesh.vectors.reshape(p*q, r), return_inverse=True, axis=0)

    I = np.take(ixr, [3*k for k in range(p)])

    J = np.take(ixr, [3*k+1 for k in range(p)])

    K = np.take(ixr, [3*k+2 for k in range(p)])

    return vertices, I, J, K
mymesh = [

    mesh.Mesh.from_file('/kaggle/input/humanshapestlfiles/stature_1773_shs_0p52_age_38_bmi_16.stl'), 

    mesh.Mesh.from_file('/kaggle/input/humanshapestlfiles/stature_1773_shs_0p52_age_38_bmi_37.stl'),]
fig = make_subplots(rows=1, cols=2,

                    specs=[[{'is_3d': True}, {'is_3d': True}]],

                    subplot_titles=("BMI: 16", "BMI: 37"),

                    print_grid=False)



for i in [1,2]:

        vertices, I, J, K = stl2mesh3d(mymesh[i-1])

        triangles = np.stack((I,J,K)).T

        x, y, z = vertices.T



        fig.append_trace(

                          go.Mesh3d(x=x, y=y, z=z, 

                          i=I, j=J, k=K, 

                          showscale=False,

                          flatshading=False, 

                          lighting = dict(ambient=0.5,

                                          diffuse=1,

                                          fresnel=4,        

                                          specular=0.5,

                                          roughness=0.05,

                                          facenormalsepsilon=0),

                            ),

            row=1, col=i)

        

fig.update_layout(width=800, height=700,

                  template='plotly_dark', 

                 )





# fix the ratio in the top left subplot to be a cube

camera = dict(eye=dict(x=-1.25, y=-0.25, z=-0.25))

fig.update_layout(scene_aspectmode='manual',scene_aspectratio=dict(x=0.2, y=0.6, z=1), scene_camera = camera)

# manually force the z-axis to appear twice as big as the other two

fig.update_layout(scene2_aspectmode='manual',scene2_aspectratio=dict(x=0.25, y=0.6, z=1), scene2_camera = camera)





for i in fig['layout']['annotations']:

    i['font'] = dict(size=25,color='#ffffff')

    



fig.show()
# Extract all points from the STL files



points = [[],[]]

for i in range(2):

    points_ = []

    for triangle in list(mymesh[i].vectors):

        for point in triangle:

            points_.append(point)

    points[i] = np.array(points_)



# Extract the points corresponding to the wrist circumference

wrist_points = []

for wrist in points:

    wrist_points_ = pd.DataFrame(

        np.array(list(set(

            [tuple(item) for item in wrist if (

                (np.abs(item[2]-920) < 5) and item[1] > 300)]

        ))), columns = ['X','Y','Z'])

    wrist_points.append(wrist_points_)

    

# Plot the wrist circumferene cross-sections

f, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all')

ax1.scatter(wrist_points[0]['X'], wrist_points[0]['Y'])

ax1.set_title('BMI: 16 Wrist Size')

ax2.scatter(wrist_points[1]['X'], wrist_points[1]['Y'])

ax2.set_title('BMI: 37 Wrist Size')

f.show()
# Function to sort the array to generate nearest neighbor path

def nearest_neighbour_sort(df):

    df['Id'] = list(range(df.shape[0]))

    ids = df.Id.values[1:]

    xy = np.array([df.X.values, df.Y.values]).T[1:]

    path = [0,]

    while len(ids) > 0:

        last_x, last_y = df.X[path[-1]], df.Y[path[-1]]

        dist = ((xy - np.array([last_x, last_y]))**2).sum(-1)

        nearest_index = dist.argmin()

        path.append(ids[nearest_index])

        ids = np.delete(ids, nearest_index, axis=0)

        xy = np.delete(xy, nearest_index, axis=0)

    path.append(0)

    return path



wrist_points = []

for wrist in points:

    wrist_points_ = pd.DataFrame(np.array(list(set([tuple(item) for item in wrist if ((np.abs(item[2]-920) < 5) and item[1] > 300)]))), columns = ['X','Y','Z'])

    wrist_points_ = wrist_points_.loc[nearest_neighbour_sort(wrist_points_),].reset_index(drop = True)

    wrist_points_['distance'] = np.concatenate(([0.0],

                                                np.cumsum(np.sqrt((wrist_points_.X[1:].values - wrist_points_.X[:-1].values)**2+

                                                                  (wrist_points_.Y[1:].values - wrist_points_.Y[:-1].values)**2))))

    wrist_points.append(wrist_points_)



# Lineplot showing the list circumference

f, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all')

ax1.plot(wrist_points[0]['X'], wrist_points[0]['Y'])

ax1.set_title('BMI: 16 Wrist Size')

ax2.plot(wrist_points[1]['X'], wrist_points[1]['Y'])

ax2.set_title('BMI: 37 Wrist Size')

f.show()
print("Calculated circumference for the wrist of the person with BMI = 16 is {0:8.2f} mm".format(max(wrist_points[0].distance)))
print("Calculated circumference for the wrist of the person with BMI = 37 is {0:8.2f} mm".format(max(wrist_points[1].distance)))
for gender in ['Male','Female']:

    subset = df_ansur2_all[df_ansur2_all['Gender'] == gender]

    

    # Draw the density plot

    sns.distplot(subset['wristcircumference'], hist = False, kde = True,

                 kde_kws = {'shade': True, 'linewidth': 3}, 

                  label = gender)

    

# Plot formatting

plt.legend(prop={'size': 12}, title = 'Gender')

plt.title('Density Plot comparing Wrist Circumference')

plt.xlabel('Wrist Circumference (mm)')

plt.ylabel('Density')

plt.show()
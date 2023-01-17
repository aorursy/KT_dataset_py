###################

# Imports

###################



import json

import numpy as np # Linear aljebra



from os import listdir #Navigate in pc

from os.path import isfile, join



import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np

import pandas as pd



from sklearn.decomposition import PCA

from sklearn.preprocessing import scale

from mpl_toolkits.mplot3d import Axes3D # For 3d scatterplot visualization
####################

# Load data

####################



path_in = '../input/abstraction-and-reasoning-challenge/'

path_train = path_in + "training/"

path_test = path_in + "test/"

path_evaluation = path_in + "evaluation/"

path_out = '../working/kaggle/working\\'



# Function to open json

def open_json(path,name):

    with open('%s%s'%(path,name)) as f:

          data = json.load(f)

    return(data)





# File name lists

train_file_list = [f for f in sorted(listdir(path_train)) if isfile(join(path_train, f))]

test_file_list = [f for f in sorted(listdir(path_test)) if isfile(join(path_test, f))]

evaluation_file_list = [f for f in sorted(listdir(path_evaluation)) if isfile(join(path_evaluation, f))]



# Tasks to list

train_task_list = [open_json(path_train,name) for name in train_file_list]

test_task_list = [open_json(path_test,name) for name in test_file_list]

evaluation_task_list = [open_json(path_evaluation,name) for name in evaluation_file_list]
# ATTRIBUTION to

# BY T88 and Bo in kaggle: 

# > https://www.kaggle.com/t88take/check-the-purpose

# > https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines#evaluation-set

#

# ----------------------------------------------------------------------------------------

#

# Some changes have been made to avoid showing the images

#  two optional variables are added to *plot_task*

#  for avoiding showing the plot

#  and for saving the image



def plot_one(ax, i,train_or_test,input_or_output,task):

    cmap = colors.ListedColormap(

        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)

    

    input_matrix = task[train_or_test][i][input_or_output]

    ax.imshow(input_matrix, cmap=cmap, norm=norm)

    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    

    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])

    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.set_title(train_or_test + ' '+input_or_output)

    



def plot_task(task,show=True,savedict=None):

    """

    Plots the first train and test pairs of a specified task,

    using same color scheme as the ARC app

    """    

    num_train = len(task['train'])

    fig, axs = plt.subplots(2, num_train, figsize=(3*num_train,3*2))

    for i in range(num_train):     

        plot_one(axs[0,i],i,'train','input',task)

        plot_one(axs[1,i],i,'train','output',task)        

    plt.tight_layout()

    if show:

        plt.show()  

    

    if savedict!=None:

        plt.savefig(savedict['path']+savedict["name"]+"_train"+savedict['fmt'])

        plt.close()

        

    num_test = len(task['test'])

    fig, axs = plt.subplots(2, num_test, figsize=(3*num_test,3*2))

    if num_test==1: 

        plot_one(axs[0],0,'test','input',task)

        plot_one(axs[1],0,'test','output',task)     

    else:

        for i in range(num_test):      

            plot_one(axs[0,i],i,'test','input',task)

            plot_one(axs[1,i],i,'test','output',task)  

    plt.tight_layout()

    if show:

        plt.show() 

        

    if savedict!=None:

        plt.savefig(savedict['path']+"\\"+savedict["name"]+"_test"+savedict['fmt'])

        plt.close()

       

    

# This is mine

def plot_export(task,subtask_number,train_test):

    fig, axs = plt.subplots(2,1, figsize=(3,3*2))

    plot_one(axs[0],subtask_number-1,train_test,'input',task)

    plot_one(axs[1],subtask_number-1,train_test,'output',task)        

    

    return(fig)
# ----------------------------

# Save all tasks in train

# ----------------------------

"""

task_counter = 1

for task,task_file_code in zip(train_task_list,train_file_list):

    plot_task(task,show=False,savedict={'path': path_out,

                                        'name': str(task_counter)+"_"+task_file_code,

                                        'fmt': ".png"})

    if(task_counter/5==float(task_counter//5)): print(task_counter)

    task_counter+=1

"""
#################################

# Single feature extractors

#############################



# Different *task* feature extractor functions

def n_subtask_train(task):

    return(len(task['train']))





# Different *subtask* feature extractor



def check_is_subtask(subtask):

    assert(isinstance(subtask,dict))

    assert('input' in subtask.keys())

    assert('output' in subtask.keys())



def n_shape(subtask,origin='input'):

    # Make sure we have a subtask structure

    check_is_subtask(subtask)     

    # Constraint for origin

    assert(origin in ['input','output'])



    # return shape of origin

    return(np.array(subtask[origin]).shape)



def n_colors(subtask,origin='input'):

    # Make sure we have a subtask structure

    check_is_subtask(subtask)

    # Constraint for origin

    assert(origin in ['input','output'])    



    # return shape of origin

    return(len(np.unique(np.array(subtask[origin]))))             





##############################

#  All features of a task

###########################



def get_features_task(task,train_test='train'):



    features_list = []

    for subtask in task[train_test]:

            aux_feature_list = []

            # SHAPES of INPUT/OUTPUT

            input_shape = n_shape(subtask, origin='input')

            # number of ROWS of subtask INPUT

            aux_feature_list.append(input_shape[0])

            # number of COLUMNS of subtask INPUT

            aux_feature_list.append(input_shape[1])

            

            output_shape = n_shape(subtask, origin='output')

            # number of ROWS of subtask OUTPUT

            aux_feature_list.append(output_shape[0])

            # number of ROWS of subtask OUTPUT

            aux_feature_list.append(output_shape[1])

            

               

            # NUMBER OF COLORS

            # Input

            aux_feature_list.append(n_colors(subtask,origin='input'))

            # Output

            aux_feature_list.append(n_colors(subtask,origin='output'))

    

            features_list.append(aux_feature_list)

    return(features_list)
#########################################

#  Extract features of TRAIN tasks

#########################################



feature_rows = []



task_idx = 1 # Counter of task, will also be added for reference

for task in train_task_list:

    

    # Add all TRAIN subtask features

    subtask_count=1

    for subtask_feature_row in get_features_task(task,train_test='train'):

        feature_rows.append(['train',task_idx,subtask_count,train_file_list[task_idx-1]]+subtask_feature_row)

        subtask_count+=1

        

    # Add TEST subtask features

    feature_rows.append(['test',task_idx,1,train_file_list[task_idx-1]]+

                       get_features_task(task,train_test='test')[0])



    

    task_idx+=1 # Add for next task number



    

# Header following the added features

columns = ["type","#task","#subtask","task_id","n_row_IN","n_col_IN","n_row_OUT","n_col_OUT","#colors_IN","#colors_OUT"]



# Create a pandas dataframe to work on it

features_train = pd.DataFrame(feature_rows,columns=columns)



# Add additional features

features_train['cells_IN'] = features_train['n_row_IN']*features_train['n_col_IN']

features_train['cells_OUT'] = features_train['n_row_OUT']*features_train['n_col_OUT']

features_train['ratio_cells'] = features_train['cells_OUT']/features_train['cells_IN']

features_train['ratio_color'] = features_train['#colors_OUT']/features_train['#colors_IN']
################################

# Explore the features

###########################



features_train.describe()


# Take just columns we are interested in

column_numerics = ['n_row_IN', 'n_col_IN', 'n_row_OUT', 'n_col_OUT', '#colors_IN',

       '#colors_OUT', 'cells_IN', 'cells_OUT', 'ratio_cells', 'ratio_color']



# SCALING 

# Scale data to avoid just getting principal components too similar to 

#  features in data

X = scale(features_train[column_numerics])

features_train_scaled =pd.DataFrame(X,columns=column_numerics)



###########################

# PCA

###########################





print("%s\n#    EXPLAINED VARIANCE\n%s"%("#"+"-"*60,"#"+"-"*60))



pca = PCA().fit(features_train_scaled)

plt.plot(list(range(1,len(pca.explained_variance_ratio_)+1)),   

    np.cumsum(pca.explained_variance_ratio_), marker=".")

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.title("Explained variance vs. Number of components")

plt.show()

plt.close()





print("%s\n#    COMPONENT CORRELATIONS\n%s"%("#"+"-"*60,"#"+"-"*60))

# PC explained variance 

pca3 = PCA(n_components=3)

pca3.fit(features_train_scaled)

# Component meaning

plt.matshow(pca3.components_,cmap="bwr",vmin=-1,vmax=1)

plt.yticks([0,1,2],['1st Comp','2nd Comp', '3rd Comp'],fontsize=10)

plt.colorbar()

plt.xticks(range(1,len(column_numerics)+1),column_numerics,rotation=5,ha='left')

plt.show()

plt.close()





print("%s\n#    3D PLOT\n%s"%("#"+"-"*60,"#"+"-"*60))

# 3PC plot

X3= pca3.transform(features_train_scaled)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

# >> Colors depending on OUT/IN number of cells

color_list= features_train['ratio_cells']<1

plt.scatter(X3[:, 0], X3[:, 1], X3[:,2], alpha=0.8, marker=".",c=color_list)

ax.set_xlabel('PC1')

ax.set_ylabel('PC2')

ax.set_zlabel('PC3')

plt.title("3PC scatterplot")

plt.show()

plt.close()





print("%s\n#    2D PLOT\n%s"%("#"+"-"*60,"#"+"-"*60))



# 2PC plot

# > Performe PCA

pca2 = PCA(n_components=2)

pca2.fit(features_train_scaled)

X2 = pca2.transform(features_train_scaled)

# > Plot dimensionality reduction to 2PCs

# >> Colors depending on OUT/IN number of cells

color_list= features_train['ratio_cells']<1

# >> Scatter all points, with the corresponding color. We set 

#      transparency to 20%

plt.scatter(X2[:, 0], X2[:, 1], alpha=0.2, marker=".",c=color_list)

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.title("2PC Scatterplot")

plt.show()

plt.close()





###################

# Imports

###################



from bokeh.io import output_notebook, show

from bokeh.layouts import column, layout

from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput

from bokeh.plotting import figure
# Create a pandas dataframne with the 2PC DIMENSIONALITY REDUCTION

X2_pd = pd.DataFrame(X2,columns=["PC1","PC2"])

# Add the features of the subtaks

X2_pd['task_id']=features_train['task_id']

X2_pd['type']=features_train['type']

X2_pd['#task']=features_train['#task']

X2_pd['#subtask']=features_train['#subtask']
# We will use the visualizations attributed to bo (https://www.kaggle.com/boliu0)

# "https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines#training-set"

# in the bokeh hoover.



# You should go to the previous link and copy the url link of one of the images 

#  Take all except the last digit and ".png".



# Paste that to the following variable:

kaggle_bo_visuals_html = "https://www.kaggleusercontent.com/kf/28659886/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..fc-KLYaYs_niFe6hIBdOJg.5tv8EGf6GCHzIBRwdkJiIflkyyM5_YPKkbZ7cIWNrictbL27VWHnfOgZvYg2BW0sazthBATNnRiQzE4c_k4fAaesAuHAuuwkjrAUC5C-GY-F_ur-rCl1h1Brv4GrA6gGsRoWfMebUh8bz2zk0jCf5JWufr8FYEydFknD175ODao.XlVrPiYKwoh2MwqNUiK1BQ/__results___files/__results___5_"



# The number of the link is calculated to match the task

X2_pd['img_number'] = [str(3*(task_number)-2+int(task_type!="train"))+".png" for task_number,task_type in zip(X2_pd['#task'],X2_pd['type'])]

# The url for the task image

X2_pd['img'] = kaggle_bo_visuals_html+X2_pd['img_number']





path_aux = "../working/..\\working\\kaggle\\working\\"

path_aux ="https://strath-my.sharepoint.com/personal/clb19159_uni_strath_ac_uk/Documents/task_plots/"

X2_pd['img'] = [path_aux + str(task_number)+"_"+task_id+"_"+type_str+".png" for task_number,task_id,type_str in

               zip(X2_pd['#task'],X2_pd['task_id'],X2_pd['type'])]







# FILTER TASKS

X2_filtered = X2_pd.copy()



X2_filtered[X2_filtered['task_id']=="94f9d214.json"]['img'].unique()
######################################################  

# BOKEH figure

################################################



# Data pandas dataframe

task_pd = X2_filtered



# Add color for TEST TRAIN and different transparency since there are more TRAIN than TEST

task_pd["color"] = np.where(task_pd["type"] == "test", "green","red")

task_pd["alpha"] = np.where(task_pd["type"] == "test", 0.5, 0.3)



# Create Column Data Source that will be used by the plot

source = ColumnDataSource(data=dict(x=[], y=[], color=[], type_task=[],task_id=[],task_number=[],subtask_number=[],alpha=[],img=[]))



# Hoover: 

#   This information will be show when the cursor is on a task on the Principal Component plane

#  We add:

#    -> an image with all the train substaks, 

#    -> the subtask number that tells which task

#         starting from the left the point refers to, and

#    -> the json file name



TOOLTIPS = """

    <div>

        <div>

            <span style="font-size: 13px;">Subtask</span>

            <span style="font-size: 13px; color: #696;">@subtask_number</span>

            <span style="font-size: 13px;">  &#160&#160&#160     </span>

            <span style="font-size: 10px; font-weight: bold;">@task_id</span>

        </div>

        <div>

            <img

                src="@img" height="100" alt="@img" width="200"

                style="float: left; margin: 0px 15px 15px 0px;"

                border="2"

            ></img>

        </div>

    </div>

"""



# Create the figure

p = figure(plot_height=500, plot_width=650, title="", tooltips=TOOLTIPS, sizing_mode="fixed")

# Add the tasks for the PC1, PC2 components, specify the color and the tranparency

p.circle(x="x", y="y", source=source, size=7, color="color", line_color=None, fill_alpha="alpha")

# Add axis labels, we describe which overall feature of the tasks

#  the Principal Components are related to.

p.xaxis.axis_label = "PC1: Shape-color complexity"

p.yaxis.axis_label = "PC2: Cell_ratio"

# Add title

p.title.text = "Tasks over feature principal components"



# This function is not compulsory, we can edit it to add updates if needed

def update():

    df = task_pd

    

    source.data = dict(

        x=df['PC1'],

        y=df['PC2'],

        color=df["color"],

        type_task=df["type"],

        task_id=df["task_id"],

        task_number=df['#task'],

        subtask_number=df['#subtask'],

        alpha=df["alpha"],

        img=df['img']

    )

    

# Load data

update()  # initial load of the data

# Load notebook output

output_notebook()

# Show the figure p (if we show before updating it will lead a blank plot)

show(p)
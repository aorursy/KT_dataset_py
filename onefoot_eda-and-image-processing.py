import pandas as pd
import seaborn as sns
import numpy as np
import re 
import os
from glob import glob
import matplotlib.pyplot as plt
import ast

%matplotlib inline

plt.style.use('seaborn') #make plots prettier
train_dir =  "../input/train_simplified/"
csv_files = glob(train_dir + "*.csv")
def extract_classname(filename):
    return re.search( r"fied/(.+)\.csv",filename).group(1) 
class_names = [ extract_classname(file) for file in csv_files]
def count_lines(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i - 1 #minus one for header row

%%time
#is a little slow, many lines to count
line_counts = [ count_lines(file) for file in csv_files]
counts = pd.DataFrame({"line_counts":line_counts})
counts["class"] = class_names
sns.distplot(counts.line_counts.values,kde=False)
counts.describe()
i = counts.line_counts.nlargest(10).index
ax = counts.iloc[i].line_counts.plot.bar()
ax.set_xticklabels(counts.loc[i,"class"]);
def read_csvs(csv_files, nrows=1000):
    df =  pd.concat([ pd.read_csv(file,nrows=nrows) for file in csv_files])
    df.reset_index(inplace=True,drop=True)
    df['drawing'] = df.drawing.apply(ast.literal_eval)
    return df
%%time
#takes a litte while
df = read_csvs(csv_files)
df["n_strokes"] = df.drawing.apply(len)
s = df[df.word.str.contains("rabbit|sun|hot dog")] #look at a subset of rabbit, sun and hot dog drawings
s = s.groupby('word').n_strokes.plot.kde()
s.apply(lambda ax: ax.legend());
def list2numpy(points_list,size=1):

    """
    Takes a list of points and converts it to a boolean
    numpy array of size 72 by 72. Increase size to
    double the output size.
    """
    
    fig, ax = plt.subplots(figsize=(size,size))
    fig.tight_layout(pad=0)
    ax.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_axis_off()


    for points in points_list:
        ax.set_xlim(0,255)
        ax.set_ylim(0,255)
        ax.invert_yaxis()
        ax.plot(points[0],points[1])

    fig.canvas.draw()

    X = np.array(fig.canvas.renderer._renderer)
    plt.close()

    return X[:,:,1] == 255 
plt.imshow(list2numpy(df.drawing[7],size=4),cmap="gray")
plt.axis('off')
#default size is 72 b 72
list2numpy(df.drawing[7],size=1).shape
#but larger size possible
list2numpy(df.drawing[7],size=2).shape
def plot_images(df, w = 5, h =5):

    fig, axes = plt.subplots(w,h, figsize=(10,10))

    for i, ax in enumerate(axes.flatten()):
            ax.imshow(df.drawing[i], cmap="gray")
            ax.set_title(df.word[i])
            ax.set_axis_off()

n = 5 # change me to plot more images
#run cell a few times 
unrecognized = df[df.recognized == False].sample(n**2)
unrecognized.reset_index(inplace=True,drop=True)
unrecognized["drawing"] =   unrecognized.drawing.apply(list2numpy)
plot_images(unrecognized)
recognized = df[df.recognized == True].sample(n**2)
recognized.reset_index(inplace=True,drop=True)
recognized["drawing"] =   recognized.drawing.apply(list2numpy)
plot_images(recognized)
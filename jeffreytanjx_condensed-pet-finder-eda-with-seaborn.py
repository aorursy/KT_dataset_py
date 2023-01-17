import numpy as np # linear algebra
import pandas as pd # data processing
import os # importing data
import seaborn as sns # visualization
import matplotlib.pyplot as plt # visualization

print('drives in environment:')
print(os.listdir("../input"))
data = pd.read_csv("../input/train/train.csv")
data.head()
print('This dataset has {} rows and {} columns'.format(data.shape[0],data.shape[1]))
print('Number of entries and data type: \n\n')
data.info()
print('Basic statistics of numeric columns:')
data.describe()
# change encoding back to descriptive text
cleanup_cat = {
    "Type": {1: "Dog", 2: "Cat"},
    "MaturitySize": {1: "Small", 2: "Medium", 3: "Large", 4: "Extra Large", 5:"Unsure"},
    "FurLength": {1: "Short", 2: "Medium", 3: "Long", 4: "Unsure" },
    "Gender": {1: "Male",2:"Female",3:"Group"},
    "Vaccinated": {1: "Yes",2:"No",3:"Unsure"},
    "Dewormed": {1: "Yes",2:"No",3:"Unsure"},
    "Sterilized": {1: "Yes",2:"No",3:"Unsure"},
    "Health": {1: "Healthy",2:"Minor Injury",3:"Serious Injury",0:"Unsure"}
    
}

data.replace(cleanup_cat, inplace = True)


# convert categorical
convert = ['Type','State','Health','Sterilized','Dewormed','Vaccinated',
     'FurLength','MaturitySize','Gender','Color1','Color2','Color3','Breed1','Breed2']
data[convert] = data[convert].apply(lambda x: x.astype('category'))
data.head()
sns.set(font_scale = 5)
cat_col = list(data.select_dtypes("category").columns)
excluded_cat_col = [col for col in cat_col if col not in ['Breed1','Breed2','State']]
f, axes = plt.subplots(round(len(excluded_cat_col)), 2, figsize=(100,550))  # create plot
axes_list = [item for sublist in axes for item in sublist]  # flatten axes
f.suptitle('Analysis of AdoptionSpeed for categorical variable', y = 0.89, fontsize = 100)
for i, c in zip(range(0,len(excluded_cat_col)*2,2),excluded_cat_col):
    g1 = sns.countplot(x = c, data = data, ax = axes_list[i])

    # percentage
    total = data[c].count()
    for p in g1.patches:
        height = p.get_height()
        g1.text(p.get_x()+p.get_width()/2., height+40, '{0:.1%}'.format(height/total),ha = 'center')
        
    # stacked bar chart
    counter = data.groupby(c)['AdoptionSpeed'].value_counts().unstack()
    percentage_dist = 100 * counter.divide(counter.sum(axis = 1), axis = 0)
    g2 = percentage_dist.plot.bar(stacked = True, ax = axes_list[i+1], rot = 0)
    #g2 = sns.countplot(x = c, data = data, ax = axes_list[i+1], hue = "AdoptionSpeed", dodge = False)
    for p in g2.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        g2.annotate('{:.0f} %'.format(height), (p.get_x()+.15*width, p.get_y()+.4*height))
    
    
sns.set()
g = sns.FacetGrid(data, col = "Gender", xlim = (0,30))
g.map(sns.distplot,"Age", kde = False)
g.fig.suptitle("Histogram of Age for different gender")
g.fig.subplots_adjust(top = 0.8) # adjust title position

g2 = sns.FacetGrid(data, col = "Type")
g2.map(sns.countplot,"Gender")
g2.fig.suptitle("Count of Type by Gender")
g2.fig.subplots_adjust(top = 0.8) # adjust title position

g = sns.FacetGrid(data, col = "Vaccinated", row = "Dewormed", margin_titles = True)
g.map(sns.countplot,"Sterilized")
g.fig.suptitle("Count of Sterilized Animals")
g.fig.subplots_adjust(top = 0.8)
[plt.setp(ax.texts, text="") for ax in g.axes.flat] # remove the original texts
                                                    # important to add this before setting titles
g.set_titles(row_template = 'Dewormed - {row_name}', col_template = 'Vaccinated - {col_name}')
num_col = list(data.select_dtypes(np.number).columns)
f, axes = plt.subplots(round(len(num_col)/3),3, figsize = (20,10))
f.suptitle("Distribution of numeric variables")
axes_list = [item for sublist in axes for item in sublist]
for i,c in enumerate(num_col):
    sns.distplot(data[c], ax = axes_list[i])
sns.heatmap(data[num_col].corr())
included_num = ['Age','PhotoAmt']
f, axes = plt.subplots(1, 2, figsize=(25,10))  # create plot
#axes_list = [item for sublist in axes for item in sublist]  # flatten axes
f.suptitle('Analysis of AdoptionSpeed for numerical variable')
for i, c in enumerate(included_num):
    g = sns.boxplot(x = "AdoptionSpeed", y = c, data = data, ax = axes[i], showfliers = False)
    

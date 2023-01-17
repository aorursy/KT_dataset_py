# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



#load packages

import sys #access to system parameters https://docs.python.org/3/library/sys.html

print("Python version: {}". format(sys.version))



import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features

print("pandas version: {}". format(pd.__version__))



import matplotlib #collection of functions for scientific and publication-ready visualization

print("matplotlib version: {}". format(matplotlib.__version__))



import numpy as np #foundational package for scientific computing

print("NumPy version: {}". format(np.__version__))



import scipy as sp #collection of functions for scientific computing and advance mathematics

print("SciPy version: {}". format(sp.__version__)) 



import IPython

from IPython import display #pretty printing of dataframes in Jupyter notebook

print("IPython version: {}". format(IPython.__version__)) 



import sklearn #collection of machine learning algorithms

print("scikit-learn version: {}". format(sklearn.__version__))



#misc libraries

import random

import time





#ignore warnings

import warnings

warnings.filterwarnings('ignore')

from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=ConvergenceWarning)



print('-'*25)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



# Evaluation

from sklearn.metrics import cohen_kappa_score,make_scorer

from sklearn.model_selection import StratifiedKFold



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix

from wordcloud import WordCloud

from matplotlib.colors import ListedColormap



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('darkgrid')

pylab.rcParams['figure.figsize'] = 12,8
breeds = pd.read_csv('../input/breed_labels.csv')

colors = pd.read_csv('../input/color_labels.csv')

states = pd.read_csv('../input/state_labels.csv')



train = pd.read_csv('../input/train/train.csv')

test = pd.read_csv('../input/test/test.csv')



train['dataset_type'] = 'train'

test['dataset_type'] = 'test'
train['Type'] = train['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')

test['Type'] = test['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')
color2 = ["#ffa600","#003f5c"]

color3 = ["#ffa600","#bc5090","#003f5c"]

color4 = ["#ffa600","#ef5675","#7a5195","#003f5c"]

color5 = ["#ffa600","#ff6361","#bc5090","#58508d","#003f5c"]

color6 = ["#ffa600","#ff6e54","#dd5182","#955196","#444e86","#003f5c"]

color7 = ["#ffa600","#ff764a","#ef5675","#bc5090","#7a5195","#374c80","#003f5c"]
sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='Type', palette=color2);

plt.title('Number of cats and dogs in train and test data');
g = sns.countplot(x='AdoptionSpeed', hue='Type', data=train, palette=color2)

plt.title('Adoption speed classes rates');

ax=g.axes

for p in ax.patches:

     ax.annotate(f"{p.get_height() * 100 / train.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),

         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),

         textcoords='offset points')  

cats = train.loc[train['Type'] == 'Cat']



g = sns.countplot(x='AdoptionSpeed', data=cats, palette=color5)

plt.title('Adoption speed for cats');

ax=g.axes

for p in ax.patches:

      ax.annotate(f"{p.get_height() * 100 / cats.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),

         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),

         textcoords='offset points')  
dogs = train.loc[train['Type'] == 'Dog']



g = sns.countplot(x='AdoptionSpeed', data=dogs, palette=color5)

plt.title('Adoption speed for dogs');

ax=g.axes

for p in ax.patches:

      ax.annotate(f"{p.get_height() * 100 / dogs.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),

         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),

         textcoords='offset points')  
# copy from: https://www.kaggle.com/artgor/exploration-of-data-step-by-step

main_count = train['AdoptionSpeed'].value_counts(normalize=True).sort_index()



def prepare_plot_dict(df, col, main_count):

    main_count = dict(main_count)

    plot_dict = {}

    for i in df[col].unique():

        val_count = dict(df.loc[df[col] == i, 'AdoptionSpeed'].value_counts().sort_index())

        for k, v in main_count.items():

            if k in val_count:

                plot_dict[val_count[k]] = ((val_count[k] / sum(val_count.values())) / main_count[k]) * 100 - 100

            else:

                plot_dict[0] = 0

    return plot_dict



def make_count_plot(df, x, hue='AdoptionSpeed', title='', main_count=main_count):

    """

    Plotting countplot with correct annotations.

    """

    g = sns.countplot(x=x, data=df, hue=hue, palette=color5);

    plt.title(f'AdoptionSpeed {title}');

    plt.legend(["1st day", "1st week", "1st month", "2nd & 3rd month", "never"]);

    ax = g.axes

   

    plot_dict = prepare_plot_dict(df, x, main_count)



    for p in ax.patches:

        h = p.get_height() if str(p.get_height()) != 'nan' else 0

        text = f"{plot_dict[h]:.0f}%" if plot_dict[h] < 0 else f"+{plot_dict[h]:.0f}%"

        ax.annotate(text, (p.get_x() + p.get_width() / 2., h),

             ha='center', va='center', fontsize=11, color='green' if plot_dict[h] > 0 else 'red', rotation=0, xytext=(0, 10),

             textcoords='offset points')  



make_count_plot(df=train, x='Type', title='by pet Type')
def show_adaptionspeed_barplot(df, compare_column, column_names=None, label=None):

    

    # prepare columns to show

    unique_column_names = sorted(df[compare_column].unique())

    index = list(range(0,len(unique_column_names)))

    if column_names is None:

        column_names = unique_column_names

    

    # calculate % for all AdoptionSpeeds

    df = df.groupby([compare_column, 'AdoptionSpeed']).size().reset_index().pivot(columns='AdoptionSpeed', index=compare_column, values=0)

    totals = [i+j+k+l+m for i,j,k,l,m in zip(df[0], df[1], df[2], df[3], df[4])]

    speed0 = [i / j * 100 for i,j in zip(df[0], totals)]

    speed1 = [i / j * 100 for i,j in zip(df[1], totals)]

    speed2 = [i / j * 100 for i,j in zip(df[2], totals)]

    speed3 = [i / j * 100 for i,j in zip(df[3], totals)]

    speed4 = [i / j * 100 for i,j in zip(df[4], totals)]



    # plot

    barWidth = 0.85

    plt.bar(index, speed0, color='#ffa600', edgecolor='white', width=barWidth, label="1st day")

    plt.bar(index, speed1, bottom=speed0, color='#ff6361', edgecolor='white', width=barWidth, label="1st week")

    plt.bar(index, speed2, bottom=[i+j for i,j in zip(speed0, speed1)], color='#bc5090', edgecolor='white', width=barWidth, label="1st month")

    plt.bar(index, speed3, bottom=[i+j+k for i,j,k in zip(speed0, speed1, speed2)], color='#58508d', edgecolor='white', width=barWidth, label="2nd & 3rd month")

    plt.bar(index, speed4, bottom=[i+j+k+l for i,j,k,l in zip(speed0, speed1, speed2, speed3)], color='#003f5c', edgecolor='white', width=barWidth, label="never")



    # Custom x axis     

    plt.xticks(index,column_names)

    plt.xlabel(label)

    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)



    # Show graphic

    plt.show()

     
show_adaptionspeed_barplot(train, 'Type', ['Dog','Cat'], 'AdoptionSpeed by Type')
print("Missing names in train set: %d" % pd.isna(train['Name']).sum())

print("Missing names in test set: %d" % pd.isna(test['Name']).sum())
train['Name'] = train['Name'].fillna('Unnamed')

test['Name'] = test['Name'].fillna('Unnamed')



train['has_name'] = train['Name'].apply(lambda x: 0 if x == 'Unnamed' else 1)

test['has_name'] = test['Name'].apply(lambda x: 0 if x == 'Unnamed' else 1)
pd.crosstab(train['has_name'], train['AdoptionSpeed'], normalize='index')
make_count_plot(df=train, x='has_name', title='by name available')
train.Name.value_counts().head(10)
train['has_name'] = train['Name'].apply(lambda x: 0 if x == 'No Name' or x == 'Unnamed' else 1)

test['has_name'] = test['Name'].apply(lambda x: 0 if x == 'No Name' or x == 'Unnamed' else 1)
make_count_plot(df=train, x='has_name', title='by name available')
show_adaptionspeed_barplot(train, 'has_name', ['No','Yes'], 'AdoptionSpeed by has_name')
# defining a function which returns a list of top names

def top_names(df, top_percent):

    df_withnames = df[df.has_name != 0]

    items = df_withnames.shape[0]

    top_names = []

    counter = 0

    for i,v in df_withnames.Name.value_counts().items():

        if (counter/items)>top_percent:

            break

        top_names.append(i)

        counter = counter + v  

    return top_names
top_names(train, 0.05)
topnames = top_names(train, 0.2)

train['has_topname'] = train['Name'].apply(lambda row: 1 if row in topnames else 0)

make_count_plot(df=train, x='has_topname', title='by topname')
show_adaptionspeed_barplot(train, 'has_topname', ['No','Yes'], 'AdoptionSpeed by has_topname')
print("Missing Age in train set: %d" % pd.isna(train['Age']).sum())

print("Missing Age in test set: %d" % pd.isna(test['Age']).sum())
fig, ax = plt.subplots(figsize = (16, 6))

plt.subplot(1, 2, 1)

plt.title('Distribution of pets age')

train['Age'].plot('hist', label='train',colormap=ListedColormap(color2[0]))

test['Age'].plot('hist', label='test',colormap=ListedColormap(color2[1]))

plt.legend();



plt.subplot(1, 2, 2)

plt.title('Distribution of pets age (log)')

np.log1p(train['Age']).plot('hist', label='train', colormap=ListedColormap(color2[0]))

np.log1p(test['Age']).plot('hist', label='test', colormap=ListedColormap(color2[1]))

plt.legend();
sns.distplot(train["Age"], kde=True, color=color2[0])
print("pets with age 0: %d" % len(train[train.Age ==0]))

print("pets from a group: %d" % len(train[train.Gender==3]))

print("pets with age 0 and in a group: %d" % len(train[(train.Gender==3) & (train.Age==0)]))
train[(train.Gender==3) & (train.Age==0)].head()
print("Missing Breed1 in train set: %d" % (pd.isna(train['Breed1']).sum() + len(train[train.Breed1 == 0])))

print("Missing Breed1 in test set: %d" %  (pd.isna(test['Breed1']).sum() + len(test[test.Breed1 == 0])))



print("Missing Breed2 in train set: %d" %  (pd.isna(train['Breed2']).sum() + len(train[train.Breed2 == 0])))

print("Missing Breed2 in test set: %d" %  (pd.isna(test['Breed2']).sum() + len(test[test.Breed2 == 0])))
breeds_dict = {k: v for k, v in zip(breeds['BreedID'], breeds['BreedName'])}



train['Breed1_name'] = train['Breed1'].apply(lambda x: breeds_dict[x] if x in breeds_dict else 'Unknown')

train['Breed2_name'] = train['Breed2'].apply(lambda x: breeds_dict[x] if x in breeds_dict else '')



test['Breed1_name'] = test['Breed1'].apply(lambda x: breeds_dict[x] if x in breeds_dict else 'Unknown')

test['Breed2_name'] = test['Breed2'].apply(lambda x: breeds_dict[x] if x in breeds_dict else '')



train[['Breed1_name', 'Breed2_name']].sample(10)
fig, ax = plt.subplots(figsize = (20, 18))

plt.subplot(2, 2, 1)

text_cat1 = ' '.join(train.loc[train['Type'] == 'Cat', 'Breed1_name'].fillna('').values)

wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,

                      width=1200, height=1000).generate(text_cat1)

plt.imshow(wordcloud)

plt.title('Top cat breed1')

plt.axis("off")



plt.subplot(2, 2, 2)

text_dog1 = ' '.join(train.loc[train['Type'] == 'Dog', 'Breed1_name'].fillna('').values)

wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,

                      width=1200, height=1000).generate(text_dog1)

plt.imshow(wordcloud)

plt.title('Top dog breed1')

plt.axis("off")



plt.subplot(2, 2, 3)

text_cat2 = ' '.join(train.loc[train['Type'] == 'Cat', 'Breed2_name'].fillna('').values)

wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,

                      width=1200, height=1000).generate(text_cat2)

plt.imshow(wordcloud)

plt.title('Top cat breed2')

plt.axis("off")



plt.subplot(2, 2, 4)

text_dog2 = ' '.join(train.loc[train['Type'] == 'Dog', 'Breed2_name'].fillna('').values)

wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,

                      width=1200, height=1000).generate(text_dog2)

plt.imshow(wordcloud)

plt.title('Top dog breed2')

plt.axis("off")

plt.show()
def mixed_breed(row):

    if row['Breed1'] == 307:

        return 1

    elif row['Breed2'] == 0:

        return 0 

    elif row['Breed2'] != row['Breed1']:

        return 1

    else:

        return 0



train['mixed_breed'] = train.apply(mixed_breed, axis=1)

test['mixed_breed'] = test.apply(mixed_breed, axis=1)
make_count_plot(df=train, x='mixed_breed', title='by mixed_breed')
show_adaptionspeed_barplot(train, 'mixed_breed', ['No','Yes'], 'AdoptionSpeed by mixed_breed')
sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='mixed_breed', palette=color2);

plt.title('Mixed breeds in train and test data');
plt.figure(figsize=(18, 6));

plt.subplot(1, 2, 1)

make_count_plot(df=train, x='Gender', title='by gender')



plt.subplot(1,2,2)

show_adaptionspeed_barplot(train, 'Gender', ['Male','Female', 'Group'], 'AdoptionSpeed by gender')
sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='Gender', palette=color3);

plt.title('Number of pets by gender in train and test data');
colors_dict = {k: v for k, v in zip(colors['ColorID'], colors['ColorName'])}

train['Color1_name'] = train['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

train['Color2_name'] = train['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

train['Color3_name'] = train['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')



test['Color1_name'] = test['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

test['Color2_name'] = test['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

test['Color3_name'] = test['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
print("Missing Color1 in train set: %d" % (pd.isna(train['Color1']).sum() + len(train[train.Color1 == 0])))

print("Missing Color1 in test set: %d" %  (pd.isna(test['Color1']).sum() + len(test[test.Color1 == 0])))



print("Missing Color2 in train set: %d" %  (pd.isna(train['Color2']).sum() + len(train[train.Color2 == 0])))

print("Missing Color2 in test set: %d" %  (pd.isna(test['Color2']).sum() + len(test[test.Color2 == 0])))



print("Missing Color3 in train set: %d" %  (pd.isna(train['Color3']).sum() + len(train[train.Color3 == 0])))

print("Missing Color3 in test set: %d" %  (pd.isna(test['Color3']).sum() + len(test[test.Color3 == 0])))
plt.figure(figsize=(18, 6));

plt.subplot(1, 2, 1)

make_count_plot(df=train, x='Color1', title='by main color')



plt.subplot(1,2,2)

show_adaptionspeed_barplot(train, 'Color1', ['Black','Brown','Golden','Yellow','Cream','Gray','White'], 'AdoptionSpeed by main color')
sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='Color1', palette=color7);

plt.title('Number of pets by gender in train and test data');
def number_of_colors(row):

    if row['Color1'] == 0:

        return 0

    elif (row['Color2'] != 0 and row['Color3'] == 0):

        return 2

    elif (row['Color2'] != 0 and row['Color3'] != 0):

        return 3

    else:

        return 1



train['number_of_colors'] = train.apply(number_of_colors, axis=1)

test['number_of_colors'] = test.apply(number_of_colors, axis=1)
plt.figure(figsize=(18, 6));

plt.subplot(1, 2, 1)

make_count_plot(df=train, x='number_of_colors', title='by number of colors')



plt.subplot(1,2,2)

show_adaptionspeed_barplot(train, 'number_of_colors', ['One','Two', 'Three'], 'AdoptionSpeed by number of colors')
print("Missing MaturitySize in train set: %d" % (pd.isna(train['MaturitySize']).sum() + len(train[train.MaturitySize == 0])))

print("Missing MaturitySize in test set: %d" %  (pd.isna(test['MaturitySize']).sum() + len(test[test.MaturitySize == 0])))

print("Unique values of MaturitySize in train set: %s" %  train.MaturitySize.unique())
train.MaturitySize.replace([4],[3], inplace=True)
plt.figure(figsize=(18, 6));

plt.subplot(1, 2, 1)

make_count_plot(df=train, x='MaturitySize', title='by MaturitySize')



plt.subplot(1,2,2)

show_adaptionspeed_barplot(train, 'MaturitySize', ['Small','Medium','Large','Extra Large'], 'AdoptionSpeed by MaturitySize')
sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='MaturitySize', palette=color4);

plt.title('Number of pets by MaturitySize in train and test data');
print("Missing FurLength in train set: %d" % (pd.isna(train['FurLength']).sum() + len(train[train.FurLength == 0])))

print("Missing FurLength in test set: %d" %  (pd.isna(test['FurLength']).sum() + len(test[test.FurLength == 0])))
plt.figure(figsize=(18, 6));

plt.subplot(1, 2, 1)

make_count_plot(df=train, x='FurLength', title='by FurLength')



plt.subplot(1,2,2)

show_adaptionspeed_barplot(train, 'FurLength', ['Short','Medium','Long'], 'AdoptionSpeed by FurLength')
sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='FurLength', palette=color3);

plt.title('Number of pets by FurLength in train and test data');
wrong_hair_length = []

all_data = pd.concat([train, test])

for i, row in all_data[(all_data.Breed1_name.str.contains('Hair')) | (all_data.Breed2_name.str.contains("Hair"))].iterrows():

    if ('Short' in row['Breed1_name'] or 'Short' in row['Breed2_name']) and row['FurLength'] == 1:

        continue

    if ('Medium' in row['Breed1_name'] or 'Medium' in row['Breed2_name']) and row['FurLength'] == 2:

        continue

    if ('Long' in row['Breed1_name'] or 'Long' in row['Breed2_name']) and row['FurLength'] == 3:

        continue

    wrong_hair_length.append((row['PetID'], row['Breed1_name'], row['Breed2_name'], row['FurLength'], row['dataset_type']))



wrong_df = pd.DataFrame(wrong_hair_length)

print(f"There are {len(wrong_df[wrong_df[4] == 'train'])} pets whose breed and fur length don't match in train")

print(f"There are {len(wrong_df[wrong_df[4] == 'test'])} pets whose breed and fur length don't match in test")

wrong_df.sample(8)
print("Missing Vaccinated in train set: %d" % pd.isna(train['Vaccinated']).sum())

print("Missing Vaccinated in test set: %d" %  pd.isna(test['Vaccinated']).sum())

      

print("Missing Dewormed in train set: %d" % pd.isna(train['Dewormed']).sum())

print("Missing Dewormed in test set: %d" %  pd.isna(test['Dewormed']).sum())

      

print("Missing Sterilized in train set: %d" % pd.isna(train['Sterilized']).sum())

print("Missing Sterilized in test set: %d" %  pd.isna(test['Sterilized']).sum())

      

print("Missing Health in train set: %d" % pd.isna(train['Health']).sum())

print("Missing Health in test set: %d" %  pd.isna(test['Health']).sum())
plt.figure(figsize=(24, 16));

plt.subplot(2, 2, 1)

make_count_plot(df=train, x='Vaccinated', title='by Vaccinated')



plt.subplot(2,2,2)

show_adaptionspeed_barplot(train, 'Vaccinated', ['Yes','No','Not sure'], 'AdoptionSpeed by Vaccinated')



plt.figure(figsize=(24, 16));

plt.subplot(2,2,3)

sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='Vaccinated', palette=color3);

plt.title('Number of pets by Vaccinated in train and test data');
plt.figure(figsize=(24, 16));

plt.subplot(2, 2, 1)

make_count_plot(df=train, x='Dewormed', title='by Dewormed')



plt.subplot(2,2,2)

show_adaptionspeed_barplot(train, 'Dewormed', ['Yes','No','Not sure'], 'AdoptionSpeed by Dewormed')



plt.figure(figsize=(24, 16));

plt.subplot(2,2,3)

sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='Dewormed', palette=color3);

plt.title('Number of pets by Dewormed in train and test data');
plt.figure(figsize=(24, 16));

plt.subplot(2, 2, 1)

make_count_plot(df=train, x='Sterilized', title='by Sterilized')



plt.subplot(2,2,2)

show_adaptionspeed_barplot(train, 'Sterilized', ['Yes','No','Not sure'], 'AdoptionSpeed by Sterilized')



plt.figure(figsize=(24, 16));

plt.subplot(2,2,3)

sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='Sterilized', palette=color3);

plt.title('Number of pets by Sterilized in train and test data');
print("Healthy in train set: %d" % (len(train[train.Health == 1])))

print("Minor Injury in test set: %d" %  (len(test[test.Health == 2])))

print("Serious Injury in test set: %d" %  (len(test[test.Health == 3])))

print("Not Specified in test set: %d" %  (len(test[test.Health == 0])))



plt.figure(figsize=(24, 16));

plt.subplot(2, 2, 1)

make_count_plot(df=train, x='Health', title='by Health')



plt.subplot(2,2,2)

show_adaptionspeed_barplot(train, 'Health',['Healthy','Minor Injury','Serious Injury'], 'AdoptionSpeed by Health')



plt.figure(figsize=(24, 16));

plt.subplot(2,2,3)

sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='Health', palette=color3);

plt.title('Number of pets by Health in train and test data');
train['Quantity'].value_counts()
test['Quantity'].value_counts()
train['is_group'] = train['Quantity'].apply(lambda x: True if x > 1 else False)

test['is_group'] = test['Quantity'].apply(lambda x: True if x > 1 else False)
plt.figure(figsize=(24, 16));

plt.subplot(2, 2, 1)

make_count_plot(df=train, x='is_group', title='by is_group')



plt.subplot(2,2,2)

show_adaptionspeed_barplot(train, 'is_group',['No','Yes'], 'AdoptionSpeed by is_group')



plt.figure(figsize=(24, 16));

plt.subplot(2,2,3)

sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='is_group', palette=color2);

plt.title('Number of pets by is_group in train and test data');
fig, ax = plt.subplots(figsize = (16, 6))

plt.subplot(1, 2, 1)

plt.title('Distribution of pets fee')

train['Fee'].plot('hist', label='train',colormap=ListedColormap(color2[0]))

test['Fee'].plot('hist', label='test',colormap=ListedColormap(color2[1]))

plt.legend();



plt.subplot(1, 2, 2)

plt.title('Distribution of pets fee (log)')

np.log1p(train['Fee']).plot('hist', label='train', colormap=ListedColormap(color2[0]))

np.log1p(test['Fee']).plot('hist', label='test', colormap=ListedColormap(color2[1]))

plt.legend();
train['is_free'] = train['Fee'].apply(lambda x: True if x == 0 else False)

test['is_free'] = test['Fee'].apply(lambda x: True if x == 0 else False)
plt.figure(figsize=(24, 16));

plt.subplot(2, 2, 1)

make_count_plot(df=train, x='is_free', title='by is_free')



plt.subplot(2,2,2)

show_adaptionspeed_barplot(train, 'is_free',label='AdoptionSpeed by is_free')



plt.figure(figsize=(24, 16));

plt.subplot(2,2,3)

sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='is_free', palette=color2);

plt.title('Number of pets by is_free in train and test data');
states_dict = {k: v for k, v in zip(states['StateID'], states['StateName'])}

train['state_name'] = train['State'].apply(lambda x: states_dict[x] if x in states_dict else 'Unknown')

test['state_name'] = test['State'].apply(lambda x: states_dict[x] if x in states_dict else 'Unknown')
fig= plt.subplots(figsize=(18,8))

ax = sns.countplot(x="state_name", data=train, order = train["state_name"].value_counts().index,palette=color7)

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 

            fontsize=12, color='grey', ha='center', va='bottom')
print("Number of pets by top rescuer in train set: %d" % train['RescuerID'].value_counts().head(1))

print("Number of pets by top rescuer in test set: %d" %  test['RescuerID'].value_counts().head(1))

print("Unique rescuer in train set: %d" %  len(train['RescuerID'].unique()))

print("Unique rescuer in test set: %d" %  len(test['RescuerID'].unique()))

print("Rescuer from train also in test set: %d" %  train.RescuerID.isin(test.RescuerID).sum())

print("Rescuer from test also in train set: %d" %  test.RescuerID.isin(train.RescuerID).sum())
train['VideoAmt'].value_counts()
train['has_video'] = train['VideoAmt'].apply(lambda x: True if x > 0 else False)

test['has_video'] = test['VideoAmt'].apply(lambda x: True if x > 0 else False)



plt.figure(figsize=(24, 16));

plt.subplot(2, 2, 1)

make_count_plot(df=train, x='has_video', title='by has_video')



plt.subplot(2,2,2)

show_adaptionspeed_barplot(train, 'has_video',label='AdoptionSpeed by has_video')



plt.figure(figsize=(24, 16));

plt.subplot(2,2,3)

sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='has_video', palette=color2);

plt.title('Number of pets by has_video in train and test data');
fig, ax = plt.subplots(figsize = (16, 6))

plt.subplot(1, 2, 1)

plt.title('Distribution of PhotoAmt')

train['PhotoAmt'].plot('hist', label='train',colormap=ListedColormap(color2[0]))

test['PhotoAmt'].plot('hist', label='test',colormap=ListedColormap(color2[1]))

plt.legend();
train['has_photo'] = train['PhotoAmt'].apply(lambda x: True if x > 0 else False)

test['has_photo'] = test['PhotoAmt'].apply(lambda x: True if x > 0 else False)



plt.figure(figsize=(24, 16));

plt.subplot(2, 2, 1)

make_count_plot(df=train, x='has_photo', title='by has_photo')



plt.subplot(2,2,2)

show_adaptionspeed_barplot(train, 'has_photo',label='AdoptionSpeed by has_photo')



plt.figure(figsize=(24, 16));

plt.subplot(2,2,3)

sns.countplot(x='dataset_type', data=pd.concat([train, test]), hue='has_photo', palette=color2);

plt.title('Number of pets by has_photo in train and test data');
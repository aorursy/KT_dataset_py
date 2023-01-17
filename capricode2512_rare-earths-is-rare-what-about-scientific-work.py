# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_jacs=pd.read_json('/kaggle/input/american-chemical-society-journals/jacs.json')
df_chem_revs=pd.read_json('/kaggle/input/american-chemical-society-journals/chem_revs.json')
df_inorg_chem=pd.read_json('/kaggle/input/american-chemical-society-journals/inorg_chem.json')
df_chem_revs.info()
df_jacs.info()
df_inorg_chem.info()

# Preparing single file containing research on Uranium and rare earths from above three datasets

pattern="[Uu]ranium|[Rr]are earth|[Cc]erium|[Dd]ysprosium|[Ee]rbium|[Ee]uropium|[Gg]adolinium|[Hh]olmium|[Ll]anthanum|[Ll]utetium|[Nn]eodymium|[Pp]raseodymium|[Pp]romethium|[Ss]amarium|[Ss]candium|[Tt]erbium|[Tt]hulium|[Yy]tterbium|[Yy]ttrium"
list_df=[df_chem_revs,df_jacs,df_inorg_chem]
dataset=pd.concat(list_df)
dataset_v1=dataset[(dataset["abstract"].str.contains(pattern))|(dataset["title"].str.contains(pattern))]
# Let us explore the information available and whether it is sufficient for our analysis

dataset_v1.info()
dataset_v1["article_type"].value_counts()
bool_book_review=dataset_v1["article_type"]=="Addition/Correction"
dataset_v2=dataset_v1[~bool_book_review]
dataset_v2.reset_index(inplace=True)
dataset_v2.head()
# Splitting the dataset into two categories.
# One containing uranium papers.
# Second containing rare earth papers.

pattern_u=r"[Uu]ranium"
pattern_rare_earth=r"[Rr]are earth|[Cc]erium|[Dd]ysprosium|[Ee]rbium|[Ee]uropium|[Gg]adolinium|[Hh]olmium|[Ll]anthanum|[Ll]utetium|[Nn]eodymium|[Pp]raseodymium|[Pp]romethium|[Ss]amarium|[Ss]candium|[Tt]erbium|[Tt]hulium|[Yy]tterbium|[Yy]ttrium"
dataset_uranium=dataset_v2[(dataset_v2["abstract"].str.contains(pattern_u))|(dataset_v2["title"].str.contains(pattern_u))]
dataset_rare_earth=dataset_v2[(dataset_v2["abstract"].str.contains(pattern_rare_earth))|(dataset_v2["title"].str.contains(pattern_rare_earth))]
dataset_uranium.info()
# Divding years into bins and counting number of papers in each 10 years
import matplotlib.pyplot as plt

ax=dataset_uranium.hist(column="year",bins=72,figsize=(15,5),grid=False,color='blue', alpha=0.4, edgecolor='black')
ax=ax[0]
for x in ax:
    x.set_title("SCIENTIFIC PUBLICATIONS ON URANIUM FROM 1900 TO 2020",size= 20)
    x.set_xlabel("YEAR", labelpad=10, size=15)
    x.set_ylabel("NUMBER OF ARTICLES", labelpad=15, size=15)
    x.tick_params(axis='both', labelsize=13)
    x.set_xticks([1940,1950,1960,1970,1980,1990,2000,2010,2020])
    
ax2=dataset_rare_earth.hist(column="year",bins=72,figsize=(15,5),grid=False,color='GREEN', alpha=0.5, edgecolor='black')
ax2=ax2[0]
for x in ax2:
    x.set_title("SCIENTIFIC PUBLICATIONS ON RARE-EARTH FROM 1900 TO 2020",size= 20)
    x.set_xlabel("YEAR", labelpad=10, size=15)
    x.set_ylabel("NUMBER OF ARTICLES", labelpad=15, size=15)
    x.tick_params(axis='both', labelsize=13)
    x.set_xticks([1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020])

# Creating a function to find length of each article and adding a column to the dataset
def length_of_article(row):
    return row["page_end"]-row["page_start"]

len_article=dataset_uranium.apply(length_of_article,axis=1)
dataset_uranium = dataset_uranium.assign(page_length=len_article.values)
len_article_re=dataset_rare_earth.apply(length_of_article,axis=1)
dataset_rare_earth = dataset_rare_earth.assign(page_length=len_article_re.values)
datasets=[dataset_uranium["page_length"],dataset_rare_earth["page_length"]]
fig =plt.figure(figsize=(5,4))
ax=fig.add_axes([0,0,1,1])
bp=ax.boxplot(datasets, patch_artist=True)
colors=["#e7298a","#e7298a"]
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_ylim(0,30)
ax.set_xticklabels(["Uranium","Rare-earth"], fontsize=15)
ax.set_yticklabels([0,5,10,5,20,25,30], fontsize=15)
ax.set_ylabel("NUMBER OF PAGES", FONTSIZE=15)
plt.title("LENGTH OF ARTICLES", fontsize=20)
plt.show()
# Creating a dictionary of authors and counting their contributions in Uranium
dict_of_authors={}
def make_dict(column):
    
    for name in column:
        
        if name in dict_of_authors:
            dict_of_authors[name]+=1
        else:
            dict_of_authors[name]=1

dataset_uranium["authors"].apply(make_dict)
sorted_authors=sorted(dict_of_authors.items(), key= lambda value: value[1], reverse=True)
print("For uranium:")
print("{} has largest number of papers, that is {}.".format(sorted_authors[0][0],sorted_authors[0][1]))
print("{} has 2nd largest number of papers, that is {}.".format(sorted_authors[1][0],sorted_authors[1][1]))
print("{} has 3rd largest number of papers, that is {}.".format(sorted_authors[2][0],sorted_authors[2][1]))
# Creating a dictionary of authors and counting their contributions in rare earth
dict_of_authors_re={}
def make_dict_re(column):
    
    for name in column:
        
        if name in dict_of_authors_re:
            dict_of_authors_re[name]+=1
        else:
            dict_of_authors_re[name]=1

dataset_rare_earth["authors"].apply(make_dict_re)
sorted_authors_re=sorted(dict_of_authors_re.items(), key= lambda value: value[1], reverse=True)
print("For rare earths:")
print("{} has largest number of papers, that is {}.".format(sorted_authors_re[0][0],sorted_authors_re[0][1]))
print("{} has 2nd largest number of papers, that is {}.".format(sorted_authors_re[1][0],sorted_authors_re[1][1]))
print("{} has 3rd largest number of papers, that is {}.".format(sorted_authors_re[2][0],sorted_authors_re[2][1]))
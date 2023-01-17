# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from skimage import io

from scipy import ndimage

from skimage.transform import resize

import sys

#reload(sys)

#sys.setdefaultencoding("utf-8")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

menu = pd.read_csv('../input/menu.csv')

#Breakfast = io.imread('D:/Mcdonalds/Breakfast.png')

#Beef_and_Pork = io.imread('D:/Mcdonalds/Beef & Pork.png')

#Chicken_and_Fish = io.imread('D:/Mcdonalds/Chicken & Fish.png')

#Coffee_and_Tea = io.imread('D:/Mcdonalds/Coffee & Tea.png')

#Salads = io.imread('D:/Mcdonalds/Salads.png')

#Smoothies_and_Shakes = io.imread('D:/Mcdonalds/Smoothies & Shakes.png')

#Snacks_and_Sides = io.imread('D:/Mcdonalds/Snacks & Sides.png')

#Beverages = io.imread('D:/Mcdonalds/Beverages.png')

#Desserts = io.imread('D:/Mcdonalds/Desserts.png')

#images = [Breakfast,Beef_and_Pork,Chicken_and_Fish,Coffee_and_Tea,Salads,Smoothies_and_Shakes,Snacks_and_Sides,Beverages,Desserts]

x1 = menu['Category'].values

x2 = menu['Item'].values

x3 = menu['Calories'].values

x4 = menu['Calories from Fat'].values

x5 = menu['Trans Fat'].values

x6 = menu['Sugars'].values

x7 = menu['Protein'].values



x8 = menu['Total Fat (% Daily Value)'].values

x9 = menu['Saturated Fat (% Daily Value)'].values

x10 = menu['Cholesterol (% Daily Value)'].values

x11 = menu['Sodium (% Daily Value)'].values

x12 = menu['Carbohydrates (% Daily Value)'].values



x13 = menu['Dietary Fiber (% Daily Value)'].values

x14 = menu['Vitamin A (% Daily Value)'].values

x15 = menu['Vitamin C (% Daily Value)'].values

x16 = menu['Calcium (% Daily Value)'].values

x17 = menu['Iron (% Daily Value)'].values

x = np.vstack((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17))

print(x.shape)

r = 26

p_max = 10

for p in range(p_max):

    plt.figure(figsize=(18.5, r*0.5+2.5),facecolor='#be0c0c')

    plt.subplots_adjust(bottom=0, left=.01, right=1.4, top=0.9, hspace=.35)

    plt.subplot(1, 1, 1)

    plt.title('McDonald\'s Menu - Comparative Nutrition Values - Page '+str(p+1),fontsize=26,fontweight='bold',color='#ffd600')

    plt.ylim([r+0.2,-6])

    plt.xlim([-14,16])



    plt.scatter(-12.5,-3, s=int(100)*3, c='grey', alpha=0.15,zorder=1)

    plt.text(-11.75,-3, 'Daily Limit',verticalalignment='center',fontsize=11,color='black', fontstyle='italic',alpha=1,zorder=1)

    plt.text(-12.75,-2.3, '2000',fontweight='bold',verticalalignment='center',fontsize=11,color='darkorange', fontstyle='italic',alpha=1,zorder=1)

    plt.text(-11.75,-2.3, 'calories a day is used for general nutrition advice,',verticalalignment='center',fontsize=11,color='black', fontstyle='italic',alpha=1,zorder=1)

    plt.text(-11.75,-1.8, 'but calorie needs vary.',verticalalignment='center',fontsize=11,color='black', fontstyle='italic',alpha=1,zorder=1)

    plt.text(0.5,-5, 'Calories',horizontalalignment='right',verticalalignment='top',rotation='vertical',fontsize=13,color='black', alpha=1,zorder=1)

    plt.text(2,-5, 'Calories from Fat',horizontalalignment='right',verticalalignment='top',rotation='vertical',fontsize=13,color='black', alpha=1,zorder=1)

    plt.text(3,-5, 'Trans Fat',horizontalalignment='right',verticalalignment='top',rotation='vertical',fontsize=13,color='black', alpha=1,zorder=1)

    plt.text(4,-5, 'Sugars',horizontalalignment='right',verticalalignment='top',rotation='vertical',fontsize=13,color='black', alpha=1,zorder=1)

    plt.text(5,-5, 'Protein',horizontalalignment='right',verticalalignment='top',rotation='vertical',fontsize=13,color='black', alpha=1,zorder=1)

    plt.text(6,-5, 'Total Fat',horizontalalignment='right',verticalalignment='top',rotation='vertical',fontsize=13,color='black', alpha=1,zorder=1)

    plt.text(7,-5, 'Saturated Fat',horizontalalignment='right',verticalalignment='top',rotation='vertical',fontsize=13,color='black', alpha=1,zorder=1)

    plt.text(8,-5, 'Cholesterol',horizontalalignment='right',verticalalignment='top',rotation='vertical',fontsize=13,color='black', alpha=1,zorder=1)

    plt.text(9,-5, 'Sodium',horizontalalignment='right',verticalalignment='top',rotation='vertical',fontsize=13,color='black', alpha=1,zorder=1)

    plt.text(10,-5, 'Carbohydrates',horizontalalignment='right',verticalalignment='top',rotation='vertical',fontsize=13,color='black', alpha=1,zorder=1)

    plt.text(11,-5, 'Dietary Fiber',horizontalalignment='right',verticalalignment='top',rotation='vertical',fontsize=13,color='black', alpha=1,zorder=1)

    plt.text(12,-5, 'Vitamin A',horizontalalignment='right',verticalalignment='top',rotation='vertical',fontsize=13,color='black', alpha=1,zorder=1)

    plt.text(13,-5, 'Vitamin C',horizontalalignment='right',verticalalignment='top',rotation='vertical',fontsize=13,color='black', alpha=1,zorder=1)

    plt.text(14,-5, 'Calcium',horizontalalignment='right',verticalalignment='top',rotation='vertical',fontsize=13,color='black', alpha=1,zorder=1)

    plt.text(15,-5, 'Iron',horizontalalignment='right',verticalalignment='top',rotation='vertical',fontsize=13,color='black', alpha=1,zorder=1)

    j = 0

    for i in range(p*r,p*r+r):

        if x1[i] == 'Breakfast':

            k = 0

        if x1[i] == 'Beef & Pork':

            k = 1

        if x1[i] == 'Chicken & Fish':

            k = 2

        if x1[i] == 'Coffee & Tea':

            k = 3

        if x1[i] == 'Salads':

            k = 4

        if x1[i] == 'Smoothies & Shakes':

            k = 5

        if x1[i] == 'Snacks & Sides':

            k = 6  

        if x1[i] == 'Beverages':

            k = 7  

        if x1[i] == 'Desserts':

            k = 8  

        #plt.imshow(ndimage.rotate(images[k], 180), extent=[-13.5,-12.75,j-0.5,j+0.25],aspect= 1,cmap=plt.cm.gray,interpolation = 'none',zorder=0)

        plt.text(-12.25, j, str(x2[i]), fontsize=13,color='black', alpha=1,zorder=1)

        if x3[i] > 0:

            plt.text(0.5,j, x3[i], horizontalalignment='right',fontsize=13,color='darkorange', alpha=1,zorder=1)

        else:

            plt.text(0.5, j, '-', horizontalalignment='center',fontsize=13,color='darkorange', alpha=1,zorder=1)

        if x4[i] > 0:

            plt.text(2,j, x4[i], horizontalalignment='right',fontsize=13,color='sienna', alpha=1,zorder=1)

        else:

            plt.text(2, j, '-', horizontalalignment='center',fontsize=13,color='sienna', alpha=1,zorder=1)

        if x5[i] > 0:

            plt.text(3, j, x5[i], horizontalalignment='right',fontsize=13,color='crimson', alpha=1,zorder=1)

        else:

            plt.text(3, j, '-', horizontalalignment='center',fontsize=13,color='crimson', alpha=1,zorder=1)

        if x6[i] > 0:

            plt.text(4, j, x6[i], horizontalalignment='right',fontsize=13,color='sienna', alpha=1,zorder=1)

        else:

            plt.text(4, j, '-', horizontalalignment='center',fontsize=13,color='sienna', alpha=1,zorder=1)

        if x7[i] > 0:

            plt.text(5, j, x7[i], horizontalalignment='right',fontsize=13,color='sienna', alpha=1,zorder=1)

        else:

            plt.text(5, j, '-', horizontalalignment='center',fontsize=13,color='sienna', alpha=1,zorder=1)

       

        plt.scatter(6,j, s=int(100)*3, c='grey', alpha=0.15,zorder=1)

        plt.scatter(7,j, s=int(100)*3, c='grey', alpha=0.15,zorder=1)

        plt.scatter(8,j, s=int(100)*3, c='grey', alpha=0.15,zorder=1)

        plt.scatter(9,j, s=int(100)*3, c='grey', alpha=0.15,zorder=1)

        plt.scatter(10,j, s=int(100)*3, c='grey', alpha=0.15,zorder=1)

    

        plt.scatter(11,j, s=int(100)*3, c='grey', alpha=0.15,zorder=1)

        plt.scatter(12,j, s=int(100)*3, c='grey', alpha=0.15,zorder=1)

        plt.scatter(13,j, s=int(100)*3, c='grey', alpha=0.15,zorder=1)

        plt.scatter(14,j, s=int(100)*3, c='grey', alpha=0.15,zorder=1)

        plt.scatter(15,j, s=int(100)*3, c='grey', alpha=0.15,zorder=1)

        

        plt.scatter(6,j, s=int(x8[i])*3, c='crimson', alpha=0.8,zorder=2)

        plt.scatter(7,j, s=int(x9[i])*3, c='crimson', alpha=0.8,zorder=2)

        plt.scatter(8,j, s=int(x10[i])*3, c='crimson', alpha=0.8,zorder=2)

        plt.scatter(9,j, s=int(x11[i])*3, c='crimson', alpha=0.8,zorder=2)

        plt.scatter(10,j, s=int(x12[i])*3, c='crimson', alpha=0.8,zorder=2)

    

        plt.scatter(11,j, s=int(x13[i])*3, c='seagreen', alpha=0.8,zorder=2)

        plt.scatter(12,j, s=int(x14[i])*3, c='seagreen', alpha=0.8,zorder=2)

        plt.scatter(13,j, s=int(x15[i])*3, c='seagreen', alpha=0.8,zorder=2)

        plt.scatter(14,j, s=int(x16[i])*3, c='seagreen', alpha=0.8,zorder=2)

        plt.scatter(15,j, s=int(x17[i])*3, c='seagreen', alpha=0.8,zorder=2)

        j = j + 1

    plt.xticks([])

    plt.yticks([])

    plt.legend()

    plt.savefig('McDonald_Nutrition_'+str(p+1)+'.png',bbox_inches='tight',facecolor='#be0c0c',dpi=200)

    plt.show()

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Correlation Plot

from matplotlib import cm as cm



import matplotlib as mpl

%pylab inline

label_size = 18

mpl.rcParams['xtick.labelsize'] = label_size

mpl.rcParams['ytick.labelsize'] = label_size



sns.set_style("whitegrid")

mask = np.zeros_like(menu[menu.columns[3:]].corr(), dtype = np.bool)

mask[np.triu_indices_from(mask)] = True



plt.figure(figsize = (22,18))



cmap = sns.diverging_palette(10,220, as_cmap = True)  #10,133



#Draw

sns.heatmap(menu[menu.columns[3:]].corr(), mask = mask, vmax = 1, vmin = -.45, center = 0,cmap=cmap,

           annot = True, square = True, linewidth = .5, cbar_kws = {'shrink':.5}, fmt= '.2f')

plt.text(8, 1, 'Courtesy: https://seaborn.pydata.org/examples/many_pairwise_correlations.html', fontsize=16,alpha=0.2)

plt.title("Nutrition Value Correlation", loc = 'center', size = 32, color = '#34495E')

#plt.savefig('Correlation Matrix'+'.png', bbox_inches = 'tight')
sns.pairplot(menu[['Category','Calories', 'Total Fat (% Daily Value)','Carbohydrates (% Daily Value)', 'Dietary Fiber (% Daily Value)']],hue='Category')
sns.set_style("whitegrid")

plt.figure(figsize=(22,10))

#plt.figure()



ax = sns.boxenplot(x="Category", y='Calories', data=menu, color='#eeeeee', palette="tab10")



# Add transparency to colors

for patch in ax.artists:

    r, g, b, a = patch.get_facecolor()

    patch.set_facecolor((r, g, b, .9))

    

#ax = sns.stripplot(x='Category', y='Cholesterol (% Daily Value)', data=menu, color="orange", jitter=0.5, size=5,alpha=0.15)

#

plt.title("Calories  \n", loc="center",size=32,color='#be0c0c',alpha=0.6)

plt.xlabel('Category',color='#34495E',fontsize=20) 

plt.ylabel('Calories',color='#34495E',fontsize=20)

plt.xticks(size=16,color='#008abc',rotation='horizontal', wrap=True)  

plt.yticks(size=15,color='#006600')

#plt.text(2.5, 1, 'Courtesy: https://seaborn.pydata.org/examples/grouped_boxplot.html', fontsize=13,alpha=0.2)

#plt.ylim(0,200)

#plt.legend(loc="upper right",fontsize=14,ncol=5,title='Category',title_fontsize=22,framealpha=0.99)

plt.show()
sns.set_style("whitegrid")

plt.figure(figsize=(22,10))

#plt.figure()



ax = sns.boxenplot(x="Category", y='Total Fat (% Daily Value)', data=menu, color='#eeeeee', palette="tab10")



# Add transparency to colors

for patch in ax.artists:

    r, g, b, a = patch.get_facecolor()

    patch.set_facecolor((r, g, b, .9))

    

#ax = sns.stripplot(x='Category', y='Cholesterol (% Daily Value)', data=menu, color="orange", jitter=0.5, size=5,alpha=0.15)

#

plt.title("Total Fat (% Daily Value)  \n", loc="center",size=32,color='#be0c0c',alpha=0.6)

plt.xlabel('Category',color='#34495E',fontsize=20) 

plt.ylabel('Total Fat (% Daily Value)',color='#34495E',fontsize=20)

plt.xticks(size=16,color='#008abc',rotation='horizontal', wrap=True)  

plt.yticks(size=15,color='#006600')

#plt.text(2.5, 1, 'Courtesy: https://seaborn.pydata.org/examples/grouped_boxplot.html', fontsize=13,alpha=0.2)

#plt.ylim(0,200)

#plt.legend(loc="upper right",fontsize=14,ncol=5,title='Category',title_fontsize=22,framealpha=0.99)

plt.show()
sns.set_style("whitegrid")

plt.figure(figsize=(22,10))

#plt.figure()



ax = sns.boxenplot(x="Category", y='Saturated Fat (% Daily Value)', data=menu, color='#eeeeee', palette="tab20")



# Add transparency to colors

for patch in ax.artists:

    r, g, b, a = patch.get_facecolor()

    patch.set_facecolor((r, g, b, .9))

    

#ax = sns.stripplot(x='Category', y='Cholesterol (% Daily Value)', data=menu, color="orange", jitter=0.5, size=5,alpha=0.15)

#

plt.title("Saturated Fat (% Daily Value)  \n", loc="center",size=32,color='#be0c0c',alpha=0.6)

plt.xlabel('Category',color='#34495E',fontsize=20) 

plt.ylabel('Saturated Fat (% Daily Value)',color='#34495E',fontsize=20)

plt.xticks(size=16,color='#008abc',rotation='horizontal', wrap=True)  

plt.yticks(size=15,color='#006600')

#plt.text(2.5, 1, 'Courtesy: https://seaborn.pydata.org/examples/grouped_boxplot.html', fontsize=13,alpha=0.2)

#plt.ylim(0,200)

#plt.legend(loc="upper right",fontsize=10,ncol=5,title='Category',title_fontsize=14,framealpha=0.99)

plt.show()
sns.set_style("whitegrid")

plt.figure(figsize=(22,10))

#plt.figure()



ax = sns.boxenplot(x="Category", y='Cholesterol (% Daily Value)', data=menu, color='#eeeeee', palette="tab10")



# Add transparency to colors

for patch in ax.artists:

    r, g, b, a = patch.get_facecolor()

    patch.set_facecolor((r, g, b, .9))

    

#ax = sns.stripplot(x='Category', y='Cholesterol (% Daily Value)', data=menu, color="orange", jitter=0.5, size=5,alpha=0.15)

#

plt.title("Cholesterol (% Daily Value)  \n", loc="center",size=32,color='#be0c0c',alpha=0.6)

plt.xlabel('Category',color='#34495E',fontsize=20) 

plt.ylabel('Cholesterol (% Daily Value)',color='#34495E',fontsize=20)

plt.xticks(size=16,color='#008abc',rotation='horizontal', wrap=True)  

plt.yticks(size=15,color='#006600')

#plt.text(2.5, 1, 'Courtesy: https://seaborn.pydata.org/examples/grouped_boxplot.html', fontsize=13,alpha=0.2)

#plt.ylim(0,200)

#plt.legend(loc="upper right",fontsize=14,ncol=5,title='Category',title_fontsize=22,framealpha=0.99)

plt.show()
sns.set_style("whitegrid")

plt.figure(figsize=(22,10))

#plt.figure()



ax = sns.boxenplot(x="Category", y='Carbohydrates (% Daily Value)', data=menu, color='#eeeeee', palette="tab10")



# Add transparency to colors

for patch in ax.artists:

    r, g, b, a = patch.get_facecolor()

    patch.set_facecolor((r, g, b, .9))

    

#ax = sns.stripplot(x='Category', y='Cholesterol (% Daily Value)', data=menu, color="orange", jitter=0.5, size=5,alpha=0.15)

#

plt.title("Carbohydrates (% Daily Value)  \n", loc="center",size=32,color='#be0c0c',alpha=0.6)

plt.xlabel('Category',color='#34495E',fontsize=20) 

plt.ylabel('Carbohydrates (% Daily Value)',color='#34495E',fontsize=20)

plt.xticks(size=16,color='#008abc',rotation='horizontal', wrap=True)  

plt.yticks(size=15,color='#006600')

#plt.text(2.5, 1, 'Courtesy: https://seaborn.pydata.org/examples/grouped_boxplot.html', fontsize=13,alpha=0.2)

#plt.ylim(0,200)

#plt.legend(loc="upper right",fontsize=14,ncol=5,title='Category',title_fontsize=22,framealpha=0.99)

plt.show()
sns.set_style("whitegrid")

plt.figure(figsize=(22,10))

#plt.figure()



ax = sns.boxenplot(x="Category", y='Sugars', data=menu, color='#eeeeee', palette="tab10")



# Add transparency to colors

for patch in ax.artists:

    r, g, b, a = patch.get_facecolor()

    patch.set_facecolor((r, g, b, .9))

    

#ax = sns.stripplot(x='Category', y='Cholesterol (% Daily Value)', data=menu, color="orange", jitter=0.5, size=5,alpha=0.15)

#

plt.title("Sugar  \n", loc="center",size=32,color='#be0c0c',alpha=0.6)

plt.xlabel('Category',color='#34495E',fontsize=20) 

plt.ylabel('Sugar',color='#34495E',fontsize=20)

plt.xticks(size=16,color='#008abc',rotation='horizontal', wrap=True)  

plt.yticks(size=15,color='#006600')

#plt.text(2.5, 1, 'Courtesy: https://seaborn.pydata.org/examples/grouped_boxplot.html', fontsize=13,alpha=0.2)

#plt.ylim(0,200)

#plt.legend(loc="upper right",fontsize=14,ncol=5,title='Category',title_fontsize=22,framealpha=0.99)

plt.show()
sns.set_style("whitegrid")

plt.figure(figsize=(22,10))

#plt.figure()



ax = sns.boxenplot(x="Category", y='Sodium (% Daily Value)', data=menu, color='#eeeeee', palette="tab10")



# Add transparency to colors

for patch in ax.artists:

    r, g, b, a = patch.get_facecolor()

    patch.set_facecolor((r, g, b, .9))

    

#ax = sns.stripplot(x='Category', y='Cholesterol (% Daily Value)', data=menu, color="orange", jitter=0.5, size=5,alpha=0.15)

#

plt.title("Sodium (% Daily Value)  \n", loc="center",size=32,color='#be0c0c',alpha=0.6)

plt.xlabel('Category',color='#34495E',fontsize=20) 

plt.ylabel('Sodium (% Daily Value)',color='#34495E',fontsize=20)

plt.xticks(size=16,color='#008abc',rotation='horizontal', wrap=True)  

plt.yticks(size=15,color='#006600')

#plt.text(2.5, 1, 'Courtesy: https://seaborn.pydata.org/examples/grouped_boxplot.html', fontsize=13,alpha=0.2)

#plt.ylim(0,200)

#plt.legend(loc="upper right",fontsize=14,ncol=5,title='Category',title_fontsize=22,framealpha=0.99)

plt.show()
sns.set_style("whitegrid")

plt.figure(figsize=(22,10))

#plt.figure()



ax = sns.boxenplot(x="Category", y='Protein', data=menu, color='#eeeeee', palette="Blues")



# Add transparency to colors

for patch in ax.artists:

    r, g, b, a = patch.get_facecolor()

    patch.set_facecolor((r, g, b, .9))

    

#ax = sns.stripplot(x='Category', y='Cholesterol (% Daily Value)', data=menu, color="orange", jitter=0.5, size=5,alpha=0.15)

#

plt.title("Protein  \n", loc="center",size=32,color='#be0c0c',alpha=0.6)

plt.xlabel('Category',color='#34495E',fontsize=20) 

plt.ylabel('Protein',color='#34495E',fontsize=20)

plt.xticks(size=16,color='#008abc',rotation='horizontal', wrap=True)  

plt.yticks(size=15,color='#006600')

#plt.text(2.5, 1, 'Courtesy: https://seaborn.pydata.org/examples/grouped_boxplot.html', fontsize=13,alpha=0.2)

#plt.ylim(0,200)

#plt.legend(loc="upper right",fontsize=14,ncol=5,title='Category',title_fontsize=22,framealpha=0.99)

plt.show()
sns.set_style("whitegrid")

plt.figure(figsize=(22,10))

#plt.figure()



ax = sns.boxenplot(x="Category", y='Dietary Fiber (% Daily Value)', data=menu, color='#eeeeee', palette="Greens")



# Add transparency to colors

for patch in ax.artists:

    r, g, b, a = patch.get_facecolor()

    patch.set_facecolor((r, g, b, .9))

    

#ax = sns.stripplot(x='Category', y='Cholesterol (% Daily Value)', data=menu, color="orange", jitter=0.5, size=5,alpha=0.15)

#

plt.title("Dietary Fiber (% Daily Value)  \n", loc="center",size=32,color='darkgreen',alpha=0.6)

plt.xlabel('Category',color='#34495E',fontsize=20) 

plt.ylabel('Dietary Fiber (% Daily Value)',color='#34495E',fontsize=20)

plt.xticks(size=16,color='#008abc',rotation='horizontal', wrap=True)  

plt.yticks(size=15,color='#006600')

#plt.text(2.5, 1, 'Courtesy: https://seaborn.pydata.org/examples/grouped_boxplot.html', fontsize=13,alpha=0.2)

#plt.ylim(0,200)

#plt.legend(loc="upper right",fontsize=14,ncol=5,title='Category',title_fontsize=22,framealpha=0.99)

plt.show()
sns.set_style("whitegrid")

plt.figure(figsize=(22,10))

#plt.figure()



ax = sns.boxenplot(x="Category", y='Vitamin C (% Daily Value)', data=menu, color='#eeeeee', palette="Greens")



# Add transparency to colors

for patch in ax.artists:

    r, g, b, a = patch.get_facecolor()

    patch.set_facecolor((r, g, b, .9))

    

#ax = sns.stripplot(x='Category', y='Cholesterol (% Daily Value)', data=menu, color="orange", jitter=0.5, size=5,alpha=0.15)

#

plt.title("Vitamin C (% Daily Value)  \n", loc="center",size=32,color='darkgreen',alpha=0.6)

plt.xlabel('Category',color='#34495E',fontsize=20) 

plt.ylabel('Vitamin C (% Daily Value)',color='#34495E',fontsize=20)

plt.xticks(size=16,color='#008abc',rotation='horizontal', wrap=True)  

plt.yticks(size=15,color='#006600')

#plt.text(2.5, 1, 'Courtesy: https://seaborn.pydata.org/examples/grouped_boxplot.html', fontsize=13,alpha=0.2)

#plt.ylim(0,200)

#plt.legend(loc="upper right",fontsize=14,ncol=5,title='Category',title_fontsize=22,framealpha=0.99)

plt.show()
sns.set_style("whitegrid")

plt.figure(figsize=(22,10))

#plt.figure()



ax = sns.boxenplot(x="Category", y='Vitamin A (% Daily Value)', data=menu, color='#eeeeee', palette="Greens")



# Add transparency to colors

for patch in ax.artists:

    r, g, b, a = patch.get_facecolor()

    patch.set_facecolor((r, g, b, .9))

    

#ax = sns.stripplot(x='Category', y='Cholesterol (% Daily Value)', data=menu, color="orange", jitter=0.5, size=5,alpha=0.15)

#

plt.title("Vitamin A (% Daily Value)  \n", loc="center",size=32,color='darkgreen',alpha=0.6)

plt.xlabel('Category',color='#34495E',fontsize=20) 

plt.ylabel('Vitamin A (% Daily Value)',color='#34495E',fontsize=20)

plt.xticks(size=16,color='#008abc',rotation='horizontal', wrap=True)  

plt.yticks(size=15,color='#006600')

#plt.text(2.5, 1, 'Courtesy: https://seaborn.pydata.org/examples/grouped_boxplot.html', fontsize=13,alpha=0.2)

#plt.ylim(0,200)

#plt.legend(loc="upper right",fontsize=14,ncol=5,title='Category',title_fontsize=22,framealpha=0.99)

plt.show()
sns.set_style("whitegrid")

plt.figure(figsize=(22,10))

#plt.figure()



ax = sns.boxenplot(x="Category", y='Calcium (% Daily Value)', data=menu, color='#eeeeee', palette="Greens")



# Add transparency to colors

for patch in ax.artists:

    r, g, b, a = patch.get_facecolor()

    patch.set_facecolor((r, g, b, .9))

    

#ax = sns.stripplot(x='Category', y='Cholesterol (% Daily Value)', data=menu, color="orange", jitter=0.5, size=5,alpha=0.15)

#

plt.title("Calcium (% Daily Value)  \n", loc="center",size=32,color='darkgreen',alpha=0.6)

plt.xlabel('Category',color='#34495E',fontsize=20) 

plt.ylabel('Calcium (% Daily Value)',color='#34495E',fontsize=20)

plt.xticks(size=16,color='#008abc',rotation='horizontal', wrap=True)  

plt.yticks(size=15,color='#006600')

#plt.text(2.5, 1, 'Courtesy: https://seaborn.pydata.org/examples/grouped_boxplot.html', fontsize=13,alpha=0.2)

#plt.ylim(0,200)

#plt.legend(loc="upper right",fontsize=14,ncol=5,title='Category',title_fontsize=22,framealpha=0.99)

plt.show()
sns.set_style("whitegrid")

plt.figure(figsize=(22,10))

#plt.figure()



ax = sns.boxenplot(x="Category", y='Iron (% Daily Value)', data=menu, color='#eeeeee', palette="Greens")



# Add transparency to colors

for patch in ax.artists:

    r, g, b, a = patch.get_facecolor()

    patch.set_facecolor((r, g, b, .9))

    

#ax = sns.stripplot(x='Category', y='Cholesterol (% Daily Value)', data=menu, color="orange", jitter=0.5, size=5,alpha=0.15)

#

plt.title("Iron (% Daily Value)  \n", loc="center",size=32,color='darkgreen',alpha=0.6)

plt.xlabel('Category',color='#34495E',fontsize=20) 

plt.ylabel('Iron (% Daily Value)',color='#34495E',fontsize=20)

plt.xticks(size=16,color='#008abc',rotation='horizontal', wrap=True)  

plt.yticks(size=15,color='#006600')

#plt.text(2.5, 1, 'Courtesy: https://seaborn.pydata.org/examples/grouped_boxplot.html', fontsize=13,alpha=0.2)

#plt.ylim(0,200)

#plt.legend(loc="upper right",fontsize=14,ncol=5,title='Category',title_fontsize=22,framealpha=0.99)

plt.show()
import plotly.express as px

#df = px.data.menu()

fig = px.scatter(menu, x="Total Fat (% Daily Value)",y='Cholesterol (% Daily Value)', color="Category",

                 size='Calories', hover_name="Item",trendline="ols",

                 template="plotly_dark",marginal_x="box",marginal_y="box")

fig.show()
fig = px.scatter(menu, x="Carbohydrates (% Daily Value)",y='Sugars', color="Category",

                 size='Calories', hover_name="Item",trendline="ols",

                 template="plotly_dark",marginal_x="box",marginal_y="box")

fig.show()
fig = px.scatter(menu, x="Carbohydrates (% Daily Value)",y='Protein', color="Category",

                 size='Calories', hover_name="Item",trendline="ols",template="plotly_dark",

                 marginal_x="box",marginal_y="box")

fig.show()
fig = px.scatter(menu, x='Dietary Fiber (% Daily Value)',y='Vitamin C (% Daily Value)', color="Category",

                 size='Calories',  hover_name="Item",

                 trendline="ols",template="plotly_dark",marginal_x="box",marginal_y="box")

fig.show()
menu.columns
fig = px.scatter(menu, x='Protein',y='Sugars', color="Category",size='Calories',

                 hover_name="Item",trendline="ols",template="plotly_dark",marginal_x="box",marginal_y="box")

fig.show()